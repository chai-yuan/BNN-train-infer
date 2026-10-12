#include "internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void BNNE_OptionsInit(BNNE_Options* options) {
    if (options)
        *options = (BNNE_Options){0, 1};
}

void BNNE_Close(BNNE_Engine engine) {
    if (!engine)
        return;
    if (engine->ops)
        engine->ops->destroy(engine);
    bnne_runtime_close(engine);
    free(engine);
}

BNNE_Status BNNE_Open(const char* model_path, const BNNE_Options* options, BNNE_Engine* out_engine) {
    if (!out_engine)
        return BNNE_EINVAL;
    *out_engine = NULL;
    if (!model_path)
        return BNNE_EINVAL;
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128)
        return BNNE_ENOTSUP;
    BNNE_Engine engine = calloc(1, sizeof *engine);
    if (!engine)
        return BNNE_ENOMEM;
    BNNE_OptionsInit(&engine->options);
    if (options)
        engine->options = *options;
    uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
    FILE* file = fopen(model_path, "rb");
    if (!file) {
        BNNE_Close(engine);
        return BNNE_EIO;
    }
    BNNE_Reader reader = {file, BNNE_OK};
    char magic[4] = {0};
    bnne_read(&reader, magic, sizeof magic);
    BNNE_Status status = reader.status;
    if (status == BNNE_OK) {
        if (memcmp(magic, "BNN1", 4) == 0)
            engine->ops = &bnne_mlp_ops;
        else if (memcmp(magic, "BNN2", 4) == 0)
            engine->ops = &bnne_cnn_ops;
        else if (memcmp(magic, "GPT1", 4) == 0)
            engine->ops = &bnne_gpt_ops;
        status = engine->ops ? engine->ops->load(engine, file) : BNNE_ENOTSUP;
    }
    if (fclose(file) != 0 && status == BNNE_OK)
        status = BNNE_EIO;
    if (status == BNNE_OK)
        status = engine->ops->prepare(engine);
    if (status != BNNE_OK) {
        BNNE_Close(engine);
        return status;
    }
    engine->info.api_version = BNNE_API_VERSION;
    if (engine->options.enable_profile)
        engine->stats.load_ns = bnne_now_ns() - start;
    *out_engine = engine;
    return BNNE_OK;
}

BNNE_Status BNNE_GetModelInfo(BNNE_Engine engine, BNNE_ModelInfo* out_info) {
    if (!engine || !out_info)
        return BNNE_EINVAL;
    *out_info = engine->info;
    return BNNE_OK;
}

static int desc_equal(const BNNE_TensorDesc* a, const BNNE_TensorDesc* b) {
    if (a->dtype != b->dtype || a->rank != b->rank || a->rank > BNNE_MAX_RANK)
        return 0;
    for (uint32_t d = 0; d < a->rank; d++)
        if (a->dims[d] != b->dims[d])
            return 0;
    return 1;
}

BNNE_Status BNNE_GetOutputDesc(BNNE_Engine engine, const BNNE_TensorDesc* input,
                              BNNE_TensorDesc* out_output) {
    if (!engine || !input || !out_output)
        return BNNE_EINVAL;
    if (engine->info.kind == BNNE_MODEL_GPT) {
        if (input->dtype != BNNE_I32 || input->rank != 1 || input->dims[0] == 0 ||
            input->dims[0] > engine->info.max_sequence_length)
            return bnne_error(engine, BNNE_EINVAL, "GPT input must be I32 [n], 1 <= n <= context length");
        *out_output = engine->info.output;
        out_output->dims[0] = input->dims[0];
    } else {
        if (!desc_equal(input, &engine->info.input))
            return bnne_error(engine, BNNE_EINVAL, "input type or shape does not match model");
        *out_output = engine->info.output;
    }
    return BNNE_OK;
}

static BNNE_Status tensor_bytes(const BNNE_TensorDesc* desc, size_t* out_bytes) {
    if (desc->rank == 0 || desc->rank > BNNE_MAX_RANK ||
        (desc->dtype != BNNE_F32 && desc->dtype != BNNE_I32))
        return BNNE_EINVAL;
    size_t bytes = sizeof(uint32_t);
    for (uint32_t d = 0; d < desc->rank; d++) {
        if (desc->dims[d] == 0 || bytes > SIZE_MAX / desc->dims[d])
            return BNNE_ERANGE;
        bytes *= desc->dims[d];
    }
    *out_bytes = bytes;
    return BNNE_OK;
}

BNNE_Status BNNE_Run(BNNE_Engine engine, const BNNE_Tensor* input, BNNE_Tensor* output) {
    if (!engine)
        return BNNE_EINVAL;
    engine->error[0] = '\0';
    if (!input || !output || !input->data || !output->data)
        return bnne_error(engine, BNNE_EINVAL, "input/output tensor and data must not be NULL");
    BNNE_TensorDesc expected;
    BNNE_Status status = BNNE_GetOutputDesc(engine, &input->desc, &expected);
    if (status != BNNE_OK)
        return status;
    if (!desc_equal(&output->desc, &expected))
        return bnne_error(engine, BNNE_EINVAL, "output type or shape does not match model");
    size_t input_bytes, output_bytes;
    if (tensor_bytes(&input->desc, &input_bytes) != BNNE_OK ||
        tensor_bytes(&output->desc, &output_bytes) != BNNE_OK ||
        input->bytes < input_bytes || output->bytes < output_bytes)
        return bnne_error(engine, BNNE_ERANGE, "input/output buffer capacity is insufficient");
    size_t input_align = input->desc.dtype == BNNE_F32 ?
        offsetof(struct { char pad; float value; }, value) :
        offsetof(struct { char pad; int32_t value; }, value);
    size_t output_align = offsetof(struct { char pad; float value; }, value);
    if ((uintptr_t)input->data % input_align != 0 || (uintptr_t)output->data % output_align != 0)
        return bnne_error(engine, BNNE_EINVAL, "input/output data is not aligned");
    uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
    for (size_t i = 0; i < input_bytes / sizeof(uint32_t); i++) {
        if (input->desc.dtype == BNNE_F32) {
            if (!isfinite(((const float*)input->data)[i]))
                return bnne_error(engine, BNNE_EINVAL, "input contains NaN or Inf");
        } else {
            int32_t id = ((const int32_t*)input->data)[i];
            if (id < 0 || (uint32_t)id >= engine->info.vocabulary_size)
                return bnne_error(engine, BNNE_EINVAL, "token ID is outside vocabulary");
        }
    }
    BNNE_Stats delta = {0};
    status = engine->ops->run(engine, input, output, &delta);
    if (status != BNNE_OK)
        return status;
    for (size_t i = 0; i < output_bytes / sizeof(float); i++)
        if (!isfinite(((const float*)output->data)[i]))
            return bnne_error(engine, BNNE_ECOMPUTE, "model produced NaN or Inf");
    if (engine->options.enable_profile) {
        delta.forward_calls = 1;
        delta.total_ns = bnne_now_ns() - start;
        delta.host_ns = delta.total_ns - delta.quantize_ns - delta.transfer_ns - delta.execute_ns;
#define ACCUMULATE(field) engine->stats.field += delta.field
        ACCUMULATE(forward_calls); ACCUMULATE(total_ns); ACCUMULATE(quantize_ns);
        ACCUMULATE(transfer_ns); ACCUMULATE(execute_ns); ACCUMULATE(host_ns);
        ACCUMULATE(device_calls); ACCUMULATE(bytes_written); ACCUMULATE(bytes_read);
        ACCUMULATE(logical_macs); ACCUMULATE(executed_macs);
#undef ACCUMULATE
    }
    return BNNE_OK;
}

BNNE_Status BNNE_GetToken(BNNE_Engine engine, uint32_t id, const char** out_text) {
    if (!engine || !out_text)
        return BNNE_EINVAL;
    *out_text = NULL;
    if (!engine->ops->token)
        return BNNE_ENOTSUP;
    if (id >= engine->info.vocabulary_size)
        return BNNE_ERANGE;
    return engine->ops->token(engine, id, out_text);
}

BNNE_Status BNNE_GetStats(BNNE_Engine engine, BNNE_Stats* out_stats) {
    if (!engine || !out_stats)
        return BNNE_EINVAL;
    *out_stats = engine->stats;
    return BNNE_OK;
}

BNNE_Status BNNE_ResetStats(BNNE_Engine engine) {
    if (!engine)
        return BNNE_EINVAL;
    uint64_t load_ns = engine->stats.load_ns;
    memset(&engine->stats, 0, sizeof engine->stats);
    engine->stats.load_ns = load_ns;
    return BNNE_OK;
}

const char* BNNE_LastError(BNNE_Engine engine) {
    return engine ? engine->error : "invalid engine";
}

const char* BNNE_StatusString(BNNE_Status status) {
    switch (status) {
    case BNNE_OK: return "success";
    case BNNE_EINVAL: return "invalid argument";
    case BNNE_ENOMEM: return "out of memory";
    case BNNE_ENOTSUP: return "unsupported model or device capability";
    case BNNE_ERANGE: return "size or capacity out of range";
    case BNNE_EIO: return "I/O error";
    case BNNE_EFORMAT: return "invalid or truncated model";
    case BNNE_ECOMPUTE: return "computation failed or produced non-finite values";
    default: return "unknown status";
    }
}
