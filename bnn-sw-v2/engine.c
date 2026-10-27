#include "bnne-internal.h"

#include "gguf.h"
#include "tokenizer/tokenizer-bpe.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int float_format_supported(void) {
    return sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128;
}

BNNE_Status BNNE_ModelLoad(BNN_Device device, const char* gguf_path, BNNE_Model* out_model) {
    if (out_model)
        *out_model = NULL;
    bnne_error_clear();
    if (!device || !gguf_path || !out_model)
        return BNNE_EINVAL;
    if (!float_format_supported())
        return bnne_error(BNNE_ENOTSUP, "IEEE-754 binary32 float is required");

    uint64_t start = bnne_now_ns();
    bnne_gguf* g = NULL;
    BNNE_Status status = bnne_gguf_open(gguf_path, &g);
    if (status != BNNE_OK)
        return status;

    const bnne_arch* arch = bnne_arch_find(g->architecture);
    if (!arch) {
        bnne_error(BNNE_ENOTSUP, "unsupported GGUF architecture '%s'", g->architecture);
        bnne_gguf_close(g);
        return BNNE_ENOTSUP;
    }

    BNNE_Model model = calloc(1, sizeof *model);
    if (!model) {
        bnne_gguf_close(g);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }
    model->device = device;
    model->gguf = g;
    model->arch = arch;

    status = bnne_accel_error(BNN_DeviceGetCaps(device, &model->caps), "BNN_DeviceGetCaps");
    if (status != BNNE_OK)
        goto fail;
    if (model->caps.api_version != BNN_API_VERSION || model->caps.group_size == 0 ||
        model->caps.max_in_features == 0 || model->caps.max_out_features == 0 ||
        model->caps.max_rows == 0) {
        status = bnne_error(BNNE_ENOTSUP, "device does not support this engine");
        goto fail;
    }

    status = arch->load(model, g, &model->impl);
    if (status != BNNE_OK)
        goto fail;

    if (arch->task == BNNE_TASK_LLM) {
        status = bnne_tokenizer_create(g, &model->tok);
        if (status != BNNE_OK)
            goto fail;
        model->info.bos_token = g->has_bos ? (int32_t)g->bos_token_id : -1;
        model->info.eos_token = g->has_eos ? (int32_t)g->eos_token_id : -1;
    }

    model->info.task = arch->task;
    model->info.arch = arch->name;
    model->info.load_ns = bnne_now_ns() - start;
    *out_model = model;
    return BNNE_OK;

fail:
    BNNE_ModelFree(model);
    return status;
}

void BNNE_ModelFree(BNNE_Model model) {
    if (!model)
        return;
    if (model->arch && model->arch->unload && model->impl)
        model->arch->unload(model->impl);
    bnne_tokenizer_destroy(model->tok);
    bnne_gguf_close(model->gguf);
    free(model);
}

BNNE_Status BNNE_ModelGetInfo(BNNE_Model model, BNNE_ModelInfo* out_info) {
    if (!model || !out_info)
        return BNNE_EINVAL;
    *out_info = model->info;
    return BNNE_OK;
}

/* ---------------- Tokenizer ---------------- */

BNNE_Status BNNE_Tokenize(BNNE_Model model, const char* text, int parse_special, int32_t* ids,
                          uint32_t capacity, uint32_t* out_count) {
    if (out_count)
        *out_count = 0;
    bnne_error_clear();
    if (!model || !text || !ids || !out_count || capacity == 0)
        return BNNE_EINVAL;
    if (!model->tok)
        return bnne_error(BNNE_ENOTSUP, "model has no text tokenizer");
    return bnne_tokenizer_encode(model->tok, text, parse_special, ids, capacity, out_count);
}

BNNE_Status BNNE_TokenPiece(BNNE_Model model, int32_t id, const char** out_bytes,
                            size_t* out_len) {
    if (out_bytes)
        *out_bytes = NULL;
    if (out_len)
        *out_len = 0;
    bnne_error_clear();
    if (!model || !out_bytes || !out_len)
        return BNNE_EINVAL;
    if (!model->tok)
        return bnne_error(BNNE_ENOTSUP, "model has no text tokenizer");
    return bnne_tokenizer_piece(model->tok, id, out_bytes, out_len);
}

int BNNE_TokenIsEog(BNNE_Model model, int32_t id) {
    if (!model || !model->tok || id < 0 || (uint32_t)id >= model->info.n_vocab)
        return 0;
    return bnne_tokenizer_is_eog(model->tok, id);
}

/* ---------------- Context ---------------- */

void BNNE_ContextParamsInit(BNNE_ContextParams* params) {
    if (params)
        *params = (BNNE_ContextParams){0, 0, 1};
}

BNNE_Status BNNE_ContextCreate(BNNE_Model model, const BNNE_ContextParams* params,
                               BNNE_Context* out_ctx) {
    if (out_ctx)
        *out_ctx = NULL;
    bnne_error_clear();
    if (!model || !out_ctx)
        return BNNE_EINVAL;

    BNNE_Context c = calloc(1, sizeof *c);
    if (!c)
        return bnne_error(BNNE_ENOMEM, "out of memory");
    c->model = model;
    BNNE_ContextParamsInit(&c->params);
    if (params)
        c->params = *params;

    if (model->info.task == BNNE_TASK_LLM) {
        if (c->params.n_ctx == 0)
            c->params.n_ctx = model->info.n_ctx_train;
        if (c->params.n_ctx == 0 || c->params.n_ctx > model->info.n_ctx_train) {
            uint32_t n_ctx = c->params.n_ctx;
            BNNE_ContextFree(c);
            return bnne_error(BNNE_ERANGE, "context length %u exceeds model capacity %u", n_ctx,
                              model->info.n_ctx_train);
        }
        if (c->params.n_batch == 0 || c->params.n_batch > c->params.n_ctx)
            c->params.n_batch = c->params.n_ctx;
    } else if (c->params.n_batch == 0) {
        c->params.n_batch = 1;
    }

    if (model->arch->ctx_init) {
        BNNE_Status status = model->arch->ctx_init(c, &c->state);
        if (status != BNNE_OK) {
            BNNE_ContextFree(c);
            return status;
        }
    }

    c->max_in = model->max_in ? model->max_in : 1;
    c->max_out = model->max_out ? model->max_out : 1;
    if ((uint64_t)c->max_in * sizeof(float) > SIZE_MAX ||
        (uint64_t)c->max_out * sizeof(float) > SIZE_MAX) {
        BNNE_ContextFree(c);
        return bnne_error(BNNE_ERANGE, "context workspace is too large");
    }
    size_t input_bytes = (size_t)c->max_in * sizeof(float);
    size_t output_bytes = (size_t)c->max_out * sizeof(float);

    BNNE_Status status = bnne_accel_error(
        BNN_BufferAlloc(model->device, input_bytes, &c->in_buf), "BNN_BufferAlloc(input)");
    if (status == BNNE_OK)
        status = bnne_accel_error(
            BNN_BufferAlloc(model->device, output_bytes, &c->out_buf), "BNN_BufferAlloc(output)");
    if (status != BNNE_OK) {
        BNNE_ContextFree(c);
        return status;
    }

    c->pos = 0;
    *out_ctx = c;
    return BNNE_OK;
}

void BNNE_ContextFree(BNNE_Context c) {
    if (!c)
        return;
    if (c->model && c->model->arch && c->model->arch->ctx_free && c->state)
        c->model->arch->ctx_free(c->state);
    BNN_BufferFree(c->in_buf);
    BNN_BufferFree(c->out_buf);
    free(c);
}

typedef struct {
    BNNE_Stats snapshot;
    uint64_t start;
} StatsMark;

static void stats_begin(BNNE_Context c, StatsMark* mark) {
    if (!c->params.enable_stats) {
        memset(&mark->snapshot, 0, sizeof mark->snapshot);
        mark->start = 0;
        return;
    }
    mark->snapshot = c->stats;
    mark->start = bnne_now_ns();
}

static void stats_end(BNNE_Context c, const StatsMark* mark, uint64_t items) {
    if (!c->params.enable_stats)
        return;
    uint64_t end = bnne_now_ns();
    uint64_t total = end > mark->start ? end - mark->start : 0;
    uint64_t stages = (c->stats.quantize_ns - mark->snapshot.quantize_ns) +
                      (c->stats.transfer_ns - mark->snapshot.transfer_ns) +
                      (c->stats.execute_ns - mark->snapshot.execute_ns);
    c->stats.total_ns += total;
    c->stats.host_ns += total > stages ? total - stages : 0;
    c->stats.calls++;
    c->stats.items += items;
}

BNNE_Status BNNE_Decode(BNNE_Context c, const int32_t* tokens, uint32_t n_tokens,
                        float* logits) {
    bnne_error_clear();
    if (!c || !tokens || n_tokens == 0)
        return BNNE_EINVAL;
    if (c->model->info.task != BNNE_TASK_LLM || !c->model->arch->decode)
        return bnne_error(BNNE_ENOTSUP, "model is not a language model");
    if (n_tokens > c->params.n_batch)
        return bnne_error(BNNE_ERANGE, "token count exceeds context batch size");
    if (c->pos > c->params.n_ctx || n_tokens > c->params.n_ctx - c->pos)
        return bnne_error(BNNE_ERANGE, "decode exceeds context length");
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= c->model->info.n_vocab)
            return bnne_error(BNNE_ERANGE, "token ID is outside vocabulary");
    }

    StatsMark mark;
    stats_begin(c, &mark);
    BNNE_Status status = c->model->arch->decode(c, tokens, n_tokens, logits);
    if (status != BNNE_OK)
        return status;

    if (logits) {
        for (uint32_t i = 0; i < c->model->info.n_vocab; i++) {
            if (!isfinite(logits[i]))
                return bnne_error(BNNE_ECOMPUTE, "model produced NaN or Inf logits");
        }
    }

    c->pos += n_tokens;
    stats_end(c, &mark, n_tokens);
    return BNNE_OK;
}

BNNE_Status BNNE_Classify(BNNE_Context c, const float* input, uint32_t n_samples,
                          float* logits) {
    bnne_error_clear();
    if (!c || !input || !logits || n_samples == 0)
        return BNNE_EINVAL;
    if (c->model->info.task != BNNE_TASK_CLASSIFY || !c->model->arch->classify)
        return bnne_error(BNNE_ENOTSUP, "model is not a classifier");
    if (n_samples > c->params.n_batch)
        return bnne_error(BNNE_ERANGE, "sample count exceeds context batch size");

    uint64_t inputs = (uint64_t)n_samples * c->model->info.input_size;
    for (uint64_t i = 0; i < inputs; i++) {
        if (!isfinite(input[i]))
            return bnne_error(BNNE_EINVAL, "input contains NaN or Inf");
    }

    StatsMark mark;
    stats_begin(c, &mark);
    BNNE_Status status = c->model->arch->classify(c, input, n_samples, logits);
    if (status != BNNE_OK)
        return status;
    uint64_t outputs = (uint64_t)n_samples * c->model->info.n_classes;
    for (uint64_t i = 0; i < outputs; i++) {
        if (!isfinite(logits[i]))
            return bnne_error(BNNE_ECOMPUTE, "model produced NaN or Inf logits");
    }
    stats_end(c, &mark, n_samples);
    return BNNE_OK;
}

uint32_t BNNE_ContextPos(BNNE_Context c) {
    return c ? c->pos : 0;
}

BNNE_Status BNNE_ContextTruncate(BNNE_Context c, uint32_t pos) {
    bnne_error_clear();
    if (!c)
        return BNNE_EINVAL;
    if (pos > c->pos)
        return bnne_error(BNNE_EINVAL, "truncate position is beyond current position");
    if (c->model->info.task == BNNE_TASK_LLM) {
        if (c->model->arch->truncate) {
            BNNE_Status status = c->model->arch->truncate(c, pos);
            if (status != BNNE_OK)
                return status;
        }
    }
    c->pos = pos;
    return BNNE_OK;
}

BNNE_Status BNNE_ContextGetStats(BNNE_Context c, BNNE_Stats* out_stats) {
    if (!c || !out_stats)
        return BNNE_EINVAL;
    *out_stats = c->stats;
    return BNNE_OK;
}

BNNE_Status BNNE_ContextResetStats(BNNE_Context c) {
    if (!c)
        return BNNE_EINVAL;
    memset(&c->stats, 0, sizeof c->stats);
    return BNNE_OK;
}
