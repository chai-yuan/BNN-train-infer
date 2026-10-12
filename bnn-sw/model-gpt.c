#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GPT_MAX_VOCAB 65536u
#define GPT_MAX_SEQUENCE 4096u
#define GPT_MAX_CHANNELS 4096u
#define GPT_MAX_LAYERS 128u

enum { WQ, WK, WV, WO, WFC, WPROJ, WEIGHTS };
enum { X, NORM, Q, K, V, Y, O, FF, SCORES, PROBS, WORKS };

typedef struct {
    uint32_t in, out, group, relu;
    uint8_t* bits;
    uint16_t* scales;
    BNN_Matrix matrix;
} GPT_Weight;

typedef struct {
    float* gain[2];
    GPT_Weight weights[WEIGHTS];
} GPT_Block;

typedef struct {
    uint32_t vocab_size, length, channels, heads, layers, group;
    float epsilon;
    float *tokens, *positions, *gain;
    GPT_Block* blocks;
    GPT_Weight head;
    char* vocabulary;
    uint32_t* offsets;
    float* work[WORKS];
    int8_t* quantized;
    float* row_scales;
    uint8_t* query_bits;
    uint16_t* query_scales;
} GPT_Model;

/* Dimensions are validated before multiplying counts; also guard byte sizes. */
static void* gpt_alloc(size_t count, size_t size) {
    return count && size <= SIZE_MAX / count ? malloc(count * size) : NULL;
}

static void gpt_weight_destroy(GPT_Weight* weight) {
    BNN_MatrixDestroy(weight->matrix);
    free(weight->bits);
    free(weight->scales);
}

static void gpt_destroy(BNNE_Engine engine) {
    GPT_Model* model = engine->model;
    if (!model)
        return;
    if (model->blocks)
        for (uint32_t l = 0; l < model->layers; l++) {
            GPT_Block* block = &model->blocks[l];
            free(block->gain[0]);
            free(block->gain[1]);
            for (uint32_t w = 0; w < WEIGHTS; w++)
                gpt_weight_destroy(&block->weights[w]);
        }
    gpt_weight_destroy(&model->head);
    free(model->blocks);
    free(model->tokens);
    free(model->positions);
    free(model->gain);
    free(model->vocabulary);
    free(model->offsets);
    for (uint32_t i = 0; i < WORKS; i++)
        free(model->work[i]);
    free(model->quantized);
    free(model->row_scales);
    free(model->query_bits);
    free(model->query_scales);
    free(model);
    engine->model = NULL;
}

static BNNE_Status gpt_read_floats(BNNE_Reader* reader, float** out, size_t count) {
    *out = gpt_alloc(count, sizeof **out);
    if (!*out)
        return BNNE_ENOMEM;
    for (size_t i = 0; i < count && reader->status == BNNE_OK; i++) {
        (*out)[i] = bnne_read_f32(reader);
        if (!isfinite((*out)[i]))
            return BNNE_EFORMAT;
    }
    return reader->status;
}

static BNNE_Status gpt_read_weight(BNNE_Reader* reader, GPT_Weight* weight,
                                    uint32_t in, uint32_t out, uint32_t group,
                                    uint32_t relu) {
    weight->in = bnne_read_u32(reader);
    weight->out = bnne_read_u32(reader);
    weight->group = bnne_read_u32(reader);
    weight->relu = bnne_read_u32(reader);
    if (reader->status != BNNE_OK)
        return reader->status;
    if (weight->in != in || weight->out != out || weight->group != group ||
        weight->relu != relu)
        return BNNE_EFORMAT;
    size_t bytes = (size_t)out * (in / 8);
    size_t count = (size_t)out * (in / group);
    weight->bits = gpt_alloc(bytes, 1);
    weight->scales = gpt_alloc(count, sizeof *weight->scales);
    if (!weight->bits || !weight->scales)
        return BNNE_ENOMEM;
    bnne_read(reader, weight->bits, bytes);
    for (size_t i = 0; i < count && reader->status == BNNE_OK; i++) {
        weight->scales[i] = bnne_read_u16(reader);
        if (!isfinite(bnne_fp16_to_fp32(weight->scales[i])))
            return BNNE_EFORMAT;
    }
    return reader->status;
}

static BNNE_Status gpt_load(BNNE_Engine engine, FILE* file) {
    GPT_Model* model = calloc(1, sizeof *model);
    if (!model)
        return BNNE_ENOMEM;
    engine->model = model;
    BNNE_Reader reader = {file, BNNE_OK};
    model->vocab_size = bnne_read_u32(&reader);
    model->length = bnne_read_u32(&reader);
    model->channels = bnne_read_u32(&reader);
    model->heads = bnne_read_u32(&reader);
    model->layers = bnne_read_u32(&reader);
    model->group = bnne_read_u32(&reader);
    model->epsilon = bnne_read_f32(&reader);
    if (reader.status != BNNE_OK)
        return reader.status;
    uint32_t C = model->channels, G = model->group, H = model->heads;
    if (!model->vocab_size || model->vocab_size > GPT_MAX_VOCAB ||
        !model->length || model->length > GPT_MAX_SEQUENCE ||
        !C || C > GPT_MAX_CHANNELS || !model->layers || model->layers > GPT_MAX_LAYERS ||
        !G || !H || G % 8 || C % G || C % H || C / H != G ||
        !isfinite(model->epsilon) || model->epsilon <= 0.0f)
        return bnne_error(engine, BNNE_EFORMAT, "invalid GPT dimensions or epsilon");

    BNNE_Status status = gpt_read_floats(&reader, &model->tokens, (size_t)model->vocab_size * C);
    if (status != BNNE_OK)
        return status;
    status = gpt_read_floats(&reader, &model->positions, (size_t)model->length * C);
    if (status != BNNE_OK)
        return status;
    model->blocks = calloc(model->layers, sizeof *model->blocks);
    if (!model->blocks)
        return BNNE_ENOMEM;
    for (uint32_t l = 0; l < model->layers; l++) {
        GPT_Block* block = &model->blocks[l];
        status = gpt_read_floats(&reader, &block->gain[0], C);
        if (status != BNNE_OK)
            return status;
        for (uint32_t w = WQ; w <= WO; w++) {
            status = gpt_read_weight(&reader, &block->weights[w], C, C, G, 0);
            if (status != BNNE_OK)
                return status;
        }
        status = gpt_read_floats(&reader, &block->gain[1], C);
        if (status != BNNE_OK)
            return status;
        status = gpt_read_weight(&reader, &block->weights[WFC], C, 4 * C, G, 1);
        if (status != BNNE_OK)
            return status;
        status = gpt_read_weight(&reader, &block->weights[WPROJ], 4 * C, C, G, 0);
        if (status != BNNE_OK)
            return status;
    }
    status = gpt_read_floats(&reader, &model->gain, C);
    if (status != BNNE_OK)
        return status;
    status = gpt_read_weight(&reader, &model->head, C, model->vocab_size, G, 0);
    if (status != BNNE_OK)
        return status;

    /* Each byte-length record needs at most 255 bytes plus its terminator. */
    model->vocabulary = gpt_alloc(model->vocab_size, 256);
    model->offsets = gpt_alloc(model->vocab_size, sizeof *model->offsets);
    if (!model->vocabulary || !model->offsets)
        return BNNE_ENOMEM;
    size_t used = 0;
    for (uint32_t i = 0; i < model->vocab_size; i++) {
        uint8_t length = 0;
        bnne_read(&reader, &length, 1);
        if (reader.status != BNNE_OK)
            return reader.status;
        if (!length)
            return BNNE_EFORMAT;
        model->offsets[i] = (uint32_t)used;
        bnne_read(&reader, model->vocabulary + used, length);
        if (reader.status != BNNE_OK)
            return reader.status;
        if (memchr(model->vocabulary + used, '\0', length))
            return BNNE_EFORMAT;
        used += length;
        model->vocabulary[used++] = '\0';
    }
    if (fgetc(file) != EOF)
        return BNNE_EFORMAT;
    if (ferror(file))
        return BNNE_EIO;
    engine->info.kind = BNNE_MODEL_GPT;
    engine->info.layer_count = model->layers;
    engine->info.input = (BNNE_TensorDesc){BNNE_I32, 1, {model->length}};
    engine->info.output = (BNNE_TensorDesc){BNNE_F32, 2, {model->length, model->vocab_size}};
    engine->info.vocabulary_size = model->vocab_size;
    engine->info.max_sequence_length = model->length;
    return BNNE_OK;
}

static BNNE_Status gpt_upload(BNNE_Engine engine, GPT_Weight* weight) {
    BNN_MatrixDesc desc = {
        .in_features = weight->in,
        .out_features = weight->out,
        .group_size = weight->group,
        .weight_bits = weight->bits,
        .weight_bytes = (size_t)weight->out * (weight->in / 8),
        .scale_fp16 = weight->scales,
        .scale_count = (size_t)weight->out * (weight->in / weight->group),
    };
    BNNE_Status status = bnne_accel_status(engine,
        BNN_MatrixCreate(engine->device, &desc, &weight->matrix), "BNN_MatrixCreate(GPT weight)");
    if (status == BNNE_OK) {
        free(weight->bits);
        free(weight->scales);
        weight->bits = NULL;
        weight->scales = NULL;
    }
    return status;
}

static BNNE_Status gpt_prepare(BNNE_Engine engine) {
    GPT_Model* model = engine->model;
    BNN_Caps caps = {0};
    BNNE_Status status = bnne_runtime_open(engine, &caps);
    if (status != BNNE_OK)
        return status;
    uint32_t max_in = 4 * model->channels;
    uint32_t max_out = model->vocab_size > max_in ? model->vocab_size : max_in;
    if (model->length > max_out)
        max_out = model->length;
    if (caps.api_version != BNN_API_VERSION || !caps.supports_gemm ||
        caps.group_size != model->group || model->length > caps.max_rows ||
        max_in > caps.max_in_features || max_out > caps.max_out_features)
        return bnne_error(engine, BNNE_ENOTSUP, "GPT GEMM shapes exceed device capabilities");
    for (uint32_t l = 0; l < model->layers; l++)
        for (uint32_t w = 0; w < WEIGHTS; w++) {
            status = gpt_upload(engine, &model->blocks[l].weights[w]);
            if (status != BNNE_OK)
                return status;
        }
    status = gpt_upload(engine, &model->head);
    if (status != BNNE_OK)
        return status;

    size_t tc = (size_t)model->length * model->channels;
    size_t tt = (size_t)model->length * model->length;
    for (uint32_t i = 0; i < WORKS; i++) {
        size_t count = i >= SCORES ? tt : (i == FF ? 4 * tc : tc);
        model->work[i] = gpt_alloc(count, sizeof(float));
        if (!model->work[i])
            return BNNE_ENOMEM;
    }
    model->quantized = gpt_alloc(model->length, max_in);
    model->row_scales = gpt_alloc(model->length, sizeof(float));
    model->query_bits = gpt_alloc(model->length, model->group / 8);
    model->query_scales = gpt_alloc(model->length, sizeof(uint16_t));
    if (!model->quantized || !model->row_scales || !model->query_bits || !model->query_scales)
        return BNNE_ENOMEM;
    return bnne_runtime_buffers(engine, (size_t)model->length * max_in,
                                (size_t)model->length * max_out * sizeof(float));
}

/* IEEE binary32 -> binary16, round to nearest even, including subnormals. */
static uint16_t gpt_fp32_to_fp16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t biased = (bits >> 23) & 0xffu;
    int32_t exponent = (int32_t)biased - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (biased == 0xffu)
        return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x200u : 0));
    if (exponent >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if (exponent < -10)
        return (uint16_t)sign;
    uint32_t shift = exponent <= 0 ? (uint32_t)(14 - exponent) : 13u;
    if (exponent <= 0)
        mantissa |= 0x800000u;
    uint32_t half = (exponent <= 0 ? 0u : (uint32_t)exponent << 10) | (mantissa >> shift);
    uint32_t rest = mantissa & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1u)))
        half++;
    return (uint16_t)(sign | half);
}

static BNNE_Status gpt_linear(BNNE_Engine engine, const GPT_Weight* weight,
                             const float* input, uint32_t rows, float* output,
                             BNNE_Stats* stats) {
    GPT_Model* model = engine->model;
    uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
    BNNE_Status status = BNNE_OK;
    for (uint32_t r = 0; r < rows && status == BNNE_OK; r++)
        status = bnne_quantize(input + (size_t)r * weight->in, weight->in,
                               model->quantized + (size_t)r * weight->in, &model->row_scales[r]);
    if (engine->options.enable_profile)
        stats->quantize_ns += bnne_now_ns() - start;
    if (status != BNNE_OK)
        return bnne_error(engine, status, "cannot quantize GPT activations");
    status = bnne_runtime_gemm(engine, weight->matrix, model->quantized, rows,
                               weight->in, output, weight->out, stats);
    if (status != BNNE_OK)
        return status;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t o = 0; o < weight->out; o++) {
            size_t i = (size_t)r * weight->out + o;
            float value = output[i] * model->row_scales[r];
            if (!isfinite(value))
                return bnne_error(engine, BNNE_ECOMPUTE, "GPT linear produced NaN or Inf");
            output[i] = weight->relu && value < 0.0f ? 0.0f : value;
        }
    if (engine->options.enable_profile) {
        uint64_t macs = (uint64_t)rows * weight->in * weight->out;
        stats->logical_macs += macs;
        stats->executed_macs += macs;
    }
    return BNNE_OK;
}

static BNNE_Status gpt_attention(BNNE_Engine engine, uint32_t n, uint32_t head,
                                BNNE_Stats* stats) {
    GPT_Model* model = engine->model;
    uint32_t C = model->channels, G = model->group;
    uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
    BNNE_Status status = BNNE_OK;
    for (uint32_t r = 0; r < n && status == BNNE_OK; r++) {
        size_t offset = (size_t)r * C + head * G;
        status = bnne_quantize(model->work[K] + offset, G,
                               model->quantized + (size_t)r * G, &model->row_scales[r]);
        float sum = 0.0f;
        uint8_t* bits = model->query_bits + (size_t)r * (G / 8);
        memset(bits, 0, G / 8);
        for (uint32_t j = 0; j < G; j++) {
            float query = model->work[Q][offset + j];
            sum += fabsf(query);
            if (query >= 0.0f)
                bits[j >> 3] |= (uint8_t)(1u << (j & 7));
        }
        model->query_scales[r] = gpt_fp32_to_fp16(sum / (float)G);
        if (!isfinite(bnne_fp16_to_fp32(model->query_scales[r])))
            status = BNNE_ECOMPUTE;
    }
    if (engine->options.enable_profile)
        stats->quantize_ns += bnne_now_ns() - start;
    if (status != BNNE_OK)
        return bnne_error(engine, status, "cannot quantize GPT attention");

    BNN_MatrixDesc desc = {
        .in_features = G, .out_features = n, .group_size = G,
        .weight_bits = model->query_bits, .weight_bytes = (size_t)n * (G / 8),
        .scale_fp16 = model->query_scales, .scale_count = n,
    };
    BNN_Matrix matrix = NULL;
    start = engine->options.enable_profile ? bnne_now_ns() : 0;
    status = bnne_accel_status(engine,
        BNN_MatrixCreate(engine->device, &desc, &matrix), "BNN_MatrixCreate(GPT query)");
    if (engine->options.enable_profile)
        stats->transfer_ns += bnne_now_ns() - start;
    if (status == BNNE_OK)
        status = bnne_runtime_gemm(engine, matrix, model->quantized, n, G,
                                   model->work[SCORES], n, stats);
    /* A failed create or execute must take the same destruction path. */
    start = engine->options.enable_profile ? bnne_now_ns() : 0;
    BNN_MatrixDestroy(matrix);
    if (engine->options.enable_profile)
        stats->transfer_ns += bnne_now_ns() - start;
    if (status != BNNE_OK)
        return status;
    if (engine->options.enable_profile) {
        uint64_t macs = (uint64_t)n * n * G;
        stats->logical_macs += macs;
        stats->executed_macs += macs;
    }

    float divisor = sqrtf((float)G);
    for (uint32_t tq = 0; tq < n; tq++) {
        float* probability = model->work[PROBS] + (size_t)tq * n;
        float maximum = -INFINITY;
        for (uint32_t tk = 0; tk <= tq; tk++) {
            /* GEMM returns [key][query]; restore [query][key] and key scale. */
            float score = model->work[SCORES][(size_t)tk * n + tq] * model->row_scales[tk] / divisor;
            if (!isfinite(score))
                return bnne_error(engine, BNNE_ECOMPUTE, "GPT attention produced NaN or Inf");
            probability[tk] = score;
            if (score > maximum)
                maximum = score;
        }
        float sum = 0.0f;
        for (uint32_t tk = 0; tk <= tq; tk++) {
            probability[tk] = expf(probability[tk] - maximum);
            sum += probability[tk];
        }
        for (uint32_t tk = 0; tk <= tq; tk++)
            probability[tk] /= sum;
        for (uint32_t tk = tq + 1; tk < n; tk++)
            probability[tk] = 0.0f;
        for (uint32_t j = 0; j < G; j++) {
            float value = 0.0f;
            for (uint32_t tk = 0; tk <= tq; tk++)
                value += probability[tk] * model->work[V][(size_t)tk * C + head * G + j];
            model->work[Y][(size_t)tq * C + head * G + j] = value;
        }
    }
    return BNNE_OK;
}

static BNNE_Status gpt_norm(BNNE_Engine engine, const float* gain, uint32_t n) {
    GPT_Model* model = engine->model;
    uint32_t C = model->channels;
    for (uint32_t t = 0; t < n; t++) {
        const float* input = model->work[X] + (size_t)t * C;
        float sum = 0.0f;
        for (uint32_t c = 0; c < C; c++)
            sum += input[c] * input[c];
        float rms = sqrtf(sum / (float)C + model->epsilon);
        if (!isfinite(rms))
            return bnne_error(engine, BNNE_ECOMPUTE, "GPT RMSNorm overflow");
        for (uint32_t c = 0; c < C; c++)
            model->work[NORM][(size_t)t * C + c] = input[c] / rms * gain[c];
    }
    return BNNE_OK;
}

static BNNE_Status gpt_run(BNNE_Engine engine, const BNNE_Tensor* input,
                          BNNE_Tensor* output, BNNE_Stats* stats) {
    GPT_Model* model = engine->model;
    uint32_t n = input->desc.dims[0], C = model->channels;
    const int32_t* ids = input->data;
    for (uint32_t t = 0; t < n; t++)
        for (uint32_t c = 0; c < C; c++)
            model->work[X][(size_t)t * C + c] =
                model->tokens[(size_t)ids[t] * C + c] + model->positions[(size_t)t * C + c];
    for (uint32_t l = 0; l < model->layers; l++) {
        GPT_Block* block = &model->blocks[l];
        BNNE_Status status = gpt_norm(engine, block->gain[0], n);
        if (status != BNNE_OK)
            return status;
        for (uint32_t w = WQ; w <= WV; w++) {
            status = gpt_linear(engine, &block->weights[w], model->work[NORM], n,
                                 model->work[Q + w], stats);
            if (status != BNNE_OK)
                return status;
        }
        for (uint32_t h = 0; h < model->heads; h++) {
            status = gpt_attention(engine, n, h, stats);
            if (status != BNNE_OK)
                return status;
        }
        status = gpt_linear(engine, &block->weights[WO], model->work[Y], n, model->work[O], stats);
        if (status != BNNE_OK)
            return status;
        for (size_t i = 0; i < (size_t)n * C; i++)
            model->work[X][i] += model->work[O][i];
        status = gpt_norm(engine, block->gain[1], n);
        if (status != BNNE_OK)
            return status;
        status = gpt_linear(engine, &block->weights[WFC], model->work[NORM], n, model->work[FF], stats);
        if (status != BNNE_OK)
            return status;
        status = gpt_linear(engine, &block->weights[WPROJ], model->work[FF], n, model->work[O], stats);
        if (status != BNNE_OK)
            return status;
        for (size_t i = 0; i < (size_t)n * C; i++)
            model->work[X][i] += model->work[O][i];
    }
    BNNE_Status status = gpt_norm(engine, model->gain, n);
    if (status != BNNE_OK)
        return status;
    return gpt_linear(engine, &model->head, model->work[NORM], n, output->data, stats);
}

static BNNE_Status gpt_token(BNNE_Engine engine, uint32_t id, const char** out_text) {
    GPT_Model* model = engine->model;
    if (!out_text)
        return BNNE_EINVAL;
    if (id >= model->vocab_size)
        return bnne_error(engine, BNNE_ERANGE, "GPT token ID is out of range");
    *out_text = model->vocabulary + model->offsets[id];
    return BNNE_OK;
}

const BNNE_ModelOps bnne_gpt_ops = {
    .load = gpt_load,
    .prepare = gpt_prepare,
    .run = gpt_run,
    .destroy = gpt_destroy,
    .token = gpt_token,
};
