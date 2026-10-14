#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LLAMA_MAX_VOCAB 65536u
#define LLAMA_MAX_SEQUENCE 4096u
#define LLAMA_MAX_CHANNELS 4096u
#define LLAMA_MAX_LAYERS 128u

enum { WQ, WK, WV, WO, W1, W2, W3, LLAMA_WEIGHTS };
enum { X, NORM, Q, K, V, Y, O, FF_GATE, FF_UP, SCORES, PROBS, LLAMA_WORKS };

typedef struct {
    uint32_t in, out, group, relu;
    uint8_t* bits;
    uint16_t* scales;
    BNN_Matrix matrix;
} Llama_Weight;

typedef struct {
    float* gain[2];  /* [0]: att_norm, [1]: ffn_norm */
    Llama_Weight weights[LLAMA_WEIGHTS];
} Llama_Block;

typedef struct {
    uint32_t vocab_size, length, channels, heads, kv_heads, layers, hidden_dim, group;
    float epsilon;
    float *tokens, *gain;
    Llama_Block* blocks;
    Llama_Weight head;
    char* vocabulary;
    uint32_t* offsets;
    float* work[LLAMA_WORKS];
    int8_t* quantized;
    float* row_scales;
    uint8_t* query_bits;
    uint16_t* query_scales;
    float* key_cache;
    float* value_cache;
    uint32_t cache_pos;
} Llama_Model;

static void* llama_alloc(size_t count, size_t size) {
    return count && size <= SIZE_MAX / count ? malloc(count * size) : NULL;
}

static void llama_weight_destroy(Llama_Weight* weight) {
    BNN_MatrixDestroy(weight->matrix);
    free(weight->bits);
    free(weight->scales);
}

static void llama_destroy(BNNE_Engine engine) {
    Llama_Model* model = engine->model;
    if (!model)
        return;
    if (model->blocks) {
        for (uint32_t l = 0; l < model->layers; l++) {
            Llama_Block* block = &model->blocks[l];
            free(block->gain[0]);
            free(block->gain[1]);
            for (uint32_t w = 0; w < LLAMA_WEIGHTS; w++)
                llama_weight_destroy(&block->weights[w]);
        }
    }
    llama_weight_destroy(&model->head);
    free(model->blocks);
    free(model->tokens);
    free(model->gain);
    free(model->vocabulary);
    free(model->offsets);
    for (uint32_t i = 0; i < LLAMA_WORKS; i++)
        free(model->work[i]);
    free(model->quantized);
    free(model->row_scales);
    free(model->query_bits);
    free(model->query_scales);
    free(model->key_cache);
    free(model->value_cache);
    free(model);
    engine->model = NULL;
}

static BNNE_Status llama_read_floats(BNNE_Reader* reader, float** out, size_t count) {
    *out = llama_alloc(count, sizeof **out);
    if (!*out)
        return BNNE_ENOMEM;
    for (size_t i = 0; i < count && reader->status == BNNE_OK; i++) {
        (*out)[i] = bnne_read_f32(reader);
        if (!isfinite((*out)[i]))
            return BNNE_EFORMAT;
    }
    return reader->status;
}

static BNNE_Status llama_read_weight(BNNE_Reader* reader, Llama_Weight* weight,
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
    weight->bits = llama_alloc(bytes, 1);
    weight->scales = llama_alloc(count, sizeof *weight->scales);
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

static BNNE_Status llama_load(BNNE_Engine engine, FILE* file) {
    Llama_Model* model = calloc(1, sizeof *model);
    if (!model)
        return BNNE_ENOMEM;
    engine->model = model;
    BNNE_Reader reader = {file, BNNE_OK};
    model->vocab_size = bnne_read_u32(&reader);
    model->length = bnne_read_u32(&reader);
    model->channels = bnne_read_u32(&reader);
    model->heads = bnne_read_u32(&reader);
    model->kv_heads = bnne_read_u32(&reader);
    model->layers = bnne_read_u32(&reader);
    model->hidden_dim = bnne_read_u32(&reader);
    model->group = bnne_read_u32(&reader);
    model->epsilon = bnne_read_f32(&reader);
    if (reader.status != BNNE_OK)
        return reader.status;

    uint32_t C = model->channels, G = model->group, H = model->heads;
    uint32_t H_kv = model->kv_heads, H_dim = model->hidden_dim;
    if (!model->vocab_size || model->vocab_size > LLAMA_MAX_VOCAB ||
        !model->length || model->length > LLAMA_MAX_SEQUENCE ||
        !C || C > LLAMA_MAX_CHANNELS || !model->layers || model->layers > LLAMA_MAX_LAYERS ||
        !G || !H || !H_kv || G % 8 || C % G || C % H || C / H != G ||
        H % H_kv != 0 || !H_dim || H_dim % G != 0 ||
        !isfinite(model->epsilon) || model->epsilon <= 0.0f)
        return bnne_error(engine, BNNE_EFORMAT, "invalid Llama-2 dimensions or epsilon");

    uint32_t kv_dim = (C * H_kv) / H;

    BNNE_Status status = llama_read_floats(&reader, &model->tokens, (size_t)model->vocab_size * C);
    if (status != BNNE_OK)
        return status;

    model->blocks = calloc(model->layers, sizeof *model->blocks);
    if (!model->blocks)
        return BNNE_ENOMEM;

    for (uint32_t l = 0; l < model->layers; l++) {
        Llama_Block* block = &model->blocks[l];
        status = llama_read_floats(&reader, &block->gain[0], C);
        if (status != BNNE_OK)
            return status;
        status = llama_read_weight(&reader, &block->weights[WQ], C, C, G, 0);
        if (status != BNNE_OK)
            return status;
        status = llama_read_weight(&reader, &block->weights[WK], C, kv_dim, G, 0);
        if (status != BNNE_OK)
            return status;
        status = llama_read_weight(&reader, &block->weights[WV], C, kv_dim, G, 0);
        if (status != BNNE_OK)
            return status;
        status = llama_read_weight(&reader, &block->weights[WO], C, C, G, 0);
        if (status != BNNE_OK)
            return status;

        status = llama_read_floats(&reader, &block->gain[1], C);
        if (status != BNNE_OK)
            return status;
        status = llama_read_weight(&reader, &block->weights[W1], C, H_dim, G, 0);  /* gate */
        if (status != BNNE_OK)
            return status;
        status = llama_read_weight(&reader, &block->weights[W2], H_dim, C, G, 0);  /* down */
        if (status != BNNE_OK)
            return status;
        status = llama_read_weight(&reader, &block->weights[W3], C, H_dim, G, 0);  /* up */
        if (status != BNNE_OK)
            return status;
    }

    status = llama_read_floats(&reader, &model->gain, C);
    if (status != BNNE_OK)
        return status;
    status = llama_read_weight(&reader, &model->head, C, model->vocab_size, G, 0);
    if (status != BNNE_OK)
        return status;

    model->vocabulary = llama_alloc(model->vocab_size, 256);
    model->offsets = llama_alloc(model->vocab_size, sizeof *model->offsets);
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

    engine->info.kind = BNNE_MODEL_LLAMA;
    engine->info.layer_count = model->layers;
    engine->info.input = (BNNE_TensorDesc){BNNE_I32, 1, {model->length}};
    engine->info.output = (BNNE_TensorDesc){BNNE_F32, 2, {model->length, model->vocab_size}};
    engine->info.vocabulary_size = model->vocab_size;
    engine->info.max_sequence_length = model->length;
    return BNNE_OK;
}

static BNNE_Status llama_upload(BNNE_Engine engine, Llama_Weight* weight) {
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
        BNN_MatrixCreate(engine->device, &desc, &weight->matrix), "BNN_MatrixCreate(Llama weight)");
    if (status == BNNE_OK) {
        free(weight->bits);
        free(weight->scales);
        weight->bits = NULL;
        weight->scales = NULL;
    }
    return status;
}

static BNNE_Status llama_prepare(BNNE_Engine engine) {
    Llama_Model* model = engine->model;
    BNN_Caps caps = {0};
    BNNE_Status status = bnne_runtime_open(engine, &caps);
    if (status != BNNE_OK)
        return status;

    uint32_t max_in = model->hidden_dim > model->channels ? model->hidden_dim : model->channels;
    uint32_t max_out = model->vocab_size > max_in ? model->vocab_size : max_in;
    if (model->length > max_out)
        max_out = model->length;

    if (caps.api_version != BNN_API_VERSION || !caps.supports_gemm ||
        caps.group_size != model->group || model->length > caps.max_rows ||
        max_in > caps.max_in_features || max_out > caps.max_out_features)
        return bnne_error(engine, BNNE_ENOTSUP, "Llama GEMM shapes exceed device capabilities");

    for (uint32_t l = 0; l < model->layers; l++)
        for (uint32_t w = 0; w < LLAMA_WEIGHTS; w++) {
            status = llama_upload(engine, &model->blocks[l].weights[w]);
            if (status != BNNE_OK)
                return status;
        }
    status = llama_upload(engine, &model->head);
    if (status != BNNE_OK)
        return status;

    size_t tc = (size_t)model->length * model->channels;
    size_t tt = (size_t)model->length * model->length;
    size_t th = (size_t)model->length * model->hidden_dim;

    model->work[X] = llama_alloc(tc, sizeof(float));
    model->work[NORM] = llama_alloc(tc, sizeof(float));
    model->work[Q] = llama_alloc(tc, sizeof(float));
    model->work[K] = llama_alloc(tc, sizeof(float));
    model->work[V] = llama_alloc(tc, sizeof(float));
    model->work[Y] = llama_alloc(tc, sizeof(float));
    model->work[O] = llama_alloc(tc, sizeof(float));
    model->work[FF_GATE] = llama_alloc(th, sizeof(float));
    model->work[FF_UP] = llama_alloc(th, sizeof(float));
    model->work[SCORES] = llama_alloc(tt, sizeof(float));
    model->work[PROBS] = llama_alloc(tt, sizeof(float));

    for (uint32_t i = 0; i < LLAMA_WORKS; i++)
        if (!model->work[i])
            return BNNE_ENOMEM;

    model->quantized = llama_alloc(model->length, max_in);
    model->row_scales = llama_alloc(model->length, sizeof(float));
    model->query_bits = llama_alloc(model->length, model->group / 8);
    model->query_scales = llama_alloc(model->length, sizeof(uint16_t));

    uint32_t kv_dim = (model->channels * model->kv_heads) / model->heads;
    size_t cache_elements = (size_t)model->layers * model->length * kv_dim;
    model->key_cache = llama_alloc(cache_elements, sizeof(float));
    model->value_cache = llama_alloc(cache_elements, sizeof(float));

    if (!model->quantized || !model->row_scales || !model->query_bits || !model->query_scales ||
        !model->key_cache || !model->value_cache)
        return BNNE_ENOMEM;

    return bnne_runtime_buffers(engine, (size_t)model->length * max_in,
                                (size_t)model->length * max_out * sizeof(float));
}

static uint16_t llama_fp32_to_fp16(float value) {
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

static BNNE_Status llama_linear(BNNE_Engine engine, const Llama_Weight* weight,
                                const float* input, uint32_t rows, float* output,
                                BNNE_Stats* stats) {
    Llama_Model* model = engine->model;
    uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
    BNNE_Status status = BNNE_OK;
    for (uint32_t r = 0; r < rows && status == BNNE_OK; r++)
        status = bnne_quantize(input + (size_t)r * weight->in, weight->in,
                               model->quantized + (size_t)r * weight->in, &model->row_scales[r]);
    if (engine->options.enable_profile)
        stats->quantize_ns += bnne_now_ns() - start;
    if (status != BNNE_OK)
        return bnne_error(engine, status, "cannot quantize Llama activations");

    status = bnne_runtime_gemm(engine, weight->matrix, model->quantized, rows,
                               weight->in, output, weight->out, stats);
    if (status != BNNE_OK)
        return status;

    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t o = 0; o < weight->out; o++) {
            size_t i = (size_t)r * weight->out + o;
            float value = output[i] * model->row_scales[r];
            if (!isfinite(value))
                return bnne_error(engine, BNNE_ECOMPUTE, "Llama linear produced NaN or Inf");
            output[i] = value;
        }

    if (engine->options.enable_profile) {
        uint64_t macs = (uint64_t)rows * weight->in * weight->out;
        stats->logical_macs += macs;
        stats->executed_macs += macs;
    }
    return BNNE_OK;
}

static BNNE_Status llama_attention(BNNE_Engine engine, uint32_t n, uint32_t head,
                                  BNNE_Stats* stats) {
    Llama_Model* model = engine->model;
    uint32_t C = model->channels, G = model->group;
    uint32_t kv_mul = model->heads / model->kv_heads;
    uint32_t kv_head = head / kv_mul;
    uint32_t kv_dim = (C * model->kv_heads) / model->heads;

    uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
    BNNE_Status status = BNNE_OK;

    /* 加速器视角：Key 为激活 int8，Query 为权重 W1 (+FP16 组 scale) */
    for (uint32_t r = 0; r < n && status == BNNE_OK; r++) {
        size_t k_offset = (size_t)r * kv_dim + kv_head * G;
        status = bnne_quantize(model->work[K] + k_offset, G,
                               model->quantized + (size_t)r * G, &model->row_scales[r]);

        size_t q_offset = (size_t)r * C + head * G;
        float sum = 0.0f;
        uint8_t* bits = model->query_bits + (size_t)r * (G / 8);
        memset(bits, 0, G / 8);
        for (uint32_t j = 0; j < G; j++) {
            float query = model->work[Q][q_offset + j];
            sum += fabsf(query);
            if (query >= 0.0f)
                bits[j >> 3] |= (uint8_t)(1u << (j & 7));
        }
        model->query_scales[r] = llama_fp32_to_fp16(sum / (float)G);
        if (!isfinite(bnne_fp16_to_fp32(model->query_scales[r])))
            status = BNNE_ECOMPUTE;
    }
    if (engine->options.enable_profile)
        stats->quantize_ns += bnne_now_ns() - start;
    if (status != BNNE_OK)
        return bnne_error(engine, status, "cannot quantize Llama attention");

    BNN_MatrixDesc desc = {
        .in_features = G, .out_features = n, .group_size = G,
        .weight_bits = model->query_bits, .weight_bytes = (size_t)n * (G / 8),
        .scale_fp16 = model->query_scales, .scale_count = n,
    };
    BNN_Matrix matrix = NULL;
    start = engine->options.enable_profile ? bnne_now_ns() : 0;
    status = bnne_accel_status(engine,
        BNN_MatrixCreate(engine->device, &desc, &matrix), "BNN_MatrixCreate(Llama query)");
    if (engine->options.enable_profile)
        stats->transfer_ns += bnne_now_ns() - start;

    if (status == BNNE_OK)
        status = bnne_runtime_gemm(engine, matrix, model->quantized, n, G,
                                   model->work[SCORES], n, stats);

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
            float score = model->work[SCORES][(size_t)tk * n + tq] * model->row_scales[tk] / divisor;
            if (!isfinite(score))
                return bnne_error(engine, BNNE_ECOMPUTE, "Llama attention produced NaN or Inf");
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
                value += probability[tk] * model->work[V][(size_t)tk * kv_dim + kv_head * G + j];
            model->work[Y][(size_t)tq * C + head * G + j] = value;
        }
    }
    return BNNE_OK;
}

static BNNE_Status llama_norm(BNNE_Engine engine, const float* gain, uint32_t n) {
    Llama_Model* model = engine->model;
    uint32_t C = model->channels;
    for (uint32_t t = 0; t < n; t++) {
        const float* input = model->work[X] + (size_t)t * C;
        float sum = 0.0f;
        for (uint32_t c = 0; c < C; c++)
            sum += input[c] * input[c];
        float rms = sqrtf(sum / (float)C + model->epsilon);
        if (!isfinite(rms))
            return bnne_error(engine, BNNE_ECOMPUTE, "Llama RMSNorm overflow");
        for (uint32_t c = 0; c < C; c++)
            model->work[NORM][(size_t)t * C + c] = input[c] / rms * gain[c];
    }
    return BNNE_OK;
}

static BNNE_Status llama_run(BNNE_Engine engine, const BNNE_Tensor* input,
                            BNNE_Tensor* output, BNNE_Stats* stats) {
    Llama_Model* model = engine->model;
    uint32_t n = input->desc.dims[0], C = model->channels, G = model->group;
    uint32_t H = model->heads, H_kv = model->kv_heads;
    uint32_t kv_dim = (C * H_kv) / H;
    const int32_t* ids = input->data;

    /* 1. Embedding lookup（Llama-2 使用 RoPE，无需绝对位置编码 pos_emb） */
    for (uint32_t t = 0; t < n; t++)
        memcpy(model->work[X] + (size_t)t * C,
               model->tokens + (size_t)ids[t] * C,
               (size_t)C * sizeof(float));

    for (uint32_t l = 0; l < model->layers; l++) {
        Llama_Block* block = &model->blocks[l];

        /* --- Attention 分支 --- */
        BNNE_Status status = llama_norm(engine, block->gain[0], n);
        if (status != BNNE_OK)
            return status;

        status = llama_linear(engine, &block->weights[WQ], model->work[NORM], n, model->work[Q], stats);
        if (status != BNNE_OK)
            return status;
        status = llama_linear(engine, &block->weights[WK], model->work[NORM], n, model->work[K], stats);
        if (status != BNNE_OK)
            return status;
        status = llama_linear(engine, &block->weights[WV], model->work[NORM], n, model->work[V], stats);
        if (status != BNNE_OK)
            return status;

        /* RoPE 旋转位置编码 (与 llama2.c 的 run.c 循环结构逐位一致) */
        for (uint32_t t = 0; t < n; t++) {
            for (uint32_t i = 0; i < G; i += 2) {
                float freq = 1.0f / powf(10000.0f, (float)i / (float)G);
                float val = (float)t * freq;
                float fcr = cosf(val);
                float fci = sinf(val);

                /* Query heads 旋转 */
                for (uint32_t h = 0; h < H; h++) {
                    float* q = model->work[Q] + (size_t)t * C + h * G;
                    float q0 = q[i], q1 = q[i + 1];
                    q[i]     = q0 * fcr - q1 * fci;
                    q[i + 1] = q0 * fci + q1 * fcr;
                }

                /* Key heads 旋转 */
                for (uint32_t h = 0; h < H_kv; h++) {
                    float* k = model->work[K] + (size_t)t * kv_dim + h * G;
                    float k0 = k[i], k1 = k[i + 1];
                    k[i]     = k0 * fcr - k1 * fci;
                    k[i + 1] = k0 * fci + k1 * fcr;
                }
            }
        }

        /* Attention 计算 */
        for (uint32_t h = 0; h < H; h++) {
            status = llama_attention(engine, n, h, stats);
            if (status != BNNE_OK)
                return status;
        }

        status = llama_linear(engine, &block->weights[WO], model->work[Y], n, model->work[O], stats);
        if (status != BNNE_OK)
            return status;

        for (size_t i = 0; i < (size_t)n * C; i++)
            model->work[X][i] += model->work[O][i];

        /* --- SwiGLU FFN 分支 --- */
        status = llama_norm(engine, block->gain[1], n);
        if (status != BNNE_OK)
            return status;

        /* w1 (gate) 与 w3 (up) */
        status = llama_linear(engine, &block->weights[W1], model->work[NORM], n, model->work[FF_GATE], stats);
        if (status != BNNE_OK)
            return status;
        status = llama_linear(engine, &block->weights[W3], model->work[NORM], n, model->work[FF_UP], stats);
        if (status != BNNE_OK)
            return status;

        /* SwiGLU: silu(w1) * w3 */
        for (size_t i = 0; i < (size_t)n * model->hidden_dim; i++) {
            float val = model->work[FF_GATE][i];
            float silu = val / (1.0f + expf(-val));
            model->work[FF_GATE][i] = silu * model->work[FF_UP][i];
        }

        /* w2 (down) */
        status = llama_linear(engine, &block->weights[W2], model->work[FF_GATE], n, model->work[O], stats);
        if (status != BNNE_OK)
            return status;

        for (size_t i = 0; i < (size_t)n * C; i++)
            model->work[X][i] += model->work[O][i];
    }

    /* 最终 RMSNorm 与 lm_head */
    BNNE_Status status = llama_norm(engine, model->gain, n);
    if (status != BNNE_OK)
        return status;

    return llama_linear(engine, &model->head, model->work[NORM], n, output->data, stats);
}

static BNNE_Status llama_token(BNNE_Engine engine, uint32_t id, const char** out_text) {
    Llama_Model* model = engine->model;
    if (!out_text)
        return BNNE_EINVAL;
    if (id >= model->vocab_size)
        return bnne_error(engine, BNNE_ERANGE, "Llama token ID is out of range");
    *out_text = model->vocabulary + model->offsets[id];
    return BNNE_OK;
}

static BNNE_Status llama_reset_cache(BNNE_Engine engine) {
    Llama_Model* model = engine->model;
    if (model)
        model->cache_pos = 0;
    return BNNE_OK;
}

static BNNE_Status llama_step(BNNE_Engine engine, int32_t token, uint32_t pos,
                             float* out_logits, BNNE_Stats* stats) {
    Llama_Model* model = engine->model;
    uint32_t C = model->channels, G = model->group;
    uint32_t H = model->heads, H_kv = model->kv_heads;
    uint32_t kv_dim = (C * H_kv) / H;
    uint32_t kv_mul = H / H_kv;

    if (pos >= model->length)
        return bnne_error(engine, BNNE_ERANGE, "Llama position exceeds max sequence length");
    if (pos == 0)
        model->cache_pos = 0;

    /* 1. Embedding lookup for current single token (rows = 1) */
    memcpy(model->work[X], model->tokens + (size_t)token * C, C * sizeof(float));

    for (uint32_t l = 0; l < model->layers; l++) {
        Llama_Block* block = &model->blocks[l];

        /* --- Attention 分支 (rows = 1) --- */
        BNNE_Status status = llama_norm(engine, block->gain[0], 1);
        if (status != BNNE_OK)
            return status;

        status = llama_linear(engine, &block->weights[WQ], model->work[NORM], 1, model->work[Q], stats);
        if (status != BNNE_OK)
            return status;
        status = llama_linear(engine, &block->weights[WK], model->work[NORM], 1, model->work[K], stats);
        if (status != BNNE_OK)
            return status;
        status = llama_linear(engine, &block->weights[WV], model->work[NORM], 1, model->work[V], stats);
        if (status != BNNE_OK)
            return status;

        /* RoPE 旋转位置编码 (只旋转当前位置 pos) */
        for (uint32_t i = 0; i < G; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)G);
            float val = (float)pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            for (uint32_t h = 0; h < H; h++) {
                float* q = model->work[Q] + h * G;
                float q0 = q[i], q1 = q[i + 1];
                q[i]     = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
            }

            for (uint32_t h = 0; h < H_kv; h++) {
                float* k = model->work[K] + h * G;
                float k0 = k[i], k1 = k[i + 1];
                k[i]     = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        /* 将旋转后的当前 K 和 V 存入 KV Cache */
        size_t cache_offset = ((size_t)l * model->length + pos) * kv_dim;
        memcpy(model->key_cache + cache_offset, model->work[K], kv_dim * sizeof(float));
        memcpy(model->value_cache + cache_offset, model->work[V], kv_dim * sizeof(float));

        /* Attention 计算：Query 为当前 1 个 Token，Key 为历史缓存 0..pos 共 (pos+1) 个 Token */
        uint32_t seq_len = pos + 1;
        float divisor = sqrtf((float)G);

        for (uint32_t h = 0; h < H; h++) {
            uint32_t kv_head = h / kv_mul;

            /* 加速器 GEMM：Key 为激活 int8 (seq_len 行)，Query 为权重 W1 (1 行，G=128 维) */
            uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;

            for (uint32_t t = 0; t < seq_len; t++) {
                size_t k_src = ((size_t)l * model->length + t) * kv_dim + kv_head * G;
                status = bnne_quantize(model->key_cache + k_src, G,
                                       model->quantized + (size_t)t * G, &model->row_scales[t]);
                if (status != BNNE_OK)
                    return bnne_error(engine, status, "cannot quantize cached key");
            }

            float* q_vec = model->work[Q] + h * G;
            float sum = 0.0f;
            memset(model->query_bits, 0, G / 8);
            for (uint32_t j = 0; j < G; j++) {
                float q_val = q_vec[j];
                sum += fabsf(q_val);
                if (q_val >= 0.0f)
                    model->query_bits[j >> 3] |= (uint8_t)(1u << (j & 7));
            }
            model->query_scales[0] = llama_fp32_to_fp16(sum / (float)G);
            if (!isfinite(bnne_fp16_to_fp32(model->query_scales[0])))
                return bnne_error(engine, BNNE_ECOMPUTE, "Llama query scale NaN or Inf");

            if (engine->options.enable_profile)
                stats->quantize_ns += bnne_now_ns() - start;

            BNN_MatrixDesc desc = {
                .in_features = G, .out_features = 1, .group_size = G,
                .weight_bits = model->query_bits, .weight_bytes = G / 8,
                .scale_fp16 = model->query_scales, .scale_count = 1,
            };
            BNN_Matrix matrix = NULL;
            start = engine->options.enable_profile ? bnne_now_ns() : 0;
            status = bnne_accel_status(engine,
                BNN_MatrixCreate(engine->device, &desc, &matrix), "BNN_MatrixCreate(Llama query step)");
            if (engine->options.enable_profile)
                stats->transfer_ns += bnne_now_ns() - start;

            if (status == BNNE_OK)
                status = bnne_runtime_gemm(engine, matrix, model->quantized, seq_len, G,
                                           model->work[SCORES], 1, stats);

            start = engine->options.enable_profile ? bnne_now_ns() : 0;
            BNN_MatrixDestroy(matrix);
            if (engine->options.enable_profile)
                stats->transfer_ns += bnne_now_ns() - start;
            if (status != BNNE_OK)
                return status;

            if (engine->options.enable_profile) {
                uint64_t macs = (uint64_t)seq_len * 1 * G;
                stats->logical_macs += macs;
                stats->executed_macs += macs;
            }

            /* Softmax 归一化并加权求和 Value Cache */
            float maximum = -INFINITY;
            for (uint32_t t = 0; t < seq_len; t++) {
                float score = model->work[SCORES][t] * model->row_scales[t] / divisor;
                if (!isfinite(score))
                    return bnne_error(engine, BNNE_ECOMPUTE, "Llama attention score NaN or Inf");
                model->work[SCORES][t] = score;
                if (score > maximum)
                    maximum = score;
            }
            float sum_exp = 0.0f;
            for (uint32_t t = 0; t < seq_len; t++) {
                float prob = expf(model->work[SCORES][t] - maximum);
                model->work[PROBS][t] = prob;
                sum_exp += prob;
            }
            for (uint32_t t = 0; t < seq_len; t++)
                model->work[PROBS][t] /= sum_exp;

            for (uint32_t j = 0; j < G; j++) {
                float val = 0.0f;
                for (uint32_t t = 0; t < seq_len; t++) {
                    size_t v_src = ((size_t)l * model->length + t) * kv_dim + kv_head * G + j;
                    val += model->work[PROBS][t] * model->value_cache[v_src];
                }
                model->work[Y][h * G + j] = val;
            }
        }

        status = llama_linear(engine, &block->weights[WO], model->work[Y], 1, model->work[O], stats);
        if (status != BNNE_OK)
            return status;

        for (uint32_t c = 0; c < C; c++)
            model->work[X][c] += model->work[O][c];

        /* --- SwiGLU FFN 分支 (rows = 1) --- */
        status = llama_norm(engine, block->gain[1], 1);
        if (status != BNNE_OK)
            return status;

        status = llama_linear(engine, &block->weights[W1], model->work[NORM], 1, model->work[FF_GATE], stats);
        if (status != BNNE_OK)
            return status;
        status = llama_linear(engine, &block->weights[W3], model->work[NORM], 1, model->work[FF_UP], stats);
        if (status != BNNE_OK)
            return status;

        for (uint32_t i = 0; i < model->hidden_dim; i++) {
            float v = model->work[FF_GATE][i];
            float silu = v / (1.0f + expf(-v));
            model->work[FF_GATE][i] = silu * model->work[FF_UP][i];
        }

        status = llama_linear(engine, &block->weights[W2], model->work[FF_GATE], 1, model->work[O], stats);
        if (status != BNNE_OK)
            return status;

        for (uint32_t c = 0; c < C; c++)
            model->work[X][c] += model->work[O][c];
    }

    /* 最终 RMSNorm 与 lm_head (rows = 1) */
    BNNE_Status status = llama_norm(engine, model->gain, 1);
    if (status != BNNE_OK)
        return status;

    status = llama_linear(engine, &model->head, model->work[NORM], 1, out_logits, stats);
    if (status != BNNE_OK)
        return status;

    model->cache_pos = pos + 1;
    return BNNE_OK;
}

const BNNE_ModelOps bnne_llama_ops = {
    .load = llama_load,
    .prepare = llama_prepare,
    .run = llama_run,
    .destroy = llama_destroy,
    .token = llama_token,
    .step = llama_step,
    .reset_cache = llama_reset_cache,
};
