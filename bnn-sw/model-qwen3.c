#include "internal.h"
#include "gguf.h"
#include "tokenizer-bpe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define QWEN3_GROUP 128u
#define QWEN3_MAX_FEATURES 4096u
#define QWEN3_MAX_SEQ 1024u
#define QWEN3_MAX_VOCAB (1u << 20)
#define QWEN3_MAX_LAYERS 128u
#define QWEN3_MAX_HEADS 256u
#define QWEN3_PI 3.14159265358979323846

typedef struct {
    uint32_t in_off, in_len, out_off, out_len;
    BNN_Matrix matrix;
} QwenChunk;

typedef struct {
    uint32_t in, out;
    QwenChunk* chunks;
    uint32_t n_chunks;
} QwenWeight;

typedef struct {
    float* attn_norm;
    float* q_norm;
    float* k_norm;
    float* ffn_norm;
    QwenWeight wq, wk, wv, wo;
    QwenWeight gate, up, down;
} QwenBlock;

typedef struct {
    BnneGguf* gguf;

    uint32_t V, C, H, Hk, HD, FF, L, kv_dim, max_seq;
    float eps;
    float rope_base, rope_freq_scale, rope_mscale;
    int32_t rope_corr0, rope_corr1;

    uint8_t* emb_bits;
    uint16_t* emb_scales;
    QwenWeight lm_head;
    float* final_norm;
    QwenBlock* blocks;

    float *x, *norm, *q, *k, *v, *y, *o, *gate, *up, *scores, *probs;
    int8_t* quantized;
    float* chunk_out;
    float* k_cache, *v_cache;
    float* rope_cos, *rope_sin;

    uint32_t cache_pos;

    char* token_text;
    uint32_t* token_off;
    uint32_t* token_len;
    uint32_t eos_id;
    int has_eos;
    uint32_t eog_ids[4];
    uint32_t eog_count;
    QwenTokenizer* tokenizer;
} QwenModel;

static void* qwen3_alloc(size_t count, size_t size) {
    if (!count || size > SIZE_MAX / count)
        return NULL;
    return malloc(count * size);
}

static uint32_t qwen3_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t qwen3_u16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static void qwen3_weight_destroy(QwenWeight* weight) {
    for (uint32_t i = 0; i < weight->n_chunks; i++)
        BNN_MatrixDestroy(weight->chunks[i].matrix);
    free(weight->chunks);
    memset(weight, 0, sizeof *weight);
}

static BNNE_Status qwen3_read_q1_0(BNNE_Engine engine, const BnneGguf* gguf,
                                   const BnneGgufTensor* tensor, uint32_t in, uint32_t out,
                                   uint8_t** out_bits, uint16_t** out_scales) {
    if (!tensor || tensor->type != 41 || tensor->rank != 2 ||
        tensor->dims[0] != in || tensor->dims[1] != out || in % QWEN3_GROUP != 0 ||
        tensor->bytes != (uint64_t)out * (in / QWEN3_GROUP) * 18)
        return bnne_error(engine, BNNE_EFORMAT, "unexpected Q1_0 tensor shape or type");

    size_t row_bytes = in / 8u;
    size_t groups = in / QWEN3_GROUP;
    uint8_t* bits = qwen3_alloc((size_t)out, row_bytes);
    uint16_t* scales = qwen3_alloc((size_t)out * groups, sizeof *scales);
    if (!bits || !scales) {
        free(bits);
        free(scales);
        return BNNE_ENOMEM;
    }

    const uint8_t* source = bnne_gguf_data(gguf, tensor);
    for (uint32_t o = 0; o < out; o++) {
        for (uint32_t g = 0; g < groups; g++) {
            const uint8_t* block = source + ((size_t)o * groups + g) * 18;
            uint16_t scale = qwen3_u16(block);
            if (!isfinite(bnne_fp16_to_fp32(scale))) {
                free(bits);
                free(scales);
                return bnne_error(engine, BNNE_EFORMAT, "Q1_0 scale is not finite");
            }
            scales[(size_t)o * groups + g] = scale;
            memcpy(bits + (size_t)o * row_bytes + g * 16, block + 2, 16);
        }
    }
    *out_bits = bits;
    *out_scales = scales;
    return BNNE_OK;
}

static BNNE_Status qwen3_chunk_create(BNNE_Engine engine, QwenChunk* chunk,
                                      uint32_t in, uint32_t out,
                                      const uint8_t* bits, const uint16_t* scales) {
    BNN_MatrixDesc desc = {
        .in_features = in,
        .out_features = out,
        .group_size = QWEN3_GROUP,
        .weight_bits = bits,
        .weight_bytes = (size_t)out * (in / 8u),
        .scale_fp16 = scales,
        .scale_count = (size_t)out * (in / QWEN3_GROUP),
    };
    return bnne_accel_status(engine,
        BNN_MatrixCreate(engine->device, &desc, &chunk->matrix), "BNN_MatrixCreate(Qwen3)");
}

static BNNE_Status qwen3_make_weight(BNNE_Engine engine, QwenWeight* weight,
                                     uint32_t in, uint32_t out,
                                     const uint8_t* bits, const uint16_t* scales) {
    memset(weight, 0, sizeof *weight);
    weight->in = in;
    weight->out = out;

    if (in <= QWEN3_MAX_FEATURES && out <= QWEN3_MAX_FEATURES) {
        weight->n_chunks = 1;
        weight->chunks = calloc(1, sizeof *weight->chunks);
        if (!weight->chunks)
            return BNNE_ENOMEM;
        QwenChunk* chunk = &weight->chunks[0];
        chunk->in_len = in;
        chunk->out_len = out;
        return qwen3_chunk_create(engine, chunk, in, out, bits, scales);
    }

    if (in <= QWEN3_MAX_FEATURES) {
        uint32_t count = (out + QWEN3_MAX_FEATURES - 1) / QWEN3_MAX_FEATURES;
        weight->chunks = calloc(count, sizeof *weight->chunks);
        if (!weight->chunks)
            return BNNE_ENOMEM;
        weight->n_chunks = count;
        for (uint32_t c = 0; c < count; c++) {
            QwenChunk* chunk = &weight->chunks[c];
            uint32_t out_off = c * QWEN3_MAX_FEATURES;
            uint32_t out_len = out - out_off < QWEN3_MAX_FEATURES ? out - out_off : QWEN3_MAX_FEATURES;
            chunk->out_off = out_off;
            chunk->out_len = out_len;
            chunk->in_len = in;
            BNNE_Status status = qwen3_chunk_create(engine, chunk, in, out_len,
                bits + (size_t)out_off * (in / 8u),
                scales + (size_t)out_off * (in / QWEN3_GROUP));
            if (status != BNNE_OK) {
                qwen3_weight_destroy(weight);
                return status;
            }
        }
        return BNNE_OK;
    }

    if (out <= QWEN3_MAX_FEATURES) {
        uint32_t count = (in + QWEN3_MAX_FEATURES - 1) / QWEN3_MAX_FEATURES;
        weight->chunks = calloc(count, sizeof *weight->chunks);
        if (!weight->chunks)
            return BNNE_ENOMEM;
        weight->n_chunks = count;
        for (uint32_t c = 0; c < count; c++) {
            QwenChunk* chunk = &weight->chunks[c];
            uint32_t in_off = c * QWEN3_MAX_FEATURES;
            uint32_t in_len = in - in_off < QWEN3_MAX_FEATURES ? in - in_off : QWEN3_MAX_FEATURES;
            size_t row_bytes = in_len / 8u;
            size_t groups = in_len / QWEN3_GROUP;
            uint8_t* part_bits = qwen3_alloc((size_t)out, row_bytes);
            uint16_t* part_scales = qwen3_alloc((size_t)out * groups, sizeof *part_scales);
            if (!part_bits || !part_scales) {
                free(part_bits);
                free(part_scales);
                qwen3_weight_destroy(weight);
                return BNNE_ENOMEM;
            }
            for (uint32_t o = 0; o < out; o++) {
                memcpy(part_bits + (size_t)o * row_bytes,
                       bits + (size_t)o * (in / 8u) + in_off / 8u, row_bytes);
                memcpy(part_scales + (size_t)o * groups,
                       scales + (size_t)o * (in / QWEN3_GROUP) + in_off / QWEN3_GROUP,
                       groups * sizeof *part_scales);
            }
            chunk->in_off = in_off;
            chunk->in_len = in_len;
            chunk->out_len = out;
            BNNE_Status status = qwen3_chunk_create(engine, chunk, in_len, out,
                                                    part_bits, part_scales);
            free(part_bits);
            free(part_scales);
            if (status != BNNE_OK) {
                qwen3_weight_destroy(weight);
                return status;
            }
        }
        return BNNE_OK;
    }

    return bnne_error(engine, BNNE_ENOTSUP, "Qwen3 weight exceeds both chunk limits");
}

static BNNE_Status qwen3_read_f32(BNNE_Engine engine, const BnneGguf* gguf, const char* name,
                                  uint32_t count, float** out) {
    const BnneGgufTensor* tensor = bnne_gguf_find(gguf, name);
    if (!tensor || tensor->type != 0 || tensor->rank != 1 ||
        tensor->dims[0] != count || tensor->bytes != (uint64_t)count * 4)
        return bnne_error(engine, BNNE_EFORMAT, "missing or invalid F32 tensor '%s'", name);
    float* values = qwen3_alloc(count, sizeof *values);
    if (!values)
        return BNNE_ENOMEM;
    const uint8_t* source = bnne_gguf_data(gguf, tensor);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t bits = qwen3_u32(source + (size_t)i * 4);
        memcpy(&values[i], &bits, sizeof bits);
        if (!isfinite(values[i])) {
            free(values);
            return bnne_error(engine, BNNE_EFORMAT, "non-finite value in F32 tensor '%s'", name);
        }
    }
    *out = values;
    return BNNE_OK;
}

static BNNE_Status qwen3_load_weight(BNNE_Engine engine, const BnneGguf* gguf,
                                     const char* name, uint32_t in, uint32_t out,
                                     QwenWeight* weight) {
    const BnneGgufTensor* tensor = bnne_gguf_find(gguf, name);
    uint8_t* bits = NULL;
    uint16_t* scales = NULL;
    BNNE_Status status = qwen3_read_q1_0(engine, gguf, tensor, in, out, &bits, &scales);
    if (status == BNNE_OK)
        status = qwen3_make_weight(engine, weight, in, out, bits, scales);
    free(bits);
    free(scales);
    return status;
}

static void qwen3_build_byte_table(int32_t map[512]) {
    for (uint32_t i = 0; i < 512; i++)
        map[i] = -1;
    int used[256] = {0};
    for (int b = 33; b <= 126; b++) {
        used[b] = 1;
        map[b] = b;
    }
    for (int b = 161; b <= 172; b++) {
        used[b] = 1;
        map[b] = b;
    }
    for (int b = 174; b <= 255; b++) {
        used[b] = 1;
        map[b] = b;
    }
    int next = 0;
    for (int b = 0; b < 256; b++) {
        if (!used[b])
            map[256 + next++] = b;
    }
}

static uint32_t qwen3_utf8_decode(const uint8_t* source, uint32_t length, uint32_t* used) {
    if (!length) {
        *used = 0;
        return UINT32_MAX;
    }
    uint8_t b0 = source[0];
    if (b0 < 0x80) {
        *used = 1;
        return b0;
    }
    if ((b0 & 0xe0) == 0xc0 && length >= 2 && (source[1] & 0xc0) == 0x80) {
        *used = 2;
        return ((uint32_t)(b0 & 0x1f) << 6) | (source[1] & 0x3f);
    }
    if ((b0 & 0xf0) == 0xe0 && length >= 3 && (source[1] & 0xc0) == 0x80 &&
        (source[2] & 0xc0) == 0x80) {
        *used = 3;
        return ((uint32_t)(b0 & 0x0f) << 12) |
               ((uint32_t)(source[1] & 0x3f) << 6) | (source[2] & 0x3f);
    }
    if ((b0 & 0xf8) == 0xf0 && length >= 4 && (source[1] & 0xc0) == 0x80 &&
        (source[2] & 0xc0) == 0x80 && (source[3] & 0xc0) == 0x80) {
        *used = 4;
        return ((uint32_t)(b0 & 0x07) << 18) |
               ((uint32_t)(source[1] & 0x3f) << 12) |
               ((uint32_t)(source[2] & 0x3f) << 6) | (source[3] & 0x3f);
    }
    *used = 1;
    return UINT32_MAX;
}

static uint32_t qwen3_detokenize(const uint8_t* source, uint32_t length,
                                 const int32_t map[512], uint8_t* dst) {
    uint32_t written = 0, pos = 0;
    while (pos < length) {
        uint32_t used = 0;
        uint32_t cp = qwen3_utf8_decode(source + pos, length - pos, &used);
        if (cp != UINT32_MAX && cp < 512 && map[cp] >= 0) {
            dst[written++] = (uint8_t)map[cp];
        } else {
            memcpy(dst + written, source + pos, used);
            written += used;
        }
        pos += used;
    }
    return written;
}

static BNNE_Status qwen3_build_tokens(BNNE_Engine engine, QwenModel* model) {
    BnneGguf* gguf = model->gguf;
    if (!gguf->tokens || gguf->token_count != model->V)
        return bnne_error(engine, BNNE_EFORMAT, "Qwen3 tokenizer vocabulary does not match model");
    int32_t map[512];
    qwen3_build_byte_table(map);

    uint64_t total = 0;
    for (uint32_t i = 0; i < model->V; i++) {
        if (total > SIZE_MAX - gguf->tokens[i].length - 1)
            return BNNE_ERANGE;
        total += gguf->tokens[i].length + 1;
    }
    model->token_text = qwen3_alloc((size_t)total, 1);
    model->token_off = qwen3_alloc(model->V, sizeof *model->token_off);
    model->token_len = qwen3_alloc(model->V, sizeof *model->token_len);
    if (!model->token_text || !model->token_off || !model->token_len)
        return BNNE_ENOMEM;

    size_t used = 0;
    for (uint32_t i = 0; i < model->V; i++) {
        const BnneGgufString* token = &gguf->tokens[i];
        uint32_t type = gguf->token_types && i < gguf->token_type_count ?
                        gguf->token_types[i] : 1;
        uint32_t offset = (uint32_t)used;
        if (type == 3 || type == 4) {
            memcpy(model->token_text + used, token->data, token->length);
            used += token->length;
        } else {
            used += qwen3_detokenize(token->data, token->length, map,
                                     (uint8_t*)model->token_text + used);
        }
        model->token_text[used++] = '\0';
        model->token_off[i] = offset;
        model->token_len[i] = (uint32_t)used - 1 - offset;
    }
    return BNNE_OK;
}

static void qwen3_add_eog(QwenModel* model, uint32_t id) {
    if (model->eog_count >= 4)
        return;
    for (uint32_t i = 0; i < model->eog_count; i++)
        if (model->eog_ids[i] == id)
            return;
    model->eog_ids[model->eog_count++] = id;
}

static BNNE_Status qwen3_load(BNNE_Engine engine, FILE* file) {
    QwenModel* model = calloc(1, sizeof *model);
    if (!model)
        return BNNE_ENOMEM;
    engine->model = model;

    char error[192];
    BNNE_Status status = bnne_gguf_open(file, &model->gguf, error, sizeof error);
    if (status != BNNE_OK)
        return bnne_error(engine, status, "%s", error);
    BnneGguf* gguf = model->gguf;

    const BnneGgufTensor* embedding = bnne_gguf_find(gguf, "token_embd.weight");
    if (!embedding || embedding->type != 41 || embedding->rank != 2) {
        return bnne_error(engine, BNNE_EFORMAT, "missing token_embd.weight");
    }
    if (gguf->block_count > QWEN3_MAX_LAYERS || gguf->embedding_length > UINT32_MAX ||
        gguf->feed_forward_length > UINT32_MAX || gguf->head_count > QWEN3_MAX_HEADS ||
        gguf->head_count_kv > QWEN3_MAX_HEADS || gguf->key_length > UINT32_MAX ||
        gguf->head_count_kv == 0 || gguf->key_length == 0 ||
        embedding->dims[1] > QWEN3_MAX_VOCAB)
        return bnne_error(engine, BNNE_EFORMAT, "unsupported Qwen3 dimensions");

    model->L = (uint32_t)gguf->block_count;
    model->C = (uint32_t)gguf->embedding_length;
    model->FF = (uint32_t)gguf->feed_forward_length;
    model->H = (uint32_t)gguf->head_count;
    model->Hk = (uint32_t)gguf->head_count_kv;
    model->HD = (uint32_t)gguf->key_length;
    model->V = (uint32_t)embedding->dims[1];
    model->eps = gguf->rms_epsilon;
    model->rope_base = gguf->rope_freq_base;
    model->max_seq = QWEN3_MAX_SEQ;
    if (gguf->context_length && gguf->context_length < model->max_seq)
        model->max_seq = (uint32_t)gguf->context_length;

    if (!model->L || !model->C || !model->FF || !model->H || !model->V ||
        !model->HD || model->HD > model->C ||
        (uint64_t)model->H * model->HD != model->C ||
        model->H % model->Hk != 0 ||
        model->C % QWEN3_GROUP != 0 || model->FF % QWEN3_GROUP != 0 ||
        model->HD % 2 != 0 || model->HD != gguf->value_length)
        return bnne_error(engine, BNNE_EFORMAT, "inconsistent Qwen3 dimensions");

    model->kv_dim = model->Hk * model->HD;
    if (model->kv_dim % QWEN3_GROUP != 0)
        return bnne_error(engine, BNNE_EFORMAT, "Qwen3 KV width is not group aligned");

    model->blocks = calloc(model->L, sizeof *model->blocks);
    if (!model->blocks)
        return BNNE_ENOMEM;

    status = qwen3_read_f32(engine, gguf, "output_norm.weight", model->C, &model->final_norm);
    if (status != BNNE_OK)
        return status;

    for (uint32_t l = 0; l < model->L; l++) {
        QwenBlock* block = &model->blocks[l];
        char name[64];
        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", l);
        status = qwen3_read_f32(engine, gguf, name, model->C, &block->attn_norm);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.ffn_norm.weight", l);
        status = qwen3_read_f32(engine, gguf, name, model->C, &block->ffn_norm);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.attn_q_norm.weight", l);
        status = qwen3_read_f32(engine, gguf, name, model->HD, &block->q_norm);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.attn_k_norm.weight", l);
        status = qwen3_read_f32(engine, gguf, name, model->HD, &block->k_norm);
        if (status != BNNE_OK) return status;
    }

    if (gguf->has_eos && gguf->eos_token_id < model->V) {
        model->eos_id = gguf->eos_token_id;
        model->has_eos = 1;
    }
    if (model->has_eos)
        qwen3_add_eog(model, model->eos_id);
    for (uint32_t i = 0; i < model->V; i++) {
        const BnneGgufString* token = &gguf->tokens[i];
        if ((token->length == 13 && memcmp(token->data, "<|endoftext|>", 13) == 0) ||
            (token->length == 10 && memcmp(token->data, "<|im_end|>", 10) == 0))
            qwen3_add_eog(model, i);
    }

    status = qwen3_build_tokens(engine, model);
    if (status != BNNE_OK)
        return status;
    status = bnne_qwen_tokenizer_create(gguf, &model->tokenizer, error, sizeof error);
    if (status != BNNE_OK)
        return bnne_error(engine, status, "%s", error);

    engine->info.kind = BNNE_MODEL_QWEN3;
    engine->info.layer_count = model->L;
    engine->info.input = (BNNE_TensorDesc){BNNE_I32, 1, {model->max_seq}};
    engine->info.output = (BNNE_TensorDesc){BNNE_F32, 2, {model->max_seq, model->V}};
    engine->info.vocabulary_size = model->V;
    engine->info.max_sequence_length = model->max_seq;
    return BNNE_OK;
}

static int32_t qwen3_clamp_dim(double value, int32_t low, int32_t high) {
    if (value <= (double)low) return low;
    if (value >= (double)high) return high;
    return (int32_t)value;
}

static BNNE_Status qwen3_init_rope(BNNE_Engine engine, QwenModel* model) {
    uint32_t half = model->HD / 2;
    float freq_scale = 1.0f / model->gguf->rope_factor;
    int yarn = strcmp(model->gguf->rope_scaling_type, "yarn") == 0;
    int32_t corr0 = 0, corr1 = (int32_t)(model->HD - 1);
    float mscale = 1.0f;

    if (yarn && model->gguf->rope_original_context) {
        double base = model->rope_base;
        double original = (double)model->gguf->rope_original_context;
        double start = floor((double)model->HD *
                             log(original / (32.0 * 2.0 * QWEN3_PI)) / (2.0 * log(base)));
        double end = ceil((double)model->HD *
                          log(original / (2.0 * QWEN3_PI)) / (2.0 * log(base)));
        corr0 = qwen3_clamp_dim(start, 0, (int32_t)model->HD - 1);
        corr1 = qwen3_clamp_dim(end, 0, (int32_t)model->HD - 1);
        mscale = 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    if (!(freq_scale > 0.0f) || !isfinite(mscale))
        return bnne_error(engine, BNNE_EFORMAT, "invalid Qwen3 rope scaling");

    model->rope_freq_scale = freq_scale;
    model->rope_mscale = mscale;
    model->rope_corr0 = corr0;
    model->rope_corr1 = corr1;

    model->rope_cos = qwen3_alloc((size_t)model->max_seq * half, sizeof(float));
    model->rope_sin = qwen3_alloc((size_t)model->max_seq * half, sizeof(float));
    if (!model->rope_cos || !model->rope_sin)
        return BNNE_ENOMEM;

    for (uint32_t pos = 0; pos < model->max_seq; pos++) {
        for (uint32_t j = 0; j < half; j++) {
            float theta_extrap = (float)pos * powf(model->rope_base,
                                                   -2.0f * (float)j / (float)model->HD);
            float theta = freq_scale * theta_extrap;
            if (yarn) {
                float span = (float)(corr1 - corr0);
                float denominator = span > 0.001f ? span : 0.001f;
                float y = ((float)j - (float)corr0) / denominator;
                float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
                theta = (freq_scale * theta_extrap) * (1.0f - ramp) + theta_extrap * ramp;
            }
            model->rope_cos[(size_t)pos * half + j] = cosf(theta) * mscale;
            model->rope_sin[(size_t)pos * half + j] = sinf(theta) * mscale;
        }
    }
    return BNNE_OK;
}

static BNNE_Status qwen3_prepare(BNNE_Engine engine) {
    QwenModel* model = engine->model;
    BNN_Caps caps = {0};
    BNNE_Status status = bnne_runtime_open(engine, &caps);
    if (status != BNNE_OK)
        return status;
    if (caps.api_version != BNN_API_VERSION || !caps.supports_gemm ||
        caps.group_size != QWEN3_GROUP ||
        caps.max_in_features < QWEN3_MAX_FEATURES ||
        caps.max_out_features < QWEN3_MAX_FEATURES || caps.max_rows < 1)
        return bnne_error(engine, BNNE_ENOTSUP,
                          "Qwen3 requires group 128 and 4096-wide GEMV/GEMM support");

    status = qwen3_read_q1_0(engine, model->gguf,
        bnne_gguf_find(model->gguf, "token_embd.weight"), model->C, model->V,
        &model->emb_bits, &model->emb_scales);
    if (status != BNNE_OK)
        return status;
    status = qwen3_make_weight(engine, &model->lm_head, model->C, model->V,
                               model->emb_bits, model->emb_scales);
    if (status != BNNE_OK)
        return status;

    for (uint32_t l = 0; l < model->L; l++) {
        QwenBlock* block = &model->blocks[l];
        char name[64];
        snprintf(name, sizeof name, "blk.%u.attn_q.weight", l);
        status = qwen3_load_weight(engine, model->gguf, name, model->C, model->C, &block->wq);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.attn_k.weight", l);
        status = qwen3_load_weight(engine, model->gguf, name, model->C, model->kv_dim, &block->wk);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.attn_v.weight", l);
        status = qwen3_load_weight(engine, model->gguf, name, model->C, model->kv_dim, &block->wv);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.attn_output.weight", l);
        status = qwen3_load_weight(engine, model->gguf, name, model->C, model->C, &block->wo);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.ffn_gate.weight", l);
        status = qwen3_load_weight(engine, model->gguf, name, model->C, model->FF, &block->gate);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.ffn_up.weight", l);
        status = qwen3_load_weight(engine, model->gguf, name, model->C, model->FF, &block->up);
        if (status != BNNE_OK) return status;
        snprintf(name, sizeof name, "blk.%u.ffn_down.weight", l);
        status = qwen3_load_weight(engine, model->gguf, name, model->FF, model->C, &block->down);
        if (status != BNNE_OK) return status;
    }

    uint32_t max_in = model->C > model->FF ? model->C : model->FF;
    model->x = qwen3_alloc(model->C, sizeof(float));
    model->norm = qwen3_alloc(model->C, sizeof(float));
    model->q = qwen3_alloc(model->C, sizeof(float));
    model->y = qwen3_alloc(model->C, sizeof(float));
    model->o = qwen3_alloc(model->C, sizeof(float));
    model->k = qwen3_alloc(model->kv_dim, sizeof(float));
    model->v = qwen3_alloc(model->kv_dim, sizeof(float));
    model->gate = qwen3_alloc(model->FF, sizeof(float));
    model->up = qwen3_alloc(model->FF, sizeof(float));
    model->scores = qwen3_alloc(model->max_seq, sizeof(float));
    model->probs = qwen3_alloc(model->max_seq, sizeof(float));
    model->quantized = qwen3_alloc(max_in, 1);
    model->chunk_out = qwen3_alloc(QWEN3_MAX_FEATURES, sizeof(float));
    size_t cache_count = (size_t)model->L * model->max_seq * model->kv_dim;
    model->k_cache = qwen3_alloc(cache_count, sizeof(float));
    model->v_cache = qwen3_alloc(cache_count, sizeof(float));
    if (!model->x || !model->norm || !model->q || !model->y || !model->o ||
        !model->k || !model->v || !model->gate || !model->up ||
        !model->scores || !model->probs || !model->quantized || !model->chunk_out ||
        !model->k_cache || !model->v_cache)
        return BNNE_ENOMEM;

    status = qwen3_init_rope(engine, model);
    if (status != BNNE_OK)
        return status;

    return bnne_runtime_buffers(engine, QWEN3_MAX_FEATURES,
                                QWEN3_MAX_FEATURES * sizeof(float));
}

static BNNE_Status qwen3_linear(BNNE_Engine engine, QwenWeight* weight,
                                const float* input, float* output, BNNE_Stats* stats) {
    QwenModel* model = engine->model;
    float scale = 0.0f;
    BNNE_Status status = bnne_quantize(input, weight->in, model->quantized, &scale);
    if (status != BNNE_OK)
        return bnne_error(engine, status, "cannot quantize Qwen3 activation");

    memset(output, 0, (size_t)weight->out * sizeof(float));
    for (uint32_t c = 0; c < weight->n_chunks; c++) {
        QwenChunk* chunk = &weight->chunks[c];
        status = bnne_runtime_gemm(engine, chunk->matrix, model->quantized + chunk->in_off,
                                   1, chunk->in_len, model->chunk_out,
                                   chunk->out_len, stats);
        if (status != BNNE_OK)
            return status;
        for (uint32_t i = 0; i < chunk->out_len; i++)
            output[chunk->out_off + i] += model->chunk_out[i] * scale;
    }
    if (stats) {
        uint64_t macs = (uint64_t)weight->in * weight->out;
        stats->logical_macs += macs;
        stats->executed_macs += macs;
    }
    return BNNE_OK;
}

static BNNE_Status qwen3_rmsnorm(BNNE_Engine engine, const float* input, uint32_t count,
                                 float epsilon, const float* gain, float* output) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++)
        sum += input[i] * input[i];
    float rms = sqrtf(sum / (float)count + epsilon);
    if (!isfinite(rms) || rms == 0.0f)
        return bnne_error(engine, BNNE_ECOMPUTE, "Qwen3 RMSNorm produced a non-finite divisor");
    for (uint32_t i = 0; i < count; i++)
        output[i] = input[i] / rms * gain[i];
    return BNNE_OK;
}

static void qwen3_embed(QwenModel* model, uint32_t token, float* output) {
    const uint8_t* bits = model->emb_bits + (size_t)token * (model->C / 8u);
    const uint16_t* scales = model->emb_scales + (size_t)token * (model->C / QWEN3_GROUP);
    uint32_t groups = model->C / QWEN3_GROUP;
    for (uint32_t g = 0; g < groups; g++) {
        float scale = bnne_fp16_to_fp32(scales[g]);
        for (uint32_t j = 0; j < QWEN3_GROUP; j++) {
            uint8_t bit = (uint8_t)((bits[g * 16 + (j >> 3)] >> (j & 7u)) & 1u);
            output[g * QWEN3_GROUP + j] = bit ? scale : -scale;
        }
    }
}

static void qwen3_rope(float* vector, const float* cosine, const float* sine, uint32_t half) {
    for (uint32_t j = 0; j < half; j++) {
        float a = vector[j];
        float b = vector[j + half];
        vector[j] = a * cosine[j] - b * sine[j];
        vector[j + half] = a * sine[j] + b * cosine[j];
    }
}

/* 环境变量 BNN_QWEN3_DUMP_DIR 存在时导出前两层的中间状态，用于数值差分。 */
static void qwen3_debug_dump(const char* dir, const char* tag, uint32_t pos,
                             const float* data, size_t count) {
    if (!dir || pos >= 2)
        return;
    char path[512];
    int length = snprintf(path, sizeof path, "%s/%s_p%u.bin", dir, tag, pos);
    if (length < 0 || (size_t)length >= sizeof path)
        return;
    FILE* file = fopen(path, "wb");
    if (!file)
        return;
    fwrite(data, sizeof(float), count, file);
    fclose(file);
}

static BNNE_Status qwen3_step_core(BNNE_Engine engine, uint32_t token, uint32_t pos,
                                   float* logits, BNNE_Stats* stats) {
    QwenModel* model = engine->model;
    if (token >= model->V || pos >= model->max_seq)
        return bnne_error(engine, BNNE_ERANGE, "Qwen3 token or position is out of range");
    if (pos == 0)
        model->cache_pos = 0;
    if (pos != model->cache_pos)
        return bnne_error(engine, BNNE_EINVAL, "Qwen3 positions must be fed sequentially");

    qwen3_embed(model, token, model->x);
    const char* dump = getenv("BNN_QWEN3_DUMP_DIR");
    qwen3_debug_dump(dump, "emb", pos, model->x, model->C);
    uint32_t kv_mul = model->H / model->Hk;
    uint32_t half = model->HD / 2;
    float attention_scale = 1.0f / sqrtf((float)model->HD);

    for (uint32_t l = 0; l < model->L; l++) {
        QwenBlock* block = &model->blocks[l];
        char tag[64];
        BNNE_Status status = qwen3_rmsnorm(engine, model->x, model->C, model->eps,
                                           block->attn_norm, model->norm);
        if (status != BNNE_OK) return status;
        status = qwen3_linear(engine, &block->wq, model->norm, model->q, stats);
        if (status != BNNE_OK) return status;
        status = qwen3_linear(engine, &block->wk, model->norm, model->k, stats);
        if (status != BNNE_OK) return status;
        status = qwen3_linear(engine, &block->wv, model->norm, model->v, stats);
        if (status != BNNE_OK) return status;
        snprintf(tag, sizeof tag, "l%u_qproj", l);
        qwen3_debug_dump(dump, tag, pos, model->q, model->C);
        snprintf(tag, sizeof tag, "l%u_kproj", l);
        qwen3_debug_dump(dump, tag, pos, model->k, model->kv_dim);

        for (uint32_t h = 0; h < model->H; h++) {
            status = qwen3_rmsnorm(engine, model->q + (size_t)h * model->HD, model->HD,
                                   model->eps, block->q_norm,
                                   model->q + (size_t)h * model->HD);
            if (status != BNNE_OK) return status;
        }
        for (uint32_t h = 0; h < model->Hk; h++) {
            status = qwen3_rmsnorm(engine, model->k + (size_t)h * model->HD, model->HD,
                                   model->eps, block->k_norm,
                                   model->k + (size_t)h * model->HD);
            if (status != BNNE_OK) return status;
        }

        const float* cosine = model->rope_cos + (size_t)pos * half;
        const float* sine = model->rope_sin + (size_t)pos * half;
        for (uint32_t h = 0; h < model->H; h++)
            qwen3_rope(model->q + (size_t)h * model->HD, cosine, sine, half);
        for (uint32_t h = 0; h < model->Hk; h++)
            qwen3_rope(model->k + (size_t)h * model->HD, cosine, sine, half);
        snprintf(tag, sizeof tag, "l%u_qrope", l);
        qwen3_debug_dump(dump, tag, pos, model->q, model->C);
        snprintf(tag, sizeof tag, "l%u_krope", l);
        qwen3_debug_dump(dump, tag, pos, model->k, model->kv_dim);

        size_t cache_base = ((size_t)l * model->max_seq + pos) * model->kv_dim;
        memcpy(model->k_cache + cache_base, model->k, (size_t)model->kv_dim * sizeof(float));
        memcpy(model->v_cache + cache_base, model->v, (size_t)model->kv_dim * sizeof(float));

        for (uint32_t h = 0; h < model->H; h++) {
            uint32_t kv_head = h / kv_mul;
            const float* query = model->q + (size_t)h * model->HD;
            float maximum = -INFINITY;
            for (uint32_t t = 0; t <= pos; t++) {
                const float* key = model->k_cache +
                    ((size_t)l * model->max_seq + t) * model->kv_dim + (size_t)kv_head * model->HD;
                float dot = 0.0f;
                for (uint32_t i = 0; i < model->HD; i++)
                    dot += query[i] * key[i];
                float score = dot * attention_scale;
                if (!isfinite(score))
                    return bnne_error(engine, BNNE_ECOMPUTE, "Qwen3 attention score is not finite");
                model->scores[t] = score;
                if (score > maximum)
                    maximum = score;
            }
            float sum = 0.0f;
            for (uint32_t t = 0; t <= pos; t++) {
                float probability = expf(model->scores[t] - maximum);
                model->probs[t] = probability;
                sum += probability;
            }
            if (!(sum > 0.0f) || !isfinite(sum))
                return bnne_error(engine, BNNE_ECOMPUTE, "Qwen3 attention softmax failed");
            for (uint32_t t = 0; t <= pos; t++)
                model->probs[t] /= sum;
            for (uint32_t i = 0; i < model->HD; i++) {
                float value = 0.0f;
                for (uint32_t t = 0; t <= pos; t++) {
                    const float* cached = model->v_cache +
                        ((size_t)l * model->max_seq + t) * model->kv_dim +
                        (size_t)kv_head * model->HD;
                    value += model->probs[t] * cached[i];
                }
                model->y[(size_t)h * model->HD + i] = value;
            }
        }

        snprintf(tag, sizeof tag, "l%u_y", l);
        qwen3_debug_dump(dump, tag, pos, model->y, model->C);
        status = qwen3_linear(engine, &block->wo, model->y, model->o, stats);
        if (status != BNNE_OK) return status;
        for (uint32_t i = 0; i < model->C; i++)
            model->x[i] += model->o[i];
        snprintf(tag, sizeof tag, "l%u_attnres", l);
        qwen3_debug_dump(dump, tag, pos, model->x, model->C);

        status = qwen3_rmsnorm(engine, model->x, model->C, model->eps,
                               block->ffn_norm, model->norm);
        if (status != BNNE_OK) return status;
        status = qwen3_linear(engine, &block->gate, model->norm, model->gate, stats);
        if (status != BNNE_OK) return status;
        status = qwen3_linear(engine, &block->up, model->norm, model->up, stats);
        if (status != BNNE_OK) return status;
        for (uint32_t i = 0; i < model->FF; i++) {
            float value = model->gate[i];
            float silu = value / (1.0f + expf(-value));
            model->gate[i] = silu * model->up[i];
        }
        status = qwen3_linear(engine, &block->down, model->gate, model->o, stats);
        if (status != BNNE_OK) return status;
        for (uint32_t i = 0; i < model->C; i++)
            model->x[i] += model->o[i];
        snprintf(tag, sizeof tag, "l%u_ffnres", l);
        qwen3_debug_dump(dump, tag, pos, model->x, model->C);
    }

    BNNE_Status status = qwen3_rmsnorm(engine, model->x, model->C, model->eps,
                                       model->final_norm, model->norm);
    if (status != BNNE_OK)
        return status;
    status = qwen3_linear(engine, &model->lm_head, model->norm, logits, stats);
    if (status != BNNE_OK)
        return status;

    model->cache_pos = pos + 1;
    return BNNE_OK;
}

static BNNE_Status qwen3_run(BNNE_Engine engine, const BNNE_Tensor* input,
                             BNNE_Tensor* output, BNNE_Stats* stats) {
    QwenModel* model = engine->model;
    uint32_t count = input->desc.dims[0];
    if (count == 0 || count > model->max_seq)
        return bnne_error(engine, BNNE_ERANGE, "Qwen3 sequence length is out of range");
    const int32_t* ids = input->data;
    float* logits = output->data;
    model->cache_pos = 0;
    for (uint32_t t = 0; t < count; t++) {
        BNNE_Status status = qwen3_step_core(engine, (uint32_t)ids[t], t,
                                             logits + (size_t)t * model->V, stats);
        if (status != BNNE_OK)
            return status;
    }
    return BNNE_OK;
}

static BNNE_Status qwen3_step(BNNE_Engine engine, int32_t token, uint32_t pos,
                              float* out_logits, BNNE_Stats* stats) {
    if (token < 0)
        return bnne_error(engine, BNNE_ERANGE, "Qwen3 token ID is out of range");
    return qwen3_step_core(engine, (uint32_t)token, pos, out_logits, stats);
}

static BNNE_Status qwen3_token(BNNE_Engine engine, uint32_t id, const char** out_text) {
    QwenModel* model = engine->model;
    if (!out_text)
        return BNNE_EINVAL;
    if (id >= model->V)
        return bnne_error(engine, BNNE_ERANGE, "Qwen3 token ID is out of range");
    *out_text = model->token_text + model->token_off[id];
    return BNNE_OK;
}

static BNNE_Status qwen3_eos(BNNE_Engine engine, uint32_t* out_id) {
    QwenModel* model = engine->model;
    if (!model->has_eos)
        return BNNE_ENOTSUP;
    *out_id = model->eos_id;
    return BNNE_OK;
}

static int qwen3_is_eog(BNNE_Engine engine, uint32_t id) {
    QwenModel* model = engine->model;
    for (uint32_t i = 0; i < model->eog_count; i++)
        if (model->eog_ids[i] == id)
            return 1;
    return 0;
}

static BNNE_Status qwen3_encode(BNNE_Engine engine, const char* text, int parse_special,
                                int32_t* ids, uint32_t capacity, uint32_t* out_count) {
    QwenModel* model = engine->model;
    return bnne_qwen_encode(model->tokenizer, text, parse_special, ids, capacity, out_count);
}

static BNNE_Status qwen3_reset_cache(BNNE_Engine engine) {
    QwenModel* model = engine->model;
    if (model)
        model->cache_pos = 0;
    return BNNE_OK;
}

static void qwen3_destroy(BNNE_Engine engine) {
    QwenModel* model = engine->model;
    if (!model)
        return;
    if (model->blocks) {
        for (uint32_t l = 0; l < model->L; l++) {
            QwenBlock* block = &model->blocks[l];
            free(block->attn_norm);
            free(block->q_norm);
            free(block->k_norm);
            free(block->ffn_norm);
            qwen3_weight_destroy(&block->wq);
            qwen3_weight_destroy(&block->wk);
            qwen3_weight_destroy(&block->wv);
            qwen3_weight_destroy(&block->wo);
            qwen3_weight_destroy(&block->gate);
            qwen3_weight_destroy(&block->up);
            qwen3_weight_destroy(&block->down);
        }
    }
    qwen3_weight_destroy(&model->lm_head);
    free(model->blocks);
    free(model->final_norm);
    free(model->emb_bits);
    free(model->emb_scales);
    free(model->x);
    free(model->norm);
    free(model->q);
    free(model->k);
    free(model->v);
    free(model->y);
    free(model->o);
    free(model->gate);
    free(model->up);
    free(model->scores);
    free(model->probs);
    free(model->quantized);
    free(model->chunk_out);
    free(model->k_cache);
    free(model->v_cache);
    free(model->rope_cos);
    free(model->rope_sin);
    free(model->token_text);
    free(model->token_off);
    free(model->token_len);
    bnne_qwen_tokenizer_destroy(model->tokenizer);
    bnne_gguf_close(model->gguf);
    free(model);
    engine->model = NULL;
}

const BNNE_ModelOps bnne_qwen3_ops = {
    .load = qwen3_load,
    .prepare = qwen3_prepare,
    .run = qwen3_run,
    .destroy = qwen3_destroy,
    .token = qwen3_token,
    .step = qwen3_step,
    .reset_cache = qwen3_reset_cache,
    .eos = qwen3_eos,
    .encode = qwen3_encode,
    .is_eog = qwen3_is_eog,
};
