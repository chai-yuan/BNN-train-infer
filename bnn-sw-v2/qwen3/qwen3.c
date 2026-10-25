#include "../bnne-internal.h"

#include "../gguf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN3_GROUP 128u
#define QWEN3_MAX_LAYERS 128u
#define QWEN3_MAX_HEADS 256u
#define QWEN3_MAX_VOCAB (1u << 20)
#define QWEN3_PI 3.14159265358979323846

typedef struct {
    float* attn_norm;
    float* q_norm;
    float* k_norm;
    float* ffn_norm;
    bnne_linear wq, wk, wv, wo;
    bnne_linear gate, up, down;
} qwen3_block;

typedef struct {
    uint32_t C, H, Hk, HD, FF, L, V, kv_dim;
    float eps;
    float rope_base, rope_freq_scale, rope_mscale;
    int32_t rope_corr0, rope_corr1;
    int rope_yarn;

    uint8_t* emb_bits;
    uint16_t* emb_scales;
    bnne_linear lm_head;
    float* final_norm;
    qwen3_block* blocks;
} qwen3_model;

typedef struct {
    uint32_t n_ctx;
    float *x, *norm, *q, *k, *v, *y, *o, *gate, *up, *scores, *probs;
    float *k_cache, *v_cache;
    float *rope_cos, *rope_sin;
} qwen3_state;

static void* qwen3_alloc(size_t count, size_t size) {
    if (!count || size > SIZE_MAX / count)
        return NULL;
    return malloc(count * size);
}

static void qwen3_unload(void* impl) {
    qwen3_model* model = impl;
    if (!model)
        return;
    if (model->blocks) {
        for (uint32_t l = 0; l < model->L; l++) {
            qwen3_block* block = &model->blocks[l];
            free(block->attn_norm);
            free(block->q_norm);
            free(block->k_norm);
            free(block->ffn_norm);
            bnne_linear_free(&block->wq);
            bnne_linear_free(&block->wk);
            bnne_linear_free(&block->wv);
            bnne_linear_free(&block->wo);
            bnne_linear_free(&block->gate);
            bnne_linear_free(&block->up);
            bnne_linear_free(&block->down);
        }
    }
    free(model->blocks);
    bnne_linear_free(&model->lm_head);
    free(model->final_norm);
    free(model->emb_bits);
    free(model->emb_scales);
    free(model);
}

static BNNE_Status qwen3_read_f32(struct bnne_model* m, const char* name, uint32_t count,
                                  float** out) {
    const float* data = NULL;
    size_t n = 0;
    BNNE_Status status = bnne_tensor_f32(m, m->gguf, name, &data, &n);
    if (status != BNNE_OK)
        return status;
    if (n != count) {
        free((void*)data);
        return bnne_error(BNNE_EFORMAT, "tensor '%s' has %zu elements, expected %u", name, n,
                          count);
    }
    *out = (float*)data;
    return BNNE_OK;
}

static BNNE_Status qwen3_load_linear(struct bnne_model* m, const char* name, uint32_t in,
                                     uint32_t out, bnne_linear* linear) {
    BNNE_Status status = bnne_linear_load(m, m->gguf, name, linear);
    if (status != BNNE_OK)
        return status;
    if (linear->in != in || linear->out != out) {
        uint32_t actual_in = linear->in, actual_out = linear->out;
        bnne_linear_free(linear);
        return bnne_error(BNNE_EFORMAT, "tensor '%s' has shape %u->%u, expected %u->%u", name,
                          actual_in, actual_out, in, out);
    }
    return BNNE_OK;
}

static int32_t qwen3_clamp_dim(double value, int32_t low, int32_t high) {
    if (value <= (double)low) return low;
    if (value >= (double)high) return high;
    return (int32_t)value;
}

static BNNE_Status qwen3_init_rope_params(qwen3_model* model, const bnne_gguf* g) {
    float freq_scale = 1.0f / g->rope_factor;
    int yarn = strcmp(g->rope_scaling_type, "yarn") == 0;
    int32_t corr0 = 0, corr1 = (int32_t)(model->HD - 1);
    float mscale = 1.0f;

    if (yarn && g->rope_original_context) {
        double base = model->rope_base;
        double original = (double)g->rope_original_context;
        double start = floor((double)model->HD *
                             log(original / (32.0 * 2.0 * QWEN3_PI)) / (2.0 * log(base)));
        double end = ceil((double)model->HD *
                          log(original / (2.0 * QWEN3_PI)) / (2.0 * log(base)));
        corr0 = qwen3_clamp_dim(start, 0, (int32_t)model->HD - 1);
        corr1 = qwen3_clamp_dim(end, 0, (int32_t)model->HD - 1);
        mscale = 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    if (!(freq_scale > 0.0f) || !isfinite(mscale))
        return bnne_error(BNNE_EFORMAT, "invalid Qwen3 rope scaling");

    model->rope_freq_scale = freq_scale;
    model->rope_mscale = mscale;
    model->rope_corr0 = corr0;
    model->rope_corr1 = corr1;
    model->rope_yarn = yarn;
    return BNNE_OK;
}

static BNNE_Status qwen3_load(struct bnne_model* m, const bnne_gguf* g, void** out_impl) {
    *out_impl = NULL;
    qwen3_model* model = calloc(1, sizeof *model);
    if (!model)
        return bnne_error(BNNE_ENOMEM, "out of memory");

    const bnne_gguf_tensor* embedding = bnne_gguf_find(g, "token_embd.weight");
    if (!embedding || embedding->type != 41 || embedding->rank != 2) {
        qwen3_unload(model);
        return bnne_error(BNNE_EFORMAT, "missing token_embd.weight");
    }
    if (g->block_count == 0 || g->block_count > QWEN3_MAX_LAYERS ||
        g->embedding_length == 0 || g->embedding_length > UINT32_MAX ||
        g->feed_forward_length == 0 || g->feed_forward_length > UINT32_MAX ||
        g->head_count == 0 || g->head_count > QWEN3_MAX_HEADS ||
        g->head_count_kv == 0 || g->head_count_kv > QWEN3_MAX_HEADS ||
        g->key_length == 0 || g->key_length > UINT32_MAX ||
        g->key_length != g->value_length || g->context_length == 0 ||
        g->context_length > UINT32_MAX || g->rms_epsilon <= 0.0f ||
        g->rope_freq_base <= 0.0f || embedding->dims[0] != g->embedding_length ||
        embedding->dims[1] == 0 || embedding->dims[1] > QWEN3_MAX_VOCAB) {
        qwen3_unload(model);
        return bnne_error(BNNE_EFORMAT, "unsupported Qwen3 dimensions");
    }

    model->L = (uint32_t)g->block_count;
    model->C = (uint32_t)g->embedding_length;
    model->FF = (uint32_t)g->feed_forward_length;
    model->H = (uint32_t)g->head_count;
    model->Hk = (uint32_t)g->head_count_kv;
    model->HD = (uint32_t)g->key_length;
    model->V = (uint32_t)embedding->dims[1];
    model->eps = g->rms_epsilon;
    model->rope_base = g->rope_freq_base;

    if (model->HD > model->C || (uint64_t)model->H * model->HD != model->C ||
        model->H % model->Hk != 0 || model->C % QWEN3_GROUP != 0 ||
        model->FF % QWEN3_GROUP != 0 || model->HD % 2 != 0) {
        qwen3_unload(model);
        return bnne_error(BNNE_EFORMAT, "inconsistent Qwen3 dimensions");
    }
    model->kv_dim = model->Hk * model->HD;
    if (model->kv_dim % QWEN3_GROUP != 0) {
        qwen3_unload(model);
        return bnne_error(BNNE_EFORMAT, "Qwen3 KV width is not group aligned");
    }
    if (m->caps.group_size != QWEN3_GROUP || !m->caps.supports_gemv ||
        !m->caps.supports_gemm) {
        qwen3_unload(model);
        return bnne_error(BNNE_ENOTSUP,
                          "Qwen3 requires group 128 with GEMV/GEMM support");
    }

    model->blocks = calloc(model->L, sizeof *model->blocks);
    if (!model->blocks) {
        qwen3_unload(model);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }

    BNNE_Status status = qwen3_init_rope_params(model, g);
    if (status != BNNE_OK) {
        qwen3_unload(model);
        return status;
    }

    status = qwen3_read_f32(m, "output_norm.weight", model->C, &model->final_norm);
    if (status != BNNE_OK) {
        qwen3_unload(model);
        return status;
    }

    for (uint32_t l = 0; l < model->L; l++) {
        qwen3_block* block = &model->blocks[l];
        char name[64];
        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", l);
        status = qwen3_read_f32(m, name, model->C, &block->attn_norm);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.ffn_norm.weight", l);
        status = qwen3_read_f32(m, name, model->C, &block->ffn_norm);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.attn_q_norm.weight", l);
        status = qwen3_read_f32(m, name, model->HD, &block->q_norm);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.attn_k_norm.weight", l);
        status = qwen3_read_f32(m, name, model->HD, &block->k_norm);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
    }

    status = bnne_q1_0_read(g, "token_embd.weight", model->C, model->V, &model->emb_bits,
                            &model->emb_scales);
    if (status != BNNE_OK) {
        qwen3_unload(model);
        return status;
    }
    status = bnne_linear_init(m, model->C, model->V, model->emb_bits, model->emb_scales,
                              &model->lm_head);
    if (status != BNNE_OK) {
        qwen3_unload(model);
        return status;
    }

    for (uint32_t l = 0; l < model->L; l++) {
        qwen3_block* block = &model->blocks[l];
        char name[64];
        snprintf(name, sizeof name, "blk.%u.attn_q.weight", l);
        status = qwen3_load_linear(m, name, model->C, model->C, &block->wq);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.attn_k.weight", l);
        status = qwen3_load_linear(m, name, model->C, model->kv_dim, &block->wk);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.attn_v.weight", l);
        status = qwen3_load_linear(m, name, model->C, model->kv_dim, &block->wv);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.attn_output.weight", l);
        status = qwen3_load_linear(m, name, model->C, model->C, &block->wo);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.ffn_gate.weight", l);
        status = qwen3_load_linear(m, name, model->C, model->FF, &block->gate);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.ffn_up.weight", l);
        status = qwen3_load_linear(m, name, model->C, model->FF, &block->up);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
        snprintf(name, sizeof name, "blk.%u.ffn_down.weight", l);
        status = qwen3_load_linear(m, name, model->FF, model->C, &block->down);
        if (status != BNNE_OK) { qwen3_unload(model); return status; }
    }

    m->info.n_vocab = model->V;
    m->info.n_ctx_train = (uint32_t)g->context_length;
    *out_impl = model;
    return BNNE_OK;
}

static void qwen3_ctx_free(void* state) {
    qwen3_state* s = state;
    if (!s)
        return;
    free(s->x);
    free(s->norm);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->y);
    free(s->o);
    free(s->gate);
    free(s->up);
    free(s->scores);
    free(s->probs);
    free(s->k_cache);
    free(s->v_cache);
    free(s->rope_cos);
    free(s->rope_sin);
    free(s);
}

static BNNE_Status qwen3_init_rope_tables(qwen3_model* model, qwen3_state* s) {
    uint32_t half = model->HD / 2;
    int yarn = model->rope_yarn;
    for (uint32_t pos = 0; pos < s->n_ctx; pos++) {
        for (uint32_t j = 0; j < half; j++) {
            float theta_extrap = (float)pos * powf(model->rope_base,
                                                   -2.0f * (float)j / (float)model->HD);
            float theta = model->rope_freq_scale * theta_extrap;
            if (yarn) {
                float span = (float)(model->rope_corr1 - model->rope_corr0);
                float denominator = span > 0.001f ? span : 0.001f;
                float y = ((float)j - (float)model->rope_corr0) / denominator;
                float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
                theta = (model->rope_freq_scale * theta_extrap) * (1.0f - ramp) +
                        theta_extrap * ramp;
            }
            s->rope_cos[(size_t)pos * half + j] = cosf(theta) * model->rope_mscale;
            s->rope_sin[(size_t)pos * half + j] = sinf(theta) * model->rope_mscale;
        }
    }
    return BNNE_OK;
}

static BNNE_Status qwen3_ctx_init(struct bnne_context* c, void** out_state) {
    *out_state = NULL;
    qwen3_model* model = c->model->impl;
    qwen3_state* s = calloc(1, sizeof *s);
    if (!s)
        return bnne_error(BNNE_ENOMEM, "out of memory");
    s->n_ctx = c->params.n_ctx;

    uint32_t half = model->HD / 2;
    s->x = qwen3_alloc(model->C, sizeof(float));
    s->norm = qwen3_alloc(model->C, sizeof(float));
    s->q = qwen3_alloc(model->C, sizeof(float));
    s->y = qwen3_alloc(model->C, sizeof(float));
    s->o = qwen3_alloc(model->C, sizeof(float));
    s->k = qwen3_alloc(model->kv_dim, sizeof(float));
    s->v = qwen3_alloc(model->kv_dim, sizeof(float));
    s->gate = qwen3_alloc(model->FF, sizeof(float));
    s->up = qwen3_alloc(model->FF, sizeof(float));
    s->scores = qwen3_alloc(s->n_ctx, sizeof(float));
    s->probs = qwen3_alloc(s->n_ctx, sizeof(float));
    s->k_cache = qwen3_alloc((size_t)model->L * s->n_ctx, (size_t)model->kv_dim * sizeof(float));
    s->v_cache = qwen3_alloc((size_t)model->L * s->n_ctx, (size_t)model->kv_dim * sizeof(float));
    s->rope_cos = qwen3_alloc((size_t)s->n_ctx * half, sizeof(float));
    s->rope_sin = qwen3_alloc((size_t)s->n_ctx * half, sizeof(float));
    if (!s->x || !s->norm || !s->q || !s->y || !s->o || !s->k || !s->v || !s->gate ||
        !s->up || !s->scores || !s->probs || !s->k_cache || !s->v_cache || !s->rope_cos ||
        !s->rope_sin) {
        qwen3_ctx_free(s);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }

    BNNE_Status status = qwen3_init_rope_tables(model, s);
    if (status != BNNE_OK) {
        qwen3_ctx_free(s);
        return status;
    }
    *out_state = s;
    return BNNE_OK;
}

static BNNE_Status qwen3_rmsnorm(const float* input, uint32_t count, float epsilon,
                                 const float* gain, float* output) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++)
        sum += input[i] * input[i];
    float rms = sqrtf(sum / (float)count + epsilon);
    if (!isfinite(rms) || rms == 0.0f)
        return bnne_error(BNNE_ECOMPUTE, "Qwen3 RMSNorm produced a non-finite divisor");
    for (uint32_t i = 0; i < count; i++)
        output[i] = input[i] / rms * gain[i];
    return BNNE_OK;
}

static void qwen3_embed(const qwen3_model* model, uint32_t token, float* output) {
    const uint8_t* bits = model->emb_bits + (size_t)token * (model->C / 8u);
    const uint16_t* scales = model->emb_scales + (size_t)token * (model->C / QWEN3_GROUP);
    uint32_t groups = model->C / QWEN3_GROUP;
    for (uint32_t grp = 0; grp < groups; grp++) {
        float scale = bnne_fp16_to_fp32(scales[grp]);
        for (uint32_t j = 0; j < QWEN3_GROUP; j++) {
            uint8_t bit = (uint8_t)((bits[grp * 16 + (j >> 3)] >> (j & 7u)) & 1u);
            output[grp * QWEN3_GROUP + j] = bit ? scale : -scale;
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

static BNNE_Status qwen3_step(struct bnne_context* c, uint32_t token, uint32_t pos,
                              float* logits) {
    qwen3_model* model = c->model->impl;
    qwen3_state* s = c->state;

    qwen3_embed(model, token, s->x);
    uint32_t kv_mul = model->H / model->Hk;
    uint32_t half = model->HD / 2;
    float attention_scale = 1.0f / sqrtf((float)model->HD);

    for (uint32_t l = 0; l < model->L; l++) {
        qwen3_block* block = &model->blocks[l];
        BNNE_Status status = qwen3_rmsnorm(s->x, model->C, model->eps, block->attn_norm,
                                           s->norm);
        if (status != BNNE_OK) return status;
        status = bnne_linear_forward(c, &block->wq, s->norm, 1, s->q);
        if (status != BNNE_OK) return status;
        status = bnne_linear_forward(c, &block->wk, s->norm, 1, s->k);
        if (status != BNNE_OK) return status;
        status = bnne_linear_forward(c, &block->wv, s->norm, 1, s->v);
        if (status != BNNE_OK) return status;

        for (uint32_t h = 0; h < model->H; h++) {
            status = qwen3_rmsnorm(s->q + (size_t)h * model->HD, model->HD, model->eps,
                                   block->q_norm, s->q + (size_t)h * model->HD);
            if (status != BNNE_OK) return status;
        }
        for (uint32_t h = 0; h < model->Hk; h++) {
            status = qwen3_rmsnorm(s->k + (size_t)h * model->HD, model->HD, model->eps,
                                   block->k_norm, s->k + (size_t)h * model->HD);
            if (status != BNNE_OK) return status;
        }

        const float* cosine = s->rope_cos + (size_t)pos * half;
        const float* sine = s->rope_sin + (size_t)pos * half;
        for (uint32_t h = 0; h < model->H; h++)
            qwen3_rope(s->q + (size_t)h * model->HD, cosine, sine, half);
        for (uint32_t h = 0; h < model->Hk; h++)
            qwen3_rope(s->k + (size_t)h * model->HD, cosine, sine, half);

        size_t cache_base = ((size_t)l * s->n_ctx + pos) * model->kv_dim;
        memcpy(s->k_cache + cache_base, s->k, (size_t)model->kv_dim * sizeof(float));
        memcpy(s->v_cache + cache_base, s->v, (size_t)model->kv_dim * sizeof(float));

        for (uint32_t h = 0; h < model->H; h++) {
            uint32_t kv_head = h / kv_mul;
            const float* query = s->q + (size_t)h * model->HD;
            float maximum = -INFINITY;
            for (uint32_t t = 0; t <= pos; t++) {
                const float* key = s->k_cache +
                    ((size_t)l * s->n_ctx + t) * model->kv_dim +
                    (size_t)kv_head * model->HD;
                float dot = 0.0f;
                for (uint32_t i = 0; i < model->HD; i++)
                    dot += query[i] * key[i];
                float score = dot * attention_scale;
                if (!isfinite(score))
                    return bnne_error(BNNE_ECOMPUTE, "Qwen3 attention score is not finite");
                s->scores[t] = score;
                if (score > maximum)
                    maximum = score;
            }
            float sum = 0.0f;
            for (uint32_t t = 0; t <= pos; t++) {
                float probability = expf(s->scores[t] - maximum);
                s->probs[t] = probability;
                sum += probability;
            }
            if (!(sum > 0.0f) || !isfinite(sum))
                return bnne_error(BNNE_ECOMPUTE, "Qwen3 attention softmax failed");
            for (uint32_t t = 0; t <= pos; t++)
                s->probs[t] /= sum;
            for (uint32_t i = 0; i < model->HD; i++) {
                float value = 0.0f;
                for (uint32_t t = 0; t <= pos; t++) {
                    const float* cached = s->v_cache +
                        ((size_t)l * s->n_ctx + t) * model->kv_dim +
                        (size_t)kv_head * model->HD;
                    value += s->probs[t] * cached[i];
                }
                s->y[(size_t)h * model->HD + i] = value;
            }
        }

        status = bnne_linear_forward(c, &block->wo, s->y, 1, s->o);
        if (status != BNNE_OK) return status;
        for (uint32_t i = 0; i < model->C; i++)
            s->x[i] += s->o[i];

        status = qwen3_rmsnorm(s->x, model->C, model->eps, block->ffn_norm, s->norm);
        if (status != BNNE_OK) return status;
        status = bnne_linear_forward(c, &block->gate, s->norm, 1, s->gate);
        if (status != BNNE_OK) return status;
        status = bnne_linear_forward(c, &block->up, s->norm, 1, s->up);
        if (status != BNNE_OK) return status;
        for (uint32_t i = 0; i < model->FF; i++) {
            float value = s->gate[i];
            float silu = value / (1.0f + expf(-value));
            s->gate[i] = silu * s->up[i];
        }
        status = bnne_linear_forward(c, &block->down, s->gate, 1, s->o);
        if (status != BNNE_OK) return status;
        for (uint32_t i = 0; i < model->C; i++)
            s->x[i] += s->o[i];
    }

    if (logits) {
        BNNE_Status status = qwen3_rmsnorm(s->x, model->C, model->eps, model->final_norm,
                                           s->norm);
        if (status != BNNE_OK)
            return status;
        status = bnne_linear_forward(c, &model->lm_head, s->norm, 1, logits);
        if (status != BNNE_OK)
            return status;
    }
    return BNNE_OK;
}

static BNNE_Status qwen3_decode(struct bnne_context* c, const int32_t* tokens, uint32_t n_tokens,
                                float* logits) {
    for (uint32_t i = 0; i < n_tokens; i++) {
        BNNE_Status status = qwen3_step(c, (uint32_t)tokens[i], c->pos + i,
                                        i + 1 == n_tokens ? logits : NULL);
        if (status != BNNE_OK)
            return status;
    }
    return BNNE_OK;
}

static BNNE_Status qwen3_truncate(struct bnne_context* c, uint32_t pos) {
    (void)c;
    (void)pos;
    return BNNE_OK;
}

const bnne_arch bnne_arch_qwen3 = {
    .name = "qwen3",
    .task = BNNE_TASK_LLM,
    .load = qwen3_load,
    .unload = qwen3_unload,
    .ctx_init = qwen3_ctx_init,
    .ctx_free = qwen3_ctx_free,
    .classify = NULL,
    .decode = qwen3_decode,
    .truncate = qwen3_truncate,
};
