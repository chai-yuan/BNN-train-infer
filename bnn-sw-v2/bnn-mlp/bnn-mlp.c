#include "../bnne-internal.h"

#include "../gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BNN_MLP_MAX_LAYERS 64u

typedef struct {
    bnne_linear weight;
    float* post_scale;
    float* post_bias;
    uint32_t in_features;
    uint32_t out_features;
    int relu;
} bnn_mlp_layer;

typedef struct {
    uint32_t layer_count;
    uint32_t input_features;
    uint32_t class_count;
    uint32_t max_dim;
    bnn_mlp_layer* layers;
} bnn_mlp_model;

typedef struct {
    float* work[2];
} bnn_mlp_state;

static void bnn_mlp_unload(void* impl) {
    bnn_mlp_model* model = impl;
    if (!model)
        return;
    if (model->layers) {
        for (uint32_t i = 0; i < model->layer_count; i++) {
            bnne_linear_free(&model->layers[i].weight);
            free(model->layers[i].post_scale);
            free(model->layers[i].post_bias);
        }
    }
    free(model->layers);
    free(model);
}

static BNNE_Status bnn_mlp_read_vector(struct bnne_model* m, const char* name, uint32_t count,
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

static BNNE_Status bnn_mlp_load(struct bnne_model* m, const bnne_gguf* g, void** out_impl) {
    *out_impl = NULL;
    if (g->mlp_block_count == 0 || g->mlp_block_count > BNN_MLP_MAX_LAYERS ||
        g->mlp_input_features == 0 || g->mlp_input_features > UINT32_MAX ||
        g->mlp_input_features % 128 != 0 || g->mlp_class_count == 0 ||
        g->mlp_class_count > UINT32_MAX ||
        (g->mlp_input_pixels && g->mlp_input_pixels > g->mlp_input_features) ||
        !(g->mlp_input_std > 0.0f))
        return bnne_error(BNNE_EFORMAT, "unsupported bnn-mlp metadata");

    bnn_mlp_model* model = calloc(1, sizeof *model);
    if (!model)
        return bnne_error(BNNE_ENOMEM, "out of memory");
    model->layer_count = (uint32_t)g->mlp_block_count;
    model->input_features = (uint32_t)g->mlp_input_features;
    model->class_count = (uint32_t)g->mlp_class_count;
    model->max_dim = model->input_features;
    model->layers = calloc(model->layer_count, sizeof *model->layers);
    if (!model->layers) {
        bnn_mlp_unload(model);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }

    uint32_t expected_in = model->input_features;
    BNNE_Status status = BNNE_OK;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        bnn_mlp_layer* layer = &model->layers[i];
        char name[64];
        snprintf(name, sizeof name, "blk.%u.weight", i);
        status = bnne_linear_load(m, g, name, &layer->weight);
        if (status != BNNE_OK)
            goto fail;
        if (layer->weight.in != expected_in) {
            status = bnne_error(BNNE_EFORMAT,
                                "layer %u input width %u does not match previous output %u", i,
                                layer->weight.in, expected_in);
            goto fail;
        }
        layer->in_features = layer->weight.in;
        layer->out_features = layer->weight.out;

        snprintf(name, sizeof name, "blk.%u.post_scale", i);
        status = bnn_mlp_read_vector(m, name, layer->out_features, &layer->post_scale);
        if (status != BNNE_OK)
            goto fail;
        snprintf(name, sizeof name, "blk.%u.post_bias", i);
        status = bnn_mlp_read_vector(m, name, layer->out_features, &layer->post_bias);
        if (status != BNNE_OK)
            goto fail;

        const char* activation = i < g->mlp_activation_count ? g->mlp_activations[i] : "";
        if (!activation[0]) {
            layer->relu = i + 1 < model->layer_count;
        } else if (strcmp(activation, "relu") == 0) {
            layer->relu = 1;
        } else if (strcmp(activation, "none") == 0) {
            layer->relu = 0;
        } else {
            status = bnne_error(BNNE_EFORMAT, "invalid activation for bnn-mlp block %u", i);
            goto fail;
        }

        if (layer->out_features > model->max_dim)
            model->max_dim = layer->out_features;
        expected_in = layer->out_features;
    }

    if (expected_in != model->class_count) {
        status = bnne_error(BNNE_EFORMAT,
                            "last bnn-mlp layer width %u does not match class_count %u",
                            expected_in, model->class_count);
        goto fail;
    }

    m->info.input_rank = 1;
    m->info.input_dims[0] = model->input_features;
    m->info.input_size = model->input_features;
    m->info.n_classes = model->class_count;
    *out_impl = model;
    return BNNE_OK;

fail:
    bnn_mlp_unload(model);
    return status;
}

static void bnn_mlp_ctx_free(void* state) {
    bnn_mlp_state* s = state;
    if (!s)
        return;
    free(s->work[0]);
    free(s->work[1]);
    free(s);
}

static BNNE_Status bnn_mlp_ctx_init(struct bnne_context* c, void** out_state) {
    bnn_mlp_model* model = c->model->impl;
    bnn_mlp_state* s = calloc(1, sizeof *s);
    if (!s)
        return bnne_error(BNNE_ENOMEM, "out of memory");
    s->work[0] = malloc((size_t)model->max_dim * sizeof(float));
    s->work[1] = malloc((size_t)model->max_dim * sizeof(float));
    if (!s->work[0] || !s->work[1]) {
        bnn_mlp_ctx_free(s);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }
    *out_state = s;
    return BNNE_OK;
}

static BNNE_Status bnn_mlp_classify(struct bnne_context* c, const float* input, uint32_t n_samples,
                                    float* logits) {
    bnn_mlp_model* model = c->model->impl;
    bnn_mlp_state* s = c->state;

    for (uint32_t sample = 0; sample < n_samples; sample++) {
        const float* x = input + (size_t)sample * model->input_features;
        for (uint32_t i = 0; i < model->layer_count; i++) {
            bnn_mlp_layer* layer = &model->layers[i];
            float* y = i + 1 == model->layer_count
                           ? logits + (size_t)sample * model->class_count
                           : s->work[i & 1];
            BNNE_Status status = bnne_linear_forward(c, &layer->weight, x, 1, y);
            if (status != BNNE_OK)
                return status;
            for (uint32_t o = 0; o < layer->out_features; o++) {
                float value = layer->post_scale[o] * y[o] + layer->post_bias[o];
                if (layer->relu && value < 0.0f)
                    value = 0.0f;
                y[o] = value;
            }
            x = y;
        }
    }
    return BNNE_OK;
}

const bnne_arch bnne_arch_bnn_mlp = {
    .name = "bnn-mlp",
    .task = BNNE_TASK_CLASSIFY,
    .load = bnn_mlp_load,
    .unload = bnn_mlp_unload,
    .ctx_init = bnn_mlp_ctx_init,
    .ctx_free = bnn_mlp_ctx_free,
    .classify = bnn_mlp_classify,
    .decode = NULL,
    .truncate = NULL,
};
