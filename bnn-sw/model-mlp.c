#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MLP_MAX_LAYERS 8u
#define MLP_MAX_DIM 4096u

typedef struct {
    uint32_t in_features, out_features, group, relu;
    uint8_t* bits;
    uint16_t* scales;
    float *post_a, *post_b;
    BNN_Matrix matrix;
} MLP_Layer;

typedef struct {
    uint32_t layer_count;
    MLP_Layer layers[MLP_MAX_LAYERS];
    float *work[2], *accum;
    int8_t* quantized;
} MLP_Model;

static void mlp_destroy(BNNE_Engine engine) {
    MLP_Model* model = engine->model;
    if (!model)
        return;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        MLP_Layer* layer = &model->layers[i];
        BNN_MatrixDestroy(layer->matrix);
        free(layer->bits);
        free(layer->scales);
        free(layer->post_a);
        free(layer->post_b);
    }
    free(model->work[0]); free(model->work[1]);
    free(model->accum); free(model->quantized);
    free(model);
}

static BNNE_Status mlp_load(BNNE_Engine engine, FILE* file) {
    MLP_Model* model = calloc(1, sizeof *model);
    if (!model)
        return BNNE_ENOMEM;
    engine->model = model;
    BNNE_Reader reader = {file, BNNE_OK};
    uint32_t count = bnne_read_u32(&reader);
    if (reader.status != BNNE_OK)
        return reader.status;
    if (count == 0 || count > MLP_MAX_LAYERS)
        return BNNE_EFORMAT;
    model->layer_count = count;
    for (uint32_t i = 0; i < count; i++) {
        MLP_Layer* layer = &model->layers[i];
        layer->in_features = bnne_read_u32(&reader);
        layer->out_features = bnne_read_u32(&reader);
        layer->group = bnne_read_u32(&reader);
        layer->relu = bnne_read_u32(&reader);
        if (reader.status != BNNE_OK)
            return reader.status;
        if (layer->in_features == 0 || layer->in_features > MLP_MAX_DIM ||
            layer->out_features == 0 || layer->out_features > MLP_MAX_DIM ||
            layer->group == 0 || layer->group % 8 != 0 ||
            layer->in_features % layer->group != 0 || layer->relu > 1 ||
            (i > 0 && layer->in_features != model->layers[i - 1].out_features))
            return BNNE_EFORMAT;
        size_t weight_bytes = (size_t)layer->out_features * (layer->in_features / 8);
        size_t scale_count = (size_t)layer->out_features * (layer->in_features / layer->group);
        layer->bits = malloc(weight_bytes);
        layer->scales = malloc(scale_count * sizeof *layer->scales);
        layer->post_a = malloc((size_t)layer->out_features * sizeof *layer->post_a);
        layer->post_b = malloc((size_t)layer->out_features * sizeof *layer->post_b);
        if (!layer->bits || !layer->scales || !layer->post_a || !layer->post_b)
            return BNNE_ENOMEM;
        bnne_read(&reader, layer->bits, weight_bytes);
        for (size_t k = 0; k < scale_count && reader.status == BNNE_OK; k++) {
            layer->scales[k] = bnne_read_u16(&reader);
            if (!isfinite(bnne_fp16_to_fp32(layer->scales[k])))
                return BNNE_EFORMAT;
        }
        for (uint32_t k = 0; k < layer->out_features && reader.status == BNNE_OK; k++) {
            layer->post_a[k] = bnne_read_f32(&reader);
            if (!isfinite(layer->post_a[k]))
                return BNNE_EFORMAT;
        }
        for (uint32_t k = 0; k < layer->out_features && reader.status == BNNE_OK; k++) {
            layer->post_b[k] = bnne_read_f32(&reader);
            if (!isfinite(layer->post_b[k]))
                return BNNE_EFORMAT;
        }
        if (reader.status != BNNE_OK)
            return reader.status;
    }
    if (fgetc(file) != EOF)
        return BNNE_EFORMAT;
    if (ferror(file))
        return BNNE_EIO;
    engine->info.kind = BNNE_MODEL_MLP;
    engine->info.layer_count = count;
    engine->info.input = (BNNE_TensorDesc){BNNE_F32, 1, {model->layers[0].in_features}};
    engine->info.output = (BNNE_TensorDesc){BNNE_F32, 1, {model->layers[count - 1].out_features}};
    return BNNE_OK;
}

static BNNE_Status mlp_prepare(BNNE_Engine engine) {
    MLP_Model* model = engine->model;
    BNN_Caps caps;
    BNNE_Status status = bnne_runtime_open(engine, &caps);
    if (status != BNNE_OK)
        return status;
    if (!caps.supports_gemv)
        return BNNE_ENOTSUP;
    uint32_t max_in = 0, max_out = 0;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        MLP_Layer* layer = &model->layers[i];
        if (layer->in_features > max_in) max_in = layer->in_features;
        if (layer->out_features > max_out) max_out = layer->out_features;
        if (layer->group != caps.group_size || layer->in_features > caps.max_in_features ||
            layer->out_features > caps.max_out_features)
            return bnne_error(engine, BNNE_ENOTSUP, "layer %u exceeds device capabilities", i);
        BNN_MatrixDesc desc = {
            .in_features = layer->in_features,
            .out_features = layer->out_features,
            .group_size = layer->group,
            .weight_bits = layer->bits,
            .weight_bytes = (size_t)layer->out_features * (layer->in_features / 8),
            .scale_fp16 = layer->scales,
            .scale_count = (size_t)layer->out_features * (layer->in_features / layer->group),
        };
        status = bnne_accel_status(engine,
            BNN_MatrixCreate(engine->device, &desc, &layer->matrix), "BNN_MatrixCreate");
        if (status != BNNE_OK)
            return status;
        free(layer->bits); layer->bits = NULL;
        free(layer->scales); layer->scales = NULL;
    }
    uint32_t max_dim = max_in > max_out ? max_in : max_out;
    model->work[0] = malloc((size_t)max_dim * sizeof(float));
    model->work[1] = malloc((size_t)max_dim * sizeof(float));
    model->quantized = malloc(max_in);
    model->accum = malloc((size_t)max_out * sizeof(float));
    if (!model->work[0] || !model->work[1] || !model->quantized || !model->accum)
        return BNNE_ENOMEM;
    return bnne_runtime_buffers(engine, max_in, (size_t)max_out * sizeof(float));
}

static BNNE_Status mlp_run(BNNE_Engine engine, const BNNE_Tensor* input,
                           BNNE_Tensor* output, BNNE_Stats* stats) {
    MLP_Model* model = engine->model;
    const float* x = input->data;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        MLP_Layer* layer = &model->layers[i];
        float* y = model->work[i & 1];
        float scale;
        uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
        BNNE_Status status = bnne_quantize(x, layer->in_features, model->quantized, &scale);
        if (engine->options.enable_profile)
            stats->quantize_ns += bnne_now_ns() - start;
        if (status != BNNE_OK)
            return bnne_error(engine, status, "layer %u cannot quantize activations", i);
        status = bnne_runtime_gemv(engine, layer->matrix, model->quantized,
                                   layer->in_features, model->accum, layer->out_features, stats);
        if (status != BNNE_OK)
            return status;
        for (uint32_t o = 0; o < layer->out_features; o++) {
            float value = layer->post_a[o] * (model->accum[o] * scale) + layer->post_b[o];
            if (!isfinite(value))
                return bnne_error(engine, BNNE_ECOMPUTE, "layer %u produced NaN or Inf", i);
            y[o] = layer->relu && value < 0.0f ? 0.0f : value;
        }
        if (engine->options.enable_profile) {
            uint64_t macs = (uint64_t)layer->in_features * layer->out_features;
            stats->logical_macs += macs;
            stats->executed_macs += macs;
        }
        x = y;
    }
    memcpy(output->data, x, (size_t)engine->info.output.dims[0] * sizeof(float));
    return BNNE_OK;
}

const BNNE_ModelOps bnne_mlp_ops = {
    .load = mlp_load, .prepare = mlp_prepare, .run = mlp_run, .destroy = mlp_destroy
};
