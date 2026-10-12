#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CNN_MAX_LAYERS 16u
#define CNN_MAX_DIM 4096u
#define CNN_MAX_CHANNELS 64u
#define CNN_MAX_MODEL_BYTES (64u * 1024u * 1024u)
#define CNN_MAX_FMAP 65536u
#define CNN_MAX_MATRIX_BYTES (32u * 1024u * 1024u)
#define CNN_MAX_WORK_BYTES (64u * 1024u * 1024u)
#define CNN_IMAGE 28u

typedef struct {
    uint32_t type, in_ch, out_ch, kernel, group, relu, in_f, out_f, pool;
    uint32_t total, groups, row_bytes, pad_k, rows, height, width;
    uint8_t* bits;
    uint16_t* scale_bits;
    float* post_a;
    float* post_b;
    BNN_Matrix matrix;
} CNN_Layer;

typedef struct {
    uint32_t count, max_fmap, device_group;
    CNN_Layer layers[CNN_MAX_LAYERS];
    float* work[2];
    float* accum;
    int8_t* quantized;
    int8_t* im2col;
} CNN_Model;

static void cnn_destroy(BNNE_Engine engine) {
    CNN_Model* model = engine->model;
    if (!model)
        return;
    for (uint32_t i = 0; i < model->count; i++) {
        CNN_Layer* layer = &model->layers[i];
        BNN_MatrixDestroy(layer->matrix);
        free(layer->bits);
        free(layer->scale_bits);
        free(layer->post_a);
        free(layer->post_b);
    }
    free(model->work[0]);
    free(model->work[1]);
    free(model->accum);
    free(model->quantized);
    free(model->im2col);
    free(model);
    engine->model = NULL;
}

static BNNE_Status cnn_load(BNNE_Engine engine, FILE* file) {
    CNN_Model* model = calloc(1, sizeof *model);
    if (!model)
        return BNNE_ENOMEM;
    engine->model = model;
    BNNE_Reader r = {file, BNNE_OK};
    uint32_t count = bnne_read_u32(&r), channels = 1, height = CNN_IMAGE, width = CNN_IMAGE;
    int dense = 0;
    size_t model_bytes = 0;
    model->max_fmap = CNN_IMAGE * CNN_IMAGE;
    if (r.status != BNNE_OK)
        return r.status;
    if (!count || count > CNN_MAX_LAYERS)
        return BNNE_EFORMAT;
    model->count = count;

    for (uint32_t i = 0; i < count; i++) {
        CNN_Layer* l = &model->layers[i];
        l->type = bnne_read_u32(&r);
        if (l->type == 2) {
            l->pool = bnne_read_u32(&r);
            if (r.status != BNNE_OK)
                return r.status;
            if (dense || !l->pool || l->pool > 4 || height % l->pool || width % l->pool)
                return BNNE_EFORMAT;
            l->height = height;
            l->width = width;
            height /= l->pool;
            width /= l->pool;
        } else if (l->type == 0 || l->type == 1) {
            if (l->type == 0) {
                l->in_ch = bnne_read_u32(&r);
                l->out_ch = bnne_read_u32(&r);
                l->kernel = bnne_read_u32(&r);
                l->group = bnne_read_u32(&r);
                l->relu = bnne_read_u32(&r);
                if (r.status != BNNE_OK)
                    return r.status;
                if (dense || l->in_ch != channels || !l->in_ch || l->in_ch > CNN_MAX_CHANNELS ||
                    !l->out_ch || l->out_ch > CNN_MAX_CHANNELS ||
                    !l->kernel || l->kernel > 7 || !(l->kernel & 1) || l->relu > 1)
                    return BNNE_EFORMAT;
                uint64_t total = (uint64_t)l->in_ch * l->kernel * l->kernel;
                if (total > CNN_MAX_DIM)
                    return BNNE_EFORMAT;
                l->total = (uint32_t)total;
                l->out_f = l->out_ch;
                l->rows = height * width;
                l->height = height;
                l->width = width;
            } else {
                l->in_f = bnne_read_u32(&r);
                l->out_f = bnne_read_u32(&r);
                l->group = bnne_read_u32(&r);
                l->relu = bnne_read_u32(&r);
                if (r.status != BNNE_OK)
                    return r.status;
                uint64_t expected = (uint64_t)channels * height * width;
                if (!l->in_f || l->in_f > CNN_MAX_DIM || l->in_f != expected ||
                    !l->out_f || l->out_f > CNN_MAX_DIM || l->relu > 1)
                    return BNNE_EFORMAT;
                l->total = l->in_f;
                l->rows = 1;
                channels = l->out_f;
                height = width = 1;
                dense = 1;
            }
            if (!l->group || l->total % l->group)
                return BNNE_EFORMAT;
            l->groups = l->total / l->group;
            l->row_bytes = (l->total + 7) / 8;
            size_t weight_bytes = (size_t)l->out_f * l->row_bytes;
            size_t scale_count = (size_t)l->out_f * l->groups;
            if (!weight_bytes || weight_bytes > CNN_MAX_MATRIX_BYTES ||
                scale_count > CNN_MAX_MATRIX_BYTES / sizeof(uint16_t))
                return BNNE_EFORMAT;
            size_t bytes = weight_bytes + scale_count * sizeof(uint16_t) +
                           (size_t)l->out_f * 2 * sizeof(float);
            if (bytes > CNN_MAX_MODEL_BYTES - model_bytes)
                return BNNE_EFORMAT;
            model_bytes += bytes;
            l->bits = malloc(weight_bytes);
            l->scale_bits = malloc(scale_count * sizeof *l->scale_bits);
            l->post_a = malloc((size_t)l->out_f * sizeof *l->post_a);
            l->post_b = malloc((size_t)l->out_f * sizeof *l->post_b);
            if (!l->bits || !l->scale_bits || !l->post_a || !l->post_b)
                return BNNE_ENOMEM;
            bnne_read(&r, l->bits, weight_bytes);
            for (size_t k = 0; k < scale_count && r.status == BNNE_OK; k++) {
                l->scale_bits[k] = bnne_read_u16(&r);
                if (!isfinite(bnne_fp16_to_fp32(l->scale_bits[k])))
                    return BNNE_EFORMAT;
            }
            for (uint32_t k = 0; k < l->out_f && r.status == BNNE_OK; k++) {
                l->post_a[k] = bnne_read_f32(&r);
                if (!isfinite(l->post_a[k]))
                    return BNNE_EFORMAT;
            }
            for (uint32_t k = 0; k < l->out_f && r.status == BNNE_OK; k++) {
                l->post_b[k] = bnne_read_f32(&r);
                if (!isfinite(l->post_b[k]))
                    return BNNE_EFORMAT;
            }
            if (r.status != BNNE_OK)
                return r.status;
            if (l->type == 0)
                channels = l->out_ch;
            uint64_t fmap = (uint64_t)channels * height * width;
            if (fmap > CNN_MAX_FMAP)
                return BNNE_EFORMAT;
            if (fmap > model->max_fmap)
                model->max_fmap = (uint32_t)fmap;
        } else {
            return BNNE_EFORMAT;
        }
    }
    if (fgetc(file) != EOF)
        return BNNE_EFORMAT;
    if (ferror(file))
        return BNNE_EIO;
    if (model->layers[0].type != 0 || model->layers[0].in_ch != 1 ||
        model->layers[count - 1].type != 1)
        return BNNE_EFORMAT;
    engine->info.kind = BNNE_MODEL_CNN;
    engine->info.layer_count = count;
    engine->info.input = (BNNE_TensorDesc){BNNE_F32, 3, {1, CNN_IMAGE, CNN_IMAGE}};
    engine->info.output = (BNNE_TensorDesc){BNNE_F32, 1, {model->layers[count - 1].out_f}};
    return BNNE_OK;
}

static BNNE_Status cnn_prepare(BNNE_Engine engine) {
    CNN_Model* model = engine->model;
    BNN_Caps caps = {0};
    BNNE_Status status = bnne_runtime_open(engine, &caps);
    if (status != BNNE_OK)
        return status;
    if (caps.api_version != BNN_API_VERSION || !caps.supports_gemm || !caps.supports_gemv ||
        !caps.group_size || caps.group_size % 8 || !caps.max_rows)
        return bnne_error(engine, BNNE_ENOTSUP, "device lacks CNN GEMM/GEMV capabilities");
    model->device_group = caps.group_size;
    size_t max_input = 0, max_output = 0, matrix_bytes = 0;
    /* Validate all device and workspace sizes before creating padded matrices. */
    for (uint32_t i = 0; i < model->count; i++) {
        CNN_Layer* l = &model->layers[i];
        if (l->type == 2)
            continue;
        uint64_t pad_k = (uint64_t)l->groups * caps.group_size;
        if (l->group > caps.group_size || pad_k > caps.max_in_features ||
            l->out_f > caps.max_out_features || l->rows > caps.max_rows)
            return bnne_error(engine, BNNE_ENOTSUP, "layer %u exceeds device limits", i);
        l->pad_k = (uint32_t)pad_k;
        uint64_t weights = (uint64_t)l->out_f * (l->pad_k / 8);
        uint64_t scales = (uint64_t)l->out_f * l->groups * sizeof(uint16_t);
        uint64_t in_bytes = (uint64_t)l->rows * l->pad_k;
        uint64_t out_bytes = (uint64_t)l->rows * l->out_f * sizeof(float);
        if (weights > CNN_MAX_MATRIX_BYTES || weights + scales > CNN_MAX_MODEL_BYTES - matrix_bytes ||
            in_bytes > CNN_MAX_WORK_BYTES || out_bytes > CNN_MAX_WORK_BYTES)
            return bnne_error(engine, BNNE_ERANGE, "layer %u exceeds CNN memory limits", i);
        matrix_bytes += (size_t)(weights + scales);
        if (in_bytes > max_input) max_input = (size_t)in_bytes;
        if (out_bytes > max_output) max_output = (size_t)out_bytes;
    }
    for (uint32_t i = 0; i < model->count; i++) {
        CNN_Layer* l = &model->layers[i];
        if (l->type == 2)
            continue;
        size_t weight_bytes = (size_t)l->out_f * (l->pad_k / 8);
        size_t scale_count = (size_t)l->out_f * l->groups;
        if (!weight_bytes || weight_bytes > CNN_MAX_MATRIX_BYTES)
            return BNNE_ERANGE;
        uint8_t* padded = calloc(1, weight_bytes);
        if (!padded)
            return BNNE_ENOMEM;
        for (uint32_t o = 0; o < l->out_f; o++)
            for (uint32_t g = 0; g < l->groups; g++)
                for (uint32_t t = 0; t < l->group; t++) {
                    uint32_t src = g * l->group + t;
                    if ((l->bits[(size_t)o * l->row_bytes + (src >> 3)] >> (src & 7)) & 1u) {
                        uint32_t dst = g * caps.group_size + t;
                        padded[(size_t)o * (l->pad_k / 8) + (dst >> 3)] |= (uint8_t)(1u << (dst & 7));
                    }
                }
        BNN_MatrixDesc desc = {
            .in_features = l->pad_k, .out_features = l->out_f,
            .group_size = caps.group_size, .weight_bits = padded,
            .weight_bytes = weight_bytes, .scale_fp16 = l->scale_bits,
            .scale_count = scale_count,
        };
        status = bnne_accel_status(engine, BNN_MatrixCreate(engine->device, &desc, &l->matrix),
                                   "BNN_MatrixCreate");
        free(padded);
        if (status != BNNE_OK)
            return status;
        free(l->bits); l->bits = NULL;
        free(l->scale_bits); l->scale_bits = NULL;
    }
    model->work[0] = malloc((size_t)model->max_fmap * sizeof(float));
    model->work[1] = malloc((size_t)model->max_fmap * sizeof(float));
    model->quantized = malloc(model->max_fmap);
    model->im2col = malloc(max_input);
    model->accum = malloc(max_output);
    if (!model->work[0] || !model->work[1] || !model->quantized || !model->im2col || !model->accum)
        return BNNE_ENOMEM;
    return bnne_runtime_buffers(engine, max_input, max_output);
}

static void cnn_pool(const CNN_Layer* l, const float* x, uint32_t channels, float* y) {
    uint32_t ho = l->height / l->pool, wo = l->width / l->pool;
    for (uint32_t c = 0; c < channels; c++)
        for (uint32_t h = 0; h < ho; h++)
            for (uint32_t w = 0; w < wo; w++) {
                float best = -INFINITY;
                for (uint32_t i = 0; i < l->pool; i++)
                    for (uint32_t j = 0; j < l->pool; j++) {
                        float v = x[(size_t)c * l->height * l->width +
                                    (h * l->pool + i) * l->width + w * l->pool + j];
                        if (v > best) best = v;
                    }
                y[(size_t)c * ho * wo + h * wo + w] = best;
            }
}

static BNNE_Status cnn_run(BNNE_Engine engine, const BNNE_Tensor* input,
                           BNNE_Tensor* output, BNNE_Stats* stats) {
    CNN_Model* model = engine->model;
    const float* x = input->data;
    float* y = model->work[0];
    uint32_t channels = 1, height = CNN_IMAGE, width = CNN_IMAGE;
    for (uint32_t i = 0; i < model->count; i++) {
        CNN_Layer* l = &model->layers[i];
        if (l->type == 2) {
            cnn_pool(l, x, channels, y);
            height /= l->pool;
            width /= l->pool;
        } else {
            uint32_t rows = l->type == 0 ? height * width : 1;
            uint32_t quant_count = l->type == 0 ? channels * height * width : l->total;
            float scale;
            uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
            BNNE_Status status = bnne_quantize(x, quant_count, model->quantized, &scale);
            if (engine->options.enable_profile) stats->quantize_ns += bnne_now_ns() - start;
            if (status != BNNE_OK)
                return bnne_error(engine, status, "layer %u cannot quantize activations", i);
            if (l->type == 0) {
                /* Quantize the entire CHW map once; spatial and group padding are zero. */
                int pad = (int)(l->kernel / 2);
                memset(model->im2col, 0, (size_t)rows * l->pad_k);
                for (uint32_t h = 0; h < height; h++)
                    for (uint32_t w = 0; w < width; w++) {
                        int8_t* row = model->im2col + ((size_t)h * width + w) * l->pad_k;
                        for (uint32_t k = 0; k < l->total; k++) {
                            uint32_t c = k / (l->kernel * l->kernel);
                            uint32_t rem = k % (l->kernel * l->kernel);
                            int ih = (int)h + (int)(rem / l->kernel) - pad;
                            int iw = (int)w + (int)(rem % l->kernel) - pad;
                            if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width)
                                row[(k / l->group) * model->device_group + k % l->group] =
                                    model->quantized[(size_t)c * rows + (uint32_t)ih * width + (uint32_t)iw];
                        }
                    }
                status = bnne_runtime_gemm(engine, l->matrix, model->im2col, rows, l->pad_k,
                                           model->accum, l->out_f, stats);
                if (status != BNNE_OK) return status;
                for (uint32_t p = 0; p < rows; p++)
                    for (uint32_t o = 0; o < l->out_f; o++) {
                        float v = l->post_a[o] * (model->accum[(size_t)p * l->out_f + o] * scale) + l->post_b[o];
                        if (!isfinite(v))
                            return bnne_error(engine, BNNE_ECOMPUTE, "layer %u produced NaN or Inf", i);
                        y[(size_t)o * rows + p] = l->relu && v < 0.0f ? 0.0f : v;
                    }
                channels = l->out_ch;
            } else {
                memset(model->im2col, 0, l->pad_k);
                for (uint32_t g = 0; g < l->groups; g++)
                    memcpy(model->im2col + (size_t)g * model->device_group,
                           model->quantized + (size_t)g * l->group, l->group);
                status = bnne_runtime_gemv(engine, l->matrix, model->im2col, l->pad_k,
                                           model->accum, l->out_f, stats);
                if (status != BNNE_OK) return status;
                for (uint32_t o = 0; o < l->out_f; o++) {
                    y[o] = l->post_a[o] * (model->accum[o] * scale) + l->post_b[o];
                    if (!isfinite(y[o]))
                        return bnne_error(engine, BNNE_ECOMPUTE, "layer %u produced NaN or Inf", i);
                    if (l->relu && y[o] < 0.0f) y[o] = 0.0f;
                }
                channels = l->out_f;
                height = width = 1;
            }
            if (engine->options.enable_profile) {
                stats->logical_macs += (uint64_t)l->total * l->out_f * rows;
                stats->executed_macs += (uint64_t)l->pad_k * l->out_f * rows;
            }
        }
        x = y;
        y = y == model->work[0] ? model->work[1] : model->work[0];
    }
    memcpy(output->data, x, (size_t)engine->info.output.dims[0] * sizeof(float));
    return BNNE_OK;
}

const BNNE_ModelOps bnne_cnn_ops = {
    .load = cnn_load, .prepare = cnn_prepare, .run = cnn_run,
    .destroy = cnn_destroy, .token = NULL,
};
