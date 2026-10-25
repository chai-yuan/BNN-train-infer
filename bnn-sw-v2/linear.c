#include "bnne-internal.h"

#include "gguf.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BNNE_Q1_0_GROUP 128u

static uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static void* alloc_array(size_t count, size_t size) {
    if (!count || size > SIZE_MAX / count)
        return NULL;
    return malloc(count * size);
}

BNNE_Status bnne_q1_0_read(const bnne_gguf* g, const char* tensor, uint32_t in, uint32_t out,
                           uint8_t** out_bits, uint16_t** out_scales) {
    const bnne_gguf_tensor* t = bnne_gguf_find(g, tensor);
    if (!t || t->type != 41 || t->rank != 2 || t->dims[0] != in || t->dims[1] != out ||
        in % BNNE_Q1_0_GROUP != 0 ||
        t->bytes != (uint64_t)out * (in / BNNE_Q1_0_GROUP) * 18)
        return bnne_error(BNNE_EFORMAT, "unexpected Q1_0 tensor '%s' shape or type", tensor);

    size_t row_bytes = in / 8u;
    size_t groups = in / BNNE_Q1_0_GROUP;
    uint8_t* bits = alloc_array((size_t)out, row_bytes);
    uint16_t* scales = alloc_array((size_t)out * groups, sizeof *scales);
    if (!bits || !scales) {
        free(bits);
        free(scales);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }

    const uint8_t* source = bnne_gguf_data(g, t);
    for (uint32_t o = 0; o < out; o++) {
        for (uint32_t grp = 0; grp < groups; grp++) {
            const uint8_t* block = source + ((size_t)o * groups + grp) * 18;
            uint16_t scale = read_u16(block);
            if (!isfinite(bnne_fp16_to_fp32(scale))) {
                free(bits);
                free(scales);
                return bnne_error(BNNE_EFORMAT, "Q1_0 scale in '%s' is not finite", tensor);
            }
            scales[(size_t)o * groups + grp] = scale;
            memcpy(bits + (size_t)o * row_bytes + grp * 16, block + 2, 16);
        }
    }

    *out_bits = bits;
    *out_scales = scales;
    return BNNE_OK;
}

static BNNE_Status chunk_create(struct bnne_model* m, bnne_linear_chunk* chunk,
                                uint32_t in, uint32_t out, const uint8_t* bits,
                                const uint16_t* scales) {
    BNN_MatrixDesc desc = {
        .in_features = in,
        .out_features = out,
        .group_size = BNNE_Q1_0_GROUP,
        .weight_bits = bits,
        .weight_bytes = (size_t)out * (in / 8u),
        .scale_fp16 = scales,
        .scale_count = (size_t)out * (in / BNNE_Q1_0_GROUP),
    };
    BNN_Status status = BNN_MatrixCreate(m->device, &desc, &chunk->matrix);
    if (status != BNN_OK)
        return bnne_accel_error(status, "BNN_MatrixCreate(Q1_0)");
    return BNNE_OK;
}

BNNE_Status bnne_linear_init(struct bnne_model* m, uint32_t in, uint32_t out_features,
                             const uint8_t* bits, const uint16_t* scales, bnne_linear* l) {
    if (!m || !l || !bits || !scales || in == 0 || out_features == 0)
        return BNNE_EINVAL;
    memset(l, 0, sizeof *l);
    l->in = in;
    l->out = out_features;

    if (m->caps.group_size != BNNE_Q1_0_GROUP)
        return bnne_error(BNNE_ENOTSUP, "Q1_0 requires accelerator group size 128");
    if (in % BNNE_Q1_0_GROUP != 0)
        return bnne_error(BNNE_EFORMAT, "Q1_0 input width is not group aligned");

    uint32_t max_in = m->caps.max_in_features;
    uint32_t max_out = m->caps.max_out_features;
    if (!max_in || !max_out)
        return bnne_error(BNNE_ENOTSUP, "accelerator does not report matrix limits");

    if (in <= max_in && out_features <= max_out) {
        l->n_chunks = 1;
        l->chunks = calloc(1, sizeof *l->chunks);
        if (!l->chunks)
            return bnne_error(BNNE_ENOMEM, "out of memory");
        l->chunks[0].in_len = in;
        l->chunks[0].out_len = out_features;
        BNNE_Status status = chunk_create(m, &l->chunks[0], in, out_features, bits, scales);
        if (status != BNNE_OK) {
            bnne_linear_free(l);
            return status;
        }
        m->max_out = m->max_out > out_features ? m->max_out : out_features;
    } else if (in <= max_in) {
        uint32_t count = (out_features + max_out - 1) / max_out;
        l->chunks = calloc(count, sizeof *l->chunks);
        if (!l->chunks)
            return bnne_error(BNNE_ENOMEM, "out of memory");
        l->n_chunks = count;
        for (uint32_t c = 0; c < count; c++) {
            bnne_linear_chunk* chunk = &l->chunks[c];
            uint32_t out_off = c * max_out;
            uint32_t out_len = out_features - out_off < max_out ? out_features - out_off : max_out;
            chunk->out_off = out_off;
            chunk->out_len = out_len;
            chunk->in_len = in;
            BNNE_Status status = chunk_create(
                m, chunk, in, out_len, bits + (size_t)out_off * (in / 8u),
                scales + (size_t)out_off * (in / BNNE_Q1_0_GROUP));
            if (status != BNNE_OK) {
                bnne_linear_free(l);
                return status;
            }
        }
        m->max_out = m->max_out > max_out ? m->max_out : max_out;
    } else if (out_features <= max_out) {
        uint32_t step = max_in - max_in % BNNE_Q1_0_GROUP;
        if (!step)
            return bnne_error(BNNE_ENOTSUP, "accelerator input chunk is smaller than a Q1_0 group");
        uint32_t count = (in + step - 1) / step;
        l->chunks = calloc(count, sizeof *l->chunks);
        if (!l->chunks)
            return bnne_error(BNNE_ENOMEM, "out of memory");
        l->n_chunks = count;
        for (uint32_t c = 0; c < count; c++) {
            bnne_linear_chunk* chunk = &l->chunks[c];
            uint32_t in_off = c * step;
            uint32_t in_len = in - in_off < step ? in - in_off : step;
            size_t row_bytes = in_len / 8u;
            size_t groups = in_len / BNNE_Q1_0_GROUP;
            if (in_len % BNNE_Q1_0_GROUP != 0) {
                bnne_linear_free(l);
                return bnne_error(BNNE_ENOTSUP, "Q1_0 input chunk is not group aligned");
            }
            uint8_t* part_bits = alloc_array((size_t)out_features, row_bytes);
            uint16_t* part_scales = alloc_array((size_t)out_features * groups, sizeof *part_scales);
            if (!part_bits || !part_scales) {
                free(part_bits);
                free(part_scales);
                bnne_linear_free(l);
                return bnne_error(BNNE_ENOMEM, "out of memory");
            }
            for (uint32_t o = 0; o < out_features; o++) {
                memcpy(part_bits + (size_t)o * row_bytes,
                       bits + (size_t)o * (in / 8u) + in_off / 8u, row_bytes);
                memcpy(part_scales + (size_t)o * groups,
                       scales + (size_t)o * (in / BNNE_Q1_0_GROUP) + in_off / BNNE_Q1_0_GROUP,
                       groups * sizeof *part_scales);
            }
            chunk->in_off = in_off;
            chunk->in_len = in_len;
            chunk->out_len = out_features;
            BNNE_Status status = chunk_create(m, chunk, in_len, out_features, part_bits, part_scales);
            free(part_bits);
            free(part_scales);
            if (status != BNNE_OK) {
                bnne_linear_free(l);
                return status;
            }
        }
        m->max_out = m->max_out > out_features ? m->max_out : out_features;
    } else {
        return bnne_error(BNNE_ENOTSUP, "Q1_0 weight exceeds both accelerator chunk limits");
    }

    m->max_in = m->max_in > in ? m->max_in : in;
    return BNNE_OK;
}

BNNE_Status bnne_linear_load(struct bnne_model* m, const bnne_gguf* g, const char* tensor,
                             bnne_linear* out) {
    const bnne_gguf_tensor* t = bnne_gguf_find(g, tensor);
    if (!t || t->rank != 2 || t->dims[0] == 0 || t->dims[1] == 0 ||
        t->dims[0] > UINT32_MAX || t->dims[1] > UINT32_MAX)
        return bnne_error(BNNE_EFORMAT, "missing or invalid Q1_0 tensor '%s'", tensor);

    uint32_t in = (uint32_t)t->dims[0];
    uint32_t outf = (uint32_t)t->dims[1];
    uint8_t* bits = NULL;
    uint16_t* scales = NULL;
    BNNE_Status status = bnne_q1_0_read(g, tensor, in, outf, &bits, &scales);
    if (status == BNNE_OK)
        status = bnne_linear_init(m, in, outf, bits, scales, out);
    free(bits);
    free(scales);
    return status;
}

void bnne_linear_free(bnne_linear* l) {
    if (!l)
        return;
    for (uint32_t i = 0; i < l->n_chunks; i++)
        BNN_MatrixDestroy(l->chunks[i].matrix);
    free(l->chunks);
    memset(l, 0, sizeof *l);
}

BNNE_Status bnne_linear_forward(struct bnne_context* c, const bnne_linear* l, const float* x,
                                uint32_t rows, float* y) {
    if (!c || !l || !x || !y || rows == 0 || !c->qbuf || !c->chunk || !c->in_buf || !c->out_buf)
        return BNNE_EINVAL;

    int stats = c->params.enable_stats;
    for (uint32_t r = 0; r < rows; r++) {
        float scale = 0.0f;
        uint64_t start = stats ? bnne_now_ns() : 0;
        BNNE_Status status = bnne_quantize(x + (size_t)r * l->in, l->in, c->qbuf, &scale);
        if (stats)
            c->stats.quantize_ns += bnne_now_ns() - start;
        if (status != BNNE_OK)
            return bnne_error(status, "activation quantization failed");

        memset(y + (size_t)r * l->out, 0, (size_t)l->out * sizeof(float));
        for (uint32_t i = 0; i < l->n_chunks; i++) {
            const bnne_linear_chunk* chunk = &l->chunks[i];
            size_t input_bytes = chunk->in_len;
            size_t output_bytes = (size_t)chunk->out_len * sizeof(float);

            start = stats ? bnne_now_ns() : 0;
            status = bnne_accel_error(
                BNN_BufferWrite(c->in_buf, 0, c->qbuf + chunk->in_off, input_bytes),
                "BNN_BufferWrite");
            if (stats)
                c->stats.transfer_ns += bnne_now_ns() - start;
            if (status != BNNE_OK)
                return status;

            BNN_Job job = {0};
            job.operation = BNN_OP_GEMV;
            job.matrix = chunk->matrix;
            job.input = c->in_buf;
            job.output = c->out_buf;
            job.rows = 1;

            start = stats ? bnne_now_ns() : 0;
            status = bnne_accel_error(BNN_Execute(c->model->device, &job), "BNN_Execute");
            if (stats)
                c->stats.execute_ns += bnne_now_ns() - start;
            if (status != BNNE_OK)
                return status;

            start = stats ? bnne_now_ns() : 0;
            status = bnne_accel_error(
                BNN_BufferRead(c->out_buf, 0, c->chunk, output_bytes), "BNN_BufferRead");
            if (stats) {
                c->stats.transfer_ns += bnne_now_ns() - start;
                c->stats.device_calls++;
                c->stats.bytes_written += input_bytes;
                c->stats.bytes_read += output_bytes;
            }
            if (status != BNNE_OK)
                return status;

            for (uint32_t j = 0; j < chunk->out_len; j++)
                y[(size_t)r * l->out + chunk->out_off + j] += c->chunk[j] * scale;
        }
        if (stats) {
            uint64_t macs = (uint64_t)l->in * l->out;
            c->stats.logical_macs += macs;
            c->stats.executed_macs += macs;
        }
    }
    return BNNE_OK;
}

BNNE_Status bnne_tensor_f32(struct bnne_model* m, const bnne_gguf* g, const char* tensor,
                            const float** out, size_t* out_n) {
    (void)m;
    if (!g || !tensor || !out || !out_n)
        return BNNE_EINVAL;
    const bnne_gguf_tensor* t = bnne_gguf_find(g, tensor);
    if (!t || (t->type != 0 && t->type != 1))
        return bnne_error(BNNE_EFORMAT, "missing or invalid F32/F16 tensor '%s'", tensor);

    uint64_t count = 1;
    for (uint32_t d = 0; d < t->rank; d++) {
        if (t->dims[d] == 0 || count > UINT64_MAX / t->dims[d])
            return bnne_error(BNNE_EFORMAT, "invalid shape for tensor '%s'", tensor);
        count *= t->dims[d];
    }
    if (count > SIZE_MAX / sizeof(float))
        return bnne_error(BNNE_ERANGE, "tensor '%s' is too large", tensor);

    float* values = alloc_array((size_t)count, sizeof *values);
    if (!values)
        return bnne_error(BNNE_ENOMEM, "out of memory");

    const uint8_t* source = bnne_gguf_data(g, t);
    for (uint64_t i = 0; i < count; i++) {
        if (t->type == 0) {
            uint32_t bits = read_u32(source + (size_t)i * 4);
            memcpy(&values[i], &bits, sizeof bits);
        } else {
            values[i] = bnne_fp16_to_fp32(read_u16(source + (size_t)i * 2));
        }
        if (!isfinite(values[i])) {
            free(values);
            return bnne_error(BNNE_EFORMAT, "non-finite value in tensor '%s'", tensor);
        }
    }

    *out = values;
    *out_n = (size_t)count;
    return BNNE_OK;
}
