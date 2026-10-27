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

    if (m->caps.group_size != BNNE_Q1_0_GROUP ||
        !(m->caps.supported_ops & BNN_CAP_GEMV_F32))
        return bnne_error(BNNE_ENOTSUP, "Q1_0 requires group size 128 and F32 GEMV support");
    if (in % BNNE_Q1_0_GROUP != 0)
        return bnne_error(BNNE_EFORMAT, "Q1_0 input width is not group aligned");

    uint32_t max_in = m->caps.max_in_features;
    uint32_t max_out = m->caps.max_out_features;
    if (!max_in || !max_out)
        return bnne_error(BNNE_ENOTSUP, "accelerator does not report matrix limits");
    if (in > max_in)
        return bnne_error(BNNE_ENOTSUP, "Q1_0 input width %u exceeds accelerator limit %u", in,
                          max_in);

    uint32_t count = out_features / max_out + (out_features % max_out != 0);
    l->chunks = calloc(count, sizeof *l->chunks);
    if (!l->chunks)
        return bnne_error(BNNE_ENOMEM, "out of memory");
    l->n_chunks = count;

    size_t row_bytes = in / 8u;
    size_t groups = in / BNNE_Q1_0_GROUP;
    for (uint32_t c = 0; c < count; c++) {
        bnne_linear_chunk* chunk = &l->chunks[c];
        uint32_t out_off = c * max_out;
        uint32_t out_len = out_features - out_off < max_out ? out_features - out_off : max_out;
        chunk->out_off = out_off;
        chunk->out_len = out_len;
        BNNE_Status status = chunk_create(
            m, chunk, in, out_len, bits + (size_t)out_off * row_bytes,
            scales + (size_t)out_off * groups);
        if (status != BNNE_OK) {
            bnne_linear_free(l);
            return status;
        }
    }

    m->max_in = m->max_in > in ? m->max_in : in;
    uint32_t chunk_max = out_features < max_out ? out_features : max_out;
    m->max_out = m->max_out > chunk_max ? m->max_out : chunk_max;
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
                                float* y) {
    if (!c || !l || !x || !y || !c->in_buf || !c->out_buf)
        return BNNE_EINVAL;

    int stats = c->params.enable_stats;
    size_t input_bytes = (size_t)l->in * sizeof(float);
    uint64_t start = stats ? bnne_now_ns() : 0;
    BNNE_Status status = bnne_accel_error(
        BNN_BufferWrite(c->in_buf, 0, x, input_bytes), "BNN_BufferWrite");
    if (stats) {
        c->stats.transfer_ns += bnne_now_ns() - start;
        c->stats.bytes_written += input_bytes;
    }
    if (status != BNNE_OK)
        return status;

    for (uint32_t i = 0; i < l->n_chunks; i++) {
        const bnne_linear_chunk* chunk = &l->chunks[i];
        size_t output_bytes = (size_t)chunk->out_len * sizeof(float);

        BNN_Job job = {0};
        job.operation = BNN_OP_GEMV_F32;
        job.matrix = chunk->matrix;
        job.input.buffer = c->in_buf;
        job.input.type = BNN_TYPE_F32;
        job.output.buffer = c->out_buf;
        job.output.type = BNN_TYPE_F32;
        job.rows = 1;

        start = stats ? bnne_now_ns() : 0;
        status = bnne_accel_error(BNN_Execute(c->model->device, &job), "BNN_Execute");
        if (stats)
            c->stats.execute_ns += bnne_now_ns() - start;
        if (status != BNNE_OK)
            return status;

        start = stats ? bnne_now_ns() : 0;
        status = bnne_accel_error(
            BNN_BufferRead(c->out_buf, 0, y + chunk->out_off, output_bytes), "BNN_BufferRead");
        if (stats) {
            c->stats.transfer_ns += bnne_now_ns() - start;
            c->stats.device_calls++;
            c->stats.bytes_read += output_bytes;
        }
        if (status != BNNE_OK)
            return status;
    }

    if (stats) {
        uint64_t macs = (uint64_t)l->in * l->out;
        c->stats.logical_macs += macs;
        c->stats.executed_macs += macs;
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
