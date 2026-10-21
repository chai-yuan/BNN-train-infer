/* Standalone public-API regression runner. Optional argv[1]: raw successful outputs. */
#include "bnn-accelerator.h"

#include <fenv.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static void status_check(BNN_Status got, BNN_Status want, const char *expr, int line) {
    if (got != want) {
        fprintf(stderr, "%s:%d: %s returned %d (%s), expected %d\n",
                __FILE__, line, expr, (int)got, BNN_StatusString(got), (int)want);
        exit(1);
    }
}
#define STATUS(expr, want) status_check((expr), (want), #expr, __LINE__)
#define OK(expr) STATUS((expr), BNN_OK)

/* Literal values are the oracle: no backend half converter is reused. */
static const struct { uint16_t half; float value; } scales[] = {
    {0x3c00, 1.0f}, {0xbc00, -1.0f}, {0x3800, 0.5f}, {0x4000, 2.0f},
    {0x0001, 0x1p-24f}, {0x8001, -0x1p-24f},
    {0x03ff, 1023.0f * 0x1p-24f}, {0x83ff, -1023.0f * 0x1p-24f},
    {0x0400, 0x1p-14f}, {0x8400, -0x1p-14f},
    {0x0000, 0.0f}, {0x8000, -0.0f},
    {0x3c01, 1.0f + 0x1p-10f}, {0x7bff, 65504.0f}
};
#define SCALE_COUNT (sizeof scales / sizeof scales[0])

static FILE *dump_file;
static size_t successful_jobs;
static size_t dumped_bytes;

typedef struct {
    BNN_Device device;
    BNN_Matrix matrix;
    uint32_t k, n, groups;
    uint8_t *weights, *uploaded_weights;
    uint16_t *half, *uploaded_half;
    float *scale;
} Matrix;

typedef struct {
    Matrix *matrix;
    BNN_Job job;
    size_t input_bytes, output_bytes, input_stride, output_stride, row_bytes;
    unsigned char *input, *actual, *expected;
} Case;

static void *allocate(size_t bytes) {
    void *p = malloc(bytes);
    CHECK(p != NULL);
    return p;
}

static void same_bytes(const void *got, const void *want, size_t bytes, const Case *c) {
    const unsigned char *a = got, *b = want;
    size_t i;
    if (memcmp(got, want, bytes) == 0) return;
    for (i = 0; i < bytes && a[i] == b[i]; ++i) {}
    fprintf(stderr, "op=%d K=%u N=%u M=%u in_stride=%zu out_stride=%zu: "
            "byte %zu got 0x%02x expected 0x%02x\n",
            (int)c->job.operation, c->matrix->k, c->matrix->n, c->job.rows,
            c->job.input_row_stride, c->job.output_row_stride,
            i, (unsigned)a[i], (unsigned)b[i]);
    exit(1);
}

static BNN_MatrixDesc description(const Matrix *m) {
    BNN_MatrixDesc d;
    memset(&d, 0, sizeof d);
    d.in_features = m->k;
    d.out_features = m->n;
    d.group_size = 128;
    d.weight_bits = m->weights;
    d.weight_bytes = (size_t)m->n * (m->k / 8);
    d.scale_fp16 = m->half;
    d.scale_count = (size_t)m->n * m->groups;
    return d;
}

static Matrix matrix_create(BNN_Device device, uint32_t k, uint32_t n, unsigned seed) {
    Matrix m;
    BNN_MatrixDesc d;
    uint32_t channel, feature, group;
    size_t weight_bytes = (size_t)n * (k / 8);
    size_t count = (size_t)n * (k / 128);
    memset(&m, 0, sizeof m);
    m.device = device; m.k = k; m.n = n; m.groups = k / 128;
    m.weights = allocate(weight_bytes);
    m.half = allocate(count * sizeof *m.half);
    m.scale = allocate(count * sizeof *m.scale);
    memset(m.weights, 0, weight_bytes);
    for (channel = 0; channel < n; ++channel) {
        for (feature = 0; feature < k; ++feature) {
            uint32_t v = feature + 977u * channel + 131u * seed;
            v = (v ^ (v >> 5)) * UINT32_C(0x45d9f3b);
            v ^= v >> 13;
            if (v & 1u)
                m.weights[(size_t)channel * (k / 8) + feature / 8] |= (uint8_t)(1u << (feature % 8));
        }
        for (group = 0; group < m.groups; ++group) {
            size_t i = (size_t)channel * m.groups + group;
            size_t s = (seed + 5u * channel + 3u * group) % SCALE_COUNT;
            m.half[i] = scales[s].half;
            m.scale[i] = scales[s].value;
        }
    }
    /* Keep poisoned upload arrays alive: an alias instead of a copy must fail. */
    m.uploaded_weights = allocate(weight_bytes);
    m.uploaded_half = allocate(count * sizeof *m.half);
    memcpy(m.uploaded_weights, m.weights, weight_bytes);
    memcpy(m.uploaded_half, m.half, count * sizeof *m.half);
    d = description(&m);
    d.weight_bits = m.uploaded_weights;
    d.scale_fp16 = m.uploaded_half;
    OK(BNN_MatrixCreate(device, &d, &m.matrix));
    CHECK(m.matrix != NULL);
    memset(m.uploaded_weights, 0, weight_bytes);
    memset(m.uploaded_half, 0, count * sizeof *m.half);
    return m;
}

static void matrix_destroy(Matrix *m) {
    BNN_MatrixDestroy(m->matrix);
    free(m->weights); free(m->half); free(m->scale);
    free(m->uploaded_weights); free(m->uploaded_half);
    memset(m, 0, sizeof *m);
}

static void fill_input(Case *c, unsigned seed) {
    uint32_t row, k;
    /* -128 in prefix, padding and suffix is legal: only logical inputs count. */
    memset(c->input, 0x80, c->input_bytes);
    for (row = 0; row < c->job.rows; ++row) {
        int8_t *x = (int8_t *)(c->input + c->job.input_offset + row * c->input_stride);
        for (k = 0; k < c->matrix->k; ++k)
            x[k] = (int8_t)((int)((k * 37u + (k / 7u) * 19u + row * 83u + seed * 29u) % 255u) - 127);
        x[0] = 127; x[1] = -127; x[c->matrix->k - 1] = (int8_t)(row + 1);
    }
}

static Case case_create(Matrix *m, BNN_Operation op, uint32_t rows, int padded, unsigned seed) {
    Case c;
    memset(&c, 0, sizeof c);
    c.matrix = m;
    c.job.operation = op; c.job.matrix = m->matrix; c.job.rows = rows;
    c.row_bytes = (size_t)m->n * (op == BNN_OP_DOT_GROUPS ? m->groups : 1u) * 4;
    c.input_stride = m->k + (padded ? 7u : 0u);
    c.output_stride = c.row_bytes + (padded ? 11u : 0u);
    c.job.input_offset = padded ? 3u : 0u;
    c.job.output_offset = padded ? 5u : 0u;
    c.job.input_row_stride = padded ? c.input_stride : 0;
    c.job.output_row_stride = padded ? c.output_stride : 0;
    c.input_bytes = c.job.input_offset + rows * c.input_stride + 9;
    c.output_bytes = c.job.output_offset + rows * c.output_stride + 13;
    c.input = allocate(c.input_bytes);
    c.actual = allocate(c.output_bytes);
    c.expected = allocate(c.output_bytes);
    OK(BNN_BufferAlloc(m->device, c.input_bytes, &c.job.input));
    OK(BNN_BufferAlloc(m->device, c.output_bytes, &c.job.output));
    fill_input(&c, seed);
    return c;
}

static void case_destroy(Case *c) {
    BNN_BufferFree(c->job.input); BNN_BufferFree(c->job.output);
    free(c->input); free(c->actual); free(c->expected);
    memset(c, 0, sizeof *c);
}

/* Independent scalar sum of signed products; no positive-bit sum shortcut. */
static int32_t reference_dot(const Matrix *m, const int8_t *x, uint32_t n, uint32_t g) {
    uint32_t k;
    int32_t sum = 0;
    for (k = g * 128; k < (g + 1) * 128; ++k) {
        uint8_t byte = m->weights[(size_t)n * (m->k / 8) + k / 8];
        int32_t weight = (byte & (1u << (k % 8))) ? 1 : -1;
        sum += (int32_t)x[k] * weight;
    }
    return sum;
}

static void execute_checked(Case *c) {
    Matrix *m = c->matrix;
    uint32_t row, n, g;
    unsigned char *input_copy = allocate(c->input_bytes);
    memset(c->expected, 0xa5, c->output_bytes);
    OK(BNN_BufferWrite(c->job.input, 0, c->input, c->input_bytes));
    OK(BNN_BufferWrite(c->job.output, 0, c->expected, c->output_bytes));
    for (row = 0; row < c->job.rows; ++row) {
        const int8_t *x = (const int8_t *)(c->input + c->job.input_offset + row * c->input_stride);
        unsigned char *y = c->expected + c->job.output_offset + row * c->output_stride;
        for (n = 0; n < m->n; ++n) {
            volatile float accumulator = 0.0f;
            for (g = 0; g < m->groups; ++g) {
                int32_t dot = reference_dot(m, x, n, g);
                if (c->job.operation == BNN_OP_DOT_GROUPS) {
                    memcpy(y + ((size_t)n * m->groups + g) * sizeof dot, &dot, sizeof dot);
                } else {
                    volatile float product = (float)dot * m->scale[(size_t)n * m->groups + g];
                    accumulator = accumulator + product;
                }
            }
            if (c->job.operation != BNN_OP_DOT_GROUPS) {
                float result = accumulator;
                memcpy(y + n * sizeof result, &result, sizeof result);
            }
        }
    }
    OK(BNN_Execute(m->device, &c->job));
    OK(BNN_BufferRead(c->job.output, 0, c->actual, c->output_bytes));
    same_bytes(c->actual, c->expected, c->output_bytes, c);
    OK(BNN_BufferRead(c->job.input, 0, input_copy, c->input_bytes));
    same_bytes(input_copy, c->input, c->input_bytes, c);
    free(input_copy);
    for (row = 0; row < c->job.rows; ++row) {
        if (dump_file)
            CHECK(fwrite(c->actual + c->job.output_offset + row * c->output_stride,
                         1, c->row_bytes, dump_file) == c->row_bytes);
        dumped_bytes += c->row_bytes;
    }
    ++successful_jobs;
}

/* Reset to a sentinel unlike valid output, so partial writes cannot hide. */
static void rejected(BNN_Device device, const BNN_Job *job, Case *watched, BNN_Status want) {
    memset(watched->expected, 0x5a, watched->output_bytes);
    OK(BNN_BufferWrite(watched->job.output, 0, watched->expected, watched->output_bytes));
    STATUS(BNN_Execute(device, job), want);
    OK(BNN_BufferRead(watched->job.output, 0, watched->actual, watched->output_bytes));
    same_bytes(watched->actual, watched->expected, watched->output_bytes, watched);
}

static void shape_sweep(BNN_Device device) {
    static const uint32_t ks[] = {128, 384, 4096}, ns[] = {1, 3, 17}, ms[] = {1, 3};
    static const BNN_Operation ops[] = {BNN_OP_DOT_GROUPS, BNN_OP_GEMV, BNN_OP_GEMM};
    size_t ki, ni, mi, oi;
    unsigned seed = 0;
    int padded;
    for (ki = 0; ki < sizeof ks / sizeof ks[0]; ++ki) {
        for (ni = 0; ni < sizeof ns / sizeof ns[0]; ++ni) {
            Matrix m = matrix_create(device, ks[ki], ns[ni], seed++);
            for (mi = 0; mi < sizeof ms / sizeof ms[0]; ++mi) {
                for (oi = 0; oi < sizeof ops / sizeof ops[0]; ++oi) {
                    if (ops[oi] == BNN_OP_GEMV && ms[mi] != 1) continue;
                    for (padded = 0; padded <= 1; ++padded) {
                        Case c = case_create(&m, ops[oi], ms[mi], padded, seed);
                        execute_checked(&c);
                        /* Same buffers, matrix and device, different activation. */
                        fill_input(&c, seed + 17);
                        execute_checked(&c);
                        case_destroy(&c);
                    }
                }
            }
            matrix_destroy(&m);
        }
    }
}

static void scale_sweep(BNN_Device device) {
    size_t s;
    for (s = 0; s < SCALE_COUNT; ++s) {
        Matrix m = matrix_create(device, 128, 1, (unsigned)s);
        Case c = case_create(&m, BNN_OP_GEMV, 1, 1, 0);
        uint32_t k;
        int8_t *x = (int8_t *)(c.input + c.job.input_offset);
        /* Exact dot=1: GEMV must return the literal scale, except -0 sums to +0. */
        memset(x, 0, 128);
        for (k = 0; k < 128; ++k) {
            if (m.weights[k / 8] & (1u << (k % 8))) { x[k] = 1; break; }
        }
        if (k == 128) x[0] = -1;
        CHECK(reference_dot(&m, x, 0, 0) == 1);
        execute_checked(&c);
        case_destroy(&c);
        matrix_destroy(&m);
    }
}

static void invalid_inputs(BNN_Device device) {
    static const BNN_Operation ops[] = {BNN_OP_DOT_GROUPS, BNN_OP_GEMM, BNN_OP_GEMV};
    Matrix m = matrix_create(device, 384, 3, 4);
    size_t oi;
    for (oi = 0; oi < sizeof ops / sizeof ops[0]; ++oi) {
        Case c = case_create(&m, ops[oi], ops[oi] == BNN_OP_GEMV ? 1 : 3, 1, 19);
        uint32_t row;
        execute_checked(&c);
        for (row = 0; row < c.job.rows; ++row) {
            static const uint32_t positions[] = {0, 127, 128, 383};
            size_t i;
            for (i = 0; i < sizeof positions / sizeof positions[0]; ++i) {
                size_t at = c.job.input_offset + row * c.input_stride + positions[i];
                unsigned char saved = c.input[at];
                c.input[at] = 0x80;
                OK(BNN_BufferWrite(c.job.input, 0, c.input, c.input_bytes));
                rejected(device, &c.job, &c, BNN_ERANGE);
                c.input[at] = saved;
                execute_checked(&c); /* Every rejected input is followed by recovery. */
            }
        }
        case_destroy(&c);
    }
    matrix_destroy(&m);
}

static void invalid_jobs(BNN_Device device) {
    Matrix m = matrix_create(device, 384, 3, 3);
    Case c = case_create(&m, BNN_OP_GEMM, 3, 1, 5);
    BNN_Job bad;
    execute_checked(&c);
#define BAD_JOB(change, want) do { bad = c.job; change; rejected(device, &bad, &c, want); } while (0)
    BAD_JOB(bad.operation = (BNN_Operation)99, BNN_ENOTSUP);
    BAD_JOB(bad.operation = BNN_OP_GEMV, BNN_EINVAL);
    BAD_JOB(bad.rows = 0, BNN_EINVAL);
    BAD_JOB(bad.rows = UINT32_MAX, BNN_ERANGE);
    BAD_JOB(bad.matrix = NULL, BNN_EINVAL);
    BAD_JOB(bad.input = NULL, BNN_EINVAL);
    BAD_JOB(bad.output = NULL, BNN_EINVAL);
    BAD_JOB(bad.input = bad.output, BNN_EINVAL);
    BAD_JOB(bad.input_row_stride = m.k - 1, BNN_EINVAL);
    BAD_JOB(bad.output_row_stride = c.row_bytes - 1, BNN_EINVAL);
    BAD_JOB(bad.input_offset = c.input_bytes, BNN_ERANGE);
    BAD_JOB(bad.output_offset = c.output_bytes, BNN_ERANGE);
    BAD_JOB(bad.input_offset = SIZE_MAX, BNN_ERANGE);
    BAD_JOB(bad.output_offset = SIZE_MAX, BNN_ERANGE);
    BAD_JOB(bad.input_row_stride = SIZE_MAX, BNN_ERANGE);
    BAD_JOB(bad.output_row_stride = SIZE_MAX, BNN_ERANGE);
    BAD_JOB(bad.input_offset = c.input_bytes - m.k - 2 * c.input_stride + 1, BNN_ERANGE);
    BAD_JOB(bad.output_offset = c.output_bytes - c.row_bytes - 2 * c.output_stride + 1, BNN_ERANGE);
    rejected(device, NULL, &c, BNN_EINVAL);
    rejected(NULL, &c.job, &c, BNN_EINVAL);
#undef BAD_JOB
    execute_checked(&c);
    case_destroy(&c);
    matrix_destroy(&m);
}

static void invalid_matrices(BNN_Device device) {
    Matrix m = matrix_create(device, 128, 3, 1);
    BNN_MatrixDesc d = description(&m), bad;
    BNN_Matrix result;
    uint16_t special[3];
    static const uint16_t invalid_scales[] = {0x7c00, 0xfc00, 0x7e00, 0x7c01, 0xfe00};
    size_t i;
#define BAD_MATRIX(change, want) do { \
    bad = d; change; result = m.matrix; \
    STATUS(BNN_MatrixCreate(device, &bad, &result), want); CHECK(result == NULL); \
} while (0)
    BAD_MATRIX(bad.in_features = 0, BNN_EINVAL);
    BAD_MATRIX(bad.out_features = 0, BNN_EINVAL);
    BAD_MATRIX(bad.group_size = 0, BNN_EINVAL);
    BAD_MATRIX(bad.in_features = 129, BNN_EINVAL);
    BAD_MATRIX(bad.in_features = 4224, BNN_ENOTSUP);
    BAD_MATRIX(bad.out_features = 4097, BNN_ENOTSUP);
    BAD_MATRIX(bad.group_size = 64, BNN_ENOTSUP);
    BAD_MATRIX(bad.weight_bits = NULL, BNN_EINVAL);
    BAD_MATRIX(bad.scale_fp16 = NULL, BNN_EINVAL);
    BAD_MATRIX(bad.weight_bytes--, BNN_EINVAL);
    BAD_MATRIX(bad.weight_bytes++, BNN_EINVAL);
    BAD_MATRIX(bad.scale_count--, BNN_EINVAL);
    BAD_MATRIX(bad.scale_count++, BNN_EINVAL);
    for (i = 0; i < sizeof invalid_scales / sizeof invalid_scales[0]; ++i) {
        memcpy(special, m.half, sizeof special);
        special[2] = invalid_scales[i]; /* Validate beyond the first scale. */
        BAD_MATRIX(bad.scale_fp16 = special, BNN_EINVAL);
    }
#undef BAD_MATRIX
    result = m.matrix;
    STATUS(BNN_MatrixCreate(NULL, &d, &result), BNN_EINVAL); CHECK(result == NULL);
    result = m.matrix;
    STATUS(BNN_MatrixCreate(device, NULL, &result), BNN_EINVAL); CHECK(result == NULL);
    STATUS(BNN_MatrixCreate(device, &d, NULL), BNN_EINVAL);
    { Case c = case_create(&m, BNN_OP_GEMV, 1, 0, 9); execute_checked(&c); case_destroy(&c); }
    matrix_destroy(&m);
}

static void transfers_and_busy(BNN_Device device) {
    BNN_Buffer buffer = NULL, result;
    unsigned char data[19], got[19], unchanged[19];
    Matrix m;
    size_t i;
    for (i = 0; i < sizeof data; ++i) data[i] = (unsigned char)(i * 13);
    OK(BNN_BufferAlloc(device, sizeof data, &buffer));
    STATUS(BNN_DeviceClose(device), BNN_EBUSY); /* Buffer alone keeps device alive. */
    OK(BNN_BufferWrite(buffer, 0, data, sizeof data));
    OK(BNN_BufferWrite(buffer, 3, data + 5, 7));
    for (i = 0; i < 7; ++i) data[3 + i] = (unsigned char)((5 + i) * 13);
    OK(BNN_BufferRead(buffer, 0, got, sizeof got)); CHECK(memcmp(data, got, sizeof data) == 0);
    OK(BNN_BufferRead(buffer, 4, got, 6)); CHECK(memcmp(data + 4, got, 6) == 0);
    OK(BNN_BufferRead(buffer, sizeof data, NULL, 0));
    OK(BNN_BufferWrite(buffer, sizeof data, NULL, 0));
    STATUS(BNN_BufferWrite(buffer, sizeof data + 1, NULL, 0), BNN_ERANGE);
    STATUS(BNN_BufferRead(buffer, sizeof data + 1, NULL, 0), BNN_ERANGE);
    STATUS(BNN_BufferWrite(buffer, sizeof data, data, 1), BNN_ERANGE);
    STATUS(BNN_BufferWrite(buffer, SIZE_MAX, data, 1), BNN_ERANGE);
    STATUS(BNN_BufferWrite(buffer, 1, data, SIZE_MAX), BNN_ERANGE);
    STATUS(BNN_BufferWrite(buffer, 0, NULL, 1), BNN_EINVAL);
    memset(got, 0x33, sizeof got); memcpy(unchanged, got, sizeof got);
    STATUS(BNN_BufferRead(buffer, sizeof data, got, 1), BNN_ERANGE);
    STATUS(BNN_BufferRead(buffer, SIZE_MAX, got, 1), BNN_ERANGE);
    STATUS(BNN_BufferRead(buffer, 1, got, SIZE_MAX), BNN_ERANGE);
    CHECK(memcmp(got, unchanged, sizeof got) == 0);
    STATUS(BNN_BufferRead(buffer, 0, NULL, 1), BNN_EINVAL);
    STATUS(BNN_BufferRead(NULL, 0, got, 1), BNN_EINVAL);
    STATUS(BNN_BufferWrite(NULL, 0, data, 1), BNN_EINVAL);
    result = buffer;
    STATUS(BNN_BufferAlloc(device, 0, &result), BNN_EINVAL); CHECK(result == NULL);
    result = buffer;
    STATUS(BNN_BufferAlloc(NULL, 1, &result), BNN_EINVAL); CHECK(result == NULL);
    STATUS(BNN_BufferAlloc(device, 1, NULL), BNN_EINVAL);
    OK(BNN_BufferRead(buffer, 0, got, sizeof got)); CHECK(memcmp(data, got, sizeof data) == 0);
    BNN_BufferFree(buffer);
    m = matrix_create(device, 128, 1, 0);
    STATUS(BNN_DeviceClose(device), BNN_EBUSY); /* Matrix alone also keeps it alive. */
    matrix_destroy(&m);
}

static void isolation(BNN_Device first) {
    BNN_Device second = NULL;
    Matrix a, b;
    Case ca, cb;
    BNN_Job bad;
    unsigned i;
    OK(BNN_DeviceOpen(0, &second)); CHECK(second != NULL && second != first);
    a = matrix_create(first, 128, 3, 2);
    b = matrix_create(second, 384, 17, 7);
    ca = case_create(&a, BNN_OP_GEMV, 1, 1, 1);
    cb = case_create(&b, BNN_OP_DOT_GROUPS, 3, 1, 8);
    for (i = 0; i < 3; ++i) {
        fill_input(&ca, i); execute_checked(&ca);
        fill_input(&cb, i + 31); execute_checked(&cb);
    }
    bad = ca.job; bad.matrix = b.matrix; rejected(first, &bad, &ca, BNN_EINVAL);
    bad = ca.job; bad.input = cb.job.input; rejected(first, &bad, &ca, BNN_EINVAL);
    bad = ca.job; bad.output = cb.job.output; rejected(first, &bad, &cb, BNN_EINVAL);
    rejected(second, &ca.job, &ca, BNN_EINVAL);
    STATUS(BNN_DeviceClose(first), BNN_EBUSY);
    STATUS(BNN_DeviceClose(second), BNN_EBUSY);
    execute_checked(&ca); execute_checked(&cb);
    case_destroy(&cb); matrix_destroy(&b); OK(BNN_DeviceClose(second));
    execute_checked(&ca); /* Destroying another device cannot disturb this one. */
    case_destroy(&ca); matrix_destroy(&a);
}

int main(int argc, char **argv) {
    BNN_Device device = NULL, failed;
    BNN_Caps caps;
    float one = 1.0f;
    uint32_t bits;
    CHECK(argc <= 2);
    CHECK(sizeof(float) == 4 && sizeof(int32_t) == 4 && FLT_MANT_DIG == 24);
    memcpy(&bits, &one, sizeof bits); CHECK(bits == UINT32_C(0x3f800000));
    CHECK(fesetround(FE_TONEAREST) == 0);
    if (argc == 2) { dump_file = fopen(argv[1], "wb"); CHECK(dump_file != NULL); }
    STATUS(BNN_DeviceOpen(0, NULL), BNN_EINVAL);
    OK(BNN_DeviceOpen(0, &device)); CHECK(device != NULL);
    failed = device;
    STATUS(BNN_DeviceOpen(UINT32_MAX, &failed), BNN_ENOTSUP); CHECK(failed == NULL);
    STATUS(BNN_DeviceClose(NULL), BNN_EINVAL);
    STATUS(BNN_DeviceGetCaps(NULL, &caps), BNN_EINVAL);
    STATUS(BNN_DeviceGetCaps(device, NULL), BNN_EINVAL);
    OK(BNN_DeviceGetCaps(device, &caps));
    CHECK(caps.api_version == BNN_API_VERSION && caps.group_size == 128);
    CHECK(caps.max_in_features >= 4096 && caps.max_out_features >= 17 && caps.max_rows >= 3);
    CHECK(caps.supports_dot_groups && caps.supports_gemv && caps.supports_gemm);
    BNN_MatrixDestroy(NULL); BNN_BufferFree(NULL);
    CHECK(BNN_StatusString(BNN_OK) != NULL && BNN_StatusString(BNN_ETIMEOUT) != NULL);
    CHECK(BNN_StatusString((BNN_Status)99) != NULL);
    transfers_and_busy(device);
    invalid_matrices(device);
    shape_sweep(device);
    scale_sweep(device);
    invalid_inputs(device);
    invalid_jobs(device);
    isolation(device);
    OK(BNN_DeviceClose(device));
    /* Reopen after full teardown as well as reusing a live device above. */
    OK(BNN_DeviceOpen(0, &device));
    { Matrix m = matrix_create(device, 128, 1, 0);
      Case c = case_create(&m, BNN_OP_DOT_GROUPS, 1, 0, 0);
      execute_checked(&c); case_destroy(&c); matrix_destroy(&m); }
    OK(BNN_DeviceClose(device));
    if (dump_file) CHECK(fclose(dump_file) == 0);
    printf("BNN accelerator: %zu successful jobs, %zu output bytes; integer/FP32 oracle, "
           "layout, validation, ownership, busy and isolation OK\n", successful_jobs, dumped_bytes);
    return 0;
}
