#include "bnn-accelerator.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if CHAR_BIT != 8
#error "This backend requires 8-bit bytes."
#endif

#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MIN_EXP != -125 || FLT_MAX_EXP != 128
#error "This backend requires IEEE-754 binary32 float."
#endif

#ifdef __FAST_MATH__
#error "Compile this backend without fast-math."
#endif

#define SOFT_MAX_FEATURES 4096u
#define SOFT_GROUP_SIZE 128u
#define SOFT_MAX_GROUPS (SOFT_MAX_FEATURES / SOFT_GROUP_SIZE)

struct bnn_accel_device {
    size_t matrix_count;
    size_t buffer_count;
};

struct bnn_accel_matrix {
    BNN_Device owner;

    uint32_t in_features;
    uint32_t out_features;
    uint32_t groups;

    uint8_t* weight_bits;
    uint16_t* scale_fp16;
};

struct bnn_accel_buffer {
    BNN_Device owner;
    size_t bytes;
    unsigned char* data;
};

/* 检查大小乘法，防止 size_t 溢出。 */
static int SizeMultiply(size_t a, size_t b, size_t* result) {
    if (a != 0 && b > SIZE_MAX / a)
        return 0;

    *result = a * b;
    return 1;
}

static int RangeFits(size_t capacity, size_t offset, size_t bytes) {
    return offset <= capacity && bytes <= capacity - offset;
}

/*
 * 检查最后一行是否落在缓冲区内。
 * 使用除法检查，避免先计算 (rows - 1) * stride 而溢出。
 */
static int RowsFit(size_t capacity, size_t offset, size_t stride, size_t row_bytes, uint32_t rows) {
    size_t available;

    if (rows == 0 || row_bytes == 0 || stride < row_bytes || offset > capacity)
        return 0;

    available = capacity - offset;

    if (row_bytes > available)
        return 0;

    return (uintmax_t)(rows - 1u) <= (uintmax_t)((available - row_bytes) / stride);
}

/* 检查 FP32 位模式与 uint32_t 的解释是否兼容。 */
static int FloatFormatSupported(void) {
    float one = 1.0f;
    uint32_t bits;

    if (sizeof(float) != sizeof(uint32_t))
        return 0;

    memcpy(&bits, &one, sizeof bits);
    return bits == UINT32_C(0x3f800000);
}

/* IEEE-754 binary16 位模式转换为 binary32，保留次正规数。 */
static float HalfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t mant = h & 0x03ffu;
    int exp = (int)((h >> 10) & 0x1fu);
    uint32_t bits;
    float result;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;

            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }

            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | UINT32_C(0x7f800000) | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }

    memcpy(&result, &bits, sizeof result);
    return result;
}

/* 分别建立 FP32 舍入点，避免乘加融合。 */
static float FloatMultiply(float a, float b) {
    volatile float result = a * b;
    return result;
}

static float FloatAdd(float a, float b) {
    volatile float result = a + b;
    return result;
}

BNN_Status BNN_DeviceOpen(uint32_t device_index, BNN_Device* out_device) {
    BNN_Device device;

    if (out_device == NULL)
        return BNN_EINVAL;

    *out_device = NULL;

    if (device_index != 0 || !FloatFormatSupported())
        return BNN_ENOTSUP;

    device = calloc(1, sizeof *device);

    if (device == NULL)
        return BNN_ENOMEM;

    *out_device = device;
    return BNN_OK;
}

BNN_Status BNN_DeviceClose(BNN_Device device) {
    if (device == NULL)
        return BNN_EINVAL;

    if (device->matrix_count != 0 || device->buffer_count != 0)
        return BNN_EBUSY;

    free(device);
    return BNN_OK;
}

BNN_Status BNN_DeviceGetCaps(BNN_Device device, BNN_Caps* out_caps) {
    if (device == NULL || out_caps == NULL)
        return BNN_EINVAL;

    memset(out_caps, 0, sizeof *out_caps);

    out_caps->api_version = BNN_API_VERSION;
    out_caps->max_in_features = SOFT_MAX_FEATURES;
    out_caps->max_out_features = SOFT_MAX_FEATURES;
    out_caps->max_rows = UINT32_MAX;
    out_caps->group_size = SOFT_GROUP_SIZE;
    out_caps->supports_gemv = 1;
    out_caps->supports_gemm = 1;
    out_caps->supports_dot_groups = 1;

    return BNN_OK;
}

BNN_Status BNN_MatrixCreate(BNN_Device device, const BNN_MatrixDesc* desc, BNN_Matrix* out_matrix) {
    BNN_Matrix matrix;
    size_t weight_bytes;
    size_t scale_count;
    size_t scale_bytes;
    size_t i;

    if (out_matrix == NULL)
        return BNN_EINVAL;

    *out_matrix = NULL;

    if (device == NULL || desc == NULL)
        return BNN_EINVAL;

    if (desc->in_features == 0 || desc->out_features == 0 || desc->group_size == 0)
        return BNN_EINVAL;

    if (desc->in_features > SOFT_MAX_FEATURES || desc->out_features > SOFT_MAX_FEATURES ||
        desc->group_size != SOFT_GROUP_SIZE)
        return BNN_ENOTSUP;

    if (desc->in_features % SOFT_GROUP_SIZE != 0)
        return BNN_EINVAL;

    if (!SizeMultiply(desc->out_features, desc->in_features / 8u, &weight_bytes) ||
        !SizeMultiply(desc->out_features, desc->in_features / SOFT_GROUP_SIZE, &scale_count) ||
        !SizeMultiply(scale_count, sizeof(uint16_t), &scale_bytes))
        return BNN_ERANGE;

    if (desc->weight_bits == NULL || desc->scale_fp16 == NULL || desc->weight_bytes != weight_bytes ||
        desc->scale_count != scale_count)
        return BNN_EINVAL;

    /* 上传之前检查所有 scale，拒绝 NaN 和 Inf。 */
    for (i = 0; i < scale_count; ++i) {
        if ((desc->scale_fp16[i] & 0x7c00u) == 0x7c00u)
            return BNN_EINVAL;
    }

    if (device->matrix_count == SIZE_MAX)
        return BNN_ERANGE;

    matrix = calloc(1, sizeof *matrix);

    if (matrix == NULL)
        return BNN_ENOMEM;

    matrix->weight_bits = malloc(weight_bytes);
    matrix->scale_fp16 = malloc(scale_bytes);

    if (matrix->weight_bits == NULL || matrix->scale_fp16 == NULL) {
        free(matrix->weight_bits);
        free(matrix->scale_fp16);
        free(matrix);
        return BNN_ENOMEM;
    }

    memcpy(matrix->weight_bits, desc->weight_bits, weight_bytes);
    memcpy(matrix->scale_fp16, desc->scale_fp16, scale_bytes);

    matrix->owner = device;
    matrix->in_features = desc->in_features;
    matrix->out_features = desc->out_features;
    matrix->groups = desc->in_features / SOFT_GROUP_SIZE;

    ++device->matrix_count;
    *out_matrix = matrix;

    return BNN_OK;
}

void BNN_MatrixDestroy(BNN_Matrix matrix) {
    if (matrix == NULL)
        return;

    --matrix->owner->matrix_count;

    free(matrix->weight_bits);
    free(matrix->scale_fp16);
    free(matrix);
}

BNN_Status BNN_BufferAlloc(BNN_Device device, size_t bytes, BNN_Buffer* out_buffer) {
    BNN_Buffer buffer;

    if (out_buffer == NULL)
        return BNN_EINVAL;

    *out_buffer = NULL;

    if (device == NULL || bytes == 0)
        return BNN_EINVAL;

    if (device->buffer_count == SIZE_MAX)
        return BNN_ERANGE;

    buffer = calloc(1, sizeof *buffer);

    if (buffer == NULL)
        return BNN_ENOMEM;

    buffer->data = malloc(bytes);

    if (buffer->data == NULL) {
        free(buffer);
        return BNN_ENOMEM;
    }

    buffer->owner = device;
    buffer->bytes = bytes;

    ++device->buffer_count;
    *out_buffer = buffer;

    return BNN_OK;
}

void BNN_BufferFree(BNN_Buffer buffer) {
    if (buffer == NULL)
        return;

    --buffer->owner->buffer_count;

    free(buffer->data);
    free(buffer);
}

BNN_Status BNN_BufferWrite(BNN_Buffer buffer, size_t offset, const void* src, size_t bytes) {
    if (buffer == NULL || (bytes != 0 && src == NULL))
        return BNN_EINVAL;

    if (!RangeFits(buffer->bytes, offset, bytes))
        return BNN_ERANGE;

    if (bytes != 0)
        memcpy(buffer->data + offset, src, bytes);

    return BNN_OK;
}

BNN_Status BNN_BufferRead(BNN_Buffer buffer, size_t offset, void* dst, size_t bytes) {
    if (buffer == NULL || (bytes != 0 && dst == NULL))
        return BNN_EINVAL;

    if (!RangeFits(buffer->bytes, offset, bytes))
        return BNN_ERANGE;

    if (bytes != 0)
        memcpy(dst, buffer->data + offset, bytes);

    return BNN_OK;
}

/*
 * 计算一行输入。
 * 参数和全部输入已由 Execute 检查，此处不再产生可恢复错误。
 *
 * 输出使用 memcpy，避免非对齐 INT32/FP32 指针访问。
 */
static void ComputeRow(BNN_Matrix matrix, const int8_t* x, unsigned char* y, BNN_Operation operation) {
    int32_t totals[SOFT_MAX_GROUPS];
    size_t weight_row_bytes = matrix->in_features / 8u;
    uint32_t n, g, k;

    /* 每组激活总和只算一次，供所有输出通道共享。 */
    for (g = 0; g < matrix->groups; ++g) {
        uint32_t begin = g * SOFT_GROUP_SIZE;
        int32_t total = 0;

        for (k = begin; k < begin + SOFT_GROUP_SIZE; ++k)
            total += (int32_t)x[k];

        totals[g] = total;
    }

    for (n = 0; n < matrix->out_features; ++n) {
        const uint8_t* w = matrix->weight_bits + (size_t)n * weight_row_bytes;
        float acc = 0.0f;

        for (g = 0; g < matrix->groups; ++g) {
            uint32_t begin = g * SOFT_GROUP_SIZE;
            size_t index = (size_t)n * matrix->groups + g;
            int32_t pos = 0;
            int32_t dot;

            for (k = begin; k < begin + SOFT_GROUP_SIZE; ++k) {
                if ((w[k >> 3] >> (k & 7u)) & 1u)
                    pos += (int32_t)x[k];
            }

            /*
             * 正权重位置之和减去负权重位置之和：
             * dot = pos - (total - pos)。
             *
             * G=128、输入范围 [-127,127] 时，
             * 最终结果范围为 [-16256,16256]。
             */
            dot = 2 * pos - totals[g];

            if (operation == BNN_OP_DOT_GROUPS) {
                memcpy(y + index * sizeof dot, &dot, sizeof dot);
            } else {
                float scale = HalfToFloat(matrix->scale_fp16[index]);
                float term = FloatMultiply(scale, (float)dot);

                acc = FloatAdd(acc, term);
            }
        }

        if (operation != BNN_OP_DOT_GROUPS)
            memcpy(y + (size_t)n * sizeof acc, &acc, sizeof acc);
    }
}

BNN_Status BNN_Execute(BNN_Device device, const BNN_Job* job) {
    BNN_Matrix matrix;
    size_t input_row_bytes;
    size_t output_row_bytes;
    size_t output_elements;
    size_t input_stride;
    size_t output_stride;
    uint32_t m, k;

    if (device == NULL || job == NULL || job->matrix == NULL || job->input == NULL || job->output == NULL ||
        job->rows == 0)
        return BNN_EINVAL;

    switch (job->operation) {
    case BNN_OP_GEMV:
        if (job->rows != 1)
            return BNN_EINVAL;
        break;

    case BNN_OP_GEMM:
    case BNN_OP_DOT_GROUPS:
        break;

    default:
        return BNN_ENOTSUP;
    }

    matrix = job->matrix;

    if (matrix->owner != device || job->input->owner != device || job->output->owner != device ||
        job->input == job->output)
        return BNN_EINVAL;

    input_row_bytes = matrix->in_features;

    if (job->operation == BNN_OP_DOT_GROUPS) {
        if (!SizeMultiply(matrix->out_features, matrix->groups, &output_elements) ||
            !SizeMultiply(output_elements, sizeof(int32_t), &output_row_bytes))
            return BNN_ERANGE;
    } else {
        if (!SizeMultiply(matrix->out_features, sizeof(float), &output_row_bytes))
            return BNN_ERANGE;
    }

    input_stride = job->input_row_stride != 0 ? job->input_row_stride : input_row_bytes;

    output_stride = job->output_row_stride != 0 ? job->output_row_stride : output_row_bytes;

    if (input_stride < input_row_bytes || output_stride < output_row_bytes)
        return BNN_EINVAL;

    if (!RowsFit(job->input->bytes, job->input_offset, input_stride, input_row_bytes, job->rows) ||
        !RowsFit(job->output->bytes, job->output_offset, output_stride, output_row_bytes, job->rows))
        return BNN_ERANGE;

    if (job->operation != BNN_OP_DOT_GROUPS && fegetround() != FE_TONEAREST)
        return BNN_ENOTSUP;

    /*
     * 先检查所有行，再开始写输出。
     * 即使最后一行包含非法输入，也不会留下部分计算结果。
     */
    for (m = 0; m < job->rows; ++m) {
        const int8_t* x = (const int8_t*)(job->input->data + job->input_offset + (size_t)m * input_stride);

        for (k = 0; k < matrix->in_features; ++k) {
            if (x[k] == INT8_MIN)
                return BNN_ERANGE;
        }
    }

    for (m = 0; m < job->rows; ++m) {
        const int8_t* x = (const int8_t*)(job->input->data + job->input_offset + (size_t)m * input_stride);

        unsigned char* y = job->output->data + job->output_offset + (size_t)m * output_stride;

        ComputeRow(matrix, x, y, job->operation);
    }

    return BNN_OK;
}

const char* BNN_StatusString(BNN_Status status) {
    switch (status) {
    case BNN_OK:
        return "success";
    case BNN_EINVAL:
        return "invalid argument";
    case BNN_ENOMEM:
        return "out of memory";
    case BNN_ENOTSUP:
        return "unsupported operation or configuration";
    case BNN_ERANGE:
        return "buffer or input range error";
    case BNN_EBUSY:
        return "device still owns resources";
    case BNN_EIO:
        return "device I/O error";
    case BNN_ETIMEOUT:
        return "device timeout";
    default:
        return "unknown status";
    }
}
