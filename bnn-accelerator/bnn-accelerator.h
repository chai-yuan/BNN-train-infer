#ifndef BNN_ACCELERATOR_H
#define BNN_ACCELERATOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BNN_API_VERSION 1u

/*
 * 不透明指针句柄。
 * 调用者不能访问内部结构，也不能自行创建或释放内部对象。
 *
 * 使用示例：
 *     BNN_Device device = NULL;
 *     BNN_Buffer buffer = NULL;
 */
typedef struct bnn_accel_device* BNN_Device;
typedef struct bnn_accel_matrix* BNN_Matrix;
typedef struct bnn_accel_buffer* BNN_Buffer;

typedef enum {
    BNN_OK = 0,
    BNN_EINVAL = -1,
    BNN_ENOMEM = -2,
    BNN_ENOTSUP = -3,
    BNN_ERANGE = -4,
    BNN_EBUSY = -5,
    BNN_EIO = -6,
    BNN_ETIMEOUT = -7
} BNN_Status;

typedef enum { BNN_OP_GEMV = 1, BNN_OP_GEMM = 2, BNN_OP_DOT_GROUPS = 3 } BNN_Operation;

/*
 * 设备能力。
 *
 * max_rows 是接口支持的行数上限。
 * 实际可执行规模还受缓冲区容量和可用内存限制。
 *
 * 本版使用固定组大小，由 group_size 返回。
 */
typedef struct {
    uint32_t api_version;
    uint32_t max_in_features;
    uint32_t max_out_features;
    uint32_t max_rows;
    uint32_t group_size;
    uint32_t supports_gemv;
    uint32_t supports_gemm;
    uint32_t supports_dot_groups;
} BNN_Caps;

/*
 * 只读 W1 权重矩阵描述。
 *
 * 逻辑权重形状：
 *     [out_features][in_features]
 *
 * weight_bits：
 *     行优先排列，每行 in_features / 8 字节。
 *     第 o 行、第 k 个权重位于：
 *
 *         weight_bits[o * (in_features / 8) + (k >> 3)]
 *
 *     的第 (k & 7) 位。
 *     bit = 1 表示 +1；bit = 0 表示 -1。
 *
 * scale_fp16：
 *     行优先排列，形状为：
 *         [out_features][in_features / group_size]
 *
 *     每个 uint16_t 保存 IEEE-754 binary16 的位模式。
 *     数组元素是主机 uint16_t 数值，不是模型文件中的原始字节流。
 *     接受有限值，包括零、负数和次正规数；拒绝 NaN 和 Inf。
 *
 * weight_bytes 和 scale_count 必须与尺寸严格匹配。
 * 两个数组均必须提供，包括只用于 DOT_GROUPS 的矩阵。
 *
 * MatrixCreate 成功返回后，数据已复制或上传到后端。
 * 调用者可以释放原数组，或者修改它们而不影响矩阵对象。
 *
 * 本版要求 in_features 是设备 group_size 的整数倍。
 */
typedef struct {
    uint32_t in_features;
    uint32_t out_features;
    uint32_t group_size;

    const uint8_t* weight_bits;
    size_t weight_bytes;

    const uint16_t* scale_fp16;
    size_t scale_count;
} BNN_MatrixDesc;

/*
 * 任务描述。
 *
 * 记：
 *     M = rows
 *     K = 矩阵的 in_features
 *     N = 矩阵的 out_features
 *     G = 矩阵的 group_size
 *     groups = K / G
 *
 * 输入：
 *     [M][K]，int8_t，元素范围限制为 [-127, 127]。
 *
 * 分组整数点积：
 *     D[m][n][g] =
 *         sum_k X[m][k] * W[n][k]
 *
 *     k 只遍历第 g 组，W[n][k] 为 +1 或 -1。
 *
 * GEMV / GEMM：
 *     Z[m][n] =
 *         sum_g FP32(scale[n][g]) * D[m][n][g]
 *
 * 输出：
 *     GEMV：       [N]，float，rows 必须为 1。
 *     GEMM：       [M][N]，float。
 *     DOT_GROUPS： [M][N][groups]，int32_t。
 *
 * 所有布局均为行优先，逻辑行内元素连续。
 *
 * offset 和 row_stride 的单位都是字节。
 * row_stride 为 0 表示紧密排列。
 * 非零 row_stride 必须不小于一行逻辑数据的字节数。
 * DOT_GROUPS 的一行包含 N * groups 个 int32_t。
 *
 * 公共接口不要求偏移或步长对齐，后端负责适配。
 * 矩阵和两个缓冲区必须属于执行任务的设备。
 * input 和 output 必须是不同缓冲区句柄。
 */
typedef struct {
    BNN_Operation operation;
    BNN_Matrix matrix;
    BNN_Buffer input;
    BNN_Buffer output;

    uint32_t rows;

    size_t input_offset;
    size_t output_offset;
    size_t input_row_stride;
    size_t output_row_stride;
} BNN_Job;

/*
 * 资源及并发规则：
 *
 * 创建函数失败时，会将有效的输出句柄参数设置为 NULL。
 * 输出句柄参数不能指向一个仍需保留的现有句柄变量。
 *
 * 同一设备上的调用必须由调用者串行化。
 * 本版所有传输和计算均为同步操作。
 *
 * 先销毁矩阵和缓冲区，再关闭设备。
 * 销毁后不得继续使用句柄；建议调用者将变量设为 NULL。
 * MatrixDestroy 和 BufferFree 接受 NULL。
 */

/* 软件后端仅支持 device_index = 0。 */
BNN_Status BNN_DeviceOpen(uint32_t device_index, BNN_Device* out_device);

/* 仍存在矩阵或缓冲区时返回 BNN_EBUSY，不关闭设备。 */
BNN_Status BNN_DeviceClose(BNN_Device device);

BNN_Status BNN_DeviceGetCaps(BNN_Device device, BNN_Caps* out_caps);

/* 创建只读权重矩阵，内部复制或上传权重与 scale。 */
BNN_Status BNN_MatrixCreate(BNN_Device device, const BNN_MatrixDesc* desc, BNN_Matrix* out_matrix);

void BNN_MatrixDestroy(BNN_Matrix matrix);

/* bytes 必须大于 0；新缓冲区内容未初始化。 */
BNN_Status BNN_BufferAlloc(BNN_Device device, size_t bytes, BNN_Buffer* out_buffer);

void BNN_BufferFree(BNN_Buffer buffer);

/*
 * 同步主机与设备数据传输。
 * bytes 为 0 时 src/dst 可以为 NULL，但 offset 仍需在范围内。
 *
 * INT32/FP32 输出读取到主机后采用主机数值表示；
 * 后端负责处理设备与主机之间可能存在的字节序差异。
 */
BNN_Status BNN_BufferWrite(BNN_Buffer buffer, size_t offset, const void* src, size_t bytes);

BNN_Status BNN_BufferRead(BNN_Buffer buffer, size_t offset, void* dst, size_t bytes);

/*
 * 同步执行，成功返回时结果可以立即读取。
 *
 * DOT_GROUPS 的整数结果必须精确一致。
 *
 * GEMV/GEMM 的数值契约：
 *     FP16 scale 精确转换为 FP32；
 *     每组乘积单独舍入为 FP32；
 *     按组索引递增顺序进行 FP32 累加；
 *     不使用融合乘加 FMA。
 *
 * 软件后端要求浮点环境使用最接近舍入，禁止 fast-math，
 * 并保留次正规数，不启用 FTZ/DAZ 等清零模式。
 *
 * 参数检查、容量检查或输入值检查失败时，输出保持不变。
 * 返回 BNN_EIO / BNN_ETIMEOUT 时，输出内容不作保证。
 * 成功时仅写入逻辑输出元素，不修改行间 padding。
 *
 * 输入逻辑元素必须事先初始化。
 */
BNN_Status BNN_Execute(BNN_Device device, const BNN_Job* job);

const char* BNN_StatusString(BNN_Status status);

#ifdef __cplusplus
}
#endif

#endif
