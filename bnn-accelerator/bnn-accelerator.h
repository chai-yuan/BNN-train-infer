#ifndef BNN_ACCELERATOR_H
#define BNN_ACCELERATOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* v2 修改了任务、能力结构和 DOT_GROUPS 输出类型，调用者与后端必须一起重新编译。 */
#define BNN_API_VERSION 2u

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

/*
 * GEMV/GEMM：I8 激活 -> 分组整数点积 -> 应用权重缩放的 F32 结果。
 * DOT_GROUPS：I8 激活 -> 精确的分组 I16 点积，不应用权重缩放。
 * *_F32：F32 激活 -> 按本版约定量化到 I8 -> 缩放后的点积 -> F32 输出。
 * F32 表示接口的输入输出类型，本版量化规则见任务描述，不提供量化模式选择。
 */
typedef enum {
    BNN_OP_GEMV = 1,
    BNN_OP_GEMM = 2,
    BNN_OP_DOT_GROUPS = 3,
    BNN_OP_GEMV_F32 = 4,
    BNN_OP_GEMM_F32 = 5
} BNN_Operation;

typedef enum { BNN_TYPE_I8 = 1, BNN_TYPE_F32 = 2, BNN_TYPE_I16 = 3 } BNN_Type;

/* 运算能力位，与 BNN_Operation 的枚举值不同。 */
typedef enum {
    BNN_CAP_GEMV = 1u << 0,
    BNN_CAP_GEMM = 1u << 1,
    BNN_CAP_DOT_GROUPS = 1u << 2,
    BNN_CAP_GEMV_F32 = 1u << 3,
    BNN_CAP_GEMM_F32 = 1u << 4
} BNN_OperationCaps;

/*
 * 完整后端向调用者提供的逻辑能力，不是 RTL 单次分块的物理限制。
 * 内部分块、数据传输和软件后处理均属于后端实现细节。
 * 各上限均包含边界值，实际可分配规模还受可用内存约束。
 * group_size 表示本版后端支持的权重缩放组大小，必须是 [8, 256] 内的 8 的倍数；
 * 矩阵必须使用这个组大小。
 * supported_ops 是 BNN_CAP_* 运算能力位的组合，包含 F32 运算时也必须支持其约定的量化规则。
 */
typedef struct {
    uint32_t api_version;
    uint32_t max_in_features;
    uint32_t max_out_features;
    uint32_t max_rows;
    uint32_t group_size;
    uint32_t supported_ops;
} BNN_Caps;

/*
 * 只读 W1 权重矩阵，逻辑形状为 [out_features][in_features]。
 *
 * weight_bits 按行优先排列，每行 in_features / 8 字节。
 * 第 o 行、第 k 个权重位于：
 *     weight_bits[o * (in_features / 8) + (k >> 3)] 的第 (k & 7) 位。
 * bit = 1 表示 +1，bit = 0 表示 -1。
 *
 * scale_fp16 按行优先排列，形状为 [out_features][in_features / group_size]。
 * 每个 uint16_t 保存 IEEE-754 binary16 的位模式，采用主机 uint16_t 数值表示，
 * 不是模型文件中的原始字节流。接受所有有限值，包括负数、带符号零和次正规数；
 * 拒绝 NaN 和 Inf。
 *
 * 各尺寸必须非零。group_size 必须是 [8, 256] 内的 8 的倍数，
 * in_features 必须能被 group_size 整除。
 * weight_bytes 和 scale_count 必须与尺寸严格匹配。
 * 两个数组都必须提供，包括只用于 DOT_GROUPS 的矩阵。
 *
 * MatrixCreate 成功返回后，数据已复制或上传到后端，调用者可以修改或释放原数组。
 * 此公共布局独立于 GGUF 的块编码和各后端的原生存储布局。
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
 * 对无类型字节缓冲区的借用、无状态视图。
 * 复制或修改视图不会改变缓冲区或其他视图，无需分配、设置、查询或销毁视图对象。
 *
 * offset 和 row_stride 的单位均为字节。
 * row_stride 为 0 表示紧密排列，否则必须不小于一行逻辑数据的字节数。
 * 行内元素连续，后端负责适配非对齐偏移和步长。
 * 形状由运算和矩阵确定，type 必须与运算要求匹配。
 * 所有逻辑行都必须落在底层缓冲区范围内，大小计算不得溢出。
 */
typedef struct {
    BNN_Buffer buffer;
    BNN_Type type;
    size_t offset;
    size_t row_stride;
} BNN_BufferView;

/*
 * 任务描述。
 * M = rows，K = 矩阵的 in_features，N = 矩阵的 out_features，
 * G = 矩阵的 group_size，groups = K / G。所有布局均为行优先。
 *
 * 运算           输入             输出
 * GEMV           [1][K] I8        [1][N] F32
 * GEMM           [M][K] I8        [M][N] F32
 * DOT_GROUPS     [M][K] I8        [M][N][groups] I16
 * GEMV_F32       [1][K] F32       [1][N] F32
 * GEMM_F32       [M][K] F32       [M][N] F32
 *
 * rows 必须非零，两种 GEMV 运算都要求 rows == 1。
 * I8 逻辑输入必须事先初始化，范围为 [-127, 127]，不接受 -128。
 * 每个权重缩放组的整数点积为：
 *     D[m][n][g] = sum_k Q[m][k] * W[n][k]，W 为 +1 或 -1。
 * 其中 k 只遍历第 g 组。G <= 256 时，|D| <= 32512，可由 I16 精确表示。
 * 内部计算可以使用更宽的整数，但不得对分组点积进行回绕或饱和处理。
 *     Z[m][n] = sum_g F32(scale[n][g]) * D[m][n][g]。
 * I8 GEMV/GEMM 返回 Z，不应用激活缩放因子；Q 就是调用者提供的 I8 输入。
 *
 * F32 运算固定按每个完整逻辑输入行进行 I8 对称量化：
 *   a = max_k abs(X[m][k])；输入含 NaN/Inf 时返回 BNN_ERANGE。
 *   a == 0 时 s = 1，否则 s = F32(a / 127)；若除法结果舍入为零，
 *   则将 s 设为最小的正 binary32 次正规数。
 *   Q = clamp(round(F32(X / s)), -127, 127)，
 *       其中 round 表示舍入到最近整数，半整数向远离零的方向舍入。
 *   Y[m][n] = F32(Z[m][n] * s)。
 * 后端内部沿 K 分块时，必须复用完整行的量化结果，不能对每块独立量化。
 *
 * 矩阵和两个缓冲区必须属于执行任务的设备。
 * 输入、输出必须使用不同缓冲区句柄，即使视图范围不重叠也不例外。
 * 视图不拥有缓冲区，所有句柄都必须保持有效，直到 Execute 返回。
 */
typedef struct {
    BNN_Operation operation;
    BNN_Matrix matrix;
    BNN_BufferView input;
    BNN_BufferView output;
    uint32_t rows;
} BNN_Job;

/*
 * 资源及并发约定：
 * - 创建失败时，将有效的输出句柄参数设为 NULL；不要传入仍需保留的现有句柄变量地址。
 * - 同一设备上的所有调用必须由调用者串行化，不同设备相互独立。
 *   数据传输与任务执行均为同步操作。
 * - 必须先销毁矩阵和缓冲区，再关闭设备。Destroy/Free 接受 NULL。
 *   销毁后句柄失效，接口不保证检测失效句柄。视图不会延长资源生命周期。
 */
BNN_Status BNN_DeviceOpen(uint32_t device_index, BNN_Device* out_device);
/* 仍存在矩阵或缓冲区时返回 BNN_EBUSY，设备保持打开。 */
BNN_Status BNN_DeviceClose(BNN_Device device);
BNN_Status BNN_DeviceGetCaps(BNN_Device device, BNN_Caps* out_caps);

BNN_Status BNN_MatrixCreate(BNN_Device device, const BNN_MatrixDesc* desc, BNN_Matrix* out_matrix);
void BNN_MatrixDestroy(BNN_Matrix matrix);

/* bytes 必须大于 0，新缓冲区内容未初始化，缓冲区本身没有元素类型。 */
BNN_Status BNN_BufferAlloc(BNN_Device device, size_t bytes, BNN_Buffer* out_buffer);
void BNN_BufferFree(BNN_Buffer buffer);
/*
 * 同步传输原始字节，并检查缓冲区边界。
 * bytes 为 0 时 src/dst 可以为 NULL，但 offset 仍必须位于 [0, 缓冲区大小] 内。
 * 缓冲区中的有类型数据采用主机数值表示；后端在按视图解释数据时适配设备字节序，
 * 原始字节传输本身不改变字节内容。
 */
BNN_Status BNN_BufferWrite(BNN_Buffer buffer, size_t offset, const void* src, size_t bytes);
BNN_Status BNN_BufferRead(BNN_Buffer buffer, size_t offset, void* dst, size_t bytes);

/*
 * 同步执行，返回 BNN_OK 时，完整结果可以立即读取。
 * DOT_GROUPS 的整数结果必须精确一致。
 * 所有 FP32 运算（包括 I8 GEMV/GEMM）采用最近舍入，恰好居中时取偶数，保留次正规数；
 * FP16 缩放因子精确转换为 FP32，每组乘积单独舍入为 FP32，
 * 按组索引递增顺序进行 FP32 累加，不使用融合乘加 FMA。
 * 量化到整数时的半整数舍入规则另见任务描述，后端分块不能改变这些数值规则。
 *
 * 使用 FP32 的软件要求浮点环境为 FE_TONEAREST，禁用 fast-math 和 FTZ/DAZ。
 * 未知或不支持的运算返回 BNN_ENOTSUP，视图类型不匹配返回 BNN_EINVAL。
 * 参数检查、输入检查或内存分配失败时，输出保持不变。
 * 返回 BNN_EIO/BNN_ETIMEOUT 时，可能已写入部分输出，设备后续是否可用由后端定义。
 * 成功时仅写入逻辑输出元素，不修改行间填充字节。
 */
BNN_Status BNN_Execute(BNN_Device device, const BNN_Job* job);

const char* BNN_StatusString(BNN_Status status);

#ifdef __cplusplus
}
#endif

#endif
