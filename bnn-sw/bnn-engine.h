#ifndef BNN_ENGINE_H
#define BNN_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BNNE_API_VERSION 2u
#define BNNE_MAX_RANK 4u

typedef struct bnne_engine* BNNE_Engine;

typedef enum {
    BNNE_OK = 0, BNNE_EINVAL = -1, BNNE_ENOMEM = -2, BNNE_ENOTSUP = -3,
    BNNE_ERANGE = -4, BNNE_EIO = -5, BNNE_EFORMAT = -6, BNNE_ECOMPUTE = -7
} BNNE_Status;
typedef enum { BNNE_F32 = 1, BNNE_I32 = 2 } BNNE_DType;
typedef enum { BNNE_MODEL_MLP = 1, BNNE_MODEL_CNN = 2, BNNE_MODEL_GPT = 3 } BNNE_ModelKind;

typedef struct {
    BNNE_DType dtype;
    uint32_t rank;
    uint32_t dims[BNNE_MAX_RANK];
} BNNE_TensorDesc;

/* 连续布局；data 按元素类型对齐；bytes 为可用容量。输入内容不会被修改。 */
typedef struct {
    BNNE_TensorDesc desc;
    void* data;
    size_t bytes;
} BNNE_Tensor;

typedef struct {
    uint32_t device_index;
    int enable_profile;
} BNNE_Options;

typedef struct {
    uint32_t api_version;
    BNNE_ModelKind kind;
    uint32_t layer_count;
    /* GPT 的 input=[max_sequence_length]，output=[max_sequence_length,vocabulary_size]。
     * 实际序列形状由 GetOutputDesc 查询。CNN input=[1,28,28]，MLP input=[in_features]。
     */
    BNNE_TensorDesc input;
    BNNE_TensorDesc output;
    uint32_t vocabulary_size;
    uint32_t max_sequence_length;
} BNNE_ModelInfo;

/* 单调墙钟纳秒，仅累计成功的 Run，不含应用预处理/校验。
 * total_ns = quantize_ns + transfer_ns + execute_ns + host_ns。
 * transfer 包括 Run 内动态矩阵的创建与销毁；load 单列模型准备成本。
 * MAC 仅计卸载的矩阵运算，不含主机 FP32 attn@V 等；executed_macs 按设备尺寸，
 * logical_macs 按未补齐的有效尺寸计算。
 */
typedef struct {
    uint64_t load_ns, forward_calls, total_ns;
    uint64_t quantize_ns, transfer_ns, execute_ns, host_ns;
    uint64_t device_calls, bytes_written, bytes_read;
    uint64_t logical_macs, executed_macs;
} BNNE_Stats;

/* 默认开启统计，所有模型仅通过构建时链接的 accelerator 后端执行。 */
void BNNE_OptionsInit(BNNE_Options* options);
/* options 可为 NULL。支持 BNN1/BNN2/GPT1。失败时 *out_engine=NULL 并回收资源。
 * 句柄独占设备、权重和工作区；同一句柄上的调用必须串行化。
 * 输出句柄参数不能覆盖仍需保留的现有句柄。
 */
BNNE_Status BNNE_Open(const char* model_path, const BNNE_Options* options, BNNE_Engine* out_engine);
BNNE_Status BNNE_GetModelInfo(BNNE_Engine engine, BNNE_ModelInfo* out_info);
BNNE_Status BNNE_GetOutputDesc(BNNE_Engine engine, const BNNE_TensorDesc* input,
                              BNNE_TensorDesc* out_output);
/* 调用者持有内存，output.desc 必须匹配 GetOutputDesc；同步返回 FP32 logits。
 * MLP/CNN 接收归一化 FP32，GPT 接收 int32 token IDs（0<=id<vocabulary_size）。
 * 参数检查失败不修改输出；计算失败时输出内容不作保证。
 */
BNNE_Status BNNE_Run(BNNE_Engine engine, const BNNE_Tensor* input, BNNE_Tensor* output);
/* 仅 GPT 支持，返回 NUL 结尾的借用 UTF-8 字符串，直到 Close 有效，不得 free。 */
BNNE_Status BNNE_GetToken(BNNE_Engine engine, uint32_t id, const char** out_text);
BNNE_Status BNNE_GetStats(BNNE_Engine engine, BNNE_Stats* out_stats);
/* 清除推理累计，保留 load_ns；关闭 profile 时所有统计字段均为零。 */
BNNE_Status BNNE_ResetStats(BNNE_Engine engine);
void BNNE_Close(BNNE_Engine engine); /* 接受 NULL。 */
const char* BNNE_StatusString(BNNE_Status status);
/* 借用字符串，在下次可能设置诊断的引擎调用或 Close 前有效；不得 free。
 * Open 失败没有句柄，使用 StatusString 获取错误类别。
 */
const char* BNNE_LastError(BNNE_Engine engine);

#ifdef __cplusplus
}
#endif
#endif
