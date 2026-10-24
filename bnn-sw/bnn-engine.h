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
typedef enum {
    BNNE_MODEL_MLP = 1, BNNE_MODEL_CNN = 2, BNNE_MODEL_GPT = 3,
    BNNE_MODEL_LLAMA = 4, BNNE_MODEL_QWEN3 = 5
} BNNE_ModelKind;

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
/* 仅 GPT/Llama/Qwen3 支持，返回 NUL 结尾的借用 UTF-8 字符串，直到 Close 有效，不得 free。 */
BNNE_Status BNNE_GetToken(BNNE_Engine engine, uint32_t id, const char** out_text);
/* 返回模型的 EOS token ID；模型未声明 EOS 时返回 BNNE_ENOTSUP。 */
BNNE_Status BNNE_GetEosToken(BNNE_Engine engine, uint32_t* out_id);
/*
 * 将 UTF-8 文本编码为 token IDs。仅带文本分词器的模型（当前为 GGUF/Qwen3）支持，
 * 其他模型返回 BNNE_ENOTSUP。
 * parse_special != 0 时识别 <|im_start|> 等控制/用户自定义 token 字面量。
 * capacity 不足时返回 BNNE_ERANGE，部分写入的 ID 不作保证。
 */
BNNE_Status BNNE_Encode(BNNE_Engine engine, const char* text, int parse_special,
                        int32_t* ids, uint32_t capacity, uint32_t* out_count);
/* 判断 token 是否应结束生成（EOS/EOG）。其他模型或无声明时返回 0。 */
int BNNE_IsEogToken(BNNE_Engine engine, uint32_t id);
/* 重置或清空当前引擎的 KV Cache 状态。清空后，下次调用解码时从 pos=0 开始。 */
BNNE_Status BNNE_ResetKVCache(BNNE_Engine engine);
/* 步骤式单 token 解码（KV Cache 模式）：
 * token: 当前单步输入 token ID (0 <= token < vocabulary_size)。
 * pos: 当前 token 在序列中的位置 (0 <= pos < max_sequence_length)。若 pos==0 自动重置缓存。
 * out_logits: 接收当前 token 产生的 FP32 logits [vocabulary_size]。
 */
BNNE_Status BNNE_StepToken(BNNE_Engine engine, int32_t token, uint32_t pos, float* out_logits);
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
