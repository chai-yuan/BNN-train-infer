#ifndef BNN_ENGINE_H
#define BNN_ENGINE_H

#include "../bnn-accelerator/bnn-accelerator.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BNNE_API_VERSION 3u
#define BNNE_MAX_RANK 4u

/*
 * 句柄：
 *   BNNE_Model   只读。GGUF 元数据、已上传到设备的权重、分词器。
 *   BNNE_Context 可变。设备工作区、KV Cache、统计。
 *
 * 生命周期（必须按此顺序）：
 *   BNN_DeviceOpen → BNNE_ModelLoad → BNNE_ContextCreate
 *   → ... →
 *   BNNE_ContextFree → BNNE_ModelFree → BNN_DeviceClose
 *
 * 并发：
 *   Model 加载完成后只读，可以被多个 Context 共享。
 *   凡是会访问同一个 BNN_Device 的调用（包括直接调用加速器 API），
 *   都必须由调用者串行化。
 *
 * 创建函数失败时 *out = NULL，并回收已分配的资源。
 * 所有 Free 函数都接受 NULL。
 */
typedef struct bnne_model* BNNE_Model;
typedef struct bnne_context* BNNE_Context;

typedef enum {
    BNNE_OK = 0,
    BNNE_EINVAL = -1,
    BNNE_ENOMEM = -2,
    BNNE_ENOTSUP = -3, /* 未注册的架构 / 任务类型不匹配 / 设备能力不足 */
    BNNE_ERANGE = -4,  /* 容量不足或超出上下文长度 */
    BNNE_EIO = -5,
    BNNE_EFORMAT = -6, /* GGUF 结构或必需字段错误 */
    BNNE_ECOMPUTE = -7 /* 加速器执行失败 */
} BNNE_Status;

typedef enum {
    BNNE_TASK_CLASSIFY = 1, /* FP32 张量 → FP32 logits[n_classes] */
    BNNE_TASK_LLM = 2       /* int32 tokens → FP32 logits[n_vocab]，带 KV Cache */
} BNNE_Task;

typedef struct {
    BNNE_Task task;
    const char* arch; /* 即 general.architecture，借用，在 ModelFree 前有效 */
    uint64_t load_ns; /* 模型加载耗时（含权重上传） */

    /* TASK_CLASSIFY；其他任务时为 0 */
    uint32_t input_rank;
    uint32_t input_dims[BNNE_MAX_RANK]; /* 例如 CNN 为 [1,28,28]，MLP 为 [in_features] */
    uint32_t input_size;                /* 单个样本的元素数 = dims 的乘积 */
    uint32_t n_classes;

    /* TASK_LLM；其他任务时为 0 / -1 */
    uint32_t n_vocab;
    uint32_t n_ctx_train;
    int32_t bos_token; /* 未声明时为 -1 */
    int32_t eos_token; /* 未声明时为 -1 */
} BNNE_ModelInfo;

typedef struct {
    uint32_t n_ctx;   /* LLM：KV Cache 容量（token 数），0 表示 n_ctx_train；分类任务忽略 */
    uint32_t n_batch; /* 单次调用的最大 token 数或样本数；0 表示默认值 */
    int enable_stats; /* 默认 1 */
} BNNE_ContextParams;

/*
 * 单调墙钟纳秒，只累计成功的 Classify/Decode。
 * quantize_ns 仅在主机侧量化时非零；soft 在设备内量化，因此恒为 0。
 * total_ns = quantize_ns + transfer_ns + execute_ns + host_ns。
 * logical_macs 按有效尺寸计算，executed_macs 按设备实际执行的尺寸计算。
 */
typedef struct {
    uint64_t calls, items; /* items：分类样本数 / LLM token 数 */
    uint64_t total_ns, quantize_ns, transfer_ns, execute_ns, host_ns;
    uint64_t device_calls, bytes_written, bytes_read;
    uint64_t logical_macs, executed_macs;
} BNNE_Stats;

/* ---------------- Model ---------------- */

/*
 * device 由调用者打开，引擎借用，Model 存续期间不得关闭。
 * 按 general.architecture 分发到已注册的实现；未注册时返回 BNNE_ENOTSUP。
 */
BNNE_Status BNNE_ModelLoad(BNN_Device device, const char* gguf_path, BNNE_Model* out_model);
void BNNE_ModelFree(BNNE_Model model);
BNNE_Status BNNE_ModelGetInfo(BNNE_Model model, BNNE_ModelInfo* out_info);

/* ---------------- Tokenizer（仅 TASK_LLM，否则返回 BNNE_ENOTSUP） ---------------- */

/*
 * parse_special != 0 时识别 <|im_start|> 等特殊 token 的字面量。
 * 无论成功与否，*out_count 都写入所需的 token 数；
 * 容量不足时返回 BNNE_ERANGE，ids 的内容不作保证（可以按 out_count 重新分配后再调用）。
 */
BNNE_Status BNNE_Tokenize(BNNE_Model model, const char* text, int parse_special, int32_t* ids, uint32_t capacity,
                          uint32_t* out_count);

/*
 * 返回 token 解码后的原始字节（已处理字节级 BPE 等映射），借用到 ModelFree 为止。
 * 单个 piece 可能是不完整的 UTF-8 序列，调用者应先拼接再输出。
 */
BNNE_Status BNNE_TokenPiece(BNNE_Model model, int32_t id, const char** out_bytes, size_t* out_len);

/* 是否为结束生成的 token（EOS/EOT/EOG）。非 LLM 或 id 越界时返回 0。 */
int BNNE_TokenIsEog(BNNE_Model model, int32_t id);

/* ---------------- Context ---------------- */

void BNNE_ContextParamsInit(BNNE_ContextParams* params);
/* params 可以为 NULL，表示使用默认值。 */
BNNE_Status BNNE_ContextCreate(BNNE_Model model, const BNNE_ContextParams* params, BNNE_Context* out_ctx);
void BNNE_ContextFree(BNNE_Context ctx);

/*
 * TASK_CLASSIFY：
 *   input  [n_samples][input_size]，已归一化的 FP32，不会被修改。
 *   logits [n_samples][n_classes]，调用者分配内存。
 *   1 <= n_samples <= n_batch。参数检查失败时不修改 logits。
 */
BNNE_Status BNNE_Classify(BNNE_Context ctx, const float* input, uint32_t n_samples, float* logits);

/*
 * TASK_LLM：把 tokens 追加到当前位置（BNNE_ContextPos）之后。
 *   1 <= n_tokens <= n_batch，且 pos + n_tokens <= n_ctx，否则返回 BNNE_ERANGE。
 *   logits 接收最后一个 token 的结果 [n_vocab]；传 NULL 表示只做 prefill，不输出。
 *   失败时 KV Cache 回到调用前的位置。
 */
BNNE_Status BNNE_Decode(BNNE_Context ctx, const int32_t* tokens, uint32_t n_tokens, float* logits);

/* 当前已缓存的 token 数，也就是下一个 token 的位置。 */
uint32_t BNNE_ContextPos(BNNE_Context ctx);
/* 丢弃位置 >= pos 的缓存；pos 为 0 表示清空。pos 大于当前位置时返回 BNNE_EINVAL。 */
BNNE_Status BNNE_ContextTruncate(BNNE_Context ctx, uint32_t pos);

/* ---------------- 统计与错误 ---------------- */

/* enable_stats == 0 时所有字段都为 0。 */
BNNE_Status BNNE_ContextGetStats(BNNE_Context ctx, BNNE_Stats* out_stats);
BNNE_Status BNNE_ContextResetStats(BNNE_Context ctx);

const char* BNNE_StatusString(BNNE_Status status);
/*
 * 线程局部：返回本线程上一次失败调用的诊断信息（包括 ModelLoad 失败时）。
 * 借用字符串，在本线程下一次 BNNE 调用前有效，不得 free。
 */
const char* BNNE_LastError(void);

#ifdef __cplusplus
}
#endif
#endif
