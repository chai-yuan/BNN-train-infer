#ifndef BNNE_INTERNAL_H
#define BNNE_INTERNAL_H

#include "bnn-engine.h"

typedef struct bnne_gguf bnne_gguf;           /* GGUF 读取器 */
typedef struct bnne_tokenizer bnne_tokenizer; /* 由 tokenizer.ggml.* 字段构建，与架构无关 */

/* ---------- 模型注册契约 ---------- */
typedef struct bnne_arch {
    const char* name; /* 与 general.architecture 精确匹配 */
    BNNE_Task task;

    /* 解析超参数，在 m->device 上创建 BNN_Matrix，并填写 m->info 中与任务相关的字段。
     * 失败时自行回收已创建的资源。 */
    BNNE_Status (*load)(struct bnne_model* m, const bnne_gguf* g, void** out_impl);
    void (*unload)(void* impl);

    /* 创建每个 Context 私有的状态（KV Cache、中间激活等）；可以为 NULL。 */
    BNNE_Status (*ctx_init)(struct bnne_context* c, void** out_state);
    void (*ctx_free)(void* state);

    /* 二选一，与 task 对应。核心层已完成参数和范围检查。 */
    BNNE_Status (*classify)(struct bnne_context* c, const float* in, uint32_t n, float* logits);
    BNNE_Status (*decode)(struct bnne_context* c, const int32_t* tok, uint32_t n, float* logits);
    /* LLM 必须提供。截断后由核心层更新 c->pos。 */
    BNNE_Status (*truncate)(struct bnne_context* c, uint32_t pos);
} bnne_arch;

struct bnne_model {
    BNN_Device device; /* 借用 */
    BNN_Caps caps;
    const bnne_arch* arch;
    void* impl;
    bnne_gguf* gguf;
    bnne_tokenizer* tok; /* 仅 LLM */
    BNNE_ModelInfo info;
    uint32_t max_in, max_out; /* bnne_linear 自动记录，用于确定工作区大小 */
};

struct bnne_context {
    struct bnne_model* model;
    BNNE_ContextParams params; /* 已补全默认值 */
    void* state;
    BNN_Buffer in_buf;  /* n_batch * max_in 字节 */
    BNN_Buffer out_buf; /* n_batch * max_out * sizeof(float) */
    int8_t* qbuf;       /* 主机端量化暂存（单行） */
    float* chunk;       /* 主机端分块输出暂存（单行） */
    uint32_t max_in, max_out;
    uint32_t pos;
    BNNE_Stats stats;
};

/* ---------- 注册表 ---------- */
const bnne_arch* bnne_arch_find(const char* name);

/* ---------- 错误 ---------- */
BNNE_Status bnne_error(BNNE_Status st, const char* fmt, ...); /* 写入线程局部错误信息 */
BNNE_Status bnne_accel_error(BNN_Status st, const char* operation);
void bnne_error_clear(void);

/* ---------- GGUF 读取 ---------- */
BNNE_Status bnne_gguf_open(const char* path, bnne_gguf** out);
void bnne_gguf_close(bnne_gguf* g);

/* ---------- 共享算子 ---------- */
typedef struct {
    uint32_t in_off, in_len, out_off, out_len; /* 在原逻辑权重中的切片 */
    BNN_Matrix matrix;
} bnne_linear_chunk;

typedef struct {
    uint32_t in, out;
    bnne_linear_chunk* chunks;
    uint32_t n_chunks;
} bnne_linear;

/* 读取 Q1_0 张量，重排后调用 BNN_MatrixCreate，并更新 m->max_in / max_out。 */
BNNE_Status bnne_linear_load(struct bnne_model* m, const bnne_gguf* g, const char* tensor,
                             bnne_linear* out);
/* 从已重排的 Q1_0 bits/scales 创建（用于 tied embedding）。 */
BNNE_Status bnne_linear_init(struct bnne_model* m, uint32_t in, uint32_t out_features,
                             const uint8_t* bits, const uint16_t* scales, bnne_linear* linear);
void bnne_linear_free(bnne_linear* l);
/* x[rows][in] (FP32) → y[rows][out] (FP32)：量化、上传、执行 GEMV/GEMM、读回，同时更新统计。 */
BNNE_Status bnne_linear_forward(struct bnne_context* c, const bnne_linear* l, const float* x,
                                uint32_t rows, float* y);

/* 读取 Q1_0 张量并重排为 accelerator 的 W1 布局。 */
BNNE_Status bnne_q1_0_read(const bnne_gguf* g, const char* tensor, uint32_t in, uint32_t out,
                           uint8_t** out_bits, uint16_t** out_scales);
/* 读取 F32/F16 张量（norm、embedding 等）为 FP32 数组，归 Model 所有。 */
BNNE_Status bnne_tensor_f32(struct bnne_model* m, const bnne_gguf* g, const char* tensor,
                            const float** out, size_t* out_n);

/* 主机端 FP16→FP32 与逐行 absmax int8 量化。 */
float bnne_fp16_to_fp32(uint16_t half);
BNNE_Status bnne_quantize(const float* input, uint32_t count, int8_t* output, float* scale);

uint64_t bnne_now_ns(void);

#endif
