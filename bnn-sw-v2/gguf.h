#ifndef BNNE_GGUF_H
#define BNNE_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "bnn-engine.h"

#define BNNE_GGUF_MAX_DIMS 4u

typedef struct bnne_gguf bnne_gguf;

/* 字符串指向 mmap 区域，不保证 NUL 结尾。 */
typedef struct {
    const uint8_t* data;
    uint32_t length;
} bnne_gguf_string;

typedef struct {
    bnne_gguf_string name;
    uint32_t rank;
    uint64_t dims[BNNE_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
    uint64_t bytes;
} bnne_gguf_tensor;

/*
 * v2 只接受 GGUF。这里保留常用元数据与分词器表的只读视图；
 * 架构相关的约束由对应 bnne_arch 校验。
 */
struct bnne_gguf {
    uint8_t* mapping;
    size_t mapping_bytes;
    uint64_t data_offset;
    uint32_t alignment;

    uint64_t tensor_count;
    bnne_gguf_tensor* tensors;

    char architecture[32];
    uint64_t block_count;
    uint64_t embedding_length;
    uint64_t feed_forward_length;
    uint64_t head_count;
    uint64_t head_count_kv;
    uint64_t key_length;
    uint64_t value_length;
    uint64_t context_length;
    float rms_epsilon;
    float rope_freq_base;
    float rope_factor;
    uint64_t rope_original_context;
    char rope_scaling_type[16];

    /* bnn-mlp 分类模型 */
    uint64_t mlp_block_count;
    uint64_t mlp_class_count;
    uint64_t mlp_input_features;
    uint64_t mlp_input_pixels;
    float mlp_input_mean;
    float mlp_input_std;
    char (*mlp_activations)[16]; /* 按 block 索引，缺失项为空串 */
    uint64_t mlp_activation_count;
    bnne_gguf_string* mlp_labels;
    uint64_t mlp_label_count;

    bnne_gguf_string* tokens;
    uint64_t token_count;
    uint32_t* token_types;
    uint64_t token_type_count;
    bnne_gguf_string* merges;
    uint64_t merge_count;
    char tokenizer_model[32];
    char tokenizer_pre[32];
    int has_eos;
    int has_bos;
    int has_padding;
    uint32_t eos_token_id;
    uint32_t bos_token_id;
    uint32_t padding_token_id;
    int add_bos_token;
};

/* 打开并映射 GGUF，失败时通过 BNNE_LastError 提供诊断。 */
BNNE_Status bnne_gguf_open(const char* path, bnne_gguf** out);
void bnne_gguf_close(bnne_gguf* g);

const bnne_gguf_tensor* bnne_gguf_find(const bnne_gguf* g, const char* name);
const uint8_t* bnne_gguf_data(const bnne_gguf* g, const bnne_gguf_tensor* tensor);

#endif
