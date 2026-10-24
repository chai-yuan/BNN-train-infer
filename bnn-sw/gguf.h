#ifndef BNNE_GGUF_H
#define BNNE_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "bnn-engine.h"

#define BNNE_GGUF_MAX_DIMS 4u

/* 字符串指向 mmap 区域，不保证 NUL 结尾。 */
typedef struct {
    const uint8_t* data;
    uint32_t length;
} BnneGgufString;

typedef struct {
    BnneGgufString name;
    uint32_t rank;
    uint64_t dims[BNNE_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
    uint64_t bytes;
} BnneGgufTensor;

/*
 * 只解析 qwen3 推理需要的字段；其余 KV 与张量仍保留索引。
 * 映射在 bnne_gguf_close 前有效。
 */
typedef struct {
    uint8_t* mapping;
    size_t mapping_bytes;
    uint64_t data_offset;
    uint32_t alignment;

    uint64_t tensor_count;
    BnneGgufTensor* tensors;

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

    BnneGgufString* tokens;
    uint64_t token_count;
    uint32_t* token_types;
    uint64_t token_type_count;
    BnneGgufString* merges;
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
} BnneGguf;

BNNE_Status bnne_gguf_open(FILE* file, BnneGguf** out, char* error, size_t error_size);
void bnne_gguf_close(BnneGguf* gguf);

const BnneGgufTensor* bnne_gguf_find(const BnneGguf* gguf, const char* name);
const uint8_t* bnne_gguf_data(const BnneGguf* gguf, const BnneGgufTensor* tensor);

#endif
