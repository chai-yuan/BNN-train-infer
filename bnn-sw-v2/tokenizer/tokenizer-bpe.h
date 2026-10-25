#ifndef BNNE_TOKENIZER_BPE_H
#define BNNE_TOKENIZER_BPE_H

#include <stddef.h>
#include <stdint.h>

#include "../bnn-engine.h"
#include "../gguf.h"

typedef struct bnne_tokenizer bnne_tokenizer;

/* 映射 GGUF 中的分词器表；GGUF 的 mmap 必须比分词器存活更久。 */
BNNE_Status bnne_tokenizer_create(const bnne_gguf* g, bnne_tokenizer** out);
void bnne_tokenizer_destroy(bnne_tokenizer* tokenizer);

/*
 * Qwen2 byte-level BPE 编码。
 * parse_special != 0 时识别 <|im_start|> 等控制/用户自定义 token 字面量。
 * 容量不足返回 BNNE_ERANGE，*out_count 写入所需 token 数。
 */
BNNE_Status bnne_tokenizer_encode(const bnne_tokenizer* tokenizer, const char* text,
                                  int parse_special, int32_t* ids, uint32_t capacity,
                                  uint32_t* out_count);

/* 返回已经处理过 byte-level 映射的原始字节，借用到 destroy 为止。 */
BNNE_Status bnne_tokenizer_piece(const bnne_tokenizer* tokenizer, int32_t id,
                                 const char** out_bytes, size_t* out_len);

/* 判断 token 是否应结束生成（EOS/EOT/EOG）。 */
int bnne_tokenizer_is_eog(const bnne_tokenizer* tokenizer, int32_t id);

#endif
