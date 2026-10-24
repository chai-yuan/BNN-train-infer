#ifndef BNNE_TOKENIZER_BPE_H
#define BNNE_TOKENIZER_BPE_H

#include <stddef.h>
#include <stdint.h>

#include "bnn-engine.h"
#include "gguf.h"

typedef struct QwenTokenizer QwenTokenizer;

/* Maps GGUF tokenizer tables; the BnneGguf mapping must outlive the tokenizer. */
BNNE_Status bnne_qwen_tokenizer_create(const BnneGguf* gguf, QwenTokenizer** out,
                                       char* error, size_t error_size);
void bnne_qwen_tokenizer_destroy(QwenTokenizer* tokenizer);

/*
 * Qwen2 byte-level BPE encoding.
 * parse_special != 0 recognizes control/user-defined token literals such as
 * "<|im_start|>" and emits their IDs directly.
 * Returns BNNE_ERANGE when the encoded sequence does not fit in capacity.
 */
BNNE_Status bnne_qwen_encode(const QwenTokenizer* tokenizer, const char* text,
                             int parse_special, int32_t* ids, uint32_t capacity,
                             uint32_t* out_count);

#endif
