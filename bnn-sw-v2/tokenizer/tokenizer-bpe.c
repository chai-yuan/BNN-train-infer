#include "tokenizer-bpe.h"

#include "../bnne-internal.h"
#include "unicode-data.h"

#include <stdlib.h>
#include <string.h>

enum { CP_LETTER = 1u, CP_NUMBER = 2u, CP_SPACE = 4u };
#define ARRAY_COUNT(array) (sizeof(array) / sizeof((array)[0]))

typedef struct {
    const uint8_t* text;
    uint32_t length;
    uint32_t id;
    uint32_t used;
} TokenEntry;

typedef struct {
    const uint8_t* left;
    uint32_t left_length;
    const uint8_t* right;
    uint32_t right_length;
    uint32_t rank;
    uint32_t used;
} MergeEntry;

typedef struct {
    bnne_gguf_string text;
    uint32_t id;
} SpecialToken;

struct bnne_tokenizer {
    const bnne_gguf* gguf;
    TokenEntry* tokens;
    uint32_t token_capacity;
    MergeEntry* merges;
    uint32_t merge_capacity;
    SpecialToken* specials;
    uint32_t special_count;
    uint32_t byte_to_cp[256];
    uint32_t byte_token_id[256];

    char* piece_text;
    uint32_t* piece_off;
    uint32_t* piece_len;
    uint32_t eog_ids[8];
    uint32_t eog_count;
};

typedef struct {
    int32_t* ids;
    uint32_t capacity;
    uint32_t count;
    int overflow;
} Output;

static uint32_t next_pow2(uint64_t value) {
    uint32_t result = 1;
    while (result < value && result < (1u << 31))
        result <<= 1;
    return result;
}

static uint64_t hash_bytes(const uint8_t* data, size_t length) {
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < length; i++) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t hash_pair(const uint8_t* left, size_t left_length,
                          const uint8_t* right, size_t right_length) {
    uint64_t a = hash_bytes(left, left_length);
    uint64_t b = hash_bytes(right, right_length);
    return a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
}

static uint32_t token_slot(const bnne_tokenizer* tokenizer, const uint8_t* text,
                           uint32_t length) {
    uint32_t mask = tokenizer->token_capacity - 1;
    uint32_t index = (uint32_t)hash_bytes(text, length) & mask;
    while (tokenizer->tokens[index].used) {
        if (tokenizer->tokens[index].length == length &&
            memcmp(tokenizer->tokens[index].text, text, length) == 0)
            return index;
        index = (index + 1) & mask;
    }
    return index;
}

static void token_insert(bnne_tokenizer* tokenizer, const uint8_t* text,
                         uint32_t length, uint32_t id) {
    uint32_t index = token_slot(tokenizer, text, length);
    if (tokenizer->tokens[index].used) {
        tokenizer->tokens[index].id = id; /* last duplicate wins */
        return;
    }
    tokenizer->tokens[index].text = text;
    tokenizer->tokens[index].length = length;
    tokenizer->tokens[index].id = id;
    tokenizer->tokens[index].used = 1;
}

static int token_lookup(const bnne_tokenizer* tokenizer, const uint8_t* text,
                        uint32_t length, uint32_t* out_id) {
    uint32_t index = token_slot(tokenizer, text, length);
    if (!tokenizer->tokens[index].used)
        return 0;
    *out_id = tokenizer->tokens[index].id;
    return 1;
}

static uint32_t merge_slot(const bnne_tokenizer* tokenizer,
                           const uint8_t* left, uint32_t left_length,
                           const uint8_t* right, uint32_t right_length) {
    uint32_t mask = tokenizer->merge_capacity - 1;
    uint32_t index = (uint32_t)hash_pair(left, left_length, right, right_length) & mask;
    while (tokenizer->merges[index].used) {
        if (tokenizer->merges[index].left_length == left_length &&
            tokenizer->merges[index].right_length == right_length &&
            memcmp(tokenizer->merges[index].left, left, left_length) == 0 &&
            memcmp(tokenizer->merges[index].right, right, right_length) == 0)
            return index;
        index = (index + 1) & mask;
    }
    return index;
}

static void merge_insert(bnne_tokenizer* tokenizer,
                         const uint8_t* left, uint32_t left_length,
                         const uint8_t* right, uint32_t right_length, uint32_t rank) {
    uint32_t index = merge_slot(tokenizer, left, left_length, right, right_length);
    if (tokenizer->merges[index].used)
        return; /* first duplicate wins */
    tokenizer->merges[index].left = left;
    tokenizer->merges[index].left_length = left_length;
    tokenizer->merges[index].right = right;
    tokenizer->merges[index].right_length = right_length;
    tokenizer->merges[index].rank = rank;
    tokenizer->merges[index].used = 1;
}

static int merge_lookup(const bnne_tokenizer* tokenizer,
                        const uint8_t* left, uint32_t left_length,
                        const uint8_t* right, uint32_t right_length, uint32_t* out_rank) {
    uint32_t index = merge_slot(tokenizer, left, left_length, right, right_length);
    if (!tokenizer->merges[index].used)
        return 0;
    *out_rank = tokenizer->merges[index].rank;
    return 1;
}

static int range_contains(const BnneUnicodeRange* ranges, size_t count, uint32_t cp) {
    size_t low = 0, high = count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (cp < ranges[middle].first)
            high = middle;
        else if (cp > ranges[middle].last)
            low = middle + 1;
        else
            return 1;
    }
    return 0;
}

static int whitespace_contains(uint32_t cp) {
    for (size_t i = 0; i < ARRAY_COUNT(bnne_unicode_whitespace); i++)
        if (bnne_unicode_whitespace[i] == cp)
            return 1;
    return 0;
}

static uint32_t codepoint_flags(uint32_t cp) {
    uint32_t flags = 0;
    if (range_contains(bnne_unicode_letters, ARRAY_COUNT(bnne_unicode_letters), cp))
        flags |= CP_LETTER;
    if (range_contains(bnne_unicode_numbers, ARRAY_COUNT(bnne_unicode_numbers), cp))
        flags |= CP_NUMBER;
    if (whitespace_contains(cp))
        flags |= CP_SPACE;
    return flags;
}

static size_t utf8_encode(uint32_t cp, uint8_t out[4]) {
    if (cp < 0x80) {
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (uint8_t)(0xc0 | (cp >> 6));
        out[1] = (uint8_t)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (uint8_t)(0xe0 | (cp >> 12));
        out[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (uint8_t)(0x80 | (cp & 0x3f));
        return 3;
    }
    out[0] = (uint8_t)(0xf0 | (cp >> 18));
    out[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3f));
    out[3] = (uint8_t)(0x80 | (cp & 0x3f));
    return 4;
}

static size_t utf8_decode_one(const uint8_t* text, size_t length, uint32_t* out_cp) {
    if (!length)
        return 0;
    uint8_t b0 = text[0];
    if (b0 < 0x80) {
        *out_cp = b0;
        return 1;
    }
    if (b0 >= 0xc2 && b0 <= 0xdf && length >= 2 && (text[1] & 0xc0) == 0x80) {
        *out_cp = ((uint32_t)(b0 & 0x1f) << 6) | (text[1] & 0x3f);
        return 2;
    }
    if (b0 >= 0xe0 && b0 <= 0xef && length >= 3 &&
        (text[1] & 0xc0) == 0x80 && (text[2] & 0xc0) == 0x80) {
        uint32_t cp = ((uint32_t)(b0 & 0x0f) << 12) |
                      ((uint32_t)(text[1] & 0x3f) << 6) | (text[2] & 0x3f);
        if (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))
            return 0;
        *out_cp = cp;
        return 3;
    }
    if (b0 >= 0xf0 && b0 <= 0xf4 && length >= 4 &&
        (text[1] & 0xc0) == 0x80 && (text[2] & 0xc0) == 0x80 &&
        (text[3] & 0xc0) == 0x80) {
        uint32_t cp = ((uint32_t)(b0 & 0x07) << 18) |
                      ((uint32_t)(text[1] & 0x3f) << 12) |
                      ((uint32_t)(text[2] & 0x3f) << 6) | (text[3] & 0x3f);
        if (cp < 0x10000 || cp > 0x10ffff)
            return 0;
        *out_cp = cp;
        return 4;
    }
    return 0;
}

static int decode_text(const uint8_t* text, size_t length, uint32_t** out_cpts,
                       size_t** out_offsets, size_t* out_count) {
    if (!length) {
        *out_cpts = NULL;
        *out_offsets = NULL;
        *out_count = 0;
        return 1;
    }
    if (length > SIZE_MAX - 1 ||
        length > SIZE_MAX / sizeof(uint32_t) ||
        length + 1 > SIZE_MAX / sizeof(size_t))
        return 0;
    uint32_t* cpts = malloc(length * sizeof *cpts);
    size_t* offsets = malloc((length + 1) * sizeof *offsets);
    if (!cpts || !offsets) {
        free(cpts);
        free(offsets);
        return 0;
    }
    size_t count = 0, pos = 0;
    while (pos < length) {
        uint32_t cp = 0;
        size_t used = utf8_decode_one(text + pos, length - pos, &cp);
        if (!used) {
            free(cpts);
            free(offsets);
            return 0;
        }
        cpts[count] = cp;
        offsets[count] = pos;
        count++;
        pos += used;
    }
    offsets[count] = length;
    *out_cpts = cpts;
    *out_offsets = offsets;
    *out_count = count;
    return 1;
}

static uint32_t ascii_lower(uint32_t cp) {
    return cp >= 'A' && cp <= 'Z' ? cp + ('a' - 'A') : cp;
}

static int cp_to_byte(const bnne_tokenizer* tokenizer, uint32_t cp) {
    for (int byte = 0; byte < 256; byte++)
        if (tokenizer->byte_to_cp[byte] == cp)
            return byte;
    return -1;
}

static size_t detokenize(const bnne_tokenizer* tokenizer, const uint8_t* source,
                         size_t length, uint8_t* dst) {
    size_t written = 0, pos = 0;
    while (pos < length) {
        uint32_t cp = 0;
        size_t used = utf8_decode_one(source + pos, length - pos, &cp);
        if (used) {
            int byte = cp_to_byte(tokenizer, cp);
            if (byte >= 0) {
                dst[written++] = (uint8_t)byte;
                pos += used;
                continue;
            }
        }
        if (!used)
            used = 1;
        memcpy(dst + written, source + pos, used);
        written += used;
        pos += used;
    }
    return written;
}

static void output_push(Output* out, uint32_t id) {
    if (out->count < out->capacity)
        out->ids[out->count++] = (int32_t)id;
    else
        out->overflow = 1;
}

static int encode_chunk(const bnne_tokenizer* tokenizer, const uint8_t* text,
                        size_t length, Output* out) {
    if (!length)
        return 1;
    if (length > SIZE_MAX / 4u || length > SIZE_MAX / sizeof(size_t) ||
        length > SIZE_MAX / sizeof(uint32_t))
        return 0;
    size_t encoded_capacity = length * 4u;
    uint8_t* encoded = malloc(encoded_capacity);
    size_t* offsets = malloc(length * sizeof *offsets);
    uint32_t* lengths = malloc(length * sizeof *lengths);
    int ok = 0;
    if (!encoded || !offsets || !lengths)
        goto cleanup;

    size_t encoded_length = 0;
    for (size_t i = 0; i < length; i++) {
        uint8_t buffer[4];
        size_t size = utf8_encode(tokenizer->byte_to_cp[text[i]], buffer);
        memcpy(encoded + encoded_length, buffer, size);
        offsets[i] = encoded_length;
        lengths[i] = (uint32_t)size;
        encoded_length += size;
    }

    size_t count = length;
    for (;;) {
        size_t best = SIZE_MAX;
        uint32_t best_rank = UINT32_MAX;
        for (size_t i = 0; i + 1 < count; i++) {
            uint32_t rank = 0;
            if (merge_lookup(tokenizer, encoded + offsets[i], lengths[i],
                             encoded + offsets[i + 1], lengths[i + 1], &rank) &&
                rank < best_rank) {
                best_rank = rank;
                best = i;
            }
        }
        if (best == SIZE_MAX)
            break;
        lengths[best] += lengths[best + 1];
        if (best + 2 < count) {
            memmove(offsets + best + 1, offsets + best + 2,
                    (count - best - 2) * sizeof *offsets);
            memmove(lengths + best + 1, lengths + best + 2,
                    (count - best - 2) * sizeof *lengths);
        }
        count--;
    }

    for (size_t i = 0; i < count; i++) {
        uint32_t id = 0;
        if (token_lookup(tokenizer, encoded + offsets[i], lengths[i], &id)) {
            output_push(out, id);
        } else {
            size_t pos = 0;
            while (pos < lengths[i]) {
                uint32_t cp = 0;
                size_t used = utf8_decode_one(encoded + offsets[i] + pos,
                                              lengths[i] - pos, &cp);
                if (!used)
                    goto cleanup;
                int byte = cp_to_byte(tokenizer, cp);
                if (byte < 0 || tokenizer->byte_token_id[byte] == UINT32_MAX)
                    goto cleanup;
                output_push(out, tokenizer->byte_token_id[byte]);
                pos += used;
            }
        }
        if (out->overflow)
            break;
    }
    ok = !out->overflow;
cleanup:
    free(encoded);
    free(offsets);
    free(lengths);
    return ok;
}

static int add_chunk(const bnne_tokenizer* tokenizer, const uint8_t* text,
                     const size_t* offsets, size_t* previous, size_t end, Output* out) {
    if (end > *previous &&
        !encode_chunk(tokenizer, text + offsets[*previous],
                      offsets[end] - offsets[*previous], out))
        return 0;
    *previous = end;
    return !out->overflow;
}

/* Port of llama.cpp's unicode_regex_split_custom_qwen2. */
static int encode_segment(const bnne_tokenizer* tokenizer, const uint8_t* text,
                          size_t length, Output* out) {
    uint32_t* cpts = NULL;
    size_t* offsets = NULL;
    uint32_t* flags = NULL;
    size_t count = 0;
    int ok = 0;
    if (!length)
        return 1;
    if (!decode_text(text, length, &cpts, &offsets, &count))
        goto cleanup;
    flags = malloc(count * sizeof *flags);
    if (!flags)
        goto cleanup;
    for (size_t i = 0; i < count; i++)
        flags[i] = codepoint_flags(cpts[i]);

    size_t previous = 0;
    size_t pos = 0;
    while (pos < count) {
        uint32_t cp = cpts[pos];
        uint32_t flag = flags[pos];

        /* (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]) */
        if (cp == '\'' && pos + 1 < count) {
            uint32_t next = ascii_lower(cpts[pos + 1]);
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                pos += 2;
                if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                    goto cleanup;
                continue;
            }
            if (pos + 2 < count) {
                uint32_t after = ascii_lower(cpts[pos + 2]);
                if ((next == 'r' && after == 'e') ||
                    (next == 'v' && after == 'e') ||
                    (next == 'l' && after == 'l')) {
                    pos += 3;
                    if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                        goto cleanup;
                    continue;
                }
            }
        }

        /* [^\r\n\p{L}\p{N}]?\p{L}+ */
        if (!(cp == '\r' || cp == '\n' || (flag & CP_NUMBER))) {
            if ((flag & CP_LETTER) || (pos + 1 < count && (flags[pos + 1] & CP_LETTER))) {
                pos++;
                while (pos < count && (flags[pos] & CP_LETTER))
                    pos++;
                if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                    goto cleanup;
                continue;
            }
        }

        /* \p{N} */
        if (flag & CP_NUMBER) {
            pos++;
            if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                goto cleanup;
            continue;
        }

        /* <space>?[^\s\p{L}\p{N}]+[\r\n]* */
        {
            size_t end = pos;
            size_t probe = cp == ' ' ? pos + 1 : pos;
            uint32_t probe_flag = probe < count ? flags[probe] : 0;
            int probe_valid = probe < count;
            if (!(probe_flag & (CP_SPACE | CP_LETTER | CP_NUMBER))) {
                if (cp == ' ')
                    end++;
                while (probe_valid && !(probe_flag & (CP_SPACE | CP_LETTER | CP_NUMBER))) {
                    end++;
                    probe = end;
                    probe_flag = probe < count ? flags[probe] : 0;
                    probe_valid = probe < count;
                }
                uint32_t next = end < count ? cpts[end] : UINT32_MAX;
                while (next == '\r' || next == '\n') {
                    end++;
                    next = end < count ? cpts[end] : UINT32_MAX;
                }
                pos = end;
                if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                    goto cleanup;
                continue;
            }
        }

        /* \s*[\r\n]+ / \s+(?!\S) / \s+ */
        {
            size_t whitespace = 0;
            size_t last_newline = 0;
            while (pos + whitespace < count && (flags[pos + whitespace] & CP_SPACE)) {
                uint32_t c = cpts[pos + whitespace];
                if (c == '\r' || c == '\n')
                    last_newline = pos + whitespace + 1;
                whitespace++;
            }
            if (last_newline > 0) {
                pos = last_newline;
                if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                    goto cleanup;
                continue;
            }
            if (whitespace > 1 && pos + whitespace < count) {
                pos += whitespace - 1;
                if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                    goto cleanup;
                continue;
            }
            if (whitespace > 0) {
                pos += whitespace;
                if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
                    goto cleanup;
                continue;
            }
        }

        pos++;
        if (!add_chunk(tokenizer, text, offsets, &previous, pos, out))
            goto cleanup;
    }
    ok = 1;
cleanup:
    free(cpts);
    free(offsets);
    free(flags);
    return ok && !out->overflow;
}

static int encode_text(const bnne_tokenizer* tokenizer, const char* text, size_t length,
                       int parse_special, Output* out) {
    if (!parse_special)
        return encode_segment(tokenizer, (const uint8_t*)text, length, out);

    size_t segment = 0, pos = 0;
    while (pos < length) {
        const SpecialToken* best = NULL;
        for (uint32_t i = 0; i < tokenizer->special_count; i++) {
            const SpecialToken* special = &tokenizer->specials[i];
            if (special->text.length <= length - pos &&
                memcmp(text + pos, special->text.data, special->text.length) == 0 &&
                (!best || special->text.length > best->text.length))
                best = special;
        }
        if (best) {
            if (pos > segment &&
                !encode_segment(tokenizer, (const uint8_t*)text + segment,
                                pos - segment, out))
                return 0;
            output_push(out, best->id);
            if (out->overflow)
                return 0;
            pos += best->text.length;
            segment = pos;
        } else {
            pos++;
        }
    }
    if (length > segment &&
        !encode_segment(tokenizer, (const uint8_t*)text + segment,
                        length - segment, out))
        return 0;
    return !out->overflow;
}

static void add_eog(bnne_tokenizer* tokenizer, uint32_t id) {
    if (tokenizer->eog_count >= ARRAY_COUNT(tokenizer->eog_ids))
        return;
    for (uint32_t i = 0; i < tokenizer->eog_count; i++)
        if (tokenizer->eog_ids[i] == id)
            return;
    tokenizer->eog_ids[tokenizer->eog_count++] = id;
}

static BNNE_Status build_pieces(bnne_tokenizer* tokenizer) {
    const bnne_gguf* g = tokenizer->gguf;
    if (!g->tokens || g->token_count > UINT32_MAX)
        return bnne_error(BNNE_EFORMAT, "tokenizer vocabulary does not match model");

    uint64_t total = 0;
    for (uint64_t i = 0; i < g->token_count; i++) {
        if (total > SIZE_MAX - g->tokens[i].length - 1)
            return bnne_error(BNNE_ERANGE, "tokenizer vocabulary is too large");
        total += g->tokens[i].length + 1;
    }
    tokenizer->piece_text = malloc((size_t)total);
    tokenizer->piece_off = malloc((size_t)g->token_count * sizeof *tokenizer->piece_off);
    tokenizer->piece_len = malloc((size_t)g->token_count * sizeof *tokenizer->piece_len);
    if (!tokenizer->piece_text || !tokenizer->piece_off || !tokenizer->piece_len)
        return bnne_error(BNNE_ENOMEM, "out of memory");

    size_t used = 0;
    for (uint64_t i = 0; i < g->token_count; i++) {
        const bnne_gguf_string* token = &g->tokens[i];
        uint32_t type = g->token_types && i < g->token_type_count ?
                        g->token_types[i] : 1;
        if (used > UINT32_MAX)
            return bnne_error(BNNE_ERANGE, "tokenizer vocabulary is too large");
        uint32_t offset = (uint32_t)used;
        if (type == 3 || type == 4) {
            memcpy(tokenizer->piece_text + used, token->data, token->length);
            used += token->length;
        } else {
            used += detokenize(tokenizer, token->data, token->length,
                               (uint8_t*)tokenizer->piece_text + used);
        }
        tokenizer->piece_text[used++] = '\0';
        tokenizer->piece_off[i] = offset;
        tokenizer->piece_len[i] = (uint32_t)(used - 1 - offset);
    }

    if (g->has_eos && g->eos_token_id < g->token_count)
        add_eog(tokenizer, g->eos_token_id);
    for (uint64_t i = 0; i < g->token_count; i++) {
        const bnne_gguf_string* token = &g->tokens[i];
        if ((token->length == 13 && memcmp(token->data, "<|endoftext|>", 13) == 0) ||
            (token->length == 10 && memcmp(token->data, "<|im_end|>", 10) == 0))
            add_eog(tokenizer, (uint32_t)i);
    }
    return BNNE_OK;
}

BNNE_Status bnne_tokenizer_create(const bnne_gguf* g, bnne_tokenizer** out) {
    if (!g || !out)
        return BNNE_EINVAL;
    *out = NULL;
    if (!g->tokens || !g->token_count)
        return bnne_error(BNNE_EFORMAT, "tokenizer has no vocabulary");
    if (g->token_count > UINT32_MAX)
        return bnne_error(BNNE_ERANGE, "tokenizer vocabulary is too large");

    bnne_tokenizer* tokenizer = calloc(1, sizeof *tokenizer);
    if (!tokenizer)
        return bnne_error(BNNE_ENOMEM, "out of memory");
    tokenizer->gguf = g;

    tokenizer->token_capacity = next_pow2((uint64_t)g->token_count * 2);
    tokenizer->tokens = calloc(tokenizer->token_capacity, sizeof *tokenizer->tokens);
    if (!tokenizer->tokens) {
        bnne_tokenizer_destroy(tokenizer);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }
    for (uint64_t i = 0; i < g->token_count; i++)
        token_insert(tokenizer, g->tokens[i].data, g->tokens[i].length, (uint32_t)i);

    tokenizer->merge_capacity = next_pow2((uint64_t)g->merge_count * 2);
    tokenizer->merges = calloc(tokenizer->merge_capacity, sizeof *tokenizer->merges);
    if (!tokenizer->merges) {
        bnne_tokenizer_destroy(tokenizer);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }
    for (uint64_t i = 0; i < g->merge_count; i++) {
        const bnne_gguf_string* merge = &g->merges[i];
        if (!merge->length)
            continue;
        const uint8_t* separator = memchr(merge->data + 1, ' ', merge->length - 1);
        if (!separator)
            continue;
        uint32_t left_length = (uint32_t)(separator - merge->data);
        const uint8_t* right = separator + 1;
        uint32_t right_length = merge->length - left_length - 1;
        if (!left_length || !right_length)
            continue;
        merge_insert(tokenizer, merge->data, left_length, right, right_length, (uint32_t)i);
    }

    for (uint64_t i = 0; i < g->token_count; i++) {
        uint32_t type = g->token_types && i < g->token_type_count ?
                        g->token_types[i] : 1;
        if (type == 3 || type == 4)
            tokenizer->special_count++;
    }
    if (tokenizer->special_count) {
        tokenizer->specials = calloc(tokenizer->special_count, sizeof *tokenizer->specials);
        if (!tokenizer->specials) {
            bnne_tokenizer_destroy(tokenizer);
            return bnne_error(BNNE_ENOMEM, "out of memory");
        }
        uint32_t index = 0;
        for (uint64_t i = 0; i < g->token_count; i++) {
            uint32_t type = g->token_types && i < g->token_type_count ?
                            g->token_types[i] : 1;
            if (type == 3 || type == 4) {
                tokenizer->specials[index].text = g->tokens[i];
                tokenizer->specials[index].id = (uint32_t)i;
                index++;
            }
        }
    }

    int used[256] = {0};
    for (int byte = 33; byte <= 126; byte++) {
        tokenizer->byte_to_cp[byte] = (uint32_t)byte;
        used[byte] = 1;
    }
    for (int byte = 161; byte <= 172; byte++) {
        tokenizer->byte_to_cp[byte] = (uint32_t)byte;
        used[byte] = 1;
    }
    for (int byte = 174; byte <= 255; byte++) {
        tokenizer->byte_to_cp[byte] = (uint32_t)byte;
        used[byte] = 1;
    }
    uint32_t extra = 0;
    for (int byte = 0; byte < 256; byte++)
        if (!used[byte])
            tokenizer->byte_to_cp[byte] = 256 + extra++;

    for (int byte = 0; byte < 256; byte++) {
        uint8_t encoded[4];
        size_t length = utf8_encode(tokenizer->byte_to_cp[byte], encoded);
        uint32_t id = 0;
        if (!token_lookup(tokenizer, encoded, (uint32_t)length, &id)) {
            bnne_tokenizer_destroy(tokenizer);
            return bnne_error(BNNE_EFORMAT, "vocabulary is missing a byte token");
        }
        tokenizer->byte_token_id[byte] = id;
    }

    BNNE_Status status = build_pieces(tokenizer);
    if (status != BNNE_OK) {
        bnne_tokenizer_destroy(tokenizer);
        return status;
    }

    *out = tokenizer;
    return BNNE_OK;
}

void bnne_tokenizer_destroy(bnne_tokenizer* tokenizer) {
    if (!tokenizer)
        return;
    free(tokenizer->tokens);
    free(tokenizer->merges);
    free(tokenizer->specials);
    free(tokenizer->piece_text);
    free(tokenizer->piece_off);
    free(tokenizer->piece_len);
    free(tokenizer);
}

BNNE_Status bnne_tokenizer_encode(const bnne_tokenizer* tokenizer, const char* text,
                                  int parse_special, int32_t* ids, uint32_t capacity,
                                  uint32_t* out_count) {
    if (!tokenizer || !text || !ids || !out_count || capacity == 0)
        return BNNE_EINVAL;
    Output out = {ids, capacity, 0, 0};
    *out_count = 0;
    int ok = encode_text(tokenizer, text, strlen(text), parse_special, &out);
    *out_count = out.count;
    if (!ok)
        return out.overflow ? BNNE_ERANGE : BNNE_EINVAL;
    return BNNE_OK;
}

BNNE_Status bnne_tokenizer_piece(const bnne_tokenizer* tokenizer, int32_t id,
                                 const char** out_bytes, size_t* out_len) {
    if (!tokenizer || !out_bytes || !out_len)
        return BNNE_EINVAL;
    if (id < 0 || (uint64_t)id >= tokenizer->gguf->token_count)
        return BNNE_ERANGE;
    *out_bytes = tokenizer->piece_text + tokenizer->piece_off[id];
    *out_len = tokenizer->piece_len[id];
    return BNNE_OK;
}

int bnne_tokenizer_is_eog(const bnne_tokenizer* tokenizer, int32_t id) {
    if (!tokenizer || id < 0 || (uint64_t)id >= tokenizer->gguf->token_count)
        return 0;
    for (uint32_t i = 0; i < tokenizer->eog_count; i++)
        if (tokenizer->eog_ids[i] == (uint32_t)id)
            return 1;
    return 0;
}
