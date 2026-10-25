#define _POSIX_C_SOURCE 200809L

#include "gguf.h"

#include "bnne-internal.h"

#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3,
    GGUF_U32 = 4, GGUF_I32 = 5, GGUF_F32 = 6, GGUF_BOOL = 7,
    GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10, GGUF_I64 = 11,
    GGUF_F64 = 12
};

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    int ok;
} Cursor;

static uint8_t read_u8(Cursor* c) {
    if (!c->ok || c->p >= c->end) {
        c->ok = 0;
        return 0;
    }
    return *c->p++;
}

static uint16_t read_u16(Cursor* c) {
    uint16_t lo = read_u8(c);
    return (uint16_t)(lo | ((uint16_t)read_u8(c) << 8));
}

static uint32_t read_u32(Cursor* c) {
    uint32_t lo = read_u16(c);
    return lo | ((uint32_t)read_u16(c) << 16);
}

static uint64_t read_u64(Cursor* c) {
    uint64_t lo = read_u32(c);
    return lo | ((uint64_t)read_u32(c) << 32);
}

static float read_f32(Cursor* c) {
    uint32_t bits = read_u32(c);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static double read_f64(Cursor* c) {
    uint64_t bits = read_u64(c);
    double value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static bnne_gguf_string read_string(Cursor* c) {
    bnne_gguf_string s = {NULL, 0};
    uint64_t length = read_u64(c);
    if (!c->ok || length > UINT32_MAX || length > (uint64_t)(c->end - c->p)) {
        c->ok = 0;
        return s;
    }
    s.data = c->p;
    s.length = (uint32_t)length;
    c->p += (size_t)length;
    return s;
}

static int skip_value(Cursor* c, uint32_t type, unsigned depth) {
    if (!c->ok || depth > 4)
        return 0;
    switch (type) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL:
        (void)read_u8(c); return c->ok;
    case GGUF_U16: case GGUF_I16:
        (void)read_u16(c); return c->ok;
    case GGUF_U32: case GGUF_I32: case GGUF_F32:
        (void)read_u32(c); return c->ok;
    case GGUF_U64: case GGUF_I64: case GGUF_F64:
        (void)read_u64(c); return c->ok;
    case GGUF_STRING:
        (void)read_string(c); return c->ok;
    case GGUF_ARRAY: {
        uint32_t element = read_u32(c);
        uint64_t count = read_u64(c);
        if (!c->ok)
            return 0;
        for (uint64_t i = 0; i < count; i++)
            if (!skip_value(c, element, depth + 1))
                return 0;
        return 1;
    }
    default:
        return 0;
    }
}

static int integer_value(Cursor* c, uint32_t type, uint64_t* out) {
    switch (type) {
    case GGUF_U8: *out = read_u8(c); break;
    case GGUF_I8: *out = (uint64_t)(int64_t)(int8_t)read_u8(c); break;
    case GGUF_U16: *out = read_u16(c); break;
    case GGUF_I16: *out = (uint64_t)(int64_t)(int16_t)read_u16(c); break;
    case GGUF_U32: *out = read_u32(c); break;
    case GGUF_I32: *out = (uint64_t)(int64_t)(int32_t)read_u32(c); break;
    case GGUF_U64: case GGUF_I64: *out = read_u64(c); break;
    case GGUF_BOOL: *out = read_u8(c); break;
    default: return 0;
    }
    return c->ok;
}

static int float_value(Cursor* c, uint32_t type, float* out) {
    if (type == GGUF_F32)
        *out = read_f32(c);
    else if (type == GGUF_F64)
        *out = (float)read_f64(c);
    else
        return 0;
    return c->ok && isfinite(*out);
}

static int copy_string(const bnne_gguf_string* s, char* dst, size_t size) {
    if (size == 0)
        return 0;
    size_t length = s->length < size - 1 ? s->length : size - 1;
    memcpy(dst, s->data, length);
    dst[length] = '\0';
    return 1;
}

static int key_is(const bnne_gguf_string* key, const char* name) {
    size_t length = strlen(name);
    return key->length == length && memcmp(key->data, name, length) == 0;
}

static int mlp_activation_key(const bnne_gguf_string* key, uint32_t* out_index) {
    static const char prefix[] = "bnn-mlp.block.";
    static const char suffix[] = ".activation";
    size_t prefix_length = sizeof prefix - 1;
    size_t suffix_length = sizeof suffix - 1;
    if (key->length <= prefix_length + suffix_length ||
        memcmp(key->data, prefix, prefix_length) != 0 ||
        memcmp(key->data + key->length - suffix_length, suffix, suffix_length) != 0)
        return 0;
    uint64_t index = 0;
    for (size_t i = prefix_length; i < key->length - suffix_length; i++) {
        if (key->data[i] < '0' || key->data[i] > '9' || index > 63)
            return 0;
        index = index * 10 + (uint64_t)(key->data[i] - '0');
    }
    *out_index = (uint32_t)index;
    return 1;
}

static BNNE_Status store_mlp_activation(bnne_gguf* g, uint32_t index,
                                        const bnne_gguf_string* text) {
    if (index >= g->mlp_activation_count) {
        uint64_t count = (uint64_t)index + 1;
        if (count > 64)
            return BNNE_EFORMAT;
        char (*activations)[16] = realloc(g->mlp_activations, (size_t)count * sizeof *activations);
        if (!activations)
            return BNNE_ENOMEM;
        memset(activations + g->mlp_activation_count, 0,
               (size_t)(count - g->mlp_activation_count) * sizeof *activations);
        g->mlp_activations = activations;
        g->mlp_activation_count = count;
    }
    if (!copy_string(text, g->mlp_activations[index], sizeof g->mlp_activations[index]))
        return BNNE_EFORMAT;
    return BNNE_OK;
}

static BNNE_Status read_token_array(Cursor* c, uint32_t type, bnne_gguf_string** out,
                                    uint64_t* out_count) {
    if (type != GGUF_ARRAY)
        return BNNE_EFORMAT;
    uint32_t element = read_u32(c);
    uint64_t count = read_u64(c);
    if (!c->ok || element != GGUF_STRING ||
        count > (uint64_t)SIZE_MAX / sizeof(bnne_gguf_string))
        return BNNE_EFORMAT;
    bnne_gguf_string* values = count ? calloc((size_t)count, sizeof *values) : NULL;
    if (count && !values)
        return BNNE_ENOMEM;
    for (uint64_t i = 0; i < count; i++) {
        values[i] = read_string(c);
        if (!c->ok) {
            free(values);
            return BNNE_EFORMAT;
        }
    }
    *out = values;
    *out_count = count;
    return BNNE_OK;
}

static BNNE_Status read_token_types(Cursor* c, uint32_t type, uint32_t** out,
                                    uint64_t* out_count) {
    if (type != GGUF_ARRAY)
        return BNNE_EFORMAT;
    uint32_t element = read_u32(c);
    uint64_t count = read_u64(c);
    if (!c->ok || (element != GGUF_I32 && element != GGUF_U32) ||
        count > (uint64_t)SIZE_MAX / sizeof(uint32_t))
        return BNNE_EFORMAT;
    uint32_t* values = count ? malloc((size_t)count * sizeof *values) : NULL;
    if (count && !values)
        return BNNE_ENOMEM;
    for (uint64_t i = 0; i < count; i++)
        values[i] = read_u32(c);
    if (!c->ok) {
        free(values);
        return BNNE_EFORMAT;
    }
    *out = values;
    *out_count = count;
    return BNNE_OK;
}

static BNNE_Status parse_metadata(Cursor* c, bnne_gguf* g, uint64_t count) {
    if (!c->ok || count > 4096)
        return BNNE_EFORMAT;
    for (uint64_t i = 0; i < count; i++) {
        bnne_gguf_string key = read_string(c);
        uint32_t type = read_u32(c);
        if (!c->ok)
            return BNNE_EFORMAT;
        uint64_t integer = 0;
        uint32_t mlp_index = 0;
        bnne_gguf_string text = {NULL, 0};
        BNNE_Status status = BNNE_OK;

        if (key_is(&key, "general.architecture")) {
            if (type != GGUF_STRING) return BNNE_EFORMAT;
            text = read_string(c);
            if (!copy_string(&text, g->architecture, sizeof g->architecture)) return BNNE_EFORMAT;
        } else if (key_is(&key, "general.alignment")) {
            if (!integer_value(c, type, &integer) || integer == 0 || integer > 4096)
                return BNNE_EFORMAT;
            g->alignment = (uint32_t)integer;
        } else if (key_is(&key, "qwen3.block_count")) {
            if (!integer_value(c, type, &g->block_count)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.context_length")) {
            if (!integer_value(c, type, &g->context_length)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.embedding_length")) {
            if (!integer_value(c, type, &g->embedding_length)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.feed_forward_length")) {
            if (!integer_value(c, type, &g->feed_forward_length)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.attention.head_count")) {
            if (!integer_value(c, type, &g->head_count)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.attention.head_count_kv")) {
            if (!integer_value(c, type, &g->head_count_kv)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.attention.key_length")) {
            if (!integer_value(c, type, &g->key_length)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.attention.value_length")) {
            if (!integer_value(c, type, &g->value_length)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.attention.layer_norm_rms_epsilon")) {
            if (!float_value(c, type, &g->rms_epsilon)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.rope.freq_base")) {
            if (!float_value(c, type, &g->rope_freq_base)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.rope.scaling.type")) {
            if (type != GGUF_STRING) return BNNE_EFORMAT;
            text = read_string(c);
            if (!copy_string(&text, g->rope_scaling_type, sizeof g->rope_scaling_type))
                return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.rope.scaling.factor")) {
            if (!float_value(c, type, &g->rope_factor)) return BNNE_EFORMAT;
        } else if (key_is(&key, "qwen3.rope.scaling.original_context_length")) {
            if (!integer_value(c, type, &g->rope_original_context)) return BNNE_EFORMAT;
        } else if (key_is(&key, "bnn-mlp.block_count")) {
            if (!integer_value(c, type, &g->mlp_block_count)) return BNNE_EFORMAT;
        } else if (key_is(&key, "bnn-mlp.class_count")) {
            if (!integer_value(c, type, &g->mlp_class_count)) return BNNE_EFORMAT;
        } else if (key_is(&key, "bnn-mlp.input_features")) {
            if (!integer_value(c, type, &g->mlp_input_features)) return BNNE_EFORMAT;
        } else if (key_is(&key, "bnn-mlp.input_pixels")) {
            if (!integer_value(c, type, &g->mlp_input_pixels)) return BNNE_EFORMAT;
        } else if (key_is(&key, "bnn-mlp.input_mean")) {
            if (!float_value(c, type, &g->mlp_input_mean)) return BNNE_EFORMAT;
        } else if (key_is(&key, "bnn-mlp.input_std")) {
            if (!float_value(c, type, &g->mlp_input_std)) return BNNE_EFORMAT;
        } else if (key_is(&key, "bnn-mlp.labels")) {
            if (g->mlp_labels)
                return BNNE_EFORMAT;
            status = read_token_array(c, type, &g->mlp_labels, &g->mlp_label_count);
        } else if (mlp_activation_key(&key, &mlp_index)) {
            if (type != GGUF_STRING) return BNNE_EFORMAT;
            text = read_string(c);
            if (!c->ok) return BNNE_EFORMAT;
            status = store_mlp_activation(g, mlp_index, &text);
        } else if (key_is(&key, "tokenizer.ggml.model")) {
            if (type != GGUF_STRING) return BNNE_EFORMAT;
            text = read_string(c);
            if (!copy_string(&text, g->tokenizer_model, sizeof g->tokenizer_model))
                return BNNE_EFORMAT;
        } else if (key_is(&key, "tokenizer.ggml.pre")) {
            if (type != GGUF_STRING) return BNNE_EFORMAT;
            text = read_string(c);
            if (!copy_string(&text, g->tokenizer_pre, sizeof g->tokenizer_pre))
                return BNNE_EFORMAT;
        } else if (key_is(&key, "tokenizer.ggml.tokens")) {
            if (g->tokens)
                return BNNE_EFORMAT;
            status = read_token_array(c, type, &g->tokens, &g->token_count);
        } else if (key_is(&key, "tokenizer.ggml.token_type")) {
            if (g->token_types)
                return BNNE_EFORMAT;
            status = read_token_types(c, type, &g->token_types, &g->token_type_count);
        } else if (key_is(&key, "tokenizer.ggml.merges")) {
            if (g->merges)
                return BNNE_EFORMAT;
            status = read_token_array(c, type, &g->merges, &g->merge_count);
        } else if (key_is(&key, "tokenizer.ggml.eos_token_id")) {
            if (!integer_value(c, type, &integer)) return BNNE_EFORMAT;
            g->eos_token_id = (uint32_t)integer;
            g->has_eos = 1;
        } else if (key_is(&key, "tokenizer.ggml.bos_token_id")) {
            if (!integer_value(c, type, &integer)) return BNNE_EFORMAT;
            g->bos_token_id = (uint32_t)integer;
            g->has_bos = 1;
        } else if (key_is(&key, "tokenizer.ggml.padding_token_id")) {
            if (!integer_value(c, type, &integer)) return BNNE_EFORMAT;
            g->padding_token_id = (uint32_t)integer;
            g->has_padding = 1;
        } else if (key_is(&key, "tokenizer.ggml.add_bos_token")) {
            if (!integer_value(c, type, &integer)) return BNNE_EFORMAT;
            g->add_bos_token = integer != 0;
        } else if (!skip_value(c, type, 0)) {
            return BNNE_EFORMAT;
        }
        if (status != BNNE_OK)
            return status;
    }
    return c->ok ? BNNE_OK : BNNE_EFORMAT;
}
static int element_count(const bnne_gguf_tensor* t, uint64_t* out) {
    uint64_t count = 1;
    for (uint32_t d = 0; d < t->rank; d++) {
        if (t->dims[d] == 0 || count > UINT64_MAX / t->dims[d])
            return 0;
        count *= t->dims[d];
    }
    *out = count;
    return 1;
}

static BNNE_Status tensor_bytes(bnne_gguf_tensor* t) {
    uint64_t count = 0;
    if (!element_count(t, &count))
        return BNNE_EFORMAT;
    if (t->type == 0) {
        if (count > UINT64_MAX / 4) return BNNE_EFORMAT;
        t->bytes = count * 4;
    } else if (t->type == 1) {
        if (count > UINT64_MAX / 2) return BNNE_EFORMAT;
        t->bytes = count * 2;
    } else if (t->type == 41) {
        if (t->rank != 2 || t->dims[0] % 128 != 0 || t->dims[1] > UINT64_MAX / 18)
            return BNNE_EFORMAT;
        uint64_t groups = t->dims[0] / 128;
        if (t->dims[1] > UINT64_MAX / groups)
            return BNNE_EFORMAT;
        t->bytes = t->dims[1] * groups * 18;
    } else {
        return BNNE_ENOTSUP;
    }
    return BNNE_OK;
}

static BNNE_Status parse_tensors(Cursor* c, bnne_gguf* g, uint64_t count) {
    if (!c->ok || count == 0 || count > (uint64_t)SIZE_MAX / sizeof(bnne_gguf_tensor))
        return BNNE_EFORMAT;
    g->tensors = calloc((size_t)count, sizeof *g->tensors);
    if (!g->tensors)
        return BNNE_ENOMEM;
    g->tensor_count = count;
    for (uint64_t i = 0; i < count; i++) {
        bnne_gguf_tensor* t = &g->tensors[i];
        t->name = read_string(c);
        t->rank = read_u32(c);
        if (!c->ok || t->rank == 0 || t->rank > BNNE_GGUF_MAX_DIMS)
            return BNNE_EFORMAT;
        for (uint32_t d = 0; d < t->rank; d++)
            t->dims[d] = read_u64(c);
        t->type = read_u32(c);
        t->offset = read_u64(c);
        if (!c->ok)
            return BNNE_EFORMAT;
    }
    return BNNE_OK;
}

BNNE_Status bnne_gguf_open(const char* path, bnne_gguf** out) {
    if (!path || !out)
        return BNNE_EINVAL;
    *out = NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return bnne_error(BNNE_EIO, "%s: cannot open GGUF file", path);

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return bnne_error(BNNE_EIO, "%s: cannot stat GGUF file", path);
    }
    size_t size = (size_t)st.st_size;
    uint8_t* mapping = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED)
        return bnne_error(BNNE_EIO, "%s: cannot mmap GGUF file", path);

    bnne_gguf* g = calloc(1, sizeof *g);
    if (!g) {
        munmap(mapping, size);
        return bnne_error(BNNE_ENOMEM, "out of memory");
    }
    g->mapping = mapping;
    g->mapping_bytes = size;
    g->alignment = 32;

    Cursor c = {mapping, mapping + size, 1};
    char magic[4] = {0};
    for (size_t i = 0; i < sizeof magic && c.ok; i++)
        magic[i] = (char)read_u8(&c);
    uint32_t version = read_u32(&c);
    uint64_t tensor_count = read_u64(&c);
    uint64_t metadata_count = read_u64(&c);
    if (!c.ok || memcmp(magic, "GGUF", 4) != 0 || version != 3) {
        bnne_gguf_close(g);
        return bnne_error(BNNE_EFORMAT, "%s: unsupported GGUF header or version", path);
    }

    BNNE_Status status = parse_metadata(&c, g, metadata_count);
    if (status == BNNE_OK)
        status = parse_tensors(&c, g, tensor_count);
    if (status != BNNE_OK) {
        bnne_gguf_close(g);
        return bnne_error(status, "%s: invalid GGUF metadata or tensor table", path);
    }
    if (!g->architecture[0]) {
        bnne_gguf_close(g);
        return bnne_error(BNNE_EFORMAT, "%s: missing general.architecture", path);
    }

    size_t info_end = (size_t)(c.p - mapping);
    size_t align = g->alignment;
    if (align == 0 || info_end > SIZE_MAX - (align - 1)) {
        bnne_gguf_close(g);
        return bnne_error(BNNE_EFORMAT, "%s: invalid GGUF alignment", path);
    }
    g->data_offset = (uint64_t)((info_end + align - 1) / align * align);
    if (g->data_offset > size) {
        bnne_gguf_close(g);
        return bnne_error(BNNE_EFORMAT, "%s: GGUF data section is out of bounds", path);
    }

    for (uint64_t i = 0; i < g->tensor_count; i++) {
        bnne_gguf_tensor* t = &g->tensors[i];
        status = tensor_bytes(t);
        if (status != BNNE_OK) {
            bnne_gguf_close(g);
            return bnne_error(status, "%s: unsupported tensor type or shape", path);
        }
        uint64_t available = (uint64_t)(size - (size_t)g->data_offset);
        if (t->offset > available || t->bytes > available - t->offset) {
            bnne_gguf_close(g);
            return bnne_error(BNNE_EFORMAT, "%s: GGUF tensor data is out of bounds", path);
        }
    }

    if (g->rope_factor <= 0.0f)
        g->rope_factor = 1.0f;
    if (g->rope_original_context == 0)
        g->rope_original_context = g->context_length;

    *out = g;
    return BNNE_OK;
}

void bnne_gguf_close(bnne_gguf* g) {
    if (!g)
        return;
    free(g->tensors);
    free(g->tokens);
    free(g->token_types);
    free(g->merges);
    free(g->mlp_activations);
    free(g->mlp_labels);
    if (g->mapping)
        munmap(g->mapping, g->mapping_bytes);
    free(g);
}

const bnne_gguf_tensor* bnne_gguf_find(const bnne_gguf* g, const char* name) {
    if (!g || !name)
        return NULL;
    size_t length = strlen(name);
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        const bnne_gguf_tensor* t = &g->tensors[i];
        if (t->name.length == length && memcmp(t->name.data, name, length) == 0)
            return t;
    }
    return NULL;
}

const uint8_t* bnne_gguf_data(const bnne_gguf* g, const bnne_gguf_tensor* t) {
    if (!g || !t)
        return NULL;
    return g->mapping + g->data_offset + t->offset;
}
