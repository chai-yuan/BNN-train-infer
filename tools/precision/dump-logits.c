#include "bnn-engine.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* model;
    const char* prompt;
    const char* ids_file;
    const char* out;
    uint32_t ctx;
    int ctx_set;
} Options;

static void usage(const char* prog) {
    fprintf(stderr,
            "usage: %s --model MODEL.gguf (--prompt TEXT | --ids-file FILE) --out FILE [--ctx N]\n"
            "\n"
            "Decode each token one at a time and dump per-position logits from the C engine.\n"
            "Output format matches tools/bonsai-ref.py --out:\n"
            "  uint32 n, uint32 vocab, int32 ids[n], float32 logits[n*vocab], little-endian.\n"
            "Prompts are tokenized with parse_special=1.\n",
            prog);
}

static int parse_u32(const char* text, uint32_t* out) {
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 || value > UINT32_MAX)
        return 0;
    *out = (uint32_t)value;
    return 1;
}

static int host_little_endian(void) {
    const uint16_t one = 1;
    return *(const unsigned char*)&one == 1;
}

static int write_u32_le(FILE* file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)value,
        (unsigned char)(value >> 8),
        (unsigned char)(value >> 16),
        (unsigned char)(value >> 24),
    };
    return fwrite(bytes, 1, sizeof bytes, file) == sizeof bytes;
}

static int write_i32_le(FILE* file, int32_t value) {
    return write_u32_le(file, (uint32_t)value);
}

static int write_f32_array_le(FILE* file, const float* values, size_t count) {
    if (host_little_endian())
        return fwrite(values, sizeof *values, count, file) == count;

    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof bits);
        if (!write_u32_le(file, bits))
            return 0;
    }
    return 1;
}

static int read_ids_file(const char* path, uint32_t vocab, int32_t** out_ids, uint32_t* out_count) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "error: cannot open '%s': %s\n", path, strerror(errno));
        return 0;
    }

    int32_t* ids = NULL;
    size_t capacity = 0;
    size_t count = 0;
    long value;

    while (fscanf(file, "%ld", &value) == 1) {
        if (value < 0 || (uint64_t)value >= vocab) {
            fprintf(stderr, "error: token %ld in '%s' is outside vocabulary %u\n", value, path,
                    vocab);
            free(ids);
            fclose(file);
            return 0;
        }
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 64;
            if (next < capacity || next > SIZE_MAX / sizeof *ids) {
                fprintf(stderr, "error: token file is too large\n");
                free(ids);
                fclose(file);
                return 0;
            }
            int32_t* grown = realloc(ids, next * sizeof *ids);
            if (!grown) {
                fprintf(stderr, "error: out of memory\n");
                free(ids);
                fclose(file);
                return 0;
            }
            ids = grown;
            capacity = next;
        }
        ids[count++] = (int32_t)value;
    }

    if (ferror(file)) {
        fprintf(stderr, "error: read failed for '%s'\n", path);
        free(ids);
        fclose(file);
        return 0;
    }
    if (!feof(file)) {
        fprintf(stderr, "error: '%s' contains non-numeric data\n", path);
        free(ids);
        fclose(file);
        return 0;
    }
    fclose(file);

    if (count == 0) {
        fprintf(stderr, "error: '%s' contains no token IDs\n", path);
        free(ids);
        return 0;
    }
    if (count > UINT32_MAX) {
        fprintf(stderr, "error: too many token IDs\n");
        free(ids);
        return 0;
    }

    *out_ids = ids;
    *out_count = (uint32_t)count;
    return 1;
}

static int parse_options(int argc, char** argv, Options* opts) {
    memset(opts, 0, sizeof *opts);

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "--model") == 0 && i + 1 < argc) {
            opts->model = argv[++i];
        } else if (strcmp(arg, "--prompt") == 0 && i + 1 < argc) {
            opts->prompt = argv[++i];
        } else if (strcmp(arg, "--ids-file") == 0 && i + 1 < argc) {
            opts->ids_file = argv[++i];
        } else if (strcmp(arg, "--out") == 0 && i + 1 < argc) {
            opts->out = argv[++i];
        } else if (strcmp(arg, "--ctx") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opts->ctx))
                return 0;
            opts->ctx_set = 1;
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            return 0;
        }
    }

    return opts->model != NULL && opts->out != NULL &&
           ((opts->prompt != NULL) != (opts->ids_file != NULL));
}

int main(int argc, char** argv) {
    Options opts;
    if (!parse_options(argc, argv, &opts)) {
        usage(argv[0]);
        return 2;
    }

    BNN_Device device = NULL;
    BNNE_Model model = NULL;
    BNNE_Context context = NULL;
    int32_t* ids = NULL;
    float* logits = NULL;
    FILE* out = NULL;
    int ok = 0;

    BNN_Status device_status = BNN_DeviceOpen(0, &device);
    if (device_status != BNN_OK) {
        fprintf(stderr, "error: BNN_DeviceOpen: %s\n", BNN_StatusString(device_status));
        goto cleanup;
    }

    BNNE_Status status = BNNE_ModelLoad(device, opts.model, &model);
    if (status != BNNE_OK) {
        fprintf(stderr, "error: %s: %s\n", BNNE_StatusString(status), BNNE_LastError());
        goto cleanup;
    }

    BNNE_ModelInfo info;
    status = BNNE_ModelGetInfo(model, &info);
    if (status != BNNE_OK) {
        fprintf(stderr, "error: %s: %s\n", BNNE_StatusString(status), BNNE_LastError());
        goto cleanup;
    }
    if (info.task != BNNE_TASK_LLM) {
        fprintf(stderr, "error: model is not a language model\n");
        goto cleanup;
    }
    if (info.n_vocab == 0 || info.n_ctx_train == 0) {
        fprintf(stderr, "error: model reports an empty vocabulary or context\n");
        goto cleanup;
    }
    if ((uint64_t)info.n_vocab * sizeof(float) > SIZE_MAX) {
        fprintf(stderr, "error: vocabulary is too large\n");
        goto cleanup;
    }

    uint32_t count = 0;
    if (opts.prompt) {
        uint32_t capacity = opts.ctx_set ? opts.ctx : info.n_ctx_train;
        if ((uint64_t)capacity * sizeof(int32_t) > SIZE_MAX) {
            fprintf(stderr, "error: context capacity is too large\n");
            goto cleanup;
        }
        ids = malloc((size_t)capacity * sizeof *ids);
        if (!ids) {
            fprintf(stderr, "error: out of memory\n");
            goto cleanup;
        }
        status = BNNE_Tokenize(model, opts.prompt, 1, ids, capacity, &count);
        if (status != BNNE_OK) {
            fprintf(stderr, "error: BNNE_Tokenize: %s: %s\n", BNNE_StatusString(status),
                    BNNE_LastError());
            goto cleanup;
        }
    } else {
        if (!read_ids_file(opts.ids_file, info.n_vocab, &ids, &count))
            goto cleanup;
    }

    if (count == 0) {
        fprintf(stderr, "error: prompt produced no tokens\n");
        goto cleanup;
    }
    if (count > info.n_ctx_train) {
        fprintf(stderr, "error: %u tokens exceed model context %u\n", count, info.n_ctx_train);
        goto cleanup;
    }

    uint32_t ctx_len = opts.ctx_set ? opts.ctx : count;
    if (ctx_len < count) {
        fprintf(stderr, "error: --ctx %u is smaller than token count %u\n", ctx_len, count);
        goto cleanup;
    }
    if (ctx_len > info.n_ctx_train) {
        fprintf(stderr, "error: --ctx %u exceeds model context %u\n", ctx_len, info.n_ctx_train);
        goto cleanup;
    }

    BNNE_ContextParams params;
    BNNE_ContextParamsInit(&params);
    params.n_ctx = ctx_len;
    params.n_batch = 1;
    params.enable_stats = 0;
    status = BNNE_ContextCreate(model, &params, &context);
    if (status != BNNE_OK) {
        fprintf(stderr, "error: %s: %s\n", BNNE_StatusString(status), BNNE_LastError());
        goto cleanup;
    }

    logits = malloc((size_t)info.n_vocab * sizeof *logits);
    if (!logits) {
        fprintf(stderr, "error: out of memory\n");
        goto cleanup;
    }

    out = fopen(opts.out, "wb");
    if (!out) {
        fprintf(stderr, "error: cannot open '%s': %s\n", opts.out, strerror(errno));
        goto cleanup;
    }

    if (!write_u32_le(out, count) || !write_u32_le(out, info.n_vocab)) {
        fprintf(stderr, "error: write failed for '%s'\n", opts.out);
        goto cleanup;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (!write_i32_le(out, ids[i])) {
            fprintf(stderr, "error: write failed for '%s'\n", opts.out);
            goto cleanup;
        }
    }

    for (uint32_t i = 0; i < count; ++i) {
        status = BNNE_Decode(context, &ids[i], 1, logits);
        if (status != BNNE_OK) {
            fprintf(stderr, "error: BNNE_Decode(position %u): %s: %s\n", i,
                    BNNE_StatusString(status), BNNE_LastError());
            goto cleanup;
        }
        if (!write_f32_array_le(out, logits, info.n_vocab)) {
            fprintf(stderr, "error: write failed for '%s'\n", opts.out);
            goto cleanup;
        }
    }

    if (fclose(out) != 0) {
        out = NULL;
        fprintf(stderr, "error: close failed for '%s': %s\n", opts.out, strerror(errno));
        goto cleanup;
    }
    out = NULL;

    printf("wrote %s: n=%u V=%u activation=w1a8\n", opts.out, count, info.n_vocab);
    ok = 1;

cleanup:
    if (out)
        fclose(out);
    if (!ok && opts.out)
        remove(opts.out);
    free(logits);
    free(ids);
    BNNE_ContextFree(context);
    BNNE_ModelFree(model);
    if (device) {
        BNN_Status close_status = BNN_DeviceClose(device);
        if (ok && close_status != BNN_OK) {
            fprintf(stderr, "error: BNN_DeviceClose: %s\n", BNN_StatusString(close_status));
            ok = 0;
        }
    }
    return ok ? 0 : 1;
}
