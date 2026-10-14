/* Unified BNNE v2 demo: two task types, selected by the model file.
 * Classification (MLP/CNN): labeled dataset in, class predictions scored as accuracy.
 * LLM (GPT): prompt in, text out; --eval scores next-token prediction and --ref
 * compares logits against the Python export. --profile works for both tasks.
 * ./bnn-infer [--profile] [--ref FILE] model.bnn images.idx labels.idx
 * ./bnn-infer [--profile] [--ref FILE] [--eval FILE] [--prompt TEXT] [--steps N] model.bnn
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bnn-sw/bnn-engine.h"

#define IMAGE_PIXELS 784u
#define CLASSES 10u
#define MAX_ITEMS 100000u

typedef struct {
    const char *ref, *eval, *prompt;
    uint32_t steps;
    int profile, generate, count, kv_cache;
    const char* paths[3];
} Args;

typedef struct {
    uint32_t count;
    int32_t* ids;
    float* logits;
} Reference;

typedef struct {
    uint32_t agree;
    double max_relative;
} Comparison;

static int error(const char* message) {
    fprintf(stderr, "error: %s\n", message);
    return 0;
}

static int api_ok(BNNE_Engine engine, BNNE_Status status, const char* operation) {
    if (status == BNNE_OK)
        return 1;
    const char* detail = engine ? BNNE_LastError(engine) : "";
    fprintf(stderr, "error: %s: %s%s%s\n", operation, BNNE_StatusString(status),
            detail && *detail ? ": " : "", detail ? detail : "");
    return 0;
}

static void* alloc_array(uint64_t count, size_t size) {
    if (!count || count > SIZE_MAX / size) {
        error("invalid or overflowing allocation size");
        return NULL;
    }
    void* data = malloc((size_t)count * size);
    if (!data)
        error("out of memory");
    return data;
}

static FILE* open_file(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file)
        fprintf(stderr, "error: %s: %s\n", path, strerror(errno));
    return file;
}

static int read_exact(FILE* file, void* data, size_t bytes, const char* path) {
    if (fread(data, 1, bytes, file) == bytes)
        return 1;
    fprintf(stderr, "error: %s: %s\n", path,
            ferror(file) ? "read failed" : "unexpected end of file");
    return 0;
}

static uint32_t decode_le32(const unsigned char* bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int read_u32(FILE* file, const char* path, int big_endian, uint32_t* value) {
    unsigned char bytes[4];
    if (!read_exact(file, bytes, sizeof bytes, path))
        return 0;
    *value = big_endian ? ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
                         ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3]
                       : decode_le32(bytes);
    return 1;
}

static int check_end(FILE* file, const char* path) {
    if (fgetc(file) == EOF && !ferror(file))
        return 1;
    fprintf(stderr, "error: %s: trailing bytes or I/O error\n", path);
    return 0;
}

static int close_file(FILE* file, const char* path) {
    if (fclose(file) == 0)
        return 1;
    fprintf(stderr, "error: %s: close failed: %s\n", path, strerror(errno));
    return 0;
}

static uint32_t argmax(const float* logits, uint32_t width) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < width; ++i)
        if (logits[i] > logits[best])
            best = i;
    return best;
}

static void compare_row(Comparison* comparison, const float* logits,
                        const float* ref, uint32_t width) {
    comparison->agree += argmax(logits, width) == argmax(ref, width);
    for (uint32_t i = 0; i < width; ++i) {
        double relative = fabs((double)logits[i] - ref[i]) / (1.0 + fabs((double)ref[i]));
        if (relative > comparison->max_relative)
            comparison->max_relative = relative;
    }
}

static int report_comparison(const Comparison* comparison, uint32_t count,
                             double tolerance, int require_argmax) {
    int mismatch = (require_argmax && comparison->agree != count) ||
                   !(comparison->max_relative < tolerance);
    printf("C vs Python reference: argmax agree %" PRIu32 "/%" PRIu32
           ", max relative error %.3e (limit %.1e) %s\n",
           comparison->agree, count, comparison->max_relative, tolerance,
           mismatch ? "MISMATCH" : "OK");
    return mismatch;
}

static int monotonic(struct timespec* time) {
    if (clock_gettime(CLOCK_MONOTONIC, time) == 0)
        return 1;
    return error("cannot read monotonic clock");
}

static int phase_begin(BNNE_Engine engine, struct timespec* start) {
    return api_ok(engine, BNNE_ResetStats(engine), "BNNE_ResetStats") && monotonic(start);
}

static int phase_end(BNNE_Engine engine, const struct timespec* start,
                     const char* phase, int profile) {
    struct timespec end;
    BNNE_Stats stats;
    if (!monotonic(&end) || !api_ok(engine, BNNE_GetStats(engine, &stats), "BNNE_GetStats"))
        return 0;
    if (profile) {
        double wall_ms = (double)(end.tv_sec - start->tv_sec) * 1e3 +
                         (double)(end.tv_nsec - start->tv_nsec) / 1e6;
        printf("profile %s: wall=%.3f ms, forward_calls=%" PRIu64 ", total=%.3f ms\n"
               "  quantize=%.3f ms, transfer=%.3f ms, execute=%.3f ms, host=%.3f ms\n"
               "  device_calls=%" PRIu64 ", bytes_written=%" PRIu64 ", bytes_read=%" PRIu64
               ", logical_macs=%" PRIu64 ", executed_macs=%" PRIu64 "\n",
               phase, wall_ms, stats.forward_calls, (double)stats.total_ns / 1e6,
               (double)stats.quantize_ns / 1e6, (double)stats.transfer_ns / 1e6,
               (double)stats.execute_ns / 1e6, (double)stats.host_ns / 1e6,
               stats.device_calls, stats.bytes_written, stats.bytes_read,
               stats.logical_macs, stats.executed_macs);
    }
    return 1;
}

static int read_ids(FILE* file, const char* path, int32_t* ids, uint32_t count,
                    uint32_t vocabulary) {
    if (!read_exact(file, ids, (size_t)count * sizeof *ids, path))
        return 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t id = decode_le32((const unsigned char*)ids + (size_t)i * 4);
        if (id > INT32_MAX || id >= vocabulary) {
            fprintf(stderr, "error: %s: invalid token ID at position %" PRIu32 "\n", path, i);
            return 0;
        }
        ids[i] = (int32_t)id;
    }
    return 1;
}

/* Classification: n,10,logits; LLM: n,V,ids,logits. All words are little-endian. */
static int load_ref(const char* path, uint32_t limit, uint32_t width, int with_ids,
                    Reference* ref) {
    FILE* file = open_file(path);
    Reference loaded = {0};
    uint32_t columns;
    int ok = 0;
    if (!file)
        return 0;
    if (!read_u32(file, path, 0, &loaded.count) || !read_u32(file, path, 0, &columns))
        goto cleanup;
    if (!loaded.count || loaded.count > limit || columns != width) {
        error("reference count or logit width does not match model/data");
        goto cleanup;
    }
    if (with_ids) {
        loaded.ids = alloc_array(loaded.count, sizeof *loaded.ids);
        if (!loaded.ids || !read_ids(file, path, loaded.ids, loaded.count, width))
            goto cleanup;
    }
    uint64_t elements = (uint64_t)loaded.count * width;
    loaded.logits = alloc_array(elements, sizeof *loaded.logits);
    if (!loaded.logits ||
        !read_exact(file, loaded.logits, (size_t)elements * sizeof(float), path) ||
        !check_end(file, path))
        goto cleanup;
    for (size_t i = 0; i < (size_t)elements; ++i) {
        uint32_t bits = decode_le32((const unsigned char*)loaded.logits + i * 4);
        memcpy(loaded.logits + i, &bits, sizeof bits);
        if (!isfinite(loaded.logits[i])) {
            error("non-finite reference logit");
            goto cleanup;
        }
    }
    ok = 1;
cleanup:
    if (!close_file(file, path))
        ok = 0;
    if (ok) {
        *ref = loaded;
    } else {
        free(loaded.ids);
        free(loaded.logits);
    }
    return ok;
}

/* ---- task: classification (MNIST MLP/CNN), dataset in, accuracy out ---- */

static int load_idx(const char* path, int images, uint8_t** out, uint32_t* out_count) {
    FILE* file = open_file(path);
    uint8_t* data = NULL;
    uint32_t magic, count, rows, cols;
    int ok = 0;
    if (!file)
        return 0;
    if (!read_u32(file, path, 1, &magic) || !read_u32(file, path, 1, &count))
        goto cleanup;
    if (magic != (images ? 0x803u : 0x801u) || !count || count > MAX_ITEMS) {
        error("invalid IDX magic or count (expected 1..100000)");
        goto cleanup;
    }
    if (images) {
        if (!read_u32(file, path, 1, &rows) || !read_u32(file, path, 1, &cols))
            goto cleanup;
        if (rows != 28 || cols != 28) {
            error("MNIST images must be 28x28");
            goto cleanup;
        }
    }
    size_t bytes = (size_t)count * (images ? IMAGE_PIXELS : 1u);
    data = alloc_array(bytes, 1);
    if (!data || !read_exact(file, data, bytes, path) || !check_end(file, path))
        goto cleanup;
    if (!images)
        for (uint32_t i = 0; i < count; ++i)
            if (data[i] >= CLASSES) {
                error("MNIST label outside 0..9");
                goto cleanup;
            }
    ok = 1;
cleanup:
    if (!close_file(file, path))
        ok = 0;
    if (ok) {
        *out = data;
        *out_count = count;
    } else {
        free(data);
    }
    return ok;
}

static int classify_check(const BNNE_ModelInfo* info, const Args* args) {
    if (args->eval || args->generate)
        return error("--eval/--prompt/--steps belong to the LLM task");
    if ((info->kind != BNNE_MODEL_MLP && info->kind != BNNE_MODEL_CNN) ||
        info->input.dtype != BNNE_F32 || info->output.dtype != BNNE_F32 ||
        info->output.rank != 1 || info->output.dims[0] != CLASSES)
        return error("classification expects an MNIST MLP/CNN with 10 FP32 output classes");
    if ((info->kind == BNNE_MODEL_MLP &&
         (info->input.rank != 1 || info->input.dims[0] < IMAGE_PIXELS)) ||
        (info->kind == BNNE_MODEL_CNN &&
         (info->input.rank != 3 || info->input.dims[0] != 1 ||
          info->input.dims[1] != 28 || info->input.dims[2] != 28)))
        return error("invalid MNIST input shape");
    return 1;
}

static int classify_forward(BNNE_Engine engine, const uint8_t* pixels,
                            BNNE_Tensor* input, BNNE_Tensor* output) {
    float* data = input->data;
    for (uint32_t i = 0; i < IMAGE_PIXELS; ++i)
        data[i] = ((float)pixels[i] / 255.0f - 0.1307f) / 0.3081f;
    /* Only the MLP vector has padding; CNN is exactly [1,28,28]. */
    for (size_t i = IMAGE_PIXELS; i < input->bytes / sizeof(float); ++i)
        data[i] = 0.0f;
    return api_ok(engine, BNNE_Run(engine, input, output), "BNNE_Run");
}

static int classify_task(BNNE_Engine engine, const BNNE_ModelInfo* info, const Args* args) {
    uint8_t *images = NULL, *labels = NULL;
    uint32_t count = 0, label_count = 0, correct = 0;
    Reference ref = {0};
    struct timespec start;
    float logits[CLASSES];
    uint32_t width = info->kind == BNNE_MODEL_MLP ? info->input.dims[0] : IMAGE_PIXELS;
    BNNE_Tensor input = {info->input, NULL, 0};
    BNNE_Tensor output = {info->output, logits, sizeof logits};
    int result = 1;
    if (!classify_check(info, args))
        return result;
    if (!load_idx(args->paths[1], 1, &images, &count) ||
        !load_idx(args->paths[2], 0, &labels, &label_count))
        goto cleanup;
    if (count != label_count) {
        error("image/label count mismatch");
        goto cleanup;
    }
    if (args->ref && !load_ref(args->ref, count, CLASSES, 0, &ref))
        goto cleanup;
    input.data = alloc_array(width, sizeof(float));
    if (!input.data)
        goto cleanup;
    input.bytes = (size_t)width * sizeof(float);
    if (!phase_begin(engine, &start))
        goto cleanup;
    for (uint32_t i = 0; i < count; ++i) {
        if (!classify_forward(engine, images + (size_t)i * IMAGE_PIXELS, &input, &output))
            goto cleanup;
        correct += argmax(logits, CLASSES) == labels[i];
    }
    printf("C inference accuracy: %.4f (%" PRIu32 "/%" PRIu32 ")\n",
           (double)correct / count, correct, count);
    if (!phase_end(engine, &start, "eval", args->profile))
        goto cleanup;
    int mismatch = 0;
    if (args->ref) {
        Comparison comparison = {0};
        if (!phase_begin(engine, &start))
            goto cleanup;
        for (uint32_t i = 0; i < ref.count; ++i) {
            if (!classify_forward(engine, images + (size_t)i * IMAGE_PIXELS, &input, &output))
                goto cleanup;
            compare_row(&comparison, logits, ref.logits + (size_t)i * CLASSES, CLASSES);
        }
        mismatch = report_comparison(&comparison, ref.count, 1e-3, 0);
        if (!phase_end(engine, &start, "check", args->profile))
            goto cleanup;
    }
    result = mismatch;
cleanup:
    free(input.data);
    free(ref.logits);
    free(labels);
    free(images);
    return result;
}

/* ---- task: LLM (nanoGPT), prompt in, text out ---- */

static int load_tokens(const char* path, const BNNE_ModelInfo* info,
                       int32_t** out, uint32_t* out_count) {
    FILE* file = open_file(path);
    int32_t* ids = NULL;
    uint32_t count;
    int ok = 0;
    if (!file)
        return 0;
    if (!read_u32(file, path, 0, &count))
        goto cleanup;
    if (count <= info->max_sequence_length) {
        error("evaluation requires at least T+1 tokens");
        goto cleanup;
    }
    ids = alloc_array(count, sizeof *ids);
    if (!ids || !read_ids(file, path, ids, count, info->vocabulary_size) ||
        !check_end(file, path))
        goto cleanup;
    ok = 1;
cleanup:
    if (!close_file(file, path))
        ok = 0;
    if (ok) {
        *out = ids;
        *out_count = count;
    } else {
        free(ids);
    }
    return ok;
}

static int llm_forward(BNNE_Engine engine, int32_t* ids, uint32_t count, BNNE_Tensor* output) {
    BNNE_Tensor input = {{BNNE_I32, 1, {count}}, ids, (size_t)count * sizeof *ids};
    return api_ok(engine, BNNE_GetOutputDesc(engine, &input.desc, &output->desc),
                  "BNNE_GetOutputDesc") &&
           api_ok(engine, BNNE_Run(engine, &input, output), "BNNE_Run");
}

static int llm_eval(BNNE_Engine engine, const BNNE_ModelInfo* info, const char* path,
                    BNNE_Tensor* output, int profile) {
    int32_t* ids = NULL;
    uint32_t count = 0, T = info->max_sequence_length, V = info->vocabulary_size;
    uint64_t positions = 0, correct = 0;
    double loss_sum = 0.0;
    struct timespec start;
    int ok = 0;
    if (!load_tokens(path, info, &ids, &count) || !phase_begin(engine, &start))
        goto cleanup;
    /* Complete T-token windows, stride=T, matching the original demo protocol. */
    for (uint64_t offset = 0; offset < (uint64_t)count - T; offset += T) {
        if (!llm_forward(engine, ids + (size_t)offset, T, output))
            goto cleanup;
        for (uint32_t t = 0; t < T; ++t) {
            const float* row = (const float*)output->data + (size_t)t * V;
            uint32_t target = (uint32_t)ids[(size_t)offset + t + 1];
            uint32_t best = argmax(row, V);
            double sum = 0.0, maximum = row[best];
            for (uint32_t v = 0; v < V; ++v)
                sum += exp((double)row[v] - maximum);
            loss_sum += maximum - row[target] + log(sum);
            correct += best == target;
            ++positions;
        }
    }
    double loss = loss_sum / (double)positions, perplexity = exp(loss);
    if (!positions || !isfinite(loss) || !isfinite(perplexity)) {
        error("invalid evaluation loss/perplexity or position count");
        goto cleanup;
    }
    printf("C token eval: %" PRIu64 " positions, loss %.4f, perplexity %.4f, top1 %.4f\n",
           positions, loss, perplexity, (double)correct / (double)positions);
    ok = phase_end(engine, &start, "eval", profile);
cleanup:
    free(ids);
    return ok;
}

static void append_id(int32_t* ids, uint32_t* count, uint32_t capacity, int32_t id) {
    if (*count == capacity) {
        memmove(ids, ids + 1, (size_t)(capacity - 1) * sizeof *ids);
        --*count;
    }
    ids[(*count)++] = id;
}

static int llm_generate(BNNE_Engine engine, const BNNE_ModelInfo* info, const Args* args,
                        BNNE_Tensor* output) {
    int32_t* ids = alloc_array(info->max_sequence_length, sizeof *ids);
    uint32_t count = 0;
    struct timespec start;
    int ok = 0;
    if (!ids || !phase_begin(engine, &start))
        goto cleanup;
    const char* prompt = args->prompt ? args->prompt : "春天";
    for (const char* p = prompt; *p;) {
        size_t best_length = 0;
        uint32_t best = 0;
        for (uint32_t id = 0; id < info->vocabulary_size; ++id) {
            const char* token;
            if (!api_ok(engine, BNNE_GetToken(engine, id, &token), "BNNE_GetToken"))
                goto cleanup;
            size_t length = strlen(token);
            if (length > best_length && strncmp(p, token, length) == 0) {
                best = id;
                best_length = length;
            }
        }
        if (!best_length) {
            error("prompt contains text outside the vocabulary");
            goto cleanup;
        }
        append_id(ids, &count, info->max_sequence_length, (int32_t)best);
        p += best_length;
    }
    if (!count) {
        error("prompt must contain at least one token");
        goto cleanup;
    }
    printf("generate: %s", prompt);

    if (args->kv_cache) {
        float* step_logits = alloc_array(info->vocabulary_size, sizeof *step_logits);
        if (!step_logits)
            goto cleanup;

        /* Prefill: 将 Prompt 注入 KV cache */
        BNNE_ResetKVCache(engine);
        for (uint32_t i = 0; i < count; ++i) {
            if (!api_ok(engine, BNNE_StepToken(engine, ids[i], i, step_logits), "BNNE_StepToken(prefill)")) {
                free(step_logits);
                goto cleanup;
            }
        }

        uint32_t next = argmax(step_logits, info->vocabulary_size);
        const char* token;
        if (!api_ok(engine, BNNE_GetToken(engine, next, &token), "BNNE_GetToken")) {
            free(step_logits);
            goto cleanup;
        }
        fputs(token, stdout);
        append_id(ids, &count, info->max_sequence_length, (int32_t)next);

        /* Decode: 增量自回归生成 */
        for (uint32_t step = 1; step < args->steps; ++step) {
            if (count >= info->max_sequence_length)
                break;
            uint32_t pos = count - 1;
            if (!api_ok(engine, BNNE_StepToken(engine, (int32_t)next, pos, step_logits), "BNNE_StepToken(decode)")) {
                free(step_logits);
                goto cleanup;
            }
            next = argmax(step_logits, info->vocabulary_size);
            if (!api_ok(engine, BNNE_GetToken(engine, next, &token), "BNNE_GetToken")) {
                free(step_logits);
                goto cleanup;
            }
            fputs(token, stdout);
            append_id(ids, &count, info->max_sequence_length, (int32_t)next);
        }
        free(step_logits);
    } else {
        for (uint32_t step = 0; step < args->steps; ++step) {
            if (!llm_forward(engine, ids, count, output))
                goto cleanup;
            uint32_t next = argmax((const float*)output->data +
                                   (size_t)(count - 1) * info->vocabulary_size, info->vocabulary_size);
            const char* token;
            if (!api_ok(engine, BNNE_GetToken(engine, next, &token), "BNNE_GetToken"))
                goto cleanup;
            fputs(token, stdout);
            append_id(ids, &count, info->max_sequence_length, (int32_t)next);
        }
    }
    putchar('\n');
    ok = phase_end(engine, &start, "generate", args->profile);
cleanup:
    free(ids);
    return ok;
}

static int llm_check(const BNNE_ModelInfo* info) {
    if ((info->kind != BNNE_MODEL_GPT && info->kind != BNNE_MODEL_LLAMA) ||
        info->input.dtype != BNNE_I32 || info->input.rank != 1 ||
        !info->max_sequence_length || !info->vocabulary_size ||
        info->vocabulary_size > INT32_MAX || info->input.dims[0] != info->max_sequence_length ||
        info->output.dtype != BNNE_F32 || info->output.rank != 2 ||
        info->output.dims[0] != info->max_sequence_length ||
        info->output.dims[1] != info->vocabulary_size)
        return error("LLM task expects a GPT/LLaMA model with int32 [T] input and FP32 [T,V] output");
    return 1;
}

static int llm_task(BNNE_Engine engine, const BNNE_ModelInfo* info, const Args* args) {
    Reference ref = {0};
    BNNE_Tensor output = {info->output, NULL, 0};
    struct timespec start;
    int result = 1, mismatch = 0;
    uint64_t elements = (uint64_t)info->max_sequence_length * info->vocabulary_size;
    if (!llm_check(info))
        return result;
    output.data = alloc_array(elements, sizeof(float));
    if (!output.data)
        goto cleanup;
    output.bytes = (size_t)elements * sizeof(float);
    if (args->ref) {
        Comparison comparison = {0};
        if (!load_ref(args->ref, info->max_sequence_length, info->vocabulary_size, 1, &ref) ||
            !phase_begin(engine, &start) || !llm_forward(engine, ref.ids, ref.count, &output))
            goto cleanup;
        for (uint32_t i = 0; i < ref.count; ++i)
            compare_row(&comparison, (const float*)output.data + (size_t)i * info->vocabulary_size,
                        ref.logits + (size_t)i * info->vocabulary_size, info->vocabulary_size);
        mismatch = report_comparison(&comparison, ref.count, 1e-2, 1);
        if (!phase_end(engine, &start, "check", args->profile))
            goto cleanup;
    }
    if (args->eval && !llm_eval(engine, info, args->eval, &output, args->profile))
        goto cleanup;
    if (args->generate && !llm_generate(engine, info, args, &output))
        goto cleanup;
    result = mismatch;
cleanup:
    free(ref.ids);
    free(ref.logits);
    free(output.data);
    return result;
}

static int parse_args(int argc, char** argv, Args* args) {
    int steps_set = 0;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if (strcmp(option, "--profile") == 0 && !args->profile) {
            args->profile = 1;
        } else if (strcmp(option, "--kv-cache") == 0 && !args->kv_cache) {
            args->kv_cache = 1;
        } else if ((strcmp(option, "--ref") == 0 && !args->ref) ||
                   (strcmp(option, "--eval") == 0 && !args->eval) ||
                   (strcmp(option, "--prompt") == 0 && !args->prompt)) {
            if (++i == argc)
                return error("option requires a value");
            if (strcmp(option, "--ref") == 0) args->ref = argv[i];
            else if (strcmp(option, "--eval") == 0) args->eval = argv[i];
            else { args->prompt = argv[i]; args->generate = 1; }
        } else if (strcmp(option, "--steps") == 0 && !steps_set) {
            if (++i == argc)
                return error("--steps requires an unsigned integer");
            const char* text = argv[i];
            if (!*text)
                return error("invalid --steps");
            for (const char* p = text; *p; ++p)
                if (*p < '0' || *p > '9')
                    return error("invalid --steps");
            errno = 0;
            char* end;
            unsigned long value = strtoul(text, &end, 10);
            if (errno || *end || value > UINT32_MAX)
                return error("--steps out of range");
            args->steps = (uint32_t)value;
            steps_set = args->generate = 1;
        } else if (option[0] == '-' || args->count == 3) {
            return error("unknown/duplicate option or too many paths");
        } else {
            args->paths[args->count++] = option;
        }
    }
    return args->count > 0;
}

int main(int argc, char** argv) {
    Args args = {.steps = 40};
    BNNE_Engine engine = NULL;
    BNNE_Options options;
    BNNE_ModelInfo info;
    int result = 1, llm;
    if (!parse_args(argc, argv, &args))
        goto usage;
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) {
        error("IEEE-754 binary32 float is required");
        goto cleanup;
    }
    BNNE_OptionsInit(&options);
    options.enable_profile = args.profile;
    BNNE_Status status = BNNE_Open(args.paths[0], &options, &engine);
    if (!api_ok(engine, status, "BNNE_Open") ||
        !api_ok(engine, BNNE_GetModelInfo(engine, &info), "BNNE_GetModelInfo"))
        goto cleanup;
    if (info.api_version != BNNE_API_VERSION) {
        error("incompatible BNNE API version");
        goto cleanup;
    }
    llm = info.kind == BNNE_MODEL_GPT || info.kind == BNNE_MODEL_LLAMA;
    if (args.count != (llm ? 1 : 3)) {
        error(llm ? "LLM task expects only the model path; use --ref/--eval/--prompt"
                  : "classification expects model.bnn images.idx labels.idx");
        goto usage;
    }
    if (llm && !args.ref && !args.eval)
        args.generate = 1;
    if (args.profile) {
        BNNE_Stats stats;
        if (!api_ok(engine, BNNE_GetStats(engine, &stats), "BNNE_GetStats"))
            goto cleanup;
        printf("profile load: %.3f ms\n", (double)stats.load_ns / 1e6);
    }
    result = llm ? llm_task(engine, &info, &args) : classify_task(engine, &info, &args);
    goto cleanup;
usage:
    fprintf(stderr,
            "usage: %s [--profile] [--ref FILE] model.bnn images.idx labels.idx\n"
            "       %s [--profile] [--ref FILE] [--eval FILE] [--prompt TEXT] "
            "[--steps N] [--kv-cache] model.bnn\n", argv[0], argv[0]);
cleanup:
    BNNE_Close(engine);
    return result;
}
