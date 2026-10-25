/* bnn-infer-v2: v2 引擎的命令行接口。
 *
 *   bnn-infer-v2 llm MODEL.gguf [--prompt TEXT | --ids FILE]
 *                                  [--steps N] [--ctx N] [--chat]
 *   bnn-infer-v2 classify MODEL.gguf --input FILE [--labels FILE]
 *                                     [--batch N] [--verbose]
 */
#define _POSIX_C_SOURCE 200809L

#include "bnn-sw-v2/bnn-engine.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_STEPS 40u
#define DEFAULT_CTX 1024u
#define DEFAULT_PROMPT "春天"

typedef struct {
    const char* model;
    const char* prompt;
    const char* ids;
    uint32_t steps;
    uint32_t ctx;
    int chat;
} LlmArgs;

static void usage(const char* program) {
    fprintf(stderr,
            "usage: %s llm MODEL.gguf [--prompt TEXT | --ids FILE]\n"
            "       %*s [--steps N] [--ctx N] [--chat]\n"
            "       %s classify MODEL.gguf --input FILE [--labels FILE]\n"
            "       %*s [--batch N] [--verbose]\n",
            program, (int)strlen(program), "", program, (int)strlen(program), "");
}

static int report_api_error(BNNE_Status status, const char* operation) {
    const char* detail = BNNE_LastError();
    fprintf(stderr, "error: %s: %s%s%s\n", operation, BNNE_StatusString(status),
            detail && *detail ? ": " : "", detail ? detail : "");
    return 0;
}

static uint32_t argmax(const float* logits, uint32_t width) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < width; ++i)
        if (logits[i] > logits[best])
            best = i;
    return best;
}

static int parse_u32(const char* text, uint32_t* out) {
    if (!text || !*text)
        return 0;
    for (const char* p = text; *p; ++p)
        if (*p < '0' || *p > '9')
            return 0;
    errno = 0;
    char* end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || *end || value > UINT32_MAX)
        return 0;
    *out = (uint32_t)value;
    return 1;
}

static int parse_llm_args(int argc, char** argv, LlmArgs* args) {
    memset(args, 0, sizeof *args);
    args->steps = DEFAULT_STEPS;
    args->ctx = DEFAULT_CTX;
    int prompt_set = 0, ids_set = 0, steps_set = 0, ctx_set = 0;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--chat") == 0) {
            if (args->chat)
                return 0;
            args->chat = 1;
        } else if (strcmp(arg, "--prompt") == 0 || strcmp(arg, "--ids") == 0 ||
                   strcmp(arg, "--steps") == 0 || strcmp(arg, "--ctx") == 0) {
            if (++i == argc)
                return 0;
            if (strcmp(arg, "--prompt") == 0) {
                if (prompt_set) return 0;
                args->prompt = argv[i];
                prompt_set = 1;
            } else if (strcmp(arg, "--ids") == 0) {
                if (ids_set) return 0;
                args->ids = argv[i];
                ids_set = 1;
            } else if (strcmp(arg, "--steps") == 0) {
                if (steps_set || !parse_u32(argv[i], &args->steps)) return 0;
                steps_set = 1;
            } else {
                if (ctx_set || !parse_u32(argv[i], &args->ctx) || args->ctx == 0) return 0;
                ctx_set = 1;
            }
        } else if (arg[0] == '-' || args->model) {
            return 0;
        } else {
            args->model = arg;
        }
    }

    if (!args->model || (args->prompt && args->ids))
        return 0;
    return 1;
}

static int build_chat_prompt(const char* prompt, char** out) {
    const char* prefix = "<|im_start|>user\n";
    const char* suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    size_t size = strlen(prefix) + strlen(prompt) + strlen(suffix) + 1;
    if (size < strlen(prompt))
        return 0;
    char* text = malloc(size);
    if (!text)
        return 0;
    snprintf(text, size, "%s%s%s", prefix, prompt, suffix);
    *out = text;
    return 1;
}

static int load_ids(const char* path, int32_t* ids, uint32_t capacity, uint32_t vocabulary,
                    uint32_t* out_count) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "error: %s: cannot open token file\n", path);
        return 0;
    }
    uint32_t count = 0;
    long value = 0;
    while (fscanf(file, "%ld", &value) == 1) {
        if (value < 0 || value >= (long)vocabulary || count == capacity) {
            fprintf(stderr, "error: %s: invalid or too many token IDs\n", path);
            fclose(file);
            return 0;
        }
        ids[count++] = (int32_t)value;
    }
    int ch = 0;
    int trailing = 0;
    while ((ch = fgetc(file)) != EOF) {
        if (!isspace((unsigned char)ch)) {
            trailing = 1;
            break;
        }
    }
    int failed = ferror(file) || trailing || count == 0;
    fclose(file);
    if (failed) {
        fprintf(stderr, "error: %s: invalid token file\n", path);
        return 0;
    }
    *out_count = count;
    return 1;
}

static int llm_command(const char* program, int argc, char** argv) {
    LlmArgs args;
    if (!parse_llm_args(argc, argv, &args)) {
        usage(program);
        return 1;
    }

    BNN_Device device = NULL;
    BNNE_Model model = NULL;
    BNNE_Context context = NULL;
    int32_t* ids = NULL;
    float* logits = NULL;
    char* chat = NULL;
    int result = 1;

    BNN_Status device_status = BNN_DeviceOpen(0, &device);
    if (device_status != BNN_OK) {
        fprintf(stderr, "error: BNN_DeviceOpen: %s\n", BNN_StatusString(device_status));
        goto cleanup;
    }

    BNNE_Status status = BNNE_ModelLoad(device, args.model, &model);
    if (status != BNNE_OK) {
        report_api_error(status, "BNNE_ModelLoad");
        goto cleanup;
    }

    BNNE_ModelInfo info;
    status = BNNE_ModelGetInfo(model, &info);
    if (status != BNNE_OK) {
        report_api_error(status, "BNNE_ModelGetInfo");
        goto cleanup;
    }
    if (info.task != BNNE_TASK_LLM) {
        fprintf(stderr, "error: '%s' is not a language model\n", args.model);
        goto cleanup;
    }

    BNNE_ContextParams params;
    BNNE_ContextParamsInit(&params);
    params.n_ctx = args.ctx;
    params.enable_stats = 0;
    status = BNNE_ContextCreate(model, &params, &context);
    if (status != BNNE_OK) {
        report_api_error(status, "BNNE_ContextCreate");
        goto cleanup;
    }

    ids = malloc((size_t)args.ctx * sizeof *ids);
    if (!ids) {
        fprintf(stderr, "error: out of memory\n");
        goto cleanup;
    }

    const char* text = args.prompt ? args.prompt : DEFAULT_PROMPT;
    if (args.chat) {
        if (!build_chat_prompt(text, &chat)) {
            fprintf(stderr, "error: out of memory\n");
            goto cleanup;
        }
        text = chat;
    }

    uint32_t count = 0;
    if (args.ids) {
        if (!load_ids(args.ids, ids, args.ctx, info.n_vocab, &count))
            goto cleanup;
    } else {
        status = BNNE_Tokenize(model, text, 1, ids, args.ctx, &count);
        if (status != BNNE_OK) {
            report_api_error(status, "BNNE_Tokenize");
            goto cleanup;
        }
        if (count == 0) {
            fprintf(stderr, "error: prompt encoded to zero tokens\n");
            goto cleanup;
        }
    }
    if (count >= args.ctx && args.steps > 0) {
        fprintf(stderr, "error: prompt leaves no room for generation\n");
        goto cleanup;
    }

    logits = malloc((size_t)info.n_vocab * sizeof *logits);
    if (!logits) {
        fprintf(stderr, "error: out of memory\n");
        goto cleanup;
    }

    status = BNNE_Decode(context, ids, count, logits);
    if (status != BNNE_OK) {
        report_api_error(status, "BNNE_Decode(prefill)");
        goto cleanup;
    }

    int32_t next = (int32_t)argmax(logits, info.n_vocab);
    for (uint32_t step = 0; step < args.steps; ++step) {
        if (BNNE_TokenIsEog(model, next))
            break;
        const char* piece = NULL;
        size_t piece_len = 0;
        status = BNNE_TokenPiece(model, next, &piece, &piece_len);
        if (status != BNNE_OK) {
            report_api_error(status, "BNNE_TokenPiece");
            goto cleanup;
        }
        fwrite(piece, 1, piece_len, stdout);
        fflush(stdout);

        if (step + 1 == args.steps || BNNE_ContextPos(context) >= params.n_ctx)
            break;
        status = BNNE_Decode(context, &next, 1, logits);
        if (status != BNNE_OK) {
            report_api_error(status, "BNNE_Decode");
            goto cleanup;
        }
        next = (int32_t)argmax(logits, info.n_vocab);
    }
    putchar('\n');
    result = 0;

cleanup:
    free(logits);
    free(ids);
    free(chat);
    BNNE_ContextFree(context);
    BNNE_ModelFree(model);
    if (device && BNN_DeviceClose(device) != BNN_OK)
        fprintf(stderr, "warning: device is still busy\n");
    return result;
}

typedef struct {
    const char* model;
    const char* input;
    const char* labels;
    uint32_t batch;
    int verbose;
} ClassifyArgs;

static int parse_classify_args(int argc, char** argv, ClassifyArgs* args) {
    memset(args, 0, sizeof *args);
    args->batch = 64;
    int input_set = 0, labels_set = 0, batch_set = 0;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--verbose") == 0) {
            if (args->verbose)
                return 0;
            args->verbose = 1;
        } else if (strcmp(arg, "--input") == 0 || strcmp(arg, "--labels") == 0 ||
                   strcmp(arg, "--batch") == 0) {
            if (++i == argc)
                return 0;
            if (strcmp(arg, "--input") == 0) {
                if (input_set) return 0;
                args->input = argv[i];
                input_set = 1;
            } else if (strcmp(arg, "--labels") == 0) {
                if (labels_set) return 0;
                args->labels = argv[i];
                labels_set = 1;
            } else {
                if (batch_set || !parse_u32(argv[i], &args->batch) || args->batch == 0)
                    return 0;
                batch_set = 1;
            }
        } else if (arg[0] == '-' || args->model) {
            return 0;
        } else {
            args->model = arg;
        }
    }
    return args->model && args->input;
}

static int read_file(const char* path, void** out, size_t* out_size) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "error: %s: cannot open file\n", path);
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "error: %s: cannot seek\n", path);
        fclose(file);
        return 0;
    }
    long length = ftell(file);
    if (length <= 0 || (uint64_t)length > SIZE_MAX) {
        fprintf(stderr, "error: %s: invalid file size\n", path);
        fclose(file);
        return 0;
    }
    rewind(file);
    void* data = malloc((size_t)length);
    if (!data) {
        fprintf(stderr, "error: out of memory\n");
        fclose(file);
        return 0;
    }
    if (fread(data, 1, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "error: %s: read failed\n", path);
        free(data);
        fclose(file);
        return 0;
    }
    fclose(file);
    *out = data;
    *out_size = (size_t)length;
    return 1;
}

static int classify_command(const char* program, int argc, char** argv) {
    ClassifyArgs args;
    if (!parse_classify_args(argc, argv, &args)) {
        usage(program);
        return 1;
    }

    BNN_Device device = NULL;
    BNNE_Model model = NULL;
    BNNE_Context context = NULL;
    void* input_data = NULL;
    void* label_data = NULL;
    float* logits = NULL;
    uint64_t* counts = NULL;
    int result = 1;

    BNN_Status device_status = BNN_DeviceOpen(0, &device);
    if (device_status != BNN_OK) {
        fprintf(stderr, "error: BNN_DeviceOpen: %s\n", BNN_StatusString(device_status));
        goto cleanup;
    }

    BNNE_Status status = BNNE_ModelLoad(device, args.model, &model);
    if (status != BNNE_OK) {
        report_api_error(status, "BNNE_ModelLoad");
        goto cleanup;
    }

    BNNE_ModelInfo info;
    status = BNNE_ModelGetInfo(model, &info);
    if (status != BNNE_OK) {
        report_api_error(status, "BNNE_ModelGetInfo");
        goto cleanup;
    }
    if (info.task != BNNE_TASK_CLASSIFY) {
        fprintf(stderr, "error: '%s' is not a classification model\n", args.model);
        goto cleanup;
    }
    if (!info.input_size || !info.n_classes) {
        fprintf(stderr, "error: model has an invalid input/class shape\n");
        goto cleanup;
    }

    size_t input_bytes = 0;
    if (!read_file(args.input, &input_data, &input_bytes))
        goto cleanup;
    uint64_t sample_bytes = (uint64_t)info.input_size * sizeof(float);
    if (input_bytes % sample_bytes != 0) {
        fprintf(stderr, "error: input file size is not a multiple of %" PRIu64 " bytes\n",
                sample_bytes);
        goto cleanup;
    }
    uint64_t sample_count = input_bytes / sample_bytes;
    if (sample_count == 0 || sample_count > UINT32_MAX) {
        fprintf(stderr, "error: invalid sample count\n");
        goto cleanup;
    }
    uint32_t n_samples = (uint32_t)sample_count;

    const uint8_t* labels = NULL;
    if (args.labels) {
        size_t label_bytes = 0;
        if (!read_file(args.labels, &label_data, &label_bytes))
            goto cleanup;
        if (label_bytes != n_samples) {
            fprintf(stderr, "error: labels file must contain %" PRIu32 " bytes\n", n_samples);
            goto cleanup;
        }
        labels = label_data;
        for (uint32_t i = 0; i < n_samples; i++) {
            if (labels[i] >= info.n_classes) {
                fprintf(stderr, "error: label %u outside class range\n", (unsigned)labels[i]);
                goto cleanup;
            }
        }
    }

    BNNE_ContextParams params;
    BNNE_ContextParamsInit(&params);
    params.n_batch = args.batch < n_samples ? args.batch : n_samples;
    params.enable_stats = 0;
    status = BNNE_ContextCreate(model, &params, &context);
    if (status != BNNE_OK) {
        report_api_error(status, "BNNE_ContextCreate");
        goto cleanup;
    }

    uint32_t batch = params.n_batch;
    if ((uint64_t)batch * info.n_classes > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "error: logits buffer is too large\n");
        goto cleanup;
    }
    logits = malloc((size_t)batch * info.n_classes * sizeof(float));
    counts = calloc(info.n_classes, sizeof *counts);
    if (!logits || !counts) {
        fprintf(stderr, "error: out of memory\n");
        goto cleanup;
    }

    uint64_t correct = 0;
    for (uint32_t offset = 0; offset < n_samples; offset += batch) {
        uint32_t count = n_samples - offset < batch ? n_samples - offset : batch;
        status = BNNE_Classify(context, (const float*)input_data +
                                             (size_t)offset * info.input_size,
                               count, logits);
        if (status != BNNE_OK) {
            report_api_error(status, "BNNE_Classify");
            goto cleanup;
        }
        for (uint32_t i = 0; i < count; i++) {
            const float* row = logits + (size_t)i * info.n_classes;
            uint32_t predicted = argmax(row, info.n_classes);
            counts[predicted]++;
            if (labels && predicted == labels[offset + i])
                correct++;
            if (args.verbose) {
                if (labels)
                    printf("sample %" PRIu32 ": class %" PRIu32 " label %u\n", offset + i,
                           predicted, (unsigned)labels[offset + i]);
                else
                    printf("sample %" PRIu32 ": class %" PRIu32 "\n", offset + i, predicted);
            }
        }
    }

    if (labels)
        printf("accuracy: %" PRIu64 "/%" PRIu32 " (%.4f)\n", correct, n_samples,
               (double)correct / (double)n_samples);
    else
        printf("samples: %" PRIu32 "\n", n_samples);
    printf("class counts:");
    for (uint32_t c = 0; c < info.n_classes; c++)
        printf(" %" PRIu64, counts[c]);
    putchar('\n');
    result = 0;

cleanup:
    free(counts);
    free(logits);
    free(label_data);
    free(input_data);
    BNNE_ContextFree(context);
    BNNE_ModelFree(model);
    if (device && BNN_DeviceClose(device) != BNN_OK)
        fprintf(stderr, "warning: device is still busy\n");
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "llm") == 0)
        return llm_command(argv[0], argc - 1, argv + 1);
    if (strcmp(argv[1], "classify") == 0)
        return classify_command(argv[0], argc - 1, argv + 1);
    usage(argv[0]);
    return 1;
}
