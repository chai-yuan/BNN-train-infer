/* Public API v2 regression runner; offline fixtures come from test_mlp.py. */
#include "bnn-engine.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)
#define OK(c) CHECK((c) == BNNE_OK)

typedef struct {
    uint32_t length, inputs, outputs;
    void *input;
    float *expected;
    uint64_t device_calls, written, read, logical, executed;
} Sample;
typedef struct {
    BNNE_ModelInfo info;
    uint32_t count;
    Sample *samples;
} Fixture;

static void path_join(char *out, size_t size, const char *dir, const char *name) {
    int n = snprintf(out, size, "%s/%s", dir, name);
    CHECK(n >= 0 && (size_t)n < size);
}
static uint32_t u32(FILE *f) {
    unsigned char b[4]; CHECK(fread(b, 1, 4, f) == 4);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static uint64_t u64(FILE *f) { uint64_t lo = u32(f); return lo | ((uint64_t)u32(f) << 32); }
static float f32(FILE *f) { uint32_t bits = u32(f); float x; memcpy(&x, &bits, 4); return x; }
static BNNE_TensorDesc read_desc(FILE *f) {
    BNNE_TensorDesc d; uint32_t i;
    d.dtype = (BNNE_DType)u32(f); d.rank = u32(f);
    CHECK(d.rank > 0 && d.rank <= BNNE_MAX_RANK);
    for (i = 0; i < BNNE_MAX_RANK; ++i) d.dims[i] = u32(f);
    return d;
}
static int same_desc(const BNNE_TensorDesc *a, const BNNE_TensorDesc *b) {
    uint32_t i;
    if (a->dtype != b->dtype || a->rank != b->rank) return 0;
    for (i = 0; i < a->rank; ++i) if (a->dims[i] != b->dims[i]) return 0;
    return 1;
}
static Fixture read_fixture(const char *dir, const char *stem) {
    Fixture x; char name[128], path[4096], magic[4]; FILE *f; uint32_t i, j;
    memset(&x, 0, sizeof x);
    snprintf(name, sizeof name, "%s.bin", stem); path_join(path, sizeof path, dir, name);
    f = fopen(path, "rb"); CHECK(f != NULL);
    CHECK(fread(magic, 1, 4, f) == 4 && memcmp(magic, "TST2", 4) == 0);
    x.info.api_version = BNNE_API_VERSION;
    x.info.kind = (BNNE_ModelKind)u32(f); x.info.layer_count = u32(f);
    x.info.input = read_desc(f); x.info.output = read_desc(f);
    x.info.vocabulary_size = u32(f); x.info.max_sequence_length = u32(f);
    x.count = u32(f); CHECK(x.count > 0 && x.count <= 16);
    x.samples = calloc(x.count, sizeof *x.samples); CHECK(x.samples);
    for (i = 0; i < x.count; ++i) {
        Sample *s = &x.samples[i];
        s->length = u32(f); s->inputs = u32(f); s->outputs = u32(f);
        CHECK(s->inputs > 0 && s->inputs <= 4096 && s->outputs > 0 && s->outputs <= 4096);
        s->device_calls = u64(f); s->written = u64(f); s->read = u64(f);
        s->logical = u64(f); s->executed = u64(f);
        s->input = malloc(s->inputs * 4); s->expected = malloc(s->outputs * 4); CHECK(s->input && s->expected);
        for (j = 0; j < s->inputs; ++j) {
            if (x.info.input.dtype == BNNE_I32) ((int32_t *)s->input)[j] = (int32_t)u32(f);
            else ((float *)s->input)[j] = f32(f);
        }
        for (j = 0; j < s->outputs; ++j) s->expected[j] = f32(f);
    }
    CHECK(fgetc(f) == EOF); CHECK(fclose(f) == 0); return x;
}
static void free_fixture(Fixture *f) {
    uint32_t i; for (i = 0; i < f->count; ++i) { free(f->samples[i].input); free(f->samples[i].expected); }
    free(f->samples);
}
static BNNE_Engine open_engine(const char *dir, const char *stem, int profile) {
    BNNE_Options options; BNNE_Engine engine = NULL; char name[128], path[4096];
    BNNE_OptionsInit(&options); options.enable_profile = profile;
    snprintf(name, sizeof name, "%s.bnn", stem); path_join(path, sizeof path, dir, name);
    OK(BNNE_Open(path, &options, &engine)); CHECK(engine); return engine;
}
static void model_info(BNNE_Engine e, const Fixture *f) {
    BNNE_ModelInfo info; BNNE_TensorDesc out;
    CHECK(BNNE_GetModelInfo(e, NULL) == BNNE_EINVAL);
    CHECK(BNNE_GetOutputDesc(e, NULL, &out) == BNNE_EINVAL);
    CHECK(BNNE_GetOutputDesc(e, &f->info.input, NULL) == BNNE_EINVAL);
    CHECK(BNNE_GetStats(e, NULL) == BNNE_EINVAL);
    OK(BNNE_GetModelInfo(e, &info));
    CHECK(info.api_version == BNNE_API_VERSION && info.kind == f->info.kind);
    CHECK(info.layer_count == f->info.layer_count);
    CHECK(same_desc(&info.input, &f->info.input) && same_desc(&info.output, &f->info.output));
    CHECK(info.vocabulary_size == f->info.vocabulary_size && info.max_sequence_length == f->info.max_sequence_length);
    OK(BNNE_GetOutputDesc(e, &info.input, &out)); CHECK(same_desc(&out, &info.output));
}
static void tensors(const Fixture *f, const Sample *s, BNNE_Tensor *in, BNNE_Tensor *out) {
    in->desc = f->info.input; out->desc = f->info.output;
    if (f->info.kind == BNNE_MODEL_GPT) { in->desc.dims[0] = s->length; out->desc.dims[0] = s->length; }
    in->data = s->input; in->bytes = s->inputs * 4; out->bytes = s->outputs * 4;
}
static void run_sample(BNNE_Engine e, const Fixture *f, uint32_t index, int profile) {
    const Sample *s = &f->samples[index]; BNNE_Tensor in, out; BNNE_TensorDesc desc;
    BNNE_Stats before, after; uint32_t i;
    void *copy = malloc(s->inputs * 4); float *guard = malloc((s->outputs + 4) * 4);
    CHECK(copy && guard); memcpy(copy, s->input, s->inputs * 4);
    tensors(f, s, &in, &out); in.data = copy; out.data = guard + 2;
    for (i = 0; i < s->outputs + 4; ++i) guard[i] = -98765.25f;
    OK(BNNE_GetStats(e, &before));
    OK(BNNE_GetOutputDesc(e, &in.desc, &desc)); CHECK(same_desc(&desc, &out.desc));
    OK(BNNE_Run(e, &in, &out));
    CHECK(memcmp(copy, s->input, in.bytes) == 0);
    CHECK(guard[0] == -98765.25f && guard[1] == -98765.25f);
    CHECK(guard[s->outputs + 2] == -98765.25f && guard[s->outputs + 3] == -98765.25f);
    for (i = 0; i < s->outputs; ++i) {
        float tolerance = f->info.kind == BNNE_MODEL_GPT ? 1e-5f : 2e-6f;
        CHECK(isfinite(guard[i + 2]));
        if (fabsf(guard[i + 2] - s->expected[i]) > tolerance * (1 + fabsf(s->expected[i]))) {
            fprintf(stderr, "kind %d sample %u logit %u: %.9g expected %.9g\n", f->info.kind, index, i, guard[i+2], s->expected[i]); exit(1);
        }
    }
    OK(BNNE_GetStats(e, &after));
    if (profile) {
        CHECK(after.forward_calls == before.forward_calls + 1);
        CHECK(after.load_ns == before.load_ns);
        CHECK(after.total_ns == after.quantize_ns + after.transfer_ns + after.execute_ns + after.host_ns);
        CHECK(after.device_calls - before.device_calls == s->device_calls);
        CHECK(after.bytes_written - before.bytes_written == s->written);
        CHECK(after.bytes_read - before.bytes_read == s->read);
        CHECK(after.logical_macs - before.logical_macs == s->logical);
        CHECK(after.executed_macs - before.executed_macs == s->executed);
    } else { BNNE_Stats zero = {0}; CHECK(memcmp(&after, &zero, sizeof zero) == 0); }
    free(copy); free(guard);
}
static void check_reset(BNNE_Engine e) {
    BNNE_Stats before, after, zero = {0}; OK(BNNE_GetStats(e, &before)); OK(BNNE_ResetStats(e)); OK(BNNE_GetStats(e, &after));
    CHECK(before.load_ns == after.load_ns); after.load_ns = 0; CHECK(memcmp(&after, &zero, sizeof zero) == 0);
}
static void rejected(BNNE_Engine e, BNNE_Tensor *in, BNNE_Tensor *out, float *guard, size_t size, BNNE_Status expected) {
    BNNE_Stats before, after; void *saved = malloc(size); BNNE_Status status; CHECK(saved); memcpy(saved, guard, size);
    if (e) OK(BNNE_GetStats(e, &before));
    status = BNNE_Run(e, in, out);
    if (status != expected) { fprintf(stderr, "rejected Run status %d expected %d: %s\n", status, expected, BNNE_LastError(e)); exit(1); }
    CHECK(memcmp(saved, guard, size) == 0); free(saved);
    if (e) { OK(BNNE_GetStats(e, &after)); CHECK(memcmp(&before, &after, sizeof before) == 0); }
}
static void invalid_tensors(BNNE_Engine e, const Fixture *f) {
    const Sample *s = &f->samples[f->info.kind == BNNE_MODEL_GPT ? 1 : 0]; BNNE_Tensor in, out, bad; uint32_t i;
    void *copy = malloc(s->inputs * 4); float *guard = malloc((s->outputs + 4) * 4); BNNE_TensorDesc desc;
    CHECK(copy && guard); memcpy(copy, s->input, s->inputs * 4);
    tensors(f, s, &in, &out); in.data = copy; out.data = guard + 2;
    for (i = 0; i < s->outputs + 4; ++i) guard[i] = 31415.25f;
#define BAD_IN(change, status) do { bad = in; change; rejected(e, &bad, &out, guard, (s->outputs + 4)*4, status); } while (0)
#define BAD_OUT(change, status) do { bad = out; change; rejected(e, &in, &bad, guard, (s->outputs + 4)*4, status); } while (0)
    BAD_IN(bad.desc.dtype = (BNNE_DType)99, BNNE_EINVAL);
    BAD_IN(bad.desc.rank = 0, BNNE_EINVAL);
    BAD_IN(bad.desc.rank = BNNE_MAX_RANK + 1, BNNE_EINVAL);
    BAD_IN(bad.desc.dims[0] = 0, BNNE_EINVAL);
    BAD_IN(bad.desc.dims[0] = UINT32_MAX, BNNE_EINVAL);
    BAD_IN(bad.bytes--, BNNE_ERANGE);
    BAD_IN(bad.bytes = 0, BNNE_ERANGE);
    BAD_IN(bad.data = NULL, BNNE_EINVAL);
    BAD_IN(bad.data = (char *)copy + 1, BNNE_EINVAL);
    BAD_OUT(bad.desc.dtype = BNNE_I32, BNNE_EINVAL);
    BAD_OUT(bad.desc.rank = 0, BNNE_EINVAL);
    BAD_OUT(bad.desc.dims[0]++, BNNE_EINVAL);
    BAD_OUT(bad.bytes--, BNNE_ERANGE);
    BAD_OUT(bad.data = NULL, BNNE_EINVAL);
    BAD_OUT(bad.data = (char *)guard + 1, BNNE_EINVAL);
    rejected(e, NULL, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
    rejected(e, &in, NULL, guard, (s->outputs+4)*4, BNNE_EINVAL);
    rejected(NULL, &in, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
    bad = in; bad.desc.rank = f->info.input.rank == 1 ? 2 : 1;
    CHECK(BNNE_GetOutputDesc(e, &bad.desc, &desc) == BNNE_EINVAL);
    if (f->info.kind == BNNE_MODEL_GPT) {
        BAD_IN(bad.desc.dims[0] = f->info.max_sequence_length + 1, BNNE_EINVAL);
        ((int32_t *)copy)[0] = -1; rejected(e, &in, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
        memcpy(copy, s->input, s->inputs*4);
        ((int32_t *)copy)[s->inputs-1] = (int32_t)f->info.vocabulary_size;
        rejected(e, &in, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
        ((int32_t *)copy)[s->inputs-1] = INT32_MAX;
        rejected(e, &in, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
    } else {
        if (f->info.input.rank == 3) BAD_IN(bad.desc.dims[2]--, BNNE_EINVAL);
        ((float *)copy)[0] = NAN; rejected(e, &in, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
        memcpy(copy, s->input, s->inputs*4);
        ((float *)copy)[s->inputs-1] = INFINITY; rejected(e, &in, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
        ((float *)copy)[s->inputs-1] = -INFINITY; rejected(e, &in, &out, guard, (s->outputs+4)*4, BNNE_EINVAL);
    }
#undef BAD_IN
#undef BAD_OUT
    free(copy); free(guard);
}
static void token_test(BNNE_Engine e, const Fixture *f) {
    static const char *words[] = {"春", "天", "春天", "a", "word", "!"};
    const char *text = NULL, *borrowed = NULL; BNNE_Stats before, after; uint32_t i;
    OK(BNNE_GetStats(e, &before));
    if (f->info.kind == BNNE_MODEL_GPT) {
        CHECK(f->info.vocabulary_size == sizeof words/sizeof words[0]);
        for (i = 0; i < f->info.vocabulary_size; ++i) { OK(BNNE_GetToken(e, i, &text)); CHECK(text && strcmp(text, words[i]) == 0); }
        OK(BNNE_GetToken(e, 4, &borrowed));
        CHECK(BNNE_GetToken(e, f->info.vocabulary_size, &text) != BNNE_OK);
        CHECK(BNNE_GetToken(e, UINT32_MAX, &text) != BNNE_OK);
        CHECK(strcmp(borrowed, "word") == 0);
    } else CHECK(BNNE_GetToken(e, 0, &text) == BNNE_ENOTSUP);
    CHECK(BNNE_GetToken(e, 0, NULL) == BNNE_EINVAL);
    OK(BNNE_GetStats(e, &after)); CHECK(memcmp(&before, &after, sizeof before) == 0);
}
static void failure_test(const char *dir, const Fixture *f) {
    char path[4096], name[256]; FILE *list; int expected, repeat; BNNE_Stats before, after;
    BNNE_Engine e, survivor = open_engine(dir, "small-a", 1);
    OK(BNNE_GetStats(survivor, &before));
    path_join(path, sizeof path, dir, "bad-models.txt"); list = fopen(path, "r"); CHECK(list);
    while (fscanf(list, "%255s %d", name, &expected) == 2) {
        path_join(path, sizeof path, dir, name);
        for (repeat = 0; repeat < 2; ++repeat) {
            e = survivor; /* Failure clears the output slot; survivor remains separately owned. */
            BNNE_Status status = BNNE_Open(path, NULL, &e);
            if (status != (BNNE_Status)expected) { fprintf(stderr, "%s: Open %d expected %d\n", name, status, expected); exit(1); }
            CHECK(e == NULL);
        }
    }
    CHECK(feof(list)); CHECK(fclose(list) == 0);
    OK(BNNE_GetStats(survivor, &after)); CHECK(memcmp(&before, &after, sizeof before) == 0);
    run_sample(survivor, f, 0, 1); BNNE_Close(survivor);
    e = open_engine(dir, "small-a", 1); run_sample(e, f, 0, 1); BNNE_Close(e);
    e = NULL; CHECK(BNNE_Open(NULL, NULL, &e) == BNNE_EINVAL && e == NULL);
    path_join(path, sizeof path, dir, "small-a.bnn"); CHECK(BNNE_Open(path, NULL, NULL) == BNNE_EINVAL);
    { BNNE_Options opts; BNNE_OptionsInit(&opts); opts.device_index = UINT32_MAX;
      CHECK(BNNE_Open(path, &opts, &e) != BNNE_OK && e == NULL); }
}
int main(int argc, char **argv) {
    static const char *stems[] = {"small-a", "small-b", "mnist", "deep", "cnn", "cnn-export", "gpt"};
    Fixture f[7]; BNNE_Engine handles[7]; BNNE_Options defaults; size_t i; uint32_t j;
    CHECK(argc == 2); BNNE_OptionsInit(&defaults); BNNE_OptionsInit(NULL);
    CHECK(defaults.device_index == 0 && defaults.enable_profile != 0);
    CHECK(BNNE_GetModelInfo(NULL, NULL) == BNNE_EINVAL);
    CHECK(BNNE_GetOutputDesc(NULL, NULL, NULL) == BNNE_EINVAL);
    CHECK(BNNE_GetStats(NULL, NULL) == BNNE_EINVAL && BNNE_ResetStats(NULL) == BNNE_EINVAL);
    CHECK(BNNE_GetToken(NULL, 0, NULL) == BNNE_EINVAL);
    CHECK(BNNE_LastError(NULL) && BNNE_StatusString(BNNE_OK) && BNNE_StatusString((BNNE_Status)99));
    for (i = 0; i < 7; ++i) {
        f[i] = read_fixture(argv[1], stems[i]); handles[i] = open_engine(argv[1], stems[i], 1);
        model_info(handles[i], &f[i]); token_test(handles[i], &f[i]);
        for (j = 0; j < f[i].count; ++j) run_sample(handles[i], &f[i], j, 1);
        invalid_tensors(handles[i], &f[i]); run_sample(handles[i], &f[i], f[i].count-1, 1);
        check_reset(handles[i]);
    }
    /* Interleave different kinds/dimensions; reset and close one must leave others intact. */
    for (j = 0; j < 2; ++j) for (i = 0; i < 7; ++i) run_sample(handles[i], &f[i], j % f[i].count, 1);
    for (i = 0; i < 7; ++i) {
        BNNE_Close(handles[i]);
        if (i+1 < 7) run_sample(handles[i+1], &f[i+1], 0, 1);
        handles[i] = open_engine(argv[1], stems[i], 0);
        run_sample(handles[i], &f[i], 0, 0); invalid_tensors(handles[i], &f[i]);
        run_sample(handles[i], &f[i], f[i].count-1, 0); token_test(handles[i], &f[i]); check_reset(handles[i]); BNNE_Close(handles[i]);
    }
    failure_test(argv[1], &f[0]); BNNE_Close(NULL);
    for (i = 0; i < 7; ++i) free_fixture(&f[i]);
    puts("BNNE v2 API: MLP/CNN/GPT oracle, dynamic shapes, tokens, isolation, validation, stats and failed loads OK");
    return 0;
}
