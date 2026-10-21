/* Public BNN API only; compile as C, then link with the complete RTL archive.
 * --expect-create-failure checks BNN_DeviceOpen's forwarding of BNN_EIO.
 * --checkpoints lets the Python driver inspect flushed VCDs while devices live.
 */
#include "bnn-accelerator.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static void status_check(BNN_Status got, BNN_Status want, const char *expr, int line) {
    if (got != want) {
        fprintf(stderr, "%s:%d: %s returned %d (%s), expected %d\n",
                __FILE__, line, expr, (int)got, BNN_StatusString(got), (int)want);
        exit(1);
    }
}
#define STATUS(expr, want) status_check((expr), (want), #expr, __LINE__)
#define OK(expr) STATUS((expr), BNN_OK)

enum { K = 128, N = 2, G = 128 };

typedef struct {
    BNN_Device device;
    BNN_Matrix matrix;
    BNN_Buffer input, output;
    float scale;
} Fixture;

static int checkpoints;

static void checkpoint(const char *name) {
    if (!checkpoints) return;
    printf("CHECKPOINT %s\n", name);
    CHECK(fflush(stdout) == 0);
    CHECK(getchar() == '\n');
}

static void setup(Fixture *f, uint16_t half, float scale) {
    uint8_t weights[N * K / 8];
    uint16_t scales[N];
    int8_t input[K];
    BNN_MatrixDesc desc;
    memset(weights, 0, sizeof weights);
    memset(weights, 0xff, K / 8); /* Channel 0: +1; channel 1: -1. */
    memset(input, 1, sizeof input);
    scales[0] = scales[1] = half;
    memset(&desc, 0, sizeof desc);
    desc.in_features = K;
    desc.out_features = N;
    desc.group_size = G;
    desc.weight_bits = weights;
    desc.weight_bytes = sizeof weights;
    desc.scale_fp16 = scales;
    desc.scale_count = N;
    f->scale = scale;
    OK(BNN_MatrixCreate(f->device, &desc, &f->matrix));
    OK(BNN_BufferAlloc(f->device, sizeof input, &f->input));
    OK(BNN_BufferAlloc(f->device, N * sizeof(uint32_t), &f->output));
    OK(BNN_BufferWrite(f->input, 0, input, sizeof input));
}

static void execute(Fixture *f, BNN_Operation operation) {
    BNN_Job job;
    const uint32_t poison[N] = {UINT32_C(0x5a5a5a5a), UINT32_C(0x5a5a5a5a)};
    int8_t input[K];
    size_t k;
    memset(&job, 0, sizeof job);
    job.operation = operation;
    job.matrix = f->matrix;
    job.input = f->input;
    job.output = f->output;
    job.rows = 1;
    OK(BNN_BufferWrite(f->output, 0, poison, sizeof poison));
    OK(BNN_Execute(f->device, &job));
    if (operation == BNN_OP_DOT_GROUPS) {
        int32_t result[N];
        OK(BNN_BufferRead(f->output, 0, result, sizeof result));
        CHECK(result[0] == 128 && result[1] == -128);
    } else {
        float result[N];
        OK(BNN_BufferRead(f->output, 0, result, sizeof result));
        CHECK(result[0] == 128.0f * f->scale && result[1] == -128.0f * f->scale);
    }
    OK(BNN_BufferRead(f->input, 0, input, sizeof input));
    for (k = 0; k < sizeof input; ++k) CHECK(input[k] == 1);
}

static void teardown(Fixture *f) {
    BNN_BufferFree(f->input);
    BNN_BufferFree(f->output);
    BNN_MatrixDestroy(f->matrix);
    OK(BNN_DeviceClose(f->device));
    memset(f, 0, sizeof *f);
}

int main(int argc, char **argv) {
    Fixture first = {0}, second = {0};
    if (argc == 2 && strcmp(argv[1], "--expect-create-failure") == 0) {
        BNN_Device device = NULL;
        BNN_Status status = BNN_DeviceOpen(0, &device);
        if (status != BNN_EIO || device != NULL) {
            if (device != NULL) BNN_DeviceClose(device);
            fprintf(stderr, "trace directory: DeviceOpen returned %d (%s), "
                    "expected BNN_EIO and NULL output handle\n",
                    (int)status, BNN_StatusString(status));
            return 1;
        }
        puts("Trace create failure: BNN_DeviceOpen forwarded BNN_EIO; handle is NULL.");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--checkpoints") == 0) checkpoints = 1;
    else if (argc != 1) {
        fprintf(stderr, "Usage: %s [--expect-create-failure | --checkpoints]\n", argv[0]);
        return 2;
    }
    CHECK(sizeof(float) == sizeof(uint32_t));
    OK(BNN_DeviceOpen(0, &first.device));
    OK(BNN_DeviceOpen(0, &second.device));
    CHECK(first.device != NULL && second.device != NULL && first.device != second.device);
    checkpoint("opened");
    setup(&first, UINT16_C(0x3c00), 1.0f);
    setup(&second, UINT16_C(0x3800), 0.5f);
    execute(&first, BNN_OP_DOT_GROUPS);
    checkpoint("first-dot");
    execute(&second, BNN_OP_DOT_GROUPS);
    checkpoint("second-dot");
    execute(&first, BNN_OP_GEMV);
    checkpoint("first-gemv");
    execute(&second, BNN_OP_GEMV);
    checkpoint("second-gemv");
    execute(&first, BNN_OP_DOT_GROUPS);
    execute(&second, BNN_OP_DOT_GROUPS);
    execute(&first, BNN_OP_GEMV);
    execute(&second, BNN_OP_GEMV);
    checkpoint("repeated");
    teardown(&first);
    checkpoint("first-closed");
    execute(&second, BNN_OP_DOT_GROUPS);
    execute(&second, BNN_OP_GEMV);
    checkpoint("second-survives");
    teardown(&second);
    checkpoint("closed");
    puts("Trace public API: two independent devices, K128 N2 G128, all-one activations; "
         "DOT [128, -128], GEMV [128, -128] / [64, -64], repeated jobs and teardown OK.");
    return 0;
}
