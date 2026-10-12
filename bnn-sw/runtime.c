#include "internal.h"

BNNE_Status bnne_runtime_open(BNNE_Engine engine, BNN_Caps* caps) {
    BNNE_Status status = bnne_accel_status(engine,
        BNN_DeviceOpen(engine->options.device_index, &engine->device), "BNN_DeviceOpen");
    if (status != BNNE_OK)
        return status;
    status = bnne_accel_status(engine, BNN_DeviceGetCaps(engine->device, caps), "BNN_DeviceGetCaps");
    if (status != BNNE_OK)
        return status;
    engine->caps = *caps;
    if (caps->api_version != BNN_API_VERSION || caps->group_size == 0 || caps->max_rows == 0)
        return bnne_error(engine, BNNE_ENOTSUP, "device does not support this runtime");
    return BNNE_OK;
}

BNNE_Status bnne_runtime_buffers(BNNE_Engine engine, size_t input_bytes, size_t output_bytes) {
    BNNE_Status status = bnne_accel_status(engine,
        BNN_BufferAlloc(engine->device, input_bytes, &engine->input), "BNN_BufferAlloc(input)");
    if (status != BNNE_OK)
        return status;
    return bnne_accel_status(engine,
        BNN_BufferAlloc(engine->device, output_bytes, &engine->output), "BNN_BufferAlloc(output)");
}

static BNNE_Status execute(BNNE_Engine engine, BNN_Operation operation, BNN_Matrix matrix,
                           const int8_t* input, uint32_t rows, uint32_t in_features,
                           float* output, uint32_t out_features, BNNE_Stats* stats) {
    if (rows == 0 || rows > engine->caps.max_rows ||
        (operation == BNN_OP_GEMV && !engine->caps.supports_gemv) ||
        (operation == BNN_OP_GEMM && !engine->caps.supports_gemm))
        return bnne_error(engine, BNNE_ENOTSUP, "device does not support job shape/operation");
    size_t input_bytes = (size_t)rows * in_features;
    size_t output_bytes = (size_t)rows * out_features * sizeof(float);
    uint64_t start = engine->options.enable_profile ? bnne_now_ns() : 0;
    BNNE_Status status = bnne_accel_status(engine,
        BNN_BufferWrite(engine->input, 0, input, input_bytes), "BNN_BufferWrite");
    if (engine->options.enable_profile)
        stats->transfer_ns += bnne_now_ns() - start;
    if (status != BNNE_OK)
        return status;
    BNN_Job job = {0};
    job.operation = operation;
    job.matrix = matrix;
    job.input = engine->input;
    job.output = engine->output;
    job.rows = rows;
    start = engine->options.enable_profile ? bnne_now_ns() : 0;
    status = bnne_accel_status(engine, BNN_Execute(engine->device, &job), "BNN_Execute");
    if (engine->options.enable_profile)
        stats->execute_ns += bnne_now_ns() - start;
    if (status != BNNE_OK)
        return status;
    start = engine->options.enable_profile ? bnne_now_ns() : 0;
    status = bnne_accel_status(engine,
        BNN_BufferRead(engine->output, 0, output, output_bytes), "BNN_BufferRead");
    if (engine->options.enable_profile) {
        stats->transfer_ns += bnne_now_ns() - start;
        stats->device_calls++;
        stats->bytes_written += input_bytes;
        stats->bytes_read += output_bytes;
    }
    return status;
}

BNNE_Status bnne_runtime_gemv(BNNE_Engine engine, BNN_Matrix matrix,
                              const int8_t* input, uint32_t in_features,
                              float* output, uint32_t out_features, BNNE_Stats* stats) {
    return execute(engine, BNN_OP_GEMV, matrix, input, 1, in_features, output, out_features, stats);
}

BNNE_Status bnne_runtime_gemm(BNNE_Engine engine, BNN_Matrix matrix,
                              const int8_t* input, uint32_t rows, uint32_t in_features,
                              float* output, uint32_t out_features, BNNE_Stats* stats) {
    return execute(engine, BNN_OP_GEMM, matrix, input, rows, in_features, output, out_features, stats);
}

void bnne_runtime_close(BNNE_Engine engine) {
    BNN_BufferFree(engine->input);
    BNN_BufferFree(engine->output);
    if (engine->device)
        BNN_DeviceClose(engine->device);
}
