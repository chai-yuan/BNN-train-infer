#ifndef BNNE_INTERNAL_H
#define BNNE_INTERNAL_H

#include <stdio.h>

#include "bnn-engine.h"
#include "../bnn-accelerator/bnn-accelerator.h"

typedef struct {
    BNNE_Status (*load)(BNNE_Engine engine, FILE* file);
    BNNE_Status (*prepare)(BNNE_Engine engine);
    BNNE_Status (*run)(BNNE_Engine engine, const BNNE_Tensor* input,
                       BNNE_Tensor* output, BNNE_Stats* stats);
    void (*destroy)(BNNE_Engine engine);
    BNNE_Status (*token)(BNNE_Engine engine, uint32_t id, const char** out_text);
} BNNE_ModelOps;

struct bnne_engine {
    BNNE_Options options;
    BNNE_ModelInfo info;
    BNNE_Stats stats;
    const BNNE_ModelOps* ops;
    void* model;
    BNN_Device device;
    BNN_Buffer input;
    BNN_Buffer output;
    BNN_Caps caps;
    char error[192];
};

typedef struct {
    FILE* file;
    BNNE_Status status;
} BNNE_Reader;

extern const BNNE_ModelOps bnne_mlp_ops;
extern const BNNE_ModelOps bnne_cnn_ops;
extern const BNNE_ModelOps bnne_gpt_ops;

BNNE_Status bnne_error(BNNE_Engine engine, BNNE_Status status, const char* format, ...);
BNNE_Status bnne_accel_status(BNNE_Engine engine, BNN_Status status, const char* operation);
uint64_t bnne_now_ns(void);
void bnne_read(BNNE_Reader* reader, void* dst, size_t bytes);
uint16_t bnne_read_u16(BNNE_Reader* reader);
uint32_t bnne_read_u32(BNNE_Reader* reader);
float bnne_read_f32(BNNE_Reader* reader);
float bnne_fp16_to_fp32(uint16_t half);
BNNE_Status bnne_quantize(const float* input, uint32_t count, int8_t* output, float* scale);
BNNE_Status bnne_runtime_open(BNNE_Engine engine, BNN_Caps* caps);
BNNE_Status bnne_runtime_buffers(BNNE_Engine engine, size_t input_bytes, size_t output_bytes);
BNNE_Status bnne_runtime_gemv(BNNE_Engine engine, BNN_Matrix matrix,
                              const int8_t* input, uint32_t in_features,
                              float* output, uint32_t out_features, BNNE_Stats* stats);
BNNE_Status bnne_runtime_gemm(BNNE_Engine engine, BNN_Matrix matrix,
                              const int8_t* input, uint32_t rows, uint32_t in_features,
                              float* output, uint32_t out_features, BNNE_Stats* stats);
void bnne_runtime_close(BNNE_Engine engine);

#endif
