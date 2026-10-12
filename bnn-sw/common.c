#define _POSIX_C_SOURCE 200809L
#include "internal.h"

#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

BNNE_Status bnne_error(BNNE_Engine engine, BNNE_Status status, const char* format, ...) {
    if (engine) {
        va_list args;
        va_start(args, format);
        vsnprintf(engine->error, sizeof engine->error, format, args);
        va_end(args);
    }
    return status;
}

BNNE_Status bnne_accel_status(BNNE_Engine engine, BNN_Status status, const char* operation) {
    if (status == BNN_OK)
        return BNNE_OK;
    BNNE_Status result;
    switch (status) {
    case BNN_ENOMEM: result = BNNE_ENOMEM; break;
    case BNN_ENOTSUP: result = BNNE_ENOTSUP; break;
    case BNN_ERANGE: result = BNNE_ERANGE; break;
    case BNN_EIO: case BNN_ETIMEOUT: result = BNNE_EIO; break;
    default: result = BNNE_ECOMPUTE; break;
    }
    return bnne_error(engine, result, "%s: %s", operation, BNN_StatusString(status));
}

uint64_t bnne_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

void bnne_read(BNNE_Reader* reader, void* dst, size_t bytes) {
    if (reader->status == BNNE_OK && fread(dst, 1, bytes, reader->file) != bytes)
        reader->status = ferror(reader->file) ? BNNE_EIO : BNNE_EFORMAT;
}

uint16_t bnne_read_u16(BNNE_Reader* reader) {
    uint8_t bytes[2] = {0};
    bnne_read(reader, bytes, sizeof bytes);
    return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

uint32_t bnne_read_u32(BNNE_Reader* reader) {
    uint8_t bytes[4] = {0};
    bnne_read(reader, bytes, sizeof bytes);
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

float bnne_read_f32(BNNE_Reader* reader) {
    uint32_t bits = bnne_read_u32(reader);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

float bnne_fp16_to_fp32(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    int32_t exponent = (half >> 10) & 0x1f;
    uint32_t mantissa = half & 0x3ffu;
    uint32_t bits;
    if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3ffu;
            bits = sign | ((uint32_t)(exponent + 112) << 23) | (mantissa << 13);
        }
    } else {
        bits = sign | ((uint32_t)(exponent + 112) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

BNNE_Status bnne_quantize(const float* input, uint32_t count, int8_t* output, float* scale) {
    float amax = 0.0f;
    for (uint32_t k = 0; k < count; k++) {
        if (!isfinite(input[k]))
            return BNNE_ECOMPUTE;
        if (fabsf(input[k]) > amax)
            amax = fabsf(input[k]);
    }
    *scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    if (*scale == 0.0f)
        return BNNE_ECOMPUTE;
    for (uint32_t k = 0; k < count; k++) {
        float q = roundf(input[k] / *scale);
        output[k] = (int8_t)(q > 127.0f ? 127 : (q < -127.0f ? -127 : q));
    }
    return BNNE_OK;
}
