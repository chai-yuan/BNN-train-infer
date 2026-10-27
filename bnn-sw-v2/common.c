#define _POSIX_C_SOURCE 200809L

#include "bnne-internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static _Thread_local char bnne_error_text[192];

void bnne_error_clear(void) {
    bnne_error_text[0] = '\0';
}

BNNE_Status bnne_error(BNNE_Status status, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(bnne_error_text, sizeof bnne_error_text, format, args);
    va_end(args);
    return status;
}

const char* BNNE_LastError(void) {
    return bnne_error_text;
}

BNNE_Status bnne_accel_error(BNN_Status status, const char* operation) {
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
    return bnne_error(result, "%s: %s", operation, BNN_StatusString(status));
}

uint64_t bnne_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
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

const char* BNNE_StatusString(BNNE_Status status) {
    switch (status) {
    case BNNE_OK: return "success";
    case BNNE_EINVAL: return "invalid argument";
    case BNNE_ENOMEM: return "out of memory";
    case BNNE_ENOTSUP: return "unsupported model or device capability";
    case BNNE_ERANGE: return "size or capacity out of range";
    case BNNE_EIO: return "I/O error";
    case BNNE_EFORMAT: return "invalid or truncated model";
    case BNNE_ECOMPUTE: return "computation failed or produced non-finite values";
    default: return "unknown status";
    }
}
