/*
Combinational bnn_fp_math regression (run via `make -C bnn-hw-v2/sim fp-test`):
  verilator --cc --exe --build --assert -Wall --top-module fp_math_test \
    ../bnn_core/bnn_fp_math.sv fp_math_test.sv fp_math_test.cpp \
    --Mdir build/fp_math_test -CFLAGS "-std=c++17 -fno-fast-math -ffp-contract=off"
  ./build/fp_math_test/Vfp_math_test
Keep normal IEEE host floating-point semantics; do not compile with fast-math.
*/
#include "Vfp_math_test.h"
#include "verilated.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using U32 = std::uint32_t;
constexpr U32 kSign = 0x80000000u;
constexpr U32 kFraction = 0x007fffffu;
constexpr U32 kInfinity = 0x7f800000u;
constexpr U32 kNaN = 0x7fc00000u;
constexpr U32 kSeed = 0x127f7a32u;
constexpr unsigned kRandomCount = 50000;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

U32 bits(float value) {
    U32 result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(U32 value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

unsigned exponent(U32 value) { return (value >> 23) & 255u; }

double flushed(U32 value) {
    // Decode through the host, flushing exponent-zero inputs but retaining sign.
    return static_cast<double>(from_bits(exponent(value) == 0 ? value & kSign : value));
}

// high + low is an unevaluated, nonoverlapping expansion. Compare without
// adding its low part back into high, which would lose distant cancellation.
int compare_expansion(double high, double low, double boundary) {
    if (high < boundary) return -1;
    if (high > boundary) return 1;
    return low < 0 ? -1 : low > 0 ? 1 : 0;
}

U32 rtz(double high, double low = 0.0) {
    if (std::isnan(high)) return kNaN;
    const bool negative = std::signbit(high);
    const U32 sign = negative ? kSign : 0;
    if (std::isinf(high)) return sign | kInfinity;
    const double magnitude = negative ? -high : high;
    const double tail = negative ? -low : low;
    if (compare_expansion(magnitude, tail, std::numeric_limits<float>::min()) < 0)
        return sign;
    // The package contract overflows to infinity at 2^128, not to max finite.
    if (compare_expansion(magnitude, tail, 0x1p128) >= 0)
        return sign | kInfinity;
    float rounded = magnitude > static_cast<double>(std::numeric_limits<float>::max())
        ? std::numeric_limits<float>::max() : static_cast<float>(magnitude);
    if (compare_expansion(magnitude, tail, static_cast<double>(rounded)) < 0)
        rounded = std::nextafter(rounded, 0.0f);
    return sign | bits(rounded);
}

U32 half_reference(std::uint16_t value) {
    const unsigned e = (value >> 10) & 31u;
    const unsigned fraction = value & 1023u;
    if (e == 31) return fraction ? kNaN : (U32{value & 0x8000u} << 16) | kInfinity;
    const double magnitude = e == 0
        ? std::ldexp(static_cast<double>(fraction), -24)
        : std::ldexp(1.0 + fraction / 1024.0, static_cast<int>(e) - 15);
    return bits(static_cast<float>((value & 0x8000u) ? -magnitude : magnitude));
}

U32 multiply_reference(U32 a, U32 b) {
    // A binary32 product has at most 48 significant bits: exact in binary64.
    return rtz(flushed(a) * flushed(b));
}

U32 add_reference(U32 a, U32 b) {
    const double x = flushed(a), y = flushed(b);
    const double sum = x + y;
    if (!std::isfinite(sum)) return rtz(sum);
    // Knuth TwoSum recovers the EXACT addition residual, even for huge exponent
    // gaps. A plain double sum would get 1.0 + (-min_normal) RTZ wrong.
    const double recovered_y = sum - x;
    const double residual = (x - (sum - recovered_y)) + (y - recovered_y);
    return rtz(sum, residual);
}

U32 divide_reference(U32 a) {
    const double x = flushed(a);
    if (!std::isfinite(x)) return rtz(x);
    const U32 sign = a & kSign;
    const double magnitude = std::abs(x);
    if (magnitude < static_cast<double>(std::numeric_limits<float>::min()) * 127.0)
        return sign;
    float candidate = static_cast<float>(magnitude / 127.0);
    // A candidate binary32 times 127 needs only 31 bits and is exact in double.
    // These comparisons remove dependence on the host division's rounding.
    if (static_cast<double>(candidate) * 127.0 > magnitude)
        candidate = std::nextafter(candidate, 0.0f);
    const float next = std::nextafter(candidate, std::numeric_limits<float>::infinity());
    if (static_cast<double>(next) * 127.0 <= magnitude) candidate = next;
    return sign | bits(candidate);
}

std::uint8_t quantize_reference(U32 value, U32 maximum) {
    if (exponent(value) == 0 || exponent(value) == 255 || (maximum & kSign) ||
        exponent(maximum) == 0 || exponent(maximum) == 255)
        return 0;
    const double magnitude = std::abs(flushed(value));
    const double denominator = flushed(maximum);
    int q = 127;
    if (magnitude < denominator) {
        const double numerator = magnitude * 127.0;
        q = static_cast<int>(numerator / denominator);
        // All products below have <=31 significant bits. Compare exactly at
        // integer thresholds instead of trusting a rounded floating quotient.
        while (q > 0 && q * denominator > numerator) --q;
        while (q < 127 && (q + 1) * denominator <= numerator) ++q;
    }
    return static_cast<std::uint8_t>((value & kSign) ? -q : q);
}

void check_reference() {
    require(sizeof(float) == sizeof(U32) && std::numeric_limits<float>::is_iec559 &&
            std::numeric_limits<float>::radix == 2 && std::numeric_limits<float>::digits == 24 &&
            std::numeric_limits<float>::max_exponent == 128 &&
            std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
            "IEEE binary32/binary64 host required");
    require(std::fegetround() == FE_TONEAREST, "host rounding must be FE_TONEAREST");
    require(half_reference(0x0001) == 0x33800000u && half_reference(0x8000) == kSign &&
            half_reference(0xfe01) == kNaN, "half reference self-check");
    require(rtz(2147483647.0) == 0x4effffffu && rtz(-2147483648.0) == 0xcf000000u,
            "integer RTZ reference self-check");
    require(add_reference(0x3f800000u, 0x34400000u) == 0x3f800001u &&
            add_reference(0x3f800000u, 0xb4400000u) == 0x3f7ffffdu &&
            add_reference(0x3f800000u, 0x80800000u) == 0x3f7fffffu,
            "exact-sum/large-gap RTZ reference self-check");
    require(add_reference(kSign, kSign) == kSign && add_reference(kSign, 0) == 0 &&
            add_reference(0x3f800000u, 0xbf800000u) == 0 &&
            add_reference(0x00800000u, 0x80800001u) == kSign &&
            multiply_reference(0x80000001u, 0x3f800000u) == kSign &&
            multiply_reference(0x00800000u, 0xbf000000u) == kSign,
            "signed-zero/cancellation/FTZ reference self-check");
    require(add_reference(0x7f7fffffu, 0x73000000u) == 0x7f7fffffu &&
            add_reference(0x7f7fffffu, 0x73800000u) == kInfinity &&
            add_reference(kInfinity, kInfinity | kSign) == kNaN &&
            multiply_reference(kInfinity, 1) == kNaN,
            "overflow/Inf/NaN reference self-check");
    require(divide_reference(0x3f800000u) == 0x3c010204u &&
            divide_reference(0x00800000u) == 0 &&
            quantize_reference(0xbf000000u, 0x3f800000u) == 193 &&
            quantize_reference(0x3f800000u, 0x3f800000u) == 127 &&
            quantize_reference(kInfinity, 0x3f800000u) == 0,
            "division/quantization reference self-check");
}

constexpr std::array<U32, 32> kEdges = {{
    0, kSign, 1, 0x80000001u, 0x007fffffu, 0x807fffffu,
    0x00800000u, 0x80800000u, 0x00800001u, 0x80800001u,
    0x00ffffffu, 0x3f000000u, 0xbf000000u, 0x3f7fffffu,
    0x3f800000u, 0xbf800000u, 0x3f800001u, 0xbf800001u,
    0x3fffffffu, 0x34400000u, 0xb4400000u, 0x73000000u,
    0x73800000u, 0x7f7fffffu, 0xff7fffffu, kInfinity,
    kInfinity | kSign, kNaN, kNaN | kSign, 0x7f800001u, 0xff800001u, 0x7fffffffu
}};

U32 random_fp(std::mt19937& rng) {
    const U32 raw = static_cast<U32>(rng());
    switch (rng() & 7u) {
        case 0: return raw & (kSign | kFraction); // signed zeros/subnormals
        case 1: return (raw & (kSign | kFraction)) | kInfinity; // NaNs/Infs
        case 2: return kEdges[rng() % kEdges.size()];
        default: return raw;
    }
}

class Tests {
public:
    explicit Tests(Vfp_math_test& dut) : dut_(dut) {
        dut_.half_bits = 0;
        dut_.int_value = 0;
        dut_.fp_a = 0;
        dut_.fp_b = 0;
        dut_.positive_max = 0;
    }

    void half(std::uint16_t value) {
        dut_.half_bits = value;
        dut_.eval();
        check(0, dut_.half_result, half_reference(value), value);
    }

    void integer(std::int32_t value) {
        dut_.int_value = static_cast<U32>(value);
        dut_.eval();
        check(1, dut_.int_result, rtz(static_cast<double>(value)), static_cast<U32>(value));
    }

    void pair(U32 a, U32 b) {
        dut_.fp_a = a;
        dut_.fp_b = b;
        dut_.eval();
        check(2, dut_.mul_result, multiply_reference(a, b), a, b);
        check(3, dut_.add_result, add_reference(a, b), a, b);
    }

    void divide(U32 a) {
        dut_.fp_a = a;
        dut_.eval();
        check(4, dut_.div127_result, divide_reference(a), a);
    }

    void quantize(U32 value, U32 maximum) {
        dut_.fp_a = value;
        dut_.positive_max = maximum;
        dut_.eval();
        check(5, dut_.quantize_result, quantize_reference(value, maximum), value, maximum);
    }

    void report() const {
        std::cout << "PASS fp_math_test seed=0x" << std::hex << kSeed << std::dec << '\n';
        for (std::size_t i = 0; i < counts_.size(); ++i)
            std::cout << "  " << names_[i] << ": " << counts_[i] << " checks\n";
    }

private:
    void check(std::size_t op, U32 actual, U32 expected, U32 a, U32 b = 0) {
        ++counts_[op];
        if (actual != expected) {
            std::ostringstream message;
            message << names_[op] << " check " << std::dec << counts_[op]
                    << std::hex << " a=0x" << a << " b=0x" << b
                    << " expected=0x" << expected << " actual=0x" << actual;
            require(false, message.str());
        }
    }

    Vfp_math_test& dut_;
    std::array<std::uint64_t, 6> counts_{};
    const std::array<const char*, 6> names_{{
        "fp16_to_fp32", "int32_to_fp32", "fp32_mul_ftz", "fp32_add_ftz",
        "fp32_div127_ftz", "quantize_int8_ftz"
    }};
};

void run(Tests& test) {
    std::mt19937 rng(kSeed);
    for (unsigned h = 0; h < 65536; ++h) test.half(static_cast<std::uint16_t>(h));
    test.integer(std::numeric_limits<std::int32_t>::min());
    test.integer(std::numeric_limits<std::int32_t>::max());
    for (unsigned bit = 0; bit < 32; ++bit) {
        for (int delta : {-1, 0, 1}) {
            for (int sign : {-1, 1}) {
                const std::int64_t value = sign * ((std::int64_t{1} << bit) + delta);
                if (value >= std::numeric_limits<std::int32_t>::min() &&
                    value <= std::numeric_limits<std::int32_t>::max())
                    test.integer(static_cast<std::int32_t>(value));
            }
        }
    }
    for (U32 a : kEdges) {
        test.divide(a & ~kSign); // div127's public input contract is positive.
        for (U32 b : kEdges) {
            test.pair(a, b);
            test.quantize(a, b); // includes invalid max guards and nonfinite samples.
        }
    }
    for (unsigned e = 1; e < 255; ++e) {
        for (U32 fraction : {0u, 1u, 0x3fffffu, kFraction}) {
            const U32 a = (e << 23) | fraction;
            test.pair(a, a | kSign);
            test.pair(a, (a - 1) | kSign);
            test.pair(a, 0x00800000u);
            test.pair(a, 0x80800000u);
            for (unsigned gap : {1u, 23u, 24u, 25u, 52u, 53u, 54u, 127u, 253u}) {
                if (e <= gap) continue;
                const U32 small = ((e - gap) << 23) | fraction;
                test.pair(a, small);
                test.pair(a, small | kSign);
                test.pair(a | kSign, small);
                test.pair(a | kSign, small | kSign);
            }
        }
    }
    for (unsigned e = 0; e < 256; ++e) {
        for (U32 fraction : {0u, 1u, 0x3fffffu, 0x7dffffu, 0x7e0000u,
                             0x7e0001u, 0x7ffffeu, kFraction})
            test.divide((e << 23) | fraction);
    }
    // Exact integer quantization thresholds and the immediately adjacent floats.
    for (int power : {-126, -120, -16, -7, 0, 24, 100, 120}) {
        const U32 maximum = bits(std::ldexp(127.0f, power));
        for (unsigned q = 0; q <= 127; ++q) {
            const float x = std::ldexp(static_cast<float>(q), power);
            for (float value : {std::nextafter(x, 0.0f), x,
                                std::nextafter(x, std::numeric_limits<float>::infinity())}) {
                test.quantize(bits(value), maximum);
                test.quantize(bits(value) ^ kSign, maximum);
            }
        }
    }
    for (unsigned e : {1u, 7u, 8u, 127u, 254u}) {
        for (unsigned gap = 0; gap <= 10 && gap < e; ++gap) {
            for (U32 fraction : {0u, 1u, 0x7dffffu, 0x7e0000u, kFraction}) {
                for (U32 max_fraction : {0u, kFraction}) {
                    const U32 value = ((e - gap) << 23) | fraction;
                    const U32 maximum = (e << 23) | max_fraction;
                    test.quantize(value, maximum);
                    test.quantize(value | kSign, maximum);
                }
            }
        }
    }
    for (unsigned i = 0; i < kRandomCount; ++i) {
        const U32 a = random_fp(rng), b = random_fp(rng);
        test.pair(a, b);
        const U32 raw = static_cast<U32>(rng());
        const std::int64_t signed_value = raw < kSign ? std::int64_t{raw}
            : std::int64_t{raw} - (std::int64_t{1} << 32);
        test.integer(static_cast<std::int32_t>(signed_value));
        test.divide(random_fp(rng) & ~kSign);
        const U32 maximum = ((1u + static_cast<U32>(rng() % 254u)) << 23) |
                            (static_cast<U32>(rng()) & kFraction);
        test.quantize(a, maximum);
        if (i % 8 == 0) {
            const unsigned q = rng() % 128u;
            const float threshold = static_cast<float>(
                static_cast<double>(from_bits(maximum)) * q / 127.0);
            for (float x : {std::nextafter(threshold, 0.0f), threshold,
                            std::nextafter(threshold, std::numeric_limits<float>::infinity())}) {
                test.quantize(bits(x), maximum);
                test.quantize(bits(x) ^ kSign, maximum);
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        check_reference();
        VerilatedContext context;
        context.commandArgs(argc, argv);
        Vfp_math_test dut{&context};
        Tests test(dut);
        run(test);
        dut.final();
        test.report();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL fp_math_test: " << error.what() << '\n';
        return 1;
    }
}
