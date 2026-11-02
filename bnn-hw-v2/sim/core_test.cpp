// Standalone C++17 regression: link virtual_memory.cpp, NOT main.cpp.
// Target: Vbnn_sim_top, default 128 activations / 4 output lanes / M,N <= 128.
// Compile with Verilator --trace (Simulation supports optional runtime VCD).
#include "Vbnn_sim_top.h"
#include "host_driver.hpp"
#include "simulation.hpp"
#include "virtual_memory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace bnn::sim;
constexpr unsigned kWidth = 128;
constexpr std::size_t kMemorySize = 128 * 1024;
constexpr std::uint32_t kWeights = 0x1000, kWeights2 = 0x3000;
constexpr std::uint32_t kInput = 0x6000, kInput2 = 0x10000;
// Deliberately overlap the read-store weight address: the two DPI stores differ.
constexpr std::uint32_t kOutput = kWeights, kOutput2 = 0x14000;
constexpr std::uint8_t kSentinel = 0xa5;
using IntRow = std::array<std::int8_t, kWidth>;
using FloatRow = std::array<float, kWidth>;
using Matrix = std::vector<std::vector<float>>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24, "IEEE binary32 host required");
static_assert(std::numeric_limits<double>::digits >= 53, "binary64 host required");

std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(std::uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::string hex(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

void put_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned b = 0; b < 4; ++b)
        bytes.at(offset + b) = static_cast<std::uint8_t>(value >> (8 * b));
}

std::uint32_t get_u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned b = 0; b < 4; ++b)
        value |= std::uint32_t{bytes.at(offset + b)} << (8 * b);
    return value;
}

std::uint32_t output_stride(unsigned n) { return ((n + 3) / 4) * 16; }

// Known half encodings, paired with exact host values; no float-to-half library.
struct Scale { std::uint16_t half; float value; };
constexpr Scale kScales[] = {
    {0x3c00, 1.0f}, {0xbc00, -1.0f}, {0x3800, 0.5f}, {0xb800, -0.5f},
    {0x0000, 0.0f}, {0x4000, 2.0f}, {0x3400, 0.25f}, {0x3e00, 1.5f},
    {0x3fff, 1.9990234375f}, {0x0001, 0x1p-24f}, {0x0400, 0x1p-14f},
    {0x7bff, 65504.0f}, {0xbfff, -1.9990234375f}
};

struct Weight {
    Scale scale = kScales[0];
    std::array<int, kWidth> sign{};
};

// Independent numerical reference, intentionally not a translation of RTL.
// double represents an exact product of two finite binary32 operands. A host
// conversion followed by at most one nextafter implements truncation, not RNE.
float fp32_rtz(double value) {
    require(std::isfinite(value), "reference overflow/nonfinite value");
    if (std::abs(value) < std::numeric_limits<float>::min())
        return std::copysign(0.0f, value);
    float rounded = static_cast<float>(value);
    if (std::abs(static_cast<double>(rounded)) > std::abs(value))
        rounded = std::nextafter(rounded, 0.0f);
    if (std::fpclassify(rounded) == FP_SUBNORMAL) return std::copysign(0.0f, value);
    return rounded;
}

float normal_or_zero(float value) {
    return std::fpclassify(value) == FP_NORMAL ? value : std::copysign(0.0f, value);
}

float multiply(float a, float b) {
    return fp32_rtz(static_cast<double>(normal_or_zero(a)) * normal_or_zero(b));
}

// Match exact addition followed by ONE RTZ conversion, with no operand truncation.
// The directed sums below are exact in double; this is not a full-range FP32 adder.
float add_rtz(float a, float b) {
    return fp32_rtz(static_cast<double>(normal_or_zero(a)) + normal_or_zero(b));
}

struct Quantized {
    IntRow values{};
    float scale = 0.0f;
};

Quantized quantize(const FloatRow& row) {
    Quantized result;
    float maximum = 0.0f;
    // Quantization discards exponent-0/255 samples, independently of arithmetic FTZ signs.
    for (float value : row)
        if (std::isnormal(value)) maximum = std::max(maximum, std::abs(value));
    if (maximum == 0.0f) return result;
    result.scale = fp32_rtz(static_cast<double>(maximum) / 127.0);
    for (unsigned i = 0; i < kWidth; ++i) {
        const float value = std::isnormal(row[i]) ? row[i] : 0.0f;
        // Chosen cases have exact integer/dyadic ratios, far from ambiguous
        // boundaries. Never quantize by dividing by the rounded row scale.
        const long double ratio = std::abs(static_cast<long double>(value)) * 127.0L /
                                  static_cast<long double>(maximum);
        const int magnitude = static_cast<int>(std::trunc(ratio));
        require(magnitude >= 0 && magnitude <= 127, "reference INT8 quantization range");
        result.values[i] = static_cast<std::int8_t>(value < 0 ? -magnitude : magnitude);
    }
    return result;
}

float dot(const IntRow& row, const Weight& weight, float row_scale, bool dequantize) {
    std::int32_t sum = 0;
    for (unsigned i = 0; i < kWidth; ++i) {
        require(row[i] != -128, "reference input must not contain -128");
        sum += static_cast<std::int32_t>(row[i]) * weight.sign[i];
    }
    // |sum| <= 16256, so conversion to FP32 is exact; keep the two multiplies separate.
    const float weighted = multiply(static_cast<float>(sum), weight.scale.value);
    return dequantize ? multiply(weighted, row_scale) : weighted;
}

Matrix reference(const std::vector<IntRow>& rows, const std::vector<Weight>& weights) {
    Matrix result(rows.size(), std::vector<float>(weights.size()));
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (std::size_t c = 0; c < weights.size(); ++c)
            result[r][c] = dot(rows[r], weights[c], 0.0f, false);
    return result;
}

Matrix reference(const std::vector<FloatRow>& rows, const std::vector<Weight>& weights) {
    Matrix result(rows.size(), std::vector<float>(weights.size()));
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto q = quantize(rows[r]);
        for (std::size_t c = 0; c < weights.size(); ++c)
            result[r][c] = dot(q.values, weights[c], q.scale, true);
    }
    return result;
}

Matrix accumulate(Matrix old, const Matrix& next) {
    require(old.size() == next.size(), "reference accumulation row mismatch");
    for (std::size_t r = 0; r < old.size(); ++r) {
        require(old[r].size() == next[r].size(), "reference accumulation column mismatch");
        for (std::size_t c = 0; c < old[r].size(); ++c)
            old[r][c] = add_rtz(old[r][c], next[r][c]);
    }
    return old;
}

void check_reference() {
    require(bits(multiply(16253.0f, kScales[8].value)) == 0x46fdd441u,
            "reference RTZ product self-check");
    require(bits(add_rtz(1.0f, 3.0f * 0x1p-24f)) == 0x3f800001u,
            "reference exact-sum RTZ addition self-check");
    require(bits(add_rtz(1.0f, -3.0f * 0x1p-24f)) == 0x3f7ffffdu,
            "reference exact-sum cancellation self-check");
    require(bits(fp32_rtz(-0.0)) == 0x80000000u &&
            bits(fp32_rtz(-0x1p-150)) == 0x80000000u &&
            bits(normal_or_zero(-0.0f)) == 0x80000000u &&
            bits(normal_or_zero(from_bits(0x80000001u))) == 0x80000000u,
            "reference signed-zero FTZ self-check");
    require(bits(multiply(0.0f, -1.0f)) == 0x80000000u &&
            bits(multiply(-std::numeric_limits<float>::min(), 0.5f)) == 0x80000000u &&
            bits(multiply(-0.0f, -1.0f)) == 0 &&
            bits(add_rtz(-0.0f, -0.0f)) == 0x80000000u &&
            bits(add_rtz(-0.0f, 0.0f)) == 0 && bits(add_rtz(1.0f, -1.0f)) == 0,
            "reference arithmetic zero-sign self-check");
    FloatRow row{};
    row[0] = 1.0f; row[1] = -0.5f; row[2] = 0.25f;
    const auto q = quantize(row);
    require(q.values[0] == 127 && q.values[1] == -63 && q.values[2] == 31,
            "reference dyadic quantization self-check");
    require(bits(q.scale) == 0x3c010204u, "reference 1/127 truncation self-check");
    row.fill(0.0f);
    row[0] = from_bits(0x80000001u); row[1] = -0.0f;
    row[2] = from_bits(0xff800000u); row[3] = from_bits(0xffc12345u);
    const auto ignored = quantize(row);
    require(bits(ignored.scale) == 0 &&
            std::all_of(ignored.values.begin(), ignored.values.end(), [](auto x) { return x == 0; }),
            "reference ignored samples and positive row-scale zero self-check");
    row[0] = -std::numeric_limits<float>::min();
    require(bits(quantize(row).scale) == 0, "reference row-scale FTZ must be positive zero");
    std::vector<std::uint8_t> packed(12, kSentinel);
    put_u32(packed, 4, bits(multiply(0.0f, -1.0f)));
    require(get_u32(packed, 4) == 0x80000000u &&
            std::all_of(packed.begin(), packed.begin() + 4, [](auto x) { return x == kSentinel; }) &&
            std::all_of(packed.begin() + 8, packed.end(), [](auto x) { return x == kSentinel; }),
            "reference signed-zero packing and guard self-check");
}

class CoreTest {
public:
    CoreTest(std::uint32_t seed, const std::string& trace) : sim_(trace), rng_(seed) {
        sim_.dut.cmd_valid = 0;
        sim_.dut.cmd_inst = 0;
        sim_.dut.cfg_rd_latency = 2;
        sim_.dut.cfg_rd_gap = 1;
        sim_.dut.cfg_wr_latency = 3;
        sim_.dut.cfg_cmd_stall_period = 3;
        sim_.dut.cfg_wr_stall_period = 2;
    }

    void run() {
        check_reference();
        run_case("raw-errors-and-state", [&] { invalid_commands(); });
        run_case("INT8-shapes-strides-and-masks", [&] { int8_shapes(); });
        run_case("INT8-product-truncation", [&] { product_truncation(); });
        run_case("FP32-quantization-specials-and-FTZ", [&] { fp32_rows(); });
        run_case("reload-accumulate-and-overwrite", [&] { blocks(); });
        run_case("exact-sum-RTZ-accumulation", [&] { exact_sum_accumulation(); });
        run_case("reset-in-flight-and-reload", [&] { reset_in_flight(); });
        std::cout << "PASS core regression: " << commands_ << " completed commands, "
                  << sim_.cycles() << " cycles\n";
    }

private:
    Simulation<Vbnn_sim_top> sim_;
    std::mt19937 rng_;
    std::vector<std::uint8_t> expected_output_;
    std::uint64_t commands_ = 0;

    template <class F>
    void run_case(const std::string& name, F run) {
        const auto start = sim_.cycles();
        try {
            fresh();
            run();
            check_output();
            idle();
        } catch (const std::exception& error) {
            throw std::runtime_error(name + " at cycle " + std::to_string(sim_.cycles()) +
                                     ": " + error.what());
        }
        std::cout << "PASS " << name << " (" << sim_.cycles() - start << " cycles)\n";
    }

    void tick() {
        sim_.tick();
        require(dpi_error().empty(), "DPI error: " + dpi_error());
    }

    void idle() {
        require(!sim_.dut.status_busy && !sim_.dut.rd_busy && !sim_.dut.wr_busy,
                "core/DMA not idle after terminal event");
    }

    void fresh() {
        sim_.dut.cmd_valid = 0;
        sim_.dut.cmd_inst = 0;
        sim_.reset(); // Reset both core and DMA BEFORE changing either DPI store.
        clear_dpi_error();
        idle();
        memory(kReadMemory).configure(kMemorySize, 0, 0x6d);
        memory(kWriteMemory).configure(kMemorySize, 0, kSentinel);
        expected_output_.assign(kMemorySize, kSentinel);
        timing();
    }

    void timing() {
        idle();
        auto& dut = sim_.dut;
        dut.cfg_rd_latency = 1 + rng_() % 4;
        dut.cfg_rd_gap = 1 + rng_() % 2;
        dut.cfg_wr_latency = 1 + rng_() % 5;
        dut.cfg_cmd_stall_period = 2 + rng_() % 5;
        dut.cfg_wr_stall_period = 2 + rng_() % 4;
        dut.eval();
    }

    void command(std::uint64_t instruction, CommandResult expected = CommandResult::done) {
        HostDriver driver(sim_.dut, [this] { tick(); });
        // A serial 128-unit implementation can need ~600k cycles for M=N=128.
        // Test cases keep maximum N at M=1; budgets must still allow slow loaders.
        const auto actual = driver.execute(instruction, 1000, 2000000);
        require(actual == expected, "instruction " + hex(instruction) +
                (expected == CommandResult::done ? " returned error, expected done" :
                                                   " returned done, expected error"));
        require(!(sim_.dut.cmd_done && sim_.dut.cmd_error), "done/error asserted together");
        ++commands_;
        idle();
        tick();
        require(!sim_.dut.cmd_done && !sim_.dut.cmd_error, "terminal event is not a pulse");
        idle();
    }

    void strides(std::uint32_t source, std::uint32_t destination) {
        command(Instruction::set_source_stride(source));
        command(Instruction::set_destination_stride(destination));
    }

    std::vector<Weight> weights(unsigned n, unsigned variant = 0) {
        std::vector<Weight> result(n);
        for (unsigned c = 0; c < n; ++c) {
            result[c].scale = kScales[(c + variant) % std::size(kScales)];
            for (unsigned i = 0; i < kWidth; ++i) {
                bool plus = false;
                switch (c % 5) {
                case 0: plus = true; break;
                case 1: plus = false; break;
                case 2: plus = (i % 2 == 0); break;
                case 3: plus = (i == 127); break;
                default: plus = (rng_() & 1) != 0; break;
                }
                result[c].sign[i] = (plus != (variant % 2 != 0)) ? 1 : -1;
            }
        }
        return result;
    }

    std::vector<IntRow> int_rows(unsigned m, int limit = 127) {
        std::vector<IntRow> rows(m);
        for (unsigned r = 0; r < m; ++r) {
            for (unsigned i = 0; i < kWidth; ++i) {
                int value = static_cast<int>(rng_() % (2 * limit + 1)) - limit;
                if (r % 4 == 1) value = (i % 2 == 0) ? limit : -limit;
                if (r % 4 == 2) value = 0;
                rows[r][i] = static_cast<std::int8_t>(value);
            }
            if (r % 4 != 2) {
                rows[r][0] = static_cast<std::int8_t>(limit);
                rows[r][1] = static_cast<std::int8_t>(-limit);
                rows[r][2] = 0;
            }
        }
        return rows;
    }

    void pack_weights(std::uint32_t address, const std::vector<Weight>& data) {
        idle();
        std::vector<std::uint8_t> packed(data.size() * 18, 0);
        for (std::size_t c = 0; c < data.size(); ++c) {
            packed[18 * c] = static_cast<std::uint8_t>(data[c].scale.half);
            packed[18 * c + 1] = static_cast<std::uint8_t>(data[c].scale.half >> 8);
            for (unsigned i = 0; i < kWidth; ++i) {
                require(data[c].sign[i] == 1 || data[c].sign[i] == -1, "bad weight sign");
                if (data[c].sign[i] > 0) packed[18 * c + 2 + i / 8] |= 1u << (i % 8);
            }
        }
        // No per-channel padding: 18-byte records intentionally cross DMA beats.
        memory(kReadMemory).write(address, packed);
    }

    void pack_rows(std::uint32_t address, const std::vector<IntRow>& rows,
                   std::uint32_t stride) {
        idle();
        const auto step = stride ? stride : kWidth;
        require(step >= kWidth && step % 16 == 0, "invalid INT8 test stride");
        memory(kReadMemory).write(address, std::vector<std::uint8_t>(rows.size() * step, 0x55));
        for (std::size_t r = 0; r < rows.size(); ++r) {
            std::vector<std::uint8_t> packed(kWidth);
            for (unsigned i = 0; i < kWidth; ++i)
                packed[i] = static_cast<std::uint8_t>(rows[r][i]);
            memory(kReadMemory).write(address + r * step, packed);
        }
    }

    void pack_rows(std::uint32_t address, const std::vector<FloatRow>& rows,
                   std::uint32_t stride) {
        idle();
        const auto step = stride ? stride : kWidth * 4;
        require(step >= kWidth * 4 && step % 16 == 0, "invalid FP32 test stride");
        // Poison padding with a large finite value, not another valid input row.
        memory(kReadMemory).write(address, std::vector<std::uint8_t>(rows.size() * step, 0x55));
        for (std::size_t r = 0; r < rows.size(); ++r) {
            std::vector<std::uint8_t> packed(kWidth * 4);
            for (unsigned i = 0; i < kWidth; ++i) put_u32(packed, 4 * i, bits(rows[r][i]));
            memory(kReadMemory).write(address + r * step, packed);
        }
    }

    void check_output() const {
        const auto actual = memory(kWriteMemory).read(0, kMemorySize);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected_output_[i])
                throw std::runtime_error("output/guard mismatch at " + hex(i) + ": got " +
                    hex(actual[i]) + ", expected " + hex(expected_output_[i]));
        }
    }

    void writeback(const Matrix& expected, std::uint32_t address = kOutput,
                   std::uint32_t stride = 0) {
        require(!expected.empty() && !expected[0].empty(), "empty expected matrix");
        const unsigned n = static_cast<unsigned>(expected[0].size());
        const auto step = stride ? stride : output_stride(n);
        require(step >= output_stride(n) && step % 16 == 0, "invalid output test stride");
        check_output(); // COMPUTE/LOAD/invalid instructions must not write memory1.
        const auto source_before = memory(kReadMemory).read(0, kMemorySize);
        command(Instruction::writeback(address));
        const auto actual = memory(kWriteMemory).read(0, kMemorySize);
        for (std::size_t r = 0; r < expected.size(); ++r) {
            for (unsigned c = 0; c < n; ++c) {
                const auto offset = address + r * step + 4 * c;
                const auto want = bits(expected[r][c]);
                const auto got = get_u32(actual, offset);
                // All current inputs have independently reliable exact references.
                // Compare signed zero bit-for-bit too; preserve its bits in the guard image.
                require(got == want, "row " + std::to_string(r) + " channel " +
                        std::to_string(c) + " at " + hex(offset) + ": got " + hex(got) +
                        ", expected " + hex(want));
                put_u32(expected_output_, offset, want);
            }
        }
        // Includes every tail-beat byte, inter-row padding and both outer guards.
        check_output();
        require(memory(kReadMemory).read(0, kMemorySize) == source_before,
                "writeback corrupted independent read store");
    }

    void invalid_commands() {
        const auto error = CommandResult::error;
        command(0xff, error);
        command(Instruction::compute(kInput, 1, false, true), error);
        command(Instruction::writeback(kOutput), error);
        command((std::uint64_t{kWeights} << 32) | 0x10, error); // N=0, raw encoding.
        command((std::uint64_t{kInput} << 32) | 0x20, error);   // M=0.
        command(Instruction::load_weights(kWeights, 129), error);
        command((std::uint64_t{1} << 32) | 0x01, error);
        command((std::uint64_t{17} << 32) | 0x02, error);
        command((std::uint64_t{kWeights + 1} << 32) | (1u << 16) | 0x10, error);
        strides(0, 0);
        auto w = weights(3);
        auto rows = int_rows(2);
        pack_weights(kWeights, w);
        pack_rows(kInput, rows, 0);
        command(Instruction::load_weights(kWeights, 3));
        command(Instruction::writeback(kOutput), error); // Weights alone are not a result.
        command(Instruction::compute(kInput, 2, true, true), error); // No valid result to accumulate.
        command((std::uint64_t{kInput} << 32) | (1u << 26) | (1u << 8) | 0x20, error); // Reserved flag.
        command((std::uint64_t{kInput} << 32) | (1u << 25) | 0x20, error);
        command(Instruction::compute(kInput, 129, false, true), error);
        command((std::uint64_t{kInput + 1} << 32) | (1u << 8) | (1u << 25) | 0x20, error);
        // Errors must not wedge the next valid task or silently change its strides.
        command(Instruction::compute(kInput, 2, false, true));
        command(Instruction::compute(kInput, 1, true, true), error); // M mismatch preserves prior result.
        command((std::uint64_t{kOutput + 1} << 32) | 0x30, error); // Now a valid result exists.
        writeback(reference(rows, w));
        // Reloading with a different N invalidates the buffered result for WRITEBACK and accum.
        auto w5 = weights(5, 1);
        pack_weights(kWeights2, w5);
        command(Instruction::load_weights(kWeights2, 5));
        command(Instruction::writeback(kOutput2), error);
        command(Instruction::compute(kInput, 2, true, true), error);
        command(Instruction::compute(kInput, 2, false, true));
        writeback(reference(rows, w5), kOutput2);
    }

    void int8_shapes() {
        struct Shape { unsigned n, m; bool padded; };
        const Shape shapes[] = {
            {1, 3, false}, {2, 2, true}, {3, 2, true}, {4, 1, false},
            {5, 3, true}, {7, 4, false}, {128, 1, true}, {1, 128, true}
        };
        for (const auto shape : shapes) {
            timing();
            // Include a continuous, zero-latency DMA configuration as well as stalls.
            if (shape.n == 4) {
                sim_.dut.cfg_rd_latency = 0;
                sim_.dut.cfg_rd_gap = 0;
                sim_.dut.cfg_wr_latency = 0;
                sim_.dut.eval();
            }
            const auto src = shape.padded ? 160u : 0u;
            const auto dst = shape.padded ? output_stride(shape.n) + 32 : 0u;
            strides(src, dst);
            auto w = weights(shape.n);
            auto rows = int_rows(shape.m);
            pack_weights(kWeights, w);
            pack_rows(kInput, rows, src);
            command(Instruction::load_weights(kWeights, static_cast<std::uint16_t>(shape.n)));
            command(Instruction::compute(kInput, static_cast<std::uint16_t>(shape.m), false, true));
            writeback(reference(rows, w), kOutput, dst);
        }
    }

    void product_truncation() {
        auto w = weights(3);
        w[0].scale = kScales[8];
        w[1].scale = kScales[11];
        w[2].scale = kScales[12];
        for (auto& channel : w) channel.sign.fill(1);
        std::vector<IntRow> rows(1);
        rows[0].fill(127);
        rows[0][127] = 124; // dot=16253; products tie where RNE rounds away from zero.
        pack_weights(kWeights, w);
        pack_rows(kInput, rows, 0);
        command(Instruction::load_weights(kWeights, 3));
        command(Instruction::compute(kInput, 1, false, true));
        writeback(reference(rows, w));
    }

    void fp32_rows() {
        std::vector<FloatRow> rows(6);
        const float fractions[] = {1.0f, -1.0f, 0.5f, -0.5f, 0.25f, -0.25f, 0.125f, 0.0f};
        for (unsigned i = 0; i < kWidth; ++i) {
            rows[0][i] = static_cast<float>(static_cast<int>(i % 31) - 15);
            rows[1][i] = fractions[i % std::size(fractions)];
            rows[3][i] = 0.5f * static_cast<float>(static_cast<int>(i % 17) - 8);
            rows[4][i] = -std::numeric_limits<float>::min();
            rows[5][i] = 0x1p-118f;
        }
        rows[0][0] = 127.0f; rows[0][127] = -127.0f;
        rows[3][0] = -63.5f; rows[3][127] = 31.75f; // row scale exactly 0.5.
        const std::uint32_t ignored[] = {0x00000001, 0x807fffff, 0x7fc12345,
                                         0xffc12345, 0x7f800000, 0xff800000, 0x80000000};
        for (unsigned i = 0; i < std::size(ignored); ++i) {
            rows[1][16 + i] = from_bits(ignored[i]);
            rows[2][i] = from_bits(ignored[i]); // No finite normal values: all-zero row.
        }
        auto w = weights(7);
        w[5].scale = kScales[9];  // Tiny half scale, FP32 product result underflows on row 5.
        w[6].scale = kScales[10];
        strides(544, 64);
        pack_weights(kWeights, w);
        pack_rows(kInput, rows, 544);
        command(Instruction::load_weights(kWeights, 7));
        command(Instruction::compute(kInput, 6));
        writeback(reference(rows, w), kOutput, 64);
        // Also exercise FP32's default tight 512-byte source row stride.
        strides(0, 0);
        pack_rows(kInput2, rows, 0);
        command(Instruction::compute(kInput2, 6));
        writeback(reference(rows, w), kOutput2);
    }

    void blocks() {
        auto first_w = weights(5);
        auto next_w = weights(5, 1);
        // Keep this common accumulation case exactly representable and bit checked.
        for (unsigned c = 0; c < 5; ++c) {
            first_w[c].scale = kScales[c % 4];
            next_w[c].scale = kScales[(c + 1) % 4];
        }
        auto first = int_rows(3, 8), next = int_rows(3, 7);
        strides(160, 64);
        pack_weights(kWeights, first_w);
        pack_weights(kWeights2, next_w);
        pack_rows(kInput, first, 160);
        pack_rows(kInput2, next, 160);
        command(Instruction::load_weights(kWeights, 5));
        command(Instruction::compute(kInput, 3, false, true));
        const auto a = reference(first, first_w);
        writeback(a, kOutput, 64);
        command(Instruction::load_weights(kWeights2, 5)); // Same N must retain accumulator validity.
        command(Instruction::compute(kInput2, 3, true, true));
        writeback(accumulate(a, reference(next, next_w)), kOutput, 64);
        // New overwrite task changes BOTH M and N, with no reset or SRAM clearing.
        auto replacement = int_rows(2, 5);
        auto replacement_w = weights(7, 2);
        pack_rows(kInput, replacement, 160);
        pack_weights(kWeights, replacement_w);
        command(Instruction::load_weights(kWeights, 7));
        command(Instruction::compute(kInput, 2, false, true));
        writeback(reference(replacement, replacement_w), kOutput, 64);
    }

    void exact_sum_accumulation() {
        auto a = weights(3), b = weights(3);
        for (unsigned c = 0; c < 3; ++c) {
            a[c].sign.fill(1); a[c].scale = kScales[0];
            b[c].sign.fill(c == 1 ? -1 : 1); b[c].scale = kScales[9];
        }
        b[2].scale = {0x8001, -0x1p-24f};
        std::vector<IntRow> first(1), next(1);
        first[0][0] = 1; next[0][0] = 3;
        pack_weights(kWeights, a);
        pack_weights(kWeights2, b);
        pack_rows(kInput, first, 0);
        pack_rows(kInput2, next, 0);
        command(Instruction::load_weights(kWeights, 3));
        command(Instruction::compute(kInput, 1, false, true));
        command(Instruction::load_weights(kWeights2, 3));
        command(Instruction::compute(kInput2, 1, true, true));
        writeback(accumulate(reference(first, a), reference(next, b)));
    }

    void reset_in_flight() {
        auto w = weights(7);
        auto rows = int_rows(4);
        strides(160, 64);
        pack_weights(kWeights, w);
        pack_rows(kInput, rows, 160);
        command(Instruction::load_weights(kWeights, 7));
        sim_.dut.cfg_rd_latency = 2;
        sim_.dut.cfg_rd_gap = 2;
        // Only this intentionally interrupted command bypasses execute's completion wait.
        auto& dut = sim_.dut;
        dut.cmd_inst = Instruction::compute(kInput, 4, false, true);
        dut.cmd_valid = 1;
        dut.eval();
        bool accepted = false;
        for (unsigned i = 0; i < 1000 && !accepted; ++i) {
            accepted = dut.cmd_ready;
            tick();
        }
        dut.cmd_valid = 0;
        dut.eval();
        require(accepted, "reset-test command acceptance timeout");
        for (unsigned i = 0; i < 1000 && !dut.rd_busy; ++i) {
            require(!dut.cmd_done && !dut.cmd_error, "compute terminated before reset point");
            tick();
        }
        require(dut.rd_busy && dut.status_busy, "compute did not start a read before reset");
        for (unsigned i = 0; i < 12; ++i) tick(); // Partial input burst; not merely reset while idle.
        require(dut.rd_busy && dut.status_busy && !dut.cmd_done && !dut.cmd_error,
                "reset point is not an in-flight compute/read");
        sim_.reset();
        idle();
        for (unsigned i = 0; i < 8; ++i) {
            tick();
            idle();
            require(!dut.cmd_done && !dut.cmd_error, "stale completion after reset");
        }
        check_output();
        command(Instruction::compute(kInput, 1, false, true), CommandResult::error);
        command(Instruction::writeback(kOutput), CommandResult::error);
        timing();
        auto new_w = weights(3);
        auto new_rows = int_rows(2);
        pack_weights(kWeights2, new_w);
        pack_rows(kInput2, new_rows, 0);
        // Do not set strides: reset must have restored tight input/output defaults.
        command(Instruction::load_weights(kWeights2, 3));
        command(Instruction::compute(kInput2, 2, false, true));
        writeback(reference(new_rows, new_w), kOutput2);
    }
};

std::uint32_t parse_seed(const std::string& text) {
    require(!text.empty() && text[0] != '-', "seed must be an unsigned uint32");
    std::size_t consumed = 0;
    const bool hex_prefix = text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X');
    const auto value = std::stoull(text, &consumed, hex_prefix ? 16 : 10);
    require(consumed == text.size() && value <= std::numeric_limits<std::uint32_t>::max(),
            "seed must be an unsigned uint32");
    return static_cast<std::uint32_t>(value);
}
} // namespace

int main(int argc, char** argv) {
    std::uint32_t seed = 1;
    try {
        std::string trace;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help" || option == "-h") {
                std::cout << "BNN core end-to-end regression\n"
                          << "  --seed UINT32  deterministic inputs/timing (default 1; run 1,7,42)\n"
                          << "  --trace FILE   optional VCD (build with --trace)\n";
                return 0;
            }
            require(option == "--seed" || option == "--trace", "unknown option: " + option);
            require(i + 1 < argc, "missing argument for " + option);
            const std::string value = argv[++i];
            if (option == "--seed") seed = parse_seed(value);
            else {
                require(!value.empty(), "trace path must not be empty");
                trace = value;
            }
        }
        std::cout << "Core regression seed=" << seed << '\n';
        CoreTest test(seed, trace);
        test.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL core regression seed=" << seed << ": " << error.what() << '\n';
        return 1;
    }
}
