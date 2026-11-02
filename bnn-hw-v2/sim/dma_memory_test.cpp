#include "Vbnn_dma_memory.h"
#include "simulation.hpp"
#include "virtual_memory.hpp"


#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef TEST_DMA_WIDTH
#define TEST_DMA_WIDTH 128
#endif

namespace {
using bnn::sim::memory;
using bnn::sim::kReadMemory;
using bnn::sim::kWriteMemory;
using Sim = bnn::sim::Simulation<Vbnn_dma_memory>;
static_assert(TEST_DMA_WIDTH == 64 || TEST_DMA_WIDTH == 128 || TEST_DMA_WIDTH == 256,
              "unsupported test DMA width");
constexpr unsigned kBytes = TEST_DMA_WIDTH / 8;
constexpr std::uint64_t kFullStrobe = (std::uint64_t{1} << kBytes) - 1;
constexpr std::uint64_t kBase = 0x100000000ULL;
constexpr std::size_t kSize = 16384;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Signal>
std::uint8_t byte_at(const Signal& signal, unsigned index) {
    if constexpr (std::is_integral_v<Signal>) {
        return static_cast<std::uint8_t>(signal >> (index * 8));
    } else {
        return static_cast<std::uint8_t>(signal[index / 4] >> ((index % 4) * 8));
    }
}

template <class Signal>
void set_byte(Signal& signal, unsigned index, std::uint8_t value) {
    if constexpr (std::is_integral_v<Signal>) {
        const auto shift = index * 8;
        signal = static_cast<Signal>((signal & ~(std::uint64_t{255} << shift)) |
                                     (std::uint64_t{value} << shift));
    } else {
        const auto shift = (index % 4) * 8;
        signal[index / 4] = (signal[index / 4] & ~(std::uint32_t{255} << shift)) |
                            (std::uint32_t{value} << shift);
    }
}

std::vector<std::uint8_t> read_signal(const Vbnn_dma_memory& dut) {
    std::vector<std::uint8_t> result(kBytes);
    for (unsigned b = 0; b < kBytes; ++b) result[b] = byte_at(dut.mem_rd_data, b);
    return result;
}

void initialize(Sim& sim) {
    auto& d = sim.dut;
    d.mem_rd_cmd_valid = d.mem_rd_data_ready = 0;
    d.mem_rd_cmd_addr = kBase;
    d.mem_rd_cmd_len = 1;
    d.mem_wr_cmd_valid = d.mem_wr_data_valid = d.mem_wr_data_last = 0;
    d.mem_wr_cmd_addr = kBase;
    d.mem_wr_cmd_len = 1;
    d.mem_wr_data_strb = 0;
    for (unsigned b = 0; b < kBytes; ++b) set_byte(d.mem_wr_data, b, 0);
    d.cfg_rd_latency = 2;
    d.cfg_rd_gap = 0;
    d.cfg_wr_latency = 3;
    d.cfg_cmd_stall_period = 3;
    d.cfg_wr_stall_period = 4;
    memory(kReadMemory).configure(kSize, kBase);
    memory(kWriteMemory).configure(kSize, kBase, 0xa5);
    sim.reset();
    require(!d.rd_busy && !d.wr_busy && !d.mem_wr_done && !d.mem_rd_data_valid,
            "reset did not clear DMA state");
}

template <class Predicate>
void wait_for(Sim& sim, Predicate predicate, const std::string& what) {
    sim.dut.eval();
    for (unsigned cycle = 0; cycle < 10000; ++cycle) {
        if (predicate()) return;
        sim.tick();
    }
    throw std::runtime_error("timeout: " + what);
}

void read_command(Sim& sim, std::uint64_t address, unsigned beats) {
    auto& d = sim.dut;
    d.mem_rd_cmd_addr = address;
    d.mem_rd_cmd_len = beats;
    d.mem_rd_cmd_valid = 1;
    wait_for(sim, [&] { return d.mem_rd_cmd_ready; }, "read command");
    sim.tick();
    d.mem_rd_cmd_valid = 0;
    d.eval();
}

void write_command(Sim& sim, std::uint64_t address, unsigned beats) {
    auto& d = sim.dut;
    d.mem_wr_cmd_addr = address;
    d.mem_wr_cmd_len = beats;
    d.mem_wr_cmd_valid = 1;
    wait_for(sim, [&] { return d.mem_wr_cmd_ready; }, "write command");
    sim.tick();
    d.mem_wr_cmd_valid = 0;
    d.eval();
}

void write_beat(Sim& sim, std::uint8_t value, bool last) {
    auto& d = sim.dut;
    for (unsigned b = 0; b < kBytes; ++b) set_byte(d.mem_wr_data, b, value);
    d.mem_wr_data_strb = kFullStrobe;
    d.mem_wr_data_last = last;
    d.mem_wr_data_valid = 1;
    wait_for(sim, [&] { return d.mem_wr_data_ready; }, "write data");
    sim.tick();
    d.mem_wr_data_valid = 0;
    d.eval();
}

void idle_cycles(Sim& sim, unsigned cycles) {
    auto& d = sim.dut;
    d.eval();
    for (unsigned cycle = 0; cycle <= cycles; ++cycle) {
        require(!d.rd_busy && !d.wr_busy && !d.mem_rd_data_valid &&
                !d.mem_rd_data_last && !d.mem_wr_done && !d.mem_wr_data_ready,
                "unexpected DMA activity while idle");
        if (cycle < cycles) sim.tick();
    }
}

void timing_and_boundary_tests(Sim& sim) {
    constexpr unsigned beats = 3;
    const auto address = kBase + kSize - beats * kBytes;
    for (unsigned latency : {0u, 1u, 4u}) {
        for (unsigned gap : {0u, 1u, 3u}) {
            initialize(sim);
            auto& d = sim.dut;
            d.cfg_cmd_stall_period = d.cfg_wr_stall_period = 0;
            d.cfg_rd_latency = d.cfg_wr_latency = latency;
            d.cfg_rd_gap = gap;
            d.eval();
            require(d.mem_rd_cmd_ready && d.mem_wr_cmd_ready,
                    "idle command ready depends on valid");
            idle_cycles(sim, 3);

            std::vector<std::uint8_t> source(beats * kBytes);
            for (unsigned b = 0; b < source.size(); ++b) {
                source[b] = static_cast<std::uint8_t>(b + 1);
            }
            memory(kReadMemory).write(address, source);
            read_command(sim, address, beats);
            // Zero latency still presents data on the edge AFTER command acceptance.
            for (unsigned delay = 0; delay <= latency; ++delay) {
                require(d.rd_busy && !d.mem_rd_cmd_ready && !d.mem_rd_data_valid,
                        "incorrect initial read latency");
                sim.tick();
            }
            for (unsigned beat = 0; beat < beats; ++beat) {
                const std::vector<std::uint8_t> wanted(source.begin() + beat * kBytes,
                                                       source.begin() + (beat + 1) * kBytes);
                // The receiver withholds ready until valid; the model must not deadlock.
                for (unsigned hold = 0; hold <= 3; ++hold) {
                    require(d.rd_busy && !d.mem_rd_cmd_ready && d.mem_rd_data_valid &&
                            read_signal(d) == wanted &&
                            static_cast<bool>(d.mem_rd_data_last) == (beat + 1 == beats),
                            "read valid/data/last depends on ready or changed while held");
                    if (hold < 3) sim.tick();
                }
                d.mem_rd_data_ready = 1;
                sim.tick();
                d.mem_rd_data_ready = 0;
                d.eval();
                if (beat + 1 < beats) {
                    for (unsigned delay = 0; delay < gap; ++delay) {
                        require(d.rd_busy && !d.mem_rd_data_valid && !d.mem_rd_data_last,
                                "incorrect inter-beat read gap");
                        sim.tick();
                    }
                }
            }
            idle_cycles(sim, 3);

            auto expected = memory(kWriteMemory).read(kBase, kSize);
            write_command(sim, address, beats);
            for (unsigned beat = 0; beat < beats; ++beat) {
                for (unsigned idle = 0; idle < 3; ++idle) {
                    require(d.wr_busy && !d.mem_wr_cmd_ready && !d.mem_wr_done &&
                            d.mem_wr_data_ready, "write data ready depends on valid");
                    // Inputs may change freely while data valid is low.
                    set_byte(d.mem_wr_data, 0, static_cast<std::uint8_t>(idle));
                    d.mem_wr_data_strb = kFullStrobe;
                    d.mem_wr_data_last = idle % 2;
                    sim.tick();
                    require(memory(kWriteMemory).read(kBase, kSize) == expected,
                            "write committed without valid");
                }
                for (unsigned b = 0; b < kBytes; ++b) {
                    const auto value = source[beat * kBytes + b];
                    set_byte(d.mem_wr_data, b, value);
                    if (beat != 1) expected[kSize - beats * kBytes + beat * kBytes + b] = value;
                }
                d.mem_wr_data_strb = beat == 1 ? 0 : kFullStrobe;
                d.mem_wr_data_last = beat + 1 == beats;
                d.mem_wr_data_valid = 1;
                d.eval();
                require(d.mem_wr_data_ready, "write data ready dropped when valid asserted");
                sim.tick();
                d.mem_wr_data_valid = 0;
                d.eval();
                require(memory(kWriteMemory).read(kBase, kSize) == expected,
                        "boundary write/strobe/guard mismatch");
            }
            for (unsigned delay = 0; delay <= latency; ++delay) {
                require(d.wr_busy && !d.mem_wr_done && !d.mem_wr_cmd_ready &&
                        !d.mem_wr_data_ready, "incorrect write response latency");
                sim.tick();
            }
            require(d.mem_wr_done && d.wr_busy && !d.mem_wr_cmd_ready && !d.mem_wr_data_ready,
                    "missing write completion or premature channel reuse");
            sim.tick();
            idle_cycles(sim, 5);
            require(d.mem_rd_cmd_ready && d.mem_wr_cmd_ready, "command ready did not recover");
            require(memory(kWriteMemory).read(kBase, kSize) == expected &&
                    memory(kReadMemory).read(address, source.size()) == source,
                    "idle activity or aliased read/write memory");
        }
    }
}

void native_memory_tests() {
    using bnn::sim::VirtualMemory;
    VirtualMemory local;
    local.configure(32, 0x100, 0xa5);
    local.write(0x100, {1, 2, 3, 4});
    local.write_masked(0x100, {9, 9, 9, 9}, {0, 1, 0, 1});
    require(local.read(0x100, 4) == std::vector<std::uint8_t>({1, 9, 3, 9}), "native strobe");
    unsigned failures = 0;
    const auto expect_throw = [&](auto action) {
        try { action(); } catch (const std::exception&) { ++failures; }
    };
    expect_throw([&] { local.read(0xff, 1); });
    expect_throw([&] { local.read(0x120, 1); });
    expect_throw([&] { local.check(0x110, std::numeric_limits<std::uint64_t>::max()); });
    expect_throw([&] { local.write_masked(0x100, {1, 2}, {1}); });
    expect_throw([&] { local.configure(32, std::numeric_limits<std::uint64_t>::max() - 15); });
    expect_throw([&] { memory(2); });
    require(failures == 6, "native range/overflow/id checks");
    local.configure(16, std::numeric_limits<std::uint64_t>::max() - 15);
    local.write(std::numeric_limits<std::uint64_t>::max(), {0x42});
    require(local.read(std::numeric_limits<std::uint64_t>::max(), 1)[0] == 0x42,
            "highest uint64 address");
}

void concurrent_transfer(Sim& sim, std::mt19937& random, unsigned read_beats,
                         unsigned write_beats, unsigned gap) {
    auto& d = sim.dut;
    const auto address = kBase + 256;
    std::vector<std::uint8_t> source(kSize);
    for (auto& b : source) b = static_cast<std::uint8_t>(random());
    memory(kReadMemory).write(kBase, source);
    memory(kWriteMemory).configure(kSize, kBase, 0xa5);
    auto expected = memory(kWriteMemory).read(kBase, kSize);
    d.cfg_rd_gap = gap;
    d.cfg_rd_latency = random() % 5;
    d.cfg_wr_latency = random() % 5;
    d.mem_rd_cmd_addr = d.mem_wr_cmd_addr = address;
    d.mem_rd_cmd_len = read_beats;
    d.mem_wr_cmd_len = write_beats;
    d.mem_rd_cmd_valid = d.mem_wr_cmd_valid = 1;
    d.mem_rd_data_ready = 0;
    unsigned read_count = 0, write_count = 0, completions = 0;
    bool read_accepted = false, write_accepted = false, pending_data = false;
    bool was_blocked = false;
    std::vector<std::uint8_t> held;
    bool held_last = false;
    unsigned last_write_cycle = 0;

    for (unsigned cycle = 0; cycle < 10000; ++cycle) {
        d.mem_rd_data_ready = random() % 4 != 0;
        // Once offered, a beat is kept stable until the next handshake.
        if (!pending_data && write_count < write_beats && random() % 3 != 0) {
            d.mem_wr_data_strb = 0;
            for (unsigned b = 0; b < kBytes; ++b) {
                const auto value = static_cast<std::uint8_t>(random());
                set_byte(d.mem_wr_data, b, value);
                // Exercise a 1-FP32 tail, arbitrary sparse masks and all-zero strobes.
                const bool enabled = write_count + 1 == write_beats ? b < 4 :
                                     (write_count % 5 != 2 && random() % 2 != 0);
                if (enabled) d.mem_wr_data_strb |= std::uint64_t{1} << b;
            }
            d.mem_wr_data_last = write_count + 1 == write_beats;
            d.mem_wr_data_valid = 1;
            pending_data = true;
        }
        d.eval();
        if (was_blocked) {
            require(d.mem_rd_data_valid && read_signal(d) == held &&
                    static_cast<bool>(d.mem_rd_data_last) == held_last, "read changed under backpressure");
        }
        was_blocked = d.mem_rd_data_valid && !d.mem_rd_data_ready;
        if (was_blocked) { held = read_signal(d); held_last = d.mem_rd_data_last; }
        const bool rd_cmd = d.mem_rd_cmd_valid && d.mem_rd_cmd_ready;
        const bool wr_cmd = d.mem_wr_cmd_valid && d.mem_wr_cmd_ready;
        const bool rd_data = d.mem_rd_data_valid && d.mem_rd_data_ready;
        const bool wr_data = d.mem_wr_data_valid && d.mem_wr_data_ready;
        if (rd_data) {
            require(read_accepted && read_count < read_beats, "unexpected read beat");
            const auto offset = 256 + read_count * kBytes;
            const std::vector<std::uint8_t> wanted(source.begin() + offset, source.begin() + offset + kBytes);
            require(read_signal(d) == wanted, "DPI little-endian read mismatch");
            require(static_cast<bool>(d.mem_rd_data_last) == (read_count + 1 == read_beats), "read last");
            ++read_count;
        }
        if (wr_data) {
            require(write_accepted && write_count < write_beats, "unexpected write beat");
            for (unsigned b = 0; b < kBytes; ++b) {
                if ((d.mem_wr_data_strb >> b) & 1) {
                    expected[256 + write_count * kBytes + b] = byte_at(d.mem_wr_data, b);
                }
            }
            ++write_count;
            if (write_count == write_beats) last_write_cycle = cycle;
        }
        if (d.mem_wr_done) {
            require(write_count == write_beats && cycle > last_write_cycle, "premature write completion");
            ++completions;
            require(completions == 1, "duplicate write completion");
        }
        sim.tick();
        if (rd_cmd) { read_accepted = true; d.mem_rd_cmd_valid = 0; }
        if (wr_cmd) { write_accepted = true; d.mem_wr_cmd_valid = 0; }
        if (wr_data) { pending_data = false; d.mem_wr_data_valid = 0; }
        d.eval();
        if (read_count == read_beats && completions == 1 && !d.rd_busy && !d.wr_busy) {
            idle_cycles(sim, 5);
            require(memory(kWriteMemory).read(kBase, kSize) == expected, "masked write/guard mismatch");
            require(memory(kReadMemory).read(kBase, kSize) == source, "read/write stores are not independent");
            return;
        }
    }
    throw std::runtime_error("concurrent transfer timeout");
}

void snapshot_and_reset_tests(Sim& sim) {
    initialize(sim);
    auto& d = sim.dut;
    d.cfg_rd_latency = 0;
    d.cfg_rd_gap = 0;
    memory(kReadMemory).write(kBase, std::vector<std::uint8_t>(kBytes, 0x12));
    read_command(sim, kBase, 1);
    wait_for(sim, [&] { return d.mem_rd_data_valid; }, "snapshot data");
    memory(kReadMemory).write(kBase, std::vector<std::uint8_t>(kBytes, 0x34));
    for (unsigned i = 0; i <= 8; ++i) {
        require(d.mem_rd_data_valid && d.mem_rd_data_last &&
                read_signal(d) == std::vector<std::uint8_t>(kBytes, 0x12),
                "held read is not a stable valid snapshot");
        if (i < 8) sim.tick();
    }
    d.mem_rd_data_ready = 1;
    sim.tick();
    d.mem_rd_data_ready = 0;
    read_command(sim, kBase, 1);
    wait_for(sim, [&] { return d.mem_rd_data_valid; }, "reconfigured data");
    require(read_signal(d) == std::vector<std::uint8_t>(kBytes, 0x34), "dynamic memory update not visible");
    write_command(sim, kBase, 2);
    write_beat(sim, 0x56, false);
    require(!d.mem_wr_done, "partial write completed");
    sim.reset(); // Drop the held read and remaining write; committed bytes persist.
    require(!d.rd_busy && !d.wr_busy && !d.mem_rd_data_valid && !d.mem_wr_done, "reset failed to flush");
    require(memory(kWriteMemory).read(kBase, kBytes) == std::vector<std::uint8_t>(kBytes, 0x56),
            "reset erased committed external data");
    require(memory(kWriteMemory).read(kBase + kBytes, kBytes) == std::vector<std::uint8_t>(kBytes, 0xa5),
            "reset committed unaccepted data");
    write_command(sim, kBase + 2 * kBytes, 1);
    write_beat(sim, 0x78, true);
    sim.reset(); // Also cancel a pending completion response.
    idle_cycles(sim, 10);
    require(memory(kWriteMemory).read(kBase + 2 * kBytes, kBytes) ==
                std::vector<std::uint8_t>(kBytes, 0x78), "last beat was not committed");
    std::mt19937 random(99);
    concurrent_transfer(sim, random, 2, 2, 0);
}

void negative_test(Sim& sim, const std::string& name) {
    auto& d = sim.dut;
    d.cfg_cmd_stall_period = d.cfg_wr_stall_period = 0;
    if (name == "zero-read") read_command(sim, kBase, 0);
    else if (name == "zero-write") write_command(sim, kBase, 0);
    else if (name == "unaligned-read") read_command(sim, kBase + 1, 1);
    else if (name == "unaligned-write") write_command(sim, kBase + 1, 1);
    else if (name == "range-read") read_command(sim, kBase + kSize - kBytes, 2);
    else if (name == "range-write") write_command(sim, kBase + kSize - kBytes, 2);
    else if (name == "early-last") { write_command(sim, kBase, 2); write_beat(sim, 1, true); }
    else if (name == "missing-last") { write_command(sim, kBase, 1); write_beat(sim, 1, false); }
    else if (name == "unstable-write") {
        write_command(sim, kBase, 1);
        d.cfg_wr_stall_period = 1;
        d.mem_wr_data_valid = d.mem_wr_data_last = 1;
        sim.tick();
        set_byte(d.mem_wr_data, 0, 0xab);
        sim.tick();
    } else if (name == "unstable-read-cmd" || name == "unstable-write-cmd") {
        d.cfg_cmd_stall_period = 1;
        const bool read = name == "unstable-read-cmd";
        if (read) d.mem_rd_cmd_valid = 1; else d.mem_wr_cmd_valid = 1;
        sim.tick();
        if (read) d.mem_rd_cmd_addr += kBytes; else d.mem_wr_cmd_addr += kBytes;
        sim.tick();
    } else throw std::invalid_argument("unknown negative case: " + name);
    throw std::runtime_error("negative case did not trigger RTL fatal: " + name);
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::uint32_t seed = 1;
        std::string trace_path, negative;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (++i >= argc) throw std::invalid_argument("missing argument");
            if (option == "--seed") seed = static_cast<std::uint32_t>(std::stoul(argv[i]));
            else if (option == "--trace") trace_path = argv[i];
            else if (option == "--negative") negative = argv[i];
            else throw std::invalid_argument("unknown option: " + option);
        }
        native_memory_tests();
        Sim sim(trace_path);
        initialize(sim);
        if (!negative.empty()) negative_test(sim, negative);
        timing_and_boundary_tests(sim);
        initialize(sim);
        std::mt19937 random(seed);
        for (unsigned beats : {1u, 2u, 3u, 9u, 33u}) {
            for (unsigned gap : {0u, 1u, 3u}) {
                concurrent_transfer(sim, random, beats, beats + 1, gap);
            }
        }
        snapshot_and_reset_tests(sim);
        std::cout << "PASS DMA/DPI width=" << TEST_DMA_WIDTH << " seed=" << seed
                  << " cycles=" << sim.cycles() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
