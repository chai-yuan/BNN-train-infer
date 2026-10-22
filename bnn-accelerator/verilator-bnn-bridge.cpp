#include "verilator-bnn-bridge.h"

#include "Vbnn_core_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
// Verilator also keeps a thread-local current context for runtime helpers.
class ContextScope {
    VerilatedContext* previous;
public:
    explicit ContextScope(VerilatedContext* current = nullptr)
        : previous(Verilated::threadContextp()) {
        if (current != nullptr) Verilated::threadContextp(current);
    }
    ~ContextScope() { Verilated::threadContextp(previous); }
};
}

struct bnn_rtl_sim {
    VerilatedContext context;
    std::unique_ptr<Vbnn_core_top> core;
    std::unique_ptr<VerilatedVcdC> trace;

    bnn_rtl_sim() {
        context.threads(1);
        // Trace registration must happen before the model's first evaluation.
        context.traceEverOn(true);
        core.reset(new Vbnn_core_top(&context));
    }

    ~bnn_rtl_sim() {
        core->final();
        if (trace && trace->isOpen()) {
            trace->dump(context.time());
            trace->close();
        }
    }

    bool open_trace() {
        const char* directory = std::getenv(BNN_RTL_TRACE_DIR_ENV);
        if (directory == nullptr || directory[0] == '\0') return true;
        std::string pattern = std::string(directory) + "/bnn-core-XXXXXX.vcd";
        std::vector<char> path(pattern.begin(), pattern.end());
        path.push_back('\0');
        const int fd = mkstemps(path.data(), 4);
        if (fd < 0) return false;
        close(fd);
        try {
            trace.reset(new VerilatedVcdC);
            core->trace(trace.get(), 99);
            trace->open(path.data());
            if (trace->isOpen()) return true;
        } catch (...) {
            unlink(path.data());
            throw;
        }
        unlink(path.data());
        return false;
    }

    void flush_trace() {
        if (trace) trace->flush();
    }

    void tick() {
        // eval() has already settled low-phase input changes before sampling.
        if (trace) trace->dump(context.time());
        context.timeInc(5000);
        core->clock = 1;
        core->eval();
        if (trace) trace->dump(context.time());
        context.timeInc(5000);
        core->clock = 0;
        core->eval();
    }

    void reset() {
        core->clock = 0;
        core->reset = 1;
        core->cmd_valid = 0;
        core->cmd_k_blocks = 0;
        core->cmd_group_div = 0;
        core->cmd_out_chans = 0;
        core->in_valid = 0;
        core->weight_data = 0;
        for (unsigned i = 0; i < 8; ++i) core->act_data[i] = 0;
        core->res_ready = 0;
        core->eval();
        tick();
        tick();
        core->reset = 0;
        core->eval();
    }
};

static BNN_Status recover(BNN_RtlSim* sim, BNN_Status status) noexcept {
    try {
        ContextScope scope(&sim->context);
        sim->reset();
        sim->flush_trace();
    } catch (...) {
        return BNN_EIO;
    }
    return status;
}

extern "C" BNN_Status bnn_rtl_create(BNN_RtlSim** out_sim) {
    if (out_sim == nullptr) return BNN_EINVAL;
    *out_sim = nullptr;
    try {
        ContextScope scope;
        std::unique_ptr<BNN_RtlSim> sim(new BNN_RtlSim);
        if (!sim->open_trace()) return BNN_EIO;
        sim->reset();
        sim->flush_trace();
        *out_sim = sim.release();
        return BNN_OK;
    } catch (const std::bad_alloc&) {
        return BNN_ENOMEM;
    } catch (...) {
        return BNN_EIO;
    }
}

extern "C" void bnn_rtl_destroy(BNN_RtlSim* sim) {
    if (sim == nullptr) return;
    ContextScope scope(&sim->context);
    delete sim;
}

extern "C" BNN_Status bnn_rtl_dot_groups(BNN_RtlSim* sim, const int8_t* input,
                                         const uint8_t* weights, uint32_t in_features,
                                         uint32_t out_features, uint32_t group_size,
                                         int32_t* output, uint64_t max_cycles) {
    if (sim == nullptr || input == nullptr || weights == nullptr || output == nullptr)
        return BNN_EINVAL;
    if (in_features == 0 || out_features == 0 || group_size == 0 ||
        in_features % 32 != 0 || group_size % 32 != 0 || in_features % group_size != 0 ||
        in_features / 32 > 65535 || group_size / 32 > 255 ||
        out_features > 65535 || in_features / group_size > 256)
        return BNN_EINVAL;

    const uint32_t blocks = in_features / 32;
    const uint32_t groups = in_features / group_size;
    const uint64_t input_beats = uint64_t(blocks) * out_features;
    const uint64_t result_count = uint64_t(groups) * out_features;
    if (uint64_t(out_features) * (in_features / 8) > SIZE_MAX ||
        result_count > SIZE_MAX / sizeof(int32_t))
        return BNN_ERANGE;
    // With a valid operand pair and the sink ready, one input beat fits per clock.
    // Leave room for command, pipeline drain and future small changes.
    if (max_cycles == 0) max_cycles = input_beats + result_count + 64;

    ContextScope scope(&sim->context);
    try {
        auto& core = *sim->core;
        if (!core.status_idle || !core.cmd_ready) return recover(sim, BNN_EIO);
        uint64_t sent = 0, received = 0;
        bool accepted = false;
        core.cmd_k_blocks = blocks;
        core.cmd_group_div = group_size / 32;
        core.cmd_out_chans = out_features;
        core.res_ready = 1;

        for (uint64_t cycle = 0; cycle < max_cycles; ++cycle) {
            core.cmd_valid = !accepted;
            core.in_valid = accepted && sent < input_beats;
            if (core.in_valid) {
                const size_t block = size_t(sent % blocks);
                const size_t channel = size_t(sent / blocks);
                const size_t offset = block * 32;
                for (unsigned word = 0; word < 8; ++word) {
                    uint32_t packed = 0;
                    for (unsigned byte = 0; byte < 4; ++byte)
                        packed |= uint32_t(uint8_t(input[offset + word * 4 + byte])) << (byte * 8);
                    core.act_data[word] = packed;
                }
                const size_t weight_offset = channel * (in_features / 8) + block * 4;
                uint32_t packed = 0;
                for (unsigned byte = 0; byte < 4; ++byte)
                    packed |= uint32_t(weights[weight_offset + byte]) << (byte * 8);
                core.weight_data = packed;
            }
            core.eval();
            if (sim->context.gotFinish()) return recover(sim, BNN_EIO);
            const bool cmd_fire = core.cmd_valid && core.cmd_ready;
            const bool in_fire = core.in_valid && core.in_ready;
            const bool res_fire = core.res_valid && core.res_ready;
            const bool task_complete = res_fire && core.res_last;
            if (res_fire) {
                if (received >= result_count || core.res_group_idx != received % groups ||
                    bool(core.res_chan_last) != (received % groups == groups - 1) ||
                    bool(core.res_last) != (received == result_count - 1))
                    return recover(sim, BNN_EIO);
                const uint32_t bits = core.res_data;
                std::memcpy(output + size_t(received), &bits, sizeof bits);
                ++received;
            }
            if (cmd_fire) accepted = true;
            if (in_fire) ++sent;
            sim->tick();
            if (task_complete) {
                if (!accepted || sent != input_beats || received != result_count || !core.cmd_ready)
                    return recover(sim, BNN_EIO);
                core.cmd_valid = 0;
                core.in_valid = 0;
                core.eval();
                sim->flush_trace();
                return BNN_OK;
            }
        }
        return recover(sim, BNN_ETIMEOUT);
    } catch (const std::bad_alloc&) {
        return recover(sim, BNN_ENOMEM);
    } catch (...) {
        return recover(sim, BNN_EIO);
    }
}
