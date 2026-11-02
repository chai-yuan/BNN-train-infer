#pragma once

#include <verilated.h>
#include <verilated_vcd_c.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace bnn::sim {

template <class Dut>
class Simulation {
public:
    VerilatedContext context;
    Dut dut{&context};

    explicit Simulation(const std::string& trace_path = {}) {
        if (!trace_path.empty()) {
            context.traceEverOn(true);
            trace_ = std::make_unique<VerilatedVcdC>();
            dut.trace(trace_.get(), 99);
            trace_->open(trace_path.c_str());
            if (!trace_->isOpen()) {
                throw std::runtime_error("cannot open waveform: " + trace_path);
            }
        }
        dut.clock = 0;
    }

    ~Simulation() {
        dut.final();
        if (trace_) {
            trace_->close();
        }
    }

    // Settled low -> rising edge -> settled low. Sample handshakes BEFORE tick().
    void tick() {
        dut.clock = 0;
        evaluate();
        dut.clock = 1;
        evaluate();
        dut.clock = 0;
        evaluate();
        ++cycles_;
    }

    void reset(unsigned cycles = 3) {
        dut.reset = 1;
        for (unsigned i = 0; i < cycles; ++i) {
            tick();
        }
        dut.reset = 0;
        dut.eval();
    }

    std::uint64_t cycles() const { return cycles_; }

private:
    void evaluate() {
        dut.eval();
        if (trace_) {
            trace_->dump(context.time());
        }
        context.timeInc(1);
        if (context.gotFinish()) {
            throw std::runtime_error("RTL simulation requested finish");
        }
    }

    std::unique_ptr<VerilatedVcdC> trace_;
    std::uint64_t cycles_ = 0;
};

} // namespace bnn::sim
