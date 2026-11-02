#include "Vbnn_sim_top.h"
#include "host_driver.hpp"
#include "simulation.hpp"
#include "virtual_memory.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::uint64_t number(const std::string& text, int base = 0) {
    std::size_t consumed = 0;
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("invalid unsigned number: " + text);
    }
    if (base == 0) {
        const bool hex = text.size() > 2 && text[0] == '0' &&
                         (text[1] == 'x' || text[1] == 'X');
        base = hex ? 16 : 10;
    }
    const auto value = std::stoull(text, &consumed, base);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid number: " + text);
    }
    return value;
}

struct Load { std::uint64_t address; std::string path; };
struct Dump { std::uint64_t address; std::size_t size; std::string path; };
} // namespace

int main(int argc, char** argv) {
    using namespace bnn::sim;
    try {
        std::string commands_path, trace_path;
        std::size_t memory_size = 1024 * 1024;
        std::uint64_t read_base = 0, write_base = 0;
        std::uint64_t accept_timeout = 1000, timeout = 100000, idle_cycles = 16;
        std::uint32_t read_latency = 2, read_gap = 0, write_latency = 2;
        std::uint32_t command_stall = 0, write_stall = 0;
        std::vector<Load> loads;
        std::vector<Dump> dumps;
        const auto next = [&](int& i) -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument("missing option argument");
            }
            return argv[i];
        };
        const auto u32 = [&](int& i) {
            const auto value = number(next(i));
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("timing setting exceeds uint32_t");
            }
            return static_cast<std::uint32_t>(value);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--commands") commands_path = next(i);
            else if (option == "--trace") trace_path = next(i);
            else if (option == "--memory-size") memory_size = number(next(i));
            else if (option == "--read-base") read_base = number(next(i));
            else if (option == "--write-base") write_base = number(next(i));
            else if (option == "--accept-timeout") accept_timeout = number(next(i));
            else if (option == "--timeout") timeout = number(next(i));
            else if (option == "--idle") idle_cycles = number(next(i));
            else if (option == "--rd-latency") read_latency = u32(i);
            else if (option == "--rd-gap") read_gap = u32(i);
            else if (option == "--wr-latency") write_latency = u32(i);
            else if (option == "--cmd-stall-period") command_stall = u32(i);
            else if (option == "--wr-stall-period") write_stall = u32(i);
            else if (option == "--load") {
                const auto address = number(next(i));
                loads.push_back({address, next(i)});
            } else if (option == "--dump") {
                const auto address = number(next(i));
                const auto size = number(next(i));
                dumps.push_back({address, static_cast<std::size_t>(size), next(i)});
            } else if (option == "--help") {
                std::cout << "BNN core Verilator/DPI simulator\n"
                          << "  --commands FILE         one 64-bit hex instruction per line; # comments\n"
                          << "  --load ADDRESS FILE     load raw bytes into read memory (repeatable)\n"
                          << "  --dump ADDRESS SIZE FILE dump raw write memory bytes (repeatable)\n"
                          << "  --memory-size BYTES     capacity of EACH store (default 1 MiB)\n"
                          << "  --read-base ADDR --write-base ADDR  simulated byte bases (default 0)\n"
                          << "  --trace FILE            optional VCD waveform\n"
                          << "  --accept-timeout N --timeout N     per-command cycle budgets\n"
                          << "  --idle N                idle cycles after reset/commands (default 16)\n"
                          << "  --rd-latency N --rd-gap N --wr-latency N\n"
                          << "  --cmd-stall-period N --wr-stall-period N\n"
                          << "Addresses/sizes accept decimal or 0x hex. No commands means idle smoke only.\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }

        clear_dpi_error();
        memory(kReadMemory).configure(memory_size, read_base);
        memory(kWriteMemory).configure(memory_size, write_base);
        for (const auto& load : loads) {
            memory(kReadMemory).load_binary(load.path, load.address);
        }
        Simulation<Vbnn_sim_top> simulation(trace_path);
        auto& dut = simulation.dut;
        dut.cmd_valid = 0;
        dut.cmd_inst = 0;
        dut.cfg_rd_latency = read_latency;
        dut.cfg_rd_gap = read_gap;
        dut.cfg_wr_latency = write_latency;
        dut.cfg_cmd_stall_period = command_stall;
        dut.cfg_wr_stall_period = write_stall;
        simulation.reset();
        HostDriver driver(dut, [&] {
            simulation.tick();
            if (!dpi_error().empty()) {
                throw std::runtime_error("DPI error: " + dpi_error());
            }
        });
        std::size_t executed = 0;
        if (!commands_path.empty()) {
            std::ifstream commands(commands_path);
            if (!commands) {
                throw std::runtime_error("cannot open command file: " + commands_path);
            }
            std::string line;
            std::size_t line_number = 0;
            while (std::getline(commands, line)) {
                ++line_number;
                line = line.substr(0, line.find('#'));
                std::istringstream tokens(line);
                std::string instruction, extra;
                if (!(tokens >> instruction)) continue;
                if (tokens >> extra) {
                    throw std::runtime_error("extra text on command line " + std::to_string(line_number));
                }
                const auto result = driver.execute(number(instruction, 16), accept_timeout, timeout);
                if (result != CommandResult::done) {
                    throw std::runtime_error("cmd_error on command file line " + std::to_string(line_number));
                }
                ++executed;
            }
            if (commands.bad()) {
                throw std::runtime_error("cannot read command file: " + commands_path);
            }
        }
        for (std::uint64_t i = 0; i < idle_cycles; ++i) simulation.tick();
        if (dut.status_busy || dut.rd_busy || dut.wr_busy) {
            throw std::runtime_error("core/DMA still active at simulation end");
        }
        for (const auto& dump : dumps) {
            memory(kWriteMemory).dump_binary(dump.path, dump.address, dump.size);
        }
        std::cout << "Simulation finished: " << executed << " commands, " << simulation.cycles()
                  << " cycles" << (executed == 0 ? " (idle smoke only; NOT a functional core test)" : "")
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Simulation failed: " << error.what() << '\n';
        return 1;
    }
}
