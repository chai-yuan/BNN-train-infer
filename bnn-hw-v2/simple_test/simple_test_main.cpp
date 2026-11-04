#include "Vbnn_sim_top.h"
#include "Vbnn_sim_top___024root.h"
#include "verilated.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kMinimumMemoryBytes = 1u << 20;
constexpr std::uint64_t kCycleLimit = 10'000'000;

std::vector<std::uint64_t> g_commands;
std::size_t g_command_cursor = 0;
std::vector<std::uint8_t> g_memory;
std::uint64_t g_highest_write_end = 0;
std::size_t g_initial_memory_size = 0;

[[noreturn]] void dpi_fatal(const std::string& message) {
    std::fprintf(stderr, "simple_test_main: %s\n", message.c_str());
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void dpi_range_fatal(const char* operation, std::uint64_t address) {
    char message[160];
    std::snprintf(message, sizeof(message),
                  "%s out of range at 0x%llx (memory capacity %zu bytes)",
                  operation, static_cast<unsigned long long>(address),
                  g_memory.size());
    dpi_fatal(message);
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("cannot open " + path.string());
    const std::streamoff end = file.tellg();
    if (end < 0)
        throw std::runtime_error("cannot size " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    file.seekg(0);
    if (!bytes.empty() &&
        !file.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return bytes;
}

bool is_ascii_text(const std::vector<std::uint8_t>& bytes) {
    for (const auto byte : bytes) {
        if (byte == '\n' || byte == '\r' || byte == '\t')
            continue;
        if (byte < 0x20 || byte > 0x7e)
            return false;
    }
    return true;
}

std::vector<std::uint64_t> parse_text_commands(
    const std::vector<std::uint8_t>& bytes, const std::string& path) {
    std::vector<std::uint64_t> commands;
    std::istringstream stream(std::string(bytes.begin(), bytes.end()));
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(stream, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos)
            line.resize(comment);

        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos)
            continue;
        const auto last = line.find_last_not_of(" \t\r");
        std::string token = line.substr(first, last - first + 1);
        if (token.size() > 2 && token[0] == '0' &&
            (token[1] == 'x' || token[1] == 'X')) {
            token = token.substr(2);
        }
        if (token.empty() || token.size() > 16)
            throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                     ": expected a 64-bit hexadecimal instruction");

        std::uint64_t instruction = 0;
        const auto result = std::from_chars(token.data(), token.data() + token.size(),
                                            instruction, 16);
        if (result.ec != std::errc{} || result.ptr != token.data() + token.size())
            throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                     ": invalid hexadecimal instruction");
        commands.push_back(instruction);
    }
    return commands;
}

std::vector<std::uint64_t> parse_binary_commands(
    const std::vector<std::uint8_t>& bytes, const std::string& path) {
    if (bytes.size() % 8 != 0)
        throw std::runtime_error(path +
                                 ": binary command file size must be a multiple of 8 bytes");
    std::vector<std::uint64_t> commands;
    commands.reserve(bytes.size() / 8);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 8) {
        std::uint64_t instruction = 0;
        for (int b = 0; b < 8; ++b)
            instruction |= std::uint64_t{bytes[offset + b]} << (8 * b);
        commands.push_back(instruction);
    }
    return commands;
}

std::vector<std::uint64_t> load_commands(const std::filesystem::path& path) {
    const auto bytes = read_file(path);
    if (is_ascii_text(bytes))
        return parse_text_commands(bytes, path.string());
    return parse_binary_commands(bytes, path.string());
}

std::vector<std::uint8_t> load_memory(const std::filesystem::path& path) {
    const auto bytes = read_file(path);
    g_initial_memory_size = bytes.size();
    const std::size_t rounded =
        (bytes.size() + 7u) & ~static_cast<std::size_t>(7u);
    const std::size_t capacity = std::max(kMinimumMemoryBytes, rounded);
    std::vector<std::uint8_t> memory(capacity, 0);
    std::copy(bytes.begin(), bytes.end(), memory.begin());
    return memory;
}

void tick(Vbnn_sim_top& dut) {
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
    dut.clock = 0;
    dut.eval();
}

std::filesystem::path output_path_for(const std::filesystem::path& memory_path) {
    std::filesystem::path output = memory_path;
    output.replace_filename(memory_path.stem().string() + "_out" +
                            memory_path.extension().string());
    return output;
}

bool range_ok(std::uint64_t address, std::uint64_t bytes) {
    const auto size = static_cast<std::uint64_t>(g_memory.size());
    return address <= size && bytes <= size - address;
}

// =========================================================================
// Lightweight cycle-level profiler.
//
// The RTL already exposes one FSM state per functional module.  Sampling those
// registers every cycle is enough to build a timeline ("flame chart") of what
// each module was doing, and the valid/ready pairs at module boundaries tell us
// which side of a producer/consumer pair was actually holding the system back.
//
// The internal signal names below are generated by Verilator for the current
// hierarchy.  They are a simulation-only dependency: none of this changes the
// synthesizable RTL.
// =========================================================================
using Root = Vbnn_sim_top___024root;

constexpr std::size_t kMaxModuleStates = 10;
constexpr int kModuleCount = 10;
constexpr int kPhaseCount = 4;

enum ProfilePhase {
    PHASE_IDLE = 0,
    PHASE_LOAD = 1,
    PHASE_COMPUTE = 2,
    PHASE_WRITE = 3
};

const char* const kPhaseNames[kPhaseCount] = {"IDLE", "LOAD", "COMPUTE", "WRITE"};

const char* const kControllerStates[] = {
    "IDLE", "LOAD_DRAIN", "LOAD_START", "LOAD_WAIT", "LOAD_CONFIG",
    "COMPUTE_START", "COMPUTE_WAIT", "COMPUTE_DRAIN", "WRITE_START", "WRITE_WAIT"
};
const char* const kWeightLoaderStates[] = {
    "IDLE", "COMMAND", "STREAM", "WRITE_PADDING"
};
const char* const kIqbufStates[] = {
    "IDLE", "COMMAND", "RECEIVE", "SCAN", "SCALE", "QUANTIZE", "OUTPUT_ROW"
};
const char* const kArrayStates[] = {
    "IDLE", "READ_REQUEST", "READ_RETURN", "SUBMIT", "WAIT_OUTPUT"
};
const char* const kUnitStates[] = {
    "IDLE", "DOT", "FP_CONV", "WEIGHT_SCALE", "ROW_SCALE", "OUTPUT_DATA"
};
const char* const kPostStates[] = {
    "IDLE", "RECEIVE", "SETTLE", "WRITE_BACK"
};
const char* const kWriterStates[] = {
    "IDLE", "COMMAND", "READ_REQUEST", "READ_RETURN", "SEND_DATA",
    "WAIT_RESPONSE", "ROW_COMPLETE"
};

int phase_of_controller(int state) {
    if (state == 0)
        return PHASE_IDLE;
    if (state <= 4)
        return PHASE_LOAD;
    if (state <= 7)
        return PHASE_COMPUTE;
    return PHASE_WRITE;
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char escaped[8];
                std::snprintf(escaped, sizeof(escaped), "\\u%04x",
                              static_cast<unsigned>(static_cast<unsigned char>(c)));
                out += escaped;
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string hex_value(std::uint64_t value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << value;
    return stream.str();
}

std::string format_states(const char* const* names, int count,
                          const std::array<std::uint64_t, kMaxModuleStates>& cycles) {
    std::ostringstream stream;
    bool first = true;
    for (int i = 0; i < count; ++i) {
        if (cycles[i] == 0)
            continue;
        if (!first)
            stream << ", ";
        stream << names[i] << "=" << cycles[i];
        first = false;
    }
    return first ? "-" : stream.str();
}

struct ModuleTrack {
    std::string name;
    std::string folded_name;
    int tid = 0;
    const char* const* states = nullptr;
    int state_count = 0;
    const CData* signal = nullptr;
    int current = -1;
    std::uint64_t start = 0;
    std::array<std::uint64_t, kMaxModuleStates> cycles{};
    std::array<std::uint64_t, kMaxModuleStates> entries{};
};

struct CommandInfo {
    int opcode = 0;
    const char* name = "UNKNOWN";
    std::uint32_t address = 0;
    int m = 0;
    int n = 0;
    bool accum = false;
    bool int8 = false;
};

CommandInfo decode_command(std::uint64_t inst) {
    CommandInfo info;
    info.opcode = static_cast<int>(inst & 0xffu);
    info.address = static_cast<std::uint32_t>(inst >> 32);
    info.m = static_cast<int>((inst >> 8) & 0xffu);
    info.n = static_cast<int>((inst >> 16) & 0xffu);
    info.accum = ((inst >> 24) & 1u) != 0;
    info.int8 = ((inst >> 25) & 1u) != 0;
    switch (info.opcode) {
    case 0x01: info.name = "SET_SRC_STRIDE"; break;
    case 0x02: info.name = "SET_DST_STRIDE"; break;
    case 0x10: info.name = "LOAD_WEIGHT"; break;
    case 0x20: info.name = "COMPUTE"; break;
    case 0x30: info.name = "WRITEBACK"; break;
    default: info.name = "UNKNOWN"; break;
    }
    return info;
}

struct TxRecord {
    std::string type;
    std::string label;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::string detail;
    std::string args;

    std::uint64_t duration() const {
        return end >= start ? end - start : 0;
    }
};

struct ActiveCommand {
    bool active = false;
    std::uint64_t start = 0;
    std::uint64_t inst = 0;
    CommandInfo info;
};

struct ActiveWeightLoad {
    bool active = false;
    std::uint64_t start = 0;
    std::uint32_t addr = 0;
    int n = 0;
    int beats = 0;
    std::array<std::uint64_t, kMaxModuleStates> states{};
    std::uint64_t dma_stall = 0;
    std::uint64_t dma_starve = 0;
};

struct ActiveCompute {
    bool active = false;
    std::uint64_t start = 0;
    int m = 0;
    int n = 0;
    int g = 0;
    bool accum = false;
    bool int8 = false;
    std::uint32_t addr = 0;
    std::uint64_t rows = 0;
    std::uint64_t groups = 0;
};

struct ActiveWriteback {
    bool active = false;
    std::uint64_t start = 0;
    int m = 0;
    int n = 0;
    int g = 0;
    std::uint32_t addr = 0;
    std::uint64_t beats = 0;
};

struct ActiveDmaRead {
    bool active = false;
    std::uint64_t start = 0;
    std::uint32_t addr = 0;
    int len = 0;
    int source = 0; // 0 = weight loader, 1 = input quant buffer
    int beats = 0;
    std::uint64_t first_beat_latency = 0;
    std::uint64_t last_beat = 0;
    std::uint64_t max_gap = 0;
    std::uint64_t sum_gap = 0;
    std::uint64_t stall = 0;
    std::uint64_t starve = 0;
};

struct ActiveDmaWrite {
    bool active = false;
    std::uint64_t start = 0;
    std::uint32_t addr = 0;
    int len = 0;
    int beats = 0;
    std::uint64_t first_beat_latency = 0;
    std::uint64_t last_beat = 0;
    std::uint64_t data_done = 0;
    std::uint64_t max_gap = 0;
    std::uint64_t sum_gap = 0;
    std::uint64_t stall = 0;
    std::uint64_t starve = 0;
};

struct ActiveRow {
    bool active = false;
    std::uint64_t prep_start = 0;
    std::uint64_t prep_cycles = 0;
    std::uint64_t start = 0;
    int index = 0;
    int m = 0;
    int g = 0;
    bool int8 = false;
    std::uint64_t iqbuf_wait_array = 0;
    std::uint64_t array_starved = 0;
    std::uint64_t odma_wait = 0;
    std::array<std::uint64_t, kMaxModuleStates> iq_states{};
    std::array<std::uint64_t, kMaxModuleStates> array_states{};
    std::array<std::uint64_t, kMaxModuleStates> unit_states{};
};

struct ActiveGroup {
    bool active = false;
    std::uint64_t start = 0;
    int row = -1;
    int group = 0;
    int channels = 0;
    std::uint64_t odma_wait = 0;
    std::array<std::uint64_t, kMaxModuleStates> array_states{};
    std::array<std::uint64_t, kMaxModuleStates> unit_states{};
};

struct ActiveWriteRow {
    bool active = false;
    std::uint64_t start = 0;
    int index = 0;
    std::uint32_t addr = 0;
    int groups = 0;
    std::uint64_t mem_stall = 0;
    std::uint64_t writer_gap = 0;
    std::array<std::uint64_t, kMaxModuleStates> states{};
};

struct InternalSignals {
    const CData* ctrl_state = nullptr;
    const CData* ctrl_done = nullptr;
    const CData* ctrl_error = nullptr;
    const IData* ctrl_address = nullptr;
    const CData* ctrl_loaded_channels = nullptr;
    const CData* ctrl_groups = nullptr;
    const CData* ctrl_rows = nullptr;
    const CData* ctrl_result_rows = nullptr;
    const CData* ctrl_result_channels = nullptr;
    const CData* ctrl_accum = nullptr;
    const CData* ctrl_int8 = nullptr;
    const CData* ctrl_rd_select = nullptr;

    const CData* wl_state = nullptr;
    const CData* wl_done = nullptr;
    const IData* wl_source_addr = nullptr;
    const SData* wl_burst_len = nullptr;

    const CData* iq_state = nullptr;
    const CData* iq_done = nullptr;
    const IData* iq_row_address = nullptr;

    const CData* arr_state = nullptr;
    const CData* arr_groups = nullptr;
    const CData* arr_group_index = nullptr;
    const CData* arr_last_group = nullptr;
    const CData* unit_state[4] = {nullptr, nullptr, nullptr, nullptr};

    const CData* post_state = nullptr;
    const CData* post_done = nullptr;
    const CData* post_accum = nullptr;
    const SData* post_remaining = nullptr;

    const CData* wr_state = nullptr;
    const CData* wr_done = nullptr;
    const CData* wr_rows_left = nullptr;
    const SData* wr_groups = nullptr;
    const IData* wr_row_addr = nullptr;
    const CData* wr_last_beat = nullptr;

    const CData* cmd_stream_state = nullptr;
    const CData* cmd_ready = nullptr;
    const QData* cmd_inst = nullptr;

    const CData* idma_valid = nullptr;
    const CData* idma_ready = nullptr;
    const CData* odma_valid = nullptr;
    const CData* odma_ready = nullptr;
    const IData* mem_rd_cmd_addr = nullptr;
    const SData* mem_rd_cmd_len = nullptr;
    const CData* mem_rd_cmd_ready = nullptr;
    const CData* mem_rd_data_valid = nullptr;
    const CData* mem_rd_data_last = nullptr;
    const CData* mem_wr_cmd_ready = nullptr;
    const CData* mem_wr_data_valid = nullptr;
    const CData* mem_wr_data_ready = nullptr;
    const CData* mem_wr_done = nullptr;
};

class Profiler {
public:
    Profiler(Vbnn_sim_top& dut, const std::filesystem::path& trace_path,
             bool enable_trace)
        : root_(dut.rootp), trace_enabled_(enable_trace), trace_path_(trace_path) {
        bind_signals();
        bind_modules();
        if (trace_enabled_) {
            trace_.open(trace_path_, std::ios::out | std::ios::trunc);
            if (!trace_)
                throw std::runtime_error("cannot create " + trace_path_.string());
            begin_trace();
        }
    }

    void sample_pre(std::uint64_t cycle) {
        const int wl = state_of(signals_.wl_state);
        const int iq = state_of(signals_.iq_state);
        const bool rd_select = *signals_.ctrl_rd_select != 0;
        const bool rd_cmd_valid =
            (rd_select ? iq : wl) == 1;
        const bool rd_data_ready =
            (rd_select ? iq : wl) == 2;
        const bool rd_data_valid = *signals_.mem_rd_data_valid != 0;
        const bool rd_data_last = *signals_.mem_rd_data_last != 0;
        const bool idma_valid = *signals_.idma_valid != 0;
        const bool idma_ready = *signals_.idma_ready != 0;
        const bool odma_valid = *signals_.odma_valid != 0;
        const bool odma_ready = *signals_.odma_ready != 0;
        const bool wr_data_valid = *signals_.mem_wr_data_valid != 0;
        const bool wr_data_ready = *signals_.mem_wr_data_ready != 0;
        const bool wr_cmd_valid = state_of(signals_.wr_state) == 1;
        const bool wr_data_last =
            state_of(signals_.wr_state) == 4 && *signals_.wr_last_beat != 0;
        const int phase = phase_of_controller(state_of(signals_.ctrl_state));

        if (active_dma_read_.active) {
            if (rd_data_valid && !rd_data_ready)
                ++active_dma_read_.stall;
            else if (!rd_data_valid && rd_data_ready)
                ++active_dma_read_.starve;
        }
        if (active_dma_write_.active) {
            if (wr_data_valid && !wr_data_ready)
                ++active_dma_write_.stall;
            else if (!wr_data_valid && wr_data_ready)
                ++active_dma_write_.starve;
        }

        if (phase == PHASE_LOAD) {
            if (rd_data_valid && !rd_data_ready) {
                ++attrib_load_data_stall_;
                if (active_weight_load_.active)
                    ++active_weight_load_.dma_stall;
            } else if (!rd_data_valid && rd_data_ready) {
                ++attrib_load_data_starve_;
                if (active_weight_load_.active)
                    ++active_weight_load_.dma_starve;
            }
            if (rd_cmd_valid && *signals_.mem_rd_cmd_ready == 0)
                ++attrib_load_cmd_wait_;
        } else if (phase == PHASE_COMPUTE) {
            if (idma_valid && !idma_ready) {
                ++attrib_compute_iqbuf_blocked_;
                if (row_prep_active_)
                    ++row_prep_wait_array_;
            } else if (!idma_valid && idma_ready) {
                ++attrib_compute_array_starved_;
                if (row_prep_active_)
                    ++row_prep_array_starved_;
            }
            if (odma_valid && !odma_ready) {
                ++attrib_compute_odma_stall_;
                if (active_group_.active)
                    ++active_group_.odma_wait;
                if (active_row_.active)
                    ++active_row_.odma_wait;
            }
        } else if (phase == PHASE_WRITE) {
            if (wr_data_valid && !wr_data_ready) {
                ++attrib_write_mem_stall_;
                if (active_write_row_.active)
                    ++active_write_row_.mem_stall;
            } else if (!wr_data_valid && wr_data_ready) {
                ++attrib_write_writer_gap_;
                if (active_write_row_.active)
                    ++active_write_row_.writer_gap;
            }
            if (wr_cmd_valid && *signals_.mem_wr_cmd_ready == 0)
                ++attrib_write_cmd_wait_;
        }

        if (state_of(signals_.cmd_stream_state) == 1 && *signals_.cmd_ready != 0) {
            active_command_ = ActiveCommand{};
            active_command_.active = true;
            active_command_.start = cycle;
            active_command_.inst = *signals_.cmd_inst;
            active_command_.info = decode_command(active_command_.inst);
            ++accepted_commands_;
        }

        if (rd_cmd_valid && *signals_.mem_rd_cmd_ready != 0) {
            active_dma_read_ = ActiveDmaRead{};
            active_dma_read_.active = true;
            active_dma_read_.start = cycle;
            active_dma_read_.addr = *signals_.mem_rd_cmd_addr;
            active_dma_read_.len = static_cast<int>(*signals_.mem_rd_cmd_len);
            active_dma_read_.source = rd_select ? 1 : 0;
            active_dma_read_.last_beat = cycle;
        }

        if (rd_data_valid && rd_data_ready && active_dma_read_.active) {
            if (active_dma_read_.beats == 0) {
                active_dma_read_.first_beat_latency = cycle - active_dma_read_.start;
            } else {
                const std::uint64_t gap = cycle - active_dma_read_.last_beat;
                active_dma_read_.max_gap = std::max(active_dma_read_.max_gap, gap);
                active_dma_read_.sum_gap += gap;
            }
            ++active_dma_read_.beats;
            ++mem_rd_beats_;
            active_dma_read_.last_beat = cycle;
            if (rd_data_last)
                finalize_dma_read(cycle);
        }

        if (wr_cmd_valid && *signals_.mem_wr_cmd_ready != 0) {
            active_dma_write_ = ActiveDmaWrite{};
            active_dma_write_.active = true;
            active_dma_write_.start = cycle;
            active_dma_write_.addr = *signals_.wr_row_addr;
            active_dma_write_.len = static_cast<int>(*signals_.wr_groups);
            active_dma_write_.last_beat = cycle;
        }

        if (wr_data_valid && wr_data_ready && active_dma_write_.active) {
            if (active_dma_write_.beats == 0) {
                active_dma_write_.first_beat_latency = cycle - active_dma_write_.start;
            } else {
                const std::uint64_t gap = cycle - active_dma_write_.last_beat;
                active_dma_write_.max_gap = std::max(active_dma_write_.max_gap, gap);
                active_dma_write_.sum_gap += gap;
            }
            ++active_dma_write_.beats;
            ++mem_wr_beats_;
            if (active_writeback_.active)
                ++active_writeback_.beats;
            active_dma_write_.last_beat = cycle;
            if (wr_data_last)
                active_dma_write_.data_done = cycle;
        }
        if (*signals_.mem_wr_done != 0 && active_dma_write_.active)
            finalize_dma_write(cycle);

        if (idma_valid && idma_ready) {
            ++idma_handshakes_;
            if (active_compute_.active)
                ++active_compute_.rows;
            start_compute_row(cycle);
        }
        if (odma_valid && odma_ready) {
            ++odma_handshakes_;
            if (active_compute_.active)
                ++active_compute_.groups;
            finalize_group(cycle);
            if (*signals_.arr_last_group != 0)
                finalize_row(cycle);
        }
    }

    void sample_post(std::uint64_t cycle, bool count_cycle) {
        const int prev_ctrl = modules_[0].current;
        const int prev_wl = modules_[1].current;
        const int prev_iq = modules_[2].current;
        const int prev_arr = modules_[3].current;
        const int prev_wr = modules_[9].current;

        for (auto& module : modules_) {
            int state = state_of(module.signal);
            if (state >= module.state_count)
                state = module.state_count - 1;
            if (state == module.current)
                continue;
            if (module.current >= 0) {
                const std::uint64_t duration = cycle - module.start;
                if (duration != 0) {
                    std::ostringstream args;
                    args << "\"module\":\"" << json_escape(module.name)
                         << "\",\"state\":\""
                         << json_escape(module.states[module.current]) << "\"";
                    emit_complete(module.tid, module.states[module.current],
                                  module.name, module.start, duration, args.str());
                }
            }
            module.current = state;
            module.start = cycle;
            ++module.entries[state];
        }

        const int ctrl = modules_[0].current;
        const int wl = modules_[1].current;
        const int iq = modules_[2].current;
        const int arr = modules_[3].current;
        const int wr = modules_[9].current;
        const bool done = *signals_.ctrl_done != 0;
        const bool error = *signals_.ctrl_error != 0;

        if (wl == 1 && prev_wl == 0)
            start_weight_load(cycle);
        if (ctrl == 5 && prev_ctrl != 5)
            start_compute(cycle);
        if (ctrl == 8 && prev_ctrl != 8)
            start_writeback(cycle);
        if (iq == 1 && prev_iq != 1)
            start_row_prep(cycle);
        if (wr == 1 && prev_wr != 1)
            start_write_row(cycle);
        if (arr == 1 && prev_arr != 1 && (prev_arr == 0 || prev_arr == 4) &&
            active_compute_.active)
            start_group(cycle);
        if (arr == 3 && prev_arr == 4 && active_compute_.active)
            start_group(cycle);

        if (*signals_.wl_done != 0)
            finalize_weight_load(cycle);
        if (wr == 6 && prev_wr != 6)
            finalize_write_row(cycle);
        if (done || error) {
            if (active_compute_.active)
                finalize_compute(cycle, error);
            if (active_writeback_.active)
                finalize_writeback(cycle, error);
            if (active_weight_load_.active)
                finalize_weight_load(cycle);
            finalize_command(cycle, error);
        }

        if (!count_cycle)
            return;

        const int phase = phase_of_controller(ctrl);
        ++phase_cycles_[phase];
        ++cycle_samples_;
        for (int mod = 0; mod < kModuleCount; ++mod) {
            auto& module = modules_[mod];
            if (module.current >= 0) {
                ++module.cycles[module.current];
                ++phase_module_state_[pms_index(phase, mod, module.current)];
            }
        }

        if (active_weight_load_.active && wl < 4)
            ++active_weight_load_.states[wl];
        if (row_prep_active_ && iq < 7)
            ++row_prep_iq_states_[iq];
        if (active_row_.active) {
            if (arr < 5)
                ++active_row_.array_states[arr];
            if (modules_[4].current >= 0 && modules_[4].current < 6)
                ++active_row_.unit_states[modules_[4].current];
        }
        if (active_group_.active) {
            if (arr < 5)
                ++active_group_.array_states[arr];
            if (modules_[4].current >= 0 && modules_[4].current < 6)
                ++active_group_.unit_states[modules_[4].current];
        }
        if (active_write_row_.active && wr < 7)
            ++active_write_row_.states[wr];
    }

    void finish(std::uint64_t total_cycles) {
        total_cycles_ = total_cycles;
        for (auto& module : modules_) {
            if (module.current >= 0) {
                const std::uint64_t duration = total_cycles - module.start;
                if (duration != 0) {
                    std::ostringstream args;
                    args << "\"module\":\"" << json_escape(module.name)
                         << "\",\"state\":\""
                         << json_escape(module.states[module.current]) << "\"";
                    emit_complete(module.tid, module.states[module.current],
                                  module.name, module.start, duration, args.str());
                }
            }
        }
        if (active_group_.active)
            finalize_group(total_cycles);
        if (active_row_.active)
            finalize_row(total_cycles);
        if (active_write_row_.active)
            finalize_write_row(total_cycles);
        if (active_weight_load_.active)
            finalize_weight_load(total_cycles);
        if (active_dma_read_.active)
            finalize_dma_read(total_cycles);
        if (active_dma_write_.active)
            finalize_dma_write(total_cycles);
        if (active_compute_.active)
            finalize_compute(total_cycles, false);
        if (active_writeback_.active)
            finalize_writeback(total_cycles, false);
        if (active_command_.active)
            finalize_command(total_cycles, false);
        if (trace_enabled_ && trace_.is_open()) {
            trace_ << "\n]}\n";
            trace_.close();
        }
    }

    void write_reports(const std::filesystem::path& folded_path,
                       const std::filesystem::path& report_path, int top_n) const {
        write_folded(folded_path);
        write_text_report(report_path, folded_path, top_n);
    }

    void print_summary(std::ostream& out) const {
        out << "profile: " << total_cycles_ << " cycles, "
            << accepted_commands_ << " commands\n";
        write_bottleneck(out);
    }

private:
    Root* root_ = nullptr;
    InternalSignals signals_;
    std::array<ModuleTrack, kModuleCount> modules_{};
    bool trace_enabled_ = false;
    std::filesystem::path trace_path_;
    std::ofstream trace_;
    bool first_entry_ = true;
    std::uint64_t total_cycles_ = 0;
    std::uint64_t cycle_samples_ = 0;
    std::uint64_t accepted_commands_ = 0;

    std::array<std::uint64_t, kPhaseCount> phase_cycles_{};
    std::array<std::uint64_t, kPhaseCount * kModuleCount * kMaxModuleStates>
        phase_module_state_{};

    std::uint64_t attrib_load_data_stall_ = 0;
    std::uint64_t attrib_load_data_starve_ = 0;
    std::uint64_t attrib_load_cmd_wait_ = 0;
    std::uint64_t attrib_compute_iqbuf_blocked_ = 0;
    std::uint64_t attrib_compute_array_starved_ = 0;
    std::uint64_t attrib_compute_odma_stall_ = 0;
    std::uint64_t attrib_write_mem_stall_ = 0;
    std::uint64_t attrib_write_writer_gap_ = 0;
    std::uint64_t attrib_write_cmd_wait_ = 0;

    std::uint64_t mem_rd_beats_ = 0;
    std::uint64_t mem_wr_beats_ = 0;
    std::uint64_t idma_handshakes_ = 0;
    std::uint64_t odma_handshakes_ = 0;

    ActiveCommand active_command_;
    ActiveWeightLoad active_weight_load_;
    ActiveCompute active_compute_;
    ActiveWriteback active_writeback_;
    ActiveDmaRead active_dma_read_;
    ActiveDmaWrite active_dma_write_;
    ActiveRow active_row_;
    ActiveGroup active_group_;
    ActiveWriteRow active_write_row_;

    bool row_prep_active_ = false;
    std::uint64_t row_prep_start_ = 0;
    std::uint64_t row_prep_wait_array_ = 0;
    std::uint64_t row_prep_array_starved_ = 0;
    std::array<std::uint64_t, kMaxModuleStates> row_prep_iq_states_{};
    int row_counter_ = 0;
    int current_row_index_ = -1;

    int weight_load_count_ = 0;
    int compute_count_ = 0;
    int writeback_count_ = 0;
    int command_count_ = 0;

    std::vector<TxRecord> transactions_;

    static int state_of(const CData* signal) {
        return signal ? static_cast<int>(*signal) : 0;
    }

    static std::size_t pms_index(int phase, int module, int state) {
        return (static_cast<std::size_t>(phase) * kModuleCount + module) *
                   kMaxModuleStates +
               state;
    }

    static std::uint64_t busy_cycles(const ModuleTrack& module) {
        std::uint64_t total = 0;
        for (int i = 1; i < module.state_count; ++i)
            total += module.cycles[i];
        return total;
    }

    static int tid_for_tx(const std::string& type) {
        if (type == "command") return 11;
        if (type == "weight_load") return 12;
        if (type == "compute") return 13;
        if (type == "writeback") return 14;
        if (type == "dma_read") return 15;
        if (type == "dma_write") return 16;
        if (type == "compute_row") return 17;
        if (type == "compute_group") return 18;
        if (type == "write_row") return 19;
        return 20;
    }

    void bind_signals() {
        Root* r = root_;
#define P(field) (&r->bnn_sim_top__DOT__##field)
        signals_.ctrl_state = P(u_core__DOT__u_top_controller__DOT__state);
        signals_.ctrl_done = P(u_core__DOT__u_top_controller__DOT__done_reg);
        signals_.ctrl_error = P(u_core__DOT__u_top_controller__DOT__error_reg);
        signals_.ctrl_address = P(u_core__DOT__u_top_controller__DOT__address_reg);
        signals_.ctrl_loaded_channels =
            P(u_core__DOT__u_top_controller__DOT__loaded_channels);
        signals_.ctrl_groups = P(u_core__DOT__u_top_controller__DOT__groups);
        signals_.ctrl_rows = P(u_core__DOT__u_top_controller__DOT__operation_rows);
        signals_.ctrl_result_rows =
            P(u_core__DOT__u_top_controller__DOT__result_rows);
        signals_.ctrl_result_channels =
            P(u_core__DOT__u_top_controller__DOT__result_channels);
        signals_.ctrl_accum = P(u_core__DOT__u_top_controller__DOT__accum);
        signals_.ctrl_int8 = P(u_core__DOT__u_top_controller__DOT__input_int8);
        signals_.ctrl_rd_select = P(u_core__DOT__u_top_controller__DOT__rd_select);

        signals_.wl_state = P(u_core__DOT__u_weight_loader__DOT__state);
        signals_.wl_done = P(u_core__DOT__u_weight_loader__DOT__done_reg);
        signals_.wl_source_addr = P(u_core__DOT__u_weight_loader__DOT__source_addr);
        signals_.wl_burst_len = P(u_core__DOT__u_weight_loader__DOT__burst_len);

        signals_.iq_state = P(u_core__DOT__u_input_quant_buffer__DOT__state);
        signals_.iq_done = P(u_core__DOT__u_input_quant_buffer__DOT__done_reg);
        signals_.iq_row_address =
            P(u_core__DOT__u_input_quant_buffer__DOT__row_address);

        signals_.arr_state = P(u_core__DOT__u_core_array__DOT__state);
        signals_.arr_groups = P(u_core__DOT__u_core_array__DOT__groups);
        signals_.arr_group_index = P(u_core__DOT__u_core_array__DOT__group_index);
        signals_.arr_last_group = P(u_core__DOT__u_core_array__DOT__last_group);
#define U(n) (&r->bnn_sim_top__DOT__u_core__DOT__u_core_array__DOT__g_unit__BRA__##n##__KET____DOT__u_unit__DOT__state)
        signals_.unit_state[0] = U(0);
        signals_.unit_state[1] = U(1);
        signals_.unit_state[2] = U(2);
        signals_.unit_state[3] = U(3);
#undef U

        signals_.post_state = P(u_core__DOT__u_output_accum__DOT__state);
        signals_.post_done = P(u_core__DOT__u_output_accum__DOT__done_reg);
        signals_.post_accum = P(u_core__DOT__u_output_accum__DOT__accum_reg);
        signals_.post_remaining =
            P(u_core__DOT__u_output_accum__DOT__remaining_reg);

        signals_.wr_state = P(u_core__DOT__u_output_writer__DOT__state);
        signals_.wr_done = P(u_core__DOT__u_output_writer__DOT__done_reg);
        signals_.wr_rows_left = P(u_core__DOT__u_output_writer__DOT__rows_left);
        signals_.wr_groups = P(u_core__DOT__u_output_writer__DOT__groups);
        signals_.wr_row_addr = P(u_core__DOT__u_output_writer__DOT__row_addr);
        signals_.wr_last_beat = P(u_core__DOT__u_output_writer__DOT__last_beat);

        signals_.cmd_stream_state = P(u_cmd_stream__DOT__state);
        signals_.cmd_ready = P(cmd_ready);
        signals_.cmd_inst = P(u_cmd_stream__DOT__inst);

        signals_.idma_valid = P(u_core__DOT__idma_valid);
        signals_.idma_ready = P(u_core__DOT__idma_ready);
        signals_.odma_valid = P(u_core__DOT__odma_valid);
        signals_.odma_ready = P(u_core__DOT__odma_ready);
        signals_.mem_rd_cmd_addr = P(mem_rd_cmd_addr);
        signals_.mem_rd_cmd_len = P(mem_rd_cmd_len);
        signals_.mem_rd_cmd_ready = P(mem_rd_cmd_ready);
        signals_.mem_rd_data_valid = P(mem_rd_data_valid);
        signals_.mem_rd_data_last = P(mem_rd_data_last);
        signals_.mem_wr_cmd_ready = P(mem_wr_cmd_ready);
        signals_.mem_wr_data_valid = P(mem_wr_data_valid);
        signals_.mem_wr_data_ready = P(mem_wr_data_ready);
        signals_.mem_wr_done = P(mem_wr_done);
#undef P
    }

    void bind_modules() {
        modules_[0] = ModuleTrack{"u_top_controller", "u_top_controller", 1,
                                  kControllerStates, 10, signals_.ctrl_state};
        modules_[1] = ModuleTrack{"u_weight_loader", "u_weight_loader", 2,
                                  kWeightLoaderStates, 4, signals_.wl_state};
        modules_[2] = ModuleTrack{"u_input_quant_buffer", "u_input_quant_buffer", 3,
                                  kIqbufStates, 7, signals_.iq_state};
        modules_[3] = ModuleTrack{"u_core_array", "u_core_array", 4,
                                  kArrayStates, 5, signals_.arr_state};
        for (int i = 0; i < 4; ++i) {
            const std::string unit_name =
                "u_core_array/g_unit[" + std::to_string(i) + "]";
            modules_[4 + i] = ModuleTrack{
                unit_name, "u_core_array;g_unit[" + std::to_string(i) + "]",
                5 + i, kUnitStates, 6, signals_.unit_state[i]};
        }
        modules_[8] = ModuleTrack{"u_output_accum", "u_output_accum", 9,
                                  kPostStates, 4, signals_.post_state};
        modules_[9] = ModuleTrack{"u_output_writer", "u_output_writer", 10,
                                  kWriterStates, 7, signals_.wr_state};
    }

    void begin_trace() {
        trace_ << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
        bool first = true;
        auto meta = [&](int tid, const std::string& name) {
            if (!first)
                trace_ << ",\n";
            first = false;
            trace_ << "{\"ph\":\"M\",\"pid\":1,\"tid\":" << tid
                   << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
                   << json_escape(name) << "\"}}";
        };
        trace_ << "{\"ph\":\"M\",\"pid\":1,\"tid\":0,\"name\":\"process_name\","
                  "\"args\":{\"name\":\"bnn_sim_top\"}}";
        first = false;
        for (const auto& module : modules_)
            meta(module.tid, module.name);
        meta(11, "commands");
        meta(12, "tx/weight_load");
        meta(13, "tx/compute");
        meta(14, "tx/writeback");
        meta(15, "dma_read");
        meta(16, "dma_write");
        meta(17, "compute_rows");
        meta(18, "compute_groups");
        meta(19, "write_rows");
        first_entry_ = false;
    }

    void emit_complete(int tid, const std::string& name, const std::string& category,
                       std::uint64_t start, std::uint64_t duration,
                       const std::string& args) {
        if (!trace_enabled_)
            return;
        if (!first_entry_)
            trace_ << ",\n";
        first_entry_ = false;
        trace_ << "{\"ph\":\"X\",\"pid\":1,\"tid\":" << tid
               << ",\"name\":\"" << json_escape(name) << "\",\"cat\":\""
               << json_escape(category) << "\",\"ts\":" << start
               << ",\"dur\":" << duration;
        if (!args.empty())
            trace_ << ",\"args\":{" << args << "}";
        trace_ << "}";
    }

    void record_tx(TxRecord tx) {
        emit_complete(tid_for_tx(tx.type), tx.label, tx.type, tx.start,
                      tx.duration(), tx.args);
        transactions_.push_back(std::move(tx));
    }

    void start_weight_load(std::uint64_t cycle) {
        active_weight_load_ = ActiveWeightLoad{};
        active_weight_load_.active = true;
        active_weight_load_.start = cycle;
        active_weight_load_.addr = *signals_.wl_source_addr;
        active_weight_load_.n = static_cast<int>(*signals_.ctrl_loaded_channels);
        active_weight_load_.beats = static_cast<int>(*signals_.wl_burst_len);
        ++weight_load_count_;
    }

    void finalize_weight_load(std::uint64_t cycle) {
        if (!active_weight_load_.active)
            return;
        TxRecord tx;
        tx.type = "weight_load";
        tx.start = active_weight_load_.start;
        tx.end = cycle;
        tx.label = "weight_load #" + std::to_string(weight_load_count_ - 1);
        std::ostringstream detail;
        detail << "N=" << active_weight_load_.n
               << " beats=" << active_weight_load_.beats
               << " wloader["
               << format_states(kWeightLoaderStates, 4,
                                active_weight_load_.states)
               << "] dma_stall=" << active_weight_load_.dma_stall
               << " dma_starve=" << active_weight_load_.dma_starve;
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"address\":\"" << hex_value(active_weight_load_.addr)
             << "\",\"N\":" << active_weight_load_.n
             << ",\"beats\":" << active_weight_load_.beats
             << ",\"dma_stall\":" << active_weight_load_.dma_stall
             << ",\"dma_starve\":" << active_weight_load_.dma_starve;
        tx.args = args.str();
        record_tx(std::move(tx));
        active_weight_load_ = ActiveWeightLoad{};
    }

    void start_compute(std::uint64_t cycle) {
        active_compute_ = ActiveCompute{};
        active_compute_.active = true;
        active_compute_.start = cycle;
        active_compute_.m = static_cast<int>(*signals_.ctrl_rows);
        active_compute_.n = static_cast<int>(*signals_.ctrl_loaded_channels);
        active_compute_.g = static_cast<int>(*signals_.ctrl_groups);
        active_compute_.accum = *signals_.ctrl_accum != 0;
        active_compute_.int8 = *signals_.ctrl_int8 != 0;
        active_compute_.addr = *signals_.ctrl_address;
        row_counter_ = 0;
        current_row_index_ = -1;
        row_prep_active_ = false;
        ++compute_count_;
    }

    void finalize_compute(std::uint64_t cycle, bool error) {
        if (!active_compute_.active)
            return;
        if (active_group_.active)
            finalize_group(cycle);
        if (active_row_.active)
            finalize_row(cycle);
        TxRecord tx;
        tx.type = "compute";
        tx.start = active_compute_.start;
        tx.end = cycle;
        tx.label = "compute #" + std::to_string(compute_count_ - 1);
        std::ostringstream detail;
        detail << "M=" << active_compute_.m << " N=" << active_compute_.n
               << " G=" << active_compute_.g
               << " accum=" << (active_compute_.accum ? 1 : 0)
               << " int8=" << (active_compute_.int8 ? 1 : 0)
               << " rows=" << active_compute_.rows
               << " groups=" << active_compute_.groups
               << (error ? " error=1" : "");
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"address\":\"" << hex_value(active_compute_.addr)
             << "\",\"M\":" << active_compute_.m
             << ",\"N\":" << active_compute_.n
             << ",\"G\":" << active_compute_.g
             << ",\"accum\":" << (active_compute_.accum ? 1 : 0)
             << ",\"int8\":" << (active_compute_.int8 ? 1 : 0)
             << ",\"error\":" << (error ? 1 : 0);
        tx.args = args.str();
        record_tx(std::move(tx));
        active_compute_ = ActiveCompute{};
        row_prep_active_ = false;
    }

    void start_writeback(std::uint64_t cycle) {
        active_writeback_ = ActiveWriteback{};
        active_writeback_.active = true;
        active_writeback_.start = cycle;
        active_writeback_.m = static_cast<int>(*signals_.ctrl_result_rows);
        active_writeback_.n = static_cast<int>(*signals_.ctrl_result_channels);
        active_writeback_.g = (active_writeback_.n + 3) / 4;
        active_writeback_.addr = *signals_.ctrl_address;
        ++writeback_count_;
    }

    void finalize_writeback(std::uint64_t cycle, bool error) {
        if (!active_writeback_.active)
            return;
        if (active_write_row_.active)
            finalize_write_row(cycle);
        TxRecord tx;
        tx.type = "writeback";
        tx.start = active_writeback_.start;
        tx.end = cycle;
        tx.label = "writeback #" + std::to_string(writeback_count_ - 1);
        std::ostringstream detail;
        detail << "M=" << active_writeback_.m << " N=" << active_writeback_.n
               << " G=" << active_writeback_.g << " beats=" << active_writeback_.beats
               << (error ? " error=1" : "");
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"address\":\"" << hex_value(active_writeback_.addr)
             << "\",\"M\":" << active_writeback_.m
             << ",\"N\":" << active_writeback_.n
             << ",\"G\":" << active_writeback_.g
             << ",\"error\":" << (error ? 1 : 0);
        tx.args = args.str();
        record_tx(std::move(tx));
        active_writeback_ = ActiveWriteback{};
    }

    void start_write_row(std::uint64_t cycle) {
        if (!active_writeback_.active)
            return;
        active_write_row_ = ActiveWriteRow{};
        active_write_row_.active = true;
        active_write_row_.start = cycle;
        active_write_row_.index =
            active_writeback_.m - static_cast<int>(*signals_.wr_rows_left);
        active_write_row_.addr = *signals_.wr_row_addr;
        active_write_row_.groups = static_cast<int>(*signals_.wr_groups);
    }

    void finalize_write_row(std::uint64_t cycle) {
        if (!active_write_row_.active)
            return;
        TxRecord tx;
        tx.type = "write_row";
        tx.start = active_write_row_.start;
        tx.end = cycle;
        tx.label = "write row " + std::to_string(active_write_row_.index);
        std::ostringstream detail;
        detail << "addr=" << hex_value(active_write_row_.addr)
               << " G=" << active_write_row_.groups << " writer["
               << format_states(kWriterStates, 7, active_write_row_.states)
               << "] mem_stall=" << active_write_row_.mem_stall
               << " writer_gap=" << active_write_row_.writer_gap;
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"row\":" << active_write_row_.index
             << ",\"address\":\"" << hex_value(active_write_row_.addr)
             << "\",\"G\":" << active_write_row_.groups
             << ",\"mem_stall\":" << active_write_row_.mem_stall
             << ",\"writer_gap\":" << active_write_row_.writer_gap;
        tx.args = args.str();
        record_tx(std::move(tx));
        active_write_row_ = ActiveWriteRow{};
    }

    void start_row_prep(std::uint64_t cycle) {
        if (!active_compute_.active)
            return;
        row_prep_active_ = true;
        row_prep_start_ = cycle;
        row_prep_wait_array_ = 0;
        row_prep_array_starved_ = 0;
        row_prep_iq_states_.fill(0);
    }

    void start_compute_row(std::uint64_t cycle) {
        active_row_ = ActiveRow{};
        active_row_.active = true;
        active_row_.prep_start =
            row_prep_active_ ? row_prep_start_ : cycle;
        active_row_.prep_cycles =
            cycle >= active_row_.prep_start ? cycle - active_row_.prep_start : 0;
        active_row_.start = cycle;
        active_row_.index = row_counter_++;
        active_row_.m = active_compute_.m;
        active_row_.g = active_compute_.g;
        active_row_.int8 = active_compute_.int8;
        active_row_.iq_states = row_prep_iq_states_;
        active_row_.iqbuf_wait_array = row_prep_wait_array_;
        active_row_.array_starved = row_prep_array_starved_;
        current_row_index_ = active_row_.index;
        row_prep_active_ = false;
    }

    void finalize_row(std::uint64_t cycle) {
        if (!active_row_.active)
            return;
        TxRecord tx;
        tx.type = "compute_row";
        tx.start = active_row_.prep_start;
        tx.end = cycle;
        tx.label = "compute row " + std::to_string(active_row_.index);
        const std::uint64_t compute_cycles =
            cycle >= active_row_.start ? cycle - active_row_.start : 0;
        std::ostringstream detail;
        detail << "prep=" << active_row_.prep_cycles
               << " compute=" << compute_cycles << " iqbuf["
               << format_states(kIqbufStates, 7, active_row_.iq_states)
               << "] array["
               << format_states(kArrayStates, 5, active_row_.array_states)
               << "] unit["
               << format_states(kUnitStates, 6, active_row_.unit_states)
               << "] iqbuf_wait_array=" << active_row_.iqbuf_wait_array
               << " array_starved=" << active_row_.array_starved
               << " odma_wait=" << active_row_.odma_wait;
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"row\":" << active_row_.index
             << ",\"M\":" << active_row_.m
             << ",\"G\":" << active_row_.g
             << ",\"int8\":" << (active_row_.int8 ? 1 : 0)
             << ",\"prep_cycles\":" << active_row_.prep_cycles
             << ",\"compute_cycles\":" << compute_cycles
             << ",\"iqbuf_wait_array\":" << active_row_.iqbuf_wait_array
             << ",\"array_starved\":" << active_row_.array_starved
             << ",\"odma_wait\":" << active_row_.odma_wait;
        tx.args = args.str();
        record_tx(std::move(tx));
        active_row_ = ActiveRow{};
    }

    void start_group(std::uint64_t cycle) {
        active_group_ = ActiveGroup{};
        active_group_.active = true;
        active_group_.start = cycle;
        active_group_.row = current_row_index_;
        active_group_.group = static_cast<int>(*signals_.arr_group_index);
        const int remaining =
            active_compute_.n - active_group_.group * 4;
        active_group_.channels = std::max(0, std::min(4, remaining));
    }

    void finalize_group(std::uint64_t cycle) {
        if (!active_group_.active)
            return;
        TxRecord tx;
        tx.type = "compute_group";
        tx.start = active_group_.start;
        tx.end = cycle;
        tx.label = "compute row " + std::to_string(active_group_.row) +
                   " group " + std::to_string(active_group_.group);
        std::ostringstream detail;
        detail << "channels=" << active_group_.channels << " array["
               << format_states(kArrayStates, 5, active_group_.array_states)
               << "] unit["
               << format_states(kUnitStates, 6, active_group_.unit_states)
               << "] odma_wait=" << active_group_.odma_wait;
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"row\":" << active_group_.row
             << ",\"group\":" << active_group_.group
             << ",\"channels\":" << active_group_.channels
             << ",\"odma_wait\":" << active_group_.odma_wait;
        tx.args = args.str();
        record_tx(std::move(tx));
        active_group_ = ActiveGroup{};
    }

    void finalize_dma_read(std::uint64_t cycle) {
        if (!active_dma_read_.active)
            return;
        TxRecord tx;
        tx.type = "dma_read";
        tx.start = active_dma_read_.start;
        tx.end = cycle;
        tx.label = std::string("dma_read ") +
                   (active_dma_read_.source == 0 ? "weights" : "input");
        const std::uint64_t average_gap =
            active_dma_read_.beats > 1
                ? active_dma_read_.sum_gap /
                      static_cast<std::uint64_t>(active_dma_read_.beats - 1)
                : 0;
        std::ostringstream detail;
        detail << "source="
               << (active_dma_read_.source == 0 ? "weight_loader"
                                                : "input_quant_buffer")
               << " addr=" << hex_value(active_dma_read_.addr)
               << " len=" << active_dma_read_.len
               << " beats=" << active_dma_read_.beats
               << " first_beat_latency=" << active_dma_read_.first_beat_latency
               << " max_gap=" << active_dma_read_.max_gap
               << " avg_gap=" << average_gap
               << " consumer_stall=" << active_dma_read_.stall
               << " memory_starve=" << active_dma_read_.starve;
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"source\":\""
             << (active_dma_read_.source == 0 ? "weight_loader"
                                              : "input_quant_buffer")
             << "\",\"address\":\"" << hex_value(active_dma_read_.addr)
             << "\",\"len\":" << active_dma_read_.len
             << ",\"beats\":" << active_dma_read_.beats
             << ",\"first_beat_latency\":" << active_dma_read_.first_beat_latency
             << ",\"max_gap\":" << active_dma_read_.max_gap
             << ",\"avg_gap\":" << average_gap
             << ",\"consumer_stall\":" << active_dma_read_.stall
             << ",\"memory_starve\":" << active_dma_read_.starve;
        tx.args = args.str();
        record_tx(std::move(tx));
        active_dma_read_ = ActiveDmaRead{};
    }

    void finalize_dma_write(std::uint64_t cycle) {
        if (!active_dma_write_.active)
            return;
        TxRecord tx;
        tx.type = "dma_write";
        tx.start = active_dma_write_.start;
        tx.end = cycle;
        tx.label = "dma_write output";
        const std::uint64_t average_gap =
            active_dma_write_.beats > 1
                ? active_dma_write_.sum_gap /
                      static_cast<std::uint64_t>(active_dma_write_.beats - 1)
                : 0;
        const std::uint64_t done_latency =
            active_dma_write_.data_done >= active_dma_write_.start
                ? cycle - active_dma_write_.data_done
                : 0;
        std::ostringstream detail;
        detail << "addr=" << hex_value(active_dma_write_.addr)
               << " len=" << active_dma_write_.len
               << " beats=" << active_dma_write_.beats
               << " first_beat_latency=" << active_dma_write_.first_beat_latency
               << " max_gap=" << active_dma_write_.max_gap
               << " avg_gap=" << average_gap
               << " write_done_latency=" << done_latency
               << " memory_stall=" << active_dma_write_.stall
               << " writer_gap=" << active_dma_write_.starve;
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"address\":\"" << hex_value(active_dma_write_.addr)
             << "\",\"len\":" << active_dma_write_.len
             << ",\"beats\":" << active_dma_write_.beats
             << ",\"first_beat_latency\":" << active_dma_write_.first_beat_latency
             << ",\"max_gap\":" << active_dma_write_.max_gap
             << ",\"avg_gap\":" << average_gap
             << ",\"write_done_latency\":" << done_latency
             << ",\"memory_stall\":" << active_dma_write_.stall
             << ",\"writer_gap\":" << active_dma_write_.starve;
        tx.args = args.str();
        record_tx(std::move(tx));
        active_dma_write_ = ActiveDmaWrite{};
    }

    void finalize_command(std::uint64_t cycle, bool error) {
        if (!active_command_.active)
            return;
        TxRecord tx;
        tx.type = "command";
        tx.start = active_command_.start;
        tx.end = cycle;
        tx.label = std::string(active_command_.info.name) + " #" +
                   std::to_string(command_count_++);
        std::ostringstream detail;
        detail << "inst=" << hex_value(active_command_.inst)
               << " addr=" << hex_value(active_command_.info.address)
               << " M_field=" << active_command_.info.m
               << " N_field=" << active_command_.info.n
               << " accum=" << (active_command_.info.accum ? 1 : 0)
               << " int8=" << (active_command_.info.int8 ? 1 : 0)
               << " result=" << (error ? "error" : "done");
        tx.detail = detail.str();
        std::ostringstream args;
        args << "\"inst\":\"" << hex_value(active_command_.inst)
             << "\",\"opcode\":" << active_command_.info.opcode
             << ",\"address\":\"" << hex_value(active_command_.info.address)
             << "\",\"M_field\":" << active_command_.info.m
             << ",\"N_field\":" << active_command_.info.n
             << ",\"accum\":" << (active_command_.info.accum ? 1 : 0)
             << ",\"int8\":" << (active_command_.info.int8 ? 1 : 0)
             << ",\"error\":" << (error ? 1 : 0);
        tx.args = args.str();
        record_tx(std::move(tx));
        active_command_ = ActiveCommand{};
    }

    void write_bottleneck(std::ostream& out) const {
        const std::uint64_t total = std::max<std::uint64_t>(1, total_cycles_);
        auto percentage = [total](std::uint64_t value) {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(2)
                   << (100.0 * static_cast<double>(value) /
                       static_cast<double>(total));
            return stream.str();
        };
        auto line = [&](int phase, const std::string& label,
                        std::uint64_t stall_cycles) {
            out << "  " << std::left << std::setw(8) << kPhaseNames[phase]
                << std::right << std::setw(6) << phase_cycles_[phase]
                << " cycles (" << percentage(phase_cycles_[phase])
                << "%): " << label;
            if (stall_cycles != 0)
                out << " (" << stall_cycles << " stall cycles)";
            out << '\n';
        };

        out << "bottleneck:\n";
        if (phase_cycles_[PHASE_LOAD] != 0) {
            std::uint64_t best = attrib_load_data_stall_;
            std::string label = "weight_loader unpack/self-limit";
            if (attrib_load_data_starve_ > best) {
                best = attrib_load_data_starve_;
                label = "external memory read latency/bandwidth";
            }
            if (attrib_load_cmd_wait_ > best) {
                best = attrib_load_cmd_wait_;
                label = "DMA read command acceptance";
            }
            if (best == 0)
                label = "no explicit stall (throughput-bound)";
            line(PHASE_LOAD, label, best);
        }
        if (phase_cycles_[PHASE_COMPUTE] != 0) {
            std::uint64_t best = attrib_compute_array_starved_;
            std::string label =
                "input_quant_buffer (array starved waiting for input)";
            if (attrib_compute_iqbuf_blocked_ > best) {
                best = attrib_compute_iqbuf_blocked_;
                label = "core_array (iqbuf blocked waiting for array)";
            }
            if (attrib_compute_odma_stall_ > best) {
                best = attrib_compute_odma_stall_;
                label = "output_post_sram (odma backpressure)";
            }
            if (best == 0)
                label = "no explicit stall (throughput-bound)";
            line(PHASE_COMPUTE, label, best);
        }
        if (phase_cycles_[PHASE_WRITE] != 0) {
            std::uint64_t best = attrib_write_mem_stall_;
            std::string label = "external memory write backpressure";
            if (attrib_write_writer_gap_ > best) {
                best = attrib_write_writer_gap_;
                label = "output_writer SRAM/read gap";
            }
            if (attrib_write_cmd_wait_ > best) {
                best = attrib_write_cmd_wait_;
                label = "DMA write command acceptance";
            }
            if (best == 0)
                label = "no explicit stall (throughput-bound)";
            line(PHASE_WRITE, label, best);
        }
    }

    void write_folded(const std::filesystem::path& path) const {
        std::map<std::string, std::uint64_t> folded;
        for (int phase = 0; phase < kPhaseCount; ++phase) {
            for (int mod = 0; mod < kModuleCount; ++mod) {
                const auto& module = modules_[mod];
                for (int state = 0; state < module.state_count; ++state) {
                    const std::uint64_t cycles =
                        phase_module_state_[pms_index(phase, mod, state)];
                    if (cycles == 0)
                        continue;
                    folded["bnn_sim_top;" + std::string(kPhaseNames[phase]) +
                           ";" + module.folded_name + ";" + module.states[state]] +=
                        cycles;
                }
            }
        }
        for (int mod = 0; mod < kModuleCount; ++mod) {
            const auto& module = modules_[mod];
            for (int state = 0; state < module.state_count; ++state) {
                if (module.cycles[state] == 0)
                    continue;
                folded["bnn_sim_top;TOTAL;" + module.folded_name + ";" +
                       module.states[state]] += module.cycles[state];
            }
        }
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot create " + path.string());
        for (const auto& entry : folded)
            out << entry.first << ' ' << entry.second << '\n';
    }

    void write_text_report(const std::filesystem::path& path,
                           const std::filesystem::path& folded_path,
                           int top_n) const {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot create " + path.string());

        const std::uint64_t total = std::max<std::uint64_t>(1, total_cycles_);
        auto percent = [total](std::uint64_t value) {
            return 100.0 * static_cast<double>(value) /
                   static_cast<double>(total);
        };

        out << "BNN v2 performance profile\n";
        out << "==========================\n";
        out << "simulation cycles : " << total_cycles_ << '\n';
        out << "accepted commands : " << accepted_commands_ << '\n';
        out << "idma handshakes   : " << idma_handshakes_ << '\n';
        out << "odma handshakes   : " << odma_handshakes_ << '\n';
        out << "dma read beats    : " << mem_rd_beats_ << '\n';
        out << "dma write beats   : " << mem_wr_beats_ << '\n';
        out << "\nPhase breakdown\n";
        out << "---------------\n";
        out << std::left << std::setw(10) << "phase" << std::right
            << std::setw(12) << "cycles" << std::setw(10) << "share" << '\n';
        for (int phase = 0; phase < kPhaseCount; ++phase) {
            out << std::left << std::setw(10) << kPhaseNames[phase]
                << std::right << std::setw(12) << phase_cycles_[phase]
                << std::setw(9) << std::fixed << std::setprecision(2)
                << percent(phase_cycles_[phase]) << "%\n";
        }

        out << "\nBottleneck hint\n";
        out << "---------------\n";
        write_bottleneck(out);

        out << "\nModule busy time (busy = state != IDLE; phases overlap)\n";
        out << "Note: busy includes waiting for the other side.  For example,\n";
        out << "      u_output_accum RECEIVE can be waiting for odma_valid.\n";
        out << "      Use the stall attribution below to tell work from waiting.\n";
        out << "--------------------------------------------------------\n";
        out << std::left << std::setw(24) << "module" << std::right
            << std::setw(12) << "busy" << std::setw(10) << "share"
            << std::setw(12) << "idle" << '\n';
        for (const auto& module : modules_) {
            const std::uint64_t busy = busy_cycles(module);
            const std::uint64_t idle = module.cycles[0];
            out << std::left << std::setw(24) << module.name << std::right
                << std::setw(12) << busy << std::setw(9) << std::fixed
                << std::setprecision(2) << percent(busy) << "%"
                << std::setw(12) << idle << '\n';
        }

        out << "\nModule state breakdown\n";
        out << "----------------------\n";
        for (const auto& module : modules_) {
            const std::uint64_t busy = busy_cycles(module);
            out << '[' << module.name << "] busy=" << busy
                << " idle=" << module.cycles[0] << '\n';
            for (int state = 0; state < module.state_count; ++state) {
                if (module.cycles[state] == 0)
                    continue;
                out << "  " << std::left << std::setw(18) << module.states[state]
                    << std::right << std::setw(12) << module.cycles[state]
                    << " cycles, " << std::setw(8) << module.entries[state]
                    << " entries\n";
            }
        }

        out << "\nStall attribution\n";
        out << "-----------------\n";
        out << "LOAD phase (" << phase_cycles_[PHASE_LOAD] << " cycles)\n";
        out << "  memory data valid, wloader not ready : "
            << attrib_load_data_stall_ << " cycles (wloader unpack/self limit)\n";
        out << "  wloader ready, memory data missing  : "
            << attrib_load_data_starve_ << " cycles (memory latency/bandwidth)\n";
        out << "  read command not accepted           : "
            << attrib_load_cmd_wait_ << " cycles\n";
        out << "COMPUTE phase (" << phase_cycles_[PHASE_COMPUTE] << " cycles)\n";
        out << "  iqbuf blocked by array              : "
            << attrib_compute_iqbuf_blocked_ << " cycles (array/compute limited)\n";
        out << "  array starved by iqbuf              : "
            << attrib_compute_array_starved_ << " cycles (iqbuf limited)\n";
        out << "  array blocked by post_sram          : "
            << attrib_compute_odma_stall_ << " cycles (post/backpressure limited)\n";
        out << "WRITE phase (" << phase_cycles_[PHASE_WRITE] << " cycles)\n";
        out << "  writer data valid, memory not ready : "
            << attrib_write_mem_stall_ << " cycles (memory backpressure)\n";
        out << "  memory ready, writer data missing   : "
            << attrib_write_writer_gap_ << " cycles (writer SRAM/read gap)\n";
        out << "  write command not accepted          : "
            << attrib_write_cmd_wait_ << " cycles\n";

        std::map<std::string, std::vector<const TxRecord*>> by_type;
        for (const auto& tx : transactions_)
            by_type[tx.type].push_back(&tx);

        out << "\nTransaction summary (elapsed cycles, transactions may overlap)\n";
        out << "----------------------------------------------------------------\n";
        out << std::left << std::setw(16) << "type" << std::right
            << std::setw(8) << "count" << std::setw(12) << "total"
            << std::setw(12) << "avg" << std::setw(12) << "max" << '\n';
        for (const auto& entry : by_type) {
            const auto& records = entry.second;
            std::uint64_t total_cycles = 0;
            std::uint64_t max_cycles = 0;
            for (const auto* tx : records) {
                total_cycles += tx->duration();
                max_cycles = std::max(max_cycles, tx->duration());
            }
            const std::uint64_t avg =
                records.empty() ? 0 : total_cycles / records.size();
            out << std::left << std::setw(16) << entry.first << std::right
                << std::setw(8) << records.size() << std::setw(12)
                << total_cycles << std::setw(12) << avg << std::setw(12)
                << max_cycles << '\n';
        }

        std::vector<const TxRecord*> sorted;
        sorted.reserve(transactions_.size());
        for (const auto& tx : transactions_)
            sorted.push_back(&tx);
        std::sort(sorted.begin(), sorted.end(),
                  [](const TxRecord* lhs, const TxRecord* rhs) {
                      return lhs->duration() > rhs->duration();
                  });
        if (top_n > 0 && sorted.size() > static_cast<std::size_t>(top_n))
            sorted.resize(static_cast<std::size_t>(top_n));

        out << "\nSlowest transactions\n";
        out << "--------------------\n";
        int rank = 1;
        for (const auto* tx : sorted) {
            out << rank++ << ". " << tx->type << " / " << tx->label << " : "
                << tx->duration() << " cycles [" << tx->start << ", "
                << tx->end << "]\n";
            if (!tx->detail.empty())
                out << "     " << tx->detail << '\n';
        }

        out << "\nSlowest transaction per type\n";
        out << "----------------------------\n";
        for (const auto& entry : by_type) {
            auto records = entry.second;
            std::sort(records.begin(), records.end(),
                      [](const TxRecord* lhs, const TxRecord* rhs) {
                          return lhs->duration() > rhs->duration();
                      });
            const std::size_t limit = std::min<std::size_t>(3, records.size());
            out << '[' << entry.first << "]\n";
            for (std::size_t i = 0; i < limit; ++i) {
                const auto* tx = records[i];
                out << "  " << tx->label << " : " << tx->duration()
                    << " cycles [" << tx->start << ", " << tx->end << "]\n";
                if (!tx->detail.empty())
                    out << "      " << tx->detail << '\n';
            }
        }

        out << "\nHow to view\n";
        out << "-----------\n";
        out << "Chrome trace : open " << trace_path_.string()
            << " in chrome://tracing or https://ui.perfetto.dev\n";
        out << "Folded stacks: flamegraph.pl " << folded_path.string()
            << " > flame.svg\n";
        out << "Trace ts/dur are simulation cycles (displayed as ns).\n";
        out << "The trace has one track per module FSM plus command, DMA, row, "
               "group and write-row transaction tracks.\n";
    }
};

void tick_profiled(Vbnn_sim_top& dut, VerilatedContext& context,
                   Profiler* profiler, std::uint64_t cycle) {
    dut.clock = 0;
    dut.eval();
    if (profiler)
        profiler->sample_pre(cycle);
    dut.clock = 1;
    dut.eval();
    const bool finished = context.gotFinish();
    if (profiler)
        profiler->sample_post(cycle + 1, !finished);
    dut.clock = 0;
    dut.eval();
}

} // namespace

// -------------------------------------------------------------------------
// DPI-C virtual memories
// -------------------------------------------------------------------------
extern "C" int cmd_check() {
    return g_command_cursor < g_commands.size();
}

extern "C" unsigned long long cmd_get() {
    if (g_command_cursor >= g_commands.size())
        dpi_fatal("cmd_get() called with no pending instruction");
    return g_commands[g_command_cursor++];
}

extern "C" unsigned long long data_read(unsigned long long address) {
    if ((address & 7ull) != 0)
        dpi_fatal("data_read() address is not 8-byte aligned");
    if (!range_ok(address, 8))
        dpi_range_fatal("data_read()", address);

    std::uint64_t value = 0;
    for (int b = 0; b < 8; ++b)
        value |= std::uint64_t{g_memory[address + b]} << (8 * b);
    return value;
}

// The file-driven testbench reads words directly from g_memory; the DMA burst
// prefetch hook only exists for network-backed testbenches.
extern "C" void data_read_burst(unsigned long long address,
                                unsigned long long bytes) {
    (void)address;
    (void)bytes;
}

extern "C" void data_write(unsigned long long address, unsigned long long data,
                           unsigned char strobe) {
    if ((address & 7ull) != 0)
        dpi_fatal("data_write() address is not 8-byte aligned");
    if (!range_ok(address, 8))
        dpi_range_fatal("data_write()", address);

    for (int b = 0; b < 8; ++b) {
        if (strobe & (1u << b))
            g_memory[address + b] = static_cast<std::uint8_t>(data >> (8 * b));
    }
    g_highest_write_end = std::max(
        g_highest_write_end, static_cast<std::uint64_t>(address + 8));
}

// -------------------------------------------------------------------------
struct Options {
    std::filesystem::path instructions;
    std::filesystem::path memory;
    std::filesystem::path trace_path;
    std::filesystem::path folded_path;
    std::filesystem::path report_path;
    bool write_model = true;
    bool profile = true;
    int top_n = 10;
};

void print_usage(const char* program) {
    std::cerr
        << "usage: " << program << " INSTRUCTIONS MEMORY [options]\n"
        << "  INSTRUCTIONS  64-bit hex text (one per line, # comments)\n"
        << "                or raw little-endian binary words\n"
        << "  MEMORY        raw binary memory image loaded at address 0\n"
        << "  default       MEMORY stem + \"_out\" + extension\n"
        << "options:\n"
        << "  --no-model        do not write the output memory image\n"
        << "  --no-profile      disable trace/folded/report generation\n"
        << "  --trace PATH      Chrome trace JSON (default <stem>_trace.json)\n"
        << "  --folded PATH     folded stacks (default <stem>_flame.folded)\n"
        << "  --report PATH     text report (default <stem>_profile.txt)\n"
        << "  --top N           slowest transactions shown in report (default 10)\n"
        << "  -h, --help        show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--no-model") {
            options.write_model = false;
        } else if (argument == "--no-profile") {
            options.profile = false;
        } else if (argument == "--top") {
            if (++i >= argc)
                throw std::runtime_error("--top requires a value");
            options.top_n = std::stoi(argv[i]);
        } else if (argument == "--trace") {
            if (++i >= argc)
                throw std::runtime_error("--trace requires a path");
            options.trace_path = argv[i];
        } else if (argument == "--folded") {
            if (++i >= argc)
                throw std::runtime_error("--folded requires a path");
            options.folded_path = argv[i];
        } else if (argument == "--report") {
            if (++i >= argc)
                throw std::runtime_error("--report requires a path");
            options.report_path = argv[i];
        } else if (!argument.empty() && argument[0] == '+') {
            continue;
        } else if (!argument.empty() && argument[0] == '-') {
            throw std::runtime_error("unknown option " + argument);
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.size() != 2) {
        print_usage(argv[0]);
        std::exit(2);
    }
    options.instructions = positional[0];
    options.memory = positional[1];

    const std::filesystem::path base_dir = options.memory.parent_path();
    const std::string stem = options.memory.stem().string();
    if (options.trace_path.empty())
        options.trace_path = base_dir / (stem + "_trace.json");
    if (options.folded_path.empty())
        options.folded_path = base_dir / (stem + "_flame.folded");
    if (options.report_path.empty())
        options.report_path = base_dir / (stem + "_profile.txt");
    return options;
}

// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        g_commands = load_commands(options.instructions);
        g_memory = load_memory(options.memory);

        std::cout << "commands: " << g_commands.size()
                  << ", memory image: " << g_initial_memory_size
                  << " bytes (capacity " << g_memory.size() << ")\n";

        VerilatedContext context;
        context.commandArgs(1, argv);
        Vbnn_sim_top dut(&context);

        dut.clock = 0;
        dut.reset = 1;
        dut.finish = 0;
        for (int i = 0; i < 4; ++i)
            tick(dut);
        dut.reset = 0;
        dut.eval();

        std::unique_ptr<Profiler> profiler;
        if (options.profile) {
            profiler = std::make_unique<Profiler>(dut, options.trace_path, true);
            profiler->sample_post(0, true);
        }

        std::uint64_t cycles = 0;
        while (!context.gotFinish() && cycles < kCycleLimit) {
            dut.finish = g_command_cursor >= g_commands.size() ? 1 : 0;
            tick_profiled(dut, context, profiler.get(), cycles);
            if (dut.cmd_error_out)
                throw std::runtime_error("core reported cmd_error");
            ++cycles;
        }
        if (!context.gotFinish())
            throw std::runtime_error("simulation did not finish within " +
                                     std::to_string(kCycleLimit) + " cycles");

        dut.final();

        if (profiler) {
            profiler->finish(cycles);
            profiler->write_reports(options.folded_path, options.report_path,
                                    options.top_n);
        }

        std::size_t output_size = 0;
        std::filesystem::path output_path;
        if (options.write_model) {
            output_path = output_path_for(options.memory);
            output_size = std::min<std::size_t>(
                g_memory.size(),
                std::max<std::size_t>(
                    g_initial_memory_size,
                    static_cast<std::size_t>(g_highest_write_end)));

            std::ofstream output(output_path,
                                 std::ios::binary | std::ios::trunc);
            if (!output)
                throw std::runtime_error("cannot create " +
                                         output_path.string());
            if (output_size != 0 &&
                !output.write(reinterpret_cast<const char*>(g_memory.data()),
                              static_cast<std::streamsize>(output_size))) {
                throw std::runtime_error("cannot write " +
                                         output_path.string());
            }
            if (!output)
                throw std::runtime_error("cannot finish writing " +
                                         output_path.string());
        }

        std::cout << "finished: " << g_command_cursor << " commands, "
                  << cycles << " cycles";
        if (options.write_model)
            std::cout << " -> " << output_path.string() << " ("
                      << output_size << " bytes)";
        std::cout << '\n';
        if (!options.write_model)
            std::cout << "model output: skipped (--no-model)\n";
        if (!options.profile)
            std::cout << "profile: disabled (--no-profile)\n";
        if (profiler) {
            profiler->print_summary(std::cout);
            std::cout << "trace  : " << options.trace_path.string() << '\n'
                      << "folded : " << options.folded_path.string() << '\n'
                      << "report : " << options.report_path.string() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "simple_test_main: " << error.what() << '\n';
        return 1;
    }
}
