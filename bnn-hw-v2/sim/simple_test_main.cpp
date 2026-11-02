#include "Vbnn_sim_top.h"
#include "verilated.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " INSTRUCTIONS MEMORY\n"
                  << "  INSTRUCTIONS  64-bit hex text (one per line, # comments)\n"
                  << "                or raw little-endian binary words\n"
                  << "  MEMORY        raw binary memory image loaded at address 0\n"
                  << "  output        MEMORY stem + \"_out\" + extension\n";
        return 2;
    }

    try {
        const std::filesystem::path instruction_path = argv[1];
        const std::filesystem::path memory_path = argv[2];

        g_commands = load_commands(instruction_path);
        g_memory = load_memory(memory_path);

        std::cout << "commands: " << g_commands.size()
                  << ", memory image: " << g_initial_memory_size
                  << " bytes (capacity " << g_memory.size() << ")\n";

        VerilatedContext context;
        context.commandArgs(argc, argv);
        Vbnn_sim_top dut(&context);

        dut.clock = 0;
        dut.reset = 1;
        for (int i = 0; i < 4; ++i)
            tick(dut);
        dut.reset = 0;
        dut.eval();

        std::uint64_t cycles = 0;
        while (!context.gotFinish() && cycles < kCycleLimit) {
            tick(dut);
            ++cycles;
        }
        if (!context.gotFinish())
            throw std::runtime_error("simulation did not finish within " +
                                     std::to_string(kCycleLimit) + " cycles");

        dut.final();

        const std::filesystem::path output_path = output_path_for(memory_path);
        const std::size_t output_size = std::min<std::size_t>(
            g_memory.size(),
            std::max<std::size_t>(g_initial_memory_size,
                                  static_cast<std::size_t>(g_highest_write_end)));

        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("cannot create " + output_path.string());
        if (output_size != 0 &&
            !output.write(reinterpret_cast<const char*>(g_memory.data()),
                          static_cast<std::streamsize>(output_size))) {
            throw std::runtime_error("cannot write " + output_path.string());
        }
        if (!output)
            throw std::runtime_error("cannot finish writing " + output_path.string());

        std::cout << "finished: " << g_command_cursor << " commands, " << cycles
                  << " cycles -> " << output_path.string() << " ("
                  << output_size << " bytes)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "simple_test_main: " << error.what() << '\n';
        return 1;
    }
}
