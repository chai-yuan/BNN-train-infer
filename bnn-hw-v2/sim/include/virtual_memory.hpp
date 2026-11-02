#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bnn::sim {

// DMA uses byte addresses; base is a simulated address, never a host pointer.
// Configure/load while idle. A presented read beat is a snapshot until accepted.
class VirtualMemory {
public:
    void configure(std::size_t size, std::uint64_t base = 0, std::uint8_t fill = 0);
    void check(std::uint64_t address, std::uint64_t size) const;
    std::vector<std::uint8_t> read(std::uint64_t address, std::size_t size) const;
    void write(std::uint64_t address, const std::vector<std::uint8_t>& data);
    void write_masked(std::uint64_t address, const std::vector<std::uint8_t>& data,
                      const std::vector<std::uint8_t>& strobe);
    void load_binary(const std::string& path, std::uint64_t address);
    void dump_binary(const std::string& path, std::uint64_t address, std::size_t size) const;
    std::uint64_t base() const { return base_; }
    std::size_t size() const { return bytes_.size(); }

private:
    std::uint64_t base_ = 0;
    std::vector<std::uint8_t> bytes_;
};

constexpr unsigned kReadMemory = 0;
constexpr unsigned kWriteMemory = 1;
// Independent stores, even at equal simulated addresses. No hidden allocation.
VirtualMemory& memory(unsigned id);
const std::string& dpi_error();
void clear_dpi_error();

} // namespace bnn::sim
