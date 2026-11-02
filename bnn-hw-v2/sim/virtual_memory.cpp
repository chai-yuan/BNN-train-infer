#include "virtual_memory.hpp"

#include <svdpi.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace bnn::sim {
namespace {
std::array<VirtualMemory, 2> memories;
std::string last_error;
}

void VirtualMemory::configure(std::size_t size, std::uint64_t base, std::uint8_t fill) {
    if (size != 0 && size - 1 > std::numeric_limits<std::uint64_t>::max() - base) {
        throw std::invalid_argument("virtual memory address range wraps uint64_t");
    }
    std::vector<std::uint8_t> replacement(size, fill);
    bytes_.swap(replacement);
    base_ = base;
}

void VirtualMemory::check(std::uint64_t address, std::uint64_t size) const {
    if (address < base_ || address - base_ > bytes_.size() ||
        size > bytes_.size() - (address - base_)) {
        std::ostringstream message;
        message << "virtual memory access out of range: address=0x" << std::hex << address
                << " bytes=0x" << size << " base=0x" << base_ << " capacity=0x" << bytes_.size();
        throw std::out_of_range(message.str());
    }
}

std::vector<std::uint8_t> VirtualMemory::read(std::uint64_t address, std::size_t size) const {
    check(address, size);
    const auto offset = static_cast<std::size_t>(address - base_);
    return {bytes_.begin() + offset, bytes_.begin() + offset + size};
}

void VirtualMemory::write(std::uint64_t address, const std::vector<std::uint8_t>& data) {
    check(address, data.size());
    std::copy(data.begin(), data.end(), bytes_.begin() + static_cast<std::size_t>(address - base_));
}

void VirtualMemory::write_masked(std::uint64_t address, const std::vector<std::uint8_t>& data,
                                 const std::vector<std::uint8_t>& strobe) {
    if (data.size() != strobe.size()) {
        throw std::invalid_argument("write data/strobe byte counts differ");
    }
    check(address, data.size());
    const auto offset = static_cast<std::size_t>(address - base_);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (strobe[i] != 0) {
            bytes_[offset + i] = data[i];
        }
    }
}

void VirtualMemory::load_binary(const std::string& path, std::uint64_t address) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open input file: " + path);
    }
    const std::vector<std::uint8_t> data{std::istreambuf_iterator<char>(file), {}};
    if (file.bad()) {
        throw std::runtime_error("cannot read input file: " + path);
    }
    write(address, data);
}

void VirtualMemory::dump_binary(const std::string& path, std::uint64_t address,
                                std::size_t size) const {
    const auto data = read(address, size);
    std::ofstream file(path, std::ios::binary);
    if (!file || !file.write(reinterpret_cast<const char*>(data.data()),
                             static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("cannot write output file: " + path);
    }
    file.close();
    if (!file) {
        throw std::runtime_error("cannot close output file: " + path);
    }
}

VirtualMemory& memory(unsigned id) {
    if (id >= memories.size()) {
        throw std::out_of_range("unknown DPI memory id " + std::to_string(id));
    }
    return memories[id];
}

const std::string& dpi_error() { return last_error; }
void clear_dpi_error() { last_error.clear(); }

// Exceptions must not cross the DPI ABI. The SV caller turns a failure into $fatal.
template <class Function>
int dpi_call(Function function) noexcept {
    try {
        function();
        return 0;
    } catch (const std::exception& error) {
        if (last_error.empty()) {
            last_error = error.what();
            std::cerr << "DPI memory error: " << last_error << '\n';
        }
        return -1;
    }
}

std::size_t array_size(const svOpenArrayHandle array) {
    if (!array || svDimensions(array) != 1 || svSize(array, 1) <= 0) {
        throw std::invalid_argument("DPI expects a nonempty one-dimensional byte array");
    }
    return static_cast<std::size_t>(svSize(array, 1));
}

std::uint8_t& array_byte(const svOpenArrayHandle array, std::size_t offset) {
    const int index = svLeft(array, 1) + static_cast<int>(offset) *
                                           (svLeft(array, 1) <= svRight(array, 1) ? 1 : -1);
    auto* pointer = static_cast<std::uint8_t*>(svGetArrElemPtr1(array, index));
    if (!pointer) {
        throw std::runtime_error("DPI byte array element is inaccessible");
    }
    return *pointer;
}
} // namespace bnn::sim

extern "C" int bnn_mem_check(int id, unsigned long long address, unsigned long long size) {
    return bnn::sim::dpi_call([&] { bnn::sim::memory(id).check(address, size); });
}

extern "C" int bnn_mem_read(int id, unsigned long long address, const svOpenArrayHandle data) {
    return bnn::sim::dpi_call([&] {
        const auto bytes = bnn::sim::memory(id).read(address, bnn::sim::array_size(data));
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bnn::sim::array_byte(data, i) = bytes[i];
        }
    });
}

extern "C" int bnn_mem_write(int id, unsigned long long address, const svOpenArrayHandle data,
                             const svOpenArrayHandle strobe) {
    return bnn::sim::dpi_call([&] {
        const auto size = bnn::sim::array_size(data);
        if (size != bnn::sim::array_size(strobe)) {
            throw std::invalid_argument("DPI data/strobe array sizes differ");
        }
        std::vector<std::uint8_t> bytes(size), mask(size);
        for (std::size_t i = 0; i < size; ++i) {
            bytes[i] = bnn::sim::array_byte(data, i);
            mask[i] = bnn::sim::array_byte(strobe, i);
        }
        bnn::sim::memory(id).write_masked(address, bytes, mask);
    });
}
