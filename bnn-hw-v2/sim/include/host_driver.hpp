#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace bnn::sim {

class Instruction {
public:
    static constexpr std::uint64_t set_source_stride(std::uint32_t stride) {
        return aligned_high(stride) | 0x01u;
    }

    static constexpr std::uint64_t set_destination_stride(std::uint32_t stride) {
        return aligned_high(stride) | 0x02u;
    }

    static constexpr std::uint64_t load_weights(std::uint32_t address,
                                               std::uint16_t n) {
        return aligned_high(address) | (dimension(n) << 16) | 0x10u;
    }

    static constexpr std::uint64_t compute(std::uint32_t address, std::uint16_t m,
                                         bool accum = false, bool int8 = false) {
        return aligned_high(address) | (dimension(m) << 8) |
               (std::uint64_t{accum} << 24) | (std::uint64_t{int8} << 25) | 0x20u;
    }

    static constexpr std::uint64_t writeback(std::uint32_t address) {
        return aligned_high(address) | 0x30u;
    }

private:
    static constexpr std::uint64_t aligned_high(std::uint32_t value) {
        if ((value & 0x0fu) != 0) {
            throw std::invalid_argument("Instruction: address/stride must be 16-byte aligned");
        }
        return std::uint64_t{value} << 32;
    }

    static constexpr std::uint64_t dimension(std::uint16_t value) {
        // 这里只验证 ISA 的表示范围；DUT 配置的 MAX_ROWS/MAX_OUT_CHANS 可能更小。
        if (value == 0 || value > 255) {
            throw std::invalid_argument("Instruction: M/N must be in [1, 255]");
        }
        return value;
    }
};

enum class CommandResult { done, error };

// DUT 提供 cmd_valid/cmd_inst/cmd_ready/cmd_done/cmd_error 及 eval()，无需包含
// Verilator 头文件。status_busy 不参与握手或完成判定。驱动器独占命令接口，单指令在途。
// tick() 推进一个完整周期，并在各时钟沿求值；调用前后均须 clock=0 且组合逻辑已稳定。
// 可使用 C++17 CTAD：HostDriver driver(dut, [&] { tick(); });
// execute 返回 error 不抛异常；编码参数错误抛 invalid_argument，超时抛 runtime_error。
template <class Dut, class Tick>
class HostDriver {
public:
    HostDriver(Dut& dut, Tick tick) : dut_(dut), tick_(std::move(tick)) {}

    // 接收预算包含清除上条指令残留完成脉冲所需的空闲周期；完成预算从握手后开始。
    // 两个预算均按 tick() 次数计，允许为零；握手当拍完成不消耗完成等待预算。
    // instruction 不做合法性检查，以允许发送 raw 非法指令用于负测。
    [[nodiscard]] CommandResult execute(std::uint64_t instruction,
                                       std::uint64_t accept_cycles = 1000,
                                       std::uint64_t completion_cycles = 100000) {
        release_command();
        std::uint64_t elapsed = 0;
        // 连续 execute 时，不能把上一条指令尚未撤销的 done/error 当成本条的结果。
        while (dut_.cmd_done || dut_.cmd_error) {
            if (elapsed == accept_cycles) {
                throw std::runtime_error(
                    "HostDriver: reception/acceptance timeout waiting for previous done/error to clear");
            }
            tick_();
            ++elapsed;
        }

        dut_.cmd_inst = instruction;
        dut_.cmd_valid = 1;
        dut_.eval(); // ready/done/error 可能组合依赖 valid，首次检查前必须 settle。

        bool accepted = false;
        bool done = false;
        bool error = false;
        const auto sample_result = [&] {
            error = error || static_cast<bool>(dut_.cmd_error);
            done = done || static_cast<bool>(dut_.cmd_done);
        };

        while (elapsed < accept_cycles) {
            // 只能用沿前 valid && ready 判断本次 tick 是否接收；沿后 ready 可能已改变。
            accepted = dut_.cmd_valid && dut_.cmd_ready;
            if (accepted) {
                sample_result();
            }
            tick_();
            ++elapsed;
            if (accepted) {
                sample_result(); // 撤销 valid/eval 前保存紧随接收沿后的单拍结果。
                break;
            }
        }

        release_command(); // 直到接收沿已执行（或超时）才撤销 valid；cmd_inst 始终不变。
        if (!accepted) {
            throw std::runtime_error("HostDriver: reception/acceptance timeout waiting for cmd_ready handshake");
        }
        sample_result();

        for (std::uint64_t waited = 0;; ++waited) {
            // 合并沿前、沿后及撤销 valid 后的观测，任何 error 都优先于 done。
            if (error) {
                return CommandResult::error;
            }
            if (done) {
                return CommandResult::done;
            }
            if (waited == completion_cycles) {
                throw std::runtime_error("HostDriver: completion timeout waiting for cmd_done/cmd_error");
            }
            sample_result();
            tick_();
            sample_result();
        }
    }

private:
    void release_command() {
        dut_.cmd_valid = 0;
        dut_.eval();
    }

    Dut& dut_;
    Tick tick_;
};

} // namespace bnn::sim
