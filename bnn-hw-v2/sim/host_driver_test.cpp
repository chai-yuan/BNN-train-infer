#include "include/host_driver.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using bnn::sim::CommandResult;
using bnn::sim::HostDriver;
using bnn::sim::Instruction;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Exception, class Action>
std::string expect_throw(Action action) {
    try {
        action();
    } catch (const Exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected exception was not thrown");
}

enum class Reply { none, done, error, both };

bool has_done(Reply reply) {
    return reply == Reply::done || reply == Reply::both;
}

bool has_error(Reply reply) {
    return reply == Reply::error || reply == Reply::both;
}

CommandResult expected_result(Reply reply) {
    require(reply != Reply::none, "test requires a terminal reply");
    return has_error(reply) ? CommandResult::error : CommandResult::done;
}

struct Command {
    std::uint64_t instruction;
    unsigned ready_delay = 0;
    bool never_ready = false;
    Reply before_edge = Reply::none;
    Reply after_edge = Reply::done;
    unsigned completion_delay = 0;
    bool post_reply_requires_valid = false;
    Reply on_release = Reply::none;
};

// One configured command per execute. configure() preserves the previous registered
// pulse so consecutive-execute tests can detect reuse of stale done/error.
class FakeDut {
public:
    bool clock = false;
    bool cmd_valid = false;
    std::uint64_t cmd_inst = 0;
    bool cmd_ready = false;
    bool cmd_done = false;
    bool cmd_error = false;
    bool status_busy = false;

    unsigned cycles = 0;
    unsigned valid_cycles = 0;
    unsigned stalled_cycles = 0;
    std::vector<std::uint64_t> received;

    explicit FakeDut(Command command) : command_(command), ready_at_(command.ready_delay) {
        eval();
    }

    void configure(Command command) {
        require(!cmd_valid && pending_cycles_ == 0, "cannot reconfigure an active command");
        command_ = command;
        ready_at_ = cycles + command.ready_delay;
        accepted_ = false;
        release_reply_ = Reply::none;
        eval();
    }

    void eval() {
        // Deliberately depend on valid: the driver must settle after driving it.
        cmd_ready = cmd_valid && !accepted_ && !command_.never_ready && cycles >= ready_at_;
        const Reply before = cmd_ready ? command_.before_edge : Reply::none;
        const Reply after = command_.post_reply_requires_valid && !cmd_valid
                                ? Reply::none : registered_reply_;
        const Reply released = !cmd_valid ? release_reply_ : Reply::none;
        cmd_done = has_done(before) || has_done(after) || has_done(released);
        cmd_error = has_error(before) || has_error(after) || has_error(released);
    }

    void tick() {
        require(!clock, "tick must start with clock=0");
        // Also bound failing timeout tests if a driver regression loses its budget.
        require(cycles < 100, "fake-DUT watchdog: driver exceeded 100 cycles");
        const bool handshake = cmd_valid && cmd_ready;
        if (cmd_valid) {
            require(!accepted_, "cmd_valid was not lowered after acceptance");
            require(cmd_inst == command_.instruction, "cmd_inst changed while waiting for ready");
            ++valid_cycles;
            if (!cmd_ready) {
                ++stalled_cycles;
            }
        }

        clock = true;
        ++cycles;
        registered_reply_ = Reply::none;
        release_reply_ = Reply::none;
        if (pending_cycles_ != 0 && --pending_cycles_ == 0) {
            registered_reply_ = command_.after_edge;
        }
        if (handshake) {
            accepted_ = true; // ready and any pre-edge reply disappear after this edge.
            received.push_back(cmd_inst);
            if (command_.completion_delay == 0) {
                registered_reply_ = command_.after_edge;
            } else {
                pending_cycles_ = command_.completion_delay;
            }
            release_reply_ = command_.on_release;
        }
        eval();
        clock = false;
        eval();
    }

private:
    Command command_;
    unsigned ready_at_;
    unsigned pending_cycles_ = 0;
    bool accepted_ = false;
    Reply registered_reply_ = Reply::none;
    Reply release_reply_ = Reply::none;
};

void test_encoding() {
    require(Instruction::set_source_stride(0x12345670) == 0x1234567000000001ULL,
            "source stride encoding");
    require(Instruction::set_destination_stride(0xfffffff0) == 0xfffffff000000002ULL,
            "destination stride encoding");
    require(Instruction::load_weights(0x1000, 128) == 0x0000100000800010ULL,
            "weight address/N encoding");
    require(Instruction::compute(0x2000, 128) == 0x0000200000008020ULL,
            "compute defaults/address/M encoding");
    require(Instruction::compute(0xfffffff0, 255, true, true) == 0xfffffff00300ff20ULL,
            "maximum aligned address and both compute flags");
    require(Instruction::writeback(0x3000) == 0x0000300000000030ULL,
            "writeback encoding");
    require(Instruction::set_source_stride(0) == 0x01 &&
            Instruction::set_destination_stride(0) == 0x02,
            "zero strides must be allowed");
    require(Instruction::writeback(0) == 0x30, "zero address must be allowed");

    for (std::uint16_t count = 1; count <= 255; ++count) {
        require(Instruction::load_weights(0, count) == ((std::uint64_t{count} << 16) | 0x10),
                "N encoding or unused bits incorrect");
        for (unsigned flags = 0; flags < 4; ++flags) {
            require(Instruction::compute(0, count, (flags & 1u) != 0, (flags & 2u) != 0) ==
                        ((std::uint64_t{count} << 8) | (std::uint64_t{flags} << 24) | 0x20),
                    "M/flags encoding or unused bits incorrect");
        }
    }
}

void test_invalid_parameters() {
    for (std::uint32_t offset = 1; offset < 16; ++offset) {
        const auto address = 0x12345000u + offset;
        expect_throw<std::invalid_argument>([&] { Instruction::set_source_stride(address); });
        expect_throw<std::invalid_argument>([&] { Instruction::set_destination_stride(address); });
        expect_throw<std::invalid_argument>([&] { Instruction::load_weights(address, 1); });
        expect_throw<std::invalid_argument>([&] { Instruction::compute(address, 1); });
        expect_throw<std::invalid_argument>([&] { Instruction::writeback(address); });
    }
    const std::uint16_t invalid_dimensions[] = {0, 256, 65535};
    for (auto count : invalid_dimensions) {
        expect_throw<std::invalid_argument>([&] { Instruction::load_weights(0, count); });
        expect_throw<std::invalid_argument>([&] { Instruction::compute(0, count); });
    }
}

void test_backpressure_and_settle() {
    Command command{Instruction::compute(0x2000, 255, true)};
    command.ready_delay = 3;
    command.completion_delay = 2;
    FakeDut dut(command);
    HostDriver driver(dut, [&] { dut.tick(); });
    require(driver.execute(command.instruction, 4, 2) == CommandResult::done,
            "command must succeed on the last acceptance/completion cycles");
    require(dut.cycles == 6 && dut.valid_cycles == 4 && dut.stalled_cycles == 3,
            "backpressure must hold valid and instruction until the pre-edge handshake");
    require(dut.received == std::vector<std::uint64_t>{command.instruction},
            "command must be accepted exactly once");
    require(!dut.cmd_valid, "valid must be low after completion");
}

void test_pre_edge_pulses() {
    for (auto reply : {Reply::done, Reply::error, Reply::both}) {
        Command command{0x123};
        command.before_edge = reply;
        command.after_edge = Reply::none;
        FakeDut dut(command);
        HostDriver driver(dut, [&] { dut.tick(); });
        require(driver.execute(command.instruction, 1, 0) == expected_result(reply),
                "pre-edge reply lost or error priority incorrect");
        require(dut.cycles == 1 && dut.received.size() == 1 && !dut.cmd_valid,
                "pre-edge result must not bypass the actual acceptance edge");
        require(!dut.cmd_done && !dut.cmd_error, "pre-edge pulse must disappear after the edge");
    }
}

void test_post_edge_pulses_and_raw_instruction() {
    for (auto reply : {Reply::done, Reply::error, Reply::both}) {
        Command command{0xffffffffffffffffULL}; // Invalid opcode/address: driver must pass raw bits.
        command.after_edge = reply;
        command.post_reply_requires_valid = true;
        FakeDut dut(command);
        HostDriver driver(dut, [&] { dut.tick(); });
        require(driver.execute(command.instruction, 1, 0) == expected_result(reply),
                "post-edge reply must be captured before lowering valid/eval");
        require(dut.cycles == 1 && dut.received == std::vector<std::uint64_t>{command.instruction},
                "raw instruction must reach the DUT unchanged");
        require(!dut.cmd_valid && !dut.cmd_done && !dut.cmd_error,
                "test pulse must disappear when valid is lowered");
    }
}

void test_error_priority_across_edge() {
    for (auto before : {Reply::done, Reply::error}) {
        Command command{0x10};
        command.before_edge = before;
        command.after_edge = before == Reply::done ? Reply::error : Reply::done;
        FakeDut dut(command);
        HostDriver driver(dut, [&] { dut.tick(); });
        require(driver.execute(command.instruction, 1, 0) == CommandResult::error,
                "error on either side of the acceptance edge must override done");
    }
}

void test_reply_after_valid_release() {
    for (auto reply : {Reply::done, Reply::error}) {
        Command command{0x20};
        command.after_edge = Reply::none;
        command.on_release = reply;
        FakeDut dut(command);
        HostDriver driver(dut, [&] { dut.tick(); });
        require(driver.execute(command.instruction, 1, 0) == expected_result(reply),
                "reply produced by eval after lowering valid must be observed");
    }
}

void test_delayed_pulses() {
    for (auto reply : {Reply::done, Reply::error, Reply::both}) {
        for (unsigned delay = 1; delay <= 4; ++delay) {
            Command command{0x30};
            command.after_edge = reply;
            command.completion_delay = delay;
            FakeDut dut(command);
            HostDriver driver(dut, [&] { dut.tick(); });
            require(driver.execute(command.instruction, 1, delay) == expected_result(reply),
                    "delayed one-cycle reply lost or error priority incorrect");
            require(dut.cycles == delay + 1 && dut.received.size() == 1 && !dut.cmd_valid,
                    "completion must be observed on its first visible cycle");
            dut.tick();
            require(!dut.cmd_done && !dut.cmd_error, "registered reply must last only one cycle");
        }
    }
}

void test_consecutive_execute() {
    for (auto first_reply : {Reply::done, Reply::error}) {
        Command first{0x111};
        first.after_edge = first_reply;
        FakeDut dut(first);
        HostDriver driver(dut, [&] { dut.tick(); });
        require(driver.execute(first.instruction, 1, 0) == expected_result(first_reply),
                "first execute returned the wrong result");

        Command second{0x222};
        second.completion_delay = 3;
        dut.configure(second);
        require(dut.cmd_done || dut.cmd_error, "previous reply must still be visible for this test");
        const auto before = dut.cycles;
        require(driver.execute(second.instruction, 2, 3) == CommandResult::done,
                "previous done/error must not be reused by the next execute");
        require(dut.cycles - before == 5, "expected one drain, one acceptance and three completion cycles");
        require(dut.received == std::vector<std::uint64_t>{first.instruction, second.instruction},
                "consecutive commands must each be accepted exactly once");
        require(!dut.cmd_valid, "valid must be low after consecutive execute");
    }
}

void test_acceptance_timeout() {
    for (bool never_ready : {false, true}) {
        Command command{0x456};
        command.never_ready = never_ready;
        command.ready_delay = 3;
        FakeDut dut(command);
        HostDriver driver(dut, [&] { dut.tick(); });
        const auto message = expect_throw<std::runtime_error>([&] {
            (void)driver.execute(command.instruction, 3, 10);
        });
        require(message.find("acceptance") != std::string::npos,
                "acceptance timeout must identify its stage");
        require(dut.cycles == 3 && dut.received.empty() && !dut.cmd_valid,
                "no-ready timeout must use exactly its budget and lower valid");
        require(dut.cmd_inst == command.instruction, "timeout must not change cmd_inst");
    }
}

void test_completion_timeout() {
    for (unsigned delay : {0u, 4u}) {
        Command command{0x789};
        command.after_edge = delay == 0 ? Reply::none : Reply::done;
        command.completion_delay = delay;
        FakeDut dut(command);
        HostDriver driver(dut, [&] { dut.tick(); });
        const auto message = expect_throw<std::runtime_error>([&] {
            (void)driver.execute(command.instruction, 1, 3);
        });
        require(message.find("completion") != std::string::npos,
                "completion timeout must identify its stage");
        require(dut.cycles == 4 && dut.received.size() == 1 && !dut.cmd_valid,
                "no-done/late-done timeout must use exactly its completion budget");
        require(!dut.status_busy, "this timeout must occur even with busy=0");
    }
}

void test_busy_is_not_completion() {
    Command command{0x20};
    command.completion_delay = 3;
    FakeDut dut(command);
    HostDriver driver(dut, [&] { dut.tick(); });
    require(!dut.status_busy, "fake busy must remain zero");
    require(driver.execute(command.instruction, 1, 3) == CommandResult::done,
            "busy=0 must not replace waiting for done");
    require(dut.cycles == 4, "busy=0 or ready was mistaken for completion");

    command.completion_delay = 0;
    FakeDut busy_dut(command);
    busy_dut.status_busy = true;
    HostDriver busy_driver(busy_dut, [&] { busy_dut.tick(); });
    require(busy_driver.execute(command.instruction, 1, 0) == CommandResult::done,
            "busy=1 must not suppress a real done pulse");
}

void test_zero_budgets() {
    Command command{0x10};
    FakeDut dut(command);
    HostDriver driver(dut, [&] { dut.tick(); });
    const auto message = expect_throw<std::runtime_error>([&] {
        (void)driver.execute(command.instruction, 0, 0);
    });
    require(message.find("acceptance") != std::string::npos,
            "zero acceptance budget must report acceptance timeout");
    require(dut.cycles == 0 && dut.received.empty() && !dut.cmd_valid,
            "zero acceptance budget must not tick");

    command.completion_delay = 1;
    FakeDut delayed_dut(command);
    HostDriver delayed_driver(delayed_dut, [&] { delayed_dut.tick(); });
    const auto completion_message = expect_throw<std::runtime_error>([&] {
        (void)delayed_driver.execute(command.instruction, 1, 0);
    });
    require(completion_message.find("completion") != std::string::npos,
            "zero completion budget must still allow acceptance then report completion timeout");
    require(delayed_dut.cycles == 1 && delayed_dut.received.size() == 1 && !delayed_dut.cmd_valid,
            "zero completion budget must not tick after acceptance");
}

} // namespace

int main() {
    struct Test {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        {"encoding and dimension boundaries", test_encoding},
        {"invalid alignment and dimensions", test_invalid_parameters},
        {"ready backpressure and eval settle", test_backpressure_and_settle},
        {"pre-edge done/error pulses", test_pre_edge_pulses},
        {"post-edge pulses and raw instruction", test_post_edge_pulses_and_raw_instruction},
        {"error priority across acceptance edge", test_error_priority_across_edge},
        {"reply after valid release", test_reply_after_valid_release},
        {"delayed one-cycle done/error", test_delayed_pulses},
        {"consecutive execute", test_consecutive_execute},
        {"acceptance timeout", test_acceptance_timeout},
        {"completion timeout", test_completion_timeout},
        {"busy does not determine completion", test_busy_is_not_completion},
        {"zero cycle budgets", test_zero_budgets},
    };
    unsigned passed = 0;
    unsigned failed = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            ++passed;
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    std::cout << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
