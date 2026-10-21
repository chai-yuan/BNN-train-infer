/* Runtime interoperation regression: public bridge plus Verilator runtime only. */
#include "verilator-bnn-bridge.h"
#include <verilated.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); \
} } while (0)

namespace {
constexpr uint32_t K = 128, N = 2, G = 128;
constexpr uint64_t HOST_TIME = 123, EXTERNAL_TIME = 456;
using Input = std::array<int8_t, K>;
using Weights = std::array<uint8_t, N * K / 8>;
using Output = std::array<int32_t, N>;

void check_context(VerilatedContext *saved, uint64_t saved_time,
                   const VerilatedContext &host, const VerilatedContext &external) {
    CHECK(Verilated::threadContextp() == saved);
    CHECK(saved->time() == saved_time);
    CHECK(Verilated::time() == saved_time);
    CHECK(host.time() == HOST_TIME);
    CHECK(external.time() == EXTERNAL_TIME);
}

BNN_RtlSim *create(const VerilatedContext &host, const VerilatedContext &external) {
    VerilatedContext *saved = Verilated::threadContextp();
    const uint64_t saved_time = saved->time();
    BNN_RtlSim *sim = nullptr;
    const BNN_Status status = bnn_rtl_create(&sim);
    check_context(saved, saved_time, host, external);
    CHECK(status == BNN_OK && sim != nullptr);
    return sim;
}

void execute(BNN_RtlSim *sim, const Input &input, const Weights &weights,
             uint64_t max_cycles, BNN_Status expected_status,
             const VerilatedContext &host, const VerilatedContext &external) {
    VerilatedContext *saved = Verilated::threadContextp();
    const uint64_t saved_time = saved->time();
    Output output{};
    const BNN_Status status = bnn_rtl_dot_groups(
        sim, input.data(), weights.data(), K, N, G, output.data(), max_cycles);
    /* Check TLS immediately, including the timeout/reset path. */
    check_context(saved, saved_time, host, external);
    CHECK(status == expected_status);
    if (status == BNN_OK) {
        Output reference{};
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t k = 0; k < K; ++k) {
                const int32_t sign = (weights[n * K / 8 + k / 8] & (1u << (k % 8))) ? 1 : -1;
                reference[n] += static_cast<int32_t>(input[k]) * sign;
            }
        }
        CHECK(output == reference);
    }
}

void destroy(BNN_RtlSim *sim, const VerilatedContext &host, const VerilatedContext &external) {
    VerilatedContext *saved = Verilated::threadContextp();
    const uint64_t saved_time = saved->time();
    bnn_rtl_destroy(sim);
    check_context(saved, saved_time, host, external);
}
} // namespace

int main() {
    VerilatedContext *original = Verilated::threadContextp();
    const uint64_t original_time = original->time();
    {
        /* Both constructors change TLS. Explicitly install the caller's host context. */
        VerilatedContext externalContext;
        VerilatedContext hostContext;
        hostContext.time(HOST_TIME);
        externalContext.time(EXTERNAL_TIME);
        Verilated::threadContextp(&hostContext);
        VerilatedContext *host_tls = Verilated::threadContextp();
        CHECK(host_tls == &hostContext);
        check_context(host_tls, HOST_TIME, hostContext, externalContext);

        Input input{};
        Weights weights{};
        for (uint32_t k = 0; k < K; ++k)
            input[k] = static_cast<int8_t>(static_cast<int>((k * 37u + 11u) % 256u) - 128);
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = static_cast<uint8_t>(i * 53u + 0xa5u);

        BNN_RtlSim *simA = create(hostContext, externalContext);
        BNN_RtlSim *simB = create(hostContext, externalContext);
        CHECK(simA != simB);
        execute(simA, input, weights, 0, BNN_OK, hostContext, externalContext);
        execute(simB, input, weights, 0, BNN_OK, hostContext, externalContext);
        execute(simA, input, weights, 1, BNN_ETIMEOUT, hostContext, externalContext);
        execute(simA, input, weights, 0, BNN_OK, hostContext, externalContext);
        execute(simB, input, weights, 0, BNN_OK, hostContext, externalContext);

        /* Restore the actual caller, not a cached host context or another sim's context. */
        Verilated::threadContextp(&externalContext);
        execute(simA, input, weights, 0, BNN_OK, hostContext, externalContext);
        execute(simA, input, weights, 1, BNN_ETIMEOUT, hostContext, externalContext);
        execute(simA, input, weights, 0, BNN_OK, hostContext, externalContext);
        destroy(simA, hostContext, externalContext);
        simA = nullptr;
        execute(simB, input, weights, 0, BNN_OK, hostContext, externalContext);
        /* Creation under a second caller context must preserve that context too. */
        simA = create(hostContext, externalContext);
        execute(simA, input, weights, 0, BNN_OK, hostContext, externalContext);

        Verilated::threadContextp(host_tls);
        execute(simA, input, weights, 0, BNN_OK, hostContext, externalContext);
        destroy(simB, hostContext, externalContext);
        execute(simA, input, weights, 0, BNN_OK, hostContext, externalContext);
        destroy(simA, hostContext, externalContext);
        destroy(nullptr, hostContext, externalContext);
        check_context(host_tls, HOST_TIME, hostContext, externalContext);
        /* Do not leave TLS referring to a stack context after its lifetime ends. */
        Verilated::threadContextp(original);
    }
    CHECK(Verilated::threadContextp() == original);
    CHECK(original->time() == original_time);
    std::puts("Verilator context: create/run/timeout/destroy preserve caller TLS and host time; external context OK");
    return 0;
}
