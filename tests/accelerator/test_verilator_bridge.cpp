/* Test only the public C bridge: no Verilator/generated-model headers. */
#include "verilator-bnn-bridge.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); \
} } while (0)

static void status_check(BNN_Status got, BNN_Status want, const char *expr, int line) {
    if (got != want) {
        std::fprintf(stderr, "%s:%d: %s returned %d, expected %d\n",
                     __FILE__, line, expr, static_cast<int>(got), static_cast<int>(want));
        std::exit(1);
    }
}
#define STATUS(expr, want) status_check((expr), (want), #expr, __LINE__)
#define OK(expr) STATUS((expr), BNN_OK)

namespace {
constexpr uint32_t K = 128, N = 2, G = 128;
using Input = std::array<int8_t, K>;
using Weights = std::array<uint8_t, N * K / 8>;
using Output = std::array<int32_t, N>;

Output reference(const Input &input, const Weights &weights) {
    Output result{};
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t k = 0; k < K; ++k) {
            int32_t sign = (weights[n * K / 8 + k / 8] & (1u << (k % 8))) ? 1 : -1;
            result[n] += static_cast<int32_t>(input[k]) * sign;
        }
    }
    return result;
}

void run(BNN_RtlSim *sim, const Input &input, const Weights &weights, Output &output) {
    const Input saved_input = input;
    const Weights saved_weights = weights;
    const Output expected = reference(input, weights);
    /* Canaries catch a bridge writing beyond [N][K/G]. */
    std::array<int32_t, N + 2> guard;
    guard.fill(INT32_C(0x13579bdf));
    OK(bnn_rtl_dot_groups(sim, input.data(), weights.data(), K, N, G, guard.data() + 1, 0));
    CHECK(guard.front() == INT32_C(0x13579bdf) && guard.back() == INT32_C(0x13579bdf));
    std::memcpy(output.data(), guard.data() + 1, sizeof output);
    if (output != expected) {
        std::fprintf(stderr, "dot: got [%d, %d], expected [%d, %d]\n",
                     output[0], output[1], expected[0], expected[1]);
        std::exit(1);
    }
    CHECK(input == saved_input && weights == saved_weights);
}
} // namespace

int main() {
    BNN_RtlSim *first = nullptr, *second = nullptr;
    Input a{}, b{};
    Weights wa{}, wb{};
    Output oa{}, ob{};
    for (uint32_t k = 0; k < K; ++k) {
        a[k] = static_cast<int8_t>(static_cast<int>((k * 37u + k / 3u) % 255u) - 127);
        b[k] = static_cast<int8_t>(static_cast<int>((k * 71u + 19u) % 256u) - 128);
    }
    for (size_t i = 0; i < wa.size(); ++i) {
        wa[i] = static_cast<uint8_t>(i * 53u + 0xa5u);
        wb[i] = static_cast<uint8_t>(i * 29u + 0x3cu);
    }
    STATUS(bnn_rtl_create(nullptr), BNN_EINVAL);
    bnn_rtl_destroy(nullptr);
    OK(bnn_rtl_create(&first)); CHECK(first != nullptr);
    OK(bnn_rtl_create(&second)); CHECK(second != nullptr && second != first);

    /* One command, one input, four pipeline/drain cycles: no extra completion cycle. */
    int32_t single_expected = 0;
    for (uint32_t k = 0; k < 32; ++k) {
        const int32_t sign = (wa[k / 8] & (1u << (k % 8))) ? 1 : -1;
        single_expected += static_cast<int32_t>(a[k]) * sign;
    }
    std::array<int32_t, 3> single;
    single.fill(INT32_C(0x13579bdf));
    STATUS(bnn_rtl_dot_groups(first, a.data(), wa.data(), 32, 1, 32,
                              single.data() + 1, 5), BNN_ETIMEOUT);
    CHECK(single[1] == INT32_C(0x13579bdf));
    for (unsigned i = 0; i < 3; ++i) {
        OK(bnn_rtl_dot_groups(first, a.data(), wa.data(), 32, 1, 32,
                              single.data() + 1, 6));
        CHECK(single[0] == INT32_C(0x13579bdf) && single[2] == INT32_C(0x13579bdf));
        CHECK(single[1] == single_expected);
    }

    /* A one-cycle job must time out; the same instance then runs normally. */
    STATUS(bnn_rtl_dot_groups(first, a.data(), wa.data(), K, N, G, oa.data(), 1), BNN_ETIMEOUT);
    run(first, a, wa, oa);
    run(second, b, wb, ob);

#define BAD_CONFIG(ki, ni, gi) do { \
    STATUS(bnn_rtl_dot_groups(first, a.data(), wa.data(), (ki), (ni), (gi), oa.data(), 0), BNN_EINVAL); \
    run(first, a, wa, oa); run(second, b, wb, ob); \
} while (0)
    BAD_CONFIG(0, N, G);
    BAD_CONFIG(K, 0, G);
    BAD_CONFIG(K, N, 0);
    BAD_CONFIG(129, N, G);
    BAD_CONFIG(K, N, 33);
    BAD_CONFIG(K, N, 96);
    BAD_CONFIG(K, 65536, G);
#undef BAD_CONFIG
    STATUS(bnn_rtl_dot_groups(nullptr, a.data(), wa.data(), K, N, G, oa.data(), 0), BNN_EINVAL);
    STATUS(bnn_rtl_dot_groups(first, nullptr, wa.data(), K, N, G, oa.data(), 0), BNN_EINVAL);
    STATUS(bnn_rtl_dot_groups(first, a.data(), nullptr, K, N, G, oa.data(), 0), BNN_EINVAL);
    STATUS(bnn_rtl_dot_groups(first, a.data(), wa.data(), K, N, G, nullptr, 0), BNN_EINVAL);
    run(first, a, wa, oa);

    /* Bridge supports all int8 values, independently of the public API restriction. */
    a.fill(INT8_MIN);
    wa.fill(0);
    for (size_t i = 0; i < K / 8; ++i) wa[i] = 0xff;
    run(first, a, wa, oa);
    CHECK(oa[0] == -16384 && oa[1] == 16384);
    a.fill(0); a[0] = INT8_MIN; a[127] = 127;
    run(first, a, wa, oa);
    CHECK(oa[0] == -1 && oa[1] == 1);

    /* Interleaving/resetting/destroying a model must not change another model. */
    for (unsigned i = 0; i < 3; ++i) {
        run(second, b, wb, ob);
        run(first, a, wa, oa);
    }
    STATUS(bnn_rtl_dot_groups(first, a.data(), wa.data(), K, N, G, oa.data(), 1), BNN_ETIMEOUT);
    run(second, b, wb, ob);
    run(first, a, wa, oa);
    bnn_rtl_destroy(first);
    run(second, b, wb, ob);
    OK(bnn_rtl_create(&first)); CHECK(first != nullptr && first != second);
    run(first, a, wa, oa);
    bnn_rtl_destroy(second);
    run(first, a, wa, oa);
    bnn_rtl_destroy(first);
    std::puts("Verilator public bridge: dot oracle, timeout recovery, int8 -128, validation and model isolation OK");
    return 0;
}
