// BNN v2 remote co-simulation: Verilator + DPI-C over a local TCP socket.
//
// The upper software stack connects as a TCP client. The simulation sends
// memory read/write requests and command-completion replies; the client sends
// 64-bit commands, read responses and a quit request.
//
// Message framing (little-endian, 16-byte header + payload):
//   magic  u32  = 0x424E4E53 ("BNNS")
//   type   u8
//   flags  u8
//   seq    u16
//   addr   u32
//   length u32   payload bytes following the header
//
// Types:
//   0x01 CMD            client -> sim   payload = 8-byte instruction
//   0x02 READ           sim -> client   addr; always reads 8 bytes, no payload
//   0x03 WRITE          sim -> client   flags = byte strobe, payload = 8 bytes
//   0x04 QUIT           client -> sim
//   0x05 READ_BURST     sim -> client   addr; payload = 4-byte byte count
//   0x81 READ_RSP       client -> sim   seq/addr echo, payload = 8 bytes
//   0x82 CMD_DONE       sim -> client   seq echo, flags bit0 = error
//   0x83 QUIT_ACK       sim -> client
//   0x85 READ_BURST_RSP client -> sim   seq/addr echo, payload = data
//
// Commands are issued one at a time: the client waits for CMD_DONE before
// sending the next CMD. Reads are synchronous; writes are fire-and-forget and
// ordered by TCP.
//
// The DMA read path prefetches a whole burst with READ_BURST at the read
// command handshake, so the per-word READ path normally hits a local cache.
//
// The main loop contains a cycle-probe framework. A probe is sampled once per
// simulated cycle and accumulates active cycles, rising edges, and run lengths.
// Add or remove probes in register_probes(); the loop itself stays untouched.
// --no-probes disables sampling, --probe-interval N prints deltas every N
// cycles, and the final report is printed when the simulation exits.

#include "Vbnn_sim_top.h"
#include "Vbnn_sim_top___024root.h"
#include "verilated.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kMagic = 0x424E4E53u;
constexpr std::uint8_t kTypeCmd = 0x01;
constexpr std::uint8_t kTypeRead = 0x02;
constexpr std::uint8_t kTypeWrite = 0x03;
constexpr std::uint8_t kTypeQuit = 0x04;
constexpr std::uint8_t kTypeReadBurst = 0x05;
constexpr std::uint8_t kTypeReadRsp = 0x81;
constexpr std::uint8_t kTypeCmdDone = 0x82;
constexpr std::uint8_t kTypeQuitAck = 0x83;
constexpr std::uint8_t kTypeReadBurstRsp = 0x85;
constexpr std::size_t kHeaderBytes = 16;
constexpr std::uint32_t kMaxPayload = 1u << 20;
constexpr std::uint64_t kCycleLimit = 1'000'000'000ull;

void put_u16(std::uint8_t* p, std::uint16_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8);
}

void put_u32(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8);
    p[2] = static_cast<std::uint8_t>(value >> 16);
    p[3] = static_cast<std::uint8_t>(value >> 24);
}

std::uint16_t get_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) |
           static_cast<std::uint16_t>(p[1]) << 8;
}

std::uint32_t get_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           static_cast<std::uint32_t>(p[1]) << 8 |
           static_cast<std::uint32_t>(p[2]) << 16 |
           static_cast<std::uint32_t>(p[3]) << 24;
}

bool read_exact(int fd, void* data, std::size_t size) {
    auto* bytes = static_cast<std::uint8_t*>(data);
    std::size_t done = 0;
    while (done < size) {
        const ssize_t n = ::recv(fd, bytes + done, size - done, 0);
        if (n == 0)
            return false;
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

bool send_exact(int fd, const std::uint8_t* data, std::size_t size) {
    std::size_t done = 0;
    while (done < size) {
        const ssize_t n = ::send(fd, data + done, size - done, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

struct Message {
    std::uint8_t type = 0;
    std::uint8_t flags = 0;
    std::uint16_t seq = 0;
    std::uint32_t addr = 0;
    std::vector<std::uint8_t> payload;
};

bool recv_message(int fd, Message& message) {
    std::uint8_t header[kHeaderBytes];
    if (!read_exact(fd, header, sizeof(header)))
        return false;
    if (get_u32(header) != kMagic)
        return false;
    const std::uint32_t length = get_u32(header + 12);
    if (length > kMaxPayload)
        return false;
    message.type = header[4];
    message.flags = header[5];
    message.seq = get_u16(header + 6);
    message.addr = get_u32(header + 8);
    message.payload.resize(length);
    return length == 0 || read_exact(fd, message.payload.data(), length);
}

struct PendingCommand {
    std::uint16_t seq;
    std::uint64_t inst;
};

class RemoteServer {
public:
    explicit RemoteServer(int fd) : fd_(fd) {
        receiver_ = std::thread([this] { receiver_loop(); });
    }

    ~RemoteServer() { stop(); }

    RemoteServer(const RemoteServer&) = delete;
    RemoteServer& operator=(const RemoteServer&) = delete;

    bool has_command() {
        std::lock_guard<std::mutex> lock(command_mutex_);
        return !commands_.empty();
    }

    std::uint64_t pop_command() {
        PendingCommand command{};
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            if (commands_.empty()) {
                protocol_error_ = true;
                return 0;
            }
            command = commands_.front();
            commands_.pop_front();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mutex_);
            inflight_.push_back(command.seq);
        }
        return command.inst;
    }

    std::uint64_t read_memory(std::uint64_t address) {
        if (burst_valid_ && address >= burst_addr_ &&
            address - burst_addr_ <= burst_data_.size() &&
            burst_data_.size() - (address - burst_addr_) >= 8) {
            const std::size_t base =
                static_cast<std::size_t>(address - burst_addr_);
            std::uint64_t value = 0;
            for (int byte = 0; byte < 8; ++byte)
                value |= std::uint64_t{burst_data_[base + byte]} << (8 * byte);
            return value;
        }
        const std::vector<std::uint8_t> data =
            request_read(address, 8, kTypeRead, nullptr);
        std::uint64_t value = 0;
        for (int byte = 0; byte < 8; ++byte)
            value |= std::uint64_t{data[byte]} << (8 * byte);
        return value;
    }

    void read_burst(std::uint64_t address, std::uint64_t bytes) {
        if (bytes == 0 || bytes > kMaxPayload)
            fatal("invalid DMA burst prefetch size");
        std::vector<std::uint8_t> count(4);
        for (int byte = 0; byte < 4; ++byte)
            count[byte] = static_cast<std::uint8_t>(bytes >> (8 * byte));
        burst_data_ = request_read(address, static_cast<std::uint32_t>(bytes),
                                   kTypeReadBurst, &count);
        burst_addr_ = address;
        burst_valid_ = true;
    }

    void write_memory(std::uint64_t address, std::uint64_t data, std::uint8_t strobe) {
        std::vector<std::uint8_t> payload(8);
        for (int byte = 0; byte < 8; ++byte)
            payload[byte] = static_cast<std::uint8_t>(data >> (8 * byte));
        send(kTypeWrite, strobe, 0, static_cast<std::uint32_t>(address), payload);
    }

    void send_cmd_done(bool error) {
        std::uint16_t seq = 0;
        {
            std::lock_guard<std::mutex> lock(inflight_mutex_);
            if (inflight_.empty()) {
                protocol_error_ = true;
                return;
            }
            seq = inflight_.front();
            inflight_.pop_front();
        }
        send(kTypeCmdDone, error ? 1 : 0, seq, 0, {});
    }

    void send_quit_ack() { send(kTypeQuitAck, 0, 0, 0, {}); }

    bool quit_requested() const { return quit_.load(); }
    bool failed() const { return disconnected_.load() || protocol_error_.load(); }

    void stop() {
        if (!running_.exchange(false))
            return;
        ::shutdown(fd_, SHUT_RDWR);
        if (receiver_.joinable())
            receiver_.join();
        ::close(fd_);
        fd_ = -1;
    }

private:
    void fatal(const char* message) {
        std::fprintf(stderr, "remote_sim: %s\n", message);
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::uint8_t> request_read(
        std::uint64_t address, std::uint32_t bytes, std::uint8_t type,
        const std::vector<std::uint8_t>* request_payload) {
        const std::uint16_t seq = ++next_seq_;
        {
            std::lock_guard<std::mutex> lock(read_mutex_);
            read_seq_ = seq;
            read_length_ = bytes;
            read_ready_ = false;
            read_data_.clear();
        }
        send(type, 0, seq, static_cast<std::uint32_t>(address),
             request_payload != nullptr ? *request_payload
                                        : std::vector<std::uint8_t>{});
        std::vector<std::uint8_t> data;
        {
            std::unique_lock<std::mutex> lock(read_mutex_);
            read_cv_.wait(lock, [this] { return read_ready_ || failed(); });
            if (!read_ready_)
                fatal("connection lost while waiting for a read response");
            data = read_data_;
        }
        return data;
    }

    void send(std::uint8_t type, std::uint8_t flags, std::uint16_t seq,
              std::uint32_t address, const std::vector<std::uint8_t>& payload) {
        std::vector<std::uint8_t> packet(kHeaderBytes + payload.size());
        put_u32(packet.data(), kMagic);
        packet[4] = type;
        packet[5] = flags;
        put_u16(packet.data() + 6, seq);
        put_u32(packet.data() + 8, address);
        put_u32(packet.data() + 12, static_cast<std::uint32_t>(payload.size()));
        if (!payload.empty())
            std::memcpy(packet.data() + kHeaderBytes, payload.data(), payload.size());

        std::lock_guard<std::mutex> lock(send_mutex_);
        if (!send_exact(fd_, packet.data(), packet.size()))
            disconnected_ = true;
    }

    void receiver_loop() {
        while (running_.load()) {
            Message message;
            if (!recv_message(fd_, message)) {
                if (running_.load())
                    disconnected_ = true;
                read_cv_.notify_all();
                break;
            }
            switch (message.type) {
            case kTypeCmd: {
                if (message.payload.size() != 8) {
                    protocol_error_ = true;
                    break;
                }
                PendingCommand command{};
                command.seq = message.seq;
                command.inst = 0;
                for (int byte = 0; byte < 8; ++byte)
                    command.inst |= std::uint64_t{message.payload[byte]} << (8 * byte);
                {
                    std::lock_guard<std::mutex> lock(command_mutex_);
                    commands_.push_back(command);
                }
                break;
            }
            case kTypeReadRsp:
            case kTypeReadBurstRsp: {
                std::lock_guard<std::mutex> lock(read_mutex_);
                if (message.seq == read_seq_ &&
                    message.payload.size() == read_length_) {
                    read_data_ = message.payload;
                    read_ready_ = true;
                } else {
                    protocol_error_ = true;
                }
                read_cv_.notify_all();
                break;
            }
            case kTypeQuit:
                quit_ = true;
                read_cv_.notify_all();
                break;
            default:
                protocol_error_ = true;
                break;
            }
            if (protocol_error_.load()) {
                read_cv_.notify_all();
                break;
            }
        }
    }

    int fd_;
    std::atomic<bool> running_{true};
    std::atomic<bool> quit_{false};
    std::atomic<bool> disconnected_{false};
    std::atomic<bool> protocol_error_{false};
    std::uint16_t next_seq_ = 0;
    std::thread receiver_;

    std::mutex send_mutex_;
    std::mutex command_mutex_;
    std::deque<PendingCommand> commands_;
    std::mutex inflight_mutex_;
    std::deque<std::uint16_t> inflight_;

    std::mutex read_mutex_;
    std::condition_variable read_cv_;
    std::uint16_t read_seq_ = 0;
    std::uint32_t read_length_ = 0;
    std::vector<std::uint8_t> read_data_;
    bool read_ready_ = false;

    std::vector<std::uint8_t> burst_data_;
    std::uint64_t burst_addr_ = 0;
    bool burst_valid_ = false;
};

RemoteServer* g_server = nullptr;

void tick(Vbnn_sim_top& dut) {
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
    dut.clock = 0;
    dut.eval();
}

// ---------------------------------------------------------------------------
// Performance probes
// ---------------------------------------------------------------------------
// Each probe is a name, a description and a predicate over the current DUT
// state. The main loop samples every probe once per simulated cycle, after the
// clock edge, and tracks:
//   active     cycles where the predicate was true
//   starts     number of 0 -> 1 transitions
//   avg-run    active / starts (mean run length)
//   longest    longest consecutive active run
//
// ProbeContext exposes the top-level DUT ports and the Verilator root instance,
// so probes can also read internal RTL signals (context.dut.rootp->..., using
// the mangled names listed in build/Vbnn_sim_top___024root.h) when a top-level
// port is not enough. Predicates may capture mutable state when a probe needs
// more than level/edge tracking.
//
// State probes sample an internal FSM state register once per cycle and build a
// per-state cycle histogram. They answer "how is this module spending its
// cycles" faster than adding one level probe per state.
//
// To add a probe, append one entry in register_probes(). To remove a probe,
// delete its entry. Neither change touches the main loop.

struct ProbeContext {
    const Vbnn_sim_top& dut;
    std::uint64_t cycle;
};

struct ProbeStats {
    std::uint64_t active_cycles = 0;
    std::uint64_t activations = 0;
    std::uint64_t longest_run = 0;
    std::uint64_t current_run = 0;
    std::uint64_t reported_active = 0;
    std::uint64_t reported_activations = 0;
    bool active = false;
};

struct Probe {
    const char* name;
    const char* description;
    std::function<bool(const ProbeContext&)> predicate;
    ProbeStats stats;
};

struct StateProbe {
    const char* name;
    std::function<int(const ProbeContext&)> state_of;
    const char* const* state_names;
    int state_count;
    std::vector<std::uint64_t> cycles;
    std::vector<std::uint64_t> reported_cycles;
};

class ProbeSet {
public:
    void add(const char* name, const char* description,
             std::function<bool(const ProbeContext&)> predicate) {
        probes_.push_back(Probe{name, description, std::move(predicate), ProbeStats{}});
    }

    template <std::size_t N>
    void add_state(const char* name, std::function<int(const ProbeContext&)> state_of,
                   const char* const (&state_names)[N]) {
        state_probes_.push_back(StateProbe{name, std::move(state_of), state_names,
                                           static_cast<int>(N),
                                           std::vector<std::uint64_t>(N, 0),
                                           std::vector<std::uint64_t>(N, 0)});
    }

    void sample(const ProbeContext& context) {
        for (Probe& probe : probes_) {
            const bool active = probe.predicate(context);
            ProbeStats& stats = probe.stats;
            if (active) {
                ++stats.active_cycles;
                ++stats.current_run;
                if (!stats.active)
                    ++stats.activations;
                if (stats.current_run > stats.longest_run)
                    stats.longest_run = stats.current_run;
            } else {
                stats.current_run = 0;
            }
            stats.active = active;
        }
        for (StateProbe& probe : state_probes_) {
            const int state = probe.state_of(context);
            if (state < 0)
                continue;
            const int index = state < probe.state_count ? state : probe.state_count - 1;
            ++probe.cycles[index];
        }
    }

    // Periodic delta: level probes in "name=+active/+starts" form, state probes
    // in "name{STATE=+cycles,...}" form.
    void print_delta(std::uint64_t cycles) {
        std::printf("[probe] %llu cycles:", static_cast<unsigned long long>(cycles));
        for (Probe& probe : probes_) {
            ProbeStats& stats = probe.stats;
            const std::uint64_t active = stats.active_cycles - stats.reported_active;
            const std::uint64_t starts =
                stats.activations - stats.reported_activations;
            std::printf(" %s=+%llu/+%llu", probe.name,
                        static_cast<unsigned long long>(active),
                        static_cast<unsigned long long>(starts));
            stats.reported_active = stats.active_cycles;
            stats.reported_activations = stats.activations;
        }
        std::printf("\n");
        if (state_probes_.empty())
            return;
        std::printf("[state] %llu cycles:", static_cast<unsigned long long>(cycles));
        for (StateProbe& probe : state_probes_) {
            std::printf(" %s{", probe.name);
            bool first = true;
            for (int i = 0; i < probe.state_count; ++i) {
                const std::uint64_t delta = probe.cycles[i] - probe.reported_cycles[i];
                probe.reported_cycles[i] = probe.cycles[i];
                if (delta == 0)
                    continue;
                std::printf("%s%s=+%llu", first ? "" : ",", probe.state_names[i],
                            static_cast<unsigned long long>(delta));
                first = false;
            }
            std::printf("}");
        }
        std::printf("\n");
    }

    void print_summary(std::uint64_t cycles) const {
        if (!probes_.empty()) {
            std::printf("remote_sim: performance probes (%llu cycles)\n",
                        static_cast<unsigned long long>(cycles));
            std::printf("  %-18s %12s %9s %9s %10s %10s  %s\n", "probe", "active",
                        "active%", "starts", "avg-run", "longest", "description");
            for (const Probe& probe : probes_) {
                const ProbeStats& stats = probe.stats;
                const double share = cycles == 0
                                         ? 0.0
                                         : 100.0 * static_cast<double>(stats.active_cycles) /
                                               static_cast<double>(cycles);
                const std::uint64_t average =
                    stats.activations == 0 ? 0 : stats.active_cycles / stats.activations;
                std::printf("  %-18s %12llu %8.2f%% %9llu %10llu %10llu  %s\n",
                            probe.name,
                            static_cast<unsigned long long>(stats.active_cycles), share,
                            static_cast<unsigned long long>(stats.activations),
                            static_cast<unsigned long long>(average),
                            static_cast<unsigned long long>(stats.longest_run),
                            probe.description);
            }
        }
        if (!state_probes_.empty()) {
            std::printf("remote_sim: FSM state cycles (%llu cycles)\n",
                        static_cast<unsigned long long>(cycles));
            for (const StateProbe& probe : state_probes_) {
                std::printf("  %-22s", probe.name);
                bool any = false;
                for (int i = 0; i < probe.state_count; ++i) {
                    if (probe.cycles[i] == 0)
                        continue;
                    const double share = cycles == 0
                                             ? 0.0
                                             : 100.0 * static_cast<double>(probe.cycles[i]) /
                                                   static_cast<double>(cycles);
                    std::printf(" %s=%llu(%.1f%%)", probe.state_names[i],
                                static_cast<unsigned long long>(probe.cycles[i]), share);
                    any = true;
                }
                if (!any)
                    std::printf(" (never active)");
                std::printf("\n");
            }
        }
    }

private:
    std::vector<Probe> probes_;
    std::vector<StateProbe> state_probes_;
};

// ---- internal RTL signal accessors ----
// Verilator-flat names from build/Vbnn_sim_top___024root.h. These are
// simulation-only and must be updated if the RTL hierarchy changes.
int controller_state(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__u_top_controller__DOT__state;
}

bool controller_input_int8(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__u_top_controller__DOT__input_int8 != 0;
}

int weight_loader_state(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__u_weight_loader__DOT__state;
}

int iqbuf_state(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__u_input_quant_buffer__DOT__state;
}

int array_state(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__u_core_array__DOT__state;
}

int unit_state(const ProbeContext& context) {
    return context.dut.rootp
        ->bnn_sim_top__DOT__u_core__DOT__u_core_array__DOT__g_unit__BRA__0__KET____DOT__u_unit__DOT__state;
}

int output_accum_state(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__u_output_accum__DOT__state;
}

int output_writer_state(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__u_output_writer__DOT__state;
}

bool core_idma_valid(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__idma_valid != 0;
}

bool core_idma_ready(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__idma_ready != 0;
}

bool core_odma_valid(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__odma_valid != 0;
}

bool core_odma_ready(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__u_core__DOT__odma_ready != 0;
}

bool mem_wr_data_ready(const ProbeContext& context) {
    return context.dut.rootp->bnn_sim_top__DOT__mem_wr_data_ready != 0;
}

// ---- FSM encodings (must match the RTL typedefs) ----
constexpr int kCtrlIdle = 0;
constexpr int kCtrlLoadDrain = 1;
constexpr int kCtrlLoadStart = 2;
constexpr int kCtrlLoadWait = 3;
constexpr int kCtrlLoadConfig = 4;
constexpr int kCtrlComputeStart = 5;
constexpr int kCtrlComputeWait = 6;
constexpr int kCtrlComputeDrain = 7;
constexpr int kCtrlWriteStart = 8;
constexpr int kCtrlWriteWait = 9;

constexpr int kIqCommand = 1;
constexpr int kIqReceive = 2;
constexpr int kIqScan = 3;
constexpr int kIqScale = 4;
constexpr int kIqQuantize = 5;
constexpr int kIqOutputRow = 6;

constexpr int kUnitIdle = 0;
constexpr int kUnitDot = 1;
constexpr int kUnitFpConv = 2;
constexpr int kUnitWeightScale = 3;
constexpr int kUnitRowScale = 4;
constexpr int kUnitOutputData = 5;

constexpr int kWriterCommand = 1;
constexpr int kWriterReadRequest = 2;
constexpr int kWriterReadReturn = 3;
constexpr int kWriterSendData = 4;

constexpr int kWeightLoaderStream = 2;

const char* const kControllerStateNames[] = {
    "IDLE", "LOAD_DRAIN", "LOAD_START", "LOAD_WAIT", "LOAD_CONFIG",
    "COMPUTE_START", "COMPUTE_WAIT", "COMPUTE_DRAIN", "WRITE_START", "WRITE_WAIT",
};
const char* const kWeightLoaderStateNames[] = {
    "IDLE", "COMMAND", "STREAM", "WRITE_PADDING",
};
const char* const kIqbufStateNames[] = {
    "IDLE", "COMMAND", "RECEIVE", "SCAN", "SCALE", "QUANTIZE", "OUTPUT_ROW",
};
const char* const kArrayStateNames[] = {
    "IDLE", "READ_REQUEST", "READ_RETURN", "SUBMIT", "WAIT_OUTPUT",
};
const char* const kUnitStateNames[] = {
    "IDLE", "DOT", "FP_CONV", "WEIGHT_SCALE", "ROW_SCALE", "OUTPUT_DATA",
};
const char* const kOutputAccumStateNames[] = {
    "IDLE", "RECEIVE", "SETTLE", "WRITE_BACK",
};
const char* const kOutputWriterStateNames[] = {
    "IDLE", "COMMAND", "READ_REQUEST", "READ_RETURN", "SEND_DATA",
    "WAIT_RESPONSE", "ROW_COMPLETE",
};

// ---- probe table: add or remove probes here ----
void register_probes(ProbeSet& probes) {
    // Top-level command/DMA activity.
    probes.add("busy", "core or DMA busy",
               [](const ProbeContext& context) { return context.dut.busy_out != 0; });
    probes.add("cmd_active", "command in flight",
               [](const ProbeContext& context) {
                   return context.dut.command_active != 0;
               });
    probes.add("cmd_done", "command completion pulse",
               [](const ProbeContext& context) {
                   return context.dut.cmd_done_out != 0;
               });
    probes.add("cmd_error", "command rejection pulse",
               [](const ProbeContext& context) {
                   return context.dut.cmd_error_out != 0;
               });
    probes.add("idle", "no command and no DMA work",
               [](const ProbeContext& context) {
                   return context.dut.busy_out == 0 && context.dut.command_active == 0;
               });

    // Top-level command phases.
    probes.add("load_phase", "top controller in LOAD phase",
               [](const ProbeContext& context) {
                   const int state = controller_state(context);
                   return state >= kCtrlLoadDrain && state <= kCtrlLoadConfig;
               });
    probes.add("compute_phase", "top controller in COMPUTE phase",
               [](const ProbeContext& context) {
                   const int state = controller_state(context);
                   return state >= kCtrlComputeStart && state <= kCtrlComputeDrain;
               });
    probes.add("write_phase", "top controller in WRITE phase",
               [](const ProbeContext& context) {
                   const int state = controller_state(context);
                   return state >= kCtrlWriteStart && state <= kCtrlWriteWait;
               });
    probes.add("compute_int8", "COMPUTE phase with INT8 input",
               [](const ProbeContext& context) {
                   const int state = controller_state(context);
                   return state >= kCtrlComputeStart && state <= kCtrlComputeDrain &&
                          controller_input_int8(context);
               });
    probes.add("compute_fp32", "COMPUTE phase with FP32 input",
               [](const ProbeContext& context) {
                   const int state = controller_state(context);
                   return state >= kCtrlComputeStart && state <= kCtrlComputeDrain &&
                          !controller_input_int8(context);
               });

    // Compute pipeline: unit work versus row handoff.
    probes.add("unit_dot", "unit 0 dot-product cycles",
               [](const ProbeContext& context) {
                   return unit_state(context) == kUnitDot;
               });
    probes.add("unit_post", "unit 0 FP post-processing cycles",
               [](const ProbeContext& context) {
                   const int state = unit_state(context);
                   return state >= kUnitFpConv && state <= kUnitOutputData;
               });
    probes.add("idma_stall", "input row waiting for array",
               [](const ProbeContext& context) {
                   return core_idma_valid(context) && !core_idma_ready(context);
               });
    probes.add("idma_starve", "array waiting for input row",
               [](const ProbeContext& context) {
                   return !core_idma_valid(context) && core_idma_ready(context);
               });
    probes.add("idma_starve_int8", "array waiting for input row, INT8 compute",
               [](const ProbeContext& context) {
                   return !core_idma_valid(context) && core_idma_ready(context) &&
                          controller_input_int8(context);
               });
    probes.add("idma_starve_fp32", "array waiting for input row, FP32 compute",
               [](const ProbeContext& context) {
                   return !core_idma_valid(context) && core_idma_ready(context) &&
                          !controller_input_int8(context);
               });
    probes.add("iqbuf_prep", "input row receive/scan/quantize",
               [](const ProbeContext& context) {
                   const int state = iqbuf_state(context);
                   return state >= kIqCommand && state <= kIqQuantize;
               });
    probes.add("fp32_scan", "FP32 row max scan",
               [](const ProbeContext& context) {
                   return iqbuf_state(context) == kIqScan;
               });
    probes.add("fp32_quantize", "FP32 row INT8 quantization",
               [](const ProbeContext& context) {
                   return iqbuf_state(context) == kIqQuantize;
               });
    probes.add("odma_stall", "result SRAM backpressure on array output",
               [](const ProbeContext& context) {
                   return core_odma_valid(context) && !core_odma_ready(context);
               });

    // Weight load and writeback.
    probes.add("weight_stream", "weight DMA stream",
               [](const ProbeContext& context) {
                   return weight_loader_state(context) == kWeightLoaderStream;
               });
    probes.add("writer_sram_wait", "writer SRAM request/return cycles",
               [](const ProbeContext& context) {
                   const int state = output_writer_state(context);
                   return state == kWriterReadRequest || state == kWriterReadReturn;
               });
    probes.add("writer_send", "writer DMA data handshake cycles",
               [](const ProbeContext& context) {
                   return output_writer_state(context) == kWriterSendData;
               });
    probes.add("mem_wr_stall", "writer memory backpressure",
               [](const ProbeContext& context) {
                   return output_writer_state(context) == kWriterSendData &&
                          !mem_wr_data_ready(context);
               });

    // FSM histograms.
    probes.add_state("u_top_controller", controller_state, kControllerStateNames);
    probes.add_state("u_weight_loader", weight_loader_state, kWeightLoaderStateNames);
    probes.add_state("u_input_quant_buffer", iqbuf_state, kIqbufStateNames);
    probes.add_state("u_core_array", array_state, kArrayStateNames);
    probes.add_state("u_core_unit[0]", unit_state, kUnitStateNames);
    probes.add_state("u_output_accum", output_accum_state, kOutputAccumStateNames);
    probes.add_state("u_output_writer", output_writer_state, kOutputWriterStateNames);
}

int listen_on_loopback(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::perror("remote_sim: socket");
        return -1;
    }
    int reuse = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::perror("remote_sim: bind");
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 1) < 0) {
        std::perror("remote_sim: listen");
        ::close(fd);
        return -1;
    }
    return fd;
}

std::uint16_t parse_port(const char* text) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value == 0 || value > 65535) {
        std::fprintf(stderr, "remote_sim: invalid port '%s'\n", text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<std::uint16_t>(value);
}

void print_usage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s [--port N] [--no-probes] [--probe-interval N]\n"
                 "  --port N             TCP port to listen on (default 12345)\n"
                 "  --no-probes          disable cycle probes and the final report\n"
                 "  --probe-interval N   print probe deltas every N cycles\n",
                 program);
}

std::uint64_t parse_positive(const char* text, const char* option) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0) {
        std::fprintf(stderr, "remote_sim: %s expects a positive integer, got '%s'\n",
                     option, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<std::uint64_t>(value);
}

} // namespace

extern "C" int cmd_check() {
    return g_server != nullptr && g_server->has_command() ? 1 : 0;
}

extern "C" unsigned long long cmd_get() {
    if (g_server == nullptr)
        return 0;
    return g_server->pop_command();
}

extern "C" unsigned long long data_read(unsigned long long address) {
    if (g_server == nullptr)
        return 0;
    return g_server->read_memory(address);
}

extern "C" void data_read_burst(unsigned long long address,
                                unsigned long long bytes) {
    if (g_server != nullptr)
        g_server->read_burst(address, bytes);
}

extern "C" void data_write(unsigned long long address, unsigned long long data,
                           unsigned char strobe) {
    if (g_server != nullptr)
        g_server->write_memory(address, data, static_cast<std::uint8_t>(strobe));
}

int main(int argc, char** argv) {
    std::uint16_t port = 12345;
    bool probes_enabled = true;
    std::uint64_t probe_interval = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (argument == "--port") {
            if (++i >= argc) {
                std::fprintf(stderr, "remote_sim: --port requires a value\n");
                return 2;
            }
            port = parse_port(argv[i]);
        } else if (argument == "--no-probes") {
            probes_enabled = false;
        } else if (argument == "--probe-interval") {
            if (++i >= argc) {
                std::fprintf(stderr, "remote_sim: --probe-interval requires a value\n");
                return 2;
            }
            probe_interval = parse_positive(argv[i], "--probe-interval");
        } else {
            std::fprintf(stderr, "remote_sim: unknown argument '%s'\n", argument.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    ::signal(SIGPIPE, SIG_IGN);

    const int listen_fd = listen_on_loopback(port);
    if (listen_fd < 0)
        return 1;
    std::printf("remote_sim: listening on 127.0.0.1:%u\n", unsigned{port});
    std::fflush(stdout);

    sockaddr_in peer{};
    socklen_t peer_length = sizeof(peer);
    int fd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&peer), &peer_length);
    ::close(listen_fd);
    if (fd < 0) {
        std::perror("remote_sim: accept");
        return 1;
    }
    int nodelay = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    std::printf("remote_sim: client connected\n");
    std::fflush(stdout);

    RemoteServer server(fd);
    g_server = &server;

    VerilatedContext context;
    Vbnn_sim_top dut(&context);
    dut.clock = 0;
    dut.reset = 1;
    dut.finish = 0;
    for (int i = 0; i < 4; ++i)
        tick(dut);
    dut.reset = 0;
    dut.eval();

    ProbeSet probes;
    if (probes_enabled)
        register_probes(probes);

    std::uint64_t cycles = 0;
    std::uint64_t next_probe_report = probe_interval;
    std::uint64_t host_wait_ns = 0;
    bool command_error = false;
    const auto wall_start = std::chrono::steady_clock::now();
    while (!context.gotFinish() && cycles < kCycleLimit) {
        if (server.failed())
            break;

        // Do not advance simulation time while the core has no work. This
        // keeps the cycle count independent of network and client latency.
        const auto wait_start = std::chrono::steady_clock::now();
        while (!dut.busy_out && !dut.command_active &&
               !server.has_command() && !server.quit_requested()) {
            if (server.failed())
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        host_wait_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_start)
                .count());
        if (server.failed())
            break;

        dut.finish = server.quit_requested() ? 1 : 0;
        tick(dut);
        ++cycles;

        if (probes_enabled) {
            probes.sample(ProbeContext{dut, cycles});
            if (probe_interval != 0 && cycles >= next_probe_report) {
                probes.print_delta(cycles);
                next_probe_report += probe_interval;
            }
        }

        if (dut.cmd_error_out) {
            server.send_cmd_done(true);
            std::fprintf(stderr, "remote_sim: core reported cmd_error\n");
            command_error = true;
            break;
        }
        if (dut.cmd_done_out)
            server.send_cmd_done(false);
    }
    const auto wall_end = std::chrono::steady_clock::now();

    if (context.gotFinish() && server.quit_requested())
        server.send_quit_ack();
    server.stop();
    g_server = nullptr;
    dut.final();

    int status = 1;
    if (context.gotFinish()) {
        std::printf("remote_sim: finished after %llu cycles\n",
                    static_cast<unsigned long long>(cycles));
        status = 0;
    } else if (command_error) {
        status = 1;
    } else if (server.failed()) {
        std::fprintf(stderr, "remote_sim: connection or protocol failure\n");
    } else {
        std::fprintf(stderr, "remote_sim: cycle limit reached\n");
    }

    if (probes_enabled) {
        probes.print_summary(cycles);
        const double wall_seconds =
            std::chrono::duration<double>(wall_end - wall_start).count();
        const double wait_seconds = static_cast<double>(host_wait_ns) * 1e-9;
        std::printf("remote_sim: %.3f s wall, %.0f cycles/s, %.3f s host wait (%.1f%%)\n",
                    wall_seconds,
                    wall_seconds > 0.0 ? static_cast<double>(cycles) / wall_seconds : 0.0,
                    wait_seconds,
                    wall_seconds > 0.0 ? 100.0 * wait_seconds / wall_seconds : 0.0);
    }

    return status;
}
