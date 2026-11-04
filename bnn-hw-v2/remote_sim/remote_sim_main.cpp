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

#include "Vbnn_sim_top.h"
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
    std::fprintf(stderr, "usage: %s [--port N]\n", program);
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

    std::uint64_t cycles = 0;
    bool command_error = false;
    while (!context.gotFinish() && cycles < kCycleLimit) {
        if (server.failed())
            break;

        // Do not advance simulation time while the core has no work. This
        // keeps the cycle count independent of network and client latency.
        while (!dut.busy_out && !dut.command_active &&
               !server.has_command() && !server.quit_requested()) {
            if (server.failed())
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (server.failed())
            break;

        dut.finish = server.quit_requested() ? 1 : 0;
        tick(dut);
        ++cycles;

        if (dut.cmd_error_out) {
            server.send_cmd_done(true);
            std::fprintf(stderr, "remote_sim: core reported cmd_error\n");
            command_error = true;
            break;
        }
        if (dut.cmd_done_out)
            server.send_cmd_done(false);
    }

    if (context.gotFinish() && server.quit_requested())
        server.send_quit_ack();
    server.stop();
    g_server = nullptr;
    dut.final();

    if (context.gotFinish()) {
        std::printf("remote_sim: finished after %llu cycles\n",
                    static_cast<unsigned long long>(cycles));
        return 0;
    }
    if (command_error)
        return 1;
    if (server.failed()) {
        std::fprintf(stderr, "remote_sim: connection or protocol failure\n");
        return 1;
    }
    std::fprintf(stderr, "remote_sim: cycle limit reached\n");
    return 1;
}
