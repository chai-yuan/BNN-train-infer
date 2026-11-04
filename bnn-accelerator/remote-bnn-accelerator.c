/*
 * BNN accelerator backend over the remote_sim TCP co-simulation.
 *
 * This backend maps the BNN v2 device API onto the RTL's native command set:
 *   - F32 operations use the hardware's per-128-tile dynamic quantization.
 *   - I8 operations use the hardware's INT8 pass-through path.
 * The RTL is the numerical reference; differences from the soft backend are
 * expected (the soft backend remains the host-only reference).
 *
 * Device memory is a flat host allocation. The simulator prefetches each DMA
 * read burst with one READ_BURST request and falls back to 8-byte reads/writes
 * for the remaining traffic; this backend serves both from that allocation.
 * Weights are packed into the RTL's 18-byte records per (out-tile, K-tile).
 *
 * Environment:
 *   BNN_REMOTE_ADDR       host:port of the remote_sim server (127.0.0.1:12345)
 *   BNN_REMOTE_MEM_BYTES  device memory size (default 512 MiB, max 3 GiB)
 */

#define _POSIX_C_SOURCE 200809L

#include "bnn-accelerator.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define REMOTE_MAX_FEATURES   8192u
#define REMOTE_MAX_OUT_CHUNK  128u
#define REMOTE_MAX_ROWS       128u
#define REMOTE_GROUP_SIZE     128u
#define REMOTE_WEIGHT_BYTES   18u
#define REMOTE_BLOCK_BYTES    2304u /* 128 records * 18 bytes */
#define REMOTE_ALIGN          16u
#define REMOTE_ADDR_START     0x1000u
#define REMOTE_DEFAULT_MEM    (512u << 20)
#define REMOTE_MIN_MEM        (8u << 20)
#define REMOTE_MAX_MEM        (3u << 30)

#define REMOTE_MAGIC          0x424E4E53u
#define TYPE_CMD              0x01u
#define TYPE_READ             0x02u
#define TYPE_WRITE            0x03u
#define TYPE_QUIT             0x04u
#define TYPE_READ_BURST       0x05u
#define TYPE_READ_RSP         0x81u
#define TYPE_CMD_DONE         0x82u
#define TYPE_QUIT_ACK         0x83u
#define TYPE_READ_BURST_RSP   0x85u

#define HEADER_BYTES          16u
#define MAX_MESSAGE_PAYLOAD   8u

#define OP_SET_SRC_STRIDE     0x01u
#define OP_SET_DST_STRIDE     0x02u
#define OP_LOAD_WEIGHT        0x10u
#define OP_COMPUTE            0x20u
#define OP_WRITEBACK          0x30u

struct bnn_accel_device {
    int fd;
    unsigned char* memory;
    size_t memory_size;
    size_t bump;
    uint32_t input_scratch;
    size_t input_scratch_bytes;
    uint32_t output_scratch;
    size_t output_scratch_bytes;
    size_t matrix_count;
    size_t buffer_count;
    uint16_t next_seq;
};

struct bnn_accel_matrix {
    BNN_Device owner;
    uint32_t in_features;
    uint32_t out_features;
    uint32_t groups;
    uint32_t k_tiles;
    uint32_t* blocks;
};

struct bnn_accel_buffer {
    BNN_Device owner;
    size_t bytes;
    uint32_t offset;
};

typedef struct {
    uint8_t type;
    uint8_t flags;
    uint16_t seq;
    uint32_t addr;
    uint32_t length;
    uint8_t payload[MAX_MESSAGE_PAYLOAD];
} Message;

static void put_u16(uint8_t* p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t* p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static uint16_t get_u16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t get_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int send_exact(int fd, const uint8_t* data, size_t size) {
    size_t done = 0;
    while (done < size) {
        ssize_t n = send(fd, data + done, size - done, 0);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return 0;
        }
        done += (size_t)n;
    }
    return 1;
}

static int recv_exact(int fd, uint8_t* data, size_t size) {
    size_t done = 0;
    while (done < size) {
        ssize_t n = recv(fd, data + done, size - done, 0);
        if (n == 0)
            return 0;
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return 0;
        }
        done += (size_t)n;
    }
    return 1;
}

static int send_message(int fd, uint8_t type, uint8_t flags, uint16_t seq,
                        uint32_t addr, const uint8_t* payload, uint32_t length) {
    uint8_t header[HEADER_BYTES];
    put_u32(header, REMOTE_MAGIC);
    header[4] = type;
    header[5] = flags;
    put_u16(header + 6, seq);
    put_u32(header + 8, addr);
    put_u32(header + 12, length);
    if (!send_exact(fd, header, sizeof(header)))
        return 0;
    return length == 0 || send_exact(fd, payload, length);
}

static int recv_message(int fd, Message* message) {
    uint8_t header[HEADER_BYTES];
    if (!recv_exact(fd, header, sizeof(header)))
        return 0;
    if (get_u32(header) != REMOTE_MAGIC)
        return 0;
    message->type = header[4];
    message->flags = header[5];
    message->seq = get_u16(header + 6);
    message->addr = get_u32(header + 8);
    message->length = get_u32(header + 12);
    if (message->length > MAX_MESSAGE_PAYLOAD)
        return 0;
    return message->length == 0 ||
           recv_exact(fd, message->payload, message->length);
}

static int serve_memory(BNN_Device device, const Message* message) {
    if (message->type == TYPE_READ) {
        if (message->addr > device->memory_size ||
            device->memory_size - message->addr < 8)
            return 0;
        return send_message(device->fd, TYPE_READ_RSP, 0, message->seq,
                            message->addr, device->memory + message->addr, 8);
    }
    if (message->type == TYPE_WRITE) {
        if (message->length != 8 || message->addr > device->memory_size ||
            device->memory_size - message->addr < 8)
            return 0;
        for (int byte = 0; byte < 8; ++byte) {
            if (message->flags & (1u << byte))
                device->memory[message->addr + (uint32_t)byte] =
                    message->payload[byte];
        }
        return 1;
    }
    if (message->type == TYPE_READ_BURST) {
        uint32_t bytes;
        if (message->length != 4)
            return 0;
        bytes = get_u32(message->payload);
        if (bytes == 0 || message->addr > device->memory_size ||
            device->memory_size - message->addr < bytes)
            return 0;
        return send_message(device->fd, TYPE_READ_BURST_RSP, 0, message->seq,
                            message->addr, device->memory + message->addr,
                            bytes);
    }
    return 0;
}

static BNN_Status pump_completions(BNN_Device device, uint32_t expected) {
    uint32_t completed = 0;
    while (completed < expected) {
        Message message;
        if (!recv_message(device->fd, &message))
            return BNN_EIO;
        if (message.type == TYPE_CMD_DONE) {
            if (message.flags & 1u)
                return BNN_EIO;
            ++completed;
        } else if (!serve_memory(device, &message)) {
            return BNN_EIO;
        }
    }
    return BNN_OK;
}

static int send_instruction(BNN_Device device, uint64_t instruction) {
    uint8_t payload[8];
    for (int byte = 0; byte < 8; ++byte)
        payload[byte] = (uint8_t)(instruction >> (8 * byte));
    return send_message(device->fd, TYPE_CMD, 0, ++device->next_seq, 0,
                        payload, 8);
}

static uint32_t alloc_device(BNN_Device device, size_t bytes) {
    size_t offset = (device->bump + REMOTE_ALIGN - 1u) & ~(size_t)(REMOTE_ALIGN - 1u);
    if (offset > device->memory_size || bytes > device->memory_size - offset)
        return UINT32_MAX;
    device->bump = offset + bytes;
    return (uint32_t)offset;
}

static int rows_fit(size_t capacity, size_t offset, size_t stride,
                    size_t row_bytes, uint32_t rows) {
    size_t available;
    if (rows == 0 || row_bytes == 0 || offset > capacity)
        return 0;
    available = capacity - offset;
    if (row_bytes > available)
        return 0;
    if (stride == 0)
        stride = row_bytes;
    if (stride < row_bytes)
        return 0;
    return (uintmax_t)(rows - 1u) <=
           (uintmax_t)((available - row_bytes) / stride);
}

static int parse_port(const char* text, uint16_t* out_port) {
    char* end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 || value > 65535)
        return 0;
    *out_port = (uint16_t)value;
    return 1;
}

static int connect_server(const char* spec, int* out_fd) {
    char host[128];
    const char* colon = strrchr(spec, ':');
    uint16_t port = 12345;
    struct sockaddr_in address;
    int fd;

    if (colon != NULL) {
        size_t host_len = (size_t)(colon - spec);
        if (host_len == 0 || host_len >= sizeof(host))
            return 0;
        memcpy(host, spec, host_len);
        host[host_len] = '\0';
        if (!parse_port(colon + 1, &port))
            return 0;
    } else {
        if (strlen(spec) >= sizeof(host))
            return 0;
        strcpy(host, spec);
    }

    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1)
        return 0;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return 0;
    {
        struct timeval timeout;
        timeout.tv_sec = 120;
        timeout.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }
    if (connect(fd, (struct sockaddr*)&address, sizeof(address)) != 0) {
        close(fd);
        return 0;
    }
    {
        int nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    }
    *out_fd = fd;
    return 1;
}

static size_t env_memory_size(void) {
    const char* text = getenv("BNN_REMOTE_MEM_BYTES");
    unsigned long long value = 0;
    if (text == NULL || *text == '\0')
        return REMOTE_DEFAULT_MEM;
    char* end = NULL;
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' ||
        value < REMOTE_MIN_MEM || value > REMOTE_MAX_MEM)
        return REMOTE_DEFAULT_MEM;
    return (size_t)value;
}

BNN_Status BNN_DeviceOpen(uint32_t device_index, BNN_Device* out_device) {
    BNN_Device device;
    const char* spec;
    int fd;

    if (out_device == NULL)
        return BNN_EINVAL;
    *out_device = NULL;
    if (device_index != 0)
        return BNN_ENOTSUP;

    signal(SIGPIPE, SIG_IGN);
    spec = getenv("BNN_REMOTE_ADDR");
    if (spec == NULL || *spec == '\0')
        spec = "127.0.0.1:12345";
    if (!connect_server(spec, &fd))
        return BNN_EIO;

    device = (BNN_Device)calloc(1, sizeof(*device));
    if (device == NULL) {
        close(fd);
        return BNN_ENOMEM;
    }
    device->fd = fd;
    device->memory_size = env_memory_size();
    device->memory = (unsigned char*)calloc(1, device->memory_size);
    if (device->memory == NULL) {
        close(fd);
        free(device);
        return BNN_ENOMEM;
    }
    device->bump = REMOTE_ADDR_START;
    device->input_scratch_bytes =
        (size_t)REMOTE_MAX_ROWS * REMOTE_MAX_FEATURES * sizeof(float);
    device->output_scratch_bytes =
        (size_t)REMOTE_MAX_ROWS * (((size_t)REMOTE_MAX_OUT_CHUNK * 4u + 15u) & ~15u);
    device->input_scratch = alloc_device(device, device->input_scratch_bytes);
    device->output_scratch = alloc_device(device, device->output_scratch_bytes);
    if (device->input_scratch == UINT32_MAX ||
        device->output_scratch == UINT32_MAX) {
        close(fd);
        free(device->memory);
        free(device);
        return BNN_ENOMEM;
    }
    *out_device = device;
    return BNN_OK;
}

BNN_Status BNN_DeviceClose(BNN_Device device) {
    BNN_Status status = BNN_OK;
    if (device == NULL)
        return BNN_OK;
    if (device->matrix_count != 0 || device->buffer_count != 0)
        return BNN_EBUSY;
    if (device->fd >= 0) {
        if (!send_message(device->fd, TYPE_QUIT, 0, 0, 0, NULL, 0)) {
            status = BNN_EIO;
        } else {
            for (;;) {
                Message message;
                if (!recv_message(device->fd, &message)) {
                    status = BNN_EIO;
                    break;
                }
                if (message.type == TYPE_QUIT_ACK)
                    break;
                if (!serve_memory(device, &message)) {
                    status = BNN_EIO;
                    break;
                }
            }
        }
        close(device->fd);
    }
    free(device->memory);
    free(device);
    return status;
}

BNN_Status BNN_DeviceGetCaps(BNN_Device device, BNN_Caps* out_caps) {
    if (device == NULL || out_caps == NULL)
        return BNN_EINVAL;
    out_caps->api_version = BNN_API_VERSION;
    out_caps->max_in_features = REMOTE_MAX_FEATURES;
    out_caps->max_out_features = REMOTE_MAX_OUT_CHUNK;
    out_caps->max_rows = REMOTE_MAX_ROWS;
    out_caps->group_size = REMOTE_GROUP_SIZE;
    out_caps->supported_ops = BNN_CAP_GEMV | BNN_CAP_GEMM |
                              BNN_CAP_GEMV_F32 | BNN_CAP_GEMM_F32;
    return BNN_OK;
}

BNN_Status BNN_MatrixCreate(BNN_Device device, const BNN_MatrixDesc* desc,
                            BNN_Matrix* out_matrix) {
    BNN_Matrix matrix;
    size_t row_bytes;
    uint32_t k_tiles;
    if (out_matrix == NULL)
        return BNN_EINVAL;
    *out_matrix = NULL;
    if (device == NULL || desc == NULL)
        return BNN_EINVAL;
    if (desc->in_features == 0 || desc->out_features == 0 ||
        desc->in_features > REMOTE_MAX_FEATURES ||
        desc->out_features > REMOTE_MAX_OUT_CHUNK ||
        desc->group_size != REMOTE_GROUP_SIZE ||
        desc->in_features % REMOTE_GROUP_SIZE != 0 ||
        desc->weight_bits == NULL || desc->scale_fp16 == NULL)
        return BNN_EINVAL;
    row_bytes = desc->in_features / 8u;
    if (desc->weight_bytes != (size_t)desc->out_features * row_bytes ||
        desc->scale_count !=
            (size_t)desc->out_features * (desc->in_features / REMOTE_GROUP_SIZE))
        return BNN_EINVAL;

    k_tiles = desc->in_features / REMOTE_GROUP_SIZE;
    matrix = (BNN_Matrix)calloc(1, sizeof(*matrix));
    if (matrix == NULL)
        return BNN_ENOMEM;
    matrix->blocks = (uint32_t*)malloc((size_t)k_tiles * sizeof(uint32_t));
    if (matrix->blocks == NULL) {
        free(matrix);
        return BNN_ENOMEM;
    }
    matrix->owner = device;
    matrix->in_features = desc->in_features;
    matrix->out_features = desc->out_features;
    matrix->groups = desc->in_features / REMOTE_GROUP_SIZE;
    matrix->k_tiles = k_tiles;

    for (uint32_t tile = 0; tile < k_tiles; ++tile) {
        uint32_t block = alloc_device(device, REMOTE_BLOCK_BYTES);
        if (block == UINT32_MAX) {
            free(matrix->blocks);
            free(matrix);
            return BNN_ENOMEM;
        }
        matrix->blocks[tile] = block;
        memset(device->memory + block, 0, REMOTE_BLOCK_BYTES);
        for (uint32_t channel = 0; channel < desc->out_features; ++channel) {
            uint16_t scale = desc->scale_fp16[
                (size_t)channel * matrix->groups + tile];
            const uint8_t* bits = desc->weight_bits +
                (size_t)channel * row_bytes + (size_t)tile * 16u;
            uint8_t* record = device->memory + block +
                (size_t)channel * REMOTE_WEIGHT_BYTES;
            record[0] = (uint8_t)scale;
            record[1] = (uint8_t)(scale >> 8);
            memcpy(record + 2, bits, 16);
        }
    }
    ++device->matrix_count;
    *out_matrix = matrix;
    return BNN_OK;
}

void BNN_MatrixDestroy(BNN_Matrix matrix) {
    if (matrix == NULL)
        return;
    if (matrix->owner != NULL && matrix->owner->matrix_count != 0)
        --matrix->owner->matrix_count;
    free(matrix->blocks);
    free(matrix);
}

BNN_Status BNN_BufferAlloc(BNN_Device device, size_t bytes, BNN_Buffer* out_buffer) {
    BNN_Buffer buffer;
    uint32_t offset;
    if (out_buffer == NULL)
        return BNN_EINVAL;
    *out_buffer = NULL;
    if (device == NULL || bytes == 0)
        return BNN_EINVAL;
    offset = alloc_device(device, bytes);
    if (offset == UINT32_MAX)
        return BNN_ENOMEM;
    buffer = (BNN_Buffer)calloc(1, sizeof(*buffer));
    if (buffer == NULL)
        return BNN_ENOMEM;
    buffer->owner = device;
    buffer->bytes = bytes;
    buffer->offset = offset;
    ++device->buffer_count;
    *out_buffer = buffer;
    return BNN_OK;
}

void BNN_BufferFree(BNN_Buffer buffer) {
    if (buffer == NULL)
        return;
    if (buffer->owner != NULL && buffer->owner->buffer_count != 0)
        --buffer->owner->buffer_count;
    free(buffer);
}

BNN_Status BNN_BufferWrite(BNN_Buffer buffer, size_t offset, const void* src,
                           size_t bytes) {
    if (buffer == NULL || (src == NULL && bytes != 0) ||
        offset > buffer->bytes || bytes > buffer->bytes - offset)
        return BNN_EINVAL;
    if (bytes != 0)
        memcpy(buffer->owner->memory + buffer->offset + offset, src, bytes);
    return BNN_OK;
}

BNN_Status BNN_BufferRead(BNN_Buffer buffer, size_t offset, void* dst,
                          size_t bytes) {
    if (buffer == NULL || (dst == NULL && bytes != 0) ||
        offset > buffer->bytes || bytes > buffer->bytes - offset)
        return BNN_EINVAL;
    if (bytes != 0)
        memcpy(dst, buffer->owner->memory + buffer->offset + offset, bytes);
    return BNN_OK;
}

BNN_Status BNN_Execute(BNN_Device device, const BNN_Job* job) {
    BNN_Matrix matrix;
    BNN_Buffer input;
    BNN_Buffer output;
    uint32_t rows;
    int is_f32;
    size_t elem_bytes;
    size_t in_row_bytes;
    size_t out_row_bytes;
    size_t in_stride;
    size_t out_stride;
    size_t in_offset;
    size_t out_offset;
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t hw_src_stride;
    uint32_t hw_dst_stride;
    int dst_copy = 0;
    uint32_t expected;
    BNN_Status status;

    if (device == NULL || job == NULL || job->matrix == NULL ||
        job->input.buffer == NULL || job->output.buffer == NULL)
        return BNN_EINVAL;
    matrix = job->matrix;
    input = job->input.buffer;
    output = job->output.buffer;
    if (matrix->owner != device || input->owner != device ||
        output->owner != device)
        return BNN_EINVAL;
    if (input == output)
        return BNN_EINVAL;

    switch (job->operation) {
    case BNN_OP_GEMV:
        if (job->rows != 1 || job->input.type != BNN_TYPE_I8 ||
            job->output.type != BNN_TYPE_F32)
            return BNN_EINVAL;
        is_f32 = 0;
        break;
    case BNN_OP_GEMM:
        if (job->input.type != BNN_TYPE_I8 ||
            job->output.type != BNN_TYPE_F32)
            return BNN_EINVAL;
        is_f32 = 0;
        break;
    case BNN_OP_GEMV_F32:
        if (job->rows != 1 || job->input.type != BNN_TYPE_F32 ||
            job->output.type != BNN_TYPE_F32)
            return BNN_EINVAL;
        is_f32 = 1;
        break;
    case BNN_OP_GEMM_F32:
        if (job->input.type != BNN_TYPE_F32 ||
            job->output.type != BNN_TYPE_F32)
            return BNN_EINVAL;
        is_f32 = 1;
        break;
    default:
        return BNN_ENOTSUP;
    }
    rows = job->rows;
    if (rows == 0 || rows > REMOTE_MAX_ROWS)
        return BNN_EINVAL;

    elem_bytes = is_f32 ? sizeof(float) : 1u;
    in_row_bytes = (size_t)matrix->in_features * elem_bytes;
    out_row_bytes = (size_t)matrix->out_features * sizeof(float);
    in_stride = job->input.row_stride ? job->input.row_stride : in_row_bytes;
    out_stride = job->output.row_stride ? job->output.row_stride : out_row_bytes;
    if (!rows_fit(input->bytes, job->input.offset, job->input.row_stride,
                  in_row_bytes, rows) ||
        !rows_fit(output->bytes, job->output.offset, job->output.row_stride,
                  out_row_bytes, rows))
        return BNN_EINVAL;

    in_offset = (size_t)input->offset + job->input.offset;
    out_offset = (size_t)output->offset + job->output.offset;
    if ((size_t)rows * in_row_bytes > device->input_scratch_bytes ||
        (size_t)rows * (((out_row_bytes + 15u) & ~(size_t)15u)) >
            device->output_scratch_bytes)
        return BNN_ENOMEM;

    if ((in_offset % REMOTE_ALIGN) != 0 ||
        (rows > 1 && (in_stride % REMOTE_ALIGN) != 0)) {
        for (uint32_t row = 0; row < rows; ++row)
            memcpy(device->memory + device->input_scratch +
                       (size_t)row * in_row_bytes,
                   device->memory + in_offset + (size_t)row * in_stride,
                   in_row_bytes);
        src_addr = device->input_scratch;
        in_stride = in_row_bytes;
    } else {
        src_addr = (uint32_t)in_offset;
    }
    hw_src_stride = rows > 1 ? (uint32_t)in_stride : 0;

    if (rows == 1 && (out_offset % REMOTE_ALIGN) == 0) {
        dst_addr = (uint32_t)out_offset;
        hw_dst_stride = 0;
    } else {
        dst_copy = 1;
        dst_addr = device->output_scratch;
        hw_dst_stride = (uint32_t)((out_row_bytes + 15u) & ~15u);
    }

    expected = 2u + 2u * matrix->k_tiles + 1u;
    if (!send_instruction(device,
                          ((uint64_t)hw_src_stride << 32) | OP_SET_SRC_STRIDE) ||
        !send_instruction(device,
                          ((uint64_t)hw_dst_stride << 32) | OP_SET_DST_STRIDE))
        return BNN_EIO;

    for (uint32_t tile = 0; tile < matrix->k_tiles; ++tile) {
        uint64_t load = ((uint64_t)matrix->blocks[tile] << 32) |
                        ((uint64_t)matrix->out_features << 16) |
                        OP_LOAD_WEIGHT;
        uint64_t compute;
        uint32_t compute_addr = src_addr +
            (uint32_t)((size_t)tile * REMOTE_GROUP_SIZE * elem_bytes);
        if (!send_instruction(device, load))
            return BNN_EIO;
        compute = ((uint64_t)compute_addr << 32) |
                  (tile != 0 ? (1ull << 24) : 0ull) |
                  (is_f32 ? 0ull : (1ull << 25)) |
                  ((uint64_t)rows << 8) | OP_COMPUTE;
        if (!send_instruction(device, compute))
            return BNN_EIO;
    }
    if (!send_instruction(device, ((uint64_t)dst_addr << 32) | OP_WRITEBACK))
        return BNN_EIO;

    status = pump_completions(device, expected);
    if (status != BNN_OK)
        return status;

    if (dst_copy) {
        for (uint32_t row = 0; row < rows; ++row)
            memcpy(device->memory + out_offset + (size_t)row * out_stride,
                   device->memory + dst_addr +
                       (size_t)row * hw_dst_stride,
                   out_row_bytes);
    }
    return BNN_OK;
}

const char* BNN_StatusString(BNN_Status status) {
    switch (status) {
    case BNN_OK:
        return "success";
    case BNN_EINVAL:
        return "invalid argument";
    case BNN_ENOMEM:
        return "out of memory";
    case BNN_ENOTSUP:
        return "unsupported operation or configuration";
    case BNN_ERANGE:
        return "buffer or input range error";
    case BNN_EBUSY:
        return "device still owns resources";
    case BNN_EIO:
        return "device I/O error";
    case BNN_ETIMEOUT:
        return "device timeout";
    default:
        return "unknown status";
    }
}
