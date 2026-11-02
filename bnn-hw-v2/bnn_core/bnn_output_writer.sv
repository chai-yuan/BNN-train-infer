`default_nettype none

/*
============================================================================
bnn_output_writer (暂存 SRAM 矩阵 -> 2D 步长与自动尾列掩码外部内存写回)
----------------------------------------------------------------------------
控制、维度与寻址：
- start 在 !busy 时拉高 1 拍启动，锁存 cfg_dst_base_addr、cfg_dst_row_stride、cfg_row_num_m (M)
  与 cfg_weight_rows_n (N，即有效输出列数) 的全部工作配置，运行中不再读取这些配置输入。
- 每拍恰好 ODMA_FP32_NUM 个 FP32，要求 DMA_DATA_WIDTH == ODMA_FP32_NUM * 32。
  G = ceil(N / ODMA_FP32_NUM)，每行一笔 G 拍突发，最多一笔写事务在途。
  前 G-1 拍全字节有效；尾拍低 (N - (G-1) * ODMA_FP32_NUM) * 4 字节有效。
- 起始地址与非零行步长按 DMA_DATA_WIDTH / 8 对齐（默认 16 Bytes）。
  第 m 行目标地址 = base + m * effective_stride；stride == 0 时 effective_stride = G * BEAT_BYTES，
  即按整拍占用空间紧密存放，而非按 N * 4 压紧各行。
- SRAM 从地址 0 线性读取至 M * G - 1，不随外部行步长跳转。

简单串行实现，不做乒乓缓存或复杂流水：
- 每次只发一个 SRAM 读请求：T 沿请求，T+OSRAM_READ_LATENCY 沿锁存返回数据；随后保持
  mem_wr_data_valid、data、strb、last 直至握手，再发下一次读请求，valid 不依赖 ready。
- 末拍握手时若 mem_wr_done 同拍到达则立即记住完成，否则等待该行回应；收到回应后才开始
  下一行，所有 M 行完成后 done 脉冲 1 拍。
- 同步 reset 丢弃在途状态，禁用 SRAM 请求、DMA 命令及 data valid，不清除外部存储。
============================================================================
*/
module bnn_output_writer #(
    parameter int unsigned ADDR_WIDTH         = 32,
    parameter int unsigned STRIDE_WIDTH       = 32,
    parameter int unsigned DMA_DATA_WIDTH     = 128,
    parameter int unsigned ODMA_FP32_NUM      = 4,
    parameter int unsigned MAX_ROWS           = 128,
    parameter int unsigned MAX_WEIGHT_GROUPS  = 32,
    parameter int unsigned MAX_OUT_CHANS      = MAX_WEIGHT_GROUPS * ODMA_FP32_NUM,
    parameter int unsigned ROW_COUNT_WIDTH    = $clog2(MAX_ROWS + 1),
    parameter int unsigned CHAN_COUNT_WIDTH   = $clog2(MAX_OUT_CHANS + 1),
    parameter int unsigned OSRAM_DEPTH        = MAX_ROWS * MAX_WEIGHT_GROUPS,
    parameter int unsigned OSRAM_ADDR_WIDTH   = $clog2(OSRAM_DEPTH),
    parameter int unsigned OSRAM_READ_LATENCY = 1,
    parameter int unsigned DMA_LEN_WIDTH      = 16
) (
    input  wire                             clock,
    input  wire                             reset,

    input  wire                             start,
    output wire                             busy,
    output wire                             done,
    input  wire [ADDR_WIDTH-1:0]            cfg_dst_base_addr,
    input  wire [STRIDE_WIDTH-1:0]          cfg_dst_row_stride,
    input  wire [ROW_COUNT_WIDTH-1:0]       cfg_row_num_m,
    input  wire [CHAN_COUNT_WIDTH-1:0]      cfg_weight_rows_n,

    output wire                             osram_ren,
    output wire [OSRAM_ADDR_WIDTH-1:0]      osram_raddr,
    input  wire [(ODMA_FP32_NUM*32)-1:0]    osram_rdata,

    output wire                             mem_wr_cmd_valid,
    input  wire                             mem_wr_cmd_ready,
    output wire [ADDR_WIDTH-1:0]            mem_wr_cmd_addr,
    output wire [DMA_LEN_WIDTH-1:0]         mem_wr_cmd_len,

    output wire                             mem_wr_data_valid,
    input  wire                             mem_wr_data_ready,
    output wire [DMA_DATA_WIDTH-1:0]        mem_wr_data,
    output wire [(DMA_DATA_WIDTH/8)-1:0]    mem_wr_data_strb,
    output wire                             mem_wr_data_last,

    input  wire                             mem_wr_done
);
    localparam int unsigned BEAT_BYTES = DMA_DATA_WIDTH / 8;
    localparam int unsigned DELAY_WIDTH = (OSRAM_READ_LATENCY > 1) ? $clog2(OSRAM_READ_LATENCY) : 1;

    typedef enum logic [2:0] {
        IDLE, COMMAND, READ_REQUEST, READ_RETURN, SEND_DATA, WAIT_RESPONSE, ROW_COMPLETE
    } state_t;
    state_t state;

    logic done_reg;
    logic [ADDR_WIDTH-1:0] row_addr;
    logic [ADDR_WIDTH-1:0] row_stride;
    logic [ROW_COUNT_WIDTH-1:0] rows_left;
    logic [DMA_LEN_WIDTH-1:0] groups;
    logic [DMA_LEN_WIDTH-1:0] beats_left;
    logic [BEAT_BYTES-1:0] tail_strb;
    logic [OSRAM_ADDR_WIDTH-1:0] read_addr;
    logic [DELAY_WIDTH-1:0] read_delay;
    logic [DMA_DATA_WIDTH-1:0] data_buffer;
    logic [BEAT_BYTES-1:0] cfg_tail_strb;

    wire [63:0] cfg_groups = (64'(cfg_weight_rows_n) + 64'(ODMA_FP32_NUM) - 64'd1)
                            / 64'(ODMA_FP32_NUM);
    wire [63:0] cfg_tail_channels = (64'(cfg_weight_rows_n) - 64'd1)
                                   % 64'(ODMA_FP32_NUM) + 64'd1;
    wire last_beat = (beats_left == DMA_LEN_WIDTH'(1));

    always_comb begin
        for (int byte_index = 0; byte_index < BEAT_BYTES; byte_index = byte_index + 1)
            cfg_tail_strb[byte_index] = (64'(byte_index) < cfg_tail_channels * 64'd4);
    end

    assign busy = !reset && (state != IDLE);
    assign done = !reset && done_reg;
    assign osram_ren = !reset && (state == READ_REQUEST);
    assign osram_raddr = read_addr;
    assign mem_wr_cmd_valid = !reset && (state == COMMAND);
    assign mem_wr_cmd_addr = row_addr;
    assign mem_wr_cmd_len = groups;
    assign mem_wr_data_valid = !reset && (state == SEND_DATA);
    assign mem_wr_data = reset ? '0 : data_buffer;
    assign mem_wr_data_strb = reset ? '0 : (last_beat ? tail_strb : {BEAT_BYTES{1'b1}});
    assign mem_wr_data_last = !reset && (state == SEND_DATA) && last_beat;

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            done_reg <= 1'b0;
            row_addr <= '0;
            row_stride <= '0;
            rows_left <= '0;
            groups <= '0;
            beats_left <= '0;
            tail_strb <= '0;
            read_addr <= '0;
            read_delay <= '0;
            data_buffer <= '0;
        end else begin
            done_reg <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        row_addr <= cfg_dst_base_addr;
                        row_stride <= (cfg_dst_row_stride == '0)
                                      ? ADDR_WIDTH'(cfg_groups * 64'(BEAT_BYTES))
                                      : ADDR_WIDTH'(cfg_dst_row_stride);
                        rows_left <= cfg_row_num_m;
                        groups <= DMA_LEN_WIDTH'(cfg_groups);
                        beats_left <= DMA_LEN_WIDTH'(cfg_groups);
                        tail_strb <= cfg_tail_strb;
                        read_addr <= '0;
                        state <= COMMAND;
                    end
                end
                COMMAND: begin
                    if (mem_wr_cmd_ready)
                        state <= READ_REQUEST;
                end
                READ_REQUEST: begin
                    // 与顶层 SRAM pipe 一致：T 请求，T+L 采样（L=1 时为下一沿）。
                    read_delay <= DELAY_WIDTH'(OSRAM_READ_LATENCY - 1);
                    state <= READ_RETURN;
                end
                READ_RETURN: begin
                    if (read_delay != '0) begin
                        read_delay <= read_delay - DELAY_WIDTH'(1);
                    end else begin
                        data_buffer <= osram_rdata;
                        state <= SEND_DATA;
                    end
                end
                SEND_DATA: begin
                    if (mem_wr_data_ready) begin
                        read_addr <= read_addr + OSRAM_ADDR_WIDTH'(1);
                        if (last_beat) begin
                            // 不在下一状态才检查 done，避免丢失与末拍握手同沿的完成脉冲。
                            state <= mem_wr_done ? ROW_COMPLETE : WAIT_RESPONSE;
                        end else begin
                            beats_left <= beats_left - DMA_LEN_WIDTH'(1);
                            state <= READ_REQUEST;
                        end
                    end
                end
                WAIT_RESPONSE: begin
                    if (mem_wr_done)
                        state <= ROW_COMPLETE;
                end
                ROW_COMPLETE: begin
                    if (rows_left == ROW_COUNT_WIDTH'(1)) begin
                        state <= IDLE;
                        done_reg <= 1'b1;
                    end else begin
                        rows_left <= rows_left - ROW_COUNT_WIDTH'(1);
                        row_addr <= row_addr + row_stride;
                        beats_left <= groups;
                        state <= COMMAND;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (ODMA_FP32_NUM > 0 && DMA_DATA_WIDTH == ODMA_FP32_NUM * 32)
            else $fatal(1, "DMA_DATA_WIDTH must equal ODMA_FP32_NUM * 32");
        assert (OSRAM_READ_LATENCY >= 1)
            else $fatal(1, "OSRAM_READ_LATENCY must be >= 1");
        assert (MAX_ROWS > 0 && MAX_WEIGHT_GROUPS > 0 && MAX_OUT_CHANS > 0 && OSRAM_DEPTH > 0)
            else $fatal(1, "Output capacity must be positive");
    end

    always @(posedge clock) begin
        if (!reset && start && !busy) begin
            assert (cfg_row_num_m != '0)
                else $fatal(1, "Output row count must be positive");
            if (64'(MAX_ROWS) < ((64'd1 << ROW_COUNT_WIDTH) - 64'd1)) begin
                assert (64'(cfg_row_num_m) <= 64'(MAX_ROWS))
                    else $fatal(1, "Output row count exceeds MAX_ROWS");
            end
            assert (cfg_weight_rows_n != '0 && 64'(cfg_weight_rows_n) <= 64'(MAX_OUT_CHANS))
                else $fatal(1, "Invalid output channel count");
            assert (cfg_groups <= 64'(MAX_WEIGHT_GROUPS)
                    && cfg_groups == 64'(DMA_LEN_WIDTH'(cfg_groups)))
                else $fatal(1, "Output groups exceed configured capacity or DMA length");
            assert (64'(cfg_row_num_m) * cfg_groups <= 64'(OSRAM_DEPTH)
                    && 64'(cfg_row_num_m) * cfg_groups <= (64'd1 << OSRAM_ADDR_WIDTH))
                else $fatal(1, "Output matrix exceeds SRAM capacity");
            assert (64'(cfg_dst_row_stride) == 64'(ADDR_WIDTH'(cfg_dst_row_stride)))
                else $fatal(1, "Output row stride exceeds ADDR_WIDTH");
        end
    end
`endif
endmodule

`default_nettype wire
