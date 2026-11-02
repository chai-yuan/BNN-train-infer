`default_nettype none

/*
============================================================================
bnn_weight_loader (外部内存紧密排列权重 -> Weight SRAM)
----------------------------------------------------------------------------
控制接口（来自 bnn_top_controller）：
- start 在 !busy 时拉高 1 拍启动，当拍锁存 cfg_src_base_addr 与 cfg_weight_rows_n (N)。
- N 为本次分块的有效权重矩阵行数（即有效输出通道数，范围 1 .. MAX_OUT_CHANS）。

外部内存布局与时序：
- WEIGHT_WIDTH 与 DMA_DATA_WIDTH 均须为正的整字节宽度，分别记为 WEIGHT_BYTES、BEAT_BYTES。
  N 条权重按小端字节序紧密排列，不在记录之间插入填充；默认每条 18 Bytes，低 2 Bytes 为
  scale_fp16，随后 16 Bytes 为 weight_bits，对应 wsram_wdata = {weight_bits, scale_fp16}。
- 起始地址按 BEAT_BYTES 对齐；连续分块由软件将各块占用空间补齐至整拍边界。
  仅发一笔 DMA 读，长度 ceil(N * WEIGHT_BYTES / BEAT_BYTES)，完整握手接收最后一拍，忽略
  其中超出 N 条记录的高位字节。mem_rd_data_last 仅作协议断言，不用它提前终止事务。
- 简单单拍缓存，不做乒乓或流水：接收一拍后逐周期取一字节，记录可跨任意拍边界；每拼好一条
  暂停拆字节，以一个 wsram_wen 周期将完整权重写入 WEIGHT_BASE_ADDR 起的连续地址。
- 有效记录之后补全 0 至 ODMA_FP32_NUM 的整数倍；最后一次 SRAM 写入沿之后才报告 done。
  同步 reset 丢弃内部在途状态并禁用命令、ready 与写使能，不清除外部内存 / SRAM。
============================================================================
*/
module bnn_weight_loader #(
    parameter int unsigned ADDR_WIDTH        = 32,
    parameter int unsigned DMA_DATA_WIDTH    = 128,
    parameter int unsigned WEIGHT_WIDTH      = 144,
    parameter int unsigned ODMA_FP32_NUM     = 4,
    parameter int unsigned SRAM_ADDR_WIDTH   = 7,
    parameter logic [SRAM_ADDR_WIDTH-1:0] WEIGHT_BASE_ADDR = '0,
    parameter int unsigned DMA_LEN_WIDTH     = 16,
    parameter int unsigned MAX_WEIGHT_GROUPS = (1 << SRAM_ADDR_WIDTH) / ODMA_FP32_NUM,
    parameter int unsigned MAX_OUT_CHANS     = MAX_WEIGHT_GROUPS * ODMA_FP32_NUM,
    parameter int unsigned CHAN_COUNT_WIDTH  = $clog2(MAX_OUT_CHANS + 1)
) (
    input  wire                             clock,
    input  wire                             reset,

    input  wire                             start,
    output wire                             busy,
    output wire                             done,
    input  wire [ADDR_WIDTH-1:0]            cfg_src_base_addr,
    input  wire [CHAN_COUNT_WIDTH-1:0]       cfg_weight_rows_n,

    output wire                             mem_rd_cmd_valid,
    input  wire                             mem_rd_cmd_ready,
    output wire [ADDR_WIDTH-1:0]            mem_rd_cmd_addr,
    output wire [DMA_LEN_WIDTH-1:0]         mem_rd_cmd_len,

    input  wire                             mem_rd_data_valid,
    output wire                             mem_rd_data_ready,
    input  wire [DMA_DATA_WIDTH-1:0]        mem_rd_data,
    input  wire                             mem_rd_data_last,

    output wire                             wsram_wen,
    output wire [SRAM_ADDR_WIDTH-1:0]       wsram_waddr,
    output wire [WEIGHT_WIDTH-1:0]          wsram_wdata
);
    localparam int unsigned WEIGHT_BYTES = WEIGHT_WIDTH / 8;
    localparam int unsigned BEAT_BYTES = DMA_DATA_WIDTH / 8;
    localparam int unsigned BYTE_COUNT_WIDTH = (BEAT_BYTES > 1) ? $clog2(BEAT_BYTES + 1) : 1;
    localparam int unsigned WEIGHT_BYTE_WIDTH = (WEIGHT_BYTES > 1) ? $clog2(WEIGHT_BYTES) : 1;
    localparam int unsigned PAD_COUNT_WIDTH = (ODMA_FP32_NUM > 1) ? $clog2(ODMA_FP32_NUM) : 1;

    typedef enum logic [2:0] {
        IDLE, COMMAND, RECEIVE, UNPACK, WRITE_WEIGHT, WRITE_PADDING
    } state_t;
    state_t state;

    logic done_reg;
    logic [ADDR_WIDTH-1:0] source_addr;
    logic [DMA_LEN_WIDTH-1:0] burst_len;
    logic [DMA_LEN_WIDTH-1:0] beats_left;
    logic [CHAN_COUNT_WIDTH-1:0] rows_left;
    logic [PAD_COUNT_WIDTH-1:0] padding_left;
    logic [DMA_DATA_WIDTH-1:0] beat_buffer;
    logic [BYTE_COUNT_WIDTH-1:0] bytes_left;
    logic [WEIGHT_BYTE_WIDTH-1:0] weight_byte;
    logic [WEIGHT_WIDTH-1:0] weight_buffer;
    logic [SRAM_ADDR_WIDTH-1:0] write_addr;

    wire [63:0] cfg_beats = (64'(cfg_weight_rows_n) * 64'(WEIGHT_BYTES)
                           + 64'(BEAT_BYTES) - 64'd1) / 64'(BEAT_BYTES);
    wire [63:0] cfg_padding = (64'(ODMA_FP32_NUM)
                             - 64'(cfg_weight_rows_n) % 64'(ODMA_FP32_NUM))
                             % 64'(ODMA_FP32_NUM);

    assign busy = !reset && (state != IDLE);
    assign done = !reset && done_reg;
    assign mem_rd_cmd_valid = !reset && (state == COMMAND);
    assign mem_rd_cmd_addr = source_addr;
    assign mem_rd_cmd_len = burst_len;
    assign mem_rd_data_ready = !reset && (state == RECEIVE);
    assign wsram_wen = !reset && ((state == WRITE_WEIGHT) || (state == WRITE_PADDING));
    assign wsram_waddr = write_addr;
    assign wsram_wdata = (!reset && (state == WRITE_WEIGHT)) ? weight_buffer : '0;

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            done_reg <= 1'b0;
            source_addr <= '0;
            burst_len <= '0;
            beats_left <= '0;
            rows_left <= '0;
            padding_left <= '0;
            beat_buffer <= '0;
            bytes_left <= '0;
            weight_byte <= '0;
            weight_buffer <= '0;
            write_addr <= WEIGHT_BASE_ADDR;
        end else begin
            done_reg <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        source_addr <= cfg_src_base_addr;
                        burst_len <= DMA_LEN_WIDTH'(cfg_beats);
                        beats_left <= DMA_LEN_WIDTH'(cfg_beats);
                        rows_left <= cfg_weight_rows_n;
                        padding_left <= PAD_COUNT_WIDTH'(cfg_padding);
                        weight_byte <= '0;
                        weight_buffer <= '0;
                        write_addr <= WEIGHT_BASE_ADDR;
                        state <= COMMAND;
                    end
                end
                COMMAND: begin
                    if (mem_rd_cmd_ready)
                        state <= RECEIVE;
                end
                RECEIVE: begin
                    if (mem_rd_data_valid) begin
                        beat_buffer <= mem_rd_data;
                        bytes_left <= BYTE_COUNT_WIDTH'(BEAT_BYTES);
                        beats_left <= beats_left - DMA_LEN_WIDTH'(1);
                        state <= UNPACK;
                    end
                end
                UNPACK: begin
                    weight_buffer[8 * int'(weight_byte) +: 8] <= beat_buffer[7:0];
                    beat_buffer <= beat_buffer >> 8;
                    bytes_left <= bytes_left - BYTE_COUNT_WIDTH'(1);
                    if (weight_byte == WEIGHT_BYTE_WIDTH'(WEIGHT_BYTES - 1)) begin
                        weight_byte <= '0;
                        state <= WRITE_WEIGHT;
                    end else begin
                        weight_byte <= weight_byte + WEIGHT_BYTE_WIDTH'(1);
                        if (bytes_left == BYTE_COUNT_WIDTH'(1))
                            state <= RECEIVE;
                    end
                end
                WRITE_WEIGHT: begin
                    // 本沿 SRAM 真正写入；末记录所在拍已完整接收，其余高字节直接丢弃。
                    write_addr <= write_addr + SRAM_ADDR_WIDTH'(1);
                    rows_left <= rows_left - CHAN_COUNT_WIDTH'(1);
                    if (rows_left == CHAN_COUNT_WIDTH'(1)) begin
                        if (padding_left != '0) begin
                            state <= WRITE_PADDING;
                        end else begin
                            state <= IDLE;
                            done_reg <= 1'b1;
                        end
                    end else if (bytes_left != '0) begin
                        state <= UNPACK;
                    end else begin
                        state <= RECEIVE;
                    end
                end
                WRITE_PADDING: begin
                    write_addr <= write_addr + SRAM_ADDR_WIDTH'(1);
                    padding_left <= padding_left - PAD_COUNT_WIDTH'(1);
                    if (padding_left == PAD_COUNT_WIDTH'(1)) begin
                        state <= IDLE;
                        done_reg <= 1'b1;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (WEIGHT_WIDTH > 0 && WEIGHT_WIDTH % 8 == 0)
            else $fatal(1, "WEIGHT_WIDTH must be a positive whole number of bytes");
        assert (DMA_DATA_WIDTH > 0 && DMA_DATA_WIDTH % 8 == 0)
            else $fatal(1, "DMA_DATA_WIDTH must be a positive whole number of bytes");
        assert (ODMA_FP32_NUM > 0 && MAX_WEIGHT_GROUPS > 0 && MAX_OUT_CHANS > 0)
            else $fatal(1, "Weight capacity and channel count must be positive");
    end

    always @(posedge clock) begin
        if (!reset) begin
            if (start && !busy) begin
                assert (cfg_weight_rows_n != '0 && 64'(cfg_weight_rows_n) <= 64'(MAX_OUT_CHANS))
                    else $fatal(1, "Invalid weight row count");
                assert (64'(cfg_weight_rows_n) + cfg_padding
                        <= 64'(MAX_WEIGHT_GROUPS) * 64'(ODMA_FP32_NUM))
                    else $fatal(1, "Padded weights exceed configured groups");
                assert (64'(WEIGHT_BASE_ADDR) + 64'(cfg_weight_rows_n) + cfg_padding
                        <= (64'd1 << SRAM_ADDR_WIDTH))
                    else $fatal(1, "Padded weights exceed SRAM address space");
                assert (cfg_beats != 0 && cfg_beats == 64'(DMA_LEN_WIDTH'(cfg_beats)))
                    else $fatal(1, "Weight burst length does not fit DMA_LEN_WIDTH");
            end
            if (mem_rd_data_valid && mem_rd_data_ready) begin
                assert (mem_rd_data_last == (beats_left == DMA_LEN_WIDTH'(1)))
                    else $fatal(1, "Weight DMA last does not match commanded burst length");
            end
            if (state == WRITE_WEIGHT && rows_left == CHAN_COUNT_WIDTH'(1)) begin
                assert (beats_left == '0)
                    else $fatal(1, "Weight write completed before consuming the DMA burst");
            end
        end
    end
`else
    wire _unused_ok = &{1'b0, mem_rd_data_last, cfg_beats, cfg_padding};
`endif
endmodule

`default_nettype wire
