`default_nettype none

/*
============================================================================
bnn_output_post_sram (跨块 FP32 累加与结果 SRAM)
----------------------------------------------------------------------------
- 简单双端口 1R1W memory，位宽 ODMA_FP32_NUM * 32，深度 OSRAM_DEPTH；不复位清数组。
- 空闲时 session_start 锁存 cfg_accum_enable / cfg_total_beats，地址从 0 递增。
  cfg_total_beats 不得超过 OSRAM_DEPTH；零拍 session 立即返回 done。
- 每次握手锁存一拍 odma_data；覆盖模式下一拍写入，累加模式在接收沿同步读旧值，
  下一拍逐路组合计算 fp32_add_ftz 并写回。有限结果向零截断，次正规数 FTZ，非 RNE。
  不追求连续拍吞吐，写完后才接收下一拍。session_done 仅在最后写入完成沿产生一拍。
- session_busy 期间读端口归累加使用，不接受 writer 读请求；上层保证二者互斥。
  空闲时 osram_ren 发起同步读，经 OSRAM_READ_LATENCY >= 1 级寄存器输出，
  与顶层 Weight SRAM 相同；第一级无请求时保持，后续 pipeline 每拍前移。
============================================================================
*/
module bnn_output_post_sram #(
    parameter int unsigned ODMA_FP32_NUM      = 4,
    parameter int unsigned MAX_ROWS           = 128,
    parameter int unsigned MAX_WEIGHT_GROUPS  = 32,
    parameter int unsigned OSRAM_DEPTH        = MAX_ROWS * MAX_WEIGHT_GROUPS,
    parameter int unsigned OSRAM_ADDR_WIDTH   = $clog2(OSRAM_DEPTH),
    parameter int unsigned BEAT_COUNT_WIDTH   = $clog2(OSRAM_DEPTH + 1),
    parameter int unsigned OSRAM_READ_LATENCY = 1
) (
    input  wire                             clock,
    input  wire                             reset,

    input  wire                             session_start,
    output wire                             session_busy,
    output wire                             session_done,
    input  wire                             cfg_accum_enable,
    input  wire [BEAT_COUNT_WIDTH-1:0]      cfg_total_beats,

    input  wire [(ODMA_FP32_NUM*32)-1:0]    odma_data,
    input  wire                             odma_valid,
    output wire                             odma_ready,

    input  wire                             osram_ren,
    input  wire [OSRAM_ADDR_WIDTH-1:0]      osram_raddr,
    output wire [(ODMA_FP32_NUM*32)-1:0]    osram_rdata
);
    import bnn_fp_math::*;

    localparam int unsigned DATA_WIDTH = ODMA_FP32_NUM * 32;
    typedef enum logic [1:0] {IDLE, RECEIVE, WRITE_BACK} state_t;
    state_t state;
    logic accum_reg, done_reg;
    logic [BEAT_COUNT_WIDTH-1:0] remaining_reg;
    logic [OSRAM_ADDR_WIDTH-1:0] address_reg;
    logic [DATA_WIDTH-1:0] incoming_reg, write_data;
    logic [DATA_WIDTH-1:0] memory [0:OSRAM_DEPTH-1];
    logic [DATA_WIDTH-1:0] read_pipe [0:OSRAM_READ_LATENCY-1];
    wire read_enable;
    wire [OSRAM_ADDR_WIDTH-1:0] read_address;

    assign session_busy = !reset && (state != IDLE);
    assign session_done = !reset && done_reg;
    assign odma_ready = !reset && (state == RECEIVE);
    assign read_enable = !reset &&
        (((state == RECEIVE) && odma_valid && accum_reg) ||
         ((state == IDLE) && !session_start && osram_ren));
    assign read_address = (state == IDLE) ? osram_raddr : address_reg;
    assign osram_rdata = read_pipe[OSRAM_READ_LATENCY-1];

    always_comb begin
        write_data = incoming_reg;
        if (accum_reg) begin
            for (int lane = 0; lane < ODMA_FP32_NUM; lane++)
                write_data[lane*32 +: 32] = fp32_add_ftz(
                    read_pipe[0][lane*32 +: 32], incoming_reg[lane*32 +: 32]);
        end
    end

    // No reset on memory or read registers, preserving synchronous SRAM inference.
    // Internal accumulation consumes stage zero; writer uses the full pipeline.
    always_ff @(posedge clock) begin
        if (!reset && (state == WRITE_BACK))
            memory[address_reg] <= write_data;
        if (read_enable)
            read_pipe[0] <= memory[read_address];
        for (int stage = 1; stage < OSRAM_READ_LATENCY; stage++)
            read_pipe[stage] <= read_pipe[stage-1];
    end

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            accum_reg <= 1'b0;
            done_reg <= 1'b0;
            remaining_reg <= '0;
            address_reg <= '0;
            incoming_reg <= '0;
        end else begin
            done_reg <= 1'b0;
            case (state)
                IDLE: begin
                    if (session_start) begin
                        accum_reg <= cfg_accum_enable;
                        remaining_reg <= cfg_total_beats;
                        address_reg <= '0;
                        if (cfg_total_beats == 0)
                            done_reg <= 1'b1;
                        else
                            state <= RECEIVE;
                    end
                end
                RECEIVE: begin
                    if (odma_valid && odma_ready) begin
                        incoming_reg <= odma_data;
                        state <= WRITE_BACK;
                    end
                end
                WRITE_BACK: begin
                    if (remaining_reg == BEAT_COUNT_WIDTH'(1)) begin
                        state <= IDLE;
                        done_reg <= 1'b1;
                    end else begin
                        remaining_reg <= remaining_reg - 1'b1;
                        address_reg <= address_reg + 1'b1;
                        state <= RECEIVE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (ODMA_FP32_NUM > 0 && MAX_ROWS > 0 && MAX_WEIGHT_GROUPS > 0 && OSRAM_DEPTH > 0)
            else $fatal(1, "Output SRAM dimensions must be positive");
        assert (OSRAM_READ_LATENCY >= 1)
            else $fatal(1, "OSRAM_READ_LATENCY must be >= 1");
        assert ((64'd1 << OSRAM_ADDR_WIDTH) >= 64'(OSRAM_DEPTH) &&
                (64'd1 << BEAT_COUNT_WIDTH) > 64'(OSRAM_DEPTH))
            else $fatal(1, "OSRAM address/beat widths insufficient for OSRAM_DEPTH");
    end

    always @(posedge clock) begin
        if (!reset) begin
            if (session_start && !session_busy) begin
                assert (64'(cfg_total_beats) <= 64'(OSRAM_DEPTH))
                    else $fatal(1, "Session total beats exceed OSRAM_DEPTH");
            end
            assert (!(osram_ren && (session_busy || session_start)))
                else $fatal(1, "Output SRAM read port contention during active session");
        end
    end
`endif
endmodule

`default_nettype wire
