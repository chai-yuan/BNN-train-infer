`default_nettype none

/*
============================================================================
bnn_output_accum (结果 SRAM 的覆盖写 / 跨块 FP32 累加访问层)
----------------------------------------------------------------------------
职责边界：
本模块只包含"结果 SRAM 的访问逻辑与跨块后处理"，存储本体由
sram_result_wrapper（对 FPGA BRAM / ASIC 单端口 1RW SRAM 宏的浅包装）实例化
提供，访问层与存储宏解耦，便于替换厂商原语或调整深度/位宽。

单端口 1RW 适配：
- 存储宏同一拍只能读或写，本模块把三类访问合流到唯一端口：
  a) 累加读：RECEIVE 拍发起读旧值（en=1, we=0）；
  b) 累加写：WRITE_BACK 拍写回（en=1, we=1），天然与 a) 错开拍；
  c) 写回读：session 空闲时由 bnn_output_writer 发起读（en=1, we=0）。
  三者互斥，故不需要仲裁等待，单端口化不损失吞吐。
- 由于读延迟固定为 OSRAM_READ_LATENCY，累加路径在 RECEIVE 与 WRITE_BACK
  之间插入 SETTLE 等待，保证写回时读数据已对齐；OSRAM_READ_LATENCY = 1 时
  SETTLE 不进入，时序与双端口版本逐拍一致。

运行规则：
- 空闲时 session_start 锁存 cfg_accum_enable / cfg_total_beats，地址从 0 递增。
  cfg_total_beats 不得超过 OSRAM_DEPTH；零拍 session 立即返回 done。
- 每次握手锁存一拍 odma_data：
  * 覆盖模式：下一拍直接写入结果 SRAM；
  * 累加模式：接收沿同时发起同步读，等 OSRAM_READ_LATENCY 拍取回旧值后，
    逐路组合计算 fp32_add_ftz 并写回。有限结果向零截断，次正规数 FTZ，非 RNE。
  不追求连续拍吞吐，写完一拍后才接收下一拍；session_done 仅在最后写入完成沿
  产生一拍。
- session_busy 期间读端口归累加使用，不接受 writer 读请求；上层保证二者互斥。
  空闲时 osram_ren 发起同步读，经 OSRAM_READ_LATENCY 级流水输出，与
  Weight SRAM 的读延迟约定一致。
- 写端口 wstrb 恒为全 1：结果按整拍（ODMA_FP32_NUM 路 FP32）整体写回。
============================================================================
*/
module bnn_output_accum #(
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

    localparam int unsigned DATA_WIDTH  = ODMA_FP32_NUM * 32;
    localparam int unsigned STRB_WIDTH  = DATA_WIDTH / 8;
    localparam int unsigned WAIT_CYCLES = (OSRAM_READ_LATENCY > 1) ? (OSRAM_READ_LATENCY - 1) : 0;
    localparam int unsigned WAIT_WIDTH  = (WAIT_CYCLES > 1) ? $clog2(WAIT_CYCLES) : 1;

    typedef enum logic [1:0] {IDLE, RECEIVE, SETTLE, WRITE_BACK} state_t;
    state_t state;
    logic accum_reg, done_reg;
    logic [BEAT_COUNT_WIDTH-1:0] remaining_reg;
    logic [OSRAM_ADDR_WIDTH-1:0] address_reg;
    logic [WAIT_WIDTH-1:0]       wait_reg;
    logic [DATA_WIDTH-1:0]       incoming_reg, write_data;
    logic                        sram_we, sram_re, sram_en;
    logic [OSRAM_ADDR_WIDTH-1:0] sram_addr;

    wire read_enable;
    wire [OSRAM_ADDR_WIDTH-1:0] read_address;

    assign session_busy = !reset && (state != IDLE);
    assign session_done = !reset && done_reg;
    assign odma_ready = !reset && (state == RECEIVE);

    // 结果 SRAM 读请求：累加模式在接收沿读取旧值，空闲时让给 writer。
    assign read_enable = !reset &&
        (((state == RECEIVE) && odma_valid && accum_reg) ||
         ((state == IDLE) && !session_start && osram_ren));
    assign read_address = (state == IDLE) ? osram_raddr : address_reg;

    // 单端口 1RW 合流：写只发生在 WRITE_BACK 拍，读只发生在累加接收拍或
    // 空闲 writer 读拍，二者天然互斥（下方断言兜底），因此不需要仲裁等待。
    assign sram_we   = !reset && (state == WRITE_BACK);
    assign sram_re   = read_enable;
    assign sram_en   = sram_we || sram_re;
    assign sram_addr = sram_we ? address_reg : read_address;

    // 累加数据通路：读回的旧值在 WRITE_BACK 拍已对齐（见下方 SETTLE 说明）。
    always_comb begin
        write_data = incoming_reg;
        if (accum_reg) begin
            for (int lane = 0; lane < ODMA_FP32_NUM; lane++)
                write_data[lane*32 +: 32] = fp32_add_ftz(
                    osram_rdata[lane*32 +: 32], incoming_reg[lane*32 +: 32]);
        end
    end

    // ------------------------------------------------------------------------
    // 结果 SRAM 存储体：浅包装，不含任何控制逻辑，不在复位时清零。
    // ------------------------------------------------------------------------
    sram_result_wrapper #(
        .DATA_WIDTH   (DATA_WIDTH),
        .DEPTH        (OSRAM_DEPTH),
        .ADDR_WIDTH   (OSRAM_ADDR_WIDTH),
        .READ_LATENCY (OSRAM_READ_LATENCY)
    ) u_result_sram (
        .clock (clock),
        .en    (sram_en),
        .we    (sram_we),
        .addr  (sram_addr),
        .wdata (write_data),
        .wstrb ({STRB_WIDTH{1'b1}}),
        .rdata (osram_rdata)
    );

    // 状态机不加复位信号以外的存储逻辑；结果 SRAM 不在复位时清空。
    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            accum_reg <= 1'b0;
            done_reg <= 1'b0;
            remaining_reg <= '0;
            address_reg <= '0;
            wait_reg <= '0;
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
                        // T 拍发起读，rdata 在 T + OSRAM_READ_LATENCY 拍有效，
                        // 因此累加模式需额外等待 OSRAM_READ_LATENCY-1 拍再写回。
                        if (WAIT_CYCLES == 0) begin
                            state <= WRITE_BACK;
                        end else begin
                            wait_reg <= WAIT_WIDTH'(WAIT_CYCLES - 1);
                            state <= SETTLE;
                        end
                    end
                end
                SETTLE: begin
                    if (wait_reg == '0)
                        state <= WRITE_BACK;
                    else
                        wait_reg <= wait_reg - 1'b1;
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
            assert (!(sram_we && sram_re))
                else $fatal(1, "Result SRAM single-port concurrent read and write violation");
        end
    end
`endif
endmodule

`default_nettype wire
