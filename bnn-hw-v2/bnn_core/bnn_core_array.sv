`default_nettype none

/*
============================================================================
bnn_core_array (多通道计算阵列，单行锁存与单组权重缓存)
----------------------------------------------------------------------------
控制与状态接口：
- enable 低电平禁止接收新输入行、禁止发起新的 Weight SRAM 读请求；已在途的 SRAM 读取、
  单元计算与输出握手继续完成。若当前行仍需读取权重或处理后续组，则保持 busy 暂停至 enable
  恢复；正常任务应在所有输出完成后 disable，不应依赖 disable 自动排空尚未发起的读。
- busy 高电平表示阵列仍有未完成的工作（包括当前行的组循环未走完、任一 unit 非空闲、
  或仍有未返回的在途 Weight SRAM 读请求）。上层控制器在拉低 enable 后须等待 !busy，
  方可改写 Weight SRAM 或拉高 weight_cfg_valid。
- weight_cfg_valid：仅在 !enable && !busy 时拉高 1 拍，锁存本次有效权重组数 G
  (weight_cfg_groups)。复位后 G = 0，配置合法权重组数前不接收输入。

权重 SRAM 读接口：
- 固定延迟 SRAM_READ_LATENCY 拍按序返回，地址映射为 WEIGHT_BASE_ADDR + g * ODMA_FP32_NUM + ch。
- 简化为单组权重缓存，不做乒乓预取；接收输入行后，每组逐通道串行读取，最多一笔读在途。
- 请求在 T 沿发出，在 T + SRAM_READ_LATENCY 沿采样返回值；disable 不阻止在途读返回。

输入激活与反量化控制接口：
- 每握手 1 次（idma_valid && idma_ready）接收输入矩阵的 1 行（128 个 INT8）、该行的
  idma_scale 及 idma_dequant_en，使前级 bnn_input_quant_buffer 握手后即可处理下一行。
- 阵列锁存该行 data / scale / dequant_en 后拉低 idma_ready，内部自动遍历 g = 0 .. G-1 组权重，
  每次凑齐整组 ODMA_FP32_NUM 通道权重后原子广播给全部 bnn_core_unit（由 unit 在提交沿各自锁存
  scale 与 dequant_en 完成反量化，阵列输出端不再向外传递 odma_scale）。

输出接口：
- 每接收 1 行输入，按 g = 0 .. G-1 顺序产生 G 次输出握手，odma_data[32*i +: 32] 对应第 i 通道。
- odma_row_last 指示当前行的第 G-1 组输出（供调试/断言观测）。
- 仅全部 unit 输出有效且 odma_ready 时同时消费；背压期间权重、组号与输出保持稳定。
============================================================================
*/
module bnn_core_array #(
    parameter int unsigned WEIGHT_WIDTH      = 144,
    parameter int unsigned IDMA_INT8_NUM     = 128,
    parameter int unsigned ODMA_FP32_NUM     = 4,
    parameter int unsigned SRAM_ADDR_WIDTH   = 7,
    parameter int unsigned SRAM_READ_LATENCY = 1,
    parameter logic [SRAM_ADDR_WIDTH-1:0] WEIGHT_BASE_ADDR = '0,
    parameter int unsigned MAX_WEIGHT_GROUPS = (1 << SRAM_ADDR_WIDTH) / ODMA_FP32_NUM,
    parameter int unsigned GROUP_COUNT_WIDTH = $clog2(MAX_WEIGHT_GROUPS + 1)
) (
    input  wire                             clock,
    input  wire                             reset,

    input  wire                             enable,
    output wire                             busy,

    input  wire                             weight_cfg_valid,
    input  wire [GROUP_COUNT_WIDTH-1:0]     weight_cfg_groups,

    output wire                             sram_en,
    output wire [SRAM_ADDR_WIDTH-1:0]       sram_addr,
    input  wire [WEIGHT_WIDTH-1:0]          sram_rdata,

    input  wire [(IDMA_INT8_NUM*8)-1:0]     idma_data,
    input  wire [31:0]                      idma_scale,
    input  wire                             idma_dequant_en,
    input  wire                             idma_valid,
    output wire                             idma_ready,

    output wire [(ODMA_FP32_NUM*32)-1:0]    odma_data,
    output wire                             odma_row_last,
    output wire                             odma_valid,
    input  wire                             odma_ready
);
    localparam int unsigned CHANNEL_WIDTH = (ODMA_FP32_NUM > 1) ? $clog2(ODMA_FP32_NUM) : 1;
    localparam int unsigned DELAY_WIDTH = (SRAM_READ_LATENCY > 1) ? $clog2(SRAM_READ_LATENCY) : 1;

    typedef enum logic [2:0] {
        IDLE, READ_REQUEST, READ_RETURN, SUBMIT, WAIT_OUTPUT
    } state_t;
    state_t state;

    logic [GROUP_COUNT_WIDTH-1:0] groups;
    logic [GROUP_COUNT_WIDTH-1:0] group_index;
    logic [CHANNEL_WIDTH-1:0] channel_index;
    logic [DELAY_WIDTH-1:0] read_delay;
    logic [(IDMA_INT8_NUM*8)-1:0] row_data;
    logic [31:0] row_scale;
    logic row_dequant_en;
    logic [WEIGHT_WIDTH-1:0] weights [0:ODMA_FP32_NUM-1];

    wire [ODMA_FP32_NUM-1:0] unit_input_ready;
    wire [ODMA_FP32_NUM-1:0] unit_output_valid;
    wire [(ODMA_FP32_NUM*32)-1:0] unit_output_data;
    wire all_units_ready = &unit_input_ready;
    wire unit_input_valid = !reset && (state == SUBMIT);
    wire unit_output_ready = odma_valid && odma_ready;
    wire last_group = (group_index == groups - GROUP_COUNT_WIDTH'(1));

    assign busy = !reset && (state != IDLE);
    assign idma_ready = !reset && enable && (state == IDLE) && (groups != '0);
    assign sram_en = !reset && enable && (state == READ_REQUEST);
    assign sram_addr = reset ? '0 : SRAM_ADDR_WIDTH'(
        64'(WEIGHT_BASE_ADDR) + 64'(group_index) * 64'(ODMA_FP32_NUM) + 64'(channel_index));
    assign odma_valid = !reset && (state == WAIT_OUTPUT) && (&unit_output_valid);
    assign odma_data = (!reset && (state == WAIT_OUTPUT)) ? unit_output_data : '0;
    assign odma_row_last = !reset && (state == WAIT_OUTPUT) && last_group;

    for (genvar ch = 0; ch < ODMA_FP32_NUM; ch = ch + 1) begin : g_unit
        bnn_core_unit #(
            .WEIGHT_WIDTH  (WEIGHT_WIDTH),
            .IDMA_INT8_NUM (IDMA_INT8_NUM)
        ) u_unit (
            .clock            (clock),
            .reset            (reset),
            .idata_data       (row_data),
            .idata_scale      (row_scale),
            .idata_dequant_en (row_dequant_en),
            .w_data           (weights[ch]),
            .idata_valid      (unit_input_valid),
            .idata_ready      (unit_input_ready[ch]),
            .odata_data       (unit_output_data[32*ch +: 32]),
            .odata_valid      (unit_output_valid[ch]),
            .odata_ready      (unit_output_ready)
        );
    end

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            groups <= '0;
            group_index <= '0;
            channel_index <= '0;
            read_delay <= '0;
            row_data <= '0;
            row_scale <= '0;
            row_dequant_en <= 1'b0;
            for (int ch = 0; ch < ODMA_FP32_NUM; ch = ch + 1)
                weights[ch] <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (weight_cfg_valid && !enable) begin
                        groups <= weight_cfg_groups;
                        group_index <= '0;
                        channel_index <= '0;
                    end
                    if (idma_valid && idma_ready) begin
                        row_data <= idma_data;
                        row_scale <= idma_scale;
                        row_dequant_en <= idma_dequant_en;
                        group_index <= '0;
                        channel_index <= '0;
                        state <= READ_REQUEST;
                    end
                end
                READ_REQUEST: begin
                    if (sram_en) begin
                        // T 请求，T+1 首次检查；L=1 时直接在 T+1 采样。
                        read_delay <= DELAY_WIDTH'(SRAM_READ_LATENCY - 1);
                        state <= READ_RETURN;
                    end
                end
                READ_RETURN: begin
                    if (read_delay != '0) begin
                        read_delay <= read_delay - DELAY_WIDTH'(1);
                    end else begin
                        weights[channel_index] <= sram_rdata;
                        if (channel_index == CHANNEL_WIDTH'(ODMA_FP32_NUM - 1)) begin
                            state <= SUBMIT;
                        end else begin
                            channel_index <= channel_index + CHANNEL_WIDTH'(1);
                            state <= READ_REQUEST;
                        end
                    end
                end
                SUBMIT: begin
                    // valid 独立于 ready；同复位、同提交、同释放保证各 unit 起点一致。
                    // 用全组 ready 推进状态，输出也仅允许整组同时消费。
                    if (all_units_ready)
                        state <= WAIT_OUTPUT;
                end
                WAIT_OUTPUT: begin
                    if (unit_output_ready) begin
                        if (last_group) begin
                            state <= IDLE;
                        end else begin
                            group_index <= group_index + GROUP_COUNT_WIDTH'(1);
                            channel_index <= '0;
                            state <= READ_REQUEST;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (IDMA_INT8_NUM > 0 && WEIGHT_WIDTH == IDMA_INT8_NUM + 16)
            else $fatal(1, "Invalid activation/weight widths");
        assert (ODMA_FP32_NUM > 0 && SRAM_READ_LATENCY >= 1)
            else $fatal(1, "Channel count and SRAM read latency must be positive");
        assert (SRAM_ADDR_WIDTH >= 1 && SRAM_ADDR_WIDTH <= 63)
            else $fatal(1, "SRAM_ADDR_WIDTH must be in [1, 63]");
        assert (MAX_WEIGHT_GROUPS > 0 &&
                64'(MAX_WEIGHT_GROUPS) * 64'(ODMA_FP32_NUM) <=
                (64'd1 << SRAM_ADDR_WIDTH) - 64'(WEIGHT_BASE_ADDR))
            else $fatal(1, "Weight groups exceed SRAM capacity");
        assert (GROUP_COUNT_WIDTH >= $clog2(64'(MAX_WEIGHT_GROUPS) + 64'd1) &&
                GROUP_COUNT_WIDTH <= 64)
            else $fatal(1, "GROUP_COUNT_WIDTH cannot represent the configured group count");
    end

    always @(posedge clock) begin
        if (!reset) begin
            if (weight_cfg_valid) begin
                assert (!enable && !busy)
                    else $fatal(1, "Weight configuration requires !enable && !busy");
                // 无符号 G-1 范围检查也排除 G=0，且满量程参数不会产生常量比较告警。
                assert ((64'(weight_cfg_groups) - 64'd1) < 64'(MAX_WEIGHT_GROUPS))
                    else $fatal(1, "Invalid weight group count");
            end
            if (state == SUBMIT) begin
                assert ((unit_input_ready == '0) || all_units_ready)
                    else $fatal(1, "Units must accept a group atomically");
            end
        end
    end

    assert property (@(posedge clock) disable iff (reset)
        odma_valid && !odma_ready |=> odma_valid && $stable({odma_data, odma_row_last}))
        else $fatal(1, "Array output changed under backpressure");
`endif
endmodule

`default_nettype wire
