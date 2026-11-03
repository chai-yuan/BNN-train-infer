`default_nettype none

/*
============================================================================
bnn_top_controller (总控制器 —— 单阶段原子 64-bit 指令集架构)
----------------------------------------------------------------------------
接收上层单阶段原子指令（cmd_valid && cmd_ready），串行编排下级组件完成分块矩阵运算。
每条指令高 32 位 [63:32] 统一承载 16-Byte 对齐的字节地址或行步长，低 32 位 [31:0] 承载
8-bit 操作码 [7:0] 及维度/模式控制字段：

  63                              32 31      24 23          16 15           8 7        0
 +----------------------------------+----------+--------------+--------------+----------+
 |     addr / row_stride (32b)      |  flags   |  N (rows_n)  |  M (rows_m)  |  opcode  |
 +----------------------------------+----------+--------------+--------------+----------+

实现约定：串行执行、无指令队列；每笔接收指令恰好返回一次互斥的 done/error 脉冲。
WRITEBACK 要求已有完整结果；accum 要求旧结果与本次 M/N 完全相同。
同 N 重载权重保留旧结果供跨 K 块累加，改变 N 则使旧结果无效。

原子指令定义：
1. OP_SET_SRC_STRIDE (opcode = 8'h01) —— 配置输入矩阵行步长（单拍完成）
   - [63:32] src_row_stride：输入相邻行字节步长（须 16B 对齐；复位默认 0 表示单行紧密排列）。

2. OP_SET_DST_STRIDE (opcode = 8'h02) —— 配置输出矩阵行步长（单拍完成）
   - [63:32] dst_row_stride：输出相邻行字节步长（须 16B 对齐；复位默认 0 表示按 G*16B 紧密排列）。

3. OP_LOAD_WEIGHT    (opcode = 8'h10) —— 加载权重分块至 Weight SRAM
   - [63:32] wloader_base_addr：权重分块外部内存首地址（须 16B 对齐）。
   - [23:16] N (weight_rows_n)：本次权重分块有效行数（即有效输出通道数，范围 1 .. MAX_OUT_CHANS）。
   - 执行动作：
     a) 锁存 N，自动计算有效权重组数 G = (N + 3) >> 2；
     b) 保持 array_enable = 0，确认 !array_busy 后切换 dma_rd_mux_sel = 0；
     c) 启动 bnn_weight_loader 写入 N 条有效权重并自动补零对齐至 4*G；
     d) 等待 wloader_done 后，向 bnn_core_array 发出 1 拍 array_weight_cfg_valid 脉冲刷新 G。

4. OP_COMPUTE        (opcode = 8'h20) —— 读取输入 + (可选)动态量化 + 阵列计算 + 暂存累加
   - [63:32] iqbuf_base_addr：输入分块外部内存首地址（须 16B 对齐）。
   - [15:8]  M (row_num_m)  ：本次输入分块行数（范围 1 .. MAX_ROWS）。
   - [24]    accum_en       ：0 = 首块覆盖写入暂存 SRAM；1 = 与暂存 SRAM 旧值累加。
   - [25]    input_is_int8  ：0 = FP32 输入（开启行动态量化与 unit 反量化）；
                              1 = 预量化 INT8 直通输入（关闭动态量化与 unit 反量化）。
   - 执行动作：
     a) 锁存 M，结合当前已配置的 G 计算 opost_total_beats = M * G；
     b) 切换 dma_rd_mux_sel = 1，拉高 array_enable = 1；
     c) 同时启动 bnn_output_accum（结果 SRAM 访问层）与 bnn_input_quant_buffer；
     d) 等待 opost_session_done 后拉低 array_enable = 0，并等待 !array_busy 确保在途读排空。

5. OP_WRITEBACK      (opcode = 8'h30) —— 将暂存 SRAM 分块结果写回外部内存
   - [63:32] owriter_base_addr：输出分块外部内存首地址（须 16B 对齐）。
   - 执行动作：复用当前锁存的 M、N 与 dst_row_stride 启动 bnn_output_writer（由 N 自动生成
     每行拍数 G 与尾拍字节掩码），等待 owriter_done。

非法参数防护：
- 若收到未知 opcode、M/N 为 0 或超出上限、在未配置权重(N==0)时发起 OP_COMPUTE/OP_WRITEBACK、
  或地址/步长低 4 位非 0（未 16B 对齐），控制器拒绝启动下级模块并拉高 cmd_error 脉冲 1 拍。

经典 2D 分块矩阵乘法指令序列示例（计算 A_11 = B_11 * C_11^T + B_12 * C_21^T）：
- 指令 1: OP_SET_SRC_STRIDE (src_stride = K_total * 4)
- 指令 2: OP_SET_DST_STRIDE (dst_stride = N_total * 4)   // 须 16B 对齐
- 指令 3: OP_LOAD_WEIGHT    (addr = &C_11, N = 128)
- 指令 4: OP_COMPUTE        (addr = &B_11, M = 128, accum_en = 0, input_is_int8 = 0)
- 指令 5: OP_LOAD_WEIGHT    (addr = &C_21, N = 128)
- 指令 6: OP_COMPUTE        (addr = &B_12, M = 128, accum_en = 1, input_is_int8 = 0)
- 指令 7: OP_WRITEBACK      (addr = &A_11)
============================================================================
*/
module bnn_top_controller #(
    parameter int unsigned ADDR_WIDTH        = 32,
    parameter int unsigned STRIDE_WIDTH      = 32,
    parameter int unsigned ODMA_FP32_NUM     = 4,
    parameter int unsigned MAX_ROWS          = 128,
    parameter int unsigned MAX_WEIGHT_GROUPS = 32,
    parameter int unsigned MAX_OUT_CHANS     = MAX_WEIGHT_GROUPS * ODMA_FP32_NUM,
    parameter int unsigned ROW_COUNT_WIDTH   = $clog2(MAX_ROWS + 1),
    parameter int unsigned GROUP_COUNT_WIDTH = $clog2(MAX_WEIGHT_GROUPS + 1),
    parameter int unsigned CHAN_COUNT_WIDTH  = $clog2(MAX_OUT_CHANS + 1),
    parameter int unsigned OSRAM_DEPTH       = MAX_ROWS * MAX_WEIGHT_GROUPS,
    parameter int unsigned BEAT_COUNT_WIDTH  = $clog2(OSRAM_DEPTH + 1)
) (
    input  wire                             clock,
    input  wire                             reset,

    // ------------------------------------------------------------------------
    // 上层宿主机命令与状态接口
    // ------------------------------------------------------------------------
    input  wire                             cmd_valid,
    input  wire [63:0]                      cmd_inst,
    output wire                             cmd_ready,
    output wire                             cmd_done,
    output wire                             cmd_error,
    output wire                             status_busy,

    // ------------------------------------------------------------------------
    // 下级组件控制信号
    // ------------------------------------------------------------------------
    // 顶层 DMA 读通道 MUX 控制：0 选 bnn_weight_loader，1 选 bnn_input_quant_buffer
    output wire                             dma_rd_mux_sel,

    // 1. 去往 bnn_weight_loader
    output wire                             wloader_start,
    input  wire                             wloader_busy,
    input  wire                             wloader_done,
    output wire [ADDR_WIDTH-1:0]            wloader_base_addr,
    output wire [CHAN_COUNT_WIDTH-1:0]       wloader_weight_rows_n,

    // 2. 去往 bnn_core_array
    output wire                             array_enable,
    input  wire                             array_busy,
    output wire                             array_weight_cfg_valid,
    output wire [GROUP_COUNT_WIDTH-1:0]     array_weight_cfg_groups,

    // 3. 去往 bnn_input_quant_buffer
    output wire                             iqbuf_start,
    input  wire                             iqbuf_busy,
    input  wire                             iqbuf_done,
    output wire                             iqbuf_input_is_int8,
    output wire [ADDR_WIDTH-1:0]            iqbuf_base_addr,
    output wire [STRIDE_WIDTH-1:0]          iqbuf_row_stride,
    output wire [ROW_COUNT_WIDTH-1:0]       iqbuf_row_num_m,

    // 4. 去往 bnn_output_accum（结果 SRAM 访问层）
    output wire                             opost_session_start,
    input  wire                             opost_session_busy,
    input  wire                             opost_session_done,
    output wire                             opost_accum_enable,
    output wire [BEAT_COUNT_WIDTH-1:0]      opost_total_beats,

    // 5. 去往 bnn_output_writer
    output wire                             owriter_start,
    input  wire                             owriter_busy,
    input  wire                             owriter_done,
    output wire [ADDR_WIDTH-1:0]            owriter_base_addr,
    output wire [STRIDE_WIDTH-1:0]          owriter_row_stride,
    output wire [ROW_COUNT_WIDTH-1:0]       owriter_row_num_m,
    output wire [CHAN_COUNT_WIDTH-1:0]      owriter_weight_rows_n
);
    typedef enum logic [3:0] {
        IDLE, LOAD_DRAIN, LOAD_START, LOAD_WAIT, LOAD_CONFIG,
        COMPUTE_START, COMPUTE_WAIT, COMPUTE_DRAIN, WRITE_START, WRITE_WAIT
    } state_t;
    state_t state;
    logic done_reg, error_reg, rd_select;
    logic [ADDR_WIDTH-1:0] address_reg;
    logic [STRIDE_WIDTH-1:0] source_stride, destination_stride;
    logic [CHAN_COUNT_WIDTH-1:0] loaded_channels, result_channels;
    logic [GROUP_COUNT_WIDTH-1:0] groups;
    logic [ROW_COUNT_WIDTH-1:0] operation_rows, result_rows;
    logic accum, input_int8, result_valid;
    wire [7:0] opcode = cmd_inst[7:0];
    wire [7:0] rows_m = cmd_inst[15:8];
    wire [7:0] rows_n = cmd_inst[23:16];
    wire aligned_address = cmd_inst[35:32] == 0;
    wire address_fits = 64'(cmd_inst[63:32]) == 64'(ADDR_WIDTH'(cmd_inst[63:32]));
    wire stride_fits = (64'(cmd_inst[63:32]) == 64'(STRIDE_WIDTH'(cmd_inst[63:32]))) && address_fits;
    wire children_idle = !array_busy && !wloader_busy && !iqbuf_busy &&
                         !opost_session_busy && !owriter_busy;

    assign cmd_ready = state == IDLE && !done_reg && !error_reg && !reset;
    assign cmd_done = done_reg && !reset;
    assign cmd_error = error_reg && !reset;
    assign status_busy = !reset && (state != IDLE);
    assign dma_rd_mux_sel = rd_select;
    assign wloader_start = state == LOAD_START && !reset;
    assign wloader_base_addr = address_reg;
    assign wloader_weight_rows_n = loaded_channels;
    assign array_enable = (state == COMPUTE_START || state == COMPUTE_WAIT) && !reset;
    assign array_weight_cfg_valid = state == LOAD_CONFIG && !reset;
    assign array_weight_cfg_groups = groups;
    assign iqbuf_start = state == COMPUTE_START && !reset;
    assign iqbuf_input_is_int8 = input_int8;
    assign iqbuf_base_addr = address_reg;
    assign iqbuf_row_stride = source_stride;
    assign iqbuf_row_num_m = operation_rows;
    assign opost_session_start = state == COMPUTE_START && !reset;
    assign opost_accum_enable = accum;
    assign opost_total_beats = BEAT_COUNT_WIDTH'(operation_rows) * BEAT_COUNT_WIDTH'(groups);
    assign owriter_start = state == WRITE_START && !reset;
    assign owriter_base_addr = address_reg;
    assign owriter_row_stride = destination_stride;
    assign owriter_row_num_m = result_rows;
    assign owriter_weight_rows_n = result_channels;

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            done_reg <= 0;
            error_reg <= 0;
            rd_select <= 0;
            address_reg <= 0;
            source_stride <= 0;
            destination_stride <= 0;
            loaded_channels <= 0;
            result_channels <= 0;
            groups <= 0;
            operation_rows <= 0;
            result_rows <= 0;
            accum <= 0;
            input_int8 <= 0;
            result_valid <= 0;
        end else begin
            done_reg <= 0;
            error_reg <= 0;
            case (state)
                IDLE: if (cmd_valid && cmd_ready) begin
                    case (opcode)
                        8'h01, 8'h02: begin
                            if (!aligned_address || !stride_fits) error_reg <= 1;
                            else begin
                                if (opcode == 8'h01) source_stride <= STRIDE_WIDTH'(cmd_inst[63:32]);
                                else destination_stride <= STRIDE_WIDTH'(cmd_inst[63:32]);
                                done_reg <= 1;
                            end
                        end
                        8'h10: begin
                            if (!aligned_address || !address_fits || rows_n == 0 || int'(rows_n) > MAX_OUT_CHANS)
                                error_reg <= 1;
                            else begin
                                address_reg <= ADDR_WIDTH'(cmd_inst[63:32]);
                                loaded_channels <= CHAN_COUNT_WIDTH'(rows_n);
                                groups <= GROUP_COUNT_WIDTH'((int'(rows_n) + ODMA_FP32_NUM - 1) / ODMA_FP32_NUM);
                                if (CHAN_COUNT_WIDTH'(rows_n) != result_channels) result_valid <= 0;
                                state <= LOAD_DRAIN;
                            end
                        end
                        8'h20: begin
                            if (!aligned_address || !address_fits || rows_m == 0 || int'(rows_m) > MAX_ROWS ||
                                loaded_channels == 0 || cmd_inst[31:26] != 0 ||
                                (cmd_inst[24] && (!result_valid || result_channels != loaded_channels ||
                                                 result_rows != ROW_COUNT_WIDTH'(rows_m))))
                                error_reg <= 1;
                            else begin
                                address_reg <= ADDR_WIDTH'(cmd_inst[63:32]);
                                operation_rows <= ROW_COUNT_WIDTH'(rows_m);
                                accum <= cmd_inst[24];
                                input_int8 <= cmd_inst[25];
                                rd_select <= 1;
                                result_valid <= 0;
                                state <= COMPUTE_START;
                            end
                        end
                        8'h30: begin
                            if (!aligned_address || !address_fits || !result_valid ||
                                result_channels != loaded_channels)
                                error_reg <= 1;
                            else begin
                                address_reg <= ADDR_WIDTH'(cmd_inst[63:32]);
                                state <= WRITE_START;
                            end
                        end
                        default: error_reg <= 1;
                    endcase
                end
                LOAD_DRAIN: if (children_idle) begin
                    // No outstanding read may remain when changing its response owner.
                    rd_select <= 0;
                    state <= LOAD_START;
                end
                LOAD_START: state <= LOAD_WAIT;
                LOAD_WAIT: if (wloader_done) state <= LOAD_CONFIG;
                LOAD_CONFIG: begin
                    done_reg <= 1;
                    state <= IDLE;
                end
                COMPUTE_START: state <= COMPUTE_WAIT;
                COMPUTE_WAIT: if (opost_session_done) state <= COMPUTE_DRAIN;
                COMPUTE_DRAIN: if (children_idle) begin
                    result_valid <= 1;
                    result_rows <= operation_rows;
                    result_channels <= loaded_channels;
                    done_reg <= 1;
                    state <= IDLE;
                end
                WRITE_START: state <= WRITE_WAIT;
                WRITE_WAIT: if (owriter_done) begin
                    done_reg <= 1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (ADDR_WIDTH > 0 && ADDR_WIDTH <= 64 && STRIDE_WIDTH > 0 && STRIDE_WIDTH <= 64)
            else $fatal(1, "invalid controller address/stride width");
        assert (ODMA_FP32_NUM > 0 && MAX_ROWS > 0 && MAX_OUT_CHANS > 0)
            else $fatal(1, "invalid controller dimensions");
    end
    always @(posedge clock) if (!reset) begin
        assert (!(cmd_done && cmd_error))
            else $fatal(1, "cmd_done and cmd_error asserted together");
        if (cmd_valid && cmd_ready)
            assert (children_idle)
                else $fatal(1, "controller accepted command while child module busy");
        if (array_weight_cfg_valid)
            assert (!array_enable && !array_busy) else $fatal(1, "weight configuration while array busy");
        if (iqbuf_done)
            assert (state == COMPUTE_WAIT || state == COMPUTE_DRAIN)
                else $fatal(1, "unexpected input completion");
    end
`else
    wire _unused_ok = &{1'b0, iqbuf_done};
`endif
endmodule

`default_nettype wire
