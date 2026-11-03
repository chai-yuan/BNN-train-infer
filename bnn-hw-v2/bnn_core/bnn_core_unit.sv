`default_nettype none

/*
============================================================================
bnn_core_unit (并行 INT8 点积 + 权重缩放 + 可选激活反量化)
----------------------------------------------------------------------------
- WEIGHT_WIDTH == IDMA_INT8_NUM + 16，IDMA_INT8_NUM > 0，DOT_LANES > 0。
- idata_data[8*i +: 8] 为有符号 INT8；硬件也正确处理 -128，但软件不保证产生该值。
- w_data[16+i]：1 表示 +1，0 表示 -1；w_data[15:0] 为 FP16 scale。
- 输入握手时锁存全部 data / weight / scale / dequant_en，随后每拍通过移位寄存器与
  并行加法树处理至多 DOT_LANES 个 INT8（利用 -x = ~x + 1 异或拆解消除前置取反进位链）。
  点积使用精确有符号 32-bit 累加；参数应保证点积不溢出 INT32。
- 浮点后处理拆分为多拍流水并显式复用单一 FP32 乘法器：
  先于 FP_CONV 拍并行完成点积 INT32->FP32 与权重 scale FP16->FP32 转换，
  再于 WEIGHT_SCALE 拍相乘，可选于 ROW_SCALE 拍再乘 idata_scale。
  使用 bnn_fp_math：FP32 输入/输出次正规数 FTZ，有限结果向零截断，非 RNE。
  FP16 转换本身保留次正规数，NaN 统一为 7fc00000。
- 单笔事务在途，ready 不依赖 valid；输出遇背压时 valid 与 data 保持不变。
============================================================================
*/
module bnn_core_unit #(
    parameter int unsigned WEIGHT_WIDTH  = 144,
    parameter int unsigned IDMA_INT8_NUM = 128,
    parameter int unsigned DOT_LANES     = 64
) (
    input  wire                         clock,
    input  wire                         reset,

    input  wire [(IDMA_INT8_NUM*8)-1:0] idata_data,
    input  wire [31:0]                  idata_scale,
    input  wire                         idata_dequant_en,
    input  wire [WEIGHT_WIDTH-1:0]      w_data,
    input  wire                         idata_valid,
    output wire                         idata_ready,

    output wire [31:0]                  odata_data,
    output wire                         odata_valid,
    input  wire                         odata_ready
);
    import bnn_fp_math::*;

    localparam int unsigned EFF_LANES  = (IDMA_INT8_NUM < DOT_LANES) ? IDMA_INT8_NUM : DOT_LANES;
    localparam int unsigned DOT_STEPS  = (IDMA_INT8_NUM + EFF_LANES - 1) / EFF_LANES;
    localparam int unsigned STEP_WIDTH = (DOT_STEPS > 1) ? $clog2(DOT_STEPS) : 1;

    typedef enum logic [2:0] {
        IDLE,
        DOT,
        FP_CONV,
        WEIGHT_SCALE,
        ROW_SCALE,
        OUTPUT_DATA
    } state_t;
    state_t state;

    logic [(IDMA_INT8_NUM*8)-1:0] data_reg;
    logic [IDMA_INT8_NUM-1:0]     weights_reg;
    logic [15:0]                  weight_scale_reg;
    logic [31:0]                  row_scale_reg;
    logic [31:0]                  result_reg;
    logic [31:0]                  scale_fp32_reg;
    logic                         dequant_reg;
    logic [STEP_WIDTH-1:0]        step_reg;
    logic signed [31:0]           dot_reg, partial_sum, dot_next;
    logic [31:0]                  fp_mul_out;

    assign idata_ready = !reset && (state == IDLE);
    assign odata_valid = !reset && (state == OUTPUT_DATA);
    assign odata_data  = result_reg;

    always_comb begin
        partial_sum = '0;
        for (int i = 0; i < EFF_LANES; i++) begin
            logic              inv;
            logic signed [7:0] act_xored;

            if ((DOT_STEPS * EFF_LANES == IDMA_INT8_NUM) ||
                (int'(step_reg) * EFF_LANES + i < IDMA_INT8_NUM)) begin
                inv       = ~weights_reg[i];
                act_xored = signed'(data_reg[i*8 +: 8] ^ {8{inv}});
                partial_sum += 32'(act_xored) + 32'(inv);
            end
        end
    end

    assign dot_next   = dot_reg + partial_sum;
    assign fp_mul_out = fp32_mul_ftz(result_reg, scale_fp32_reg);

    always_ff @(posedge clock) begin
        if (reset) begin
            state            <= IDLE;
            data_reg         <= '0;
            weights_reg      <= '0;
            weight_scale_reg <= '0;
            row_scale_reg    <= '0;
            result_reg       <= '0;
            scale_fp32_reg   <= '0;
            dequant_reg      <= 1'b0;
            step_reg         <= '0;
            dot_reg          <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (idata_valid && idata_ready) begin
                        data_reg         <= idata_data;
                        weights_reg      <= w_data[16 +: IDMA_INT8_NUM];
                        weight_scale_reg <= w_data[15:0];
                        row_scale_reg    <= idata_scale;
                        dequant_reg      <= idata_dequant_en;
                        step_reg         <= '0;
                        dot_reg          <= '0;
                        state            <= DOT;
                    end
                end
                DOT: begin
                    dot_reg <= dot_next;
                    if (step_reg == STEP_WIDTH'(DOT_STEPS - 1)) begin
                        state <= FP_CONV;
                    end else begin
                        step_reg    <= step_reg + 1'b1;
                        data_reg    <= data_reg >> (EFF_LANES * 8);
                        weights_reg <= weights_reg >> EFF_LANES;
                    end
                end
                FP_CONV: begin
                    result_reg     <= int32_to_fp32(dot_reg);
                    scale_fp32_reg <= fp16_to_fp32(weight_scale_reg);
                    state          <= WEIGHT_SCALE;
                end
                WEIGHT_SCALE: begin
                    result_reg     <= fp_mul_out;
                    scale_fp32_reg <= row_scale_reg;
                    state          <= dequant_reg ? ROW_SCALE : OUTPUT_DATA;
                end
                ROW_SCALE: begin
                    result_reg <= fp_mul_out;
                    state      <= OUTPUT_DATA;
                end
                OUTPUT_DATA: begin
                    if (odata_ready) state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (IDMA_INT8_NUM > 0 && WEIGHT_WIDTH == IDMA_INT8_NUM + 16)
            else $fatal(1, "Invalid activation/weight widths");
        assert (DOT_LANES > 0)
            else $fatal(1, "DOT_LANES must be greater than 0");
        assert (64'(IDMA_INT8_NUM) * 64'd128 <= 64'h7fff_ffff)
            else $fatal(1, "IDMA_INT8_NUM may overflow signed 32-bit dot accumulator");
    end

    assert property (@(posedge clock) disable iff (reset)
        odata_valid && !odata_ready |=> odata_valid && $stable(odata_data))
        else $fatal(1, "Unit output changed under backpressure");
`endif
endmodule

`default_nettype wire
