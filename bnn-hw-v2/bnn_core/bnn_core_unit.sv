`default_nettype none

/*
============================================================================
bnn_core_unit (顺序 INT8 点积 + 权重缩放 + 可选激活反量化)
----------------------------------------------------------------------------
- WEIGHT_WIDTH == IDMA_INT8_NUM + 16，IDMA_INT8_NUM > 0。
- idata_data[8*i +: 8] 为有符号 INT8；硬件也正确处理 -128，但软件不保证产生该值。
- w_data[16+i]：1 表示 +1，0 表示 -1；w_data[15:0] 为 FP16 scale。
- 输入握手时锁存全部 data / weight / scale / dequant_en，随后每拍处理一个 INT8。
  点积使用精确有符号 32-bit 加减；参数应保证点积不溢出 INT32。
- 点积转 FP32 后乘 FP16 转换的权重 scale；可选再乘 idata_scale。
  使用 bnn_fp_math：FP32 输入/输出次正规数 FTZ，有限结果向零截断，非 RNE。
  FP16 转换本身保留次正规数，NaN 统一为 7fc00000。
- 单笔事务在途，ready 不依赖 valid；输出遇背压时 valid 与 data 保持不变。
============================================================================
*/
module bnn_core_unit #(
    parameter int unsigned WEIGHT_WIDTH  = 144,
    parameter int unsigned IDMA_INT8_NUM = 128
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

    localparam int unsigned INDEX_WIDTH = (IDMA_INT8_NUM > 1) ? $clog2(IDMA_INT8_NUM) : 1;
    typedef enum logic [2:0] {IDLE, DOT, WEIGHT_SCALE, ROW_SCALE, OUTPUT_DATA} state_t;
    state_t state;
    logic [(IDMA_INT8_NUM*8)-1:0] data_reg;
    logic [IDMA_INT8_NUM-1:0] weights_reg;
    logic [15:0] weight_scale_reg;
    logic [31:0] row_scale_reg, result_reg;
    logic dequant_reg;
    logic [INDEX_WIDTH-1:0] index_reg;
    logic signed [31:0] dot_reg, activation, dot_next;
    logic [7:0] activation_byte;

    assign idata_ready = !reset && (state == IDLE);
    assign odata_valid = !reset && (state == OUTPUT_DATA);
    assign odata_data = result_reg;
    assign activation_byte = data_reg[int'(index_reg)*8 +: 8];
    assign activation = {{24{activation_byte[7]}}, activation_byte};
    assign dot_next = weights_reg[index_reg] ? (dot_reg + activation) : (dot_reg - activation);

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            data_reg <= '0;
            weights_reg <= '0;
            weight_scale_reg <= '0;
            row_scale_reg <= '0;
            dequant_reg <= 1'b0;
            index_reg <= '0;
            dot_reg <= '0;
            result_reg <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (idata_valid && idata_ready) begin
                        data_reg <= idata_data;
                        weights_reg <= w_data[16 +: IDMA_INT8_NUM];
                        weight_scale_reg <= w_data[15:0];
                        row_scale_reg <= idata_scale;
                        dequant_reg <= idata_dequant_en;
                        index_reg <= '0;
                        dot_reg <= '0;
                        state <= DOT;
                    end
                end
                DOT: begin
                    dot_reg <= dot_next;
                    if (index_reg == INDEX_WIDTH'(IDMA_INT8_NUM - 1))
                        state <= WEIGHT_SCALE;
                    else
                        index_reg <= index_reg + 1'b1;
                end
                WEIGHT_SCALE: begin
                    result_reg <= fp32_mul_ftz(int32_to_fp32(dot_reg), fp16_to_fp32(weight_scale_reg));
                    state <= dequant_reg ? ROW_SCALE : OUTPUT_DATA;
                end
                ROW_SCALE: begin
                    result_reg <= fp32_mul_ftz(result_reg, row_scale_reg);
                    state <= OUTPUT_DATA;
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
        assert (64'(IDMA_INT8_NUM) * 64'd128 <= 64'h7fff_ffff)
            else $fatal(1, "IDMA_INT8_NUM may overflow signed 32-bit dot accumulator");
    end

    assert property (@(posedge clock) disable iff (reset)
        odata_valid && !odata_ready |=> odata_valid && $stable(odata_data))
        else $fatal(1, "Unit output changed under backpressure");
`endif
endmodule

`default_nettype wire
