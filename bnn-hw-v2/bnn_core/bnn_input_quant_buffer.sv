`default_nettype none

/*
============================================================================
bnn_input_quant_buffer (外部内存 FP32 动态量化 / INT8 直通 -> 单行缓冲 -> Array)
----------------------------------------------------------------------------
控制接口（来自 bnn_top_controller）：
- start 在 !busy 时拉高 1 拍启动整批 M 行的读取、（可选）动态量化与向阵列的推送；
  当拍锁存全部 cfg_* 参数。
- 当第 M 行成功与 bnn_core_array 完成 idma 握手后，拉高 done 脉冲 1 拍（共发起 M 次握手）。

2D 行步长与对齐约束：
- cfg_src_base_addr 与 cfg_src_row_stride 均须满足 16-Byte 对齐（低 4 位为 0）。
- 第 m 行（m = 0 .. M-1）外部内存首地址为：cfg_src_base_addr + m * effective_stride。
  若 cfg_src_row_stride == 0，硬件自动按单行紧密排列字节数步进（FP32 模式为 512 Bytes，
  INT8 直通模式为 128 Bytes）。

两种工作模式（由 cfg_input_is_int8 选择）：
1. FP32 动态量化模式 (cfg_input_is_int8 == 1'b0)：
   - 每行读取 IDMA_INT8_NUM (128) 个 FP32（占 32 拍 128-bit DMA 传输，512 Bytes）。
   - 硬件友好量化准则（软件参考模型同步适配硬件算术）：
     a) 次正规数（exp == 8'h00）按 FTZ（Flush-To-Zero）视为 +0.0；遍历求行最大绝对值 row_max_abs。
     b) 若 row_max_abs == 0：整行 idma_data 输出全 0，idma_scale = 32'h0000_0000。
     c) 若 row_max_abs > 0：反量化系数 idma_scale = FP32_FTZ(row_max_abs / 127.0)；
        行内各元素量化通过“指数差右移 (E_max - E_i) + 尾数定点乘除 (1.m_i * 127 / 1.m_max)”
        直接求得有符号 INT8 并截断至 [-127, +127]（禁止 -128），避免直接计算 FP32 倒数
        127.0 / row_max_abs 在小指数下产生浮点上溢。
   - 输出 idma_dequant_en = 1'b1，通知后级 bnn_core_unit 开启反量化乘法。

2. 预量化 INT8 直通模式 (cfg_input_is_int8 == 1'b1)：
   - 每行直接读取 IDMA_INT8_NUM (128) 个有符号 INT8（占 8 拍 128-bit DMA 传输，128 Bytes；
     范围 [-127, +127] 由软件侧保证）。
   - 跳过动态量化逻辑，直接装入行缓冲；输出 idma_scale = 32'h0000_0000，idma_dequant_en = 1'b0，
     通知后级 bnn_core_unit 关闭反量化通路。

简化实现：单行缓冲，串行完成 DMA 读取、逐元素最大值扫描、逐元素量化、行提交。
- FP32 运算向零截断，次正规数 FTZ；NaN/Inf 输入样本按零处理，不参与行最大值。
- 默认行步长为向上补齐到完整 DMA 拍的行字节数；默认参数仍是 FP32 512B / INT8 128B。
- 不做预取；行握手前 data/scale/dequant_en 始终保持稳定。
============================================================================
*/
module bnn_input_quant_buffer #(
    parameter int unsigned ADDR_WIDTH      = 32,
    parameter int unsigned STRIDE_WIDTH    = 32,
    parameter int unsigned DMA_DATA_WIDTH  = 128,
    parameter int unsigned IDMA_INT8_NUM   = 128,
    parameter int unsigned MAX_ROWS        = 128,
    parameter int unsigned ROW_COUNT_WIDTH = $clog2(MAX_ROWS + 1),
    parameter int unsigned DMA_LEN_WIDTH   = 16
) (
    input  wire                             clock,
    input  wire                             reset,

    input  wire                             start,
    output wire                             busy,
    output wire                             done,
    input  wire                             cfg_input_is_int8,
    input  wire [ADDR_WIDTH-1:0]            cfg_src_base_addr,
    input  wire [STRIDE_WIDTH-1:0]          cfg_src_row_stride,
    input  wire [ROW_COUNT_WIDTH-1:0]       cfg_row_num_m,

    output wire                             mem_rd_cmd_valid,
    input  wire                             mem_rd_cmd_ready,
    output wire [ADDR_WIDTH-1:0]            mem_rd_cmd_addr,
    output wire [DMA_LEN_WIDTH-1:0]         mem_rd_cmd_len,

    input  wire                             mem_rd_data_valid,
    output wire                             mem_rd_data_ready,
    input  wire [DMA_DATA_WIDTH-1:0]        mem_rd_data,
    input  wire                             mem_rd_data_last,

    output wire [(IDMA_INT8_NUM*8)-1:0]     idma_data,
    output wire [31:0]                      idma_scale,
    output wire                             idma_dequant_en,
    output wire                             idma_valid,
    input  wire                             idma_ready
);
    import bnn_fp_math::*;
    localparam int unsigned BEAT_BYTES = DMA_DATA_WIDTH / 8;
    localparam int unsigned FP32_BEATS = (IDMA_INT8_NUM * 4 + BEAT_BYTES - 1) / BEAT_BYTES;
    localparam int unsigned INT8_BEATS = (IDMA_INT8_NUM + BEAT_BYTES - 1) / BEAT_BYTES;
    typedef enum logic [2:0] {IDLE, COMMAND, RECEIVE, SCAN, SCALE, QUANTIZE, OUTPUT_ROW} state_t;
    state_t state;
    logic input_int8;
    logic [ADDR_WIDTH-1:0] row_address;
    logic [ADDR_WIDTH-1:0] row_stride;
    logic [ROW_COUNT_WIDTH-1:0] rows_left;
    logic [FP32_BEATS*DMA_DATA_WIDTH-1:0] row_storage;
    logic [IDMA_INT8_NUM*8-1:0] quantized_row;
    logic [31:0] maximum, row_scale;
    int unsigned beat_index, sample_index;
    logic done_reg;
    wire [31:0] sample = row_storage[sample_index*32 +: 32];
    wire finite_normal = sample[30:23] != 0 && sample[30:23] != 8'hff;
    wire [31:0] magnitude = {1'b0, sample[30:0]};
    wire [DMA_LEN_WIDTH-1:0] row_beats = input_int8 ? DMA_LEN_WIDTH'(INT8_BEATS) : DMA_LEN_WIDTH'(FP32_BEATS);

    assign busy = !reset && (state != IDLE);
    assign done = done_reg && !reset;
    assign mem_rd_cmd_valid = state == COMMAND && !reset;
    assign mem_rd_cmd_addr = row_address;
    assign mem_rd_cmd_len = row_beats;
    assign mem_rd_data_ready = state == RECEIVE && !reset;
    assign idma_valid = state == OUTPUT_ROW && !reset;
    assign idma_data = input_int8 ? row_storage[IDMA_INT8_NUM*8-1:0] : quantized_row;
    assign idma_scale = row_scale;
    assign idma_dequant_en = !input_int8;

    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            input_int8 <= 0;
            row_address <= 0;
            row_stride <= 0;
            rows_left <= 0;
            maximum <= 0;
            row_scale <= 0;
            beat_index <= 0;
            sample_index <= 0;
            done_reg <= 0;
        end else begin
            done_reg <= 0;
            case (state)
                IDLE: if (start) begin
                    input_int8 <= cfg_input_is_int8;
                    row_address <= cfg_src_base_addr;
                    row_stride <= cfg_src_row_stride != 0 ? ADDR_WIDTH'(cfg_src_row_stride) :
                        ADDR_WIDTH'((cfg_input_is_int8 ? INT8_BEATS : FP32_BEATS) * BEAT_BYTES);
                    rows_left <= cfg_row_num_m;
                    row_scale <= 0;
                    maximum <= 0;
                    beat_index <= 0;
                    state <= COMMAND;
                end
                COMMAND: if (mem_rd_cmd_ready) begin
                    beat_index <= 0;
                    state <= RECEIVE;
                end
                RECEIVE: if (mem_rd_data_valid) begin
                    row_storage[beat_index*DMA_DATA_WIDTH +: DMA_DATA_WIDTH] <= mem_rd_data;
                    if (beat_index + 1 == int'(row_beats)) begin
                        sample_index <= 0;
                        maximum <= 0;
                        state <= input_int8 ? OUTPUT_ROW : SCAN;
                    end else begin
                        beat_index <= beat_index + 1;
                    end
                end
                SCAN: begin
                    if (finite_normal && magnitude > maximum) maximum <= magnitude;
                    if (sample_index + 1 == IDMA_INT8_NUM) begin
                        state <= SCALE;
                        sample_index <= 0;
                    end else sample_index <= sample_index + 1;
                end
                SCALE: begin
                    row_scale <= fp32_div127_ftz(maximum);
                    if (maximum == 0) begin
                        quantized_row <= '0;
                        state <= OUTPUT_ROW;
                    end else state <= QUANTIZE;
                end
                QUANTIZE: begin
                    quantized_row[sample_index*8 +: 8] <= quantize_int8_ftz(sample, maximum);
                    if (sample_index + 1 == IDMA_INT8_NUM) begin
                        state <= OUTPUT_ROW;
                        sample_index <= 0;
                    end else sample_index <= sample_index + 1;
                end
                OUTPUT_ROW: if (idma_ready) begin
                    if (rows_left == 1) begin
                        done_reg <= 1;
                        state <= IDLE;
                    end else begin
                        rows_left <= rows_left - 1'b1;
                        row_address <= row_address + row_stride;
                        beat_index <= 0;
                        state <= COMMAND;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (DMA_DATA_WIDTH >= 8 && DMA_DATA_WIDTH % 8 == 0 && IDMA_INT8_NUM > 0)
            else $fatal(1, "invalid input buffer data widths");
        assert (64'(FP32_BEATS) < (64'd1 << DMA_LEN_WIDTH))
            else $fatal(1, "input row exceeds DMA length field");
    end
    always @(posedge clock) if (!reset) begin
        if (start && !busy) begin
            assert (cfg_row_num_m > 0 && int'(cfg_row_num_m) <= MAX_ROWS)
                else $fatal(1, "invalid input row count");
            assert (64'(cfg_src_row_stride) == 64'(ADDR_WIDTH'(cfg_src_row_stride)))
                else $fatal(1, "input row stride exceeds ADDR_WIDTH");
        end
        if (mem_rd_data_valid && mem_rd_data_ready)
            assert (mem_rd_data_last == (beat_index + 1 == int'(row_beats)))
                else $fatal(1, "input DMA last mismatch");
    end
    assert property (@(posedge clock) disable iff (reset)
        idma_valid && !idma_ready |=> idma_valid && $stable({idma_data, idma_scale, idma_dequant_en}))
        else $fatal(1, "input buffer output changed under backpressure");
`else
    wire _unused_ok = &{1'b0, mem_rd_data_last};
`endif
endmodule

`default_nettype wire
