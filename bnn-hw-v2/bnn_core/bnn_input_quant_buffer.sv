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
     d) 时序与面积优化：
        - RECEIVE 拍流水并行求取全精度 FP32 行最大绝对值，省去独立 128 拍 SCAN 遍历；
        - idma_scale 采用精确移位加恒等式替代 /127 组合除法器（零误差，两拍拆分）；
        - 行缓冲采用仅缓存 (sign + exp + 高 QUANT_FRAC_BITS 位尾数) 的移位寄存器，
          输出缓冲同样采用按字节右移寄存器，彻底消除大位宽 128:1 MUX 与 1:128 DEMUX；
        - QUANTIZE 采用 3 级流水、7 步恢复余数试商（商仅 7-bit 有效）；默认 QUANT_FRAC_BITS=12
          引入不超过 1 LSB 的极小截断误差以缩减约 34% 行缓冲寄存器与近半算术位宽（若需与参考
          模型逐比特 0 误差对齐，将 QUANT_FRAC_BITS 设为 23 即可）。
   - 输出 idma_dequant_en = 1'b1，通知后级 bnn_core_unit 开启反量化乘法。

2. 预量化 INT8 直通模式 (cfg_input_is_int8 == 1'b1)：
   - 每行直接读取 IDMA_INT8_NUM (128) 个有符号 INT8（占 8 拍 128-bit DMA 传输，128 Bytes；
     范围 [-127, +127] 由软件侧保证）。
   - 跳过动态量化逻辑，直接装入行缓冲；输出 idma_scale = 32'h0000_0000，idma_dequant_en = 1'b0，
     通知后级 bnn_core_unit 关闭反量化通路。
============================================================================
*/
module bnn_input_quant_buffer #(
    parameter int unsigned ADDR_WIDTH      = 32,
    parameter int unsigned STRIDE_WIDTH    = 32,
    parameter int unsigned DMA_DATA_WIDTH  = 128,
    parameter int unsigned IDMA_INT8_NUM   = 128,
    parameter int unsigned MAX_ROWS        = 128,
    parameter int unsigned ROW_COUNT_WIDTH = $clog2(MAX_ROWS + 1),
    parameter int unsigned DMA_LEN_WIDTH   = 16,
    parameter int unsigned QUANT_FRAC_BITS = 12
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
    localparam int unsigned BEAT_BYTES       = DMA_DATA_WIDTH / 8;
    localparam int unsigned FP32_PER_BEAT    = DMA_DATA_WIDTH / 32;
    localparam int unsigned FP32_BEATS       = (IDMA_INT8_NUM * 4 + BEAT_BYTES - 1) / BEAT_BYTES;
    localparam int unsigned INT8_BEATS       = (IDMA_INT8_NUM + BEAT_BYTES - 1) / BEAT_BYTES;

    localparam int unsigned BEAT_IDX_WIDTH   = (FP32_BEATS > 1) ? $clog2(FP32_BEATS) : 1;
    localparam int unsigned SAMPLE_IDX_WIDTH = (IDMA_INT8_NUM > 1) ? $clog2(IDMA_INT8_NUM) : 1;
    localparam logic [SAMPLE_IDX_WIDTH-1:0] LAST_SAMPLE_IDX = SAMPLE_IDX_WIDTH'(IDMA_INT8_NUM - 1);

    // FP32 压缩移位缓冲位宽：仅存 {sign(1), exp(8), frac_hi(QUANT_FRAC_BITS)}
    localparam int unsigned ENTRY_WIDTH      = 1 + 8 + QUANT_FRAC_BITS;
    localparam int unsigned BEAT_PACKED_BITS = FP32_PER_BEAT * ENTRY_WIDTH;
    localparam int unsigned FP32_BUF_BITS    = FP32_BEATS * BEAT_PACKED_BITS;

    // INT8 输出移位缓冲（兼容 INT8 直通接收与 FP32 逐字节右移装填）
    localparam int unsigned QUANT_ROW_BITS   = IDMA_INT8_NUM * 8;
    localparam int unsigned INT8_BUF_BITS    = INT8_BEATS * DMA_DATA_WIDTH;

    // 尾数定点乘除位宽（商严格落在 0..126，仅需 7 步恢复余数减法）
    localparam int unsigned M_WIDTH          = 1 + QUANT_FRAC_BITS;
    localparam int unsigned REM_WIDTH        = M_WIDTH + 1;
    localparam int unsigned NUM_WIDTH        = M_WIDTH + 7;

    // 单拍多路 FP32 最大值归约树参数
    localparam int unsigned MAX_TREE_DEPTH   = (FP32_PER_BEAT > 1) ? $clog2(FP32_PER_BEAT) : 0;
    localparam int unsigned MAX_TREE_LEAVES  = 1 << MAX_TREE_DEPTH;

    typedef enum logic [2:0] {
        IDLE,
        COMMAND,
        RECEIVE,
        MAX_DRAIN,
        SCALE,
        QUANTIZE,
        OUTPUT_ROW
    } state_t;

    // 单步无符号恢复余数试商：已知 rem_in < 2 * denom，商位与余数无溢出
    function automatic void div_step(
        input  logic [REM_WIDTH-1:0] rem_in,
        input  logic                 next_bit,
        input  logic [M_WIDTH-1:0]   denom,
        output logic                 q_bit,
        output logic [REM_WIDTH-1:0] rem_out
    );
        logic [M_WIDTH-1:0] sub;
        q_bit   = (rem_in >= {1'b0, denom});
        sub     = rem_in[M_WIDTH-1:0] - denom;
        rem_out = {q_bit ? sub : rem_in[M_WIDTH-1:0], next_bit};
    endfunction

    // 精确 /127 后半拍归一化：利用 2^35 - 1 = 127 * 35'h10204081，对 31-bit 被除数逐比特精确
    function automatic logic [31:0] finish_fp32_div127(
        input logic [30:0] max_mag,
        input logic [38:0] y_129
    );
        logic [30:0] x1;
        logic [59:0] scaled;
        logic [24:0] quotient;
        logic [22:0] fraction;
        int          exponent;
        x1       = {1'b1, max_mag[22:0], 7'd1};
        scaled   = (60'(y_129) << 21) + (60'(y_129) << 7) + 60'(x1);
        quotient = 25'(scaled >> 35);
        exponent = int'(max_mag[30:23]) - 7;
        if (quotient[24]) begin
            exponent = exponent + 1;
            fraction = quotient[23:1];
        end else begin
            fraction = quotient[22:0];
        end
        if (exponent <= 0)
            return 32'd0;
        return {1'b0, 8'(exponent), fraction};
    endfunction

    state_t                        state;
    logic                          input_int8;
    logic [ADDR_WIDTH-1:0]         row_address;
    logic [ADDR_WIDTH-1:0]         row_stride;
    logic [ROW_COUNT_WIDTH-1:0]    rows_left;
    logic [BEAT_IDX_WIDTH-1:0]     beat_index;
    logic [SAMPLE_IDX_WIDTH-1:0]   sample_index;
    logic                          done_reg;

    logic [FP32_BUF_BITS-1:0]      fp32_buf;
    logic [INT8_BUF_BITS-1:0]      int8_buf;

    // RECEIVE 拍流水求最大值寄存器（隔离外部 mem_rd_data 输入引脚组合负载）
    logic                          max_pipe_valid;
    logic [BEAT_IDX_WIDTH-1:0]     max_pipe_beat;
    logic [FP32_PER_BEAT*31-1:0]   max_pipe_mag;
    logic [30:0]                   maximum;

    // SCALE -> QUANTIZE 拆拍寄存器
    logic [38:0]                   scale_y_reg;
    logic                          scale_done_reg;
    logic [31:0]                   row_scale;
    logic [7:0]                    max_exp_reg;
    logic [M_WIDTH-1:0]            m_max_reg;

    // QUANTIZE 3 级流水寄存器（fp32_buf[0] -> S1(移位+q[6]) -> S2(q[5:3]) -> S3(q[2:0]+入队)）
    logic                          q_issue_done;
    logic                          q_s1_valid;
    logic                          q_s1_zero;
    logic                          q_s1_sat;
    logic                          q_s1_sign;
    logic                          q_s1_q6;
    logic [REM_WIDTH-1:0]          q_s1_rem;
    logic [4:0]                    q_s1_low_bits;

    logic                          q_s2_valid;
    logic                          q_s2_zero;
    logic                          q_s2_sat;
    logic                          q_s2_sign;
    logic [3:0]                    q_s2_q_hi;
    logic [REM_WIDTH-1:0]          q_s2_rem;
    logic [1:0]                    q_s2_low_bits;

    wire [DMA_LEN_WIDTH-1:0] row_beats = input_int8 ?
        DMA_LEN_WIDTH'(INT8_BEATS) : DMA_LEN_WIDTH'(FP32_BEATS);
    wire last_rx_beat = (DMA_LEN_WIDTH'(beat_index) + DMA_LEN_WIDTH'(1) == row_beats);
    wire [30:0] scale_x1 = {1'b1, maximum[22:0], 7'd1};

    // 将单拍 mem_rd_data 压缩为 FP32 移位缓冲切片并提取 31-bit 幅值（纯连线）
    logic [BEAT_PACKED_BITS-1:0]   packed_rx_beat;
    logic [FP32_PER_BEAT*31-1:0]   rx_beat_mag;
    always_comb begin
        logic [31:0] word;
        for (int unsigned lane = 0; lane < FP32_PER_BEAT; lane++) begin
            word = mem_rd_data[lane*32 +: 32];
            packed_rx_beat[lane*ENTRY_WIDTH +: ENTRY_WIDTH] = {
                word[31], word[30:23], word[22 -: QUANT_FRAC_BITS]
            };
            rx_beat_mag[lane*31 +: 31] = word[30:0];
        end
    end

    // 单拍多路 FP32 幅值二叉树归约求最大值
    logic [30:0] rx_local_max;
    always_comb begin
        logic [30:0] max_nodes [0 : 2*MAX_TREE_LEAVES - 2];
        logic [30:0] mag_k;
        int unsigned global_idx;
        logic        valid_k, normal_k;
        for (int unsigned k = 0; k < MAX_TREE_LEAVES; k++) begin
            if (k < FP32_PER_BEAT) begin
                mag_k      = max_pipe_mag[k*31 +: 31];
                global_idx = int'(max_pipe_beat) * FP32_PER_BEAT + k;
                valid_k    = (global_idx < IDMA_INT8_NUM);
                normal_k   = (mag_k[30:23] != 8'h00) && (mag_k[30:23] != 8'hff);
                max_nodes[MAX_TREE_LEAVES - 1 + k] = (valid_k && normal_k) ? mag_k : 31'd0;
            end else begin
                max_nodes[MAX_TREE_LEAVES - 1 + k] = 31'd0;
            end
        end
        for (int idx = int'(MAX_TREE_LEAVES) - 2; idx >= 0; idx--) begin
            max_nodes[idx] = (max_nodes[2*idx + 1] >= max_nodes[2*idx + 2]) ?
                max_nodes[2*idx + 1] : max_nodes[2*idx + 2];
        end
        rx_local_max = max_nodes[0];
    end

    // QUANTIZE Stage 1 组合逻辑：特殊值判别 + 指数差右移 + 第 1 步试商 (q[6])
    logic                   s1_zero_comb, s1_sat_comb, s1_sign_comb, s1_q6_comb;
    logic [REM_WIDTH-1:0]   s1_rem_comb;
    logic [4:0]             s1_low_comb;
    always_comb begin
        logic [ENTRY_WIDTH-1:0]     head_entry;
        logic                       cur_sign;
        logic [7:0]                 cur_exp, exp_diff;
        logic [QUANT_FRAC_BITS-1:0] cur_frac;
        logic                       zero_or_inf_nan, ge_max, underflow;
        logic [M_WIDTH-1:0]         m_val;
        logic [NUM_WIDTH-1:0]       num, num_shifted;

        head_entry      = fp32_buf[ENTRY_WIDTH-1:0];
        cur_sign        = head_entry[ENTRY_WIDTH-1];
        cur_exp         = head_entry[QUANT_FRAC_BITS +: 8];
        cur_frac        = head_entry[QUANT_FRAC_BITS-1:0];

        zero_or_inf_nan = (cur_exp == 8'h00) || (cur_exp == 8'hff);
        ge_max          = !zero_or_inf_nan &&
                          ({cur_exp, cur_frac} >= {max_exp_reg, m_max_reg[QUANT_FRAC_BITS-1:0]});
        exp_diff        = max_exp_reg - cur_exp;
        underflow       = !zero_or_inf_nan && !ge_max && (exp_diff >= 8'd8);

        m_val           = {1'b1, cur_frac};
        num             = ({7'b0, m_val} << 7) - {7'b0, m_val};
        num_shifted     = num >> exp_diff[2:0];

        div_step(num_shifted[NUM_WIDTH-1:6], num_shifted[5], m_max_reg, s1_q6_comb, s1_rem_comb);
        s1_zero_comb    = zero_or_inf_nan || underflow;
        s1_sat_comb     = ge_max;
        s1_sign_comb    = cur_sign;
        s1_low_comb     = num_shifted[4:0];
    end

    // QUANTIZE Stage 2 组合逻辑：第 2..4 步试商 (q[5:3])
    logic [2:0]             s2_q543_comb;
    logic [REM_WIDTH-1:0]   s2_rem_comb;
    always_comb begin
        logic [REM_WIDTH-1:0] r4, r3;
        div_step(q_s1_rem, q_s1_low_bits[4], m_max_reg, s2_q543_comb[2], r4);
        div_step(r4,       q_s1_low_bits[3], m_max_reg, s2_q543_comb[1], r3);
        div_step(r3,       q_s1_low_bits[2], m_max_reg, s2_q543_comb[0], s2_rem_comb);
    end

    // QUANTIZE Stage 3 组合逻辑：第 5..7 步试商 (q[2:0]) + 有符号二补数封装
    logic [7:0]             s3_q_byte_comb;
    always_comb begin
        logic [REM_WIDTH-1:0] r1, r0;
        logic [2:0]           q210;
        logic [7:0]           mag;
        div_step(q_s2_rem, q_s2_low_bits[1], m_max_reg, q210[2], r1);
        div_step(r1,       q_s2_low_bits[0], m_max_reg, q210[1], r0);
        q210[0] = (r0 >= {1'b0, m_max_reg});
        if (q_s2_zero)
            mag = 8'd0;
        else if (q_s2_sat)
            mag = 8'd127;
        else
            mag = {1'b0, q_s2_q_hi, q210};
        s3_q_byte_comb = q_s2_sign ? (8'd0 - mag) : mag;
    end

    assign busy              = !reset && (state != IDLE);
    assign done              = done_reg && !reset;
    assign mem_rd_cmd_valid  = (state == COMMAND) && !reset;
    assign mem_rd_cmd_addr   = row_address;
    assign mem_rd_cmd_len    = row_beats;
    assign mem_rd_data_ready = (state == RECEIVE) && !reset;
    assign idma_valid        = (state == OUTPUT_ROW) && !reset;
    assign idma_data         = int8_buf[QUANT_ROW_BITS-1:0];
    assign idma_scale        = row_scale;
    assign idma_dequant_en   = !input_int8;

    always_ff @(posedge clock) begin
        if (reset) begin
            state          <= IDLE;
            input_int8     <= 1'b0;
            row_address    <= '0;
            row_stride     <= '0;
            rows_left      <= '0;
            beat_index     <= '0;
            sample_index   <= '0;
            done_reg       <= 1'b0;
            max_pipe_valid <= 1'b0;
            max_pipe_beat  <= '0;
            maximum        <= '0;
            scale_y_reg    <= '0;
            scale_done_reg <= 1'b0;
            row_scale      <= '0;
            max_exp_reg    <= '0;
            m_max_reg      <= '0;
            q_issue_done   <= 1'b0;
            q_s1_valid     <= 1'b0;
            q_s2_valid     <= 1'b0;
        end else begin
            done_reg       <= 1'b0;
            max_pipe_valid <= (state == RECEIVE) && mem_rd_data_valid && !input_int8;

            if ((state == RECEIVE) && mem_rd_data_valid && !input_int8) begin
                max_pipe_beat <= beat_index;
                max_pipe_mag  <= rx_beat_mag;
            end
            if (max_pipe_valid && (rx_local_max > maximum)) begin
                maximum <= rx_local_max;
            end

            case (state)
                IDLE: if (start) begin
                    input_int8  <= cfg_input_is_int8;
                    row_address <= cfg_src_base_addr;
                    row_stride  <= (cfg_src_row_stride != '0) ? ADDR_WIDTH'(cfg_src_row_stride) :
                        ADDR_WIDTH'((cfg_input_is_int8 ? INT8_BEATS : FP32_BEATS) * BEAT_BYTES);
                    rows_left   <= cfg_row_num_m;
                    row_scale   <= 32'd0;
                    maximum     <= 31'd0;
                    beat_index  <= '0;
                    state       <= COMMAND;
                end

                COMMAND: if (mem_rd_cmd_ready) begin
                    beat_index <= '0;
                    maximum    <= 31'd0;
                    state      <= RECEIVE;
                end

                RECEIVE: if (mem_rd_data_valid) begin
                    if (input_int8) begin
                        int8_buf <= (int8_buf >> DMA_DATA_WIDTH) |
                            (INT8_BUF_BITS'(mem_rd_data) << (INT8_BUF_BITS - DMA_DATA_WIDTH));
                    end else begin
                        fp32_buf <= (fp32_buf >> BEAT_PACKED_BITS) |
                            (FP32_BUF_BITS'(packed_rx_beat) << (FP32_BUF_BITS - BEAT_PACKED_BITS));
                    end
                    if (last_rx_beat) begin
                        state <= input_int8 ? OUTPUT_ROW : MAX_DRAIN;
                    end else begin
                        beat_index <= beat_index + BEAT_IDX_WIDTH'(1);
                    end
                end

                MAX_DRAIN: begin
                    // 等待最后一拍 RECEIVE 数据完成流水最大值归约
                    state <= SCALE;
                end

                SCALE: begin
                    if (maximum == 31'd0) begin
                        row_scale                    <= 32'd0;
                        int8_buf[QUANT_ROW_BITS-1:0] <= '0;
                        state                        <= OUTPUT_ROW;
                    end else begin
                        scale_y_reg    <= {1'b0, scale_x1, 7'b0} + {8'b0, scale_x1};
                        scale_done_reg <= 1'b0;
                        max_exp_reg    <= maximum[30:23];
                        m_max_reg      <= {1'b1, maximum[22 -: QUANT_FRAC_BITS]};
                        sample_index   <= '0;
                        q_issue_done   <= 1'b0;
                        q_s1_valid     <= 1'b0;
                        q_s2_valid     <= 1'b0;
                        state          <= QUANTIZE;
                    end
                end

                QUANTIZE: begin
                    if (!scale_done_reg) begin
                        row_scale      <= finish_fp32_div127(maximum, scale_y_reg);
                        scale_done_reg <= 1'b1;
                    end

                    // Stage 1: 从移位缓冲头部顺序弹出并执行第 1 步试商
                    if (!q_issue_done) begin
                        fp32_buf      <= fp32_buf >> ENTRY_WIDTH;
                        q_s1_valid    <= 1'b1;
                        q_s1_zero     <= s1_zero_comb;
                        q_s1_sat      <= s1_sat_comb;
                        q_s1_sign     <= s1_sign_comb;
                        q_s1_q6       <= s1_q6_comb;
                        q_s1_rem      <= s1_rem_comb;
                        q_s1_low_bits <= s1_low_comb;
                        if (sample_index == LAST_SAMPLE_IDX)
                            q_issue_done <= 1'b1;
                        else
                            sample_index <= sample_index + SAMPLE_IDX_WIDTH'(1);
                    end else begin
                        q_s1_valid <= 1'b0;
                    end

                    // Stage 2: 执行中间 3 步试商
                    q_s2_valid <= q_s1_valid;
                    if (q_s1_valid) begin
                        q_s2_zero     <= q_s1_zero;
                        q_s2_sat      <= q_s1_sat;
                        q_s2_sign     <= q_s1_sign;
                        q_s2_q_hi     <= {q_s1_q6, s2_q543_comb};
                        q_s2_rem      <= s2_rem_comb;
                        q_s2_low_bits <= q_s1_low_bits[1:0];
                    end

                    // Stage 3: 执行末 3 步试商并按字节右移装入输出缓冲
                    if (q_s2_valid) begin
                        int8_buf[QUANT_ROW_BITS-1:0] <=
                            (int8_buf[QUANT_ROW_BITS-1:0] >> 8) |
                            (QUANT_ROW_BITS'(s3_q_byte_comb) << (QUANT_ROW_BITS - 8));
                    end

                    if (q_issue_done && !q_s1_valid && q_s2_valid) begin
                        state <= OUTPUT_ROW;
                    end
                end

                OUTPUT_ROW: if (idma_ready) begin
                    if (rows_left == ROW_COUNT_WIDTH'(1)) begin
                        done_reg <= 1'b1;
                        state    <= IDLE;
                    end else begin
                        rows_left   <= rows_left - ROW_COUNT_WIDTH'(1);
                        row_address <= row_address + row_stride;
                        beat_index  <= '0;
                        state       <= COMMAND;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (DMA_DATA_WIDTH >= 32 && DMA_DATA_WIDTH % 32 == 0 && IDMA_INT8_NUM > 0)
            else $fatal(1, "invalid input buffer data widths");
        assert (QUANT_FRAC_BITS >= 1 && QUANT_FRAC_BITS <= 23)
            else $fatal(1, "QUANT_FRAC_BITS must be in 1..23");
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
            assert (mem_rd_data_last == last_rx_beat)
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
