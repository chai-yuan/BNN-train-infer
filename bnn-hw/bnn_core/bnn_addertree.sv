`timescale 1ns / 1ps

// ============================================================================
// Module: bnn_addertree
// Description:
//   P 路并行有符号点积加法树 (2 级流水线设计)。
//   Stage 0: 32 路 W1*A8 条件求补 (W ? +A : -A) 并进行 32 -> 16 -> 8 级求和
//   Stage 1: 流水寄存器打拍，进行 8 -> 4 -> 2 -> 1 级求和，输出 16-bit 局部点积和
//   具备完备的流水线反压机制 (Stall-able Pipeline)
// ============================================================================

module bnn_addertree #(
    parameter int P_LANES    = 32,  // 物理通道数 (固定为 32)
    parameter int ACT_WIDTH  = 8,   // 激活精度 int8
    parameter int PSUM_WIDTH = 16   // 物理局部和精度
)(
    input  logic                                clock,
    input  logic                                reset,
    input  logic                                flush,

    // 输入流
    input  logic                                in_valid,
    output logic                                in_ready,
    input  logic [P_LANES*ACT_WIDTH-1:0]        in_act_data,
    input  logic [P_LANES-1:0]                  in_weight_data,

    // 伴随流水线传递的控制元数据
    input  logic                                in_is_group_last,
    input  logic                                in_is_chan_last,
    input  logic                                in_is_all_last,
    input  logic [7:0]                          in_group_idx,

    // 输出流与局部和
    output logic                                out_valid,
    input  logic                                out_ready,
    output logic signed [PSUM_WIDTH-1:0]        out_psum,
    output logic                                out_is_group_last,
    output logic                                out_is_chan_last,
    output logic                                out_is_all_last,
    output logic [7:0]                          out_group_idx
);

    initial begin
        if (P_LANES != 32 || ACT_WIDTH != 8 || PSUM_WIDTH != 16)
            $fatal(1, "bnn_addertree supports only P32/A8/PSUM16");
    end

    // -------------------------------------------------------------
    // Stage 0 组合逻辑：条件取反与前两级加法
    // -------------------------------------------------------------
    logic signed [8:0]  term [P_LANES];      // 9-bit: 支持 -128 ~ +128
    logic signed [9:0]  sum_lvl1 [16];      // 32 -> 16
    logic signed [10:0] sum_lvl2 [8];       // 16 -> 8

    always_comb begin
        for (int i = 0; i < P_LANES; i++) begin
            logic signed [ACT_WIDTH-1:0] a;
            a = in_act_data[i*ACT_WIDTH +: ACT_WIDTH];
            term[i] = in_weight_data[i] ? $signed({a[ACT_WIDTH-1], a})
                                        : -$signed({a[ACT_WIDTH-1], a});
        end

        for (int i = 0; i < 16; i++) begin
            sum_lvl1[i] = $signed(term[2*i]) + $signed(term[2*i+1]);
        end

        for (int i = 0; i < 8; i++) begin
            sum_lvl2[i] = $signed(sum_lvl1[2*i]) + $signed(sum_lvl1[2*i+1]);
        end
    end

    // -------------------------------------------------------------
    // Stage 1 流水线寄存器
    // -------------------------------------------------------------
    logic signed [10:0] s1_sum [8];
    logic               s1_valid;
    logic               s1_is_group_last;
    logic               s1_is_chan_last;
    logic               s1_is_all_last;
    logic [7:0]         s1_group_idx;

    // Stage 1 反压使能信号
    logic s1_ready;
    logic s2_ready;

    assign s2_ready = out_ready || !out_valid;
    assign s1_ready = s2_ready  || !s1_valid;
    assign in_ready = s1_ready;

    always_ff @(posedge clock) begin
        if (reset || flush) begin
            s1_valid          <= 1'b0;
            s1_is_group_last  <= 1'b0;
            s1_is_chan_last   <= 1'b0;
            s1_is_all_last    <= 1'b0;
            s1_group_idx      <= '0;
            for (int i = 0; i < 8; i++) begin
                s1_sum[i] <= '0;
            end
        end else if (s1_ready) begin
            s1_valid          <= in_valid;
            s1_is_group_last  <= in_is_group_last;
            s1_is_chan_last   <= in_is_chan_last;
            s1_is_all_last    <= in_is_all_last;
            s1_group_idx      <= in_group_idx;
            s1_sum            <= sum_lvl2;
        end
    end

    // -------------------------------------------------------------
    // Stage 1 -> Stage 2 组合逻辑：后三级加法归约
    // -------------------------------------------------------------
    logic signed [11:0] sum_lvl3 [4];       // 8 -> 4
    logic signed [12:0] sum_lvl4 [2];       // 4 -> 2
    logic signed [13:0] sum_lvl5;           // 2 -> 1 (最大值 4096)

    always_comb begin
        for (int i = 0; i < 4; i++) begin
            sum_lvl3[i] = $signed(s1_sum[2*i]) + $signed(s1_sum[2*i+1]);
        end

        for (int i = 0; i < 2; i++) begin
            sum_lvl4[i] = $signed(sum_lvl3[2*i]) + $signed(sum_lvl3[2*i+1]);
        end

        sum_lvl5 = $signed(sum_lvl4[0]) + $signed(sum_lvl4[1]);
    end

    // -------------------------------------------------------------
    // Stage 2 (输出寄存器)
    // -------------------------------------------------------------
    always_ff @(posedge clock) begin
        if (reset || flush) begin
            out_valid          <= 1'b0;
            out_psum           <= '0;
            out_is_group_last  <= 1'b0;
            out_is_chan_last   <= 1'b0;
            out_is_all_last    <= 1'b0;
            out_group_idx      <= '0;
        end else if (s2_ready) begin
            out_valid          <= s1_valid;
            out_psum           <= {{(PSUM_WIDTH-14){sum_lvl5[13]}}, sum_lvl5}; // 符号扩展到 16-bit
            out_is_group_last  <= s1_is_group_last;
            out_is_chan_last   <= s1_is_chan_last;
            out_is_all_last    <= s1_is_all_last;
            out_group_idx      <= s1_group_idx;
        end
    end

endmodule
