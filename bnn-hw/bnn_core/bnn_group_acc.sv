`timescale 1ns / 1ps

// 顺序累加完整逻辑组，组尾生成一个结果。
module bnn_group_acc #(
    parameter int PSUM_WIDTH = 16,
    parameter int RES_WIDTH  = 32
)(
    input  logic                         clock,
    input  logic                         reset,
    input  logic                         flush,
    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic signed [PSUM_WIDTH-1:0] in_psum,
    input  logic                         in_is_group_last,
    input  logic                         in_is_chan_last,
    input  logic                         in_is_all_last,
    input  logic [7:0]                   in_group_idx,
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic signed [RES_WIDTH-1:0]  out_res_data,
    output logic                         out_res_last,
    output logic                         out_res_chan_last,
    output logic [7:0]                   out_group_idx
);
    logic signed [RES_WIDTH-1:0] acc_reg;
    wire signed [RES_WIDTH-1:0] psum_ext = {{(RES_WIDTH-PSUM_WIDTH){in_psum[PSUM_WIDTH-1]}}, in_psum};
    wire signed [RES_WIDTH-1:0] new_acc = acc_reg + psum_ext;
    assign in_ready = out_ready || !out_valid;

    always_ff @(posedge clock) begin
        if (reset || flush) begin
            acc_reg           <= '0;
            out_valid         <= 1'b0;
            out_res_data      <= '0;
            out_res_last      <= 1'b0;
            out_res_chan_last <= 1'b0;
            out_group_idx     <= '0;
        end else begin
            if (out_ready && out_valid) out_valid <= 1'b0;
            if (in_valid && in_ready) begin
                if (in_is_group_last) begin
                    out_valid         <= 1'b1;
                    out_res_data      <= new_acc;
                    out_res_last      <= in_is_all_last;
                    out_res_chan_last <= in_is_chan_last;
                    out_group_idx     <= in_group_idx;
                    acc_reg           <= '0;
                end else begin
                    acc_reg <= new_acc;
                end
            end
        end
    end
endmodule
