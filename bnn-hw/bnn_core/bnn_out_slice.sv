`timescale 1ns / 1ps

// ============================================================================
// Module: bnn_out_slice
// Description:
//   输出级弹性缓冲 (Register Slice / Skid Buffer)。
//   解耦内部计算流水线与外部消费端的反压 (res_ready)，切断组合反压通路，
//   支持全吞吐 (1 拍 1 数据，零死拍 bubble-free)。
// ============================================================================

module bnn_out_slice #(
    parameter int RES_WIDTH = 32
)(
    input  logic                                clock,
    input  logic                                reset,
    input  logic                                flush,

    // 上游输入接口 (来自组累加器)
    input  logic                                in_valid,
    output logic                                in_ready,
    input  logic signed [RES_WIDTH-1:0]         in_data,
    input  logic                                in_last,
    input  logic                                in_chan_last,
    input  logic [7:0]                          in_group_idx,

    // 下游输出接口 (直出顶层)
    output logic                                out_valid,
    input  logic                                out_ready,
    output logic signed [RES_WIDTH-1:0]         out_data,
    output logic                                out_last,
    output logic                                out_chan_last,
    output logic [7:0]                          out_group_idx
);

    // 主输出寄存器与暂存 (Skid) 寄存器
    logic signed [RES_WIDTH-1:0] reg_data,      skid_data;
    logic                        reg_last,      skid_last;
    logic                        reg_chan_last, skid_chan_last;
    logic [7:0]                  reg_group_idx, skid_group_idx;
    logic                        reg_valid,     skid_valid;

    // 当 skid 缓冲为空时，上游可继续灌入数据
    assign in_ready = ~skid_valid;

    // 下游负载仅来自主寄存器
    assign out_valid     = reg_valid;
    assign out_data      = reg_data;
    assign out_last      = reg_last;
    assign out_chan_last = reg_chan_last;
    assign out_group_idx = reg_group_idx;

    always_ff @(posedge clock) begin
        if (reset || flush) begin
            reg_valid      <= 1'b0;
            skid_valid     <= 1'b0;
            reg_data       <= '0;
            reg_last       <= 1'b0;
            reg_chan_last  <= 1'b0;
            reg_group_idx  <= '0;
            skid_data      <= '0;
            skid_last      <= 1'b0;
            skid_chan_last <= 1'b0;
            skid_group_idx <= '0;
        end else begin
            // 1. 如果下游成功接收了主寄存器数据
            if (out_ready && reg_valid) begin
                if (skid_valid) begin
                    // 将 skid 缓冲中的数据移入主寄存器
                    reg_valid      <= 1'b1;
                    reg_data       <= skid_data;
                    reg_last       <= skid_last;
                    reg_chan_last  <= skid_chan_last;
                    reg_group_idx  <= skid_group_idx;
                    skid_valid     <= 1'b0;
                end else if (in_valid && in_ready) begin
                    // skid 为空且上游有新数据，直接灌入主寄存器
                    reg_valid      <= 1'b1;
                    reg_data       <= in_data;
                    reg_last       <= in_last;
                    reg_chan_last  <= in_chan_last;
                    reg_group_idx  <= in_group_idx;
                end else begin
                    // 无新数据，主寄存器变为空闲
                    reg_valid      <= 1'b0;
                end
            end else if (!reg_valid) begin
                // 主寄存器当前为空
                if (in_valid) begin
                    reg_valid      <= 1'b1;
                    reg_data       <= in_data;
                    reg_last       <= in_last;
                    reg_chan_last  <= in_chan_last;
                    reg_group_idx  <= in_group_idx;
                end
            end else begin
                // 主寄存器有数据但下游未就绪 (out_ready == 0)
                if (in_valid && in_ready) begin
                    // 将上游新进来的数据放入 skid 缓冲暂存
                    skid_valid     <= 1'b1;
                    skid_data      <= in_data;
                    skid_last      <= in_last;
                    skid_chan_last <= in_chan_last;
                    skid_group_idx <= in_group_idx;
                end
            end
        end
    end

endmodule
