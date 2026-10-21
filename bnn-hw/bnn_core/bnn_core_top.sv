`timescale 1ns / 1ps

// 与总线无关的单任务 DOT_GROUPS 核心；激活缓存和重放由外层负责。
module bnn_core_top #(
    parameter int P_LANES    = 32,
    parameter int ACT_WIDTH  = 8,
    parameter int PSUM_WIDTH = 16,
    parameter int RES_WIDTH  = 32
)(
    input  logic                           clock,
    input  logic                           reset,
    input  logic                           ctrl_abort,

    input  logic                           cmd_valid,
    output logic                           cmd_ready,
    input  logic [15:0]                    cmd_k_blocks,
    input  logic [7:0]                     cmd_group_div,
    input  logic [15:0]                    cmd_out_chans,
    output logic                           status_idle,
    output logic                           status_busy,
    output logic                           done_valid,
    input  logic                           done_ready,
    output logic [1:0]                     done_status,

    input  logic                           act_valid,
    output logic                           act_ready,
    input  logic [P_LANES*ACT_WIDTH-1:0]   act_data,
    input  logic                           act_last,
    input  logic                           weight_valid,
    output logic                           weight_ready,
    input  logic [P_LANES-1:0]             weight_data,
    input  logic                           weight_last,

    output logic                           res_valid,
    input  logic                           res_ready,
    output logic signed [RES_WIDTH-1:0]    res_data,
    output logic                           res_last,
    output logic                           res_chan_last,
    output logic [7:0]                     res_group_idx
);
    initial begin
        if (P_LANES != 32 || ACT_WIDTH != 8 || PSUM_WIDTH != 16 || RES_WIDTH != 32)
            $fatal(1, "bnn_core_top supports only P32/A8/PSUM16/RES32");
    end

    logic pipe_ready, pipe_valid, pipe_flush;
    logic pipe_group_last, pipe_chan_last, pipe_all_last;
    logic [7:0] pipe_group_idx;
    logic tree_valid, tree_ready;
    logic signed [PSUM_WIDTH-1:0] tree_psum;
    logic tree_group_last, tree_chan_last, tree_all_last;
    logic [7:0] tree_group_idx;
    logic acc_valid, acc_ready, acc_last, acc_chan_last;
    logic signed [RES_WIDTH-1:0] acc_data;
    logic [7:0] acc_group_idx;
    logic slice_valid, slice_ready;
    logic task_drain_done;

    // 取消周期禁止对外提交结果；各级有效位在同一时钟沿被清空。
    assign res_valid = slice_valid && !reset && !pipe_flush;
    assign slice_ready = res_ready && !reset && !pipe_flush;
    assign task_drain_done = res_valid && res_ready && res_last;

    bnn_ctrl_fsm u_ctrl_fsm (
        .clock(clock), .reset(reset), .ctrl_abort(ctrl_abort),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_k_blocks(cmd_k_blocks), .cmd_group_div(cmd_group_div),
        .cmd_out_chans(cmd_out_chans),
        .status_idle(status_idle), .status_busy(status_busy),
        .done_valid(done_valid), .done_ready(done_ready), .done_status(done_status),
        .act_valid(act_valid), .act_ready(act_ready), .act_last(act_last),
        .weight_valid(weight_valid), .weight_ready(weight_ready), .weight_last(weight_last),
        .pipe_ready(pipe_ready), .pipe_valid(pipe_valid),
        .pipe_is_group_last(pipe_group_last), .pipe_is_chan_last(pipe_chan_last),
        .pipe_is_all_last(pipe_all_last), .pipe_group_idx(pipe_group_idx),
        .pipe_flush(pipe_flush), .task_drain_done(task_drain_done)
    );

    bnn_addertree #(
        .P_LANES(P_LANES), .ACT_WIDTH(ACT_WIDTH), .PSUM_WIDTH(PSUM_WIDTH)
    ) u_addertree (
        .clock(clock), .reset(reset), .flush(pipe_flush),
        .in_valid(pipe_valid), .in_ready(pipe_ready),
        .in_act_data(act_data), .in_weight_data(weight_data),
        .in_is_group_last(pipe_group_last), .in_is_chan_last(pipe_chan_last),
        .in_is_all_last(pipe_all_last), .in_group_idx(pipe_group_idx),
        .out_valid(tree_valid), .out_ready(tree_ready), .out_psum(tree_psum),
        .out_is_group_last(tree_group_last), .out_is_chan_last(tree_chan_last),
        .out_is_all_last(tree_all_last), .out_group_idx(tree_group_idx)
    );

    bnn_group_acc #(.PSUM_WIDTH(PSUM_WIDTH), .RES_WIDTH(RES_WIDTH)) u_group_acc (
        .clock(clock), .reset(reset), .flush(pipe_flush),
        .in_valid(tree_valid), .in_ready(tree_ready), .in_psum(tree_psum),
        .in_is_group_last(tree_group_last), .in_is_chan_last(tree_chan_last),
        .in_is_all_last(tree_all_last), .in_group_idx(tree_group_idx),
        .out_valid(acc_valid), .out_ready(acc_ready), .out_res_data(acc_data),
        .out_res_last(acc_last), .out_res_chan_last(acc_chan_last), .out_group_idx(acc_group_idx)
    );

    bnn_out_slice #(.RES_WIDTH(RES_WIDTH)) u_out_slice (
        .clock(clock), .reset(reset), .flush(pipe_flush),
        .in_valid(acc_valid), .in_ready(acc_ready), .in_data(acc_data),
        .in_last(acc_last), .in_chan_last(acc_chan_last), .in_group_idx(acc_group_idx),
        .out_valid(slice_valid), .out_ready(slice_ready), .out_data(res_data),
        .out_last(res_last), .out_chan_last(res_chan_last), .out_group_idx(res_group_idx)
    );
endmodule
