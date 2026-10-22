`timescale 1ns / 1ps

module bnn_core_top #(
    parameter int P_LANES    = 32,
    parameter int ACT_WIDTH  = 8,
    parameter int PSUM_WIDTH = 16,
    parameter int RES_WIDTH  = 32
)(
    input  logic                           clock,
    input  logic                           reset,

    // cmd_valid && cmd_ready 锁存配置并启动一次任务；阻塞时须保持 valid 和配置。
    // 首对输入最早在下一拍接收；任务末结果交付后才能接收下一命令。
    input  logic                           cmd_valid,
    output logic                           cmd_ready,
    input  logic [15:0]                    cmd_k_blocks,  // 每输出通道的 P_LANES 元素块数。
    input  logic [7:0]                     cmd_group_div, // 每结果组的块数；须非零且整除 k_blocks。
    input  logic [15:0]                    cmd_out_chans, // 输出通道数；三个尺寸均非零，组数 <= 256。
    output logic                           status_idle,   // 空闲状态；能否接收命令以 cmd_ready 为准。
    output logic                           status_busy,   // 正在接收输入或等待最后一个结果交付。

    // in_valid && in_ready 同时消费一对操作数；阻塞时须保持 valid 和两路数据。
    // 按 [channel][k_block] 供数，每通道重放激活；外层完成两路配对和顺序保证。
    input  logic                           in_valid,
    output logic                           in_ready,
    input  logic [P_LANES*ACT_WIDTH-1:0]   act_data,      // lane i: [i*ACT_WIDTH +: ACT_WIDTH]，有符号 int8。
    input  logic [P_LANES-1:0]             weight_data,   // lane i: 位 1 表示 +1，位 0 表示 -1。

    // 结果按 [channel][group] 输出；除 reset 外，阻塞时 valid 和负载保持稳定。
    output logic                           res_valid,
    input  logic                           res_ready,
    output logic signed [RES_WIDTH-1:0]    res_data,       // 当前组的有符号整数点积。
    output logic                           res_last,       // 当前结果为任务末结果；与 valid/ready 握手即完成。
    output logic                           res_chan_last,  // 当前结果为该输出通道的最后一组。
    output logic [7:0]                     res_group_idx   // 当前通道内的组编号，从 0 开始。
);
    initial begin
        if (P_LANES != 32 || ACT_WIDTH != 8 || PSUM_WIDTH != 16 || RES_WIDTH != 32)
            $fatal(1, "bnn_core_top supports only P32/A8/PSUM16/RES32");
    end

    logic pipe_ready, pipe_valid;
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

    // reset 期间不提交旧结果；成功完成只取决于末结果被下游接收。
    assign res_valid = slice_valid && !reset;
    assign slice_ready = res_ready && !reset;
    assign task_drain_done = res_valid && res_ready && res_last;

    bnn_ctrl_fsm u_ctrl_fsm (
        .clock(clock), .reset(reset),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_k_blocks(cmd_k_blocks), .cmd_group_div(cmd_group_div),
        .cmd_out_chans(cmd_out_chans),
        .status_idle(status_idle), .status_busy(status_busy),
        .in_valid(in_valid), .in_ready(in_ready),
        .pipe_ready(pipe_ready), .pipe_valid(pipe_valid),
        .pipe_is_group_last(pipe_group_last), .pipe_is_chan_last(pipe_chan_last),
        .pipe_is_all_last(pipe_all_last), .pipe_group_idx(pipe_group_idx),
        .task_drain_done(task_drain_done)
    );

    bnn_addertree #(
        .P_LANES(P_LANES), .ACT_WIDTH(ACT_WIDTH), .PSUM_WIDTH(PSUM_WIDTH)
    ) u_addertree (
        .clock(clock), .reset(reset), .flush(1'b0),
        .in_valid(pipe_valid), .in_ready(pipe_ready),
        .in_act_data(act_data), .in_weight_data(weight_data),
        .in_is_group_last(pipe_group_last), .in_is_chan_last(pipe_chan_last),
        .in_is_all_last(pipe_all_last), .in_group_idx(pipe_group_idx),
        .out_valid(tree_valid), .out_ready(tree_ready), .out_psum(tree_psum),
        .out_is_group_last(tree_group_last), .out_is_chan_last(tree_chan_last),
        .out_is_all_last(tree_all_last), .out_group_idx(tree_group_idx)
    );

    bnn_group_acc #(.PSUM_WIDTH(PSUM_WIDTH), .RES_WIDTH(RES_WIDTH)) u_group_acc (
        .clock(clock), .reset(reset), .flush(1'b0),
        .in_valid(tree_valid), .in_ready(tree_ready), .in_psum(tree_psum),
        .in_is_group_last(tree_group_last), .in_is_chan_last(tree_chan_last),
        .in_is_all_last(tree_all_last), .in_group_idx(tree_group_idx),
        .out_valid(acc_valid), .out_ready(acc_ready), .out_res_data(acc_data),
        .out_res_last(acc_last), .out_res_chan_last(acc_chan_last), .out_group_idx(acc_group_idx)
    );

    bnn_out_slice #(.RES_WIDTH(RES_WIDTH)) u_out_slice (
        .clock(clock), .reset(reset), .flush(1'b0),
        .in_valid(acc_valid), .in_ready(acc_ready), .in_data(acc_data),
        .in_last(acc_last), .in_chan_last(acc_chan_last), .in_group_idx(acc_group_idx),
        .out_valid(slice_valid), .out_ready(slice_ready), .out_data(res_data),
        .out_last(res_last), .out_chan_last(res_chan_last), .out_group_idx(res_group_idx)
    );
endmodule
