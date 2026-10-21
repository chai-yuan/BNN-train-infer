`timescale 1ns / 1ps

// 单在途任务控制。所有计数仅在成对输入握手时推进。
module bnn_ctrl_fsm (
    input  logic        clock,
    input  logic        reset,
    input  logic        ctrl_abort,

    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [15:0] cmd_k_blocks,
    input  logic [7:0]  cmd_group_div,
    input  logic [15:0] cmd_out_chans,

    output logic        status_idle,
    output logic        status_busy,
    output logic        done_valid,
    input  logic        done_ready,
    output logic [1:0]  done_status,

    input  logic        act_valid,
    output logic        act_ready,
    input  logic        act_last,
    input  logic        weight_valid,
    output logic        weight_ready,
    input  logic        weight_last,

    input  logic        pipe_ready,
    output logic        pipe_valid,
    output logic        pipe_is_group_last,
    output logic        pipe_is_chan_last,
    output logic        pipe_is_all_last,
    output logic [7:0]  pipe_group_idx,
    output logic        pipe_flush,
    input  logic        task_drain_done
);
    localparam logic [1:0] DONE_OK       = 2'd0;
    localparam logic [1:0] DONE_CONFIG   = 2'd1;
    localparam logic [1:0] DONE_ABORTED  = 2'd2;
    localparam logic [1:0] DONE_PROTOCOL = 2'd3;

    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DRAIN, S_DONE} state_t;
    state_t state;
    logic [15:0] k_blocks_q, out_chans_q;
    logic [7:0] group_div_q;
    logic [15:0] k_cnt, ch_cnt;
    logic [7:0] div_cnt, grp_cnt;
    logic config_valid;
    logic running, pair_fire, boundary_error, protocol_error;
    logic is_group_end, is_chan_end, is_all_end;

    always_comb begin
        config_valid = 1'b0;
        if ((cmd_k_blocks != 0) && (cmd_group_div != 0) && (cmd_out_chans != 0)) begin
            config_valid = ((cmd_k_blocks % {8'b0, cmd_group_div}) == 0)
                         && (cmd_k_blocks <= {cmd_group_div, 8'b0});
        end
    end

    assign status_idle = (state == S_IDLE);
    assign status_busy = (state == S_RUN) || (state == S_DRAIN);
    assign cmd_ready   = status_idle && !reset && !ctrl_abort;
    assign done_valid  = (state == S_DONE) && !reset;
    assign running    = (state == S_RUN) && !reset && !ctrl_abort;

    assign is_group_end = (div_cnt == group_div_q - 8'd1);
    assign is_chan_end  = (k_cnt == k_blocks_q - 16'd1);
    assign is_all_end   = is_chan_end && (ch_cnt == out_chans_q - 16'd1);
    assign boundary_error = (act_last != is_chan_end) || (weight_last != is_chan_end);

    assign act_ready    = running && pipe_ready && weight_valid;
    assign weight_ready = running && pipe_ready && act_valid;
    assign pair_fire    = running && act_valid && weight_valid && pipe_ready;
    assign protocol_error = pair_fire && boundary_error;
    // 错误拍仍由输入握手消费，但不进入计算流水线。
    assign pipe_valid   = running && act_valid && weight_valid && !boundary_error;
    assign pipe_flush   = (ctrl_abort && status_busy) || protocol_error;
    assign pipe_is_group_last = is_group_end;
    assign pipe_is_chan_last  = is_chan_end;
    assign pipe_is_all_last   = is_all_end;
    assign pipe_group_idx     = grp_cnt;

    always_ff @(posedge clock) begin
        if (reset) begin
            state       <= S_IDLE;
            done_status <= DONE_OK;
            k_blocks_q  <= '0;
            group_div_q <= '0;
            out_chans_q <= '0;
            k_cnt       <= '0;
            div_cnt     <= '0;
            grp_cnt     <= '0;
            ch_cnt      <= '0;
        end else if (pipe_flush) begin
            state       <= S_DONE;
            done_status <= ctrl_abort ? DONE_ABORTED : DONE_PROTOCOL;
            k_cnt       <= '0;
            div_cnt     <= '0;
            grp_cnt     <= '0;
            ch_cnt      <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        k_blocks_q  <= cmd_k_blocks;
                        group_div_q <= cmd_group_div;
                        out_chans_q <= cmd_out_chans;
                        k_cnt       <= '0;
                        div_cnt     <= '0;
                        grp_cnt     <= '0;
                        ch_cnt      <= '0;
                        done_status <= config_valid ? DONE_OK : DONE_CONFIG;
                        state       <= config_valid ? S_RUN : S_DONE;
                    end
                end
                S_RUN: begin
                    if (pair_fire) begin
                        if (is_group_end) begin
                            div_cnt <= '0;
                            grp_cnt <= grp_cnt + 8'd1;
                        end else begin
                            div_cnt <= div_cnt + 8'd1;
                        end
                        if (is_chan_end) begin
                            k_cnt   <= '0;
                            div_cnt <= '0;
                            grp_cnt <= '0;
                            ch_cnt  <= ch_cnt + 16'd1;
                        end else begin
                            k_cnt <= k_cnt + 16'd1;
                        end
                        if (is_all_end) state <= S_DRAIN;
                    end
                end
                S_DRAIN: begin
                    if (task_drain_done) begin
                        done_status <= DONE_OK;
                        state       <= S_DONE;
                    end
                end
                S_DONE: begin
                    if (done_ready) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
