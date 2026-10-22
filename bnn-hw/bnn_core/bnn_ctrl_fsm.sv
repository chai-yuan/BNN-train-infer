`timescale 1ns / 1ps

// 单在途任务控制；命令握手启动，输入握手计数，末结果握手结束。
module bnn_ctrl_fsm (
    input  logic        clock,
    input  logic        reset,

    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [15:0] cmd_k_blocks,
    input  logic [7:0]  cmd_group_div,
    input  logic [15:0] cmd_out_chans,

    output logic        status_idle,
    output logic        status_busy,

    input  logic        in_valid,
    output logic        in_ready,

    input  logic        pipe_ready,
    output logic        pipe_valid,
    output logic        pipe_is_group_last,
    output logic        pipe_is_chan_last,
    output logic        pipe_is_all_last,
    output logic [7:0]  pipe_group_idx,
    input  logic        task_drain_done
);
    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DRAIN} state_t;
    state_t state;
    logic [15:0] k_blocks_q, out_chans_q;
    logic [7:0] group_div_q;
    logic [15:0] k_cnt, ch_cnt;
    logic [7:0] div_cnt, grp_cnt;
    logic running, in_fire;
    logic is_group_end, is_chan_end, is_all_end;

    assign status_idle = (state == S_IDLE);
    assign status_busy = (state == S_RUN) || (state == S_DRAIN);
    assign cmd_ready   = status_idle && !reset;
    assign running     = (state == S_RUN) && !reset;

    assign is_group_end = (div_cnt == group_div_q - 8'd1);
    assign is_chan_end  = (k_cnt == k_blocks_q - 16'd1);
    assign is_all_end   = is_chan_end && (ch_cnt == out_chans_q - 16'd1);

    assign in_ready   = running && pipe_ready;
    assign in_fire    = in_valid && in_ready;
    assign pipe_valid = running && in_valid;
    assign pipe_is_group_last = is_group_end;
    assign pipe_is_chan_last  = is_chan_end;
    assign pipe_is_all_last   = is_all_end;
    assign pipe_group_idx     = grp_cnt;

    // 配置合法性由外层保证；仅在实际接收命令时验证，不生成运行时校验电路。
`ifndef SYNTHESIS
    always @(posedge clock) begin
        if (!reset && cmd_valid && cmd_ready) begin
            assert (cmd_k_blocks != 0)
                else $fatal(1, "cmd_k_blocks must be nonzero");
            assert (cmd_group_div != 0)
                else $fatal(1, "cmd_group_div must be nonzero");
            assert (cmd_out_chans != 0)
                else $fatal(1, "cmd_out_chans must be nonzero");
            if (cmd_group_div != 0) begin
                assert ((cmd_k_blocks % {8'b0, cmd_group_div}) == 0)
                    else $fatal(1, "cmd_k_blocks must be divisible by cmd_group_div");
                assert (cmd_k_blocks <= {cmd_group_div, 8'b0})
                    else $fatal(1, "command must contain at most 256 groups");
            end
        end
    end
`endif

    always_ff @(posedge clock) begin
        if (reset) begin
            state       <= S_IDLE;
            k_blocks_q  <= '0;
            group_div_q <= '0;
            out_chans_q <= '0;
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
                        state       <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (in_fire) begin
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
                    if (task_drain_done) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
