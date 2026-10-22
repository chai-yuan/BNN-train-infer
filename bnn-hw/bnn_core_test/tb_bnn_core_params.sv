`timescale 1ns/1ps

// Separate top for expected-failure runs. Each Makefile run changes exactly
// one parameter. All widths follow that parameter, avoiding port-size errors
// that could otherwise masquerade as a successful parameter rejection test.
module tb_bnn_core_params #(
    parameter int TEST_P_LANES = 32,
    parameter int TEST_ACT_WIDTH = 8,
    parameter int TEST_PSUM_WIDTH = 16,
    parameter int TEST_RES_WIDTH = 32
);
    logic clock = 0;
    always #5 clock = ~clock;

    bnn_core_top #(
        .P_LANES(TEST_P_LANES), .ACT_WIDTH(TEST_ACT_WIDTH),
        .PSUM_WIDTH(TEST_PSUM_WIDTH), .RES_WIDTH(TEST_RES_WIDTH)
    ) dut (
        .clock(clock), .reset(1'b1),
        .cmd_valid(1'b0), .cmd_ready(),
        .cmd_k_blocks(16'd0), .cmd_group_div(8'd0), .cmd_out_chans(16'd0),
        .status_idle(), .status_busy(),
        .in_valid(1'b0), .in_ready(),
        .act_data({(TEST_P_LANES*TEST_ACT_WIDTH){1'b0}}),
        .weight_data({TEST_P_LANES{1'b0}}),
        .res_valid(), .res_ready(1'b0), .res_data(), .res_last(),
        .res_group_idx(), .res_chan_last()
    );

    initial begin
        #20;
        $display("[UNEXPECTED] unsupported parameters survived: P=%0d A=%0d PSUM=%0d RES=%0d",
                 TEST_P_LANES, TEST_ACT_WIDTH, TEST_PSUM_WIDTH, TEST_RES_WIDTH);
        $finish;
    end
endmodule
