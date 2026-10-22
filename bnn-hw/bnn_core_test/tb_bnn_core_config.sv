`timescale 1ns/1ps

// Each invalid command runs in its own process. First prove the same values
// cause no assertion during reset, without valid, or while command is blocked.
// +probe=0 exits successfully after guards; probes 1..8 must assert at acceptance.
module tb_bnn_core_config;
    logic clock = 0;
    always #5 clock = ~clock;
    logic reset = 1;
    logic cmd_valid = 0;
    wire cmd_ready;
    logic [15:0] cmd_k_blocks = 0;
    logic [7:0] cmd_group_div = 0;
    logic [15:0] cmd_out_chans = 0;
    wire status_idle, status_busy;
    logic in_valid = 0;
    wire in_ready;
    wire res_valid;
    logic res_ready = 0;
    wire res_last;
    int probe = 0;
    int ticks = 0;

    bnn_core_top dut (
        .clock(clock), .reset(reset),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_k_blocks(cmd_k_blocks), .cmd_group_div(cmd_group_div),
        .cmd_out_chans(cmd_out_chans),
        .status_idle(status_idle), .status_busy(status_busy),
        .in_valid(in_valid), .in_ready(in_ready),
        .act_data(256'b0), .weight_data(32'b0),
        .res_valid(res_valid), .res_ready(res_ready), .res_data(),
        .res_last(res_last), .res_group_idx(), .res_chan_last()
    );

    task automatic require(input bit condition, input string message);
        assert (condition) else $fatal(1, "[PROBE FAIL] %s", message);
    endtask

    task automatic step();
        @(posedge clock);
        @(negedge clock);
        ticks++;
        require(ticks < 200, "watchdog");
    endtask

    task automatic configure(input int which);
        case (which)
            1: begin cmd_k_blocks = 0; cmd_group_div = 1; cmd_out_chans = 1; end
            2: begin cmd_k_blocks = 4; cmd_group_div = 0; cmd_out_chans = 1; end
            3: begin cmd_k_blocks = 4; cmd_group_div = 1; cmd_out_chans = 0; end
            4: begin cmd_k_blocks = 5; cmd_group_div = 2; cmd_out_chans = 1; end
            5: begin cmd_k_blocks = 2; cmd_group_div = 3; cmd_out_chans = 1; end
            6: begin cmd_k_blocks = 257; cmd_group_div = 1; cmd_out_chans = 1; end
            7: begin cmd_k_blocks = 514; cmd_group_div = 2; cmd_out_chans = 1; end
            8: begin cmd_k_blocks = 65535; cmd_group_div = 255; cmd_out_chans = 1; end
            default: $fatal(1, "unknown configuration probe %0d", which);
        endcase
    endtask

    initial begin
        if ($value$plusargs("probe=%d", probe)) begin end
        require(probe >= 0 && probe <= 8, "probe must be in 0..8");
        @(negedge clock);
        // All original invalid configurations are harmless under active reset,
        // even when command/input valid and result ready are simultaneously high.
        cmd_valid = 1; in_valid = 1; res_ready = 1;
        for (int i = 1; i <= 8; i++) begin
            configure(i);
            require(!cmd_ready && !in_ready && !res_valid, "reset handshake masking");
            step();
        end
        reset = 0; cmd_valid = 0; in_valid = 0; res_ready = 0;
        for (int i = 1; i <= 8; i++) begin
            configure(i);
            step();
            require(status_idle && !status_busy && cmd_ready && !in_ready && !res_valid,
                    "unaccepted invalid command changed idle state");
        end

        // Keep all illegal commands valid in RUN while no input is supplied.
        cmd_k_blocks = 1; cmd_group_div = 1; cmd_out_chans = 1; cmd_valid = 1;
        step();
        require(status_busy && !cmd_ready, "legal setup command not accepted");
        for (int i = 1; i <= 8; i++) begin
            configure(i);
            step();
            require(status_busy && !cmd_ready && in_ready && !res_valid,
                    "blocked invalid command changed RUN state");
        end
        in_valid = 1;
        step();
        in_valid = 0;
        require(status_busy && !cmd_ready && !in_ready, "last input did not enter DRAIN");
        for (int i = 1; i <= 8; i++) begin
            configure(i);
            step();
            require(status_busy && !cmd_ready && !in_ready,
                    "blocked invalid command changed DRAIN state");
        end
        require(res_valid && res_last, "legal setup task did not produce final result");
        // Illegal cmd_valid remains asserted on the completion edge, which still
        // cannot be a command handshake. Withdraw it before the next idle edge.
        res_ready = 1;
        step();
        cmd_valid = 0;
        require(status_idle && cmd_ready && !status_busy, "last result did not finish setup task");
        step();
        $display("[GUARDS PASS] all 8 illegal configurations ignored during reset, no-valid, RUN, DRAIN, and last-result edge");
        if (probe == 0) begin
            $display("[PASS] configuration assertion guards");
            $finish;
        end else begin
            configure(probe);
            cmd_valid = 1;
            require(cmd_ready && !reset, "failure probe must actually handshake");
            $display("[EXPECT ASSERT] probe=%0d k=%0d div=%0d n=%0d", probe,
                     cmd_k_blocks, cmd_group_div, cmd_out_chans);
            // An immediate handshake assertion must fail on this first edge.
            // A later failure is insufficient evidence of immediate validation.
            @(posedge clock);
            @(negedge clock);
            $fatal(1, "[UNEXPECTED] invalid command survived its acceptance edge");
        end
    end
endmodule
