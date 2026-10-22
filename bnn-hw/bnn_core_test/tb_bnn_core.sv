`timescale 1ns/1ps

// Black-box tests: only external handshakes drive the reference model.
// Sources assert valid independently of ready and hold both payloads together.
module tb_bnn_core;
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
    logic [255:0] act_data = 0;
    logic [31:0] weight_data = 0;
    wire res_valid;
    logic res_ready = 1;
    wire signed [31:0] res_data;
    wire res_last;
    wire [7:0] res_group_idx;
    wire res_chan_last;

    // Default parameters are part of the public contract.
    bnn_core_top dut (
        .clock(clock), .reset(reset),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_k_blocks(cmd_k_blocks), .cmd_group_div(cmd_group_div),
        .cmd_out_chans(cmd_out_chans),
        .status_idle(status_idle), .status_busy(status_busy),
        .in_valid(in_valid), .in_ready(in_ready),
        .act_data(act_data), .weight_data(weight_data),
        .res_valid(res_valid), .res_ready(res_ready),
        .res_data(res_data), .res_last(res_last),
        .res_group_idx(res_group_idx), .res_chan_last(res_chan_last)
    );

    typedef enum {M_IDLE, M_RUN, M_DRAIN} model_state_t;
    typedef struct packed {
        logic signed [31:0] data;
        logic [7:0] group_idx;
        logic chan_last;
        logic last;
    } result_t;
    model_state_t model_state = M_IDLE;
    result_t expected[$];
    int mk = 0, mg = 0, mn = 0;
    int inputs = 0, outputs = 0;
    longint signed group_sum = 0;
    int unsigned seed = 1, rng_state = 1;
    int random_cases = 16;
    int max_cycles = 500000;
    int cycles = 0, cases = 0;
    string case_name = "initial reset";

    // Sample before NBA updates; drivers inspect events at the next negedge.
    bit ev_cmd, ev_in, ev_res, ev_last;
    bit held_input = 0, held_res = 0;
    logic [255:0] held_act_data;
    logic [31:0] held_weight_data;
    result_t held_result;
    bit throughput_check = 0;
    int previous_input_cycle = -1, previous_output_cycle = -1;
    int commands = 0, completions = 0, good_results = 0;
    int input_stalls = 0, output_stalls = 0;
    int accepted_group_255 = 0, reset_run = 0, reset_drain = 0;
    int blocked_last_cycles = 0, immediate_restarts = 0;
    int last_completion_cycle = -2;
    int idle_input_cycles = 0, drain_input_cycles = 0;

    task automatic require(input bit condition, input string message);
        assert (condition) else
            $fatal(1, "[FAIL] seed=%0d cycle=%0d case='%s': %s",
                   seed, cycles, case_name, message);
    endtask

    function automatic int unsigned random_word();
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        return rng_state;
    endfunction

    task automatic sample();
        int lane_value;
        longint signed dot;
        result_t actual, item;
        cycles++;
        require(cycles <= max_cycles, "global cycle watchdog expired");
        ev_cmd = cmd_valid && cmd_ready;
        ev_in = in_valid && in_ready;
        ev_res = res_valid && res_ready;
        ev_last = ev_res && res_last;
        actual = '{data: res_data, group_idx: res_group_idx,
                   chan_last: res_chan_last, last: res_last};

        if (reset) begin
            require(!cmd_ready && !in_ready && !res_valid,
                    "reset must mask command/input/result handshakes, including dirty valid/ready");
            if (model_state == M_RUN) reset_run++;
            if (model_state == M_DRAIN) reset_drain++;
            model_state = M_IDLE;
            expected.delete();
            mk = 0; mg = 0; mn = 0;
            inputs = 0; outputs = 0; group_sum = 0;
            held_input = 0; held_res = 0;
            previous_input_cycle = -1; previous_output_cycle = -1;
            return;
        end

        require(status_idle === (model_state == M_IDLE), "status_idle disagrees with task state");
        require(status_busy === (model_state != M_IDLE), "status_busy must indicate RUN/DRAIN");
        require(cmd_ready === (model_state == M_IDLE), "cmd_ready must indicate idle && !reset");
        if (model_state != M_RUN)
            require(!in_ready && !ev_in, "input ready/acceptance outside RUN");
        if (model_state == M_IDLE && in_valid) idle_input_cycles++;
        if (model_state == M_DRAIN && in_valid) drain_input_cycles++;
        if (held_input)
            require(in_valid && act_data === held_act_data && weight_data === held_weight_data,
                    "source changed either common input payload while stalled");
        if (held_res)
            require(res_valid && actual === held_result, "result valid/payload changed while stalled");
        if (in_valid && !in_ready && model_state == M_RUN) input_stalls++;
        if (res_valid && !res_ready) output_stalls++;
        if (res_valid && res_last && !res_ready) begin
            require(model_state == M_DRAIN && !cmd_ready && !ev_cmd,
                    "blocked final result must retain DRAIN and reject a new command");
            blocked_last_cycles++;
        end
        if (ev_last)
            require(model_state == M_DRAIN && !ev_cmd && !cmd_ready,
                    "last result must complete from DRAIN without same-edge command acceptance");

        case (model_state)
            M_IDLE: begin
                require(!res_valid, "stale result visible in IDLE");
                if (ev_cmd) begin
                    commands++;
                    if (cycles == last_completion_cycle + 1) immediate_restarts++;
                    mk = int'(cmd_k_blocks);
                    mg = int'(cmd_group_div);
                    mn = int'(cmd_out_chans);
                    require(mk > 0 && mg > 0 && mn > 0, "main suite issued zero configuration");
                    require(mk % mg == 0 && mk <= mg * 256, "main suite issued illegal grouping");
                    inputs = 0; outputs = 0; group_sum = 0;
                    previous_input_cycle = -1; previous_output_cycle = -1;
                    expected.delete();
                    model_state = M_RUN;
                end
            end
            M_RUN, M_DRAIN: begin
                require(!ev_cmd, "busy core accepted a second command");
                if (ev_in) begin
                    require(inputs < mk * mn, "too many input beats accepted");
                    if (throughput_check && previous_input_cycle >= 0)
                        require(cycles == previous_input_cycle + 1,
                                "group_div=1 input throughput is below one beat per clock");
                    previous_input_cycle = cycles;
                    dot = 0;
                    for (int lane = 0; lane < 32; lane++) begin
                        // Widen before negation: -(-128) must become +128.
                        lane_value = int'($signed(act_data[lane*8 +: 8]));
                        if (weight_data[lane]) dot += longint'(lane_value);
                        else dot -= longint'(lane_value);
                    end
                    group_sum += dot;
                    if (inputs % mg == mg - 1) begin
                        require(group_sum >= -64'sd2147483648 && group_sum <= 64'sd2147483647,
                                "reference result exceeds signed int32");
                        item.data = 32'(group_sum);
                        item.group_idx = 8'((inputs % mk) / mg);
                        item.chan_last = inputs % mk == mk - 1;
                        item.last = inputs == mk * mn - 1;
                        expected.push_back(item);
                        group_sum = 0;
                    end
                    inputs++;
                    if (inputs == mk * mn) model_state = M_DRAIN;
                end
                if (res_valid) begin
                    require(expected.size() != 0, "unexpected/duplicate/unflushed result");
                    item = expected[0];
                    require(actual === item,
                            $sformatf("result[%0d]: got data=%0d group=%0d chan_last=%0b last=%0b; expected data=%0d group=%0d chan_last=%0b last=%0b",
                                      outputs, res_data, res_group_idx, res_chan_last, res_last,
                                      item.data, item.group_idx, item.chan_last, item.last));
                    if (ev_res) begin
                        item = expected.pop_front();
                        outputs++;
                        good_results++;
                        if (res_group_idx == 8'hff) accepted_group_255++;
                        if (throughput_check && previous_output_cycle >= 0)
                            require(cycles == previous_output_cycle + 1,
                                    "group_div=1 output throughput is below one result per clock");
                        previous_output_cycle = cycles;
                        if (item.last) begin
                            require(inputs == mk * mn && outputs == (mk / mg) * mn && expected.size() == 0,
                                    "success before all input/output handshakes");
                            completions++;
                            last_completion_cycle = cycles;
                            model_state = M_IDLE;
                        end
                    end
                end
            end
            default: require(0, "invalid reference state");
        endcase

        held_input = in_valid && !in_ready;
        held_res = res_valid && !res_ready;
        held_act_data = act_data;
        held_weight_data = weight_data;
        held_result = actual;
    endtask

    always @(posedge clock) sample();

    task automatic step();
        @(posedge clock);
        @(negedge clock);
    endtask

    task automatic announce(input string name);
        case_name = name;
        cases++;
        $display("[CASE %0d] %s (cycle %0d)", cases, name, cycles);
    endtask

    task automatic quiet();
        cmd_valid = 0; in_valid = 0; res_ready = 1;
    endtask

    task automatic issue(input int k, input int g, input int n);
        require(model_state == M_IDLE, "driver issued command before last result handshake");
        cmd_valid = 1;
        cmd_k_blocks = 16'(k); cmd_group_div = 8'(g); cmd_out_chans = 16'(n);
        step();
        require(ev_cmd && !ev_in, "idle command was not accepted exclusively");
        // Poison live configuration after acceptance to check latching. These
        // values are also illegal but must not assert without a cmd handshake.
        cmd_valid = 0;
        cmd_k_blocks = 0; cmd_group_div = 0; cmd_out_chans = 0;
    endtask

    task automatic finish_task();
        int start_cycle;
        start_cycle = cycles;
        res_ready = 1;
        while (model_state != M_IDLE) begin
            step();
            require(cycles - start_cycle < 10000, "last result timeout");
        end
        require(ev_last && !ev_cmd, "success must be the final result handshake alone");
        cmd_valid = 0;
        in_valid = 0;
        // Next issue() accepts a command on the immediately following edge.
    endtask

    // 0 extrema/lane order, 1 random int8/weights, 2 repeated activations per
    // channel with channel-dependent weights, 3 maximum positive dot product.
    task automatic make_input(input int index, input int k, input int pattern);
        int value;
        for (int lane = 0; lane < 32; lane++) begin
            case (pattern)
                1: value = int'(random_word() % 256) - 128;
                2: value = ((index % k) * 17 + lane * 3) % 256 - 128;
                3: value = -128;
                default: case (index % 8)
                    0, 1: value = -128;
                    2, 3: value = 127;
                    4: value = lane - 16;
                    5: case (lane % 4)
                        0: value = -128;
                        1: value = 127;
                        2: value = -1;
                        default: value = 0;
                    endcase
                    6: value = (lane * 37 + index * 11) % 256 - 128;
                    default: value = 0;
                endcase
            endcase
            act_data[lane*8 +: 8] = 8'(value);
            case (pattern)
                1: weight_data[lane] = 1'(random_word());
                2: weight_data[lane] = 1'((lane + index / k) % 3 != 0);
                3: weight_data[lane] = 0;
                default: case (index % 8)
                    0, 3: weight_data[lane] = 0;
                    1, 2: weight_data[lane] = 1;
                    default: weight_data[lane] = 1'(lane % 2);
                endcase
            endcase
        end
    endtask

    // ready_mode: 0 always ready, 1 random backpressure, 2 blocked.
    task automatic feed(input int count, input int base, input int k,
                         input int pattern = 0, input bit bubbles = 0,
                         input int ready_mode = 0);
        int sent, start_cycle;
        sent = 0; start_cycle = cycles;
        require(!in_valid, "feed called with pending common input");
        while (sent < count) begin
            if (!in_valid && (!bubbles || random_word() % 100 < 67)) begin
                make_input(base + sent, k, pattern);
                in_valid = 1;
            end
            case (ready_mode)
                1: res_ready = random_word() % 100 < 58;
                2: res_ready = 0;
                default: res_ready = 1;
            endcase
            step();
            if (ev_in) begin sent++; in_valid = 0; end
            require(cycles - start_cycle < count * 200 + 2000, "input feed timeout/deadlock");
        end
    endtask

    task automatic success_case(input string name, input int k, input int g,
                                 input int n, input int pattern = 0,
                                 input bit random_stalls = 0);
        announce(name);
        issue(k, g, n);
        feed(k * n, 0, k, pattern, random_stalls, random_stalls ? 1 : 0);
        finish_task();
        require(inputs == k * n && outputs == (k / g) * n,
                "successful task input/output count mismatch");
        $display("  PASS inputs=%0d results=%0d", inputs, outputs);
    endtask

    task automatic recovery(input string reason);
        success_case({reason, " / immediate recovery"}, 6, 2, 2, 2);
    endtask

    task automatic wait_blocked_result();
        int start_cycle;
        start_cycle = cycles;
        res_ready = 0;
        while (!res_valid) begin
            step();
            require(cycles - start_cycle < 100, "blocked result never appeared");
        end
        repeat (3) step();
    endtask

    task automatic reset_core();
        // Force every producer/consumer active with dirty payload/configuration.
        // A synchronous reset edge must flush state and suppress all handshakes.
        reset = 1;
        cmd_valid = 1; cmd_k_blocks = 0; cmd_group_div = 0; cmd_out_chans = 0;
        in_valid = 1; act_data = '1; weight_data = '0; res_ready = 1;
        repeat (2) step();
        reset = 0;
        quiet();
        repeat (16) step();
        require(model_state == M_IDLE && !res_valid && !status_busy,
                "reset left stale pipeline data or busy state");
    endtask

    initial begin : suite
        int g, k, n, start_cycle;
        if ($value$plusargs("seed=%d", seed)) begin end
        if ($value$plusargs("random_cases=%d", random_cases)) begin end
        if ($value$plusargs("max_cycles=%d", max_cycles)) begin end
        require(seed != 0, "seed must be nonzero");
        require(random_cases > 0, "random_cases must be positive");
        rng_state = seed;
        $display("[START] BNN common-input self-test seed=%0d random_cases=%0d", seed, random_cases);
        @(negedge clock);
        reset_core();

        announce("IDLE input ignored and invalid unaccepted command values harmless");
        in_valid = 1; act_data = '1; weight_data = '0;
        cmd_k_blocks = 0; cmd_group_div = 0; cmd_out_chans = 0;
        repeat (5) step();
        // Input presented on the command edge still cannot be accepted in IDLE.
        issue(2, 1, 1);
        // Retain the pending pair until the first RUN edge accepts it.
        step();
        require(ev_in, "pending IDLE input was not accepted in RUN");
        in_valid = 0;
        feed(1, 1, 2);
        finish_task();

        success_case("int8 extrema: all 32 positive/negative weights, lane mapping", 8, 1, 1);
        success_case("multiple groups/channels and repeated external activations", 12, 3, 3, 2);
        success_case("255-beat group: full int8 sum exceeds int16", 255, 255, 2, 3);
        success_case("256 groups: index 255 then reset to zero on next channel", 512, 2, 2, 1, 1);
        success_case("maximum legal k: 65280 blocks / 255 = 256 groups", 65280, 255, 1, 3);
        success_case("16-bit channel boundary: 65535 one-block channels", 1, 1, 65535, 2);

        announce("common payloads held together through prolonged input/output backpressure");
        issue(96, 1, 1);
        res_ready = 0; in_valid = 1; make_input(0, 96, 1);
        start_cycle = cycles;
        while (in_ready) begin
            step();
            if (ev_in) make_input(inputs, 96, 1);
            require(cycles - start_cycle < 100, "backpressure never reached common input");
        end
        repeat (12) step();
        require(held_input && model_state == M_RUN, "pending common input stall not exercised");
        res_ready = 1;
        do step(); while (!ev_in);
        in_valid = 0;
        feed(96 - inputs, inputs, 96, 1);
        finish_task();

        announce("last result blocked in DRAIN; accept pending command only on next edge");
        issue(1, 1, 1);
        feed(1, 0, 1, 0, 0, 2);
        wait_blocked_result();
        cmd_valid = 1; cmd_k_blocks = 6; cmd_group_div = 2; cmd_out_chans = 2;
        in_valid = 1; make_input(0, 6, 2);
        repeat (8) begin
            step();
            require(model_state == M_DRAIN && !ev_cmd && !ev_in, "blocked last result accepted work");
        end
        res_ready = 1;
        step();
        require(ev_last && !ev_cmd && !ev_in && status_idle && cmd_ready,
                "last result did not return to IDLE exclusively");
        step();
        require(ev_cmd && !ev_in && model_state == M_RUN,
                "pending command not accepted exactly one edge after final result");
        cmd_valid = 0; cmd_k_blocks = 0; cmd_group_div = 0; cmd_out_chans = 0;
        // The input stayed valid across DRAIN/IDLE and is first accepted in RUN.
        step();
        require(ev_in, "pending common input not accepted in RUN");
        in_valid = 0;
        feed(11, 1, 6, 2);
        finish_task();

        announce("reset RUN with a just-accepted tree beat");
        issue(8, 8, 1); feed(1, 0, 8);
        reset_core(); recovery("tree reset");

        announce("reset RUN with partial accumulation");
        issue(12, 6, 1); feed(3, 0, 12);
        repeat (8) step();
        require(outputs == 0, "partial group unexpectedly produced output");
        reset_core(); recovery("partial accumulator reset");

        announce("reset RUN with blocked output and tree data");
        issue(16, 1, 1); feed(3, 0, 16, 0, 0, 2);
        wait_blocked_result();
        require(model_state == M_RUN, "blocked RUN reset reached wrong phase");
        reset_core(); recovery("blocked RUN reset");

        announce("reset DRAIN before tree result emerges");
        issue(1, 1, 1); feed(1, 0, 1, 0, 0, 2);
        require(model_state == M_DRAIN && !res_valid, "DRAIN tree reset missed pipeline window");
        reset_core(); recovery("DRAIN tree reset");

        announce("reset DRAIN while final output is blocked");
        issue(1, 1, 1); feed(1, 0, 1, 0, 0, 2);
        wait_blocked_result();
        require(model_state == M_DRAIN, "blocked DRAIN reset reached wrong phase");
        reset_core(); recovery("blocked final output reset");

        announce("reset DRAIN with multiple queued groups and channel metadata");
        issue(4, 1, 1); feed(4, 0, 4, 1, 0, 2);
        require(model_state == M_DRAIN, "queued DRAIN reset reached wrong phase");
        reset_core(); recovery("queued groups reset");

        for (int test_index = 0; test_index < random_cases; test_index++) begin
            g = 1 + int'(random_word() % 9);
            k = g * (1 + int'(random_word() % 12));
            n = 1 + int'(random_word() % 4);
            success_case($sformatf("random %0d: k=%0d div=%0d n=%0d", test_index, k, g, n),
                         k, g, n, 1, 1);
        end

        if (!$test$plusargs("no_throughput")) begin
            throughput_check = 1;
            success_case("group_div=1 sustained one input/result per clock", 96, 1, 2, 1);
            throughput_check = 0;
        end else $display("[SKIP] throughput check explicitly disabled by +no_throughput");

        announce("post-suite idle has no extra results");
        quiet(); repeat (16) step();
        require(input_stalls >= 12 && output_stalls > 0, "common input/output backpressure not covered");
        require(blocked_last_cycles >= 8, "blocked final result not covered");
        require(idle_input_cycles > 0 && drain_input_cycles >= 8, "out-of-RUN input rejection not covered");
        require(reset_run == 3 && reset_drain == 3, "RUN/DRAIN reset stages missing");
        require(immediate_restarts >= 2, "immediately following command acceptance not covered");
        require(accepted_group_255 >= 2, "group index 255 not consumed for both channels");
        require(commands == completions + reset_run + reset_drain, "command completion/reset accounting mismatch");
        $display("[PASS] seed=%0d cases=%0d cycles=%0d commands=%0d completions=%0d results=%0d",
                 seed, cases, cycles, commands, completions, good_results);
        $display("       input_stall=%0d res_stall=%0d blocked_last=%0d immediate_restart=%0d reset_RUN=%0d reset_DRAIN=%0d group255=%0d",
                 input_stalls, output_stalls, blocked_last_cycles, immediate_restarts,
                 reset_run, reset_drain, accepted_group_255);
        $finish;
    end
endmodule
