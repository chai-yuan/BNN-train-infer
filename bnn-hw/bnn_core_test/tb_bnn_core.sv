`timescale 1ns/1ps

// Black-box tests for the command/done interface. No hierarchical RTL references.
// Run with +seed=<nonzero u32> +random_cases=<count> +max_cycles=<watchdog>.
// Sources never wait for ready to raise valid, and retain pending beats.
module tb_bnn_core;
    logic clock = 0;
    always #5 clock = ~clock;

    logic reset = 1;
    logic ctrl_abort = 0;
    logic cmd_valid = 0;
    wire cmd_ready;
    logic [15:0] cmd_k_blocks = 0;
    logic [7:0] cmd_group_div = 0;
    logic [15:0] cmd_out_chans = 0;
    wire done_valid;
    logic done_ready = 0;
    wire [1:0] done_status;
    wire status_idle, status_busy;
    logic act_valid = 0;
    wire act_ready;
    logic [255:0] act_data = 0;
    logic act_last = 0;
    logic weight_valid = 0;
    wire weight_ready;
    logic [31:0] weight_data = 0;
    logic weight_last = 0;
    wire res_valid;
    logic res_ready = 1;
    wire signed [31:0] res_data;
    wire res_last;
    wire [7:0] res_group_idx;
    wire res_chan_last;

    // Deliberately use the defaults: they are part of the interface contract.
    bnn_core_top dut (
        .clock(clock), .reset(reset), .ctrl_abort(ctrl_abort),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_k_blocks(cmd_k_blocks), .cmd_group_div(cmd_group_div),
        .cmd_out_chans(cmd_out_chans),
        .done_valid(done_valid), .done_ready(done_ready), .done_status(done_status),
        .status_idle(status_idle), .status_busy(status_busy),
        .act_valid(act_valid), .act_ready(act_ready), .act_data(act_data),
        .act_last(act_last), .weight_valid(weight_valid),
        .weight_ready(weight_ready), .weight_data(weight_data),
        .weight_last(weight_last), .res_valid(res_valid), .res_ready(res_ready),
        .res_data(res_data), .res_last(res_last),
        .res_group_idx(res_group_idx), .res_chan_last(res_chan_last)
    );

    typedef enum {M_IDLE, M_RUN, M_DRAIN, M_DONE} model_state_t;
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
    logic [1:0] completion = 0;
    int unsigned seed = 1, rng_state = 1;
    int random_cases = 16;
    int max_cycles = 500000;
    int cycles = 0, cases = 0;
    string case_name = "initial reset";

    // Events sampled BEFORE the rising-edge NBA updates, for the drivers.
    bit ev_cmd, ev_act, ev_weight, ev_pair, ev_res, ev_done;
    bit held_act = 0, held_weight = 0, held_res = 0;
    logic [255:0] held_act_data;
    logic held_act_last;
    logic [31:0] held_weight_data;
    logic held_weight_last;
    result_t held_result;
    bit throughput_check = 0;
    int previous_input_cycle = -1, previous_output_cycle = -1;
    int commands = 0, completions = 0, cancelled = 0, protocol_errors = 0;
    int bad_commands = 0, good_results = 0;
    int act_only_cycles = 0, weight_only_cycles = 0;
    int output_stalls = 0, done_stalls = 0;
    int accepted_group_255 = 0, suppressed_stalled = 0;

    task automatic require(input bit condition, input string message);
        assert (condition) else
            $fatal(1, "[FAIL] seed=%0d cycle=%0d case='%s': %s",
                   seed, cycles, case_name, message);
    endtask

    // Private PRNG avoids tool-specific $urandom sequencing. Never returns zero
    // from a nonzero seed. All bubble decisions are separate draws.
    function automatic int unsigned random_word();
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        return rng_state;
    endfunction

    function automatic bit valid_config(input int k, input int g, input int n);
        if (k == 0 || g == 0 || n == 0) return 0;
        return (k % g == 0) && (k <= g * 256);
    endfunction

    // This is the only checker entry point. A single always block samples once
    // per rising edge; all stimulus changes occur on the following falling edge.
    // It models accepted transactions, never speculative driver intentions.
    task automatic sample();
        bit active, bad_last, killing;
        bit want_last;
        int lane_value;
        longint signed dot;
        result_t actual, item;
        cycles++;
        require(cycles <= max_cycles, "global cycle watchdog expired");
        ev_cmd = cmd_valid && cmd_ready;
        ev_act = act_valid && act_ready;
        ev_weight = weight_valid && weight_ready;
        ev_pair = ev_act && ev_weight;
        ev_res = res_valid && res_ready;
        ev_done = done_valid && done_ready;
        actual = '{data: res_data, group_idx: res_group_idx,
                   chan_last: res_chan_last, last: res_last};

        if (reset) begin
            require(!cmd_ready, "cmd_ready must be masked during reset");
            model_state = M_IDLE;
            expected.delete();
            mk = 0; mg = 0; mn = 0;
            inputs = 0; outputs = 0; group_sum = 0; completion = 0;
            held_act = 0; held_weight = 0; held_res = 0;
            previous_input_cycle = -1; previous_output_cycle = -1;
            return;
        end

        active = model_state == M_RUN || model_state == M_DRAIN;
        require(status_idle === (model_state == M_IDLE), "status_idle disagrees with task state");
        require(status_busy === active, "status_busy must indicate RUN/DRAIN only");
        require(cmd_ready === ((model_state == M_IDLE) && !ctrl_abort),
                "cmd_ready must indicate idle && !reset && !abort");
        require(done_valid === (model_state == M_DONE),
                "done_valid missing, premature, or not held until handshake");
        if (model_state == M_DONE)
            require(done_status === completion, "completion status changed or is incorrect");
        require(ev_act == ev_weight, "input streams consumed separately instead of as a pair");
        if (!active) require(!ev_pair, "input consumed outside RUN");
        if (model_state == M_DRAIN) require(!ev_pair, "extra input consumed during DRAIN");

        bad_last = 0;
        if (model_state == M_RUN && ev_pair) begin
            want_last = inputs % mk == mk - 1;
            bad_last = (act_last !== want_last) || (weight_last !== want_last);
        end
        killing = active && (ctrl_abort || bad_last);
        if (held_act && !killing)
            require(act_valid && act_data === held_act_data && act_last === held_act_last,
                    "test source changed activation while stalled");
        if (held_weight && !killing)
            require(weight_valid && weight_data === held_weight_data && weight_last === held_weight_last,
                    "test source changed weight while stalled");
        if (held_res && !killing)
            require(res_valid && actual === held_result, "result valid/payload changed while stalled");
        if (killing) begin
            require(!res_valid, "abort/accepted bad last must mask res_valid on the SAME cycle");
            if (held_res) suppressed_stalled++;
        end
        if (active && act_valid && !weight_valid) act_only_cycles++;
        if (active && weight_valid && !act_valid) weight_only_cycles++;
        if (res_valid && !res_ready) output_stalls++;
        if (done_valid && !done_ready) done_stalls++;

        case (model_state)
            M_IDLE: begin
                require(!res_valid, "stale result visible in IDLE");
                if (ev_cmd) begin
                    commands++;
                    mk = int'(cmd_k_blocks);
                    mg = int'(cmd_group_div);
                    mn = int'(cmd_out_chans);
                    inputs = 0; outputs = 0; group_sum = 0;
                    previous_input_cycle = -1; previous_output_cycle = -1;
                    expected.delete();
                    if (valid_config(mk, mg, mn)) model_state = M_RUN;
                    else begin
                        model_state = M_DONE;
                        completion = 1;
                        bad_commands++;
                    end
                end
            end
            M_DONE: begin
                require(!res_valid && !ev_cmd, "DONE accepted work or exposed stale output");
                if (ev_done) begin
                    completions++;
                    model_state = M_IDLE;
                end
            end
            M_RUN, M_DRAIN: begin
                require(!ev_cmd, "busy core accepted a second command");
                if (killing) begin
                    expected.delete();
                    group_sum = 0;
                    model_state = M_DONE;
                    if (ctrl_abort) begin
                        completion = 2;
                        cancelled++;
                    end else begin
                        completion = 3;
                        protocol_errors++;
                    end
                end else begin
                    if (ev_pair) begin
                        require(inputs < mk * mn, "too many input beats accepted");
                        if (throughput_check && previous_input_cycle >= 0)
                            require(cycles == previous_input_cycle + 1,
                                    "group_div=1 input throughput is below one beat per clock");
                        previous_input_cycle = cycles;
                        dot = 0;
                        for (int lane = 0; lane < 32; lane++) begin
                            // Widen BEFORE negation: -(-128) must become +128.
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
                                completion = 0;
                                model_state = M_DONE;
                            end
                        end
                    end
                end
            end
            default: require(0, "invalid reference state");
        endcase

        held_act = active && !killing && model_state != M_DONE && act_valid && !act_ready;
        held_weight = active && !killing && model_state != M_DONE && weight_valid && !weight_ready;
        held_res = !killing && res_valid && !res_ready;
        held_act_data = act_data; held_act_last = act_last;
        held_weight_data = weight_data; held_weight_last = weight_last;
        held_result = actual;
    endtask

    // Keep the sizeable scoreboard in one process instead of inlining it at
    // every directed task call. Drivers observe ev_* only at the next negedge.
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
        cmd_valid = 0; ctrl_abort = 0;
        act_valid = 0; weight_valid = 0;
        done_ready = 0; res_ready = 1;
    endtask

    task automatic issue(input int k, input int g, input int n);
        require(model_state == M_IDLE, "driver issued command before previous completion handshake");
        cmd_valid = 1;
        cmd_k_blocks = 16'(k); cmd_group_div = 8'(g); cmd_out_chans = 16'(n);
        step();
        require(ev_cmd, "idle command was not accepted");
        // Deliberately poison live configuration after handshake. The model uses
        // the accepted values, so following them instead of latching will fail.
        cmd_valid = 0;
        cmd_k_blocks = 0; cmd_group_div = 255; cmd_out_chans = 0;
    endtask

    task automatic finish_task(input int status, input int hold_cycles = 5);
        int start_cycle;
        start_cycle = cycles;
        while (!done_valid) begin
            step();
            require(cycles - start_cycle < 10000, "completion timeout");
        end
        require(done_status === 2'(status), "wrong completion status at driver");
        // A pending next command and abort must not overwrite a held completion.
        cmd_valid = 1;
        cmd_k_blocks = 1; cmd_group_div = 1; cmd_out_chans = 1;
        for (int i = 0; i < hold_cycles; i++) begin
            ctrl_abort = 1'(i % 2);
            step();
            require(!ev_cmd && !ev_done, "work/completion accepted during done backpressure");
        end
        ctrl_abort = 0;
        done_ready = 1;
        step();
        require(ev_done && !ev_cmd, "completion handshake failed or overlapped next command");
        done_ready = 0;
        cmd_valid = 0;
        require(model_state == M_IDLE, "completion did not return to idle");
        // Returns on this very falling edge; the next test may restart directly.
    endtask

    // pattern 0: directed extrema and lane ordering; 1: full-range random;
    // 2: repeated activation vector per channel, channel-dependent weights;
    // 3: maximum positive dot, exercising accumulation beyond int16.
    task automatic make_activation(input int index, input int k, input int pattern,
                                   input int error_index, input int error_mode);
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
        end
        act_last = index % k == k - 1;
        if (index == error_index) begin
            case (error_mode)
                1, 5: act_last = 1;
                2, 3, 6: act_last = 0;
                default: ;
            endcase
        end
    endtask

    task automatic make_weight(input int index, input int k, input int pattern,
                               input int error_index, input int error_mode);
        for (int lane = 0; lane < 32; lane++) begin
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
        weight_last = index % k == k - 1;
        if (index == error_index) begin
            case (error_mode)
                1, 6: weight_last = 1;
                2, 4, 5: weight_last = 0;
                default: ;
            endcase
        end
    endtask

    // ready_mode: 0 always ready; 1 random backpressure; 2 blocked.
    // base permits sending a partial group, pausing, then continuing the task.
    task automatic feed(input int count, input int base, input int k,
                         input int pattern = 0, input bit bubbles = 0,
                         input int ready_mode = 0,
                         input int error_index = -1, input int error_mode = 0);
        int a, w, start_cycle;
        a = 0; w = 0; start_cycle = cycles;
        require(!act_valid && !weight_valid, "feed called with pending source data");
        while (a < count || w < count) begin
            // Once asserted, each valid stays high and its payload stays intact
            // until its own handshake. The two sources have independent bubbles.
            if (!act_valid && a < count && (!bubbles || random_word() % 100 < 72)) begin
                make_activation(base + a, k, pattern, error_index, error_mode);
                act_valid = 1;
            end
            if (!weight_valid && w < count && (!bubbles || random_word() % 100 < 61)) begin
                make_weight(base + w, k, pattern, error_index, error_mode);
                weight_valid = 1;
            end
            case (ready_mode)
                1: res_ready = random_word() % 100 < 58;
                2: res_ready = 0;
                default: res_ready = 1;
            endcase
            step();
            if (ev_act) begin a++; act_valid = 0; end
            if (ev_weight) begin w++; weight_valid = 0; end
            require(cycles - start_cycle < count * 200 + 2000, "input feed timeout/deadlock");
            require(model_state != M_DONE || (a == count && w == count),
                    "task terminated before requested input sequence ended");
        end
        require(a == count && w == count, "source handshake count mismatch");
    endtask

    task automatic success_case(input string name, input int k, input int g,
                                 input int n, input int pattern = 0,
                                 input bit random_stalls = 0);
        announce(name);
        issue(k, g, n);
        // Hold an unrelated next command throughout RUN/DRAIN/DONE. Keep
        // its payload stable under cmd backpressure, including finish_task().
        cmd_valid = 1;
        cmd_k_blocks = 1; cmd_group_div = 1; cmd_out_chans = 1;
        feed(k * n, 0, k, pattern, random_stalls, random_stalls ? 1 : 0);
        res_ready = 1;
        finish_task(0);
        require(inputs == k * n && outputs == (k / g) * n,
                "successful task input/output count mismatch");
        $display("  PASS inputs=%0d results=%0d", inputs, outputs);
    endtask

    task automatic recovery(input string reason);
        // No idle bubble after the preceding done handshake.
        success_case({reason, " / immediate recovery"}, 6, 2, 2, 2, 0);
    endtask

    task automatic invalid_case(input string name, input int k, input int g, input int n);
        announce(name);
        // Present input even on the bad command cycle; none may be consumed.
        act_valid = 1; act_data = '1; act_last = 1;
        weight_valid = 1; weight_data = '0; weight_last = 1;
        issue(k, g, n);
        finish_task(1, 7);
        require(inputs == 0 && outputs == 0, "bad configuration consumed input or produced output");
        quiet();
        recovery(name);
    endtask

    task automatic abort_now();
        ctrl_abort = 1;
        res_ready = 1; // Exposes a stale result even if cancellation races a read.
        step();
        quiet();
        finish_task(2, 3);
    endtask

    task automatic wait_blocked_result();
        int start_cycle;
        start_cycle = cycles;
        res_ready = 0;
        while (!res_valid) begin
            step();
            require(cycles - start_cycle < 100, "blocked result never appeared");
        end
        // Sample at least three stalled cycles before cancellation/reset.
        repeat (3) step();
    endtask

    task automatic protocol_case(input string name, input int error_index, input int mode);
        announce(name);
        issue(4, 1, 2);
        feed(error_index + 1, 0, 4, 0, 1, 1, error_index, mode);
        quiet();
        finish_task(3);
        require(inputs == error_index, "malformed beat was accumulated/counted");
        recovery(name);
    endtask

    task automatic reset_core();
        reset = 1;
        ctrl_abort = 0;
        cmd_valid = 1; cmd_k_blocks = 1; cmd_group_div = 1; cmd_out_chans = 1;
        done_ready = 1;
        repeat (2) step();
        reset = 0;
        quiet();
        // The next step checks idle, no completion and no leftover result.
        step();
    endtask

    initial begin : suite
        int g, k, n;
        if ($value$plusargs("seed=%d", seed)) begin end
        if ($value$plusargs("random_cases=%d", random_cases)) begin end
        if ($value$plusargs("max_cycles=%d", max_cycles)) begin end
        require(seed != 0, "seed must be nonzero");
        require(random_cases > 0, "random_cases must be positive");
        rng_state = seed;
        $display("[START] BNN command/done self-test seed=%0d random_cases=%0d", seed, random_cases);
        @(negedge clock);
        reset_core();

        announce("IDLE abort ignores command and produces no completion");
        ctrl_abort = 1; cmd_valid = 1;
        repeat (3) step();
        quiet();
        step();

        success_case("int8 extrema: all 32 positive/negative weights, lane mapping", 8, 1, 1);
        success_case("multiple groups/channels and repeated external activations", 12, 3, 3, 2);
        success_case("255-beat group: full int8 sum exceeds int16", 255, 255, 2, 3);
        success_case("256 groups: index 255 then reset to zero on next channel", 512, 2, 2, 1, 1);
        success_case("maximum legal k: 65280 blocks / 255 = 256 groups", 65280, 255, 1, 3);
        success_case("16-bit channel boundary: 65535 one-block channels", 1, 1, 65535, 2);

        announce("independent sources may assert valid before the other source");
        issue(2, 1, 1);
        make_activation(0, 2, 0, -1, 0); act_valid = 1;
        repeat (4) begin step(); require(!ev_pair, "activation consumed without weight"); end
        make_weight(0, 2, 0, -1, 0); weight_valid = 1;
        step(); require(ev_pair, "paired first beat not accepted");
        act_valid = 0; weight_valid = 0;
        make_weight(1, 2, 0, -1, 0); weight_valid = 1;
        repeat (4) begin step(); require(!ev_pair, "weight consumed without activation"); end
        make_activation(1, 2, 0, -1, 0); act_valid = 1;
        step(); require(ev_pair, "paired second beat not accepted");
        quiet(); finish_task(0);

        announce("last result held in DRAIN: no success before its handshake");
        issue(1, 1, 1);
        feed(1, 0, 1, 0, 0, 2);
        wait_blocked_result();
        require(model_state == M_DRAIN && !done_valid, "success before last output handshake");
        repeat (8) step();
        res_ready = 1;
        finish_task(0, 9);

        announce("abort with a just-accepted tree beat");
        issue(8, 8, 2);
        feed(1, 0, 8, 0, 0, 2);
        abort_now(); recovery("tree abort");

        announce("abort with a partially accumulated group");
        issue(12, 6, 1);
        feed(3, 0, 12);
        repeat (8) step();
        require(outputs == 0, "partial group unexpectedly produced output");
        abort_now(); recovery("partial accumulator abort");

        announce("abort in RUN with blocked output and more tree data");
        issue(8, 1, 2);
        feed(3, 0, 8, 0, 0, 2);
        wait_blocked_result();
        require(model_state == M_RUN, "RUN abort scenario reached wrong phase");
        abort_now(); recovery("blocked RUN abort");

        announce("abort in DRAIN with blocked final output");
        issue(1, 1, 1);
        feed(1, 0, 1, 0, 0, 2);
        wait_blocked_result();
        require(model_state == M_DRAIN, "DRAIN abort scenario reached wrong phase");
        abort_now(); recovery("blocked DRAIN abort");

        invalid_case("bad config: zero k", 0, 1, 1);
        invalid_case("bad config: zero divisor", 4, 0, 1);
        invalid_case("bad config: zero channels", 4, 1, 0);
        invalid_case("bad config: nondivisible", 5, 2, 1);
        invalid_case("bad config: group divisor exceeds k", 2, 3, 1);
        invalid_case("bad config: 257 groups", 257, 1, 1);
        invalid_case("bad config: 257 groups with divisor 2", 514, 2, 1);
        invalid_case("bad config: 65535 / 255 = 257 groups", 65535, 255, 1);

        protocol_case("bad last: both early", 1, 1);
        protocol_case("bad last: both missing at channel end", 3, 2);
        protocol_case("bad last: activation missing at channel end", 3, 3);
        protocol_case("bad last: weight missing at channel end", 3, 4);
        protocol_case("bad last: activation early, weight low", 1, 5);
        protocol_case("bad last: weight early, activation low", 1, 6);

        announce("bad last on an unpaired source is not yet a protocol error");
        issue(4, 2, 1);
        make_activation(0, 4, 0, 0, 1); act_valid = 1;
        repeat (5) begin
            step();
            require(!done_valid && model_state == M_RUN, "unaccepted last caused protocol failure");
        end
        make_weight(0, 4, 0, 0, 1); weight_valid = 1;
        step(); require(ev_pair, "malformed pair was not accepted for detection");
        quiet(); finish_task(3); recovery("unpaired malformed last");

        announce("accepted protocol error suppresses an already blocked result immediately");
        issue(8, 1, 1);
        feed(1, 0, 8, 0, 0, 2);
        wait_blocked_result();
        feed(1, 1, 8, 0, 0, 0, 1, 1);
        quiet(); finish_task(3); recovery("protocol flush with visible output");

        announce("reset with tree data: clear without done notification");
        issue(8, 8, 1);
        feed(1, 0, 8);
        reset_core(); recovery("tree reset");

        announce("reset with partial accumulation");
        issue(12, 6, 1);
        feed(3, 0, 12);
        repeat (8) step();
        reset_core(); recovery("partial accumulator reset");

        announce("reset while final output is blocked");
        issue(1, 1, 1);
        feed(1, 0, 1, 0, 0, 2);
        wait_blocked_result();
        reset_core(); recovery("blocked output reset");

        announce("reset discards a held completion without notification");
        issue(0, 1, 1);
        repeat (3) step();
        reset_core(); recovery("held completion reset");

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

        announce("post-suite idle has no extra completion/results");
        quiet(); repeat (16) step();
        require(act_only_cycles > 0 && weight_only_cycles > 0, "independent source stalls not covered");
        require(output_stalls > 0 && done_stalls > 0, "output/done backpressure not covered");
        require(cancelled == 4, "abort scenarios missing");
        require(protocol_errors == 8, "protocol error scenarios missing");
        require(bad_commands == 9, "configuration error scenarios missing");
        require(accepted_group_255 >= 2, "group index 255 not consumed for both channels");
        require(suppressed_stalled >= 3, "same-cycle blocked output suppression not covered");
        $display("[PASS] seed=%0d cases=%0d cycles=%0d commands=%0d completions=%0d results=%0d",
                 seed, cases, cycles, commands, completions, good_results);
        $display("       aborts=%0d protocol_errors=%0d bad_commands=%0d act_only=%0d weight_only=%0d res_stall=%0d done_stall=%0d group255=%0d suppressed_blocked=%0d",
                 cancelled, protocol_errors, bad_commands, act_only_cycles, weight_only_cycles,
                 output_stalls, done_stalls, accepted_group_255, suppressed_stalled);
        $finish;
    end
endmodule
