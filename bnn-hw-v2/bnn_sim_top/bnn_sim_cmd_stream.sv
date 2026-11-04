`default_nettype none

/*
============================================================================
bnn_sim_cmd_stream (线性命令流发生器)
----------------------------------------------------------------------------
- 命令存储器由 C++ DPI-C 实现，本模块只负责取指和 valid/ready 握手。
- 契约：
    int cmd_check();              // 非 0 表示下一条指令可用（纯检查）
    longint unsigned cmd_get();   // 返回下一条 64-bit 指令并推进游标
- 空闲时若 cmd_check() 为真，调用 cmd_get() 取出一条指令并保持 cmd_valid；
  核心 cmd_ready 后立即回到空闲，继续取指，因此命令是线性连续供应的。
- allow_finish 由 testbench 在确认不会再有新指令时拉高；此前即使队列为空也保持
  done 为低，避免交互式仿真在两条指令之间提前结束。
- allow_finish 有效、无在途命令且核心/DMA 空闲时 done 保持为高。
- 核心返回 cmd_error 时停止取指并保持 error 为高。
- reset 期间不调用任何 DPI 函数；C++ 侧应在 reset 时重置命令游标。
============================================================================
*/
module bnn_sim_cmd_stream (
    input  wire       clock,
    input  wire       reset,

    output wire       cmd_valid,
    output wire [63:0] cmd_inst,
    input  wire       cmd_ready,
    input  wire       cmd_done,
    input  wire       cmd_error,

    input  wire       core_busy,
    input  wire       allow_finish,
    output wire       command_active,
    output wire       done,
    output wire       error
);
    import "DPI-C" function int cmd_check();
    import "DPI-C" function longint unsigned cmd_get();

    typedef enum logic {WAIT_CMD, PRESENT} state_t;
    state_t state;

    logic [63:0] inst;
    logic        outstanding;
    logic        done_reg;
    logic        error_reg;

    assign cmd_valid = !reset && (state == PRESENT);
    assign cmd_inst  = inst;
    assign command_active = !reset && ((state == PRESENT) || outstanding);
    assign done      = !reset && done_reg;
    assign error     = !reset && error_reg;

    always_ff @(posedge clock) begin
        if (reset) begin
            state       <= WAIT_CMD;
            inst        <= '0;
            outstanding <= 1'b0;
            done_reg    <= 1'b0;
            error_reg   <= 1'b0;
        end else if (cmd_error) begin
            error_reg   <= 1'b1;
            outstanding <= 1'b0;
            done_reg    <= 1'b0;
            state       <= WAIT_CMD;
        end else if (!error_reg && !done_reg) begin
            if (cmd_done)
                outstanding <= 1'b0;

            case (state)
                WAIT_CMD: begin
                    if (cmd_check() != 0) begin
                        inst  <= 64'(cmd_get());
                        state <= PRESENT;
                    end else if (!outstanding && !core_busy && allow_finish) begin
                        done_reg <= 1'b1;
                    end
                end
                PRESENT: if (cmd_ready) begin
                    outstanding <= 1'b1;
                    state       <= WAIT_CMD;
                end
                default: state <= WAIT_CMD;
            endcase
        end
    end
endmodule

`default_nettype wire
