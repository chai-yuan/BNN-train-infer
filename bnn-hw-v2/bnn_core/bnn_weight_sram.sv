`default_nettype none

/*
============================================================================
bnn_weight_sram (片上权重 SRAM)
----------------------------------------------------------------------------
- 简单 1R1W 推断式存储，位宽 WEIGHT_WIDTH，深度 2^ADDR_WIDTH；不复位清数组。
- 写口在 T 沿生效；读口为同步读，经 READ_LATENCY 级寄存器输出。
  READ_LATENCY = 1 时，T 沿发出读请求，T+1 沿可采样 rdata。
- 不包含读写仲裁：上层须保证同一拍 wen/ren 不冲突，并在 OP_LOAD_WEIGHT
  与 OP_COMPUTE 之间时分复用。
- 不在复位时清空阵列，保持块 RAM 推断。
============================================================================
*/
module bnn_weight_sram #(
    parameter int unsigned WEIGHT_WIDTH = 144,
    parameter int unsigned ADDR_WIDTH   = 7,
    parameter int unsigned READ_LATENCY = 1
) (
    input  wire                         clock,
    input  wire                         wen,
    input  wire [ADDR_WIDTH-1:0]        waddr,
    input  wire [WEIGHT_WIDTH-1:0]      wdata,
    input  wire                         ren,
    input  wire [ADDR_WIDTH-1:0]        raddr,
    output wire [WEIGHT_WIDTH-1:0]      rdata
);
    localparam int unsigned DEPTH = 1 << ADDR_WIDTH;

    logic [WEIGHT_WIDTH-1:0] memory [0:DEPTH-1];
    logic [WEIGHT_WIDTH-1:0] read_pipe [0:READ_LATENCY-1];

    always_ff @(posedge clock) begin
        if (wen)
            memory[waddr] <= wdata;
        if (ren)
            read_pipe[0] <= memory[raddr];
        for (int stage = 1; stage < READ_LATENCY; stage = stage + 1)
            read_pipe[stage] <= read_pipe[stage-1];
    end

    assign rdata = read_pipe[READ_LATENCY-1];

`ifndef SYNTHESIS
    initial begin
        assert (WEIGHT_WIDTH > 0)
            else $fatal(1, "Weight SRAM width must be positive");
        assert (ADDR_WIDTH >= 1 && ADDR_WIDTH <= 30)
            else $fatal(1, "Weight SRAM address width must be in [1, 30]");
        assert (READ_LATENCY >= 1)
            else $fatal(1, "Weight SRAM read latency must be >= 1");
    end
`endif
endmodule

`default_nettype wire
