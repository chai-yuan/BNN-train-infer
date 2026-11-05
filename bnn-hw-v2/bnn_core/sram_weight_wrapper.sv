`default_nettype none

// `define TARGET_ASIC

/*
============================================================================
sram_weight_wrapper (Weight SRAM 浅包装：单端口 1RW 同步存储)
----------------------------------------------------------------------------
用途：
存放推理权重（默认 128 条 × 144-bit）。本模块是对 FPGA Block RAM 或 ASIC
常见**单端口 1RW**（一个地址口，读或写二选一）SRAM 宏的薄包装：内部只有
存储阵列 + 同步读输出流水，不含任何控制状态机、复位清零、读写仲裁或数据
变换逻辑。
============================================================================
*/
module sram_weight_wrapper #(
    parameter int unsigned DATA_WIDTH   = 144,            // 读写数据位宽
    parameter int unsigned DEPTH        = 128,            // 存储深度（2 的幂）
    parameter int unsigned ADDR_WIDTH   = $clog2(DEPTH),  // 地址位宽
    parameter int unsigned READ_LATENCY = 1               // 读总延迟（>= 1 拍）
) (
    input  wire                  clock,

    input  wire                  en,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] wdata,
    output wire [DATA_WIDTH-1:0] rdata
);

`ifdef TARGET_ASIC

    // 占位代码
    logic [DATA_WIDTH-1:0] stub_reg;
    always_ff @(posedge clock) begin
        if (en) begin
            stub_reg <= wdata ^ DATA_WIDTH'({we, addr});
        end
    end
    assign rdata = stub_reg;

`elsif TARGET_FPGA


`else // SIM

    logic [DATA_WIDTH-1:0] read_stage [0:READ_LATENCY-1];

    logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];

    always_ff @(posedge clock) begin
        if (en && we)
            memory[addr] <= wdata;
        if (en && !we)
            read_stage[0] <= memory[addr];
    end

    always_ff @(posedge clock) begin
        for (int stage = 1; stage < READ_LATENCY; stage++)
            read_stage[stage] <= read_stage[stage-1];
    end

    assign rdata = read_stage[READ_LATENCY-1];

`endif

`ifndef SYNTHESIS
    initial begin
        assert (DATA_WIDTH > 0)
            else $fatal(1, "sram_weight_wrapper: DATA_WIDTH must be positive");
        assert (DEPTH > 0 && (64'd1 << ADDR_WIDTH) == 64'(DEPTH))
            else $fatal(1, "sram_weight_wrapper: DEPTH must equal 2**ADDR_WIDTH");
        assert (READ_LATENCY >= 1)
            else $fatal(1, "sram_weight_wrapper: READ_LATENCY must be >= 1");
    end
`endif
endmodule

`default_nettype wire
