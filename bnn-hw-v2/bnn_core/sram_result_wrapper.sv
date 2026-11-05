`default_nettype none

// `define TARGET_ASIC

/*
============================================================================
sram_result_wrapper (Result SRAM 浅包装：带字节写掩码的单端口 1RW 同步存储)
----------------------------------------------------------------------------
用途：
存放分块输出结果（默认 4096 拍 × 4×FP32 = 128-bit）。本模块是对 FPGA
Block RAM 或 ASIC 常见**单端口 1RW**（一个地址口，读或写二选一）SRAM 宏的
薄包装：内部只有存储阵列、字节写掩码 + 同步读输出流水，不含任何控制状态机、
复位清零或读写仲裁逻辑。
============================================================================
*/
module sram_result_wrapper #(
    parameter int unsigned DATA_WIDTH   = 128,            // 读写数据位宽（8 的整数倍）
    parameter int unsigned DEPTH        = 4096,           // 存储深度（2 的幂）
    parameter int unsigned ADDR_WIDTH   = $clog2(DEPTH),  // 地址位宽
    parameter int unsigned READ_LATENCY = 1,              // 读总延迟（>= 1 拍）
    parameter int unsigned STRB_WIDTH   = DATA_WIDTH / 8  // 字节写掩码位宽
) (
    input  wire                  clock,

    input  wire                  en,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] wdata,
    input  wire [STRB_WIDTH-1:0] wstrb,
    output wire [DATA_WIDTH-1:0] rdata
);


`ifdef TARGET_ASIC

    // 占位代码
    logic [DATA_WIDTH-1:0] stub_reg;
    always_ff @(posedge clock) begin
        if (en) begin
            stub_reg <= wdata ^ DATA_WIDTH'({we, addr, wstrb});
        end
    end
    assign rdata = stub_reg;

`elsif TARGET_FPGA


`else // SIM

    logic [DATA_WIDTH-1:0] read_stage [0:READ_LATENCY-1];

    logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];

    always_ff @(posedge clock) begin
        if (en && we) begin
            for (int byte_index = 0; byte_index < STRB_WIDTH; byte_index++) begin
                if (wstrb[byte_index])
                    memory[addr][byte_index*8 +: 8] <= wdata[byte_index*8 +: 8];
            end
        end
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
        assert (DATA_WIDTH > 0 && (DATA_WIDTH % 8) == 0)
            else $fatal(1, "sram_result_wrapper: DATA_WIDTH must be a positive multiple of 8");
        assert (STRB_WIDTH == DATA_WIDTH / 8)
            else $fatal(1, "sram_result_wrapper: STRB_WIDTH must equal DATA_WIDTH/8");
        assert (DEPTH > 0 && (64'd1 << ADDR_WIDTH) == 64'(DEPTH))
            else $fatal(1, "sram_result_wrapper: DEPTH must equal 2**ADDR_WIDTH");
        assert (READ_LATENCY >= 1)
            else $fatal(1, "sram_result_wrapper: READ_LATENCY must be >= 1");
    end
`endif
endmodule

`default_nettype wire
