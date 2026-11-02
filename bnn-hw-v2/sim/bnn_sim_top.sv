`default_nettype none

// Only host commands and timing configuration cross the C++/RTL boundary here.
// DMA traffic itself goes through bnn_dma_memory and its DPI-C callbacks.
module bnn_sim_top (
    input  wire clock,
    input  wire reset,
    input  wire cmd_valid,
    input  wire [63:0] cmd_inst,
    output wire cmd_ready,
    output wire cmd_done,
    output wire cmd_error,
    output wire status_busy,
    input  wire [31:0] cfg_rd_latency,
    input  wire [31:0] cfg_rd_gap,
    input  wire [31:0] cfg_wr_latency,
    input  wire [31:0] cfg_cmd_stall_period,
    input  wire [31:0] cfg_wr_stall_period,
    output wire rd_busy,
    output wire wr_busy
);
    wire mem_rd_cmd_valid, mem_rd_cmd_ready;
    wire [31:0] mem_rd_cmd_addr;
    wire [15:0] mem_rd_cmd_len;
    wire mem_rd_data_valid, mem_rd_data_ready, mem_rd_data_last;
    wire [127:0] mem_rd_data;
    wire mem_wr_cmd_valid, mem_wr_cmd_ready;
    wire [31:0] mem_wr_cmd_addr;
    wire [15:0] mem_wr_cmd_len;
    wire mem_wr_data_valid, mem_wr_data_ready, mem_wr_data_last, mem_wr_done;
    wire [127:0] mem_wr_data;
    wire [15:0] mem_wr_data_strb;

    bnn_core_top u_core (
        .clock(clock), .reset(reset),
        .cmd_valid(cmd_valid), .cmd_inst(cmd_inst), .cmd_ready(cmd_ready),
        .cmd_done(cmd_done), .cmd_error(cmd_error), .status_busy(status_busy),
        .mem_rd_cmd_valid(mem_rd_cmd_valid), .mem_rd_cmd_ready(mem_rd_cmd_ready),
        .mem_rd_cmd_addr(mem_rd_cmd_addr), .mem_rd_cmd_len(mem_rd_cmd_len),
        .mem_rd_data_valid(mem_rd_data_valid), .mem_rd_data_ready(mem_rd_data_ready),
        .mem_rd_data(mem_rd_data), .mem_rd_data_last(mem_rd_data_last),
        .mem_wr_cmd_valid(mem_wr_cmd_valid), .mem_wr_cmd_ready(mem_wr_cmd_ready),
        .mem_wr_cmd_addr(mem_wr_cmd_addr), .mem_wr_cmd_len(mem_wr_cmd_len),
        .mem_wr_data_valid(mem_wr_data_valid), .mem_wr_data_ready(mem_wr_data_ready),
        .mem_wr_data(mem_wr_data), .mem_wr_data_strb(mem_wr_data_strb),
        .mem_wr_data_last(mem_wr_data_last), .mem_wr_done(mem_wr_done)
    );

    bnn_dma_memory u_memory (
        .clock(clock), .reset(reset),
        .cfg_rd_latency(cfg_rd_latency), .cfg_rd_gap(cfg_rd_gap),
        .cfg_wr_latency(cfg_wr_latency), .cfg_cmd_stall_period(cfg_cmd_stall_period),
        .cfg_wr_stall_period(cfg_wr_stall_period), .rd_busy(rd_busy), .wr_busy(wr_busy),
        .mem_rd_cmd_valid(mem_rd_cmd_valid), .mem_rd_cmd_ready(mem_rd_cmd_ready),
        .mem_rd_cmd_addr(mem_rd_cmd_addr), .mem_rd_cmd_len(mem_rd_cmd_len),
        .mem_rd_data_valid(mem_rd_data_valid), .mem_rd_data_ready(mem_rd_data_ready),
        .mem_rd_data(mem_rd_data), .mem_rd_data_last(mem_rd_data_last),
        .mem_wr_cmd_valid(mem_wr_cmd_valid), .mem_wr_cmd_ready(mem_wr_cmd_ready),
        .mem_wr_cmd_addr(mem_wr_cmd_addr), .mem_wr_cmd_len(mem_wr_cmd_len),
        .mem_wr_data_valid(mem_wr_data_valid), .mem_wr_data_ready(mem_wr_data_ready),
        .mem_wr_data(mem_wr_data), .mem_wr_data_strb(mem_wr_data_strb),
        .mem_wr_data_last(mem_wr_data_last), .mem_wr_done(mem_wr_done)
    );
endmodule

`default_nettype wire
