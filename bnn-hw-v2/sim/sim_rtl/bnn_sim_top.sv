`default_nettype none

/*
============================================================================
bnn_sim_top (BNN v2 最小 Verilator 仿真顶层)
----------------------------------------------------------------------------
对外只有 clock/reset。命令流和数据访问全部由仿真存储模块通过 DPI-C
自动完成：

  bnn_sim_cmd_stream  -- cmd_check()/cmd_get() --> 线性命令流
  bnn_sim_data_memory -- data_read()/data_write() --> 统一数据存储器

所有命令提交完成且核心/DMA 空闲后自动 $finish；核心返回 cmd_error 时
直接 $fatal。具体 DPI-C 存储模型由 C++ 侧实现。
============================================================================
*/
module bnn_sim_top (
    input  wire clock,
    input  wire reset
);
    localparam int unsigned DMA_DATA_WIDTH = 128;
    localparam int unsigned DMA_LEN_WIDTH  = 16;

    wire                             cmd_valid;
    wire [63:0]                      cmd_inst;
    wire                             cmd_ready;
    wire                             cmd_done;
    wire                             cmd_error;
    wire                             status_busy;

    wire                             mem_rd_cmd_valid;
    wire                             mem_rd_cmd_ready;
    wire [31:0]                      mem_rd_cmd_addr;
    wire [DMA_LEN_WIDTH-1:0]         mem_rd_cmd_len;
    wire                             mem_rd_data_valid;
    wire                             mem_rd_data_ready;
    wire [DMA_DATA_WIDTH-1:0]        mem_rd_data;
    wire                             mem_rd_data_last;

    wire                             mem_wr_cmd_valid;
    wire                             mem_wr_cmd_ready;
    wire [31:0]                      mem_wr_cmd_addr;
    wire [DMA_LEN_WIDTH-1:0]         mem_wr_cmd_len;
    wire                             mem_wr_data_valid;
    wire                             mem_wr_data_ready;
    wire [DMA_DATA_WIDTH-1:0]        mem_wr_data;
    wire [DMA_DATA_WIDTH/8-1:0]      mem_wr_data_strb;
    wire                             mem_wr_data_last;
    wire                             mem_wr_done;

    wire                             rd_busy;
    wire                             wr_busy;
    wire                             busy;
    wire                             done;
    wire                             error;

    assign busy = status_busy || rd_busy || wr_busy;

    bnn_sim_cmd_stream u_cmd_stream (
        .clock     (clock),
        .reset     (reset),
        .cmd_valid (cmd_valid),
        .cmd_inst  (cmd_inst),
        .cmd_ready (cmd_ready),
        .cmd_done  (cmd_done),
        .cmd_error (cmd_error),
        .core_busy (busy),
        .done      (done),
        .error     (error)
    );

    bnn_core_top #(
        .ADDR_WIDTH     (32),
        .DMA_DATA_WIDTH (DMA_DATA_WIDTH),
        .DMA_LEN_WIDTH  (DMA_LEN_WIDTH)
    ) u_core (
        .clock               (clock),
        .reset               (reset),
        .cmd_valid           (cmd_valid),
        .cmd_inst            (cmd_inst),
        .cmd_ready           (cmd_ready),
        .cmd_done            (cmd_done),
        .cmd_error           (cmd_error),
        .status_busy         (status_busy),
        .mem_rd_cmd_valid    (mem_rd_cmd_valid),
        .mem_rd_cmd_ready    (mem_rd_cmd_ready),
        .mem_rd_cmd_addr     (mem_rd_cmd_addr),
        .mem_rd_cmd_len      (mem_rd_cmd_len),
        .mem_rd_data_valid   (mem_rd_data_valid),
        .mem_rd_data_ready   (mem_rd_data_ready),
        .mem_rd_data         (mem_rd_data),
        .mem_rd_data_last    (mem_rd_data_last),
        .mem_wr_cmd_valid    (mem_wr_cmd_valid),
        .mem_wr_cmd_ready    (mem_wr_cmd_ready),
        .mem_wr_cmd_addr     (mem_wr_cmd_addr),
        .mem_wr_cmd_len      (mem_wr_cmd_len),
        .mem_wr_data_valid   (mem_wr_data_valid),
        .mem_wr_data_ready   (mem_wr_data_ready),
        .mem_wr_data         (mem_wr_data),
        .mem_wr_data_strb    (mem_wr_data_strb),
        .mem_wr_data_last    (mem_wr_data_last),
        .mem_wr_done         (mem_wr_done)
    );

    bnn_sim_data_memory #(
        .ADDR_WIDTH     (32),
        .DMA_DATA_WIDTH (DMA_DATA_WIDTH),
        .DMA_LEN_WIDTH  (DMA_LEN_WIDTH)
    ) u_data_memory (
        .clock          (clock),
        .reset          (reset),
        .rd_cmd_valid   (mem_rd_cmd_valid),
        .rd_cmd_ready   (mem_rd_cmd_ready),
        .rd_cmd_addr    (mem_rd_cmd_addr),
        .rd_cmd_len     (mem_rd_cmd_len),
        .rd_data_valid  (mem_rd_data_valid),
        .rd_data_ready  (mem_rd_data_ready),
        .rd_data        (mem_rd_data),
        .rd_data_last   (mem_rd_data_last),
        .wr_cmd_valid   (mem_wr_cmd_valid),
        .wr_cmd_ready   (mem_wr_cmd_ready),
        .wr_cmd_addr    (mem_wr_cmd_addr),
        .wr_cmd_len     (mem_wr_cmd_len),
        .wr_data_valid  (mem_wr_data_valid),
        .wr_data_ready  (mem_wr_data_ready),
        .wr_data        (mem_wr_data),
        .wr_data_strb   (mem_wr_data_strb),
        .wr_data_last   (mem_wr_data_last),
        .wr_done        (mem_wr_done),
        .rd_busy        (rd_busy),
        .wr_busy        (wr_busy)
    );

    always @(posedge clock) begin
        if (!reset) begin
            if (error)
                $fatal(1, "bnn_sim_top: core reported cmd_error");
            else if (done) begin
                $display("bnn_sim_top: all commands completed");
                $finish;
            end
        end
    end
endmodule

`default_nettype wire
