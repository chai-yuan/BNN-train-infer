`default_nettype none

/*
============================================================================
bnn_core_top (BNN v2 计算核心顶层封装)
----------------------------------------------------------------------------
模块职责与边界：
将核心内部 6 个功能子模块、片上 Weight SRAM 以及内部 DMA 读通道多路选择器组装为完整、
独立于具体外部总线协议（AXI4 / Wishbone 等）的计算核心顶层。

对外仅暴露三组极简标准接口：
1. 宿主机单阶段 64-bit 原子命令与状态接口 (cmd_* / status_busy)
2. 统一外部内存突发读接口 (mem_rd_cmd_* / mem_rd_data_*) —— 128-bit 流式数据
3. 统一外部内存突发写接口 (mem_wr_cmd_* / mem_wr_data_* / mem_wr_done) —— 128-bit 带字节掩码

内部拓扑结构：
                    +-------------------------------------------+
   cmd_* / status <-|            bnn_top_controller             |
                    +-----+---------+---------+---------+-------+
                          |         |         |         |
         +----------------+   +-----+   +-----+---+     +----------------+
         v                    v         v         v                      v
+-------------------+  +-----------------------+  +------------+  +-------------------+
| bnn_weight_loader |  |bnn_input_quant_buffer |  |            |  | bnn_output_writer |
+--------+----------+  +-----------+-----------+  |            |  +---------+---------+
         |  \                   /  |              |            |            ^     |
         |   \  (dma_rd_mux)   /   | idma_*       |            |  osram_*   |     | mem_wr_*
         |    +-------+-------+    | (+scale/deq) |            |            |     v
         v            ^            v              v            |            |   (外部写)
  [Weight SRAM]   mem_rd_*   +--------------------+            |            |
   (128x144b)     (外部读)   |   bnn_core_array   |            |            |
         |                   | (4x bnn_core_unit) |            |            |
         +------------------>+---------+----------+            |            |
              sram_rdata               | odma_data (4xFP32)    |            |
                                       v                       |            |
                             +--------------------+            |            |
                             |bnn_output_post_sram|<-----------+            |
                             |  (4096x128b SDP)   |-------------------------+
                             +--------------------+

内部存储与仲裁约定：
- Weight SRAM (128 x 144-bit) 封装于本顶层内部。bnn_top_controller 保证 OP_LOAD_WEIGHT
  写操作（在 !array_enable && !array_busy 下执行）与 OP_COMPUTE 读操作严格互斥。
- DMA 读通道 MUX 由 dma_rd_mux_sel 静态切换（0：bnn_weight_loader；1：bnn_input_quant_buffer），
  未选通一侧的 cmd_ready 与 data_valid 均被门控拉低。
============================================================================
*/
module bnn_core_top #(
    parameter int unsigned ADDR_WIDTH         = 32,
    parameter int unsigned STRIDE_WIDTH       = 32,
    parameter int unsigned DMA_DATA_WIDTH     = 128,
    parameter int unsigned DMA_LEN_WIDTH      = 16,
    parameter int unsigned IDMA_INT8_NUM      = 128,
    parameter int unsigned WEIGHT_WIDTH       = IDMA_INT8_NUM + 16,
    parameter int unsigned ODMA_FP32_NUM      = 4,
    parameter int unsigned MAX_ROWS           = 128,
    parameter int unsigned WSRAM_ADDR_WIDTH   = 7,
    parameter int unsigned WSRAM_READ_LATENCY = 1,
    parameter logic [WSRAM_ADDR_WIDTH-1:0] WEIGHT_BASE_ADDR = '0,
    parameter int unsigned MAX_WEIGHT_GROUPS  = (1 << WSRAM_ADDR_WIDTH) / ODMA_FP32_NUM,
    parameter int unsigned OSRAM_READ_LATENCY = 1
) (
    input  wire                             clock,
    input  wire                             reset,

    // ------------------------------------------------------------------------
    // 1. 宿主机 64-bit 原子指令与状态接口
    // ------------------------------------------------------------------------
    input  wire                             cmd_valid,
    input  wire [63:0]                      cmd_inst,
    output wire                             cmd_ready,
    output wire                             cmd_done,
    output wire                             cmd_error,
    output wire                             status_busy,

    // ------------------------------------------------------------------------
    // 2. 统一外部内存 DMA 读接口（连接外层 AXI4 读适配引擎）
    // ------------------------------------------------------------------------
    output wire                             mem_rd_cmd_valid,
    input  wire                             mem_rd_cmd_ready,
    output wire [ADDR_WIDTH-1:0]            mem_rd_cmd_addr,
    output wire [DMA_LEN_WIDTH-1:0]         mem_rd_cmd_len,

    input  wire                             mem_rd_data_valid,
    output wire                             mem_rd_data_ready,
    input  wire [DMA_DATA_WIDTH-1:0]        mem_rd_data,
    input  wire                             mem_rd_data_last,

    // ------------------------------------------------------------------------
    // 3. 统一外部内存 DMA 写接口（连接外层 AXI4 写适配引擎）
    // ------------------------------------------------------------------------
    output wire                             mem_wr_cmd_valid,
    input  wire                             mem_wr_cmd_ready,
    output wire [ADDR_WIDTH-1:0]            mem_wr_cmd_addr,
    output wire [DMA_LEN_WIDTH-1:0]         mem_wr_cmd_len,

    output wire                             mem_wr_data_valid,
    input  wire                             mem_wr_data_ready,
    output wire [DMA_DATA_WIDTH-1:0]        mem_wr_data,
    output wire [(DMA_DATA_WIDTH/8)-1:0]    mem_wr_data_strb,
    output wire                             mem_wr_data_last,

    input  wire                             mem_wr_done
);
    localparam int unsigned MAX_OUT_CHANS     = MAX_WEIGHT_GROUPS * ODMA_FP32_NUM;
    localparam int unsigned ROW_COUNT_WIDTH   = $clog2(MAX_ROWS + 1);
    localparam int unsigned GROUP_COUNT_WIDTH = $clog2(MAX_WEIGHT_GROUPS + 1);
    localparam int unsigned CHAN_COUNT_WIDTH  = $clog2(MAX_OUT_CHANS + 1);
    localparam int unsigned OSRAM_DEPTH       = MAX_ROWS * MAX_WEIGHT_GROUPS;
    localparam int unsigned OSRAM_ADDR_WIDTH  = $clog2(OSRAM_DEPTH);
    localparam int unsigned BEAT_COUNT_WIDTH  = $clog2(OSRAM_DEPTH + 1);
    localparam int unsigned WSRAM_DEPTH       = 1 << WSRAM_ADDR_WIDTH;

    // ------------------------------------------------------------------------
    // 控制器 <-> 各子模块控制与状态连线
    // ------------------------------------------------------------------------
    wire                             dma_rd_mux_sel;

    wire                             wloader_start;
    wire                             wloader_busy;
    wire                             wloader_done;
    wire [ADDR_WIDTH-1:0]            wloader_base_addr;
    wire [CHAN_COUNT_WIDTH-1:0]      wloader_weight_rows_n;

    wire                             array_enable;
    wire                             array_busy;
    wire                             array_weight_cfg_valid;
    wire [GROUP_COUNT_WIDTH-1:0]     array_weight_cfg_groups;

    wire                             iqbuf_start;
    wire                             iqbuf_busy;
    wire                             iqbuf_done;
    wire                             iqbuf_input_is_int8;
    wire [ADDR_WIDTH-1:0]            iqbuf_base_addr;
    wire [STRIDE_WIDTH-1:0]          iqbuf_row_stride;
    wire [ROW_COUNT_WIDTH-1:0]       iqbuf_row_num_m;

    wire                             opost_session_start;
    wire                             opost_session_busy;
    wire                             opost_session_done;
    wire                             opost_accum_enable;
    wire [BEAT_COUNT_WIDTH-1:0]      opost_total_beats;

    wire                             owriter_start;
    wire                             owriter_busy;
    wire                             owriter_done;
    wire [ADDR_WIDTH-1:0]            owriter_base_addr;
    wire [STRIDE_WIDTH-1:0]          owriter_row_stride;
    wire [ROW_COUNT_WIDTH-1:0]       owriter_row_num_m;
    wire [CHAN_COUNT_WIDTH-1:0]      owriter_weight_rows_n;

    // ------------------------------------------------------------------------
    // DMA 读多路选择器信号 (0: wloader, 1: iqbuf)
    // ------------------------------------------------------------------------
    wire                             wloader_rd_cmd_valid;
    wire                             wloader_rd_cmd_ready;
    wire [ADDR_WIDTH-1:0]            wloader_rd_cmd_addr;
    wire [DMA_LEN_WIDTH-1:0]         wloader_rd_cmd_len;
    wire                             wloader_rd_data_valid;
    wire                             wloader_rd_data_ready;

    wire                             iqbuf_rd_cmd_valid;
    wire                             iqbuf_rd_cmd_ready;
    wire [ADDR_WIDTH-1:0]            iqbuf_rd_cmd_addr;
    wire [DMA_LEN_WIDTH-1:0]         iqbuf_rd_cmd_len;
    wire                             iqbuf_rd_data_valid;
    wire                             iqbuf_rd_data_ready;

    assign mem_rd_cmd_valid      = dma_rd_mux_sel ? iqbuf_rd_cmd_valid  : wloader_rd_cmd_valid;
    assign mem_rd_cmd_addr       = dma_rd_mux_sel ? iqbuf_rd_cmd_addr   : wloader_rd_cmd_addr;
    assign mem_rd_cmd_len        = dma_rd_mux_sel ? iqbuf_rd_cmd_len    : wloader_rd_cmd_len;
    assign wloader_rd_cmd_ready  = !dma_rd_mux_sel && mem_rd_cmd_ready;
    assign iqbuf_rd_cmd_ready    =  dma_rd_mux_sel && mem_rd_cmd_ready;

    assign wloader_rd_data_valid = !dma_rd_mux_sel && mem_rd_data_valid;
    assign iqbuf_rd_data_valid   =  dma_rd_mux_sel && mem_rd_data_valid;
    assign mem_rd_data_ready     = dma_rd_mux_sel ? iqbuf_rd_data_ready : wloader_rd_data_ready;

    // ------------------------------------------------------------------------
    // 片上 Weight SRAM (1R1W, 读写由控制器保证时分互斥)
    // ------------------------------------------------------------------------
    wire                             wsram_wen;
    wire [WSRAM_ADDR_WIDTH-1:0]      wsram_waddr;
    wire [WEIGHT_WIDTH-1:0]          wsram_wdata;
    wire                             wsram_ren;
    wire [WSRAM_ADDR_WIDTH-1:0]      wsram_raddr;
    wire [WEIGHT_WIDTH-1:0]          wsram_rdata;

    logic [WEIGHT_WIDTH-1:0]         wsram_mem  [0:WSRAM_DEPTH-1];
    logic [WEIGHT_WIDTH-1:0]         wsram_pipe [0:WSRAM_READ_LATENCY-1];

    always_ff @(posedge clock) begin
        if (wsram_wen)
            wsram_mem[wsram_waddr] <= wsram_wdata;
        if (wsram_ren)
            wsram_pipe[0] <= wsram_mem[wsram_raddr];
        for (int stage = 1; stage < WSRAM_READ_LATENCY; stage = stage + 1)
            wsram_pipe[stage] <= wsram_pipe[stage-1];
    end

    assign wsram_rdata = wsram_pipe[WSRAM_READ_LATENCY-1];

    // ------------------------------------------------------------------------
    // 数据流内部通路连线
    // ------------------------------------------------------------------------
    wire [(IDMA_INT8_NUM*8)-1:0]     idma_data;
    wire [31:0]                      idma_scale;
    wire                             idma_dequant_en;
    wire                             idma_valid;
    wire                             idma_ready;

    wire [(ODMA_FP32_NUM*32)-1:0]    odma_data;
    wire                             odma_row_last; // 供波形与断言观测
    wire                             odma_valid;
    wire                             odma_ready;
    wire                             _unused_odma_row_last = &{1'b0, odma_row_last};

    wire                             osram_ren;
    wire [OSRAM_ADDR_WIDTH-1:0]      osram_raddr;
    wire [(ODMA_FP32_NUM*32)-1:0]    osram_rdata;

    // ------------------------------------------------------------------------
    // 子模块实例化
    // ------------------------------------------------------------------------
    bnn_top_controller #(
        .ADDR_WIDTH        (ADDR_WIDTH),
        .STRIDE_WIDTH      (STRIDE_WIDTH),
        .ODMA_FP32_NUM     (ODMA_FP32_NUM),
        .MAX_ROWS          (MAX_ROWS),
        .MAX_WEIGHT_GROUPS (MAX_WEIGHT_GROUPS),
        .MAX_OUT_CHANS     (MAX_OUT_CHANS),
        .ROW_COUNT_WIDTH   (ROW_COUNT_WIDTH),
        .GROUP_COUNT_WIDTH (GROUP_COUNT_WIDTH),
        .CHAN_COUNT_WIDTH  (CHAN_COUNT_WIDTH),
        .OSRAM_DEPTH       (OSRAM_DEPTH),
        .BEAT_COUNT_WIDTH  (BEAT_COUNT_WIDTH)
    ) u_top_controller (
        .clock                   (clock),
        .reset                   (reset),
        .cmd_valid               (cmd_valid),
        .cmd_inst                (cmd_inst),
        .cmd_ready               (cmd_ready),
        .cmd_done                (cmd_done),
        .cmd_error               (cmd_error),
        .status_busy             (status_busy),
        .dma_rd_mux_sel          (dma_rd_mux_sel),
        .wloader_start           (wloader_start),
        .wloader_busy            (wloader_busy),
        .wloader_done            (wloader_done),
        .wloader_base_addr       (wloader_base_addr),
        .wloader_weight_rows_n   (wloader_weight_rows_n),
        .array_enable            (array_enable),
        .array_busy              (array_busy),
        .array_weight_cfg_valid  (array_weight_cfg_valid),
        .array_weight_cfg_groups (array_weight_cfg_groups),
        .iqbuf_start             (iqbuf_start),
        .iqbuf_busy              (iqbuf_busy),
        .iqbuf_done              (iqbuf_done),
        .iqbuf_input_is_int8     (iqbuf_input_is_int8),
        .iqbuf_base_addr         (iqbuf_base_addr),
        .iqbuf_row_stride        (iqbuf_row_stride),
        .iqbuf_row_num_m         (iqbuf_row_num_m),
        .opost_session_start     (opost_session_start),
        .opost_session_busy      (opost_session_busy),
        .opost_session_done      (opost_session_done),
        .opost_accum_enable      (opost_accum_enable),
        .opost_total_beats       (opost_total_beats),
        .owriter_start           (owriter_start),
        .owriter_busy            (owriter_busy),
        .owriter_done            (owriter_done),
        .owriter_base_addr       (owriter_base_addr),
        .owriter_row_stride      (owriter_row_stride),
        .owriter_row_num_m       (owriter_row_num_m),
        .owriter_weight_rows_n   (owriter_weight_rows_n)
    );

    bnn_weight_loader #(
        .ADDR_WIDTH        (ADDR_WIDTH),
        .DMA_DATA_WIDTH    (DMA_DATA_WIDTH),
        .WEIGHT_WIDTH      (WEIGHT_WIDTH),
        .ODMA_FP32_NUM     (ODMA_FP32_NUM),
        .SRAM_ADDR_WIDTH   (WSRAM_ADDR_WIDTH),
        .WEIGHT_BASE_ADDR  (WEIGHT_BASE_ADDR),
        .DMA_LEN_WIDTH     (DMA_LEN_WIDTH),
        .MAX_WEIGHT_GROUPS (MAX_WEIGHT_GROUPS),
        .MAX_OUT_CHANS     (MAX_OUT_CHANS),
        .CHAN_COUNT_WIDTH  (CHAN_COUNT_WIDTH)
    ) u_weight_loader (
        .clock             (clock),
        .reset             (reset),
        .start             (wloader_start),
        .busy              (wloader_busy),
        .done              (wloader_done),
        .cfg_src_base_addr (wloader_base_addr),
        .cfg_weight_rows_n (wloader_weight_rows_n),
        .mem_rd_cmd_valid  (wloader_rd_cmd_valid),
        .mem_rd_cmd_ready  (wloader_rd_cmd_ready),
        .mem_rd_cmd_addr   (wloader_rd_cmd_addr),
        .mem_rd_cmd_len    (wloader_rd_cmd_len),
        .mem_rd_data_valid (wloader_rd_data_valid),
        .mem_rd_data_ready (wloader_rd_data_ready),
        .mem_rd_data       (mem_rd_data),
        .mem_rd_data_last  (mem_rd_data_last),
        .wsram_wen         (wsram_wen),
        .wsram_waddr       (wsram_waddr),
        .wsram_wdata       (wsram_wdata)
    );

    bnn_input_quant_buffer #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .STRIDE_WIDTH    (STRIDE_WIDTH),
        .DMA_DATA_WIDTH  (DMA_DATA_WIDTH),
        .IDMA_INT8_NUM   (IDMA_INT8_NUM),
        .MAX_ROWS        (MAX_ROWS),
        .ROW_COUNT_WIDTH (ROW_COUNT_WIDTH),
        .DMA_LEN_WIDTH   (DMA_LEN_WIDTH)
    ) u_input_quant_buffer (
        .clock              (clock),
        .reset              (reset),
        .start              (iqbuf_start),
        .busy               (iqbuf_busy),
        .done               (iqbuf_done),
        .cfg_input_is_int8  (iqbuf_input_is_int8),
        .cfg_src_base_addr  (iqbuf_base_addr),
        .cfg_src_row_stride (iqbuf_row_stride),
        .cfg_row_num_m      (iqbuf_row_num_m),
        .mem_rd_cmd_valid   (iqbuf_rd_cmd_valid),
        .mem_rd_cmd_ready   (iqbuf_rd_cmd_ready),
        .mem_rd_cmd_addr    (iqbuf_rd_cmd_addr),
        .mem_rd_cmd_len     (iqbuf_rd_cmd_len),
        .mem_rd_data_valid  (iqbuf_rd_data_valid),
        .mem_rd_data_ready  (iqbuf_rd_data_ready),
        .mem_rd_data        (mem_rd_data),
        .mem_rd_data_last   (mem_rd_data_last),
        .idma_data          (idma_data),
        .idma_scale         (idma_scale),
        .idma_dequant_en    (idma_dequant_en),
        .idma_valid         (idma_valid),
        .idma_ready         (idma_ready)
    );

    bnn_core_array #(
        .WEIGHT_WIDTH      (WEIGHT_WIDTH),
        .IDMA_INT8_NUM     (IDMA_INT8_NUM),
        .ODMA_FP32_NUM     (ODMA_FP32_NUM),
        .SRAM_ADDR_WIDTH   (WSRAM_ADDR_WIDTH),
        .SRAM_READ_LATENCY (WSRAM_READ_LATENCY),
        .WEIGHT_BASE_ADDR  (WEIGHT_BASE_ADDR),
        .MAX_WEIGHT_GROUPS (MAX_WEIGHT_GROUPS),
        .GROUP_COUNT_WIDTH (GROUP_COUNT_WIDTH)
    ) u_core_array (
        .clock             (clock),
        .reset             (reset),
        .enable            (array_enable),
        .busy              (array_busy),
        .weight_cfg_valid  (array_weight_cfg_valid),
        .weight_cfg_groups (array_weight_cfg_groups),
        .sram_en           (wsram_ren),
        .sram_addr         (wsram_raddr),
        .sram_rdata        (wsram_rdata),
        .idma_data         (idma_data),
        .idma_scale        (idma_scale),
        .idma_dequant_en   (idma_dequant_en),
        .idma_valid        (idma_valid),
        .idma_ready        (idma_ready),
        .odma_data         (odma_data),
        .odma_row_last     (odma_row_last),
        .odma_valid        (odma_valid),
        .odma_ready        (odma_ready)
    );

    bnn_output_post_sram #(
        .ODMA_FP32_NUM      (ODMA_FP32_NUM),
        .MAX_ROWS           (MAX_ROWS),
        .MAX_WEIGHT_GROUPS  (MAX_WEIGHT_GROUPS),
        .OSRAM_DEPTH        (OSRAM_DEPTH),
        .OSRAM_ADDR_WIDTH   (OSRAM_ADDR_WIDTH),
        .BEAT_COUNT_WIDTH   (BEAT_COUNT_WIDTH),
        .OSRAM_READ_LATENCY (OSRAM_READ_LATENCY)
    ) u_output_post_sram (
        .clock            (clock),
        .reset            (reset),
        .session_start    (opost_session_start),
        .session_busy     (opost_session_busy),
        .session_done     (opost_session_done),
        .cfg_accum_enable (opost_accum_enable),
        .cfg_total_beats  (opost_total_beats),
        .odma_data        (odma_data),
        .odma_valid       (odma_valid),
        .odma_ready       (odma_ready),
        .osram_ren        (osram_ren),
        .osram_raddr      (osram_raddr),
        .osram_rdata      (osram_rdata)
    );

    bnn_output_writer #(
        .ADDR_WIDTH         (ADDR_WIDTH),
        .STRIDE_WIDTH       (STRIDE_WIDTH),
        .DMA_DATA_WIDTH     (DMA_DATA_WIDTH),
        .ODMA_FP32_NUM      (ODMA_FP32_NUM),
        .MAX_ROWS           (MAX_ROWS),
        .MAX_WEIGHT_GROUPS  (MAX_WEIGHT_GROUPS),
        .MAX_OUT_CHANS      (MAX_OUT_CHANS),
        .ROW_COUNT_WIDTH    (ROW_COUNT_WIDTH),
        .CHAN_COUNT_WIDTH   (CHAN_COUNT_WIDTH),
        .OSRAM_DEPTH        (OSRAM_DEPTH),
        .OSRAM_ADDR_WIDTH   (OSRAM_ADDR_WIDTH),
        .OSRAM_READ_LATENCY (OSRAM_READ_LATENCY),
        .DMA_LEN_WIDTH      (DMA_LEN_WIDTH)
    ) u_output_writer (
        .clock              (clock),
        .reset              (reset),
        .start              (owriter_start),
        .busy               (owriter_busy),
        .done               (owriter_done),
        .cfg_dst_base_addr  (owriter_base_addr),
        .cfg_dst_row_stride (owriter_row_stride),
        .cfg_row_num_m      (owriter_row_num_m),
        .cfg_weight_rows_n  (owriter_weight_rows_n),
        .osram_ren          (osram_ren),
        .osram_raddr        (osram_raddr),
        .osram_rdata        (osram_rdata),
        .mem_wr_cmd_valid   (mem_wr_cmd_valid),
        .mem_wr_cmd_ready   (mem_wr_cmd_ready),
        .mem_wr_cmd_addr    (mem_wr_cmd_addr),
        .mem_wr_cmd_len     (mem_wr_cmd_len),
        .mem_wr_data_valid  (mem_wr_data_valid),
        .mem_wr_data_ready  (mem_wr_data_ready),
        .mem_wr_data        (mem_wr_data),
        .mem_wr_data_strb   (mem_wr_data_strb),
        .mem_wr_data_last   (mem_wr_data_last),
        .mem_wr_done        (mem_wr_done)
    );

`ifndef SYNTHESIS
    initial begin
        assert (IDMA_INT8_NUM > 0)
            else $fatal(1, "IDMA_INT8_NUM must be positive");
        assert (WEIGHT_WIDTH == IDMA_INT8_NUM + 16)
            else $fatal(1, "WEIGHT_WIDTH must equal IDMA_INT8_NUM + 16");
        assert (ODMA_FP32_NUM > 0 && DMA_DATA_WIDTH == ODMA_FP32_NUM * 32)
            else $fatal(1, "DMA_DATA_WIDTH must equal ODMA_FP32_NUM * 32");
        assert (WSRAM_READ_LATENCY >= 1 && OSRAM_READ_LATENCY >= 1)
            else $fatal(1, "SRAM read latencies must be >= 1");
        assert (MAX_WEIGHT_GROUPS > 0 &&
                (int'(WEIGHT_BASE_ADDR) + MAX_WEIGHT_GROUPS * ODMA_FP32_NUM <= WSRAM_DEPTH))
            else $fatal(1, "Weight groups exceed Weight SRAM capacity");
    end

    always @(posedge clock) begin
        if (!reset) begin
            assert (!(wsram_wen && wsram_ren))
                else $fatal(1, "Weight SRAM concurrent read and write violation");
        end
    end
`endif

endmodule

`default_nettype wire
