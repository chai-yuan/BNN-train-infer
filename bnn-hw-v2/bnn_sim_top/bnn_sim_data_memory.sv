`default_nettype none

/*
============================================================================
bnn_sim_data_memory (统一数据存储器 DPI-C 适配层)
----------------------------------------------------------------------------
- 数据存储由 C++ DPI-C 实现，本模块只实现核心所需的 DMA 读写时序。
- DPI 数据宽度固定为 64 bit，地址为字节地址，小端序。
- 契约：
    longint unsigned data_read (longint unsigned address);
    void data_write(longint unsigned address, longint unsigned data,
                    byte unsigned strobe);
  strobe 的 bit i 对应 data 的第 i 个字节。
- 核心 DMA 一拍默认 128 bit，本模块把每拍拆成两次 64-bit DPI 访问；
  若 DMA_DATA_WIDTH == 64，则每拍一次访问。
- 读通道：命令握手后先通过 data_read_burst() 一次预取整笔突发，随后逐拍读取并
  返回 128-bit 数据，末拍 rd_data_last 为 1。离线 testbench 可将预取实现为空操作，
  仍由 data_read() 直接访问本地存储；网络化 testbench 用预取把整笔突发压缩成一次
  远端事务。预取调用阻塞期间仿真时钟不推进，故 DMA 时序与周期数不变。
- 写通道：命令握手后逐拍接收数据并写入，末拍握手后的下一拍输出单周期 wr_done。
- 地址必须按拍对齐，长度不能为 0；违反时直接 $fatal。
============================================================================
*/
module bnn_sim_data_memory #(
    parameter int unsigned ADDR_WIDTH     = 32,
    parameter int unsigned DMA_DATA_WIDTH = 128,
    parameter int unsigned DMA_LEN_WIDTH  = 16
) (
    input  wire                         clock,
    input  wire                         reset,

    input  wire                         rd_cmd_valid,
    output wire                         rd_cmd_ready,
    input  wire [ADDR_WIDTH-1:0]        rd_cmd_addr,
    input  wire [DMA_LEN_WIDTH-1:0]     rd_cmd_len,
    output logic                        rd_data_valid,
    input  wire                         rd_data_ready,
    output logic [DMA_DATA_WIDTH-1:0]   rd_data,
    output logic                        rd_data_last,

    input  wire                         wr_cmd_valid,
    output wire                         wr_cmd_ready,
    input  wire [ADDR_WIDTH-1:0]        wr_cmd_addr,
    input  wire [DMA_LEN_WIDTH-1:0]     wr_cmd_len,
    input  wire                         wr_data_valid,
    output wire                         wr_data_ready,
    input  wire [DMA_DATA_WIDTH-1:0]    wr_data,
    input  wire [DMA_DATA_WIDTH/8-1:0]  wr_data_strb,
    input  wire                         wr_data_last,
    output logic                        wr_done,

    output wire                         rd_busy,
    output wire                         wr_busy
);
    import "DPI-C" function longint unsigned data_read(
        input longint unsigned address);
    import "DPI-C" function void data_read_burst(
        input longint unsigned address, input longint unsigned bytes);
    import "DPI-C" function void data_write(
        input longint unsigned address, input longint unsigned data,
        input byte unsigned strobe);

    localparam int unsigned BEAT_BYTES = DMA_DATA_WIDTH / 8;
    localparam int unsigned WORDS64    = DMA_DATA_WIDTH / 64;

    typedef enum logic {RD_IDLE, RD_DATA} rd_state_t;
    typedef enum logic {WR_IDLE, WR_DATA} wr_state_t;

    rd_state_t rd_state;
    logic [63:0]            rd_address;
    logic [DMA_LEN_WIDTH-1:0] rd_remaining;

    wr_state_t wr_state;
    logic [63:0]            wr_address;
    logic [DMA_LEN_WIDTH-1:0] wr_remaining;

    assign rd_cmd_ready  = !reset && (rd_state == RD_IDLE);
    assign rd_busy       = !reset && (rd_state != RD_IDLE);
    assign wr_cmd_ready  = !reset && (wr_state == WR_IDLE) && !wr_done;
    assign wr_data_ready = !reset && (wr_state == WR_DATA);
    assign wr_busy       = !reset && ((wr_state != WR_IDLE) || wr_done);

    task automatic present_read(input logic [63:0] address, input logic last);
        logic [DMA_DATA_WIDTH-1:0] packed_data;
        for (int w = 0; w < WORDS64; w++)
            packed_data[64*w +: 64] = 64'(data_read(address + 64'(8*w)));
        rd_data       <= packed_data;
        rd_data_last  <= last;
        rd_data_valid <= 1'b1;
    endtask

    always_ff @(posedge clock) begin
        if (reset) begin
            rd_state      <= RD_IDLE;
            rd_address    <= '0;
            rd_remaining  <= '0;
            rd_data_valid <= 1'b0;
            rd_data       <= '0;
            rd_data_last  <= 1'b0;
        end else begin
            case (rd_state)
                RD_IDLE: if (rd_cmd_valid && rd_cmd_ready) begin
                    if (rd_cmd_len == 0)
                        $fatal(1, "DMA read command has zero beats");
                    if (64'(rd_cmd_addr) % 64'(BEAT_BYTES) != 0)
                        $fatal(1, "DMA read address is not beat aligned");
                    data_read_burst(64'(rd_cmd_addr),
                                    64'(rd_cmd_len) * 64'(BEAT_BYTES));
                    rd_address   <= 64'(rd_cmd_addr);
                    rd_remaining <= rd_cmd_len;
                    rd_state     <= RD_DATA;
                end
                RD_DATA: begin
                    if (!rd_data_valid) begin
                        present_read(rd_address, rd_remaining == 1);
                    end else if (rd_data_ready) begin
                        if (rd_remaining == 1) begin
                            rd_data_valid <= 1'b0;
                            rd_data_last  <= 1'b0;
                            rd_state      <= RD_IDLE;
                        end else begin
                            rd_address   <= rd_address + 64'(BEAT_BYTES);
                            rd_remaining <= rd_remaining - 1'b1;
                            present_read(rd_address + 64'(BEAT_BYTES),
                                         rd_remaining == 2);
                        end
                    end
                end
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    always_ff @(posedge clock) begin
        if (reset) begin
            wr_state     <= WR_IDLE;
            wr_address   <= '0;
            wr_remaining <= '0;
            wr_done      <= 1'b0;
        end else begin
            wr_done <= 1'b0;
            case (wr_state)
                WR_IDLE: if (wr_cmd_valid && wr_cmd_ready) begin
                    if (wr_cmd_len == 0)
                        $fatal(1, "DMA write command has zero beats");
                    if (64'(wr_cmd_addr) % 64'(BEAT_BYTES) != 0)
                        $fatal(1, "DMA write address is not beat aligned");
                    wr_address   <= 64'(wr_cmd_addr);
                    wr_remaining <= wr_cmd_len;
                    wr_state     <= WR_DATA;
                end
                WR_DATA: if (wr_data_valid && wr_data_ready) begin
                    if (wr_data_last != (wr_remaining == 1))
                        $fatal(1, "DMA write: last disagrees with remaining beats");
                    for (int w = 0; w < WORDS64; w++)
                        data_write(wr_address + 64'(8*w),
                                   wr_data[64*w +: 64],
                                   8'(wr_data_strb[8*w +: 8]));
                    if (wr_remaining == 1) begin
                        wr_done      <= 1'b1;
                        wr_state     <= WR_IDLE;
                        wr_remaining <= '0;
                    end else begin
                        wr_address   <= wr_address + 64'(BEAT_BYTES);
                        wr_remaining <= wr_remaining - 1'b1;
                    end
                end
                default: wr_state <= WR_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (DMA_DATA_WIDTH >= 64 && DMA_DATA_WIDTH % 64 == 0)
            else $fatal(1, "DMA data width must be a positive multiple of 64");
    end
`endif
endmodule

`default_nettype wire
