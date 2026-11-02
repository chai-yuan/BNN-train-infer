`default_nettype none

// Simulation-only DMA adapter. len is a beat count, NOT AXI AxLEN.
// The read and write channels address independent C++ byte stores via DPI-C.
module bnn_dma_memory #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DMA_DATA_WIDTH = 128,
    parameter int unsigned DMA_LEN_WIDTH = 16,
    parameter int READ_MEMORY_ID = 0,
    parameter int WRITE_MEMORY_ID = 1
) (
    input  wire clock,
    input  wire reset,
    input  wire [31:0] cfg_rd_latency,
    input  wire [31:0] cfg_rd_gap,
    input  wire [31:0] cfg_wr_latency,
    input  wire [31:0] cfg_cmd_stall_period,
    input  wire [31:0] cfg_wr_stall_period,

    input  wire mem_rd_cmd_valid,
    output wire mem_rd_cmd_ready,
    input  wire [ADDR_WIDTH-1:0] mem_rd_cmd_addr,
    input  wire [DMA_LEN_WIDTH-1:0] mem_rd_cmd_len,
    output logic mem_rd_data_valid,
    input  wire mem_rd_data_ready,
    output logic [DMA_DATA_WIDTH-1:0] mem_rd_data,
    output logic mem_rd_data_last,

    input  wire mem_wr_cmd_valid,
    output wire mem_wr_cmd_ready,
    input  wire [ADDR_WIDTH-1:0] mem_wr_cmd_addr,
    input  wire [DMA_LEN_WIDTH-1:0] mem_wr_cmd_len,
    input  wire mem_wr_data_valid,
    output wire mem_wr_data_ready,
    input  wire [DMA_DATA_WIDTH-1:0] mem_wr_data,
    input  wire [DMA_DATA_WIDTH/8-1:0] mem_wr_data_strb,
    input  wire mem_wr_data_last,
    output logic mem_wr_done,
    output wire rd_busy,
    output wire wr_busy
);
    localparam int unsigned BEAT_BYTES = DMA_DATA_WIDTH / 8;
    import "DPI-C" function int bnn_mem_check(
        input int id, input longint unsigned address, input longint unsigned size);
    import "DPI-C" function int bnn_mem_read(
        input int id, input longint unsigned address, output byte unsigned data[]);
    import "DPI-C" function int bnn_mem_write(
        input int id, input longint unsigned address,
        input byte unsigned data[], input byte unsigned strobe[]);

    initial begin
        if (ADDR_WIDTH < 1 || ADDR_WIDTH > 64 || DMA_LEN_WIDTH < 1 || DMA_LEN_WIDTH > 32)
            $fatal(1, "unsupported DMA address/length width");
        if (DMA_DATA_WIDTH < 8 || DMA_DATA_WIDTH % 8 != 0)
            $fatal(1, "DMA data width must be a positive multiple of 8");
    end

    logic [63:0] cycle_count;
    logic rd_active;
    logic [63:0] rd_address, wr_address;
    logic [DMA_LEN_WIDTH-1:0] rd_remaining, wr_remaining;
    logic [31:0] rd_wait, wr_wait;
    typedef enum logic [1:0] {WR_IDLE, WR_DATA, WR_RESPONSE} wr_state_t;
    wr_state_t wr_state;

    wire cmd_stall = (cfg_cmd_stall_period != 0) &&
                     (cycle_count % 64'(cfg_cmd_stall_period) == 64'(cfg_cmd_stall_period) - 1);
    wire wr_stall = (cfg_wr_stall_period != 0) &&
                    (cycle_count % 64'(cfg_wr_stall_period) == 64'(cfg_wr_stall_period) - 1);
    assign mem_rd_cmd_ready = !reset && !rd_active && !cmd_stall;
    assign mem_wr_cmd_ready = !reset && wr_state == WR_IDLE && !mem_wr_done && !cmd_stall;
    assign mem_wr_data_ready = !reset && wr_state == WR_DATA && !wr_stall;
    assign rd_busy = rd_active;
    assign wr_busy = wr_state != WR_IDLE || mem_wr_done;

    task automatic check_command(input int id, input logic [ADDR_WIDTH-1:0] address,
                                 input logic [DMA_LEN_WIDTH-1:0] beats);
        if (beats == 0)
            $fatal(1, "DMA memory %0d: zero-length command", id);
        if (64'(address) % 64'(BEAT_BYTES) != 0)
            $fatal(1, "DMA memory %0d: unaligned address 0x%h", id, address);
        // Check the entire burst before performing any accesses, including masked tails.
        if (bnn_mem_check(id, 64'(address), 64'(beats) * 64'(BEAT_BYTES)) != 0)
            $fatal(1, "DMA memory %0d: burst range check failed (see DPI error)", id);
    endtask

    task automatic present_read(input logic [63:0] address, input logic last);
        byte unsigned bytes_out [0:BEAT_BYTES-1];
        logic [DMA_DATA_WIDTH-1:0] packed_data;
        if (bnn_mem_read(READ_MEMORY_ID, address, bytes_out) != 0)
            $fatal(1, "DMA read failed at 0x%h (see DPI error)", address);
        for (int b = 0; b < BEAT_BYTES; b++)
            packed_data[8*b +: 8] = bytes_out[b];
        mem_rd_data <= packed_data;
        mem_rd_data_last <= last;
        mem_rd_data_valid <= 1'b1;
    endtask

    always_ff @(posedge clock) begin
        if (reset) begin
            cycle_count <= 0;
            rd_active <= 0;
            rd_address <= 0;
            rd_remaining <= 0;
            rd_wait <= 0;
            mem_rd_data_valid <= 0;
            mem_rd_data <= 0;
            mem_rd_data_last <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (mem_rd_cmd_valid && mem_rd_cmd_ready) begin
                check_command(READ_MEMORY_ID, mem_rd_cmd_addr, mem_rd_cmd_len);
                rd_active <= 1;
                rd_address <= 64'(mem_rd_cmd_addr);
                rd_remaining <= mem_rd_cmd_len;
                rd_wait <= cfg_rd_latency;
            end
            if (rd_active) begin
                if (mem_rd_data_valid) begin
                    if (mem_rd_data_ready) begin
                        if (rd_remaining == 1) begin
                            rd_active <= 0;
                            mem_rd_data_valid <= 0;
                            mem_rd_data_last <= 0;
                        end else begin
                            rd_address <= rd_address + 64'(BEAT_BYTES);
                            rd_remaining <= rd_remaining - 1'b1;
                            if (cfg_rd_gap == 0) begin
                                present_read(rd_address + 64'(BEAT_BYTES), rd_remaining == 2);
                            end else begin
                                mem_rd_data_valid <= 0;
                                mem_rd_data_last <= 0;
                                rd_wait <= cfg_rd_gap - 1;
                            end
                        end
                    end
                end else if (rd_wait != 0) begin
                    rd_wait <= rd_wait - 1;
                end else begin
                    present_read(rd_address, rd_remaining == 1);
                end
            end
        end
    end

    always_ff @(posedge clock) begin : write_channel
        byte unsigned bytes_in [0:BEAT_BYTES-1];
        byte unsigned strobes [0:BEAT_BYTES-1];
        if (reset) begin
            wr_state <= WR_IDLE;
            wr_address <= 0;
            wr_remaining <= 0;
            wr_wait <= 0;
            mem_wr_done <= 0;
        end else begin
            mem_wr_done <= 0;
            case (wr_state)
                WR_IDLE: if (mem_wr_cmd_valid && mem_wr_cmd_ready) begin
                    check_command(WRITE_MEMORY_ID, mem_wr_cmd_addr, mem_wr_cmd_len);
                    wr_address <= 64'(mem_wr_cmd_addr);
                    wr_remaining <= mem_wr_cmd_len;
                    wr_state <= WR_DATA;
                end
                WR_DATA: if (mem_wr_data_valid && mem_wr_data_ready) begin
                    if (mem_wr_data_last != (wr_remaining == 1))
                        $fatal(1, "DMA write: last disagrees with remaining beats (%0d)", wr_remaining);
                    for (int b = 0; b < BEAT_BYTES; b++) begin
                        bytes_in[b] = mem_wr_data[8*b +: 8];
                        strobes[b] = {7'b0, mem_wr_data_strb[b]};
                    end
                    if (bnn_mem_write(WRITE_MEMORY_ID, wr_address, bytes_in, strobes) != 0)
                        $fatal(1, "DMA write failed at 0x%h (see DPI error)", wr_address);
                    wr_address <= wr_address + 64'(BEAT_BYTES);
                    wr_remaining <= wr_remaining - 1'b1;
                    if (wr_remaining == 1) begin
                        wr_wait <= cfg_wr_latency;
                        wr_state <= WR_RESPONSE;
                    end
                end
                WR_RESPONSE: if (wr_wait != 0) begin
                    wr_wait <= wr_wait - 1;
                end else begin
                    mem_wr_done <= 1;
                    wr_state <= WR_IDLE;
                end
                default: $fatal(1, "invalid DMA write model state");
            endcase
        end
    end

    // Both the DUT and the environment must obey ready/valid stability rules.
    assert property (@(posedge clock) disable iff (reset)
        mem_rd_cmd_valid && !mem_rd_cmd_ready |=>
        mem_rd_cmd_valid && $stable({mem_rd_cmd_addr, mem_rd_cmd_len}))
                else $fatal(1, "DMA read command changed under backpressure");
    assert property (@(posedge clock) disable iff (reset)
        mem_wr_cmd_valid && !mem_wr_cmd_ready |=>
        mem_wr_cmd_valid && $stable({mem_wr_cmd_addr, mem_wr_cmd_len}))
                else $fatal(1, "DMA write command changed under backpressure");
    assert property (@(posedge clock) disable iff (reset)
        mem_wr_data_valid && !mem_wr_data_ready |=>
        mem_wr_data_valid && $stable({mem_wr_data, mem_wr_data_strb, mem_wr_data_last}))
                else $fatal(1, "DMA write data changed under backpressure");
    assert property (@(posedge clock) disable iff (reset)
        mem_rd_data_valid && !mem_rd_data_ready |=>
        mem_rd_data_valid && $stable({mem_rd_data, mem_rd_data_last}))
                else $fatal(1, "DMA read data changed under backpressure");
endmodule

`default_nettype wire
