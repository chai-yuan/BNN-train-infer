`default_nettype none

// Combinational package adapter: no clock, reset, DPI, or host floating point.
module fp_math_test (
    input  wire [15:0]        half_bits,
    input  wire signed [31:0] int_value,
    input  wire [31:0]        fp_a,
    input  wire [31:0]        fp_b,
    input  wire [31:0]        positive_max,
    output wire [31:0]        half_result,
    output wire [31:0]        int_result,
    output wire [31:0]        mul_result,
    output wire [31:0]        add_result,
    output wire [31:0]        div127_result,
    output wire [7:0]         quantize_result
);
    import bnn_fp_math::*;

    assign half_result = fp16_to_fp32(half_bits);
    assign int_result = int32_to_fp32(int_value);
    assign mul_result = fp32_mul_ftz(fp_a, fp_b);
    assign add_result = fp32_add_ftz(fp_a, fp_b);
    assign div127_result = fp32_div127_ftz(fp_a);
    assign quantize_result = quantize_int8_ftz(fp_a, positive_max);
endmodule

`default_nettype wire
