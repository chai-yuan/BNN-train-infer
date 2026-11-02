`default_nettype none

// Synthesizable reference arithmetic: FP32 denormals are flushed to signed
// zero; finite results truncate toward zero. No IEEE exception flags or RNE.
package bnn_fp_math;
    localparam logic [31:0] CANONICAL_NAN = 32'h7fc00000;

    function automatic logic [31:0] fp16_to_fp32(input logic [15:0] value);
        logic [7:0] exponent;
        logic [22:0] fraction;
        int leading;
        if (value[14:10] == 5'h1f) begin
            if (value[9:0] != 0)
                return CANONICAL_NAN;
            return {value[15], 8'hff, 23'b0};
        end
        if (value[14:10] != 0) begin
            exponent = 8'(int'(value[14:10]) + 112);
            return {value[15], exponent, value[9:0], 13'b0};
        end
        if (value[9:0] == 0)
            return {value[15], 31'b0};
        leading = 0;
        for (int i = 0; i < 10; i++)
            if (value[i]) leading = i;
        exponent = 8'(leading + 103);
        fraction = 23'(value[9:0]) << (23 - leading);
        return {value[15], exponent, fraction};
    endfunction

    function automatic logic [31:0] int32_to_fp32(input logic signed [31:0] value);
        logic [31:0] magnitude;
        logic [22:0] fraction;
        logic [7:0] exponent;
        int leading;
        if (value == 0)
            return 32'b0;
        // Unsigned two's-complement magnitude also handles INT32_MIN.
        magnitude = value[31] ? (~$unsigned(value) + 32'd1) : $unsigned(value);
        leading = 0;
        for (int i = 0; i < 32; i++)
            if (magnitude[i]) leading = i;
        exponent = 8'(leading + 127);
        if (leading > 23)
            fraction = 23'(magnitude >> (leading - 23));
        else
            fraction = 23'(magnitude << (23 - leading));
        return {value[31], exponent, fraction};
    endfunction

    function automatic logic [31:0] fp32_mul_ftz(
        input logic [31:0] a, input logic [31:0] b
    );
        logic sign_bit;
        logic [47:0] product;
        logic [22:0] fraction;
        int exponent;
        sign_bit = a[31] ^ b[31];
        if (((a[30:23] == 8'hff) && (a[22:0] != 0)) ||
            ((b[30:23] == 8'hff) && (b[22:0] != 0)))
            return CANONICAL_NAN;
        if ((a[30:23] == 8'hff) || (b[30:23] == 8'hff)) begin
            // Subnormal operands are zero before classifying infinity * zero.
            if ((a[30:23] == 0) || (b[30:23] == 0))
                return CANONICAL_NAN;
            return {sign_bit, 8'hff, 23'b0};
        end
        if ((a[30:23] == 0) || (b[30:23] == 0))
            return {sign_bit, 31'b0};
        product = {1'b1, a[22:0]} * {1'b1, b[22:0]};
        exponent = int'(a[30:23]) + int'(b[30:23]) - 127;
        if (product[47]) begin
            exponent = exponent + 1;
            fraction = 23'(product >> 24);
        end else begin
            fraction = 23'(product >> 23);
        end
        if (exponent >= 255)
            return {sign_bit, 8'hff, 23'b0};
        if (exponent <= 0)
            return {sign_bit, 31'b0};
        return {sign_bit, 8'(exponent), fraction};
    endfunction

    function automatic logic [31:0] fp32_add_ftz(
        input logic [31:0] a, input logic [31:0] b
    );
        // Exact fixed-point alignment in units of 2^-149. Keeping all bits
        // makes cancellation and truncation correct without sticky-bit rules.
        logic [277:0] magnitude_a, magnitude_b, magnitude;
        logic sign_bit;
        logic [22:0] fraction;
        int leading, exponent;
        if (((a[30:23] == 8'hff) && (a[22:0] != 0)) ||
            ((b[30:23] == 8'hff) && (b[22:0] != 0)))
            return CANONICAL_NAN;
        if ((a[30:23] == 8'hff) && (b[30:23] == 8'hff)) begin
            if (a[31] != b[31]) return CANONICAL_NAN;
            return {a[31], 8'hff, 23'b0};
        end
        if (a[30:23] == 8'hff) return {a[31], 8'hff, 23'b0};
        if (b[30:23] == 8'hff) return {b[31], 8'hff, 23'b0};
        magnitude_a = '0;
        magnitude_b = '0;
        if (a[30:23] != 0)
            magnitude_a = 278'({1'b1, a[22:0]}) << (int'(a[30:23]) - 1);
        if (b[30:23] != 0)
            magnitude_b = 278'({1'b1, b[22:0]}) << (int'(b[30:23]) - 1);
        if (a[31] == b[31]) begin
            magnitude = magnitude_a + magnitude_b;
            sign_bit = a[31];
        end else if (magnitude_a >= magnitude_b) begin
            magnitude = magnitude_a - magnitude_b;
            sign_bit = a[31];
        end else begin
            magnitude = magnitude_b - magnitude_a;
            sign_bit = b[31];
        end
        if (magnitude == 0)
            return {a[31] & b[31], 31'b0};
        leading = 0;
        for (int i = 0; i < 278; i++)
            if (magnitude[i]) leading = i;
        exponent = leading - 22;
        if (exponent >= 255)
            return {sign_bit, 8'hff, 23'b0};
        if (exponent <= 0)
            return {sign_bit, 31'b0};
        fraction = 23'(magnitude >> (leading - 23));
        return {sign_bit, 8'(exponent), fraction};
    endfunction

    function automatic logic [31:0] fp32_div127_ftz(input logic [31:0] positive_a);
        logic [24:0] quotient;
        logic [22:0] fraction;
        int exponent;
        if (positive_a[30:23] == 8'hff) begin
            if (positive_a[22:0] != 0) return CANONICAL_NAN;
            return {positive_a[31], 8'hff, 23'b0};
        end
        if (positive_a[30:23] == 0)
            return {positive_a[31], 31'b0};
        // Divide the exact scaled mantissa, not an approximate reciprocal.
        quotient = 25'({1'b1, positive_a[22:0], 7'b0} / 31'd127);
        exponent = int'(positive_a[30:23]) - 7;
        if (quotient[24]) begin
            exponent = exponent + 1;
            fraction = quotient[23:1];
        end else begin
            fraction = quotient[22:0];
        end
        if (exponent <= 0)
            return {positive_a[31], 31'b0};
        return {positive_a[31], 8'(exponent), fraction};
    endfunction

    function automatic logic [7:0] quantize_int8_ftz(
        input logic [31:0] value, input logic [31:0] positive_max
    );
        logic [30:0] numerator, denominator;
        logic [7:0] magnitude;
        int exponent_difference;
        if ((value[30:23] == 0) || (value[30:23] == 8'hff) ||
            positive_max[31] || (positive_max[30:23] == 0) ||
            (positive_max[30:23] == 8'hff))
            return 8'b0;
        if (value[30:0] >= positive_max[30:0])
            magnitude = 8'd127;
        else begin
            exponent_difference = int'(positive_max[30:23]) - int'(value[30:23]);
            if (exponent_difference >= 8)
                return 8'b0;
            numerator = {7'b0, 1'b1, value[22:0]} * 31'd127;
            denominator = {7'b0, 1'b1, positive_max[22:0]} << exponent_difference;
            magnitude = 8'(numerator / denominator);
            if (magnitude > 8'd127) magnitude = 8'd127;
        end
        return value[31] ? (~magnitude + 8'd1) : magnitude;
    endfunction
endpackage

`default_nettype wire
