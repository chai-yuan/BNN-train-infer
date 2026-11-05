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
        // 27-bit fixed-point alignment in units of the larger exponent's
        // significand LSB: {Carry[26], Mantissa[25:2], Guard[1], Sticky[0]}.
        // Only 24-bit significands and a capped 26-bit shift are needed.
        logic a_is_nan, b_is_nan, a_is_inf, b_is_inf;
        logic [7:0]  exp_a, exp_b, exp_l, exp_s, exp_diff;
        logic [23:0] sig_a, sig_b, sig_l, sig_s;
        logic        sign_l, sign_s, a_ge_b, sticky;
        logic [4:0]  shift_amt, lead_idx, lshift;
        logic [49:0] shifted_raw;
        logic [26:0] aligned_l, aligned_s, mag_27;
        logic [22:0] fraction;
        int          exponent;

        // 1. NaN 与 Inf 特殊值处理
        a_is_nan = (a[30:23] == 8'hff) && (a[22:0] != 23'd0);
        b_is_nan = (b[30:23] == 8'hff) && (b[22:0] != 23'd0);
        a_is_inf = (a[30:23] == 8'hff) && (a[22:0] == 23'd0);
        b_is_inf = (b[30:23] == 8'hff) && (b[22:0] == 23'd0);

        if (a_is_nan || b_is_nan) return CANONICAL_NAN;
        if (a_is_inf && b_is_inf) begin
            if (a[31] != b[31]) return CANONICAL_NAN;
            return {a[31], 8'hff, 23'b0};
        end
        if (a_is_inf) return {a[31], 8'hff, 23'b0};
        if (b_is_inf) return {b[31], 8'hff, 23'b0};

        // 2. 输入次正规数 FTZ 展开（仅 24-bit，不再左移 278 位）
        exp_a = a[30:23];
        exp_b = b[30:23];
        sig_a = (exp_a != 8'd0) ? {1'b1, a[22:0]} : 24'd0;
        sig_b = (exp_b != 8'd0) ? {1'b1, b[22:0]} : 24'd0;

        // 3. 32-bit 绝对值比较并交换大小操作数（L >= S）
        a_ge_b = ({exp_a, sig_a} >= {exp_b, sig_b});
        if (a_ge_b) begin
            sign_l = a[31]; exp_l = exp_a; sig_l = sig_a;
            sign_s = b[31]; exp_s = exp_b; sig_s = sig_b;
        end else begin
            sign_l = b[31]; exp_l = exp_b; sig_l = sig_b;
            sign_s = a[31]; exp_s = exp_a; sig_s = sig_a;
        end

        // 4. 对阶右移（最大只需移 26 位，低位折叠到 Sticky）
        exp_diff    = exp_l - exp_s;
        shift_amt   = (exp_diff > 8'd26) ? 5'd26 : exp_diff[4:0];
        shifted_raw = {sig_s, 26'd0} >> shift_amt;
        sticky      = |shifted_raw[24:0];

        // 5. 27-bit 定点加减：{Carry[26], Mantissa[25:2], Guard[1], Sticky[0]}
        aligned_l = {1'b0, sig_l, 2'b00};
        aligned_s = {1'b0, shifted_raw[49:25], sticky};
        mag_27    = (sign_l == sign_s) ? (aligned_l + aligned_s)
                                       : (aligned_l - aligned_s);

        if (mag_27[26:1] == 26'd0)
            return {a[31] & b[31], 31'b0};

        // 6. 26-bit 前导 1 搜索（替代原来的 278 次循环）
        lead_idx = 5'd1;
        for (int i = 1; i <= 26; i++) begin
            if (mag_27[i]) lead_idx = 5'(i);
        end

        // 7. 计算新指数与 26-bit 规格化左移
        exponent = int'(exp_l) + int'(lead_idx) - 25;
        if (exponent >= 255)
            return {sign_l, 8'hff, 23'b0};
        if (exponent <= 0)
            return {sign_l, 31'b0};

        lshift = 5'd26 - lead_idx;
        fraction = 23'((26'(mag_27[26:1]) << lshift) >> 2);
        return {sign_l, 8'(exponent), fraction};
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
