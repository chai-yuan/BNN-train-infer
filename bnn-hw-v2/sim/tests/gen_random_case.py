#!/usr/bin/env python3
"""Generate a deterministic random INT8 + FP32 BNN v2 test case.

Usage: gen_random_case.py SEED [OUT_DIR]

One command stream runs two independent compute blocks with shared weights:
  1. LOAD_WEIGHT
  2. COMPUTE INT8 input
  3. WRITEBACK INT8 output
  4. COMPUTE FP32 dynamic-quantized input
  5. WRITEBACK FP32 output

The FP32 reference implements the RTL's bit-level quantization and RTZ
arithmetic, so the generated expected values are bit-exact, not approximate.
"""

from __future__ import annotations

import argparse
import random
import struct
from dataclasses import dataclass
from pathlib import Path

LANES = 128
WEIGHT_ADDR = 0x1000
INT8_INPUT_ADDR = 0x2000
INT8_OUTPUT_ADDR = 0x3000
FP32_INPUT_ADDR = 0x6000
FP32_OUTPUT_ADDR = 0x8000
OUTPUT_PADDING = 0xA5

CANONICAL_NAN = 0x7FC00000

# Mix of exact powers of two and values that exercise RTZ truncation.
SCALES = (
    0x3C00,  # +1.0
    0xBC00,  # -1.0
    0x3800,  # +0.5
    0xB800,  # -0.5
    0x0000,  # +0.0
    0x4000,  # +2.0
    0x3400,  # +0.25
    0x3E00,  # +1.5
    0x3FFF,  # +1.9990234375
    0x0001,  # +2^-24 (FP16 subnormal)
    0x0400,  # +2^-14
    0x7BFF,  # +65504.0
    0xBFFF,  # -1.9990234375
)


@dataclass(frozen=True)
class OutputRegion:
    name: str
    address: int
    m: int
    groups: int
    expected: tuple[tuple[int, ...], ...]

    @property
    def end(self) -> int:
        return self.address + self.m * self.groups * 16


@dataclass(frozen=True)
class Case:
    seed: int
    n: int
    groups: int
    image: bytes
    commands: tuple[int, ...]
    int8_region: OutputRegion
    fp32_region: OutputRegion


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fp16_to_fp32_bits(value: int) -> int:
    sign = (value >> 15) & 1
    exponent = (value >> 10) & 0x1F
    fraction = value & 0x3FF
    if exponent == 0x1F:
        if fraction != 0:
            return CANONICAL_NAN
        return (sign << 31) | (0xFF << 23)
    if exponent != 0:
        return (sign << 31) | (((exponent + 112) & 0xFF) << 23) | (fraction << 13)
    if fraction == 0:
        return sign << 31
    leading = fraction.bit_length() - 1
    exponent32 = leading + 103
    return (sign << 31) | (exponent32 << 23) | ((fraction << (23 - leading)) & 0x7FFFFF)


def int32_to_fp32_bits(value: int) -> int:
    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    magnitude = abs(value)
    leading = magnitude.bit_length() - 1
    exponent = leading + 127
    if leading > 23:
        fraction = (magnitude >> (leading - 23)) & 0x7FFFFF
    else:
        fraction = (magnitude << (23 - leading)) & 0x7FFFFF
    return (sign << 31) | (exponent << 23) | fraction


def fp32_mul_ftz_bits(a: int, b: int) -> int:
    sign = ((a >> 31) ^ (b >> 31)) & 1
    a_exp, a_frac = (a >> 23) & 0xFF, a & 0x7FFFFF
    b_exp, b_frac = (b >> 23) & 0xFF, b & 0x7FFFFF

    if (a_exp == 0xFF and a_frac != 0) or (b_exp == 0xFF and b_frac != 0):
        return CANONICAL_NAN
    if a_exp == 0xFF or b_exp == 0xFF:
        if a_exp == 0 or b_exp == 0:
            return CANONICAL_NAN
        return (sign << 31) | (0xFF << 23)
    if a_exp == 0 or b_exp == 0:
        return sign << 31

    product = ((1 << 23) | a_frac) * ((1 << 23) | b_frac)
    exponent = a_exp + b_exp - 127
    if product & (1 << 47):
        exponent += 1
        fraction = (product >> 24) & 0x7FFFFF
    else:
        fraction = (product >> 23) & 0x7FFFFF

    if exponent >= 255:
        return (sign << 31) | (0xFF << 23)
    if exponent <= 0:
        return sign << 31
    return (sign << 31) | (exponent << 23) | fraction


def fp32_div127_ftz_bits(value: int) -> int:
    sign = (value >> 31) & 1
    exponent = (value >> 23) & 0xFF
    fraction = value & 0x7FFFFF

    if exponent == 0xFF:
        if fraction != 0:
            return CANONICAL_NAN
        return value
    if exponent == 0:
        return sign << 31

    quotient = (((1 << 23) | fraction) << 7) // 127
    scaled_exponent = exponent - 7
    if quotient & (1 << 24):
        scaled_exponent += 1
        fraction_out = (quotient >> 1) & 0x7FFFFF
    else:
        fraction_out = quotient & 0x7FFFFF

    if scaled_exponent <= 0:
        return sign << 31
    return (sign << 31) | (scaled_exponent << 23) | fraction_out


def quantize_int8_ftz(value: int, positive_max: int) -> int:
    value_exp = (value >> 23) & 0xFF
    max_exp = (positive_max >> 23) & 0xFF
    if (value_exp == 0 or value_exp == 0xFF or (positive_max >> 31) or
            max_exp == 0 or max_exp == 0xFF):
        return 0

    if (value & 0x7FFFFFFF) >= (positive_max & 0x7FFFFFFF):
        magnitude = 127
    else:
        exponent_difference = max_exp - value_exp
        if exponent_difference >= 8:
            return 0
        numerator = ((1 << 23) | (value & 0x7FFFFF)) * 127
        denominator = ((1 << 23) | (positive_max & 0x7FFFFF)) << exponent_difference
        magnitude = numerator // denominator
        if magnitude > 127:
            magnitude = 127

    return -magnitude if (value >> 31) else magnitude


def row_max_abs_bits(row: list[int]) -> int:
    maximum = 0
    for value in row:
        exponent = (value >> 23) & 0xFF
        if exponent == 0 or exponent == 0xFF:
            continue
        magnitude = value & 0x7FFFFFFF
        if magnitude > maximum:
            maximum = magnitude
    return maximum


def quantize_row(row: list[int]) -> tuple[list[int], int]:
    maximum = row_max_abs_bits(row)
    scale = fp32_div127_ftz_bits(maximum)
    values = [quantize_int8_ftz(value, maximum) for value in row]
    return values, scale


def dot_product(values: list[int], signs: int) -> int:
    return sum(
        value if (signs >> lane) & 1 else -value
        for lane, value in enumerate(values)
    )


def make_fp32_row(rng: random.Random) -> list[int]:
    row: list[int] = []
    normal_count = 0
    for _ in range(LANES):
        choice = rng.random()
        sign = rng.getrandbits(1) << 31
        if choice < 0.70:
            exponent = rng.randint(120, 150)
            fraction = rng.getrandbits(23)
            row.append(sign | (exponent << 23) | fraction)
            normal_count += 1
        elif choice < 0.78:
            fraction = rng.getrandbits(23) or 1
            row.append(sign | fraction)  # subnormal
        elif choice < 0.84:
            row.append(sign)  # signed zero
        elif choice < 0.92:
            row.append(sign | (0xFF << 23))  # infinity
        else:
            fraction = rng.getrandbits(23) or 1
            row.append(sign | (0xFF << 23) | fraction)  # NaN
    if normal_count == 0:
        index = rng.randrange(LANES)
        row[index] = ((rng.getrandbits(1) << 31) | (rng.randint(120, 150) << 23) |
                      rng.getrandbits(23))
    return row


def build_case(seed: int) -> Case:
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    groups = (n + 3) // 4

    weights: list[tuple[int, int]] = []
    weight_blob = bytearray()
    for _ in range(n):
        scale_bits = rng.choice(SCALES)
        signs = rng.getrandbits(LANES)
        weights.append((scale_bits, signs))
        weight_blob += struct.pack("<H", scale_bits)
        weight_blob += signs.to_bytes(LANES // 8, "little")

    int8_m = rng.randint(1, 4)
    int8_rows = [
        [rng.randint(-127, 127) for _ in range(LANES)]
        for _ in range(int8_m)
    ]
    int8_expected: list[tuple[int, ...]] = []
    for row in int8_rows:
        row_output = []
        for scale_bits, signs in weights:
            dot = dot_product(row, signs)
            weighted = fp32_mul_ftz_bits(
                int32_to_fp32_bits(dot), fp16_to_fp32_bits(scale_bits)
            )
            row_output.append(weighted)
        int8_expected.append(tuple(row_output))

    fp32_m = rng.randint(1, 4)
    fp32_rows = [make_fp32_row(rng) for _ in range(fp32_m)]
    fp32_expected: list[tuple[int, ...]] = []
    for row in fp32_rows:
        quantized, row_scale = quantize_row(row)
        row_output = []
        for scale_bits, signs in weights:
            dot = dot_product(quantized, signs)
            weighted = fp32_mul_ftz_bits(
                int32_to_fp32_bits(dot), fp16_to_fp32_bits(scale_bits)
            )
            row_output.append(fp32_mul_ftz_bits(weighted, row_scale))
        fp32_expected.append(tuple(row_output))

    weight_beats = (len(weight_blob) + 15) // 16
    weight_read_end = WEIGHT_ADDR + weight_beats * 16
    int8_input_end = INT8_INPUT_ADDR + int8_m * LANES
    int8_output_end = INT8_OUTPUT_ADDR + int8_m * groups * 16
    fp32_input_end = FP32_INPUT_ADDR + fp32_m * LANES * 4
    fp32_output_end = FP32_OUTPUT_ADDR + fp32_m * groups * 16
    image_size = max(
        weight_read_end, int8_input_end, int8_output_end,
        fp32_input_end, fp32_output_end,
    )

    image = bytearray(image_size)
    image[WEIGHT_ADDR:WEIGHT_ADDR + len(weight_blob)] = weight_blob

    for row_index, row in enumerate(int8_rows):
        offset = INT8_INPUT_ADDR + row_index * LANES
        image[offset:offset + LANES] = bytes(value & 0xFF for value in row)

    for row_index, row in enumerate(fp32_rows):
        offset = FP32_INPUT_ADDR + row_index * LANES * 4
        for lane, value in enumerate(row):
            struct.pack_into("<I", image, offset + lane * 4, value)

    image[INT8_OUTPUT_ADDR:int8_output_end] = bytes([OUTPUT_PADDING]) * (
        int8_output_end - INT8_OUTPUT_ADDR
    )
    image[FP32_OUTPUT_ADDR:fp32_output_end] = bytes([OUTPUT_PADDING]) * (
        fp32_output_end - FP32_OUTPUT_ADDR
    )

    commands = (
        (WEIGHT_ADDR << 32) | (n << 16) | 0x10,
        (INT8_INPUT_ADDR << 32) | (int8_m << 8) | (1 << 25) | 0x20,
        (INT8_OUTPUT_ADDR << 32) | 0x30,
        (FP32_INPUT_ADDR << 32) | (fp32_m << 8) | 0x20,
        (FP32_OUTPUT_ADDR << 32) | 0x30,
    )

    return Case(
        seed=seed,
        n=n,
        groups=groups,
        image=bytes(image),
        commands=commands,
        int8_region=OutputRegion(
            "INT8", INT8_OUTPUT_ADDR, int8_m, groups, tuple(int8_expected)
        ),
        fp32_region=OutputRegion(
            "FP32", FP32_OUTPUT_ADDR, fp32_m, groups, tuple(fp32_expected)
        ),
    )


def write_case(case: Case, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    command_path = out_dir / "commands.hex"
    memory_path = out_dir / "memory.img"

    lines = [
        f"# seed={case.seed} N={case.n} groups={case.groups}",
        f"# INT8 M={case.int8_region.m} @0x{INT8_INPUT_ADDR:x} "
        f"-> 0x{case.int8_region.address:x}",
        f"# FP32 M={case.fp32_region.m} @0x{FP32_INPUT_ADDR:x} "
        f"-> 0x{case.fp32_region.address:x}",
    ]
    lines.extend(f"0x{instruction:016x}" for instruction in case.commands)
    command_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    memory_path.write_bytes(case.image)


def parse_seed(text: str) -> int:
    return int(text, 0)


def self_check() -> None:
    assert fp32_mul_ftz_bits(f32_bits(16253.0), f32_bits(1.9990234375)) == 0x46FDD441
    assert fp32_div127_ftz_bits(f32_bits(1.0)) == 0x3C010204
    assert quantize_int8_ftz(f32_bits(1.0), f32_bits(1.0)) == 127
    assert quantize_int8_ftz(f32_bits(-0.5), f32_bits(1.0)) == -63
    assert quantize_int8_ftz(f32_bits(0.25), f32_bits(1.0)) == 31
    assert quantize_int8_ftz(0x00000001, f32_bits(1.0)) == 0
    assert fp32_mul_ftz_bits(f32_bits(0.0), f32_bits(-1.0)) == 0x80000000
    assert fp32_mul_ftz_bits(f32_bits(-0.0), f32_bits(-1.0)) == 0
    assert fp32_mul_ftz_bits(CANONICAL_NAN, f32_bits(1.0)) == CANONICAL_NAN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=parse_seed, help="random seed (decimal or 0x hex)")
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "case",
        help="output directory (default: tests/case)",
    )
    args = parser.parse_args()

    self_check()
    case = build_case(args.seed)
    write_case(case, args.out_dir)
    print(
        f"generated seed={case.seed} N={case.n} groups={case.groups}, "
        f"INT8 M={case.int8_region.m}, FP32 M={case.fp32_region.m} -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
