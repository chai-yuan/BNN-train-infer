#!/usr/bin/env python3
"""Verify memory_out.img against the INT8 + FP32 case generated from SEED.

Usage: check_random_case.py SEED [DATA_DIR]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from gen_random_case import OutputRegion, OUTPUT_PADDING, build_case


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def first_difference(actual: bytes, expected: bytes) -> int | None:
    for offset, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            return offset
    return None


def verify_outside_regions(actual: bytes, expected: bytes,
                           regions: list[OutputRegion]) -> None:
    cursor = 0
    for region in sorted(regions, key=lambda item: item.address):
        if actual[cursor:region.address] != expected[cursor:region.address]:
            relative = first_difference(actual[cursor:region.address],
                                        expected[cursor:region.address])
            offset = cursor + (relative or 0)
            fail(f"memory outside output regions changed at 0x{offset:x}: "
                 f"got 0x{actual[offset]:02x}, expected 0x{expected[offset]:02x}")
        cursor = region.end
    if actual[cursor:] != expected[cursor:]:
        relative = first_difference(actual[cursor:], expected[cursor:])
        offset = cursor + (relative or 0)
        fail(f"memory outside output regions changed at 0x{offset:x}: "
             f"got 0x{actual[offset]:02x}, expected 0x{expected[offset]:02x}")


def verify_region(actual: bytes, case, region: OutputRegion) -> int:
    checked = 0
    for row in range(region.m):
        base = region.address + row * region.groups * 16
        for channel in range(case.n):
            offset = base + channel * 4
            expected = region.expected[row][channel]
            actual_bits = struct.unpack_from("<I", actual, offset)[0]
            if actual_bits != expected:
                expected_float = struct.unpack("<f", struct.pack("<I", expected))[0]
                actual_float = struct.unpack("<f", struct.pack("<I", actual_bits))[0]
                fail(
                    f"{region.name} row {row} channel {channel} at 0x{offset:x}: "
                    f"got 0x{actual_bits:08x} ({actual_float!r}), "
                    f"expected 0x{expected:08x} ({expected_float!r})"
                )
            checked += 1

        pad_start = base + case.n * 4
        pad_end = base + region.groups * 16
        if actual[pad_start:pad_end] != bytes([OUTPUT_PADDING]) * (pad_end - pad_start):
            fail(f"{region.name} row {row} tail padding at "
                 f"0x{pad_start:x}..0x{pad_end:x} was overwritten")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=lambda text: int(text, 0),
                        help="random seed used by the generator")
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "case",
        help="directory containing memory_out.img (default: tests/case)",
    )
    args = parser.parse_args()

    case = build_case(args.seed)
    actual_path = args.data_dir / "memory_out.img"
    try:
        actual = actual_path.read_bytes()
    except OSError as error:
        fail(f"cannot read {actual_path}: {error}")

    if len(actual) != len(case.image):
        fail(f"{actual_path} has {len(actual)} bytes, expected {len(case.image)}")

    verify_outside_regions(actual, case.image, [case.int8_region, case.fp32_region])

    checked = 0
    for region in (case.int8_region, case.fp32_region):
        checked += verify_region(actual, case, region)

    print(
        f"PASS seed={case.seed} N={case.n} groups={case.groups}, "
        f"INT8 M={case.int8_region.m}, FP32 M={case.fp32_region.m} "
        f"({checked} FP32 results, padding preserved) -> {actual_path}"
    )


if __name__ == "__main__":
    main()
