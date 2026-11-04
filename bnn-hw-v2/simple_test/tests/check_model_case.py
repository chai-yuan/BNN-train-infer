#!/usr/bin/env python3
"""Verify memory_out.img against the fixed bnn-accelerator W1A8 case.

Checks:
  1. only the declared output regions changed;
  2. every FP32 result is within atol + rtol * |golden| of the API golden;
  3. the FP32 tile equals the exact power-of-two scaling of the INT8 tile;
  4. prints the measured simulation cycles against the recorded baseline.

Usage:
    python3 tests/check_model_case.py [DATA_DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

FLOAT_BYTES = 4
MISMATCH_LIMIT = 10


def f32_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def load_memory(path):
    try:
        return path.read_bytes()
    except OSError as error:
        raise SystemExit(f"FAIL: cannot read {path}: {error}\n"
                         f"      run `make -C bnn-hw-v2/simple_test test` first")


def load_golden(path):
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise SystemExit(f"FAIL: cannot read {path}: {error}")
    values = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values.append(f32_from_bits(int(line, 16)))
    return values


def compare(name, got_values, want_values, atol, rtol, output_base=None, columns=None):
    if len(got_values) != len(want_values):
        raise SystemExit(f"FAIL: {name} has {len(got_values)} values, "
                         f"golden has {len(want_values)}")
    max_abs = 0.0
    max_rel = 0.0
    max_index = 0
    failures = []

    for index, (got, want) in enumerate(zip(got_values, want_values)):
        difference = abs(got - want)
        limit = atol + rtol * abs(want)
        if difference > max_abs:
            max_abs = difference
            max_index = index
        if want != 0.0:
            max_rel = max(max_rel, difference / abs(want))
        if difference > limit and len(failures) < MISMATCH_LIMIT:
            if output_base is None:
                location = f"index {index}"
            else:
                row, column = divmod(index, columns)
                location = (f"row {row} channel {column} "
                            f"@0x{output_base + index * FLOAT_BYTES:x}")
            failures.append(
                f"    {name} {location}: "
                f"got {got!r} (0x{f32_bits(got):08x}), "
                f"want {want!r} (0x{f32_bits(want):08x}), "
                f"diff {difference:.9g} > {limit:.9g}")

    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        raise SystemExit(f"FAIL: {name} exceeded atol={atol:g}, rtol={rtol:g}")

    if output_base is None:
        worst = f"index {max_index}"
    else:
        row, column = divmod(max_index, columns)
        worst = f"worst row {row}, channel {column}"
    print(f"PASS {name:<9} {len(got_values)} values, "
          f"max_abs={max_abs:.9g}, max_rel={max_rel:.9g} ({worst})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "case",
                        help="case directory (default: tests/case)")
    args = parser.parse_args()
    data_dir = args.data_dir

    try:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"FAIL: cannot read {data_dir / 'meta.json'}: {error}")

    initial = load_memory(data_dir / "memory.img")
    actual = load_memory(data_dir / "memory_out.img")
    if len(actual) != len(initial):
        raise SystemExit(f"FAIL: memory_out.img is {len(actual)} bytes, "
                         f"expected exactly {len(initial)} (a write escaped the image)")

    rows = meta["rows"]
    columns = meta["matrix"]["out_features"]
    elements = rows * columns
    values_per_workload = {}
    regions = []

    print(f"case: {meta.get('description', data_dir.name)}")

    for workload in meta["workloads"]:
        output = meta["addresses"][workload["output"]]
        count = elements * FLOAT_BYTES
        if output % FLOAT_BYTES:
            raise SystemExit(f"FAIL: {workload['name']} output is not word aligned")
        if output + count > len(actual):
            raise SystemExit(f"FAIL: {workload['name']} output exceeds memory image")
        regions.append((workload["name"], output, output + count))

    covered = bytearray(len(initial))
    for _, begin, end in regions:
        covered[begin:end] = b"\x01" * (end - begin)
    changed = 0
    first_change = None
    for offset, (before, after) in enumerate(zip(initial, actual)):
        if not covered[offset] and before != after:
            changed += 1
            if first_change is None:
                first_change = (offset, before, after)
    if changed:
        offset, before, after = first_change
        raise SystemExit(
            f"FAIL: {changed} bytes changed outside output regions; first at "
            f"0x{offset:x}: 0x{before:02x} -> 0x{after:02x}")

    tolerance = meta["tolerance"]
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])

    for workload in meta["workloads"]:
        output = meta["addresses"][workload["output"]]
        count = elements * FLOAT_BYTES
        got = list(struct.unpack_from(f"<{elements}f", actual, output))
        want = load_golden(data_dir / workload["golden"])
        compare(workload["name"], got, want, atol, rtol, output, columns)
        values_per_workload[workload["name"]] = got

    for relation in meta.get("relationships", []):
        left = values_per_workload[relation["left"]]
        right = values_per_workload[relation["right"]]
        scale = float(relation["scale"])
        scaled = [f32_from_bits(f32_bits(value * scale)) for value in right]
        compare(f"{relation['left']}={relation['right']}*{scale:g}",
                left, scaled, atol, rtol)

    profile = data_dir / "memory_profile.txt"
    baseline_path = data_dir / "baseline_cycles.txt"
    baseline = None
    if baseline_path.is_file():
        match = re.search(r"\d+", baseline_path.read_text(encoding="ascii", errors="replace"))
        if match:
            baseline = int(match.group(0))
    if profile.is_file():
        text = profile.read_text(encoding="ascii", errors="replace")
        match = re.search(r"simulation cycles\s*:\s*(\d+)", text)
        if match:
            cycles = int(match.group(1))
            if baseline:
                change = 100.0 * (cycles - baseline) / baseline
                print(f"cycles {cycles} (baseline {baseline}, {change:+.2f}%)")
            else:
                print(f"cycles {cycles} (no baseline recorded)")
        else:
            print(f"cycles: no 'simulation cycles' line in {profile}")

    print(f"PASS: {len(meta['workloads'])} workloads, {elements} values each, "
          f"atol={atol:g}, rtol={rtol:g} -> {data_dir / 'memory_out.img'}")


if __name__ == "__main__":
    main()
