#!/usr/bin/env python3
"""Build a Markdown timing/area report from Yosys + OpenSTA outputs.

The Makefile runs one OpenSTA job per corner, each producing:
    sta_<corner>.json   machine-readable path data (report_checks -format json)
    sta_<corner>.rpt    human-readable path detail
    sta_<corner>.log    tool diagnostics
and a Yosys log with `stat -liberty` area information.

Fmax is extrapolated from the setup slack at the probe clock period:
    Fmax = 1 / (period - slack)

The numbers are post-synthesis, ideal-clock estimates; see the generated
report's notes for the SRAM and signoff caveats.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="report directory")
    parser.add_argument("--top", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--map-corner", required=True)
    parser.add_argument("--corners", required=True, help="space separated corner list")
    parser.add_argument("--period", type=float, required=True, help="probe clock period (ns)")
    parser.add_argument("--uncertainty", default="", help="clock uncertainty (ns)")
    parser.add_argument("--io-delay", default="", help="optional IO delay (ns)")
    parser.add_argument("--out-load", default="", help="optional output load (pF)")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--synth-log", required=True)
    parser.add_argument("--netlist", required=True)
    parser.add_argument("--liberty", default="", help="liberty file (metadata)")
    return parser.parse_args()


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError as error:
        print(f"summarize: cannot read {path}: {error}", file=sys.stderr)
        return ""


def parse_synthesis(synth_log):
    """Return area/cell statistics from the last `stat` section."""
    text = read_text(synth_log)
    sections = re.split(r"^=== .* ===$", text, flags=re.MULTILINE)
    stat = sections[-1] if len(sections) > 1 else text

    area = None
    matches = re.findall(r"Chip area for module '[^']*':\s*([0-9.]+)", stat)
    if not matches:
        matches = re.findall(r"Chip area for module '[^']*':\s*([0-9.]+)", text)
    if matches:
        area = float(matches[-1])

    cell_counts = {}
    for line in stat.splitlines():
        # `stat -liberty` prints: count, numeric area, cell type.
        # The same table also has count, "-", wires/ports; those must not be
        # counted as cells.
        match = re.match(r"^\s*(\d+)\s+([0-9][0-9.Ee+-]*)\s+(\S+)\s*$", line)
        if match:
            count = int(match.group(1))
            cell = match.group(3)
            if cell == "cells" or cell.startswith("$"):
                continue
            cell_counts[cell] = cell_counts.get(cell, 0) + count

    total = None
    total_matches = re.findall(
        r"^\s*(\d+)\s+[0-9.Ee+-]+\s+cells\s*$", stat, re.MULTILINE
    )
    if not total_matches:
        total_matches = re.findall(r"^\s*(\d+)\s+cells\s*$", stat, re.MULTILINE)
    if total_matches:
        total = int(total_matches[-1])
    if total is None and cell_counts:
        total = sum(cell_counts.values())

    dffs = sum(
        count
        for cell, count in cell_counts.items()
        if cell.startswith(("DFF", "SDFF", "EDFF"))
    )
    combinational = total - dffs if total is not None else None
    top_cells = sorted(cell_counts.items(), key=lambda item: (-item[1], item[0]))[:10]

    return {
        "area": area,
        "total": total,
        "dffs": dffs,
        "combinational": combinational,
        "top_cells": top_cells,
        "cell_counts": cell_counts,
    }


def parse_sta_json(path):
    """Return max-delay checks sorted by slack (worst first)."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        print(f"summarize: cannot parse {path}: {error}", file=sys.stderr)
        return []

    paths = []
    for check in data.get("checks", []):
        if check.get("path_type") not in (None, "max"):
            continue
        slack = check.get("slack")
        arrival = check.get("data_arrival_time")
        paths.append(
            {
                "startpoint": check.get("startpoint", ""),
                "endpoint": check.get("endpoint", ""),
                "group": check.get("path_group", ""),
                "slack_ns": None if slack is None else slack * 1e9,
                "arrival_ns": None if arrival is None else arrival * 1e9,
                "source_path": check.get("source_path", []),
            }
        )
    paths.sort(
        key=lambda path: (
            path["slack_ns"] is None,
            path["slack_ns"] if path["slack_ns"] is not None else 0.0,
        )
    )
    return paths


def start_net(path):
    for entry in path.get("source_path", []):
        net = entry.get("net")
        if net:
            return net
    return None


def endpoint_q_net(netlist_text, endpoint):
    instance = endpoint.split("/")[0]
    if not instance:
        return None
    match = re.search(
        r"\b" + re.escape(instance) + r"\s*\([^;]*?\.Q\s*\(\s*([^)\s]+)\s*\)",
        netlist_text,
        re.DOTALL,
    )
    return match.group(1) if match else None


def fmax_mhz(period_ns, slack_ns):
    if slack_ns is None:
        return None
    critical_path = period_ns - slack_ns
    if critical_path <= 0:
        return None
    return 1000.0 / critical_path


def format_number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    args = parse_args()
    corners = args.corners.split()
    period = args.period

    area_info = parse_synthesis(args.synth_log)
    netlist_text = read_text(args.netlist)

    results = {}
    missing = []
    for corner in corners:
        json_path = os.path.join(args.run_dir, f"sta_{corner}.json")
        if not os.path.isfile(json_path):
            missing.append(json_path)
            continue
        paths = parse_sta_json(json_path)
        if not paths:
            missing.append(json_path + " (no max paths)")
            continue
        worst = paths[0]
        results[corner] = {
            "paths": paths,
            "worst": worst,
            "fmax": fmax_mhz(period, worst["slack_ns"]),
        }

    if missing:
        for item in missing:
            print(f"summarize: missing or empty STA result: {item}", file=sys.stderr)
        return 1

    worst_corner = min(
        results,
        key=lambda corner: results[corner]["worst"]["slack_ns"]
        if results[corner]["worst"]["slack_ns"] is not None
        else float("inf"),
    )
    worst = results[worst_corner]

    os.makedirs(args.run_dir, exist_ok=True)
    report_path = os.path.join(args.run_dir, "report.md")
    csv_path = os.path.join(args.run_dir, "summary.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["corner", "slack_ns", "fmax_mhz", "arrival_ns"])
        for corner in corners:
            result = results[corner]
            path = result["worst"]
            writer.writerow(
                [
                    corner,
                    "" if path["slack_ns"] is None else f"{path['slack_ns']:.6f}",
                    "" if result["fmax"] is None else f"{result['fmax']:.3f}",
                    "" if path["arrival_ns"] is None else f"{path['arrival_ns']:.6f}",
                ]
            )

    lines = []
    lines.append(f"# `{args.top}` timing and area report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Top module: `{args.top}`")
    lines.append(f"- Library family: `{args.family}` ({args.map_corner} used for ABC mapping)")
    lines.append(f"- STA corners: {', '.join(f'`{corner}`' for corner in corners)}")
    lines.append(f"- Probe clock: `{period} ns`, uncertainty `{args.uncertainty or '0'} ns`")
    if args.io_delay:
        lines.append(f"- IO delay constraint: `{args.io_delay} ns`")
    if args.out_load:
        lines.append(f"- Output load constraint: `{args.out_load} pF`")
    lines.append("- Direction: post-synthesis, ideal clock, no wire/CTS/OCV")
    lines.append("")

    lines.append("## Fmax by corner")
    lines.append("")
    lines.append("| Corner | Worst slack (ns) | Data arrival (ns) | Fmax (MHz) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for corner in corners:
        result = results[corner]
        path = result["worst"]
        lines.append(
            f"| `{corner}` | {format_number(path['slack_ns'])} | "
            f"{format_number(path['arrival_ns'])} | {format_number(result['fmax'], 1)} |"
        )
    lines.append("")
    lines.append(
        f"**Worst corner:** `{worst_corner}` — "
        f"**Fmax ≈ {format_number(worst['fmax'], 1)} MHz** "
        f"(post-synthesis, periodic probe)."
    )
    lines.append("")
    lines.append("`Fmax = 1 / (probe_period - slack)`; slack includes clock uncertainty and setup time.")
    lines.append("")

    lines.append("## Area (logic only)")
    lines.append("")
    if area_info["total"] is None and area_info["area"] is None:
        lines.append("Area statistics were not found in the synthesis log.")
    else:
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        if area_info["total"] is not None:
            lines.append(f"| Standard cells | {area_info['total']} |")
        if area_info["dffs"] is not None:
            lines.append(f"| Flip-flops | {area_info['dffs']} |")
        if area_info["combinational"] is not None:
            lines.append(f"| Combinational cells | {area_info['combinational']} |")
        if area_info["area"] is not None:
            lines.append(f"| Chip area (Liberty units, µm²) | {area_info['area']:.1f} |")
        lines.append("")
        if area_info["top_cells"]:
            lines.append("Most-used cell types:")
            lines.append("")
            for cell, count in area_info["top_cells"]:
                lines.append(f"- `{cell}`: {count}")
    lines.append("")
    lines.append(
        "Area covers synthesized logic only. The `TARGET_ASIC` SRAM wrappers are "
        "placeholders, not real memory macros."
    )
    lines.append("")

    lines.append(f"## Critical paths (`{worst_corner}`)")
    lines.append("")
    lines.append("| # | Start | End | Arrival (ns) | Slack (ns) | Local Fmax (MHz) |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
    for index, path in enumerate(worst["paths"][: args.top_n], start=1):
        start_label = start_net(path) or path["startpoint"]
        end_label = endpoint_q_net(netlist_text, path["endpoint"]) or path["endpoint"]
        lines.append(
            f"| {index} | `{start_label}` | `{end_label}` | "
            f"{format_number(path['arrival_ns'])} | {format_number(path['slack_ns'])} | "
            f"{format_number(fmax_mhz(period, path['slack_ns']), 1)} |"
        )
    lines.append("")

    lines.append("## Files")
    lines.append("")
    for corner in corners:
        lines.append(
            f"- `sta_{corner}.json`, `sta_{corner}.rpt`, `sta_{corner}.log`"
        )
    lines.append("- `synth.log` (in `../../build/` relative to this report)")
    lines.append(f"- Machine-readable summary: `{os.path.basename(csv_path)}`")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- These are **post-synthesis** numbers with ideal clock and no wire delay, CTS, or OCV. Real silicon/placed Fmax is lower.")
    lines.append("- The SRAM wrappers use the `TARGET_ASIC` placeholder (a register), not a real SRAM macro. Generate macros with the ICS55 SRAM compiler and re-run for SRAM-path timing and area.")
    lines.append("- For signoff, run OpenROAD/OpenSTA through placement, CTS and routing with the PDK LEF/GDS/techLEF and real SRAM macros.")
    lines.append("- The reported Fmax is an internal register-to-register estimate unless IO_DELAY/OUT_LOAD are set.")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))

    print(f"report: {report_path}")
    print(
        f"worst: {worst_corner}  Fmax={format_number(worst['fmax'], 2)} MHz  "
        f"slack={format_number(worst['worst']['slack_ns'])} ns"
    )
    if area_info["area"] is not None or area_info["total"] is not None:
        print(
            f"area: {format_number(area_info['area'], 1)} um2  "
            f"cells={area_info['total']}  dff={area_info['dffs']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
