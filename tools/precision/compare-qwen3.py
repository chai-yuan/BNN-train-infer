#!/usr/bin/env python3
"""Compare C Qwen3 W1A8 logits with the NumPy full-precision-activation reference.

Three comparisons are reported:
  1. C W1A8 vs Python W1A8: implementation control. Large values indicate a
     mismatch between the C engine and the reference model, not quantization.
  2. Python W1A8 vs Python W1A32: expected activation-quantization loss.
  3. C W1A8 vs Python W1A32: total gap, the headline number.

The tool exits 0 when everything is within threshold, 1 on a large expected
quantization gap or an unexpected total gap, and 2 on a likely implementation
mismatch or an invalid run.
"""

import argparse
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "tools" / "bonsai-ref.py"
DUMPER = HERE / "dump-logits"

RED = "1;31"
YELLOW = "1;33"
GREEN = "1;32"
CYAN = "1;36"


def paint(text, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def banner(text, code):
    line = "=" * 78
    print(paint(line, code))
    print(paint(text, code))
    print(paint(line, code))


def run_command(command, label, verbose):
    command = [str(part) for part in command]
    if verbose:
        print("+ " + shlex.join(command), flush=True)
    else:
        print(f"[{label}] running...", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"[{label}] failed with exit code {result.returncode}", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(2)
    if verbose and result.stdout:
        print(result.stdout.rstrip())
    return result


def read_dump(path):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path}: file is too short")
    n, vocab = struct.unpack_from("<II", data, 0)
    if n == 0 or vocab == 0:
        raise ValueError(f"{path}: empty logits")
    expected = 8 + 4 * n + 4 * n * vocab
    if len(data) != expected:
        raise ValueError(f"{path}: size {len(data)} does not match n={n} vocab={vocab}")
    ids = np.frombuffer(data, dtype="<i4", count=n, offset=8).astype(np.int64)
    logits = np.frombuffer(data, dtype="<f4", count=n * vocab, offset=8 + 4 * n)
    logits = logits.reshape(n, vocab).astype(np.float64)
    if not np.all(np.isfinite(logits)):
        raise ValueError(f"{path}: logits contain NaN or Inf")
    return n, vocab, ids, logits


def write_ids(path, ids):
    Path(path).write_text(" ".join(str(int(token)) for token in ids) + "\n")


def log_softmax(logits):
    maximum = np.max(logits, axis=1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def topk_indices(values, k):
    k = max(1, min(int(k), values.shape[1]))
    return np.argpartition(-values, k - 1, axis=1)[:, :k]


def compare(reference, candidate, ids, topk):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")

    diff = candidate - reference
    absdiff = np.abs(diff)
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    difference_norm = float(np.linalg.norm(diff))
    rel_l2 = difference_norm / max(reference_norm, 1e-300)
    cosine = float(np.sum(reference * candidate) / max(reference_norm * candidate_norm, 1e-300))

    per_position_ref = np.linalg.norm(reference, axis=1)
    per_position_rel = np.linalg.norm(diff, axis=1) / np.maximum(per_position_ref, 1e-300)

    reference_top = np.argmax(reference, axis=1)
    candidate_top = np.argmax(candidate, axis=1)
    agree = reference_top == candidate_top
    mismatches = int(np.sum(~agree))

    ref_topk = topk_indices(reference, topk)
    cand_topk = topk_indices(candidate, topk)
    k = ref_topk.shape[1]
    overlap = np.array([len(set(a) & set(b)) / k for a, b in zip(ref_topk, cand_topk)])

    logp_reference = log_softmax(reference)
    logp_candidate = log_softmax(candidate)
    kl = np.sum(np.exp(logp_reference) * (logp_reference - logp_candidate), axis=1)
    kl = np.maximum(kl, 0.0)

    ppl_reference = None
    ppl_candidate = None
    ppl_delta = None
    ppl_ratio = None
    if len(ids) >= 2:
        targets = np.asarray(ids[1:], dtype=np.int64)
        rows = np.arange(len(ids) - 1)
        nll_reference = -logp_reference[rows, targets]
        nll_candidate = -logp_candidate[rows, targets]
        ppl_reference = float(np.exp(np.mean(nll_reference)))
        ppl_candidate = float(np.exp(np.mean(nll_candidate)))
        ppl_delta = ppl_candidate - ppl_reference
        ppl_ratio = ppl_candidate / max(ppl_reference, 1e-300)

    return {
        "max_abs": float(np.max(absdiff)),
        "mean_abs": float(np.mean(absdiff)),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "rel_l2": rel_l2,
        "cosine": cosine,
        "top1": float(np.mean(agree)),
        "topk": float(np.mean(overlap)),
        "topk_k": k,
        "top1_mismatches": mismatches,
        "first_top1_mismatch": int(np.argmax(~agree)) if mismatches else None,
        "kl_mean": float(np.mean(kl)),
        "kl_max": float(np.max(kl)),
        "ppl_reference": ppl_reference,
        "ppl_candidate": ppl_candidate,
        "ppl_delta": ppl_delta,
        "ppl_ratio": ppl_ratio,
        "per_position_rel": per_position_rel,
        "reference_top": reference_top,
        "candidate_top": candidate_top,
    }


def print_metrics(title, reference_label, candidate_label, metrics):
    print(f"\n{title}")
    print(f"  {reference_label}  ->  {candidate_label}")
    print(f"  max_abs={metrics['max_abs']:.6g}  mean_abs={metrics['mean_abs']:.6g}  "
          f"rms={metrics['rms']:.6g}")
    print(f"  rel_l2={metrics['rel_l2']:.6g}  cosine={metrics['cosine']:.9f}")
    print(f"  top1={metrics['top1']:.6f}  top{metrics['topk_k']}={metrics['topk']:.6f}  "
          f"top1_mismatches={metrics['top1_mismatches']}")
    print(f"  KL mean={metrics['kl_mean']:.6g}  max={metrics['kl_max']:.6g}")
    if metrics["ppl_reference"] is not None:
        print(f"  PPL: {reference_label}={metrics['ppl_reference']:.6f}  "
              f"{candidate_label}={metrics['ppl_candidate']:.6f}  "
              f"delta={metrics['ppl_delta']:+.6f}  ratio={metrics['ppl_ratio']:.6f}")


def worst_positions(metrics, ids, count=5):
    order = np.argsort(-metrics["per_position_rel"])[:max(1, min(count, len(ids)))]
    result = []
    for position in order:
        result.append({
            "position": int(position),
            "token": int(ids[position]),
            "rel_l2": float(metrics["per_position_rel"][position]),
            "reference_top": int(metrics["reference_top"][position]),
            "candidate_top": int(metrics["candidate_top"][position]),
        })
    return result


def print_worst(metrics, ids, heading, count=5):
    print(f"\n  worst positions ({heading}):")
    for item in worst_positions(metrics, ids, count):
        print(f"    pos {item['position']:4d} token {item['token']:6d}  "
              f"top1 {item['reference_top']:6d} -> {item['candidate_top']:6d}  "
              f"rel_l2={item['rel_l2']:.6g}")


def scalar_metrics(metrics):
    return {
        key: metrics[key]
        for key in (
            "max_abs", "mean_abs", "rms", "rel_l2", "cosine", "top1", "topk",
            "topk_k", "top1_mismatches", "first_top1_mismatch", "kl_mean", "kl_max",
            "ppl_reference", "ppl_candidate", "ppl_delta", "ppl_ratio",
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Qwen3 GGUF model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="text prompt, tokenized by the C engine")
    source.add_argument("--ids-file", help="whitespace-separated token IDs")
    parser.add_argument("--ctx", type=int, help="C engine context capacity")
    parser.add_argument("--topk", type=int, default=5, help="top-k overlap (default: 5)")
    parser.add_argument("--skip-a8", action="store_true",
                        help="skip the Python W1A8 implementation-control run")
    parser.add_argument("--json", help="write a JSON report")
    parser.add_argument("--verbose", action="store_true", help="show subprocess commands/output")
    parser.add_argument("--impl-max-abs", type=float, default=1.5,
                        help="hard error when max |C - Python A8| exceeds this (default: 1.5)")
    parser.add_argument("--impl-rel-l2", type=float, default=5e-2,
                        help="hard error when C vs Python A8 relative L2 exceeds this (default: 5e-2)")
    parser.add_argument("--impl-ppl-ratio-max", type=float, default=2e-2,
                        help="hard error when |C/A8 PPL ratio - 1| exceeds this (default: 2e-2)")
    parser.add_argument("--impl-top1-min", type=float, default=0.99,
                        help="note when C/Python A8 top-1 agreement is below this (default: 0.99)")
    parser.add_argument("--impl-rel-l2-note", type=float, default=5e-3,
                        help="note when the implementation-control relative L2 exceeds this (default: 5e-3)")
    parser.add_argument("--total-rel-l2-warn", type=float, default=5e-2,
                        help="warn when C vs Python A32 relative L2 exceeds this (default: 5e-2)")
    parser.add_argument("--total-rel-l2-note", type=float, default=2e-2,
                        help="note when C vs Python A32 relative L2 exceeds this (default: 2e-2)")
    parser.add_argument("--total-top1-warn", type=float, default=0.99,
                        help="warn when C vs Python A32 top-1 agreement is below this (default: 0.99)")
    parser.add_argument("--total-ppl-ratio-warn", type=float, default=1.02,
                        help="warn when C/A32 PPL ratio exceeds this (default: 1.02)")
    parser.add_argument("--unexpected-factor", type=float, default=3.0,
                        help="warn when the total gap exceeds this multiple of the Python A8->A32 gap")
    parser.add_argument("--unexpected-margin", type=float, default=1e-2,
                        help="absolute margin added to the unexpected-gap test")
    args = parser.parse_args()

    if args.ctx is not None and args.ctx <= 0:
        parser.error("--ctx must be positive")
    if args.topk <= 0:
        parser.error("--topk must be positive")

    model = Path(args.model).resolve()
    if not model.is_file():
        parser.error(f"model does not exist: {model}")
    ids_file = Path(args.ids_file).resolve() if args.ids_file else None
    if ids_file is not None and not ids_file.is_file():
        parser.error(f"ids file does not exist: {ids_file}")
    if not REFERENCE.is_file():
        parser.error(f"reference tool is missing: {REFERENCE}")

    if not DUMPER.is_file():
        print(f"[build] {DUMPER} is missing; running make", flush=True)
        run_command(["make", "-C", HERE], "build dump-logits", args.verbose)

    print(f"model: {model}")
    if args.prompt is not None:
        print(f"prompt: {args.prompt!r}")
    else:
        print(f"ids: {ids_file}")
    print("C engine: W1A8; Python reference: A32 (headline) and A8 (control)")

    with tempfile.TemporaryDirectory(prefix="qwen3-precision-") as tmp:
        tmp = Path(tmp)
        c_path = tmp / "c.bin"
        c_command = [DUMPER, "--model", model, "--out", c_path]
        if args.prompt is not None:
            c_command += ["--prompt", args.prompt]
        else:
            c_command += ["--ids-file", ids_file]
        if args.ctx is not None:
            c_command += ["--ctx", str(args.ctx)]
        run_command(c_command, "C engine", args.verbose)

        n, vocab, ids, c_logits = read_dump(c_path)
        if n <= 0:
            raise SystemExit("error: C dump contains no positions")
        ids_path = tmp / "ids.txt"
        write_ids(ids_path, ids)

        def run_reference(activation):
            out_path = tmp / f"ref-{activation}.bin"
            command = [
                sys.executable, REFERENCE, "--model", model, "--ids-file", ids_path,
                "--out", out_path, "--activation", activation, "--quiet",
            ]
            run_command(command, f"Python {activation.upper()}", args.verbose)
            ref_n, ref_vocab, ref_ids, ref_logits = read_dump(out_path)
            if ref_n != n or ref_vocab != vocab or not np.array_equal(ref_ids, ids):
                raise SystemExit(
                    f"error: reference shape/ids mismatch for {activation}: "
                    f"n={ref_n}/{n} vocab={ref_vocab}/{vocab}")
            return ref_logits

        a32_logits = run_reference("a32")
        a8_logits = None if args.skip_a8 else run_reference("a8")

    total = compare(a32_logits, c_logits, ids, args.topk)
    print_metrics("C W1A8 vs Python A32 (total gap)", "Python A32", "C W1A8", total)

    if a8_logits is not None:
        implementation = compare(a8_logits, c_logits, ids, args.topk)
        quantization = compare(a32_logits, a8_logits, ids, args.topk)
        print_metrics("C W1A8 vs Python W1A8 (implementation control)",
                      "Python W1A8", "C W1A8", implementation)
        print_metrics("Python W1A8 vs Python A32 (expected quantization loss)",
                      "Python A32", "Python W1A8", quantization)
        print_worst(implementation, ids, "implementation control")

    print_worst(total, ids, "C vs Python A32")

    errors = []
    warnings = []
    notes = []

    if a8_logits is None:
        notes.append("Python A8 implementation control skipped; only the total gap is reported.")
    else:
        expected_rel = quantization["rel_l2"]
        implementation_reasons = []
        if implementation["max_abs"] > args.impl_max_abs:
            implementation_reasons.append(
                f"max_abs={implementation['max_abs']:.6g} > {args.impl_max_abs:g}")
        if implementation["rel_l2"] > args.impl_rel_l2:
            implementation_reasons.append(
                f"rel_l2={implementation['rel_l2']:.6g} > {args.impl_rel_l2:g}")
        if (implementation["ppl_ratio"] is not None and
                abs(implementation["ppl_ratio"] - 1.0) > args.impl_ppl_ratio_max):
            implementation_reasons.append(
                f"PPL ratio={implementation['ppl_ratio']:.6f} "
                f"(|ratio-1| > {args.impl_ppl_ratio_max:g})")
        if implementation_reasons:
            errors.append(
                "C W1A8 does not match Python W1A8: " + "; ".join(implementation_reasons))
        elif (implementation["rel_l2"] > args.impl_rel_l2_note or
              implementation["top1"] < args.impl_top1_min):
            notes.append(
                "C vs Python A8 control drift: "
                f"rel_l2={implementation['rel_l2']:.6g}, "
                f"top1={implementation['top1']:.6f}, "
                f"PPL ratio={implementation['ppl_ratio'] if implementation['ppl_ratio'] is None else round(implementation['ppl_ratio'], 6)}. "
                "Batched-vs-incremental decoding and NumPy/C math-library ordering can cause this; "
                "check the implementation-control worst positions above.")

        unexpected_threshold = (
            args.unexpected_factor * max(expected_rel, 1e-6) + args.unexpected_margin)
        if (total["rel_l2"] > unexpected_threshold and
                total["rel_l2"] > args.total_rel_l2_note):
            warnings.append(
                "C vs A32 gap is larger than the Python A8->A32 gap: "
                f"total rel_l2={total['rel_l2']:.6g}, "
                f"Python A8->A32 rel_l2={expected_rel:.6g}")

    ppl_ratio = total["ppl_ratio"]
    total_reasons = []
    if total["rel_l2"] > args.total_rel_l2_warn:
        total_reasons.append(
            f"rel_l2={total['rel_l2']:.6g} > {args.total_rel_l2_warn:g}")
    if total["top1"] < args.total_top1_warn:
        total_reasons.append(
            f"top1={total['top1']:.6f} < {args.total_top1_warn:g}")
    if ppl_ratio is not None and ppl_ratio > args.total_ppl_ratio_warn:
        total_reasons.append(
            f"PPL ratio={ppl_ratio:.6f} > {args.total_ppl_ratio_warn:g}")
    if total_reasons:
        warnings.append(
            "W1A8 vs full-precision activations exceeds the warning threshold: " +
            "; ".join(total_reasons))
    elif total["rel_l2"] > args.total_rel_l2_note:
        notes.append(
            f"C vs A32 logit drift is moderate (rel_l2={total['rel_l2']:.6g}) "
            "but top-1 and PPL are within warning thresholds.")
    if (ppl_ratio is not None and ppl_ratio < 1.0 and
            abs(ppl_ratio - 1.0) > abs(args.total_ppl_ratio_warn - 1.0)):
        notes.append(
            f"PPL ratio={ppl_ratio:.6f} is below 1.0 on this sequence; short-sequence PPL "
            "is noisy and a lower value is not evidence of better accuracy.")

    if errors:
        status = "ERROR"
        banner("!!! ERROR: C ENGINE AND PYTHON W1A8 DISAGREE !!!", RED)
        for message in errors:
            print(paint("  " + message, RED))
        print(paint("  The implementation-control gap is unexpected; inspect the C path "
                    "before trusting the total gap.", RED))
    elif warnings:
        status = "WARNING"
        banner("!!! WARNING: W1A8 VS FULL PRECISION GAP IS LARGE !!!", YELLOW)
        for message in warnings:
            print(paint("  " + message, YELLOW))
        if a8_logits is not None:
            print(paint(
                "  Compare the Python A8->A32 metrics above: if they are similar, "
                "the total gap is expected activation-quantization loss, not a C bug.", YELLOW))
    else:
        status = "OK"
        banner("OK: C W1A8 is within thresholds of the Python reference", GREEN)

    if errors and warnings:
        print(paint("\nwarnings:", YELLOW))
        for message in warnings:
            print(paint("  " + message, YELLOW))
    if notes:
        print(paint("\nnotes:", CYAN))
        for message in notes:
            print(paint("  " + message, CYAN))

    print(f"\nresult: {status}")

    if args.json:
        report = {
            "model": str(model),
            "prompt": args.prompt,
            "ids_file": str(ids_file) if ids_file else None,
            "n": int(n),
            "vocab": int(vocab),
            "ids": [int(token) for token in ids],
            "c_activation": "w1a8",
            "python_reference": "a32",
            "thresholds": {
                "impl_max_abs": args.impl_max_abs,
                "impl_rel_l2": args.impl_rel_l2,
                "impl_ppl_ratio_max": args.impl_ppl_ratio_max,
                "impl_top1_min": args.impl_top1_min,
                "impl_rel_l2_note": args.impl_rel_l2_note,
                "total_rel_l2_warn": args.total_rel_l2_warn,
                "total_rel_l2_note": args.total_rel_l2_note,
                "total_top1_warn": args.total_top1_warn,
                "total_ppl_ratio_warn": args.total_ppl_ratio_warn,
                "unexpected_factor": args.unexpected_factor,
                "unexpected_margin": args.unexpected_margin,
            },
            "metrics": {
                "total_c_vs_python_a32": scalar_metrics(total),
                "implementation_c_vs_python_a8": (
                    scalar_metrics(implementation) if a8_logits is not None else None),
                "quantization_python_a8_vs_a32": (
                    scalar_metrics(quantization) if a8_logits is not None else None),
            },
            "worst_total": worst_positions(total, ids),
            "worst_implementation": (
                worst_positions(implementation, ids) if a8_logits is not None else None),
            "errors": errors,
            "warnings": warnings,
            "notes": notes,
            "status": status,
        }
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")

    return 2 if status == "ERROR" else 1 if status == "WARNING" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
