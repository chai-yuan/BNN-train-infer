#!/usr/bin/env python3
"""Differential test for the Qwen3/Q1_0 software backend.

The NumPy reference in tools/bonsai-ref.py emulates the same W1A8 contract as
bnn-sw/model-qwen3.c.  This test writes a --ref file and requires the C result
to agree on every argmax within the CLI's relative-error limit.

The GGUF model is not part of the repository; pass --model or set BONSAI_GGUF.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run(command, timeout):
    print("+", " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               timeout=timeout)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(f"command failed with exit code {completed.returncode}")
    return completed.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("BONSAI_GGUF"))
    parser.add_argument("--binary", default=str(ROOT / "bnn-infer"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ids", default="785 6722 315 9625 374",
                        help="Qwen3 token IDs for 'The capital of France is'")
    parser.add_argument("--generation", action="store_true",
                        help="also exercise the KV-cache --ids generation path")
    args = parser.parse_args()

    if not args.model:
        print("SKIP: set BONSAI_GGUF or pass --model", file=sys.stderr)
        return 0
    if not os.path.exists(args.model):
        print(f"SKIP: model not found: {args.model}", file=sys.stderr)
        return 0
    if not os.path.exists(args.binary):
        raise SystemExit(f"missing {args.binary}; run 'make ACCE=soft' first")

    with tempfile.TemporaryDirectory(prefix="qwen3-test-") as temporary:
        reference = os.path.join(temporary, "ref.bin")
        ids_file = os.path.join(temporary, "ids.txt")

        run([args.python, str(ROOT / "tools" / "bonsai-ref.py"),
             "--model", args.model, "--ids", args.ids, "--out", reference,
             "--topk", "1"], timeout=900)
        output = run([args.binary, "--profile", "--ref", reference, args.model],
                     timeout=900)
        if "OK" not in output or "argmax agree" not in output:
            raise SystemExit("reference comparison did not report OK")

        if args.generation:
            with open(ids_file, "w", encoding="utf-8") as file:
                file.write(args.ids + "\n")
            generated = run([args.binary, "--profile", "--ids", ids_file,
                             "--steps", "2", args.model], timeout=900)
            if "generate ids:" not in generated:
                raise SystemExit("generation did not start from --ids")

    print("Qwen3 Q1_0 differential test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
