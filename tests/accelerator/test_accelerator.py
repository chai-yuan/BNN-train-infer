#!/usr/bin/env python3
"""Compile and run the standalone soft accelerator API v2 regression tests.

From the repository root:
    python3 tests/accelerator/test_accelerator.py
or through the category Makefile:
    make -C tests/accelerator test

Only Python 3 and a C11 compiler are needed; no RTL, archive or root build is used.
Build products live in a temporary directory and are removed after the run.
Compiler commands can contain wrappers, e.g. --cc 'ccache gcc'. Extra compile/link
flags can be supplied with --cflags='-O1 -g -fsanitize=address,undefined'.
C11 and the strict floating-point flags are enforced after user flags.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parents[1]


def command(value):
    result = shlex.split(value)
    if not result:
        raise argparse.ArgumentTypeError("compiler command must not be empty")
    return result


def run(argv):
    argv = [str(arg) for arg in argv]
    print("+ " + shlex.join(argv), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT, timeout=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", type=command, default=os.environ.get("CC", "cc"),
                        help="C compiler command, including wrappers (default: CC or cc)")
    parser.add_argument("--cflags", type=shlex.split, default=os.environ.get("CFLAGS", ""),
                        help="extra compile/link flags (default: CFLAGS)")
    args = parser.parse_args()

    include = ROOT / "bnn-accelerator"
    # User optimization/sanitizer flags must not relax the oracle's FP rules.
    flags = ["-O2", "-Wall", "-Wextra", "-Wpedantic", "-I", include,
             *args.cflags, "-std=c11", "-fno-fast-math", "-ffp-contract=off"]
    with tempfile.TemporaryDirectory(prefix="bnn-accelerator-test-") as tmp:
        runner = Path(tmp) / "soft-runner"
        run([*args.cc, *flags, TEST_DIR / "test_bnn_accelerator.c",
             include / "soft-bnn-accelerator.c", "-lm", "-o", runner])
        run([runner])
    print("Soft accelerator API v2 tests passed (independent scalar oracle).")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"accelerator tests failed: {error}") from error
