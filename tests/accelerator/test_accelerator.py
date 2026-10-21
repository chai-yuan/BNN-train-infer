#!/usr/bin/env python3
"""Compile isolated accelerator runners against an already-built RTL archive.

From the repository root:
    python3 tests/accelerator/test_accelerator.py \
        --library build/verilator/libbnn-engine.a \
        --cc cc --cxx c++ --verilator verilator
or through the category Makefile:
    make -C tests/accelerator test

Add --trace-dir DIR (or set BNN_RTL_TRACE_DIR) to keep VCD waveforms from every
Verilator runner. The directory is created if missing; one VCD is produced per
simulated device, so the public API, bridge and context runners each add files.
Pre-existing VCDs in the directory are left untouched.

The archive must contain the backend, bridge, model and Verilator runtime objects.
This script never invokes make or changes bnn-infer. All build products and dumps
live in a temporary directory; no generated headers or third-party Python modules
are needed. The context interoperation test uses runtime headers discovered via
verilator --getenv VERILATOR_ROOT; use the same Verilator as the archive build.
Compiler commands can contain wrappers, e.g. --cc 'ccache gcc'.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parents[1]
TRACE_ENV = "BNN_RTL_TRACE_DIR"


def command(value):
    result = shlex.split(value)
    if not result:
        raise argparse.ArgumentTypeError("compiler command must not be empty")
    return result


def run(argv, env=None):
    argv = [str(arg) for arg in argv]
    print("+ " + shlex.join(argv), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT, timeout=300, env=env)


def run_with_trace(label, argv, trace_dir, trace_env):
    if trace_dir is None:
        run(argv)
        return set()
    before = set(trace_dir.glob("*.vcd"))
    run(argv, env=trace_env)
    created = sorted(set(trace_dir.glob("*.vcd")) - before)
    for path in created:
        print(f"Waveform [{label}]: {path}", flush=True)
    return set(created)


def verilator_includes(verilator):
    argv = [*verilator, "--getenv", "VERILATOR_ROOT"]
    print("+ " + shlex.join(argv), flush=True)
    result = subprocess.run(argv, check=True, cwd=ROOT, timeout=30,
                            capture_output=True, text=True)
    root_value = result.stdout.strip()
    if not root_value:
        raise RuntimeError("verilator returned an empty VERILATOR_ROOT")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    include = root / "include"
    if not (include / "verilated.h").is_file() or not (include / "vltstd").is_dir():
        raise RuntimeError(f"Verilator runtime headers not found under {include}")
    return ["-isystem", include, "-isystem", include / "vltstd"]


def compare_dumps(soft_path, rtl_path):
    soft = soft_path.read_bytes()
    rtl = rtl_path.read_bytes()
    if not soft or not rtl:
        raise RuntimeError("a successful runner produced an empty output dump")
    if soft != rtl:
        offset = next((i for i, (a, b) in enumerate(zip(soft, rtl)) if a != b),
                      min(len(soft), len(rtl)))
        begin = max(0, offset - 8)
        end = offset + 12
        raise RuntimeError(
            f"soft/RTL binary mismatch at byte {offset}; "
            f"lengths soft={len(soft)}, RTL={len(rtl)}\n"
            f"soft[{begin}:{end}]: {soft[begin:end].hex(' ')}\n"
            f"RTL [{begin}:{end}]: {rtl[begin:end].hex(' ')}"
        )
    print(f"soft/RTL byte comparison: {len(soft)} bytes identical "
          "(int32 and FP32 bit patterns)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path,
                        default=ROOT / "build/verilator/libbnn-engine.a",
                        help="existing complete Verilator static archive")
    parser.add_argument("--cc", type=command, default=os.environ.get("CC", "cc"),
                        help="C compiler command (default: CC or cc)")
    parser.add_argument("--cxx", type=command, default=os.environ.get("CXX", "c++"),
                        help="C++ compiler/linker command (default: CXX or c++)")
    parser.add_argument("--verilator", type=command, default=os.environ.get("VERILATOR", "verilator"),
                        help="Verilator command for runtime headers (default: VERILATOR or verilator)")
    parser.add_argument("--trace-dir", type=Path,
                        help="keep VCD waveforms here (created if missing); "
                             "BNN_RTL_TRACE_DIR is honored when this option is absent")
    args = parser.parse_args()
    library = args.library.expanduser().resolve()
    if not library.is_file():
        parser.error(f"archive does not exist: {library}; build ACCE=verilator first")

    trace_value = args.trace_dir if args.trace_dir is not None else os.environ.get(TRACE_ENV, "")
    trace_dir = None
    trace_env = None
    if trace_value:
        trace_dir = Path(trace_value).expanduser().resolve()
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_env = os.environ.copy()
        trace_env[TRACE_ENV] = str(trace_dir)
        print(f"Waveform trace dir: {trace_dir}", flush=True)

    include = ROOT / "bnn-accelerator"
    flags = ["-O2", "-Wall", "-Wextra", "-Wpedantic",
             "-fno-fast-math", "-ffp-contract=off", "-I", include]
    c_flags = ["-std=c99", *flags]
    cxx_flags = ["-std=c++17", *flags]
    context_includes = verilator_includes(args.verilator)
    with tempfile.TemporaryDirectory(prefix="bnn-accelerator-test-") as tmp:
        tmp = Path(tmp)
        obj = tmp / "test_bnn_accelerator.o"
        soft = tmp / "soft-runner"
        rtl = tmp / "verilator-runner"
        bridge = tmp / "bridge-runner"
        context = tmp / "context-runner"
        soft_dump = tmp / "soft.bin"
        rtl_dump = tmp / "rtl.bin"

        # Compile the public API test as C once; link the same object to each backend.
        run([*args.cc, *c_flags, "-c", TEST_DIR / "test_bnn_accelerator.c", "-o", obj])
        run([*args.cc, *c_flags, obj, include / "soft-bnn-accelerator.c", "-lm", "-o", soft])
        run([*args.cxx, obj, library, "-pthread", "-latomic", "-lm", "-o", rtl])
        run([*args.cxx, *cxx_flags, TEST_DIR / "test_verilator_bridge.cpp",
             library, "-pthread", "-latomic", "-lm", "-o", bridge])
        run([*args.cxx, *cxx_flags, *context_includes, TEST_DIR / "test_verilator_context.cpp",
             library, "-pthread", "-latomic", "-lm", "-o", context])

        run([soft, soft_dump])
        created = run_with_trace("public API", [rtl, rtl_dump], trace_dir, trace_env)
        compare_dumps(soft_dump, rtl_dump)
        created |= run_with_trace("bridge", [bridge], trace_dir, trace_env)
        created |= run_with_trace("context", [context], trace_dir, trace_env)
        if trace_dir is not None:
            print(f"Waveforms: {len(created)} new VCD file(s) in {trace_dir}", flush=True)
    print("Accelerator differential, standalone bridge and runtime context tests passed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"accelerator tests failed: {error}") from error
