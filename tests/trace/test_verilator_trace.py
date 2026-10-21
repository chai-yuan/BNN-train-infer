#!/usr/bin/env python3
"""Test runtime VCD tracing through the public BNN API, using an existing archive.

Build the normal ACCE=verilator archive first (trace support is always built in).
Run from the repository root to leave inspectable waveforms:
    python3 tests/trace/test_verilator_trace.py --output-dir build/waves
or through the category Makefile:
    make -C tests/trace waves

Optional: --library build/verilator/libbnn-engine.a --cc cc --cxx c++
Compiler commands may include wrappers, e.g. --cc 'ccache gcc'. Compilation and
linking always use a temporary directory; this script never invokes make.
Without --output-dir, all waveforms are temporary too. An existing output
directory is preserved, and only files created by this invocation are parsed.
Success-path flushing is checked while both devices are alive. Timeout flushing
cannot be triggered through the public API, which has no cycle-limit argument.
"""

import argparse
import hashlib
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import tempfile
import threading


TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parents[1]
TRACE_ENV = "BNN_RTL_TRACE_DIR"
CHECKPOINTS = (
    "opened", "first-dot", "second-dot", "first-gemv", "second-gemv",
    "repeated", "first-closed", "second-survives", "closed",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(value):
    result = shlex.split(value)
    if not result:
        raise argparse.ArgumentTypeError("compiler command must not be empty")
    return result


def run(argv, *, cwd, env=None):
    argv = [str(arg) for arg in argv]
    print("+ " + shlex.join(argv), flush=True)
    subprocess.run(argv, check=True, cwd=cwd, env=env, timeout=120)


def environment(directory=None):
    env = os.environ.copy()
    env.pop(TRACE_ENV, None)
    if directory is not None:
        env[TRACE_ENV] = str(directory)
    return env


def file_signature(path):
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return stat.st_dev, stat.st_ino, stat.st_size, digest.digest()


def snapshot(directory):
    return {path.name: file_signature(path) for path in directory.iterdir()
            if path.is_file()}


def preserved(directory, before):
    for name, signature in before.items():
        path = directory / name
        require(path.is_file() and file_signature(path) == signature,
                f"pre-existing file changed or disappeared: {path}")


def validate_vcd(path, expected_jobs):
    text = path.read_text(encoding="ascii")
    require(re.search(r"\$timescale\s+1\s*ps\s+\$end", text) is not None,
            f"{path}: expected 1ps timescale")
    sections = re.split(r"\$enddefinitions\s+\$end", text, maxsplit=1)
    require(len(sections) == 2, f"{path}: missing $enddefinitions")
    header, body = sections
    scopes = []
    modules = set()
    signals = {}
    for line in header.splitlines():
        words = line.split()
        if words[:1] == ["$scope"]:
            require(len(words) >= 4, f"{path}: malformed scope")
            scopes.append(words[2])
            if words[1] == "module":
                modules.add(tuple(scopes))
        elif words[:1] == ["$upscope"]:
            require(bool(scopes), f"{path}: unbalanced scope")
            scopes.pop()
        elif words[:1] == ["$var"]:
            require(len(words) >= 6, f"{path}: malformed variable")
            # Prefer the top-level port over aliases inside child modules.
            name = words[4]
            candidate = (len(scopes), words[3], int(words[2]))
            if name not in signals or candidate[0] < signals[name][0]:
                signals[name] = candidate
    require(not scopes, f"{path}: unclosed scope")
    require(any("bnn_core_top" in scope and scope[-1] != "bnn_core_top"
                for scope in modules), f"{path}: missing core submodule hierarchy")
    names = ("clock", "reset", "cmd_valid", "res_data")
    for name in names:
        require(name in signals, f"{path}: missing {name}")
    require(signals["clock"][2] == 1 and signals["res_data"][2] == 32,
            f"{path}: unexpected clock/result widths")
    by_code = {}
    for name in names:
        by_code.setdefault(signals[name][1], []).append(name)
    changes = {name: [] for name in names}
    timestamps = []
    now = None
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            now = int(line[1:])
            require(not timestamps or now > timestamps[-1],
                    f"{path}: timestamps do not strictly increase at {now}")
            timestamps.append(now)
            continue
        if line.startswith("$"):
            continue
        if line[0] in "bB":
            fields = line[1:].split()
            require(len(fields) == 2, f"{path}: malformed vector change")
            value, code = fields
        elif line[0] in "01xXzZ":
            value, code = line[0], line[1:]
        else:
            continue
        if code in by_code:
            require(now is not None, f"{path}: signal before first timestamp")
            require(set(value.lower()) <= {"0", "1"},
                    f"{path}: unknown value for {by_code[code]}")
            number = int(value, 2)
            for name in by_code[code]:
                if not changes[name] or changes[name][-1][1] != number:
                    changes[name].append((now, number))
    require(len(timestamps) > 10 and timestamps[0] == 0,
            f"{path}: trace must start at reset time zero and advance")
    clock = changes["clock"]
    require(len(clock) > 10 and {value for _, value in clock} == {0, 1},
            f"{path}: clock does not toggle")
    # Ignore the initial sample, which may share time zero with the first edge.
    require(all(b[0] - a[0] == 5000 for a, b in zip(clock[1:], clock[2:])),
            f"{path}: expected 5000ps half-period (10ns clock)")
    commands = [time for time, value in changes["cmd_valid"] if value == 1]
    require(len(commands) == expected_jobs,
            f"{path}: saw {len(commands)} commands, expected {expected_jobs}; "
            "possible missing jobs, trace truncation or device cross-talk")
    reset = changes["reset"]
    require(reset and reset[0] == (0, 1) and
            any(value == 0 and time <= commands[0] for time, value in reset),
            f"{path}: initial reset assertion/deassertion was not traced")
    results = {value for _, value in changes["res_data"]}
    require({128, (1 << 32) - 128} <= results,
            f"{path}: missing +128/-128 RTL results")
    print(f"VCD OK: {path} ({len(timestamps)} timestamps, "
          f"{expected_jobs} jobs, 1ps / 10ns clock)", flush=True)


def enabled_run(runner, directory, work):
    old_names = {path.name for path in directory.iterdir()}
    old = snapshot(directory)
    # Exclusive creation: never overwrite a user's file, including an old canary.
    fd, canary_name = tempfile.mkstemp(prefix="bnn-core-", suffix=".vcd", dir=directory)
    with os.fdopen(fd, "wb") as stream:
        stream.write(b"existing VCD canary: must never be overwritten\n")
    canary = Path(canary_name)
    protected = {**old, canary.name: file_signature(canary)}
    excluded = old_names | {canary.name}
    messages = queue.Queue()
    argv = [str(runner), "--checkpoints"]
    print("+ " + shlex.join(argv) + f" ({TRACE_ENV}={directory})", flush=True)
    with tempfile.TemporaryFile(mode="w+t") as errors:
        proc = subprocess.Popen(argv, cwd=work, env=environment(directory),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=errors, text=True, bufsize=1)

        def collect():
            try:
                for line in proc.stdout:
                    messages.put(line.rstrip("\n"))
            finally:
                messages.put(None)

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        paths = None
        previous = {}
        first = second = None
        first_before_close = None
        final_data = None
        try:
            for stage in CHECKPOINTS:
                try:
                    line = messages.get(timeout=30)
                except queue.Empty as error:
                    raise RuntimeError(f"runner stalled before {stage}") from error
                require(line == f"CHECKPOINT {stage}",
                        f"expected checkpoint {stage}, got {line!r}")
                names = {p.name for p in directory.iterdir()} - excluded
                require(len(names) == 2 and all(
                    re.fullmatch(r"bnn-core-[A-Za-z0-9]{6}\.vcd", name) for name in names),
                    f"{stage}: expected two unique bnn-core-XXXXXX.vcd files, got {sorted(names)}")
                current_paths = {directory / name for name in names}
                if paths is None:
                    paths = current_paths
                    require(len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) == 2,
                            "two sims share the same VCD inode")
                require(paths == current_paths, f"{stage}: VCD set changed during device lifetime")
                data = {p: p.read_bytes() for p in paths}
                for path, content in previous.items():
                    require(data[path].startswith(content),
                            f"{stage}: trace was truncated or rewritten: {path}")
                preserved(directory, protected)
                if stage == "first-dot":
                    grown = [p for p in paths if len(data[p]) > len(previous[p])]
                    require(len(grown) == 1, "first dot must flush only its own trace")
                    first = grown[0]
                    second = next(p for p in paths if p != first)
                    require(data[second] == previous[second], "first dot wrote second sim's VCD")
                    validate_vcd(first, 1)
                elif stage in ("second-dot", "first-gemv", "second-gemv", "second-survives"):
                    active = first if stage == "first-gemv" else second
                    idle = second if active == first else first
                    require(data[active].startswith(previous[active]) and
                            len(data[active]) > len(previous[active]),
                            f"{stage}: successful job did not append/flush its VCD")
                    require(data[idle] == previous[idle], f"{stage}: other sim's VCD changed")
                    validate_vcd(active, {"second-dot": 1, "first-gemv": 2,
                                          "second-gemv": 2, "second-survives": 6}[stage])
                elif stage == "repeated":
                    validate_vcd(first, 4)
                    validate_vcd(second, 4)
                elif stage == "first-closed":
                    require(data[second] == previous[second], "closing first sim changed second VCD")
                    first_before_close = data[first]
                    validate_vcd(first, 4)
                elif stage == "closed":
                    require(data[first] == first_before_close, "closed sim's trace changed")
                    validate_vcd(first, 4)
                    validate_vcd(second, 6)
                    final_data = data
                previous = data
                proc.stdin.write("\n")
                proc.stdin.flush()
            proc.stdin.close()
            require(proc.wait(timeout=30) == 0, "public API runner failed")
            reader.join(timeout=5)
            require(not reader.is_alive(), "runner output did not close")
            while not messages.empty():
                line = messages.get_nowait()
                if line is not None:
                    print(line, flush=True)
            require(final_data == {p: p.read_bytes() for p in paths},
                    "VCD changed after DeviceClose returned")
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
            reader.join(timeout=5)
            errors.seek(0)
            diagnostic = errors.read()
            if diagnostic:
                print(diagnostic, end="", flush=True)
            raise
        finally:
            if not proc.stdin.closed:
                proc.stdin.close()
            proc.stdout.close()
    preserved(directory, protected)
    canary.unlink()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path,
                        default=ROOT / "build/verilator/libbnn-engine.a",
                        help="already-built complete Verilator archive")
    parser.add_argument("--cc", type=command, default=os.environ.get("CC", "cc"))
    parser.add_argument("--cxx", type=command, default=os.environ.get("CXX", "c++"))
    parser.add_argument("--output-dir", type=Path,
                        help="keep new VCDs here; preserve all pre-existing files")
    args = parser.parse_args()
    library = args.library.expanduser().resolve()
    if not library.is_file():
        parser.error(f"archive does not exist: {library}; build ACCE=verilator first")
    with tempfile.TemporaryDirectory(prefix="bnn-trace-test-") as temporary:
        work = Path(temporary)
        obj = work / "test_verilator_trace.o"
        runner = work / "trace-runner"
        run([*args.cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic",
             "-fno-fast-math", "-ffp-contract=off", "-I", ROOT / "bnn-accelerator",
             "-c", TEST_DIR / "test_verilator_trace.c", "-o", obj], cwd=work)
        run([*args.cxx, obj, library, "-pthread", "-latomic", "-lm", "-o", runner], cwd=work)
        for label, value in (("unset", None), ("empty", "")):
            sandbox = work / label
            sandbox.mkdir()
            run([runner], cwd=sandbox, env=environment(value))
            require(not list(sandbox.iterdir()), f"{label}: tracing created files while disabled")
            print(f"Trace {label}: no files generated", flush=True)
        missing = work / "missing-parent" / "waves"
        before = snapshot(work)
        run([runner, "--expect-create-failure"], cwd=work, env=environment(missing))
        require(not missing.parent.exists(), "failed create left a directory behind")
        require(snapshot(work) == before, "failed create changed existing work files")
        require(not list(work.rglob("*.vcd")), "failed create left a VCD behind")
        print("Missing trace directory: BNN_EIO, no residual files", flush=True)
        blocked = work / "unwritable"
        blocked.mkdir(mode=0o555)
        try:
            if os.access(blocked, os.W_OK):
                print("SKIP unwritable-directory check: current credentials bypass directory permissions",
                      flush=True)
            else:
                run([runner, "--expect-create-failure"], cwd=work, env=environment(blocked))
                require(not list(blocked.iterdir()), "unwritable directory retained a trace file")
                print("Unwritable trace directory: BNN_EIO, no residual files", flush=True)
        finally:
            blocked.chmod(0o755)
        directory = (args.output_dir.expanduser().resolve() if args.output_dir else work / "waves")
        directory.mkdir(parents=True, exist_ok=True)
        paths = enabled_run(runner, directory, work)
        print("Trace tests passed: disabled/create errors, canary preservation, live flush, "
              "reset, hierarchy, clock, results and per-device lifecycle/isolation.", flush=True)
        if args.output_dir:
            for path in sorted(paths):
                print(f"Waveform retained: {path}", flush=True)
        else:
            print("Temporary waveforms removed on exit; use --output-dir build/waves to keep them.",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"trace tests failed: {error}") from error
