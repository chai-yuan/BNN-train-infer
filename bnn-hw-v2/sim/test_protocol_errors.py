#!/usr/bin/env python3
"""Check intentional RTL fatals in isolated processes; never accept an unrelated crash."""

import resource
import subprocess
import sys


CASES = {
    "zero-read": "zero-length command",
    "zero-write": "zero-length command",
    "unaligned-read": "unaligned address",
    "unaligned-write": "unaligned address",
    "range-read": "burst range check failed",
    "range-write": "burst range check failed",
    "early-last": "last disagrees with remaining beats",
    "missing-last": "last disagrees with remaining beats",
    "unstable-write": "DMA write data changed under backpressure",
    "unstable-read-cmd": "DMA read command changed under backpressure",
    "unstable-write-cmd": "DMA write command changed under backpressure",
}


def no_core_dump():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_protocol_errors.py DMA_TEST_EXECUTABLE")
    for name, expected in CASES.items():
        try:
            result = subprocess.run(
                [sys.argv[1], "--negative", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
                check=False,
                preexec_fn=no_core_dump,
            )
        except subprocess.TimeoutExpired as error:
            raise SystemExit(f"FAIL {name}: timed out") from error
        if result.returncode == 0 or expected not in result.stdout:
            print(result.stdout, file=sys.stderr)
            raise SystemExit(f"FAIL {name}: expected nonzero exit with {expected!r}")
        print(f"PASS expected protocol failure: {name}")
    print(f"PASS {len(CASES)} protocol error cases")


if __name__ == "__main__":
    main()
