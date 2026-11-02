#!/usr/bin/env python3
"""End-to-end CLI, binary memory I/O, VCD, instruction execution, and error checks."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def pack_weight(half_scale, signs_bytes):
    assert len(signs_bytes) == 16
    return struct.pack("<H", half_scale) + signs_bytes


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_cli.py CORE_SIMULATOR")
    executable = Path(sys.argv[1]).resolve()

    def run(*arguments, error=None):
        result = subprocess.run(
            [str(executable), *map(str, arguments)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        if error is None:
            if result.returncode != 0:
                raise RuntimeError(result.stdout)
        elif result.returncode == 0 or error not in result.stdout:
            raise RuntimeError(f"expected failure {error!r}:\n{result.stdout}")
        return result.stdout

    with tempfile.TemporaryDirectory(prefix="cli-", dir=executable.parent.parent) as temporary:
        directory = Path(temporary)
        source = directory / "input.bin"
        output = directory / "output.bin"
        trace = directory / "idle.vcd"
        empty = directory / "empty.bin"
        source.write_bytes(bytes(range(32)))
        empty.write_bytes(b"")
        assert "--commands" in run("--help")
        message = run(
            "--memory-size", "01024", "--read-base", "0x1000", "--write-base", "0x2000",
            "--load", "0x1000", source, "--dump", "0x2000", "32", output,
            "--idle", "08", "--trace", trace,
        )
        assert "idle smoke only" in message
        assert output.read_bytes() == bytes(32), "write store aliased the loaded read store"
        assert "$enddefinitions" in trace.read_text(), "missing VCD header"
        run("--memory-size", "010", "--load", "9", empty, "--idle", "0")
        run("--memory-size", "08", "--idle", "0")
        run("--memory-size", "-1", error="invalid unsigned number")
        run("--rd-latency", "4294967296", error="exceeds uint32_t")
        run("--load", "0xffff0", source, error="out of range")
        run("--load", "0", directory / "missing.bin", error="cannot open input file")
        run("--dump", "0xfffff", "32", output, error="out of range")
        run("--commands", directory / "missing.hex", error="cannot open command file")
        commands = directory / "commands.hex"
        commands.write_text("# comment only\n\n")
        run("--commands", commands, "--idle", "0")
        commands.write_text("0000000000000001 extra\n")
        run("--commands", commands, error="extra text on command line 1")

        # 1. End-to-end INT8 M=2, N=5 task with 2D source/destination strides and tail mask.
        weights_int8 = directory / "weights_int8.bin"
        input_int8 = directory / "input_int8.bin"
        output_int8 = directory / "output_int8.bin"
        task_trace = directory / "task.vcd"
        weights_int8.write_bytes(b"".join([
            pack_weight(0x3C00, b"\xff" * 16),                 # scale=1.0, all +1
            pack_weight(0xB800, b"\x00" * 16),                 # scale=-0.5, all -1
            pack_weight(0x3400, b"\x55" * 16),                 # scale=0.25, alternating +1/-1
            pack_weight(0x4000, b"\xff" * 8 + b"\x00" * 8),    # scale=2.0, half +1 / half -1
            pack_weight(0x3E00, b"\x01" + b"\x00" * 15),       # scale=1.5, lane 0 +1, rest -1
        ]))
        row0 = bytes([1]) * 128
        row1 = bytes([2, 256 - 2]) * 64
        input_int8.write_bytes(row0 + bytes([0x7F]) * 32 + row1)
        commands.write_text(
            "# INT8 M=2, N=5 with src_stride=160, dst_stride=48\n"
            "0x000000a000000001 # OP_SET_SRC_STRIDE\n"
            "0000003000000002   # OP_SET_DST_STRIDE\n"
            "0x0000100000050010 # OP_LOAD_WEIGHT N=5 @ 0x1000\n"
            "0000200002000220   # OP_COMPUTE M=2, int8=1, accum=0 @ 0x2000\n"
            "0x0000300000000030 # OP_WRITEBACK @ 0x3000\n"
        )
        task_msg = run(
            "--commands", commands,
            "--load", "0x1000", weights_int8,
            "--load", "0x2000", input_int8,
            "--dump", "0x3000", "96", output_int8,
            "--rd-latency", "1", "--wr-latency", "1", "--cmd-stall-period", "2",
            "--trace", task_trace,
        )
        assert "5 commands" in task_msg and "idle smoke only" not in task_msg
        assert "$enddefinitions" in task_trace.read_text(), "missing task VCD header"
        expected_int8 = (
            struct.pack("<5f", 128.0, 64.0, 0.0, 0.0, -189.0) + bytes(28) +
            struct.pack("<5I", 0x00000000, 0x80000000, 0x42800000, 0x00000000, 0x40C00000) + bytes(28)
        )
        assert output_int8.read_bytes() == expected_int8, "INT8 CLI output/tail-mask mismatch"

        # 2. End-to-end FP32 dynamic quantization + cross-block accumulate (M=1, N=3).
        w_blk0 = directory / "w_blk0.bin"
        w_blk1 = directory / "w_blk1.bin"
        x_blk0 = directory / "x_blk0.bin"
        x_blk1 = directory / "x_blk1.bin"
        output_fp32 = directory / "output_fp32.bin"
        w_blk0.write_bytes(b"".join([
            pack_weight(0x3C00, b"\xff" * 16),  # +1.0
            pack_weight(0xB800, b"\xff" * 16),  # -0.5
            pack_weight(0x4000, b"\xff" * 16),  # +2.0
        ]))
        w_blk1.write_bytes(b"".join([
            pack_weight(0x3800, b"\x00" * 16),  # +0.5, all -1
            pack_weight(0x3C00, b"\x00" * 16),  # +1.0, all -1
            pack_weight(0x3400, b"\x00" * 16),  # +0.25, all -1
        ]))
        x_blk0.write_bytes(struct.pack("<128f", 127.0, *([0.0] * 127)))
        x_blk1.write_bytes(struct.pack("<128f", *([63.5] * 4), *([0.0] * 124)))
        commands.write_text(
            "0000100000030010\n"
            "0000400000000120\n"
            "0000120000030010\n"
            "0000500001000120\n"
            "0000600000000030\n"
        )
        run(
            "--commands", commands,
            "--load", "0x1000", w_blk0,
            "--load", "0x1200", w_blk1,
            "--load", "0x4000", x_blk0,
            "--load", "0x5000", x_blk1,
            "--dump", "0x6000", "16", output_fp32,
        )
        expected_fp32 = struct.pack("<3f", 0.0, -317.5, 190.5) + bytes(4)
        assert output_fp32.read_bytes() == expected_fp32, "FP32 accumulate CLI output mismatch"

        # 3. Runtime cmd_error reporting (WRITEBACK before COMPUTE on line 2).
        commands.write_text("0000000000000001\n0000300000000030\n")
        run("--commands", commands, error="cmd_error on command file line 2")

    print("PASS core CLI: idle, file I/O, VCD, INT8/FP32 instruction execution, and cmd_error checks")


if __name__ == "__main__":
    main()
