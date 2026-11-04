#!/usr/bin/env python3
"""Generate the fixed Qwen3 W1A8 hardware regression case.

The case instantiates a real Qwen3-zh-5k layer slice through the public
bnn-accelerator v2 contract:

  matrix   : blk.0.attn_q.weight, K=512, N=512, group=128 (Q1_0)
  rows     : 16 activations derived from token embeddings + blk.0 RMSNorm
  full_i8  : BNN_OP_GEMM      K=512  (four K blocks, accum=1)
  tile_i8  : BNN_OP_GEMM      K=128  (first K tile)
  tile_f32 : BNN_OP_GEMM_F32  K=128  (same tile, exact dyadic FP32 input)

The FP32 tile input is Q * 2^-6, so its full-row absmax quantization is
exact and recovers Q.  This keeps both input types on the same logical GEMM
and lets the RTL result be compared with the soft API result at low error.

The v2 input buffer quantizes each 128-feature COMPUTE tile independently.
A K>128 FP32 job would therefore not implement BNN_OP_GEMM_F32's full-logical-
row quantization contract; the FP32 workload is intentionally K=128.

Golden outputs follow bnn-accelerator v2: FP32 RNE arithmetic, FP16 scales
converted exactly, per-group products rounded separately, groups accumulated
in increasing index order, no FMA.  They were cross-checked bit-exactly
against bnn-accelerator/soft-bnn-accelerator.c.

Usage:
    python3 tests/gen_model_case.py --model /path/to/qwen3-zh-5k.gguf \
        --out tests/case
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

GROUP = 128
K_FULL = 512
K_TILE = 128
N_FULL = 512
ROWS = 16
F32_TILE_SCALE = np.float32(1.0 / 64.0)

WEIGHT_ADDR = 0x00001000
INT8_FULL_IN = 0x00010000
INT8_TILE_IN = 0x00012000
INT8_FULL_OUT = 0x00020000
INT8_TILE_OUT = 0x00030000
F32_TILE_IN = 0x00040000
F32_TILE_OUT = 0x00050000
OUTPUT_PADDING = 0xA5

TENSOR_F32 = 0
TENSOR_Q1_0 = 41


def parse_gguf(path: Path):
    file = path.open("rb")
    if file.read(4) != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    version = struct.unpack("<I", file.read(4))[0]
    if version != 3:
        raise ValueError(f"{path}: unsupported GGUF version {version}")
    tensor_count = struct.unpack("<Q", file.read(8))[0]
    kv_count = struct.unpack("<Q", file.read(8))[0]

    def read_string():
        length = struct.unpack("<Q", file.read(8))[0]
        return file.read(length).decode("utf-8", "replace")

    def read_value(value_type):
        formats = {
            0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
            7: "?", 10: "Q", 11: "q", 12: "d",
        }
        if value_type in formats:
            fmt = "<" + formats[value_type]
            return struct.unpack(fmt, file.read(struct.calcsize(fmt)))[0]
        if value_type == 8:
            return read_string()
        if value_type == 9:
            element_type = struct.unpack("<I", file.read(4))[0]
            count = struct.unpack("<Q", file.read(8))[0]
            return [read_value(element_type) for _ in range(count)]
        raise ValueError(f"{path}: unsupported GGUF metadata type {value_type}")

    metadata = {}
    for _ in range(kv_count):
        key = read_string()
        value_type = struct.unpack("<I", file.read(4))[0]
        metadata[key] = read_value(value_type)

    tensors = {}
    for _ in range(tensor_count):
        name = read_string()
        rank = struct.unpack("<I", file.read(4))[0]
        dims = tuple(struct.unpack("<Q", file.read(8))[0] for _ in range(rank))
        tensor_type = struct.unpack("<I", file.read(4))[0]
        offset = struct.unpack("<Q", file.read(8))[0]
        tensors[name] = (dims, tensor_type, offset)

    alignment = int(metadata.get("general.alignment", 32))
    data_start = (file.tell() + alignment - 1) // alignment * alignment

    def read_tensor(name):
        if name not in tensors:
            raise ValueError(f"{path}: missing tensor {name}")
        dims, tensor_type, offset = tensors[name]
        file.seek(data_start + offset)
        if tensor_type == TENSOR_Q1_0:
            if len(dims) != 2 or dims[0] % GROUP:
                raise ValueError(f"{name}: unexpected Q1_0 shape {dims}")
            out_features = dims[1]
            groups = dims[0] // GROUP
            raw = np.frombuffer(file.read(out_features * groups * 18), dtype=np.uint8)
            return dims, raw.reshape(out_features, groups, 18)
        if tensor_type == TENSOR_F32:
            count = int(np.prod(dims))
            raw = np.frombuffer(file.read(count * 4), dtype="<f4")
            return dims, raw.reshape(tuple(reversed(dims)))
        raise ValueError(f"{name}: unsupported tensor type {tensor_type}")

    return metadata, read_tensor


def q1_0_parts(blocks):
    out_features, groups, _ = blocks.shape
    scales = blocks[:, :, 0:2].copy().view("<f2").reshape(out_features, groups).astype(np.float32)
    signs = np.unpackbits(blocks[:, :, 2:], axis=2, bitorder="little").astype(np.int8) * 2 - 1
    return scales, signs


def round_half_away(values):
    return np.where(values >= 0.0, np.floor(values + np.float32(0.5)),
                    np.ceil(values - np.float32(0.5)))


def api_quantize_row(values):
    values = np.asarray(values, dtype=np.float32)
    absmax = np.float32(np.abs(values).max())
    if not np.isfinite(absmax) or absmax == 0:
        raise ValueError("activation row must be finite and non-zero")
    scale = np.float32(absmax / np.float32(127.0))
    rounded = round_half_away(np.float32(values / scale))
    return np.clip(rounded, -127.0, 127.0).astype(np.int8), scale


def api_gemm(quantized, signs, scales, row_scale=None):
    rows, out_features = quantized.shape[0], scales.shape[0]
    groups = scales.shape[1]
    acc = np.zeros((rows, out_features), dtype=np.float32)
    for group in range(groups):
        begin = group * GROUP
        dots = (quantized[:, begin:begin + GROUP].astype(np.int32) @
                signs[:, begin:begin + GROUP].T.astype(np.int32))
        term = np.float32(scales[:, group][None, :] * dots.astype(np.float32))
        acc = np.float32(acc + term)
    if row_scale is not None:
        acc = np.float32(acc * np.asarray(row_scale, dtype=np.float32)[:, None])
    return acc


def build_case(model_path, out_dir):
    metadata, read_tensor = parse_gguf(model_path)
    if metadata.get("general.architecture") != "qwen3":
        raise ValueError("model architecture must be qwen3")
    if metadata.get("qwen3.embedding_length") != K_FULL:
        raise ValueError(f"model embedding length must be {K_FULL}")

    q_dims, attn_q = read_tensor("blk.0.attn_q.weight")
    emb_dims, emb_blocks = read_tensor("token_embd.weight")
    _, norm = read_tensor("blk.0.attn_norm.weight")
    if q_dims != (K_FULL, N_FULL):
        raise ValueError(f"blk.0.attn_q.weight shape {q_dims} is not {K_FULL}x{N_FULL}")
    if emb_dims[0] != K_FULL or emb_dims[1] < ROWS:
        raise ValueError(f"token_embd.weight shape {emb_dims} cannot provide {ROWS} rows")
    vocab_size = metadata.get("qwen3.vocab_size")
    if isinstance(vocab_size, int) and vocab_size != emb_dims[1]:
        raise ValueError(f"token_embd.weight has {emb_dims[1]} rows, metadata says {vocab_size}")
    norm = norm.reshape(-1)
    if norm.size != K_FULL:
        raise ValueError("blk.0.attn_norm.weight must have 512 values")

    # Real layer-0 activation direction: tokens 0..15 after embedding + RMSNorm.
    emb_scales, emb_signs = q1_0_parts(emb_blocks)
    embeddings = (emb_scales[:, :, None] * emb_signs).reshape(emb_blocks.shape[0], K_FULL)
    activations = embeddings[np.arange(ROWS)].astype(np.float32)
    epsilon = np.float32(metadata.get("qwen3.layer_norm_rms_epsilon", 1e-6))
    rms = np.sqrt(np.mean(activations * activations, axis=1, keepdims=True) + epsilon)
    activations = np.float32(activations / rms * norm[None, :])

    q_full = np.empty((ROWS, K_FULL), dtype=np.int8)
    for row in range(ROWS):
        q_full[row], _ = api_quantize_row(activations[row])
    q_tile = np.empty((ROWS, K_TILE), dtype=np.int8)
    for row in range(ROWS):
        q_tile[row], _ = api_quantize_row(activations[row, :K_TILE])
    if np.abs(q_full).max(axis=1).min() != 127 or np.abs(q_tile).max(axis=1).min() != 127:
        raise ValueError("every activation row must span the full int8 range")

    scales, signs = q1_0_parts(attn_q)
    flat_signs = signs.reshape(N_FULL, -1)
    golden_full_i8 = api_gemm(q_full, flat_signs, scales)
    golden_tile_i8 = api_gemm(q_tile, signs[:, 0, :], scales[:, 0:1])

    f32_tile = np.float32(q_tile.astype(np.float32) * F32_TILE_SCALE)
    for row in range(ROWS):
        recovered, scale = api_quantize_row(f32_tile[row])
        if scale != F32_TILE_SCALE or not np.array_equal(recovered, q_tile[row]):
            raise ValueError("FP32 tile input does not round-trip through absmax quantization")
    golden_tile_f32 = api_gemm(q_tile, signs[:, 0, :], scales[:, 0:1],
                               np.full(ROWS, F32_TILE_SCALE, dtype=np.float32))
    if not np.array_equal(golden_tile_f32.view(np.uint32),
                          np.float32(golden_tile_i8 * F32_TILE_SCALE).view(np.uint32)):
        raise ValueError("FP32 and INT8 tile goldens are not exact power-of-two scalings")

    n_blocks = N_FULL // GROUP
    k_groups = K_FULL // GROUP
    weight_blob = bytearray()
    for n_block in range(n_blocks):
        for k_group in range(k_groups):
            for channel in range(GROUP):
                weight_blob += attn_q[n_block * GROUP + channel, k_group, :].tobytes()
    if len(weight_blob) != N_FULL * k_groups * 18:
        raise ValueError("unexpected hardware weight layout size")

    image_size = F32_TILE_OUT + ROWS * N_FULL * 4
    image = bytearray(image_size)
    image[WEIGHT_ADDR:WEIGHT_ADDR + len(weight_blob)] = weight_blob
    for row in range(ROWS):
        offset = INT8_FULL_IN + row * K_FULL
        image[offset:offset + K_FULL] = q_full[row].astype(np.uint8).tobytes()
        offset = INT8_TILE_IN + row * K_TILE
        image[offset:offset + K_TILE] = q_tile[row].astype(np.uint8).tobytes()
        offset = F32_TILE_IN + row * K_TILE * 4
        image[offset:offset + K_TILE * 4] = f32_tile[row].astype("<f4").tobytes()
    for address in (INT8_FULL_OUT, INT8_TILE_OUT, F32_TILE_OUT):
        image[address:address + ROWS * N_FULL * 4] = bytes([OUTPUT_PADDING]) * (ROWS * N_FULL * 4)

    def load_weight(address):
        return (address << 32) | (GROUP << 16) | 0x10

    def compute(address, accum, int8):
        return ((address << 32) | (ROWS << 8) | (int(accum) << 24) |
                (int(int8) << 25) | 0x20)

    def writeback(address):
        return (address << 32) | 0x30

    def set_src_stride(stride):
        return (stride << 32) | 0x01

    def set_dst_stride(stride):
        return (stride << 32) | 0x02

    commands = [set_src_stride(K_FULL), set_dst_stride(N_FULL * 4)]
    for n_block in range(n_blocks):
        for k_group in range(k_groups):
            commands.append(load_weight(WEIGHT_ADDR + (n_block * k_groups + k_group) * GROUP * 18))
            commands.append(compute(INT8_FULL_IN + k_group * GROUP, k_group != 0, True))
        commands.append(writeback(INT8_FULL_OUT + n_block * GROUP * 4))

    commands.append(set_src_stride(K_TILE))
    for n_block in range(n_blocks):
        commands.append(load_weight(WEIGHT_ADDR + n_block * k_groups * GROUP * 18))
        commands.append(compute(INT8_TILE_IN, False, True))
        commands.append(writeback(INT8_TILE_OUT + n_block * GROUP * 4))

    commands.append(set_src_stride(K_TILE * 4))
    for n_block in range(n_blocks):
        commands.append(load_weight(WEIGHT_ADDR + n_block * k_groups * GROUP * 18))
        commands.append(compute(F32_TILE_IN, False, False))
        commands.append(writeback(F32_TILE_OUT + n_block * GROUP * 4))

    out_dir.mkdir(parents=True, exist_ok=True)
    command_lines = [
        "# Qwen3-zh-5k blk.0.attn_q W1A8 GEMM, M=16, N=512",
        f"# full_i8 : K=512 @0x{INT8_FULL_IN:x} -> 0x{INT8_FULL_OUT:x}",
        f"# tile_i8 : K=128 @0x{INT8_TILE_IN:x} -> 0x{INT8_TILE_OUT:x}",
        f"# tile_f32: K=128 @0x{F32_TILE_IN:x} -> 0x{F32_TILE_OUT:x}",
        f"# weights : @0x{WEIGHT_ADDR:x} ({len(weight_blob)} bytes)",
        "# generated by tests/gen_model_case.py",
    ]
    command_lines += [f"0x{command:016x}" for command in commands]
    (out_dir / "commands.hex").write_text("\n".join(command_lines) + "\n", encoding="ascii")
    (out_dir / "memory.img").write_bytes(image)

    golden_files = {
        "golden_full_i8.hex": golden_full_i8,
        "golden_tile_i8.hex": golden_tile_i8,
        "golden_tile_f32.hex": golden_tile_f32,
    }
    for name, values in golden_files.items():
        bits = values.astype("<f4").view("<u4").reshape(-1)
        header = f"# IEEE-754 binary32 bits, row-major [{ROWS}][{N_FULL}]\n"
        (out_dir / name).write_text(
            header + "\n".join(f"{int(value):08x}" for value in bits) + "\n",
            encoding="ascii")

    meta = {
        "description": "Qwen3-zh-5k blk.0.attn_q W1A8 GEMM regression case",
        "model": model_path.name,
        "matrix": {"in_features": K_FULL, "out_features": N_FULL, "group_size": GROUP},
        "rows": ROWS,
        "f32_tile_scale": float(F32_TILE_SCALE),
        "addresses": {
            "weights": WEIGHT_ADDR,
            "int8_full_input": INT8_FULL_IN,
            "int8_tile_input": INT8_TILE_IN,
            "int8_full_output": INT8_FULL_OUT,
            "int8_tile_output": INT8_TILE_OUT,
            "f32_tile_input": F32_TILE_IN,
            "f32_tile_output": F32_TILE_OUT,
        },
        "workloads": [
            {"name": "full_i8", "operation": "BNN_OP_GEMM", "input_type": "i8",
             "in_features": K_FULL, "input": "int8_full_input",
             "output": "int8_full_output", "golden": "golden_full_i8.hex"},
            {"name": "tile_i8", "operation": "BNN_OP_GEMM", "input_type": "i8",
             "in_features": K_TILE, "input": "int8_tile_input",
             "output": "int8_tile_output", "golden": "golden_tile_i8.hex"},
            {"name": "tile_f32", "operation": "BNN_OP_GEMM_F32", "input_type": "f32",
             "in_features": K_TILE, "input": "f32_tile_input",
             "output": "f32_tile_output", "golden": "golden_tile_f32.hex"},
        ],
        "relationships": [
            {"left": "tile_f32", "right": "tile_i8",
             "scale": float(F32_TILE_SCALE),
             "description": "FP32 tile equals the exact power-of-two scaling of the INT8 tile"},
        ],
        "tolerance": {"atol": 1e-5, "rtol": 1e-6},
        "notes": [
            "Activation source: tokens 0..15 embeddings plus blk.0 RMSNorm, quantized per row.",
            "tile_i8 and tile_f32 use the first 128 input features of the same layer matrix.",
            "FP32 tile input is INT8 * 2^-6 so both hardware and API quantization recover the same Q.",
            "v2 RTL quantizes each 128-feature COMPUTE tile independently; FP32 is kept at K=128 "
            "to preserve BNN_OP_GEMM_F32 full-logical-row quantization semantics.",
            "Goldens follow bnn-accelerator v2 and were cross-checked against the soft backend.",
            "baseline_cycles.txt is the performance reference and is not rewritten by this script.",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")

    print(f"wrote {out_dir}: {len(commands)} commands, {image_size} byte image, "
          f"{len(weight_blob)} weight bytes")
    print(f"full_i8 golden range [{golden_full_i8.min():.6f}, {golden_full_i8.max():.6f}]")
    print(f"tile_i8 golden range [{golden_tile_i8.min():.6f}, {golden_tile_i8.max():.6f}]")


def main():
    default_model = Path(__file__).resolve().parents[3] / "qwen3-zh-5k.gguf"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=default_model,
                        help="Qwen3-zh-5k GGUF file (default: repository root)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "case",
                        help="output directory (default: tests/case)")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model} (pass --model PATH)")
    build_case(args.model, args.out)


if __name__ == "__main__":
    main()
