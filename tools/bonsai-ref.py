#!/usr/bin/env python3
"""NumPy reference for the Qwen3/Q1_0 path used by bnn-sw/model-qwen3.c.

The reference deliberately emulates the project's W1A8 contract:
  * activations are quantized per row with absmax/127,
  * weights are Q1_0 (+-1 with one FP16 scale per 128 values),
  * each group dot is accumulated in FP32 in increasing group order,
  * host RMSNorm/RoPE/attention/SwiGLU stay in FP32.

It writes the existing --ref format:
  uint32 n, uint32 V, int32 ids[n], float32 logits[n][V]
"""

import argparse
import mmap
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
TYPE_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}


class Cursor:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, size):
        if self.pos + size > len(self.data):
            raise ValueError("truncated GGUF")
        out = self.data[self.pos:self.pos + size]
        self.pos += size
        return out

    def u8(self):
        return struct.unpack("<B", self.take(1))[0]

    def u16(self):
        return struct.unpack("<H", self.take(2))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def f32(self):
        return struct.unpack("<f", self.take(4))[0]

    def f64(self):
        return struct.unpack("<d", self.take(8))[0]

    def string(self):
        length = self.u64()
        return self.take(length)

    def skip(self, kind):
        if kind in TYPE_SIZES:
            self.take(TYPE_SIZES[kind])
        elif kind == 8:
            self.string()
        elif kind == 9:
            element = self.u32()
            count = self.u64()
            for _ in range(count):
                self.skip(element)
        else:
            raise ValueError(f"unknown GGUF value type {kind}")


def parse_gguf(path):
    with open(path, "rb") as f:
        mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    cur = Cursor(mapping)
    magic = cur.take(4)
    if magic != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = cur.u32()
    tensor_count = cur.u64()
    metadata_count = cur.u64()
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version}")

    meta = {}
    for _ in range(metadata_count):
        key = cur.string().decode("utf-8", "replace")
        kind = cur.u32()
        if kind == 8:
            value = cur.string().decode("utf-8", "replace")
        elif kind == 4:
            value = cur.u32()
        elif kind == 5:
            value = struct.unpack("<i", cur.take(4))[0]
        elif kind == 6:
            value = cur.f32()
        elif kind == 7:
            value = cur.u8() != 0
        elif kind == 10:
            value = cur.u64()
        elif kind == 11:
            value = struct.unpack("<q", cur.take(8))[0]
        elif kind == 12:
            value = cur.f64()
        else:
            cur.skip(kind)
            value = None
        meta[key] = value

    tensors = {}
    for _ in range(tensor_count):
        name = cur.string().decode("utf-8", "replace")
        rank = cur.u32()
        dims = tuple(cur.u64() for _ in range(rank))
        kind = cur.u32()
        offset = cur.u64()
        tensors[name] = (dims, kind, offset)

    alignment = int(meta.get("general.alignment", 32))
    data_offset = (cur.pos + alignment - 1) // alignment * alignment
    return mapping, meta, tensors, data_offset


def dequant_q1_0(mapping, data_offset, tensors, name, start=0, end=None):
    dims, kind, offset = tensors[name]
    if kind != 41 or len(dims) != 2 or dims[0] % 128 != 0:
        raise ValueError(f"{name} is not a Q1_0 2D tensor")
    in_dim, out_dim = int(dims[0]), int(dims[1])
    end = out_dim if end is None else end
    groups = in_dim // 128
    if not (0 <= start < end <= out_dim):
        raise ValueError("invalid row slice")

    block_dtype = np.dtype([("d", "<f2"), ("qs", "V16")])
    base = data_offset + offset + start * groups * 18
    blocks = np.frombuffer(mapping, dtype=block_dtype,
                           count=(end - start) * groups, offset=base)
    blocks = blocks.reshape(end - start, groups)
    scales = blocks["d"].astype(np.float32)
    qs = np.ascontiguousarray(blocks["qs"]).view(np.uint8).reshape(end - start, groups, 16)
    bits = np.unpackbits(qs, axis=2, bitorder="little")
    weight = bits.reshape(end - start, in_dim).astype(np.float32) * 2.0 - 1.0
    return weight, scales


def load_f32(mapping, data_offset, tensors, name):
    dims, kind, offset = tensors[name]
    if kind != 0:
        raise ValueError(f"{name} is not F32")
    count = int(np.prod(dims))
    return np.frombuffer(mapping, dtype="<f4", count=count,
                         offset=data_offset + offset).astype(np.float32)


def quantize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(x), axis=1)
    scale = np.where(amax > 0.0, amax / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    y = x / scale[:, None]
    q = np.where(y >= 0.0, np.floor(y + np.float32(0.5)), np.ceil(y - np.float32(0.5)))
    q = np.clip(q, -127.0, 127.0).astype(np.int8)
    return q, scale


def w1a8_linear(x, weight, scales, in_chunks=None):
    """x [n,in] float32; weight [out,in] +-1; scales [out,in/128].

    in_chunks mirrors the C backend's K-splitting (ffn_down is split at 4096).
    The activation scale is applied per chunk, exactly like qwen3_linear().
    """
    rows = x.shape[0]
    out_dim, in_dim = weight.shape
    groups = in_dim // 128
    q, activation_scale = quantize_rows(x)
    result = np.zeros((rows, out_dim), dtype=np.float32)
    cursor = 0
    for count in (in_chunks or [groups]):
        partial = np.zeros((rows, out_dim), dtype=np.float32)
        for g in range(cursor, cursor + count):
            begin = g * 128
            q_group = q[:, begin:begin + 128].astype(np.float32)
            w_group = weight[:, begin:begin + 128]
            dots = q_group @ w_group.T
            terms = (dots * scales[:, g][None, :]).astype(np.float32)
            partial = (partial + terms).astype(np.float32)
        result = (result + partial * activation_scale[:, None]).astype(np.float32)
        cursor += count
    return result


def rmsnorm(x, gain, eps):
    x = np.asarray(x, dtype=np.float32)
    square_sum = np.zeros(x.shape[:-1], dtype=np.float32)
    for i in range(x.shape[-1]):
        value = x[..., i]
        square_sum = square_sum + value * value
    rms = np.sqrt(square_sum / np.float32(x.shape[-1]) + np.float32(eps)).astype(np.float32)
    return (x / rms[..., None] * gain).astype(np.float32)


def yarn_table(max_seq, head_dim, base, factor, original_context, scaling_type):
    half = head_dim // 2
    freq_scale = np.float32(1.0 / factor)
    yarn = scaling_type == "yarn" and original_context > 0
    corr0, corr1 = 0, head_dim - 1
    mscale = np.float32(1.0)
    if yarn:
        start = np.floor(head_dim * np.log(original_context / (32.0 * 2.0 * np.pi)) /
                         (2.0 * np.log(base)))
        end = np.ceil(head_dim * np.log(original_context / (2.0 * np.pi)) /
                      (2.0 * np.log(base)))
        corr0 = int(np.clip(start, 0, head_dim - 1))
        corr1 = int(np.clip(end, 0, head_dim - 1))
        mscale = np.float32(1.0 + 0.1 * np.log(1.0 / freq_scale))

    cosines = np.empty((max_seq, half), dtype=np.float32)
    sines = np.empty((max_seq, half), dtype=np.float32)
    denominator = max(np.float32(corr1 - corr0), np.float32(0.001))
    for pos in range(max_seq):
        for j in range(half):
            theta_extrap = np.float32(pos * base ** (-2.0 * j / head_dim))
            theta = freq_scale * theta_extrap
            if yarn:
                y = (np.float32(j) - np.float32(corr0)) / denominator
                ramp = np.float32(1.0) - np.clip(y, np.float32(0.0), np.float32(1.0))
                theta = (freq_scale * theta_extrap) * (np.float32(1.0) - ramp) + theta_extrap * ramp
            cosines[pos, j] = np.cos(theta) * mscale
            sines[pos, j] = np.sin(theta) * mscale
    return cosines, sines


class Qwen3Reference:
    def __init__(self, path, max_seq=1024):
        self.mapping, self.meta, self.tensors, self.data_offset = parse_gguf(path)
        self.V = int(self.tensors["token_embd.weight"][0][1])
        self.C = int(self.meta["qwen3.embedding_length"])
        self.FF = int(self.meta["qwen3.feed_forward_length"])
        self.H = int(self.meta["qwen3.attention.head_count"])
        self.Hk = int(self.meta["qwen3.attention.head_count_kv"])
        self.HD = int(self.meta["qwen3.attention.key_length"])
        self.L = int(self.meta["qwen3.block_count"])
        self.eps = np.float32(self.meta["qwen3.attention.layer_norm_rms_epsilon"])
        self.kv_mul = self.H // self.Hk
        self.max_seq = min(max_seq, int(self.meta.get("qwen3.context_length", max_seq)))
        self.cosines, self.sines = yarn_table(
            self.max_seq, self.HD,
            float(self.meta["qwen3.rope.freq_base"]),
            float(self.meta["qwen3.rope.scaling.factor"]),
            int(self.meta["qwen3.rope.scaling.original_context_length"]),
            str(self.meta["qwen3.rope.scaling.type"]),
        )
        self.final_norm = load_f32(self.mapping, self.data_offset, self.tensors, "output_norm.weight")
        self.blocks = []
        for layer in range(self.L):
            prefix = f"blk.{layer}."
            self.blocks.append({
                "attn_norm": load_f32(self.mapping, self.data_offset, self.tensors, prefix + "attn_norm.weight"),
                "ffn_norm": load_f32(self.mapping, self.data_offset, self.tensors, prefix + "ffn_norm.weight"),
                "q_norm": load_f32(self.mapping, self.data_offset, self.tensors, prefix + "attn_q_norm.weight"),
                "k_norm": load_f32(self.mapping, self.data_offset, self.tensors, prefix + "attn_k_norm.weight"),
            })

    def weight(self, name):
        return dequant_q1_0(self.mapping, self.data_offset, self.tensors, name)

    def embedding(self, token):
        weight, scales = dequant_q1_0(self.mapping, self.data_offset, self.tensors,
                                      "token_embd.weight", token, token + 1)
        grouped = weight.reshape(self.C // 128, 128) * scales[0][:, None]
        return grouped.reshape(self.C).astype(np.float32)

    def rope(self, x, positions):
        half = self.HD // 2
        cos = self.cosines[positions][:, None, :]
        sin = self.sines[positions][:, None, :]
        first = x[..., :half]
        second = x[..., half:]
        return np.concatenate([first * cos - second * sin,
                               first * sin + second * cos], axis=-1).astype(np.float32)

    def attention(self, q, k, v):
        n = q.shape[0]
        qh = q.transpose(1, 0, 2)
        kh = np.repeat(k.transpose(1, 0, 2), self.kv_mul, axis=0)
        vh = np.repeat(v.transpose(1, 0, 2), self.kv_mul, axis=0)
        scores = np.zeros((self.H, n, n), dtype=np.float32)
        for i in range(self.HD):
            scores = (scores + qh[:, :, i][:, :, None] * kh[:, :, i][:, None, :]).astype(np.float32)
        scores = (scores * np.float32(1.0 / np.sqrt(self.HD))).astype(np.float32)
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        scores = np.where(mask, np.float32(-np.inf), scores)
        maximum = scores.max(axis=-1, keepdims=True)
        exponentials = np.exp((scores - maximum).astype(np.float32)).astype(np.float32)
        probs = np.zeros_like(exponentials)
        sums = np.zeros((self.H, n), dtype=np.float32)
        for t in range(n):
            probs[:, :, t] = exponentials[:, :, t]
            sums = (sums + exponentials[:, :, t]).astype(np.float32)
        probs = (probs / sums[:, :, None]).astype(np.float32)
        out = np.zeros((self.H, n, self.HD), dtype=np.float32)
        for t in range(n):
            out = (out + probs[:, :, t][:, :, None] * vh[:, t, :][:, None, :]).astype(np.float32)
        return out.transpose(1, 0, 2).reshape(n, self.C).astype(np.float32)

    def lm_head(self, x):
        rows = x.shape[0]
        logits = np.empty((rows, self.V), dtype=np.float32)
        for start in range(0, self.V, 4096):
            end = min(self.V, start + 4096)
            weight, scales = dequant_q1_0(self.mapping, self.data_offset, self.tensors,
                                          "token_embd.weight", start, end)
            logits[:, start:end] = w1a8_linear(x, weight, scales)
        return logits

    def forward(self, ids, dump_dir=None):
        ids = np.asarray(ids, dtype=np.int64)
        n = len(ids)
        if n == 0 or n > self.max_seq:
            raise ValueError("invalid sequence length")

        def dump(tag, pos, array):
            if dump_dir and pos < 2:
                np.ascontiguousarray(array, dtype=np.float32).tofile(
                    f"{dump_dir}/{tag}_p{pos}.bin")

        x = np.stack([self.embedding(int(t)) for t in ids])
        for pos in range(min(2, n)):
            dump("emb", pos, x[pos])
        positions = list(range(n))
        for layer, block in enumerate(self.blocks):
            prefix = f"blk.{layer}."
            norm = rmsnorm(x, block["attn_norm"], self.eps)

            wq, sq = self.weight(prefix + "attn_q.weight")
            wk, sk = self.weight(prefix + "attn_k.weight")
            wv, sv = self.weight(prefix + "attn_v.weight")
            q = w1a8_linear(norm, wq, sq)
            k = w1a8_linear(norm, wk, sk)
            v = w1a8_linear(norm, wv, sv)
            for pos in range(min(2, n)):
                dump(f"l{layer}_qproj", pos, q[pos])
                dump(f"l{layer}_kproj", pos, k[pos])

            q = q.reshape(n, self.H, self.HD)
            k = k.reshape(n, self.Hk, self.HD)
            v = v.reshape(n, self.Hk, self.HD)
            q = rmsnorm(q, block["q_norm"], self.eps)
            k = rmsnorm(k, block["k_norm"], self.eps)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
            for pos in range(min(2, n)):
                dump(f"l{layer}_qrope", pos, q[pos].reshape(self.C))
                dump(f"l{layer}_krope", pos, k[pos].reshape(self.Hk * self.HD))
            y = self.attention(q, k, v)
            for pos in range(min(2, n)):
                dump(f"l{layer}_y", pos, y[pos])

            wo, so = self.weight(prefix + "attn_output.weight")
            x = (x + w1a8_linear(y, wo, so)).astype(np.float32)
            for pos in range(min(2, n)):
                dump(f"l{layer}_attnres", pos, x[pos])

            norm = rmsnorm(x, block["ffn_norm"], self.eps)
            wg, sg = self.weight(prefix + "ffn_gate.weight")
            wu, su = self.weight(prefix + "ffn_up.weight")
            wd, sd = self.weight(prefix + "ffn_down.weight")
            gate = w1a8_linear(norm, wg, sg)
            up = w1a8_linear(norm, wu, su)
            hidden = (gate / (np.float32(1.0) + np.exp(-gate)) * up).astype(np.float32)
            down_groups = self.FF // 128
            down_chunks = []
            remaining = down_groups
            while remaining > 32:
                down_chunks.append(32)
                remaining -= 32
            down_chunks.append(remaining)
            x = (x + w1a8_linear(hidden, wd, sd, in_chunks=down_chunks)).astype(np.float32)
            for pos in range(min(2, n)):
                dump(f"l{layer}_ffnres", pos, x[pos])

        norm = rmsnorm(x, self.final_norm, self.eps)
        return self.lm_head(norm)


def read_ids(args):
    if args.ids:
        return [int(value) for value in args.ids.split()]
    if args.ids_file:
        return [int(value) for value in open(args.ids_file).read().split()]
    raise ValueError("provide --ids or --ids-file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ids", help="space-separated token IDs")
    parser.add_argument("--ids-file", help="whitespace-separated token IDs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--dump-dir", help="dump first two positions of intermediates")
    args = parser.parse_args()

    ids = read_ids(args)
    reference = Qwen3Reference(args.model)
    logits = reference.forward(ids, dump_dir=args.dump_dir)

    with open(args.out, "wb") as out:
        out.write(struct.pack("<II", len(ids), reference.V))
        out.write(np.asarray(ids, dtype="<i4").tobytes())
        out.write(logits.astype("<f4").tobytes())

    for pos, token in enumerate(ids):
        top = np.argsort(-logits[pos])[:args.topk]
        print(f"position {pos} (id {token}): " +
              ", ".join(f"{int(i)}:{logits[pos, i]:.4f}" for i in top))
    print(f"wrote {args.out}: n={len(ids)} V={reference.V}")


if __name__ == "__main__":
    sys.exit(main())
