#!/usr/bin/env python3
"""Build Qwen3-shaped Q1_0 GGUF files for the bnn-sw-v2 engine.

Q1_0 stores 128 signs (+1/-1) plus one FP16 group scale.  The engine also
requires hidden/FFN/KV widths to be 128-aligned, so the exporter validates the
same structural constraints before writing anything.

The vocabulary is a GPT-2 byte-level BPE table with no merges, plus the Qwen3
chat markers.  Every UTF-8 byte has a token, which keeps lm_head small while
still allowing arbitrary prompts to round-trip.

Metadata types follow llama.cpp's Qwen3 schema (u32 hyperparameters), so the
same Q1_0 file also loads in llama.cpp.  bnn-sw-v2 itself accepts any integer
width, which is why this distinction is easy to miss.
"""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUP = 128
ALIGNMENT = 32

META_U32 = 4
META_I32 = 5
META_F32 = 6
META_BOOL = 7
META_STRING = 8
META_ARRAY = 9
META_U64 = 10

TENSOR_F32 = 0
TENSOR_Q1_0 = 41

SPECIAL_TOKENS = (
    ("<|endoftext|>", 3),
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),
    ("<think>", 4),
    ("</think>", 4),
)


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 261
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 8
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    @property
    def kv_size(self):
        return self.num_key_value_heads * self.head_dim

    def validate(self):
        if self.hidden_size % GROUP:
            raise ValueError("hidden_size must be a multiple of 128")
        if self.intermediate_size % GROUP:
            raise ValueError("intermediate_size must be a multiple of 128")
        if self.kv_size % GROUP:
            raise ValueError("num_key_value_heads * head_dim must be a multiple of 128")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.max_position_embeddings == 0:
            raise ValueError("max_position_embeddings must be positive")

    @property
    def parameter_count(self):
        per_layer = (
            self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.kv_size
            + self.hidden_size * self.hidden_size
            + 3 * self.hidden_size * self.intermediate_size
        )
        norms = (
            self.num_hidden_layers * (2 * self.hidden_size + 2 * self.head_dim)
            + self.hidden_size
        )
        return (
            per_layer * self.num_hidden_layers
            + self.hidden_size * self.vocab_size
            + norms
        )


def bytes_to_unicode():
    """GPT-2 byte <-> printable-codepoint mapping used by the C tokenizer."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(byte_values, codepoints)}


def build_vocabulary():
    mapping = bytes_to_unicode()
    symbols = sorted(mapping.values(), key=ord)
    tokens = list(symbols)
    token_types = [1] * len(tokens)
    for text, kind in SPECIAL_TOKENS:
        tokens.append(text)
        token_types.append(kind)
    eos_id = tokens.index("<|im_end|>")
    return tokens, token_types, eos_id


@dataclass
class TokenizerData:
    tokens: list
    token_types: list
    merges: list
    eos_id: int


def byte_tokenizer():
    tokens, token_types, eos_id = build_vocabulary()
    return TokenizerData(tokens, token_types, [], eos_id)


def load_tokenizer_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = data["model"]
    vocabulary = model["vocab"]
    size = max(vocabulary.values()) + 1
    tokens = [None] * size
    for token, index in vocabulary.items():
        tokens[index] = token
    if any(token is None for token in tokens):
        raise ValueError(f"{path}: tokenizer vocabulary has gaps")

    merges = []
    for merge in model.get("merges", []):
        if isinstance(merge, str):
            merges.append(merge)
        else:
            merges.append(f"{merge[0]} {merge[1]}")

    special_types = {text: kind for text, kind in SPECIAL_TOKENS}
    token_types = [special_types.get(token, 1) for token in tokens]
    eos_id = vocabulary.get("<|endoftext|>")
    if eos_id is None:
        raise ValueError(f"{path}: tokenizer has no <|endoftext|> token")
    return TokenizerData(tokens, token_types, merges, eos_id)


def add_tokenizer_metadata(writer, tokenizer):
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen2")
    writer.add_string_array("tokenizer.ggml.tokens", tokenizer.tokens)
    writer.add_i32_array("tokenizer.ggml.token_type", tokenizer.token_types)
    writer.add_string_array("tokenizer.ggml.merges", tokenizer.merges)
    writer.add_u32("tokenizer.ggml.eos_token_id", tokenizer.eos_id)
    writer.add_bool("tokenizer.ggml.add_bos_token", False)


def add_qwen3_metadata(writer, config, name, tokenizer):
    writer.add_string("general.architecture", "qwen3")
    writer.add_string("general.type", "model")
    writer.add_string("general.name", name)
    writer.add_string("general.size_label", f"{config.parameter_count / 1e6:.1f}M")
    writer.add_u32("general.quantization_version", 2)
    writer.add_u32("general.file_type", 40)  # MOSTLY_Q1_0
    writer.add_u32("general.alignment", ALIGNMENT)
    # llama.cpp's Qwen3 schema expects u32 for every integer hyperparameter.
    writer.add_u32("qwen3.block_count", config.num_hidden_layers)
    writer.add_u32("qwen3.context_length", config.max_position_embeddings)
    writer.add_u32("qwen3.embedding_length", config.hidden_size)
    writer.add_u32("qwen3.feed_forward_length", config.intermediate_size)
    writer.add_u32("qwen3.attention.head_count", config.num_attention_heads)
    writer.add_u32("qwen3.attention.head_count_kv", config.num_key_value_heads)
    writer.add_u32("qwen3.attention.key_length", config.head_dim)
    writer.add_u32("qwen3.attention.value_length", config.head_dim)
    writer.add_f32("qwen3.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_f32("qwen3.rope.freq_base", config.rope_theta)
    writer.add_string("qwen3.rope.scaling.type", "none")
    writer.add_f32("qwen3.rope.scaling.factor", 1.0)
    writer.add_u32(
        "qwen3.rope.scaling.original_context_length",
        config.max_position_embeddings,
    )
    add_tokenizer_metadata(writer, tokenizer)


class GGUFWriter:
    def __init__(self):
        self.kvs = []
        self.tensors = []

    @staticmethod
    def _string(value):
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return struct.pack("<Q", len(data)) + data

    def _add(self, key, value_type, payload):
        self.kvs.append((key, value_type, payload))

    def add_u32(self, key, value):
        self._add(key, META_U32, struct.pack("<I", int(value)))

    def add_u64(self, key, value):
        self._add(key, META_U64, struct.pack("<Q", int(value)))

    def add_f32(self, key, value):
        self._add(key, META_F32, struct.pack("<f", float(value)))

    def add_bool(self, key, value):
        self._add(key, META_BOOL, struct.pack("<B", int(bool(value))))

    def add_string(self, key, value):
        self._add(key, META_STRING, self._string(value))

    def add_string_array(self, key, values):
        payload = struct.pack("<IQ", META_STRING, len(values))
        payload += b"".join(self._string(value) for value in values)
        self._add(key, META_ARRAY, payload)

    def add_i32_array(self, key, values):
        payload = struct.pack("<IQ", META_I32, len(values))
        payload += b"".join(struct.pack("<i", int(value)) for value in values)
        self._add(key, META_ARRAY, payload)

    def add_tensor(self, name, dims, tensor_type, payload):
        dims = tuple(int(dim) for dim in dims)
        if tensor_type == TENSOR_Q1_0:
            if len(dims) != 2 or dims[0] % GROUP:
                raise ValueError(f"{name}: Q1_0 tensor must be [128*n, rows]")
            expected = dims[1] * (dims[0] // GROUP) * 18
        elif tensor_type == TENSOR_F32:
            expected = int(np.prod(dims)) * 4
        else:
            raise ValueError(f"{name}: unsupported tensor type {tensor_type}")
        if len(payload) != expected:
            raise ValueError(
                f"{name}: payload {len(payload)} bytes, expected {expected}"
            )
        self.tensors.append((name, dims, tensor_type, bytes(payload)))

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = b"GGUF" + struct.pack(
            "<IQQ", 3, len(self.tensors), len(self.kvs)
        )
        metadata = bytearray()
        for key, value_type, payload in self.kvs:
            metadata += self._string(key) + struct.pack("<I", value_type) + payload

        infos = bytearray()
        data = bytearray()
        offset = 0
        for name, dims, tensor_type, payload in self.tensors:
            padding = (-offset) % ALIGNMENT
            data += b"\0" * padding
            offset += padding
            infos += self._string(name)
            infos += struct.pack("<I", len(dims))
            infos += struct.pack("<" + "Q" * len(dims), *dims)
            infos += struct.pack("<IQ", tensor_type, offset)
            data += payload
            offset += len(payload)

        info_end = len(header) + len(metadata) + len(infos)
        data_padding = (-info_end) % ALIGNMENT
        with open(path, "wb") as file:
            file.write(header)
            file.write(metadata)
            file.write(infos)
            file.write(b"\0" * data_padding)
            file.write(data)


def random_q1_0(out_features, in_features, rng):
    if in_features % GROUP:
        raise ValueError("Q1_0 input width must be a multiple of 128")
    groups = in_features // GROUP
    scale = np.asarray(1.0 / math.sqrt(in_features), dtype="<f2")
    scale_bytes = scale.tobytes()

    blocks = np.empty((out_features, groups, 18), dtype=np.uint8)
    blocks[:, :, 0] = scale_bytes[0]
    blocks[:, :, 1] = scale_bytes[1]
    blocks[:, :, 2:] = rng.integers(
        0, 256, size=(out_features, groups, 16), dtype=np.uint8
    )
    return blocks.tobytes()


def f32_ones(count):
    return np.ones(count, dtype="<f4").tobytes()


def pack_q1_0(weights):
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] % GROUP:
        raise ValueError("Q1_0 weights must be [out, 128*n]")
    out_features, in_features = weights.shape
    groups = in_features // GROUP
    if not np.isfinite(weights).all():
        raise ValueError("Q1_0 weights contain NaN or Inf")

    bits = (weights >= 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little").reshape(
        out_features, groups, 16
    )
    scales = np.abs(weights).reshape(out_features, groups, GROUP).mean(-1)
    scales = scales.astype("<f2")
    blocks = np.empty((out_features, groups, 18), dtype=np.uint8)
    blocks[:, :, 0:2] = scales.view(np.uint8).reshape(out_features, groups, 2)
    blocks[:, :, 2:] = packed
    return [in_features, out_features], blocks.tobytes()


def export_untrained(config, path, seed=20261025, name="qwen3-tiny"):
    config.validate()
    tokenizer = byte_tokenizer()
    if len(tokenizer.tokens) != config.vocab_size:
        raise ValueError(
            f"config.vocab_size is {config.vocab_size}, "
            f"tokenizer has {len(tokenizer.tokens)}"
        )

    rng = np.random.default_rng(seed)
    writer = GGUFWriter()
    add_qwen3_metadata(writer, config, name, tokenizer)

    writer.add_tensor(
        "output_norm.weight",
        [config.hidden_size],
        TENSOR_F32,
        f32_ones(config.hidden_size),
    )
    writer.add_tensor(
        "token_embd.weight",
        [config.hidden_size, config.vocab_size],
        TENSOR_Q1_0,
        random_q1_0(config.vocab_size, config.hidden_size, rng),
    )

    for layer in range(config.num_hidden_layers):
        prefix = f"blk.{layer}."
        writer.add_tensor(
            prefix + "attn_norm.weight",
            [config.hidden_size],
            TENSOR_F32,
            f32_ones(config.hidden_size),
        )
        writer.add_tensor(
            prefix + "attn_q.weight",
            [config.hidden_size, config.hidden_size],
            TENSOR_Q1_0,
            random_q1_0(config.hidden_size, config.hidden_size, rng),
        )
        writer.add_tensor(
            prefix + "attn_k.weight",
            [config.hidden_size, config.kv_size],
            TENSOR_Q1_0,
            random_q1_0(config.kv_size, config.hidden_size, rng),
        )
        writer.add_tensor(
            prefix + "attn_v.weight",
            [config.hidden_size, config.kv_size],
            TENSOR_Q1_0,
            random_q1_0(config.kv_size, config.hidden_size, rng),
        )
        writer.add_tensor(
            prefix + "attn_output.weight",
            [config.hidden_size, config.hidden_size],
            TENSOR_Q1_0,
            random_q1_0(config.hidden_size, config.hidden_size, rng),
        )
        writer.add_tensor(
            prefix + "attn_q_norm.weight",
            [config.head_dim],
            TENSOR_F32,
            f32_ones(config.head_dim),
        )
        writer.add_tensor(
            prefix + "attn_k_norm.weight",
            [config.head_dim],
            TENSOR_F32,
            f32_ones(config.head_dim),
        )
        writer.add_tensor(
            prefix + "ffn_norm.weight",
            [config.hidden_size],
            TENSOR_F32,
            f32_ones(config.hidden_size),
        )
        writer.add_tensor(
            prefix + "ffn_gate.weight",
            [config.hidden_size, config.intermediate_size],
            TENSOR_Q1_0,
            random_q1_0(config.intermediate_size, config.hidden_size, rng),
        )
        writer.add_tensor(
            prefix + "ffn_up.weight",
            [config.hidden_size, config.intermediate_size],
            TENSOR_Q1_0,
            random_q1_0(config.intermediate_size, config.hidden_size, rng),
        )
        writer.add_tensor(
            prefix + "ffn_down.weight",
            [config.intermediate_size, config.hidden_size],
            TENSOR_Q1_0,
            random_q1_0(config.hidden_size, config.intermediate_size, rng),
        )

    writer.write(path)
    return Path(path)


def export_trained(config, path, state_dict, tokenizer, name="qwen3-zh"):
    config.validate()
    if len(tokenizer.tokens) != config.vocab_size:
        raise ValueError(
            f"config.vocab_size is {config.vocab_size}, "
            f"tokenizer has {len(tokenizer.tokens)}"
        )

    def value(key, shape):
        if key not in state_dict:
            raise KeyError(f"missing state_dict tensor '{key}'")
        tensor = np.asarray(state_dict[key], dtype=np.float32)
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"'{key}' has shape {tuple(tensor.shape)}, expected {tuple(shape)}"
            )
        return tensor

    writer = GGUFWriter()
    add_qwen3_metadata(writer, config, name, tokenizer)

    def add_f32(name, key, count):
        writer.add_tensor(
            name, [count], TENSOR_F32, value(key, (count,)).astype("<f4").tobytes()
        )

    def add_q1(name, key, out_features, in_features):
        dims, payload = pack_q1_0(value(key, (out_features, in_features)))
        writer.add_tensor(name, dims, TENSOR_Q1_0, payload)

    add_f32("output_norm.weight", "output_norm.weight", config.hidden_size)
    add_q1(
        "token_embd.weight",
        "token_embd.weight",
        config.vocab_size,
        config.hidden_size,
    )

    for layer in range(config.num_hidden_layers):
        prefix = f"blk.{layer}."
        add_f32(
            prefix + "attn_norm.weight",
            prefix + "attn_norm.weight",
            config.hidden_size,
        )
        add_q1(
            prefix + "attn_q.weight",
            prefix + "attn_q.weight",
            config.hidden_size,
            config.hidden_size,
        )
        add_q1(
            prefix + "attn_k.weight",
            prefix + "attn_k.weight",
            config.kv_size,
            config.hidden_size,
        )
        add_q1(
            prefix + "attn_v.weight",
            prefix + "attn_v.weight",
            config.kv_size,
            config.hidden_size,
        )
        add_q1(
            prefix + "attn_output.weight",
            prefix + "attn_output.weight",
            config.hidden_size,
            config.hidden_size,
        )
        add_f32(
            prefix + "attn_q_norm.weight",
            prefix + "attn_q_norm.weight",
            config.head_dim,
        )
        add_f32(
            prefix + "attn_k_norm.weight",
            prefix + "attn_k_norm.weight",
            config.head_dim,
        )
        add_f32(
            prefix + "ffn_norm.weight",
            prefix + "ffn_norm.weight",
            config.hidden_size,
        )
        add_q1(
            prefix + "ffn_gate.weight",
            prefix + "ffn_gate.weight",
            config.intermediate_size,
            config.hidden_size,
        )
        add_q1(
            prefix + "ffn_up.weight",
            prefix + "ffn_up.weight",
            config.intermediate_size,
            config.hidden_size,
        )
        add_q1(
            prefix + "ffn_down.weight",
            prefix + "ffn_down.weight",
            config.hidden_size,
            config.intermediate_size,
        )

    writer.write(path)
    return Path(path)
