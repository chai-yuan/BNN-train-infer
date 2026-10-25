#!/usr/bin/env python3
"""Train a W1A8 binary MNIST MLP and export a bnn-mlp GGUF.

The exported model uses the same layout as the C reader:
  - weights: Q1_0 (128-weight blocks, FP16 group scale, +-1 signs)
  - per layer: FP32 post_scale / post_bias (BatchNorm folded)
  - input: 784 pixels normalized with (x/255 - mean) / std, zero-padded to 896

It also writes deployable test data:
  test-inputs.f32  N x input_features little-endian FP32 (already normalized)
  test-labels.u8   N uint8 labels
"""

import argparse
import gzip
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent

GROUP = 128
IN_RAW = 784
IN_PAD = 896  # 7 * 128
MEAN = 0.1307
STD = 0.3081
CLASSES = 10

GGUF_VERSION = 3
ALIGNMENT = 32
TENSOR_F32 = 0
TENSOR_Q1_0 = 41
META_F32 = 6
META_STRING = 8
META_ARRAY = 9
META_U64 = 10

MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _valid_gzip(path):
    try:
        with gzip.open(path, "rb") as file:
            file.read(4)
        return True
    except OSError:
        return False


def ensure_mnist(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES:
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size and _valid_gzip(dest):
            continue
        if dest.exists():
            dest.unlink()
        last_error = None
        for mirror in MIRRORS:
            url = f"{mirror}/{name}"
            part = dest.with_suffix(dest.suffix + ".part")
            try:
                print(f"download {url}", flush=True)
                urllib.request.urlretrieve(url, part)
                part.replace(dest)
                if not _valid_gzip(dest):
                    raise OSError("downloaded file is not valid gzip")
                break
            except Exception as error:  # noqa: BLE001 - report the last mirror error
                last_error = error
                if part.exists():
                    part.unlink()
                if dest.exists():
                    dest.unlink()
        else:
            raise RuntimeError(f"cannot download {name}: {last_error}")


def load_images(path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError(f"invalid image file: {path}")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    if data.size != count * rows * cols:
        raise ValueError(f"truncated image file: {path}")
    return data.reshape(count, rows * cols)


def load_labels(path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"invalid label file: {path}")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"truncated label file: {path}")
    return data.copy()


def preprocess(images):
    pixels = images.astype(np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    pixels = np.pad(pixels, ((0, 0), (0, IN_PAD - IN_RAW)))
    return np.ascontiguousarray(pixels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BinaryLinear(nn.Module):
    """Latent FP weights with a straight-through +-1 quantizer."""

    def __init__(self, in_features, out_features, group=GROUP):
        super().__init__()
        if in_features % group or in_features % 8:
            raise ValueError("input width must be 128- and 8-aligned")
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-1, 1) * in_features ** -0.5
        )

    def quantized(self):
        groups = self.weight.view(self.out_features, -1, self.group)
        scale = groups.abs().mean(-1, keepdim=True)
        sign = torch.where(groups >= 0, 1.0, -1.0)
        return (sign * scale).view(self.out_features, self.in_features)

    def forward(self, x):
        weight = self.weight + (self.quantized() - self.weight).detach()
        return F.linear(x, weight)


class MLP(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.fc1 = BinaryLinear(IN_PAD, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = BinaryLinear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = BinaryLinear(hidden, CLASSES)
        self.bias3 = nn.Parameter(torch.zeros(CLASSES))

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x) + self.bias3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(model, x, y, device, batch=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(x[start:start + batch]).to(device)
            prediction = model(xb).argmax(1).cpu().numpy()
            correct += int((prediction == y[start:start + batch]).sum())
    return correct / len(x)


def train(args, x_train, y_train, x_test, y_test, device):
    torch.manual_seed(args.seed)
    model = MLP(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(loader))
    )

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(xb)
        accuracy = evaluate(model, x_test, y_test, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}: loss {loss_sum / len(x_train):.4f}, "
            f"quantized test accuracy {accuracy:.4f}",
            flush=True,
        )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------

class GGUFWriter:
    def __init__(self):
        self.kvs = []
        self.tensors = []

    @staticmethod
    def _string(value):
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return struct.pack("<Q", len(data)) + data

    def add_u64(self, key, value):
        self.kvs.append((key, META_U64, struct.pack("<Q", int(value))))

    def add_f32(self, key, value):
        self.kvs.append((key, META_F32, struct.pack("<f", float(value))))

    def add_string(self, key, value):
        self.kvs.append((key, META_STRING, self._string(value)))

    def add_string_array(self, key, values):
        payload = struct.pack("<IQ", META_STRING, len(values))
        payload += b"".join(self._string(value) for value in values)
        self.kvs.append((key, META_ARRAY, payload))

    def add_tensor(self, name, dims, tensor_type, payload):
        dims = tuple(int(dim) for dim in dims)
        if tensor_type == TENSOR_Q1_0:
            if len(dims) != 2 or dims[0] % GROUP:
                raise ValueError("Q1_0 tensor must be [128*n, rows]")
            expected = dims[1] * (dims[0] // GROUP) * 18
        elif tensor_type == TENSOR_F32:
            expected = int(np.prod(dims)) * 4
        else:
            raise ValueError(f"unsupported tensor type {tensor_type}")
        if len(payload) != expected:
            raise ValueError(
                f"tensor {name}: payload {len(payload)} bytes, expected {expected}"
            )
        self.tensors.append((name, dims, tensor_type, bytes(payload)))

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = b"GGUF" + struct.pack(
            "<IQQ", GGUF_VERSION, len(self.tensors), len(self.kvs)
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


def pack_q1_0(weights):
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("weights must be [out, in]")
    out_features, in_features = weights.shape
    if in_features % GROUP or in_features % 8:
        raise ValueError("input width must be 128- and 8-aligned")
    groups = in_features // GROUP

    bits = (weights >= 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    scales = np.abs(weights).reshape(out_features, groups, GROUP).mean(-1)

    blocks = np.empty((out_features, groups, 18), dtype=np.uint8)
    blocks[:, :, :2] = np.asarray(scales, dtype="<f2").view(np.uint8).reshape(
        out_features, groups, 2
    )
    blocks[:, :, 2:] = packed.reshape(out_features, groups, 16)
    return [in_features, out_features], blocks.tobytes()


@torch.no_grad()
def fold_bn(bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = bn.bias - bn.running_mean * scale
    return (
        scale.cpu().numpy().astype(np.float32),
        bias.cpu().numpy().astype(np.float32),
    )


@torch.no_grad()
def export_layers(model):
    a1, b1 = fold_bn(model.bn1)
    a2, b2 = fold_bn(model.bn2)
    return [
        {
            "weight": model.fc1.weight.cpu().numpy().astype(np.float32),
            "post_scale": a1,
            "post_bias": b1,
            "activation": "relu",
        },
        {
            "weight": model.fc2.weight.cpu().numpy().astype(np.float32),
            "post_scale": a2,
            "post_bias": b2,
            "activation": "relu",
        },
        {
            "weight": model.fc3.weight.cpu().numpy().astype(np.float32),
            "post_scale": np.ones(CLASSES, dtype=np.float32),
            "post_bias": model.bias3.cpu().numpy().astype(np.float32),
            "activation": "none",
        },
    ]


def write_mlp_gguf(path, layers, labels):
    writer = GGUFWriter()
    writer.add_string("general.architecture", "bnn-mlp")
    writer.add_string("general.name", "mnist-mlp")
    writer.add_u64("bnn-mlp.block_count", len(layers))
    writer.add_u64("bnn-mlp.input_features", IN_PAD)
    writer.add_u64("bnn-mlp.class_count", CLASSES)
    writer.add_u64("bnn-mlp.input_pixels", IN_RAW)
    writer.add_f32("bnn-mlp.input_mean", MEAN)
    writer.add_f32("bnn-mlp.input_std", STD)
    writer.add_string_array("bnn-mlp.labels", labels)

    for index, layer in enumerate(layers):
        writer.add_string(f"bnn-mlp.block.{index}.activation", layer["activation"])
        dims, payload = pack_q1_0(layer["weight"])
        writer.add_tensor(f"blk.{index}.weight", dims, TENSOR_Q1_0, payload)
        out_features = int(np.asarray(layer["weight"]).shape[0])
        post_scale = np.asarray(layer["post_scale"], dtype="<f4").tobytes()
        post_bias = np.asarray(layer["post_bias"], dtype="<f4").tobytes()
        writer.add_tensor(f"blk.{index}.post_scale", [out_features], TENSOR_F32,
                          post_scale)
        writer.add_tensor(f"blk.{index}.post_bias", [out_features], TENSOR_F32,
                          post_bias)
    writer.write(path)


# ---------------------------------------------------------------------------
# Export-time numeric check
# ---------------------------------------------------------------------------

def simulate_layers(layers, inputs, batch=1000):
    outputs = []
    for start in range(0, len(inputs), batch):
        x = inputs[start:start + batch]
        for layer in layers:
            weight = np.asarray(layer["weight"], dtype=np.float32)
            out_features, in_features = weight.shape
            groups = in_features // GROUP
            scales = np.abs(weight).reshape(out_features, groups, GROUP).mean(-1)
            scales = scales.astype(np.float16).astype(np.float32)
            sign = np.where(weight >= 0, 1.0, -1.0).astype(np.float32)

            amax = np.abs(x).max(axis=1, keepdims=True)
            x_scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            quantized = np.sign(x / x_scale) * np.floor(np.abs(x / x_scale) + 0.5)
            quantized = np.clip(quantized, -127.0, 127.0).astype(np.float32)

            group_sum = np.einsum(
                "ngk,ogk->nog",
                quantized.reshape(-1, groups, GROUP),
                sign.reshape(out_features, groups, GROUP),
            )
            accumulator = (group_sum * scales[None]).sum(-1) * x_scale
            x = layer["post_scale"][None] * accumulator + layer["post_bias"][None]
            if layer["activation"] == "relu":
                x = np.maximum(x, 0.0)
        outputs.append(x.astype(np.float32))
    return np.concatenate(outputs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=1000,
                        help="samples written to test-inputs.f32 (0 = all)")
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--out-dir", type=Path, default=BASE / "out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.hidden % GROUP or IN_PAD % GROUP:
        parser.error("hidden size must be a multiple of 128")

    np.random.seed(args.seed)
    ensure_mnist(args.data_dir)

    train_images = load_images(args.data_dir / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(args.data_dir / "train-labels-idx1-ubyte.gz")
    test_images = load_images(args.data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(args.data_dir / "t10k-labels-idx1-ubyte.gz")
    x_train = preprocess(train_images)
    x_test = preprocess(test_images)
    print(
        f"MNIST: {len(x_train)} train / {len(x_test)} test samples, "
        f"input {IN_RAW} -> {IN_PAD}",
        flush=True,
    )

    model = train(args, x_train, train_labels, x_test, test_labels, args.device)
    layers = export_layers(model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "mnist_mlp.gguf"
    write_mlp_gguf(model_path, layers, [str(label) for label in range(CLASSES)])
    print(f"wrote {model_path} ({model_path.stat().st_size} bytes)")

    logits = simulate_layers(layers, x_test)
    accuracy = float((logits.argmax(1) == test_labels).mean())
    print(f"exported W1A8 test accuracy: {accuracy:.4f}")

    sample_count = len(x_test) if args.test_samples <= 0 else min(
        args.test_samples, len(x_test)
    )
    inputs_path = args.out_dir / "test-inputs.f32"
    labels_path = args.out_dir / "test-labels.u8"
    np.ascontiguousarray(x_test[:sample_count], dtype="<f4").tofile(inputs_path)
    np.ascontiguousarray(test_labels[:sample_count], dtype=np.uint8).tofile(labels_path)
    subset_accuracy = float(
        (logits[:sample_count].argmax(1) == test_labels[:sample_count]).mean()
    )
    print(
        f"wrote {inputs_path} ({inputs_path.stat().st_size} bytes) and "
        f"{labels_path} ({labels_path.stat().st_size} bytes): "
        f"{sample_count} samples, exported accuracy {subset_accuracy:.4f}"
    )
    print(
        "run: ./bnn-infer-v2 classify "
        f"{model_path} --input {inputs_path} --labels {labels_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
