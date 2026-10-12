#!/usr/bin/env python3
"""Offline MLP/CNN/GPT migration regression using only the Python standard library.

Run from any directory:
    python3 tests/test_mlp.py --binary ./bnn-infer --library build/soft/libbnn-engine.a

All generated models, IDX data, references and executables live in a temporary
folder. --sanitize builds the API runner and old CLI with ASan/UBSan; build the
engine archive with the same sanitizers to inspect its partial-load cleanup too.
"""

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import re
import shlex
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    """Round after every arithmetic operation, as the original scalar C does."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass
class Layer:
    inputs: int
    outputs: int
    relu: int
    bits: bytes
    half_bits: list
    post_a: list
    post_b: list
    group: int = 128

    def serialize(self):
        return (struct.pack("<4I", self.inputs, self.outputs, self.group, self.relu)
                + self.bits
                + struct.pack(f"<{len(self.half_bits)}H", *self.half_bits)
                + struct.pack(f"<{self.outputs}f", *self.post_a)
                + struct.pack(f"<{self.outputs}f", *self.post_b))


def model_bytes(layers):
    return b"BNN1" + struct.pack("<I", len(layers)) + b"".join(
        layer.serialize() for layer in layers)


def make_layers(dimensions, seed):
    rng = random.Random(seed)
    # Include both signs of zero, normal and subnormal scales. The random
    # significands and bit-packed weights stress sign/row/group interpretation.
    specials = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF,
                0x0400, 0x8400, 0x2800, 0xA800, 0x2C00, 0xAC00]
    layers = []
    for index, (inputs, outputs) in enumerate(zip(dimensions, dimensions[1:])):
        scale_bits = []
        for i in range(outputs * (inputs // 128)):
            scale_bits.append(specials[i] if i < len(specials)
                              else rng.choice((0x2400, 0x2800, 0x2C00, 0xA400,
                                               0xA800, 0xAC00)) + rng.randrange(256))
        post_a = [f32(rng.uniform(-1.25, 1.25)) for _ in range(outputs)]
        post_b = [f32(rng.uniform(-0.5, 0.5)) for _ in range(outputs)]
        post_a[0] = 0.0
        post_a[1] = -1.0
        layers.append(Layer(inputs, outputs, int(index < len(dimensions) - 2),
                            bytes(rng.getrandbits(8) for _ in range(outputs * inputs // 8)),
                            scale_bits, post_a, post_b))
    return layers


def quantize(vector):
    absmax = max(map(abs, vector))
    scale = f32(absmax / 127.0) if absmax else 1.0
    result = []
    for value in vector:
        ratio = f32(value / scale)
        q = math.floor(abs(ratio) + 0.5) * (1 if ratio >= 0 else -1)
        result.append(max(-127, min(127, q)))
    return result, scale


def signed_dot(layer, quantized, row):
    groups = layer.inputs // layer.group
    stride = (layer.inputs + 7) // 8
    acc = 0.0
    for group in range(groups):
        dot = 0
        for k in range(group * layer.group, (group + 1) * layer.group):
            byte = layer.bits[row * stride + k // 8]
            dot += quantized[k] if byte & (1 << (k % 8)) else -quantized[k]
        half = struct.unpack("<e", struct.pack("<H", layer.half_bits[row * groups + group]))[0]
        acc = f32(acc + f32(half * dot))
    return acc


def oracle(layers, vector):
    """Independent signed-dot FP32 oracle; no engine or accelerator code is used.

    The implementation directly sums +/-q instead of the legacy C's packed-bit
    2*positive-total algorithm. Group results are accumulated in wire order.
    """
    x = list(vector)
    for layer in layers:
        assert len(x) == layer.inputs
        quantized, scale = quantize(x)
        result = []
        for row in range(layer.outputs):
            acc = signed_dot(layer, quantized, row)
            value = f32(f32(layer.post_a[row] * f32(acc * scale)) + layer.post_b[row])
            result.append(0.0 if layer.relu and value < 0.0 else value)
        x = result
    assert all(map(math.isfinite, x))
    return x


def inputs_for(width, seed):
    rng = random.Random(seed)
    # All-zero activations, both signs, exact rounding ties, negative absmax,
    # and random inputs with substantially different quantization scales.
    values = [[0.0] * width, [1.0] * width, [-2.0] * width]
    ties = [(-1.5, -0.5, 0.5, 1.5)[i % 4] for i in range(width)]
    ties[0] = 127.0  # xs=1, so ties must be rounded away from zero.
    values.append(ties)
    for amplitude in (0.0625, 3.0, 24.0):
        values.append([f32(rng.uniform(-amplitude, amplitude)) for _ in range(width)])
    return values


def preprocess(pixels):
    mean, std = f32(0.1307), f32(0.3081)
    return [f32(f32(f32(value / 255.0) - mean) / std) for value in pixels] + [0.0] * 112


def write_fixture(folder, stem, layers, vectors):
    (folder / f"{stem}.bnn").write_bytes(model_bytes(layers))
    expected = [oracle(layers, vector) for vector in vectors]
    stats = (len(layers), sum(layer.inputs for layer in layers),
             sum(layer.outputs for layer in layers) * 4,
             sum(layer.inputs * layer.outputs for layer in layers))
    api_fixture(folder, stem, 1, len(layers), (1, [layers[0].inputs]),
                (1, [layers[-1].outputs]), vectors, expected,
                [(stats[0], stats[1], stats[2], stats[3], stats[3])] * len(vectors))
    return expected


def write_bad_models(folder, layers):
    valid = model_bytes(layers)
    offsets = []
    offset = 8
    for layer in layers:
        scales = offset + 16 + len(layer.bits)
        post_a = scales + len(layer.half_bits) * 2
        post_b = post_a + layer.outputs * 4
        end = post_b + layer.outputs * 4
        offsets.append((offset, scales, post_a, post_b, end))
        offset = end
    bad = {}

    def mutation(name, offset, fmt, value, status=-6):
        data = bytearray(valid)
        struct.pack_into(fmt, data, offset, value)
        bad[name] = (bytes(data), status)

    # Truncate every kind of field, including a later layer after allocations.
    cuts = {0, 1, 3, 4, 7, 8, len(valid) - 1}
    for header, scales, post_a, post_b, end in offsets:
        cuts.update((header + 1, header + 15, scales - 1, scales + 1,
                     post_a - 1, post_a + 1, post_b - 1, post_b + 1, end - 1))
    for cut in sorted(cuts):
        bad[f"truncated-{cut}.bnn"] = (valid[:cut], -6)
    bad["unsupported-cnn.bnn"] = (b"BNC1" + valid[4:], -3)
    bad["unsupported-random.bnn"] = (b"????" + valid[4:], -3)
    bad["trailing-byte.bnn"] = (valid + b"!", -6)
    mutation("zero-layers.bnn", 4, "<I", 0)
    mutation("too-many-layers.bnn", 4, "<I", 9)
    mutation("zero-input.bnn", 8, "<I", 0)
    mutation("huge-input.bnn", 8, "<I", 0xFFFFFFFF)
    mutation("zero-output.bnn", 12, "<I", 0)
    mutation("zero-group.bnn", 16, "<I", 0)
    mutation("invalid-group.bnn", 16, "<I", 7)
    mutation("bad-relu.bnn", 20, "<I", 2)
    mutation("mismatched-second-layer.bnn", offsets[1][0], "<I", 256)
    for index, (_, scales, post_a, post_b, _) in enumerate(offsets):
        for name, bits in (("nan", 0x7E00), ("inf", 0x7C00), ("negative-inf", 0xFC00)):
            mutation(f"layer{index}-scale-{name}.bnn", scales, "<H", bits)
        mutation(f"layer{index}-post-a-nan.bnn", post_a, "<I", 0x7FC00000)
        mutation(f"layer{index}-post-b-inf.bnn", post_b, "<I", 0x7F800000)
    caps = make_layer(128, 10, 64, 0, random.Random(9))
    bad["caps-group64.bnn"] = (model_bytes([caps]), -3)
    manifest = []
    for name, (content, status) in bad.items():
        (folder / name).write_bytes(content)
        manifest.append(f"{name} {status}\n")
    manifest.append("does-not-exist.bnn -5\n")
    (folder / "bad-models.txt").write_text("".join(manifest), encoding="ascii")
    return len(manifest)


def api_fixture(folder, stem, kind, layers, input_desc, output_desc,
                vectors, expected, stats, vocab=0, max_length=0):
    def desc(value):
        dtype, dims = value
        return struct.pack("<6I", dtype, len(dims), *(dims + [0] * (4 - len(dims))))
    data = bytearray(b"TST2" + struct.pack("<2I", kind, layers)
                     + desc(input_desc) + desc(output_desc)
                     + struct.pack("<3I", vocab, max_length, len(vectors)))
    for vector, logits, counters in zip(vectors, expected, stats):
        data += struct.pack("<3I5Q", len(vector) if kind == 3 else 0,
                            len(vector), len(logits), *counters)
        data += struct.pack(f"<{len(vector)}{'i' if kind == 3 else 'f'}", *vector)
        data += struct.pack(f"<{len(logits)}f", *logits)
    (folder / f"{stem}.bin").write_bytes(data)


def make_layer(inputs, outputs, group, relu, rng):
    return Layer(inputs, outputs, relu,
                 bytes(rng.getrandbits(8) for _ in range(outputs * ((inputs + 7) // 8))),
                 [rng.choice((0x2000, 0x2400, 0x2800, 0xA000, 0xA400))
                  for _ in range(outputs * (inputs // group))],
                 [f32(rng.uniform(0.5, 1.25)) for _ in range(outputs)],
                 [f32(rng.uniform(-0.1, 0.1)) for _ in range(outputs)], group)


def cnn_model(seed, fc_group):
    rng = random.Random(seed)
    return [make_layer(9, 8, 9, 1, rng), None,
            make_layer(72, 16, 8, 1, rng), None,
            make_layer(784, 128, 112, 1, rng), make_layer(128, 10, fc_group, 0, rng)]


def cnn_bytes(layers):
    data = bytearray(b"BNN2" + struct.pack("<I", len(layers)))
    for i, layer in enumerate(layers):
        if layer is None:
            data += struct.pack("<2I", 2, 2)
            continue
        if i in (0, 2):
            data += struct.pack("<6I", 0, 1 if i == 0 else 8,
                                layer.outputs, 3, layer.group, layer.relu)
        else:
            data += struct.pack("<5I", 1, layer.inputs, layer.outputs, layer.group, layer.relu)
        data += layer.serialize()[16:]
    return bytes(data)


def cnn_oracle(layers, vector):
    x, channels, size = list(vector), 1, 28
    for i, layer in enumerate(layers):
        if layer is None:
            small = size // 2
            x = [max(x[c*size*size + (h*2+dh)*size + w*2+dw]
                     for dh in range(2) for dw in range(2))
                 for c in range(channels) for h in range(small) for w in range(small)]
            size = small
        elif i in (0, 2):
            q, scale = quantize(x)
            y = [0.0] * (layer.outputs * size * size)
            for h in range(size):
                for w in range(size):
                    patch = [q[c*size*size + (h+dh)*size + w+dw]
                             if 0 <= h+dh < size and 0 <= w+dw < size else 0
                             for c in range(channels) for dh in (-1, 0, 1) for dw in (-1, 0, 1)]
                    for row in range(layer.outputs):
                        value = f32(f32(layer.post_a[row] * f32(signed_dot(layer, patch, row) * scale))
                                    + layer.post_b[row])
                        y[row*size*size + h*size+w] = max(0.0, value) if layer.relu else value
            x, channels = y, layer.outputs
        else:
            x = oracle([layer], x)
    return x


def cnn_stats(layers):
    written = read = logical = executed = calls = 0
    size = 28
    for i, layer in enumerate(layers):
        if layer is None:
            size //= 2
            continue
        rows = size * size if i in (0, 2) else 1
        padded = layer.inputs // layer.group * 128
        calls += 1
        written += rows * padded
        read += rows * layer.outputs * 4
        logical += rows * layer.inputs * layer.outputs
        executed += rows * padded * layer.outputs
    return calls, written, read, logical, executed


WORDS = ["春", "天", "春天", "a", "word", "!"]


def gpt_model():
    rng = random.Random(991)
    gains = [[f32(rng.uniform(0.75, 1.25)) for _ in range(128)] for _ in range(3)]
    embedding = [[f32(rng.uniform(-0.4, 0.4)) for _ in range(128)] for _ in WORDS]
    position = [[f32(rng.uniform(-0.08, 0.08)) for _ in range(128)] for _ in range(4)]
    layers = [make_layer(a, b, 128, relu, rng) for a, b, relu in
              [(128, 128, 0)] * 4 + [(128, 512, 1), (512, 128, 0), (128, len(WORDS), 0)]]
    return gains, embedding, position, layers


def gpt_bytes(model):
    gains, embedding, position, layers = model
    data = bytearray(b"GPT1" + struct.pack("<6If", len(WORDS), 4, 128, 1, 1, 128, f32(1e-5)))
    def floats(values):
        return struct.pack(f"<{len(values)}f", *values)
    def weight(layer):
        return layer.serialize()[:-layer.outputs*8]
    data += floats([v for row in embedding + position for v in row])
    data += floats(gains[0]) + b"".join(weight(layer) for layer in layers[:4])
    data += floats(gains[1]) + b"".join(weight(layer) for layer in layers[4:6])
    data += floats(gains[2]) + weight(layers[6])
    for word in WORDS:
        encoded = word.encode("utf-8")
        data += bytes([len(encoded)]) + encoded
    return bytes(data)


def gpt_oracle(model, ids):
    gains, embedding, position, layers = model
    def norm(rows, gain):
        result = []
        for row in rows:
            ss = 0.0
            for v in row:
                ss = f32(ss + f32(v*v))
            rms = f32(math.sqrt(f32(f32(ss/128) + f32(1e-5))))
            result.append([f32(f32(v/rms)*g) for v, g in zip(row, gain)])
        return result
    def linear(layer, rows):
        # GPT wire records have no post affine transform.
        result = []
        for row in rows:
            q, scale = quantize(row)
            y = [f32(signed_dot(layer, q, o)*scale) for o in range(layer.outputs)]
            result.append([max(0.0, v) for v in y] if layer.relu else y)
        return result
    x = [[f32(a+b) for a, b in zip(embedding[token], position[t])] for t, token in enumerate(ids)]
    xb = norm(x, gains[0])
    q, k, v = [linear(layer, xb) for layer in layers[:3]]
    kq, scales = zip(*(quantize(row) for row in k))
    attention = []
    for t, qr in enumerate(q):
        total = 0.0
        for value in qr:
            total = f32(total + abs(value))
        alpha = struct.unpack("<e", struct.pack("<e", f32(total/128)))[0]
        scores = [f32(f32(f32(alpha * sum(a if b >= 0 else -a for a, b in zip(kr, qr)))
                            * scale) / f32(math.sqrt(128)))
                  for kr, scale in zip(kq[:t+1], scales[:t+1])]
        peak = max(scores)
        exps = [f32(math.exp(f32(s-peak))) for s in scores]
        total = 0.0
        for e in exps:
            total = f32(total+e)
        probabilities = [f32(e/total) for e in exps]
        row = []
        for j in range(128):
            acc = 0.0
            for tk, p in enumerate(probabilities):
                acc = f32(acc + f32(p*v[tk][j]))
            row.append(acc)
        attention.append(row)
    projected = linear(layers[3], attention)
    x = [[f32(a+b) for a, b in zip(row, other)] for row, other in zip(x, projected)]
    projected = linear(layers[5], linear(layers[4], norm(x, gains[1])))
    x = [[f32(a+b) for a, b in zip(row, other)] for row, other in zip(x, projected)]
    return linear(layers[6], norm(x, gains[2]))


def gpt_stats(n):
    linear = n * (4*128*128 + 2*128*512 + 128*len(WORDS))
    return (8, n*(4*128 + 128 + 512 + 128 + 128),
            4*(n*(4*128 + 512 + 128 + len(WORDS)) + n*n),
            linear + n*n*128, linear + n*n*128)


def add_bad_models(folder, stem, valid, mutations):
    bad = [(f"{stem}-truncated.bnn", valid[:-1], -6),
           (f"{stem}-late-truncated.bnn", valid[:len(valid)//2], -6),
           (f"{stem}-trailing.bnn", valid+b"!", -6)]
    for name, offset, fmt, value, status in mutations:
        data = bytearray(valid)
        struct.pack_into(fmt, data, offset, value)
        bad.append((f"{stem}-{name}.bnn", bytes(data), status))
    with (folder / "bad-models.txt").open("a", encoding="ascii") as manifest:
        for name, data, status in bad:
            (folder / name).write_bytes(data)
            manifest.write(f"{name} {status}\n")
    return len(bad)


def generate_other(folder, images):
    for stem, group in (("cnn", 16), ("cnn-export", 128)):
        layers = cnn_model(992, group)
        valid = cnn_bytes(layers)
        (folder / f"{stem}.bnn").write_bytes(valid)
        vectors = [preprocess(image)[:784] for image in images[:2]]
        # An asymmetric impulse and image pattern reveal padding and CHW mistakes.
        impulse = [0.0]*784
        impulse[1], impulse[28], impulse[-2] = 3.0, -1.0, 2.0
        vectors += [impulse, preprocess(images[3])[:784]]
        expected = [cnn_oracle(layers, vector) for vector in vectors]
        api_fixture(folder, stem, 2, 6, (1, [1, 28, 28]), (1, [10]), vectors, expected,
                    [cnn_stats(layers)]*len(vectors))
        if stem == "cnn":
            logits = [value for row in expected[:2] for value in row]
            (folder / "cnn-ref.bin").write_bytes(struct.pack("<2I", 2, 10) + struct.pack("<20f", *logits))
            preds = [max(range(10), key=lambda i: row[i]) for row in expected[:2]]
            (folder / "cnn-images.idx").write_bytes(struct.pack(">4I", 2051, 2, 28, 28)+b"".join(images[:2]))
            (folder / "cnn-labels.idx").write_bytes(struct.pack(">2I", 2049, 2)+bytes(preds))
            bad_count = add_bad_models(folder, stem, valid,
                                      [("type", 8, "<I", 99, -6), ("group", 24, "<I", 0, -6)])
            capped = list(layers)
            capped[4] = make_layer(784, 128, 16, 1, random.Random(993))
            (folder / "cnn-caps.bnn").write_bytes(cnn_bytes(capped))
            with (folder / "bad-models.txt").open("a", encoding="ascii") as manifest:
                manifest.write("cnn-caps.bnn -3\n")
            bad_count += 1
    model = gpt_model()
    valid = gpt_bytes(model)
    (folder / "gpt.bnn").write_bytes(valid)
    sequences = [[4], [0, 1, 2, 5], [4, 3], [2, 4, 0], [5]]
    expected = [gpt_oracle(model, ids) for ids in sequences]
    api_fixture(folder, "gpt", 3, 1, (2, [4]), (1, [4, len(WORDS)]), sequences,
                [[v for row in rows for v in row] for rows in expected],
                [gpt_stats(len(ids)) for ids in sequences], len(WORDS), 4)
    ids, rows = sequences[1], expected[1]
    logits = [v for row in rows for v in row]
    reference = struct.pack("<2I", len(ids), len(WORDS)) + struct.pack("<4i", *ids)
    (folder / "gpt-logits.bin").write_bytes(reference+struct.pack(f"<{len(logits)}f", *logits))
    logits[-1] += 1.0 + abs(logits[-1])
    (folder / "gpt-wrong-logits.bin").write_bytes(reference+struct.pack(f"<{len(logits)}f", *logits))
    eval_ids = [0, 1, 4, 3, 5, 2, 4, 0, 1]
    (folder / "gpt-eval.bin").write_bytes(struct.pack("<I9i", len(eval_ids), *eval_ids))
    bad_ids = bytearray((folder / "gpt-logits.bin").read_bytes())
    struct.pack_into("<i", bad_ids, 8+3*4, len(WORDS))
    (folder / "gpt-bad-ids.bin").write_bytes(bad_ids)
    bad_count += add_bad_models(folder, "gpt", valid,
                               [("zero-vocab", 4, "<I", 0, -6),
                                ("zero-group", 24, "<I", 0, -6),
                                ("eps-nan", 28, "<I", 0x7fc00000, -6),
                                ("embedding-inf", 32, "<I", 0x7f800000, -6)])
    generated = [2, 4]  # Longest-match tokenization of 春天word.
    for _ in range(6):
        last = gpt_oracle(model, generated[-4:])[-1]
        generated.append(max(range(len(WORDS)), key=lambda i: last[i]))
    return bad_count, "".join(WORDS[i] for i in generated), model, eval_ids


def generate(folder):
    models = {"small-a": make_layers((128, 128, 10), 1729),
              "small-b": make_layers((128, 128, 10), 1730),
              "mnist": make_layers((896, 128, 10), 1731),
              "deep": make_layers((128, 256, 128, 10), 1732)}
    for index, stem in enumerate(("small-a", "small-b", "deep")):
        write_fixture(folder, stem, models[stem], inputs_for(128, index))
    rng = random.Random(5050)
    images = [bytes([0] * 784), bytes([255] * 784),
              bytes(0 if i % 2 else 255 for i in range(784)),
              bytes(i % 256 for i in range(784))]
    images.extend(bytes(rng.randrange(256) for _ in range(784)) for _ in range(4))
    expected = write_fixture(folder, "mnist", models["mnist"], list(map(preprocess, images)))
    (folder / "images.idx").write_bytes(struct.pack(">4I", 2051, len(images), 28, 28) + b"".join(images))
    predictions = [max(range(10), key=lambda i: row[i]) for row in expected]
    # Deliberately make two labels wrong, so preprocessing failures cannot pass
    # merely by producing the same trivial 100% accuracy summary.
    labels = list(predictions)
    labels[0] = (labels[0] + 1) % 10
    labels[-1] = (labels[-1] + 1) % 10
    (folder / "labels.idx").write_bytes(struct.pack(">2I", 2049, len(images)) + bytes(labels))
    flat = [value for row in expected for value in row]
    reference = struct.pack("<2I", len(images), 10) + struct.pack(f"<{len(flat)}f", *flat)
    (folder / "ref.bin").write_bytes(reference)
    # The perturbed oracle tests that the CLI actually reads and compares every
    # sample/class rather than printing an unconditional success message.
    flat[-1] += 1.0 + abs(flat[-1])
    (folder / "wrong-ref.bin").write_bytes(struct.pack("<2I", len(images), 10)
                                          + struct.pack(f"<{len(flat)}f", *flat))
    bad_count = write_bad_models(folder, models["mnist"])
    extra_bad, generated, gpt, eval_ids = generate_other(folder, images)
    return len(images) - 2, len(images), bad_count + extra_bad, generated, gpt, eval_ids


def run(command, env=None, expected_returncode=0):
    completed = subprocess.run([str(arg) for arg in command], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, timeout=120)
    if completed.returncode != expected_returncode:
        raise AssertionError(f"command exited {completed.returncode}: {shlex.join(map(str, command))}\n"
                             + completed.stdout)
    return completed.stdout


def cli_metrics(output):
    accuracy = re.search(r"C inference accuracy:\s*([\d.]+)\s*\((\d+)/(\d+)\)", output)
    error = re.search(r"C vs numpy ref\s+max rel err:\s*([\d.eE+-]+)\s+(OK|MISMATCH)", output)
    if error is None:
        error = re.search(r"C vs reference logits \(\d+ items\): max relative error "
                          r"([\d.eE+-]+), tolerance [\d.eE+-]+ (OK|MISMATCH)", output)
    if error is None:
        error = re.search(r"C vs Python reference:.*max relative error ([\d.eE+-]+).* (OK|MISMATCH)", output)
    assert accuracy and error, f"missing numerical summary:\n{output}"
    return (int(accuracy[2]), int(accuracy[3])), float(error[1]), error[2]


def cli_regression(binary, legacy, folder, correct, count, env):
    paths = [folder / "mnist.bnn", folder / "images.idx", folder / "labels.idx"]
    old_outputs = {}
    for flags in ([], ["--soft"]):
        output = run([legacy, *flags, *paths, folder / "ref.bin"], env)
        accuracy, error, verdict = cli_metrics(output)
        assert accuracy == (correct, count), output
        assert error <= 2e-6 and verdict == "OK", output
        old_outputs[bool(flags)] = (accuracy, error)
    for flags in ([], ["--profile"]):
        output = run([binary, *flags, *paths, folder / "ref.bin"], env)
        accuracy, error, verdict = cli_metrics(output)
        assert accuracy == old_outputs["--soft" in flags][0], output
        assert error <= 2e-6 and verdict == "OK", output
        print(f"CLI {' '.join(flags) or 'accelerator'}: {correct}/{count}, oracle error {error:.2e}")
        if flags:
            assert profile_calls(output) == {"eval": count, "check": count}, output
    run([binary, "--soft", *paths], env, expected_returncode=1)
    for executable in (legacy, binary):
        output = run([executable, *(["--soft"] if executable == legacy else []),
                      *paths, folder / "wrong-ref.bin"], env,
                     expected_returncode=0 if executable == legacy else 1)
        _, error, verdict = cli_metrics(output)
        assert error > 0.1 and verdict == "MISMATCH", output


def profile_calls(output):
    return {phase: int(count) for phase, count in
            re.findall(r"profile (\w+):.*forward_calls=(\d+)", output)}


def other_cli_regression(binary, legacy_cnn, legacy_gpt, folder, generated, model, eval_ids, env):
    paths = [folder / "cnn.bnn", folder / "cnn-images.idx", folder / "cnn-labels.idx"]
    for executable, flags in ((legacy_cnn, []), (legacy_cnn, ["--soft"]),
                              (binary, []), (binary, ["--profile"])):
        output = run([executable, *flags, *paths, folder / "cnn-ref.bin"], env)
        accuracy, error, verdict = cli_metrics(output)
        assert accuracy == (2, 2) and error <= 2e-6 and verdict == "OK", output
        if flags == ["--profile"]:
            assert profile_calls(output) == {"eval": 2, "check": 2}, output
    old = run([legacy_gpt, "--soft", "--steps", 0, folder / "gpt.bnn",
               folder / "gpt-logits.bin"], env)
    assert "C vs python ref:" in old and "MISMATCH" not in old, old
    for flags, positional in (([], False), (["--profile"], False), (["--profile"], True)):
        args = [folder / "gpt.bnn", folder / "gpt-logits.bin"] if positional else [
            "--ref", folder / "gpt-logits.bin", folder / "gpt.bnn"]
        output = run([binary, *flags, *args], env)
        assert "C vs Python reference:" in output and "MISMATCH" not in output, output
        assert "generate:" not in output and "token eval:" not in output, output
        if flags:
            assert profile_calls(output) == {"check": 1}, output
    run([binary, "--ref", folder / "gpt-wrong-logits.bin", folder / "gpt.bnn"], env,
        expected_returncode=1)
    run([binary, "--ref", folder / "gpt-bad-ids.bin", folder / "gpt.bnn"], env,
        expected_returncode=1)
    output = run([binary, "--profile", "--prompt", "春天word", "--steps", 6,
                  folder / "gpt.bnn"], env)
    assert f"generate: {generated}\n" in output and profile_calls(output) == {"generate": 6}, output
    # Reference/eval and explicit generation run only their requested phases.
    output = run([binary, "--profile", "--ref", folder / "gpt-logits.bin", "--steps", 1,
                  folder / "gpt.bnn"], env)
    assert profile_calls(output) == {"check": 1, "generate": 1}, output
    output = run([binary, "--profile", folder / "gpt.bnn"], env)
    assert "generate: 春天" in output and profile_calls(output) == {"generate": 40}, output
    expected_loss, correct, count = 0.0, 0, 0
    for offset in range(0, len(eval_ids)-4, 4):
        for row, target in zip(gpt_oracle(model, eval_ids[offset:offset+4]), eval_ids[offset+1:offset+5]):
            peak = max(row)
            expected_loss += peak - row[target] + math.log(sum(math.exp(v-peak) for v in row))
            correct += max(range(len(row)), key=lambda i: row[i]) == target
            count += 1
    for positional in (False, True):
        args = [folder / "gpt.bnn", folder / "gpt-logits.bin", folder / "gpt-eval.bin"] if positional else [
            "--eval", folder / "gpt-eval.bin", folder / "gpt.bnn"]
        output = run([binary, "--profile", *args], env)
        result = re.search(r"C token eval: (\d+) positions, loss ([\d.]+), perplexity ([\d.]+), top1 ([\d.]+)", output)
        assert result and int(result[1]) == count, output
        assert abs(float(result[2])-expected_loss/count) < 1e-4, output
        assert abs(float(result[3])-math.exp(expected_loss/count)) < 5e-4, output
        assert abs(float(result[4])-correct/count) < 1e-4, output
        assert "generate:" not in output, output
        assert profile_calls(output) == ({"check": 1, "eval": 2} if positional else {"eval": 2}), output
    for stem in ("cnn", "gpt"):
        run([binary, "--soft", folder / f"{stem}.bnn"], env, expected_returncode=1)
    print("CNN/GPT CLI: Python oracle, old CLI, ref/eval action selection, greedy generation and profile OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=ROOT / "bnn-infer")
    parser.add_argument("--library", type=Path, default=ROOT / "build/soft/libbnn-engine.a")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true", help="ASan/UBSan for temporary executables")
    parser.add_argument("--api-only", action="store_true", help="skip CLI migration comparison")
    parser.add_argument("--fixtures-only", type=Path, help="generate fixtures at this directory without compiling")
    args = parser.parse_args()
    if args.fixtures_only:
        args.fixtures_only.mkdir(parents=True, exist_ok=True)
        correct, count, bad_count, _, _, _ = generate(args.fixtures_only)
        print(f"Generated fixtures: {count} IDX images, {bad_count} failed-load cases in {args.fixtures_only}")
        return
    library, binary = args.library.resolve(), args.binary.resolve()
    assert library.is_file(), f"build the engine archive first: {library}"
    if not args.api_only:
        assert binary.is_file(), f"build the migrated CLI first: {binary}"
    compiler = shlex.split(args.cc)
    flags = ["-std=c99", "-O2", "-Wall", "-Wextra"]
    flags += shlex.split(os.environ.get("CFLAGS", ""))
    flags += ["-ffp-contract=off", "-fno-fast-math"]
    link_flags = shlex.split(os.environ.get("LDFLAGS", ""))
    env = os.environ.copy()
    if args.sanitize:
        flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        env.setdefault("ASAN_OPTIONS", "detect_leaks=1:halt_on_error=1")
        env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
    with tempfile.TemporaryDirectory(prefix="bnn-mlp-test-") as temporary:
        folder = Path(temporary)
        correct, count, bad_count, generated, gpt, eval_ids = generate(folder)
        runner = folder / "test-bnn-engine"
        run([*compiler, *flags, "-I", ROOT / "bnn-sw", ROOT / "tests/test_bnn_engine.c",
             library, *link_flags, "-lm", "-o", runner], env)
        print(run([runner, folder], env).strip())
        print(f"Failed-load matrix: {bad_count} malformed/missing/unsupported models, two repetitions")
        if not args.api_only:
            legacy = folder / "legacy-mlp"
            run([*compiler, *flags, "-I", ROOT, ROOT / "bnn-MNIST-MLP.c",
                 ROOT / "bnn-accelerator/soft-bnn-accelerator.c", *link_flags, "-lm", "-o", legacy], env)
            cli_regression(binary, legacy, folder, correct, count, env)
            legacy_cnn, legacy_gpt = folder / "legacy-cnn", folder / "legacy-gpt"
            for source, executable in (("bnn-MNIST-CNN.c", legacy_cnn), ("bnn-nanoGPT.c", legacy_gpt)):
                run([*compiler, *flags, "-I", ROOT, ROOT / source,
                     ROOT / "bnn-accelerator/soft-bnn-accelerator.c", *link_flags, "-lm", "-o", executable], env)
            other_cli_regression(binary, legacy_cnn, legacy_gpt, folder, generated, gpt, eval_ids, env)
    print("MLP/CNN/GPT migration regressions passed (no downloads or training dependencies)")


if __name__ == "__main__":
    main()
