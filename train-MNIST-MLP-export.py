# train-MNIST-MLP-export.py — W1A8 MNIST MLP：训练 + 导出 + numpy 参考验证
# 唯一支持的推理工作模式：±1 权重 + int8 激活（组内整数累加，g128 FP16 scale）
import argparse
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

GROUP = 128
IN_RAW, IN_PAD = 784, 896          # 784 补零到 7*128
MEAN, STD = 0.1307, 0.3081
N_REF = 100                        # 导出前 100 张图的参考 logits 给 C 端对比


class BinaryLinearG(nn.Module):
    """权重 ±1，每 GROUP 个输入共享一个 scale；训练用 latent FP 权重 + STE"""

    def __init__(self, in_f, out_f, group=GROUP, binary=True):
        super().__init__()
        assert in_f % group == 0 and in_f % 8 == 0
        self.in_f, self.out_f, self.group, self.binary = in_f, out_f, group, binary
        self.weight = nn.Parameter(torch.empty(out_f, in_f).uniform_(-1, 1) * in_f ** -0.5)

    def scales(self):
        # 对 sign 量化，L2 误差最小的组 scale = 组内 |w| 均值
        w = self.weight.view(self.out_f, -1, self.group)
        return w.abs().mean(-1)                                  # [out, n_groups]

    def quantized(self):
        w = self.weight.view(self.out_f, -1, self.group)
        alpha = w.abs().mean(-1, keepdim=True)
        sign = torch.where(w >= 0, 1.0, -1.0)                    # 0 归为 +1，与导出一致
        return (sign * alpha).view(self.out_f, self.in_f)

    def forward(self, x):
        if not self.binary:
            return F.linear(x, self.weight)
        w = self.weight
        w_ste = w + (self.quantized() - w).detach()              # 前向用量化值，反向梯度直通
        return F.linear(x, w_ste)


class MLP(nn.Module):
    def __init__(self, binary=True, hidden=512):
        super().__init__()
        self.fc1 = BinaryLinearG(IN_PAD, hidden, binary=binary)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = BinaryLinearG(hidden, hidden, binary=binary)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = BinaryLinearG(hidden, 10, binary=binary)
        self.bias3 = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x) + self.bias3


def flat_pad(x):
    return F.pad(x.view(x.size(0), -1), (0, IN_PAD - IN_RAW))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = 0
    for x, y in loader:
        correct += (model(flat_pad(x).to(dev)).argmax(1).cpu() == y).sum().item()
    return correct / len(loader.dataset)


def fold_bn(bn):
    # BN(z) = gamma*(z-mean)/sqrt(var+eps) + beta  ==>  a*z + b
    a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bn.bias - bn.running_mean * a
    return a, b


@torch.no_grad()
def export(model, path):
    a1, b1 = fold_bn(model.bn1)
    a2, b2 = fold_bn(model.bn2)
    specs = [(model.fc1, a1, b1, 1),
             (model.fc2, a2, b2, 1),
             (model.fc3, torch.ones(10), model.bias3, 0)]
    total_bits = total_scales = total_params = 0
    with open(path, "wb") as f:
        f.write(b"BNN1")
        f.write(struct.pack("<I", len(specs)))
        for lin, a, b, relu in specs:
            w = lin.weight.cpu()
            bits = (w >= 0).numpy().astype(np.uint8)                       # 1 表示 +1
            packed = np.packbits(bits, axis=1, bitorder="little")          # 第 k 个权重在字节 k>>3 的第 k&7 位
            scales = lin.scales().cpu().numpy().astype("<f2")
            f.write(struct.pack("<4I", lin.in_f, lin.out_f, lin.group, relu))
            f.write(packed.tobytes())
            f.write(scales.tobytes())
            f.write(a.cpu().numpy().astype("<f4").tobytes())
            f.write(b.cpu().numpy().astype("<f4").tobytes())
            total_bits += packed.nbytes
            total_scales += scales.nbytes
            total_params += w.numel()
    print(f"exported {path}: sign bits {total_bits} B, fp16 scales {total_scales} B, "
          f"FP32 equivalent {total_params * 4} B ({total_params * 4 / (total_bits + total_scales):.1f}x)")


def load_bnn(path):
    """按 C 端完全相同的格式读回，顺带验证文件格式"""
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == b"BNN1"
    n = struct.unpack_from("<I", buf, 4)[0]
    off, layers = 8, []
    for _ in range(n):
        in_f, out_f, group, relu = struct.unpack_from("<4I", buf, off); off += 16
        nb = out_f * in_f // 8
        packed = np.frombuffer(buf, np.uint8, nb, off).reshape(out_f, in_f // 8); off += nb
        ns = out_f * (in_f // group)
        scales = np.frombuffer(buf, "<f2", ns, off).astype(np.float32).reshape(out_f, -1); off += 2 * ns
        a = np.frombuffer(buf, "<f4", out_f, off); off += 4 * out_f
        b = np.frombuffer(buf, "<f4", out_f, off); off += 4 * out_f
        layers.append((in_f, out_f, group, relu, packed, scales, a, b))
    assert off == len(buf)
    return layers


def ref_forward(layers, x):
    """只用导出的 bit + fp16 scale 做推理，计算方式与 C 端 W1A8 完全一致：
    每层输入按 absmax 对称量化为 int8，组内整数累加，组外每组乘一次 scale。"""
    for in_f, out_f, group, relu, packed, scales, a, b in layers:
        s = np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32) * 2 - 1   # [out, in] ∈ {±1}

        # 激活量化：对称 int8，scale = absmax / 127（逐样本）
        amax = np.abs(x).max(axis=1, keepdims=True)
        xs = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q = x / xs
        q = np.sign(q) * np.floor(np.abs(q) + 0.5)     # round half away from zero，对齐 C 的 roundf
        xq = np.clip(q, -127.0, 127.0).astype(np.float32)

        ng = in_f // group
        gsum = np.einsum("ngk,ogk->nog", xq.reshape(-1, ng, group), s.reshape(out_f, ng, group))
        acc = (gsum * scales[None]).sum(-1) * xs       # 每组乘一次 scale，最后乘回激活 scale
        x = a * acc + b
        if relu:
            x = np.maximum(x, 0)
    return x.astype(np.float32)


def verify_with_numpy(path, test_set):
    layers = load_bnn(path)
    raw = test_set.data.numpy().reshape(-1, IN_RAW).astype(np.float32)
    x = (raw / 255.0 - MEAN) / STD
    x = np.pad(x, ((0, 0), (0, IN_PAD - IN_RAW))).astype(np.float32)
    logits = np.concatenate([ref_forward(layers, x[i:i + 1000]) for i in range(0, len(x), 1000)])
    acc = (logits.argmax(1) == test_set.targets.numpy()).mean()
    print(f"numpy reference (from exported file) accuracy: {acc:.4f}")
    with open("ref_logits.bin", "wb") as f:
        f.write(struct.pack("<II", N_REF, 10))
        f.write(logits[:N_REF].astype("<f4").tobytes())
    print("wrote ref_logits.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fp", action="store_true", help="训练全精度基线做对比（不导出）")
    ap.add_argument("--out", default="model.bnn")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = MLP(binary=not args.fp).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = flat_pad(x).to(dev), y.to(dev)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1}: loss {loss.item():.4f}, test acc {evaluate(model, test_loader, dev):.4f}")

    if args.fp:
        return
    export(model, args.out)
    verify_with_numpy(args.out, test_set)


if __name__ == "__main__":
    main()