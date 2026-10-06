# train-MNIST-CNN-export.py — W1A8 MNIST CNN：训练 + 导出(BNN2) + numpy 参考验证
# 与 MLP 版同一套量化思想：±1 权重 + int8 激活（组内整数累加，组外 FP16 scale），
# 只是把算子从 Linear 扩展到 Conv2d / MaxPool2d，用新文件格式 BNN2 承载。
import argparse
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN, STD = 0.1307, 0.3081
N_REF = 100                     # 导出前 100 张图的参考 logits 给 C 端对比

# 结构参数：group 必须整除每行权重个数（conv 为 in_ch*k*k，fc 为 in_f）
CONV1 = dict(in_ch=1, out_ch=8, k=3, group=9)      # 1*3*3 = 9  -> 1 组
CONV2 = dict(in_ch=8, out_ch=16, k=3, group=8)     # 8*3*3 = 72 -> 9 组
FLAT = 16 * 7 * 7                                  # pool 两次后 16x7x7
FC1 = dict(in_f=FLAT, out_f=128, group=112)        # 784/112 = 7 组
FC2 = dict(in_f=128, out_f=10, group=128)          # 128/128 = 1 组


def pack_rows(bits):
    """[rows, total] 的 0/1 位打包成 [rows, ceil(total/8)]，行尾补 0（不会被访问）"""
    rows, total = bits.shape
    row_bytes = (total + 7) // 8
    flat = np.zeros((rows, row_bytes * 8), np.uint8)
    flat[:, :total] = bits
    return np.packbits(flat, axis=1, bitorder="little")


class BinaryConv(nn.Module):
    """权重 ±1，每 group 个（沿 in_ch*k*k 展平）共享 scale；训练用 latent FP + STE"""

    def __init__(self, in_ch, out_ch, k=3, group=8, binary=True, pad=1):
        super().__init__()
        total = in_ch * k * k
        assert total % group == 0, "group 必须整除 in_ch*k*k"
        self.in_ch, self.out_ch, self.k, self.group = in_ch, out_ch, k, group
        self.total, self.pad, self.binary = total, pad, binary
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k).uniform_(-1, 1) * total ** -0.5)

    def _groups(self):
        return self.weight.view(self.out_ch, self.total).view(self.out_ch, -1, self.group)

    def scales(self):                                              # [out_ch, n_groups]
        return self._groups().abs().mean(-1)

    def quantized(self):
        w = self._groups()
        alpha = w.abs().mean(-1, keepdim=True)
        sign = torch.where(w >= 0, 1.0, -1.0)
        return (sign * alpha).view(self.out_ch, self.in_ch, self.k, self.k)

    def forward(self, x):
        if not self.binary:
            return F.conv2d(x, self.weight, padding=self.pad)
        w_ste = self.weight + (self.quantized() - self.weight).detach()
        return F.conv2d(x, w_ste, padding=self.pad)


class BinaryLinearG(nn.Module):
    """FC 版二值层：权重 [out, in]，每 group 个输入共享一个 scale"""

    def __init__(self, in_f, out_f, group, binary=True):
        super().__init__()
        assert in_f % group == 0 and in_f % 8 == 0
        self.in_f, self.out_f, self.group, self.binary = in_f, out_f, group, binary
        self.weight = nn.Parameter(torch.empty(out_f, in_f).uniform_(-1, 1) * in_f ** -0.5)

    def _groups(self):
        return self.weight.view(self.out_f, -1, self.group)

    def scales(self):
        return self._groups().abs().mean(-1)

    def quantized(self):
        w = self._groups()
        alpha = w.abs().mean(-1, keepdim=True)
        sign = torch.where(w >= 0, 1.0, -1.0)
        return (sign * alpha).view(self.out_f, self.in_f)

    def forward(self, x):
        if not self.binary:
            return F.linear(x, self.weight)
        w_ste = self.weight + (self.quantized() - self.weight).detach()
        return F.linear(x, w_ste)


class CNN(nn.Module):
    def __init__(self, binary=True):
        super().__init__()
        self.c1 = BinaryConv(**CONV1, binary=binary)
        self.bn1 = nn.BatchNorm2d(CONV1["out_ch"])
        self.c2 = BinaryConv(**CONV2, binary=binary)
        self.bn2 = nn.BatchNorm2d(CONV2["out_ch"])
        self.fc1 = BinaryLinearG(**FC1, binary=binary)
        self.bn3 = nn.BatchNorm1d(FC1["out_f"])
        self.fc2 = BinaryLinearG(**FC2, binary=binary)
        self.bias = nn.Parameter(torch.zeros(FC2["out_f"]))

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.c2(x)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x) + self.bias


def fold_bn(bn):
    """BN(z)=gamma*(z-mean)/sqrt(var+eps)+beta -> a*z+b"""
    a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bn.bias - bn.running_mean * a
    return a, b


@torch.no_grad()
def write_conv(f, conv, a, b, relu):
    """conv 记录: type=0, in_ch, out_ch, k, group, relu(6*U32) + bits + scales + a + b"""
    w = conv.weight.cpu().view(conv.out_ch, conv.total)
    f.write(struct.pack("<6I", 0, conv.in_ch, conv.out_ch, conv.k, conv.group, relu))
    f.write(pack_rows((w >= 0).numpy().astype(np.uint8)).tobytes())
    f.write(conv.scales().cpu().numpy().astype("<f2").tobytes())
    f.write(a.cpu().numpy().astype("<f4").tobytes())
    f.write(b.cpu().numpy().astype("<f4").tobytes())


@torch.no_grad()
def write_fc(f, lin, a, b, relu):
    """fc 记录: type=1, in_f, out_f, group, relu(5*U32) + bits + scales + a + b"""
    w = lin.weight.cpu()
    f.write(struct.pack("<5I", 1, lin.in_f, lin.out_f, lin.group, relu))
    f.write(pack_rows((w >= 0).numpy().astype(np.uint8)).tobytes())
    f.write(lin.scales().cpu().numpy().astype("<f2").tobytes())
    f.write(a.cpu().numpy().astype("<f4").tobytes())
    f.write(b.cpu().numpy().astype("<f4").tobytes())


@torch.no_grad()
def export(model, path):
    a1, b1 = fold_bn(model.bn1)
    a2, b2 = fold_bn(model.bn2)
    a3, b3 = fold_bn(model.bn3)
    with open(path, "wb") as f:
        f.write(b"BNN2")
        f.write(struct.pack("<I", 6))
        write_conv(f, model.c1, a1, b1, 1)
        f.write(struct.pack("<2I", 2, 2))          # pool: type=2, size=2
        write_conv(f, model.c2, a2, b2, 1)
        f.write(struct.pack("<2I", 2, 2))
        write_fc(f, model.fc1, a3, b3, 1)
        write_fc(f, model.fc2, torch.ones(FC2["out_f"]), model.bias, 0)
    print(f"exported {path} (BNN2, 6 layers)")


def load_bnn2(path):
    """按 C 端完全相同的格式读回（顺带校验文件格式）"""
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == b"BNN2"
    n = struct.unpack_from("<I", buf, 4)[0]
    off, layers = 8, []
    for _ in range(n):
        typ = struct.unpack_from("<I", buf, off)[0]
        off += 4
        if typ == 2:
            size = struct.unpack_from("<I", buf, off)[0]
            off += 4
            layers.append(("pool", size))
        elif typ == 0:
            in_ch, out_ch, k, group, relu = struct.unpack_from("<5I", buf, off)
            off += 20
            total, rb = in_ch * k * k, (in_ch * k * k + 7) // 8
            packed = np.frombuffer(buf, np.uint8, out_ch * rb, off).reshape(out_ch, rb)
            off += out_ch * rb
            ng = total // group
            scales = np.frombuffer(buf, "<f2", out_ch * ng, off).astype(np.float32).reshape(out_ch, ng)
            off += 2 * out_ch * ng
            a = np.frombuffer(buf, "<f4", out_ch, off).copy()
            off += 4 * out_ch
            b = np.frombuffer(buf, "<f4", out_ch, off).copy()
            off += 4 * out_ch
            layers.append(("conv", in_ch, out_ch, k, group, relu, packed, scales, a, b))
        else:
            in_f, out_f, group, relu = struct.unpack_from("<4I", buf, off)
            off += 16
            rb = (in_f + 7) // 8
            packed = np.frombuffer(buf, np.uint8, out_f * rb, off).reshape(out_f, rb)
            off += out_f * rb
            ng = in_f // group
            scales = np.frombuffer(buf, "<f2", out_f * ng, off).astype(np.float32).reshape(out_f, ng)
            off += 2 * out_f * ng
            a = np.frombuffer(buf, "<f4", out_f, off).copy()
            off += 4 * out_f
            b = np.frombuffer(buf, "<f4", out_f, off).copy()
            off += 4 * out_f
            layers.append(("fc", in_f, out_f, group, relu, packed, scales, a, b))
    assert off == len(buf), "文件长度与记录不符"
    return layers


def quant_int8(x):
    """逐样本 absmax 对称 int8 量化，与 C 端 roundf 对齐（round half away from zero）"""
    flat = x.reshape(x.shape[0], -1)
    amax = np.abs(flat).max(1)
    xs = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = x / xs.reshape((-1,) + (1,) * (x.ndim - 1))
    q = np.sign(q) * np.floor(np.abs(q) + 0.5)
    return np.clip(q, -127.0, 127.0).astype(np.float32), xs


def unpack_sign(packed, total):
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :total]
    return bits.astype(np.float32) * 2 - 1          # {0,1} -> {-1,+1}


def grouped_binary(Xq, sign, scales, group):
    """组内整数累加：out[p,o] = Σ_g scales[o,g] · Σ_{k∈g} sign[o,k]·Xq[p,k]"""
    out, ng = scales.shape
    g = Xq.shape[1] // group
    gsum = np.einsum("pgk,ogk->pog", Xq.reshape(Xq.shape[0], g, group), sign.reshape(out, g, group))
    return (gsum * scales[None]).sum(-1)


def im2col(xq, k):
    """[N,C,H,W] -> [N*H*W, C*k*k]，patch 内顺序为 (c, kh, kw)"""
    xp = np.pad(xq, ((0, 0), (0, 0), (k // 2, k // 2), (k // 2, k // 2)))
    N, C, H, W = xq.shape
    win = sliding_window_view(xp, (k, k), axis=(2, 3))       # [N,C,H,W,k,k]
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * k * k), H, W


def ref_forward(layers, x):
    """只用导出的 bit + fp16 scale 做推理，算术与 C 端 W1A8 完全一致"""
    for L in layers:
        if L[0] == "pool":
            N, C, H, W = x.shape
            x = x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))
            continue
        xq, xs = quant_int8(x)
        if L[0] == "conv":
            _, in_ch, out_ch, k, group, relu, packed, scales, a, b = L
            sign = unpack_sign(packed, in_ch * k * k)
            cols, H, W = im2col(xq, k)
            acc = grouped_binary(cols, sign, scales, group).reshape(x.shape[0], H, W, out_ch)
            acc = acc.transpose(0, 3, 1, 2) * xs[:, None, None, None]
            y = a[None, :, None, None] * acc + b[None, :, None, None]
        else:
            _, in_f, out_f, group, relu, packed, scales, a, b = L
            sign = unpack_sign(packed, in_f)
            acc = grouped_binary(xq.reshape(x.shape[0], -1), sign, scales, group)
            y = a * acc * xs[:, None] + b
        x = np.maximum(y, 0) if relu else y
    return x.astype(np.float32)


def verify_with_numpy(path, test_set):
    layers = load_bnn2(path)
    x = ((test_set.data.numpy().astype(np.float32) / 255.0 - MEAN) / STD)[:, None, :, :]
    logits = np.concatenate([ref_forward(layers, x[i:i + 1000]) for i in range(0, len(x), 1000)])
    acc = (logits.argmax(1) == test_set.targets.numpy()).mean()
    print(f"numpy reference (from exported file) accuracy: {acc:.4f}")
    with open("ref_logits.bin", "wb") as f:
        f.write(struct.pack("<II", N_REF, 10))
        f.write(logits[:N_REF].astype("<f4").tobytes())
    print("wrote ref_logits.bin (CNN)")


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = 0
    for x, y in loader:
        correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
    return correct / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fp", action="store_true", help="训练全精度基线做对比（不导出）")
    ap.add_argument("--out", default="model-cnn.bnn")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = CNN(binary=not args.fp).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            loss = F.cross_entropy(model(x.to(dev)), y.to(dev))
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
