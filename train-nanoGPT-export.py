#!/usr/bin/env python3
# train-nanoGPT-export.py — 最小 nanoGPT 演示（中文 + 单汉字分词）的 W1A8 量化训练与推理评估
#
# 参考 karpathy/nanoGPT 的 decoder-only Transformer，规模压缩到 CPU 可训练：
#   字符级分词（一个汉字 = 一个 token）、n_layer=2 / n_head=1 / n_embd=128 / block_size=32
#
# 唯一的推理工作模式与 bnn-accelerator.h 完全对齐（W1A8）：
#   权重  ±1，每 GROUP=128 个输入共享一个 FP16 scale（组内整数点积，组外一次浮点乘）
#   激活  逐 token（逐行）对称量化为 int8，scale = absmax / 127
# 训练时对权重和激活都做量化感知（STE：前向走量化值，反向梯度直通）。
#
# 加速器只负责大矩阵乘：Q/K/V 投影、注意力分数（q·k）、注意力输出投影、MLP、lm_head；
# softmax / attn@v / 残差加 / RMSNorm / embedding 查表保持在 FP32。
#
# 注意力分数的加速器视角（注意方向）：X = 逐行 int8 的 key，W = ±1(+FP16 组 scale) 的 query。
# 这样动态矩阵（query）在增量解码时只有 N=1 行，最省。
#
# 训练结束后把权重按 sign bit + FP16 组 scale 导出成 GPT1 文件（含词表），
# 并用"硬件精确内核"（组内整数累加 + 组外 FP16 scale + 逐行 int8 激活）跑一遍，
# 写出参考 logits 供 C 端比对。
#
# 用法：
#   python3 train-nanoGPT-export.py                    # W1A8 量化训练 + 导出（默认）
#   python3 train-nanoGPT-export.py --fp               # 同结构的全精度基线，用于对比
import argparse
import math
import os
import random
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128  # 加速器固定组大小（soft-bnn-accelerator.c: SOFT_GROUP_SIZE）
RMS_EPS = 1e-5  # RMSNorm 的 eps，写进导出文件，C 端必须一致
HW_MODE = False  # True 时前向完全走硬件精确内核（组 scale 也要过一遍 FP16）

# ---------------------------------------------------------------------------
# 1. 语料与分词：单汉字字符级
# ---------------------------------------------------------------------------
SEASON = ["春", "夏", "秋", "冬"]
SCENE = ["风", "雨", "花", "叶", "雪", "月", "云", "山", "水", "树"]
PLACE = ["山", "水", "田", "园", "村", "城", "江", "林"]
VERB = ["来", "去", "开", "落", "飞", "生", "长", "流"]
ADJ = ["红", "绿", "白", "青", "黄"]

# 语料由模板组合出来：结构强、规律明显，小模型在 CPU 上几分钟就能学到东西
TEMPLATES = [
    "{season}天到了，{place}里的{scene}{verb}了。",
    "{place}上的{scene}是{adj}色的。",
    "{season}风一吹，{scene}就{verb}了。",
    "{scene}{verb}了，{adj}色的{scene}落满了{place}。",
    "我爱{season}天的{place}，也爱{place}里的{scene}。",
    "{season}{scene}{verb}，{place}{place}都是{adj}的。",
]

# 掺几句固定古诗，丰富一下词表，也让生成结果好看一点
POEMS = [
    "床前明月光，疑是地上霜。",
    "举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。",
    "夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。",
    "欲穷千里目，更上一层楼。",
    "红豆生南国，春来发几技。",
    "空山新雨后，天气晚来秋。",
    "明月松间照，清泉石上流。",
    "野火烧不尽，春风吹又生。",
]


def build_corpus(n_per_template=300, n_poem_repeat=20, seed=0):
    """合成一份小型中文语料（确定性）。真实语料可用 --data 指定文本文件。"""
    rng = random.Random(seed)
    lines = []
    for t in TEMPLATES:
        for _ in range(n_per_template):
            lines.append(t.format(season=rng.choice(SEASON), place=rng.choice(PLACE),
                                  scene=rng.choice(SCENE), verb=rng.choice(VERB), adj=rng.choice(ADJ)))
    lines += POEMS * n_poem_repeat
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


class CharTokenizer:
    """单汉字分词：一个字符一个 id，无 BPE、无词表文件"""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# 2. 量化算子（W1A8）：激活逐行 int8 + 权重 sign × 组 scale，均带 STE
# ---------------------------------------------------------------------------
def quantize_int8(x):
    """逐行（最后一个维度）对称量化：xs = absmax/127，round half away from zero（对齐 C 的 roundf）"""
    amax = x.abs().amax(dim=-1, keepdim=True)
    xs = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = x / xs
    q = torch.sign(q) * torch.floor(q.abs() + 0.5)
    return q.clamp_(-127.0, 127.0), xs


def fake_quant_int8(x):
    """量化感知训练的前向：值为去量化后的 q·xs，梯度直通"""
    q, xs = quantize_int8(x)
    return x + (q * xs - x).detach()


def binarize_groups(w, group=GROUP):
    """sign × 组内 |w| 均值（对 sign 量化 L2 最优），带 STE；最后一个维度按 group 分组"""
    wg = w.reshape(*w.shape[:-1], w.shape[-1] // group, group)
    alpha = wg.abs().mean(-1, keepdim=True)
    if HW_MODE:
        alpha = alpha.to(torch.float16).to(torch.float32)  # 加速器的组 scale 是 FP16
    sign = torch.where(wg >= 0, 1.0, -1.0)  # 0 归为 +1，与导出一致
    q = (sign * alpha).reshape(w.shape)
    return w + (q - w).detach()


def hw_gemm(x, sign, scales, group=GROUP):
    """硬件精确 GEMM：x [A,M,K] 逐行 int8；sign [N,K] 或 [A,N,K]（±1）；scales [N,K/group] 或 [A,N,K/group]
    组内整数点积 -> 乘 FP16 组 scale -> 按组序 FP32 累加 -> 乘回逐行激活 scale。
    与 bnn-accelerator.h 的 GEMM 数值契约逐位一致（整数中间量，fp32 精确）。"""
    A, M, K = x.shape
    N, G = sign.shape[-2], K // group
    amax = x.abs().amax(-1, keepdim=True)
    xs = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    r = x / xs
    q = torch.clamp(torch.sign(r) * torch.floor(torch.abs(r) + 0.5), -127.0, 127.0)
    sg = sign.reshape(-1, N, G, group)
    sc = scales.reshape(-1, N, G)
    if sg.shape[0] == 1:
        sg, sc = sg.expand(A, N, G, group), sc.expand(A, N, G)
    d = torch.einsum("amgk,angk->amng", q.reshape(A, M, G, group), sg)  # 整数，fp32 精确
    return (d * sc[:, None]).sum(-1) * xs


class BinaryLinear(nn.Module):
    """W1 权重（±1 + 每 group 一个 scale）、A8 激活（逐行 int8）"""

    def __init__(self, in_f, out_f, group=GROUP, std=0.02):
        super().__init__()
        assert in_f % group == 0 and in_f % 8 == 0
        self.in_f, self.out_f, self.group = in_f, out_f, group
        self.weight = nn.Parameter(torch.empty(out_f, in_f).normal_(0.0, std))
        self.binary = True     # 权重是否走 ±1 + 组 scale
        self.act_quant = True  # 激活是否走 int8
        self.hw_exact = False  # True：用导出文件里的 sign bit + FP16 scale 走硬件精确内核
        self.hw_sign = None
        self.hw_scales = None

    def quantized(self):
        """量化后的有效权重（sign × 组 scale）"""
        return binarize_groups(self.weight.detach(), self.group)

    def forward(self, x):
        if self.hw_exact:
            return hw_gemm(x, self.hw_sign, self.hw_scales, self.group)
        w = binarize_groups(self.weight, self.group) if self.binary else self.weight
        return F.linear(fake_quant_int8(x) if self.act_quant else x, w)


def set_quant(model, binary, act):
    for m in model.modules():
        if isinstance(m, BinaryLinear):
            m.binary, m.act_quant = binary, act


def set_hw(model, on):
    """切换"硬件精确内核"（用导出文件里的 sign bit + FP16 组 scale 前向）"""
    global HW_MODE
    HW_MODE = on
    for m in model.modules():
        if isinstance(m, (BinaryLinear, CausalSelfAttention)):
            m.hw_exact = on


# ---------------------------------------------------------------------------
# 5. 导出：GPT1 文件（权重 sign bit + FP16 组 scale、FP32 embedding/RMSNorm gain、UTF-8 词表）
#    格式见 doc/gpt-file.md
# ---------------------------------------------------------------------------
def pack_w1(w, group=GROUP):
    """FP32 权重 -> (packed sign bit [out,in/8], fp16 组 scale [out,in/group])"""
    w = np.ascontiguousarray(w, dtype=np.float32)
    packed = np.packbits((w >= 0).astype(np.uint8), axis=1, bitorder="little")
    scales = np.abs(w.reshape(w.shape[0], -1, group)).mean(-1).astype("<f2")
    return packed, scales


def unpack_w1(packed, scales, group=GROUP):
    """导出文件里的内容 -> (sign [out,in] ∈ ±1 fp32, group scale [out,in/group] fp32)"""
    sign = np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32) * 2.0 - 1.0
    return sign, scales.astype(np.float32).reshape(scales.shape[0], -1)


@torch.no_grad()
def export(model, tok, path):
    """写出 GPT1 模型文件，并把模型切到"硬件精确"模式（用文件里同样的 bit / FP16 scale 前向）"""
    global HW_MODE
    cfg = model.config
    V, T, C, H, L = cfg["vocab_size"], cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    linears = []  # (module, packed_bits, fp16_scales)

    with open(path, "wb") as f:
        f.write(b"GPT1")
        f.write(struct.pack("<6I", V, T, C, H, L, GROUP))
        f.write(struct.pack("<f", RMS_EPS))
        f.write(model.tok_emb.weight.cpu().numpy().astype("<f4").tobytes())
        f.write(model.pos_emb.weight.cpu().numpy().astype("<f4").tobytes())

        def w1(lin, relu):
            packed, scales = pack_w1(lin.weight.detach())
            f.write(struct.pack("<4I", lin.in_f, lin.out_f, GROUP, relu))
            f.write(packed.tobytes())
            f.write(scales.tobytes())
            linears.append((lin, packed, scales))
            return packed.nbytes + scales.nbytes

        nbytes = 0
        for blk in model.blocks:
            f.write(blk.ln1.gain.cpu().numpy().astype("<f4").tobytes())
            nbytes += sum(w1(p, 0) for p in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj, blk.attn.out_proj))
            f.write(blk.ln2.gain.cpu().numpy().astype("<f4").tobytes())
            nbytes += w1(blk.mlp.fc, 1) + w1(blk.mlp.proj, 0)
        f.write(model.ln_f.gain.cpu().numpy().astype("<f4").tobytes())
        nbytes += w1(model.lm_head, 0)
        for i in range(V):  # 词表：长度前缀 + UTF-8 字节，供 C 端解码输出
            b = tok.itos[i].encode("utf-8")
            f.write(struct.pack("<B", len(b)))
            f.write(b)

    # 切到硬件精确内核：sign 取文件里的 bit，scale 取文件里的 FP16
    HW_MODE = True
    for lin, packed, scales in linears:
        sign, sc = unpack_w1(packed, scales)
        lin.hw_sign = torch.from_numpy(sign)
        lin.hw_scales = torch.from_numpy(sc)
    set_hw(model, True)
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.eps = RMS_EPS
    n_fp32 = sum(p.numel() * 4 for p in model.parameters())
    print(f"exported {path}: {os.path.getsize(path)} B（W1 权重 {nbytes / 1024:.1f} KB，"
          f"FP32 等价 {n_fp32 / 1024:.1f} KB）词表 {V}")
    return path


@torch.no_grad()
def write_ref_logits(model, ids, path):
    """参考 logits：C 端要用同一份 id 序列复现出同样的 logits"""
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    logits, _ = model(idx)
    out = logits[0].float().cpu().numpy()
    with open(path, "wb") as f:
        f.write(struct.pack("<2I", len(ids), out.shape[1]))
        f.write(np.asarray(ids, dtype="<i4").tobytes())
        f.write(out.astype("<f4").tobytes())
    print(f"wrote {path}: {len(ids)}×{out.shape[1]} logits")
    return out


def write_tokens(ids, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(np.asarray(ids, dtype="<i4").tobytes())
    print(f"wrote {path}: {len(ids)} tokens")


def quant_size_report(model):
    """统计所有 W1 层的落地体积：sign bit（1 bit/权重）+ FP16 组 scale"""
    bits = scales = params = 0
    for m in model.modules():
        if isinstance(m, BinaryLinear):
            bits += m.out_f * m.in_f // 8
            scales += m.out_f * (m.in_f // m.group) * 2
            params += m.weight.numel()
    total = bits + scales
    return (f"sign bit {bits / 1024:.1f} KB + fp16 scale {scales / 1024:.1f} KB = {total / 1024:.1f} KB, "
            f"FP32 等价 {params * 4 / 1024:.1f} KB（{params * 4 / total:.1f}x）")


# ---------------------------------------------------------------------------
# 3. 模型：最小 decoder-only Transformer（pre-norm + RMSNorm）
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """FP32 逐 token 归一化（softmax/RMSNorm 不量化，C 端也用 FP32 实现）"""

    def __init__(self, dim, eps=RMS_EPS):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.gain


class CausalSelfAttention(nn.Module):
    """n_head 个头，head_size = n_embd // n_head = GROUP，保证 q·k 的点积维度是 128"""

    def __init__(self, config):
        super().__init__()
        C, H = config["n_embd"], config["n_head"]
        assert C % H == 0 and C // H == GROUP, "head_size 必须等于加速器的 group_size"
        self.n_head, self.head_size = H, C // H
        self.q_proj = BinaryLinear(C, C)
        self.k_proj = BinaryLinear(C, C)
        self.v_proj = BinaryLinear(C, C)
        self.out_proj = BinaryLinear(C, C, std=config["residual_std"])
        self.attn_quant = True
        self.hw_exact = False
        self.register_buffer("mask", torch.tril(torch.ones(config["block_size"], config["block_size"]))
                             .view(1, 1, config["block_size"], config["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        hs = self.head_size
        q = self.q_proj(x).view(B, T, self.n_head, hs).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, hs).transpose(1, 2)
        if self.hw_exact and self.attn_quant:
            # 加速器视角：X = 逐行 int8 的 key，W = ±1(+FP16 组 scale) 的 query
            A = B * self.n_head
            kf, qf = k.reshape(A, T, hs), q.reshape(A, T, hs)
            alpha = qf.abs().mean(-1, keepdim=True).to(torch.float16).to(torch.float32)
            scores = hw_gemm(kf, torch.where(qf >= 0, 1.0, -1.0), alpha, hs)
            scores = scores.view(B, self.n_head, T, T).transpose(-1, -2)
        elif self.attn_quant:
            scores = torch.matmul(fake_quant_int8(k), binarize_groups(q).transpose(-1, -2)).transpose(-1, -2)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(hs)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        y = attn @ v  # O(T^2·C)，相对投影是小头，保持 FP32
        return self.out_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config["n_embd"]
        self.fc = BinaryLinear(C, 4 * C)
        self.proj = BinaryLinear(4 * C, C, std=config["residual_std"])

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        C, V = config["n_embd"], config["vocab_size"]
        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(config["block_size"], C)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f = RMSNorm(C)
        self.lm_head = BinaryLinear(C, V)
        # 残差分支（attn.out_proj / mlp.proj）按 1/sqrt(2·n_layer) 缩放初始化，保持残差流稳定
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config["block_size"] else idx[:, -self.config["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 4. 数据批与评估
# ---------------------------------------------------------------------------
def get_batch(data, block_size, batch_size, dev, gen=None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix]).to(dev)
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(dev)
    return x, y


@torch.no_grad()
def evaluate(model, batches, dev):
    """返回 (平均 CE loss, 下一汉字 top-1 准确率, top-5 准确率)；batches 固定，保证各模式可比"""
    model.eval()
    losses, top1, top5, n = [], 0, 0, 0
    for (x, y) in batches:
        logits, loss = model(x, y)
        losses.append(loss.item())
        top1 += (logits.argmax(-1) == y).sum().item()
        top5 += (logits.topk(5, -1).indices == y.unsqueeze(-1)).any(-1).sum().item()
        n += y.numel()
    model.train()
    return float(np.mean(losses)), top1 / n, top5 / n


@torch.no_grad()
def eval_attention_quant(model, batches, dev):
    """消融：注意力分数不走量化（权重/激活量化保持不变）"""
    old = [m.attn_quant for m in model.modules() if isinstance(m, CausalSelfAttention)]
    for m in model.modules():
        if isinstance(m, CausalSelfAttention):
            m.attn_quant = False
    out = evaluate(model, batches, dev)
    for m, v in zip([m for m in model.modules() if isinstance(m, CausalSelfAttention)], old):
        m.attn_quant = v
    return out


def main():
    ap = argparse.ArgumentParser(description="最小 nanoGPT（中文单汉字）W1A8 量化训练")
    ap.add_argument("--data", default=None, help="可选：外部中文文本文件；缺省用内置合成语料")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=1)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=24)
    ap.add_argument("--sample", type=int, default=3, help="结束时生成几段样例")
    ap.add_argument("--fp", action="store_true", help="训练全精度基线做对比（权重/激活都不量化，不导出）")
    ap.add_argument("--out", default="nanogpt.bnn", help="导出的 GPT1 模型文件")
    ap.add_argument("--ref-out", default="ref_logits.bin", help="导出的参考 logits（供 C 端比对）")
    ap.add_argument("--tokens-out", default="val_tokens.bin", help="导出的验证集 token 流（供 C 端评估）")
    ap.add_argument("--no-attn-quant", action="store_true", help="注意力分数保持 FP32（其余仍 W1A8）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 1)))
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    text = open(args.data, encoding="utf-8").read() if args.data else build_corpus()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    if len(data) < 4 * args.block:
        raise SystemExit(f"语料太短：{len(data)} token < 4×block_size({args.block})，请换更长的语料或减小 --block")
    n_tr = int(0.9 * len(data))
    train_data, val_data = data[:n_tr], data[n_tr:]
    print(f"corpus: {len(text)} chars, vocab {tok.vocab_size}（单汉字）, tokens {len(data)} "
          f"(train {len(train_data)} / val {len(val_data)})")

    config = {
        "vocab_size": tok.vocab_size, "block_size": args.block, "n_layer": args.n_layer,
        "n_head": args.n_head, "n_embd": args.n_embd, "residual_std": 0.02 / math.sqrt(2 * args.n_layer),
    }
    model = NanoGPT(config).to(dev)
    for m in model.modules():
        if isinstance(m, CausalSelfAttention):
            m.attn_quant = not args.no_attn_quant and not args.fp
    if args.fp:
        set_quant(model, binary=False, act=False)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"model: {args.n_layer} layer / {args.n_head} head / embd {args.n_embd} / block {args.block} "
          f"| params {n_param / 1e3:.0f}K | device {dev} | 模式 {'FP32 基线' if args.fp else 'W1A8 量化'}")
    if not args.fp:
        print(f"weights: {quant_size_report(model)}")

    # 固定评估集：三种量化模式必须看到完全相同的 batch
    g = torch.Generator().manual_seed(1234)
    eval_batches = [get_batch(val_data, args.block, args.batch, dev, g) for _ in range(args.eval_batches)]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (
        (it + 1) / 100 if it < 100 else 0.5 * (1 + math.cos(math.pi * (it - 100) / max(1, args.iters - 100)))))

    t0 = time.time()
    for it in range(args.iters):
        x, y = get_batch(train_data, args.block, args.batch, dev)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % args.eval_interval == 0 or it == 0:
            vl, a1, a5 = evaluate(model, eval_batches, dev)
            print(f"iter {it + 1:5d}: train loss {loss.item():.4f} | val loss {vl:.4f} "
                  f"top1 {a1:.4f} top5 {a5:.4f} | {(time.time() - t0):.0f}s")
    print(f"训练用时 {time.time() - t0:.0f}s")

    # ---------------- 量化训练结果评估 ----------------
    print("\n=== 评估（固定 val batch，%d×%d×%d token） ===" % (args.eval_batches, args.batch, args.block))
    rows = []
    if args.fp:
        rows.append(("FP32 基线（权重/激活全精度）", evaluate(model, eval_batches, dev)))
    else:
        rows.append(("W1A8（STE 前向，训练态）", evaluate(model, eval_batches, dev)))
        export(model, tok, args.out)  # 导出后模型切到硬件精确内核（sign bit + FP16 组 scale）
        rows.append(("导出精度（硬件精确内核，C 端要复现的）", evaluate(model, eval_batches, dev)))
        ref_ids = val_data[:args.block].tolist()
        write_ref_logits(model, ref_ids, args.ref_out)
        write_tokens(val_data.tolist(), args.tokens_out)
        set_hw(model, False)  # 退回 latent FP 权重做参考
        set_quant(model, binary=False, act=False)
        rows.append(("同模型 latent FP 权重 + FP32 激活（参考）", evaluate(model, eval_batches, dev)))
        set_quant(model, binary=True, act=True)
        set_hw(model, True)
        if not args.no_attn_quant:
            rows.append(("消融：注意力分数用 FP32（其余仍 W1A8）", eval_attention_quant(model, eval_batches, dev)))
    print(f"{'模式':<40}{'val loss':>10}{'top-1':>10}{'top-5':>10}{'ppl':>10}")
    for name, (l, a1, a5) in rows:
        print(f"{name:<40}{l:>10.4f}{a1:>10.4f}{a5:>10.4f}{math.exp(min(l, 20)):>10.2f}")
    print("注：公平的全精度基线需要不带 --fp / 带 --fp 各跑一次再对比；"
          "latent FP 权重那一行只是说明 STE 训练出来的浮点权重本身不能直接用。")

    # ---------------- 生成样例 ----------------
    if args.sample:
        print("\n=== 生成样例（prompt 为语料中常见开头；贪婪那条可与 C 端逐字比对） ===")
        for prompt in ["春天", "床前明月", "山上的花"]:
            ids = tok.encode(prompt)
            if len(ids) < 1:
                continue
            idx = torch.tensor([ids], dtype=torch.long, device=dev)
            out = model.generate(idx, max_new_tokens=40, temperature=1.0, top_k=1)
            print(f"  [贪婪] {prompt} -> {tok.decode(out[0].tolist())}")
            out = model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=10)
            print(f"  [采样] {prompt} -> {tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
