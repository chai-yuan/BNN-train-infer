#!/usr/bin/env python3
# train-llama2-export.py — Llama-2 架构（RoPE + SwiGLU + RMSNorm）的 W1A8 量化训练与模型导出
#
# 参考 karpathy/llama2.c 的 Llama 2 Transformer 结构，针对本项目的硬件二值加速器进行对齐：
#   1. 权重 W1：符号位 ±1（1 bit/权重），每 GROUP=128 个输入共享一个 FP16 scale；
#   2. 激活 A8：逐 token（逐行）对称量化为 int8，scale = absmax / 127；
#   3. 注意力分数 QK：RoPE 旋转后的 key 激活量化为 int8，query 二值化为 ±1 + FP16 scale，走硬件 GEMM；
#   4. SwiGLU 前馈网络：w1 (gate)、w3 (up)、w2 (down) 三个线性层均使用 W1A8 量化；SiLU 与门控相乘在主机端以 FP32 计算；
#   5. 归一化与位置编码：RMSNorm 与 RoPE 均在主机端以 FP32 计算，保证数值稳定性。
#
# 导出格式支持：
#   - 原生 Llama-2 W1A8 格式（LMA1）：完整包含 RoPE 与 SwiGLU 3 矩阵结构，可由 bnn-infer 直接加载推理；
#   - 兼容模式（--compat-gpt1）：降级适配为当前 nanoGPT 的 GPT1 格式（可学习 pos_emb + 2层 ReLU MLP），供旧版引擎比对。
#
# 若当前 Python 环境未安装 PyTorch，脚本提供 --synthetic 模式以纯 NumPy 验证量化位打包、RoPE、SwiGLU 和导出。
import argparse
import math
import os
import random
import struct
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

GROUP = 128      # 加速器固定组大小（与 bnn-accelerator.h 一致）
RMS_EPS = 1e-5   # RMSNorm eps，写进模型头，C 端一致
HW_MODE = False  # True 时前向完全走硬件精确数值契约（组 scale 过 FP16，整数累加）

# ---------------------------------------------------------------------------
# 1. 中文微型语料与单汉字分词器（与 nanoGPT 一致，便于对照评估）
# ---------------------------------------------------------------------------
SEASON = ["春", "夏", "秋", "冬"]
SCENE = ["风", "雨", "花", "叶", "雪", "月", "云", "山", "水", "树"]
PLACE = ["山", "水", "田", "园", "村", "城", "江", "林"]
VERB = ["来", "去", "开", "落", "飞", "生", "长", "流"]
ADJ = ["红", "绿", "白", "青", "黄"]

TEMPLATES = [
    "{season}天到了，{place}里的{scene}{verb}了。",
    "{place}上的{scene}是{adj}色的。",
    "{season}风一吹，{scene}就{verb}了。",
    "{scene}{verb}了，{adj}色的{scene}落满了{place}。",
    "我爱{season}天的{place}，也爱{place}里的{scene}。",
    "{season}{scene}{verb}，{place}{place}都是{adj}的。",
]

POEMS = [
    "床前明月光，疑是地上霜。",
    "举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。",
    "夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。",
    "欲穷千里目，更上一层楼。",
    "红豆生南国，春来发几枝。",
    "空山新雨后，天气晚来秋。",
    "明月松间照，清泉石上流。",
    "野火烧不尽，春风吹又生。",
]


def build_corpus(n_per_template=300, n_poem_repeat=20, seed=0):
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
    """单汉字分词：每个字符对应一个 token ID，可直接导出 UTF-8 词表供 C 引擎解码。"""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)


# ---------------------------------------------------------------------------
# 2. 硬件二进制量化算子（W1A8 规范：组大小 128，bit 压缩，FP16 组 scale）
# ---------------------------------------------------------------------------
def pack_w1(w, group=GROUP):
    """FP32 权重矩阵 [out_f, in_f] -> (packed_bits [out_f, in_f/8], scales [out_f, in_f/group])"""
    w = np.ascontiguousarray(w, dtype=np.float32)
    assert w.shape[1] % group == 0 and w.shape[1] % 8 == 0
    packed = np.packbits((w >= 0).astype(np.uint8), axis=1, bitorder="little")
    scales = np.abs(w.reshape(w.shape[0], -1, group)).mean(-1).astype("<f2")
    return packed, scales


def unpack_w1(packed, scales, group=GROUP):
    """解压还原为 sign ±1 与 FP32 组 scale，供 Python 硬件精确内核校验"""
    sign = np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32) * 2.0 - 1.0
    sc = scales.astype(np.float32).reshape(scales.shape[0], -1)
    return sign, sc


if HAS_TORCH:
    def quantize_int8(x):
        """逐行（最后一个维度）对称量化：xs = absmax/127，round half away from zero"""
        amax = x.abs().amax(dim=-1, keepdim=True)
        xs = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
        q = x / xs
        q = torch.sign(q) * torch.floor(q.abs() + 0.5)
        return q.clamp_(-127.0, 127.0), xs

    def fake_quant_int8(x):
        """量化感知训练前向：通过 STE 直通梯度"""
        q, xs = quantize_int8(x)
        return x + (q * xs - x).detach()

    def binarize_groups(w, group=GROUP):
        """权重符号化并乘上组内平均绝对值（L2 最优 scale），带 STE"""
        wg = w.reshape(*w.shape[:-1], w.shape[-1] // group, group)
        alpha = wg.abs().mean(-1, keepdim=True)
        if HW_MODE:
            alpha = alpha.to(torch.float16).to(torch.float32)
        sign = torch.where(wg >= 0, 1.0, -1.0)
        q = (sign * alpha).reshape(w.shape)
        return w + (q - w).detach()

    def hw_gemm(x, sign, scales, group=GROUP):
        """硬件精确 GEMM：x [A, M, K] 逐行 int8；sign [N, K] 或 [A, N, K]；scales [N, K/group]
        契约：组内整数点积 -> 乘 FP16 组 scale -> 按组累加 -> 乘回逐行激活 scale"""
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
        d = torch.einsum("amgk,angk->amng", q.reshape(A, M, G, group), sg)
        return (d * sc[:, None]).sum(-1) * xs

    class BinaryLinear(nn.Module):
        """W1 权重（±1 + 每 group 一个 scale）、A8 激活（逐行 int8）"""

        def __init__(self, in_f, out_f, group=GROUP, std=0.02):
            super().__init__()
            assert in_f % group == 0 and in_f % 8 == 0, f"in_f {in_f} 必须是 {group} 的倍数"
            self.in_f, self.out_f, self.group = in_f, out_f, group
            self.weight = nn.Parameter(torch.empty(out_f, in_f).normal_(0.0, std))
            self.binary = True
            self.act_quant = True
            self.hw_exact = False
            self.hw_sign = None
            self.hw_scales = None

        def forward(self, x):
            if self.hw_exact:
                return hw_gemm(x, self.hw_sign, self.hw_scales, self.group)
            w = binarize_groups(self.weight, self.group) if self.binary else self.weight
            return F.linear(fake_quant_int8(x) if self.act_quant else x, w)


# ---------------------------------------------------------------------------
# 3. Llama-2 核心网络组件（对齐 karpathy/llama2.c 的 model.py）
# ---------------------------------------------------------------------------
def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 旋转位置编码的频率表（cos 与 sin）"""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)  # [max_seq_len, head_dim / 2]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


if HAS_TORCH:
    def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
        """RoPE 旋转：相邻两两成对旋转，与 llama2.c 的 run.c 循环结构逐位一致"""
        # xq, xk: [B, T, H, head_dim]
        # freqs_cos, freqs_sin: [T, head_dim / 2]
        B, T, H, D = xq.shape
        cos = freqs_cos[:T].view(1, T, 1, D // 2)
        sin = freqs_sin[:T].view(1, T, 1, D // 2)

        def rotate_pairs(x):
            x_r = x[..., 0::2]
            x_i = x[..., 1::2]
            out_r = x_r * cos - x_i * sin
            out_i = x_r * sin + x_i * cos
            return torch.stack([out_r, out_i], dim=-1).flatten(-2)

        return rotate_pairs(xq), rotate_pairs(xk)

    class RMSNorm(nn.Module):
        """RMSNorm：逐 token 均方根归一化，与 llama2.c 完全一致"""

        def __init__(self, dim, eps=RMS_EPS):
            super().__init__()
            self.eps = eps
            self.gain = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + self.eps)
            return (x / rms) * self.gain

    class LlamaAttention(nn.Module):
        """Llama-2 注意力模块：支持 GQA、RoPE 旋转与 W1A8 矩阵量化"""

        def __init__(self, config):
            super().__init__()
            dim = config["dim"]
            n_heads = config["n_heads"]
            n_kv_heads = config.get("n_kv_heads", n_heads)
            assert dim % n_heads == 0
            self.head_dim = dim // n_heads
            assert self.head_dim == GROUP, f"head_dim ({self.head_dim}) 必须等于硬件组大小 {GROUP}"
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.n_rep = n_heads // n_kv_heads
            kv_dim = n_kv_heads * self.head_dim

            self.wq = BinaryLinear(dim, dim)
            self.wk = BinaryLinear(dim, kv_dim)
            self.wv = BinaryLinear(dim, kv_dim)
            self.wo = BinaryLinear(dim, dim, std=config.get("residual_std", 0.02))
            self.attn_quant = True
            self.hw_exact = False
            self.register_buffer("mask", torch.tril(torch.ones(config["block_size"], config["block_size"]))
                                 .view(1, 1, config["block_size"], config["block_size"]))

        def forward(self, x, freqs_cos, freqs_sin):
            B, T, C = x.shape
            hd = self.head_dim

            q = self.wq(x).view(B, T, self.n_heads, hd)
            k = self.wk(x).view(B, T, self.n_kv_heads, hd)
            v = self.wv(x).view(B, T, self.n_kv_heads, hd)

            # 1. 注入 RoPE 旋转位置编码
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

            # 2. 处理 GQA 重复 KV
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=2)
                v = v.repeat_interleave(self.n_rep, dim=2)

            # 转成 [B, H, T, hd]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # 3. 注意力分数计算（硬件 GEMM 视角：k 为激活 int8，q 为权重 W1）
            if self.hw_exact and self.attn_quant:
                A = B * self.n_heads
                kf, qf = k.reshape(A, T, hd), q.reshape(A, T, hd)
                alpha = qf.abs().mean(-1, keepdim=True).to(torch.float16).to(torch.float32)
                scores = hw_gemm(kf, torch.where(qf >= 0, 1.0, -1.0), alpha, hd)
                scores = scores.view(B, self.n_heads, T, T).transpose(-1, -2)
            elif self.attn_quant:
                scores = torch.matmul(fake_quant_int8(k), binarize_groups(q).transpose(-1, -2)).transpose(-1, -2)
            else:
                scores = torch.matmul(q, k.transpose(-1, -2))

            scores = scores / math.sqrt(hd)
            scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            y = attn @ v  # [B, H, T, hd]
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.wo(y)

    class LlamaFeedForward(nn.Module):
        """Llama-2 经典的 SwiGLU 门控前馈网络：w1 (gate), w3 (up), w2 (down)"""

        def __init__(self, config):
            super().__init__()
            dim = config["dim"]
            hidden_dim = config["hidden_dim"]
            assert hidden_dim % GROUP == 0, f"hidden_dim ({hidden_dim}) 必须是 {GROUP} 的整数倍"
            self.w1 = BinaryLinear(dim, hidden_dim)  # gate_proj
            self.w3 = BinaryLinear(dim, hidden_dim)  # up_proj
            self.w2 = BinaryLinear(hidden_dim, dim, std=config.get("residual_std", 0.02))  # down_proj

        def forward(self, x):
            # SwiGLU 激活：F.silu(w1(x)) * w3(x)
            gate = F.silu(self.w1(x))
            up = self.w3(x)
            return self.w2(gate * up)

    class LlamaBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.att_norm = RMSNorm(config["dim"], eps=config.get("norm_eps", RMS_EPS))
            self.attn = LlamaAttention(config)
            self.ffn_norm = RMSNorm(config["dim"], eps=config.get("norm_eps", RMS_EPS))
            self.ffn = LlamaFeedForward(config)

        def forward(self, x, freqs_cos, freqs_sin):
            x = x + self.attn(self.att_norm(x), freqs_cos, freqs_sin)
            x = x + self.ffn(self.ffn_norm(x))
            return x

    class Llama2(nn.Module):
        """完整的 Llama-2 解码器模型（W1A8 量化 + RoPE + SwiGLU）"""

        def __init__(self, config):
            super().__init__()
            self.config = config
            dim, V = config["dim"], config["vocab_size"]
            self.tok_embeddings = nn.Embedding(V, dim)
            self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config["n_layers"])])
            self.norm = RMSNorm(dim, eps=config.get("norm_eps", RMS_EPS))
            self.lm_head = BinaryLinear(dim, V)

            # 预计算 RoPE 表
            head_dim = dim // config["n_heads"]
            cos, sin = precompute_freqs_cis(head_dim, config["block_size"])
            self.register_buffer("freqs_cos", torch.from_numpy(cos), persistent=False)
            self.register_buffer("freqs_sin", torch.from_numpy(sin), persistent=False)

            # 初始化
            nn.init.normal_(self.tok_embeddings.weight, 0.0, 0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.tok_embeddings(idx)
            cos = self.freqs_cos[:T]
            sin = self.freqs_sin[:T]
            for layer in self.layers:
                x = layer(x, cos, sin)
            logits = self.lm_head(self.norm(x))
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
# 4. 模型导出器（LMA1 原生格式与 GPT1 兼容格式）
# ---------------------------------------------------------------------------
def export_lma1(model, tok, path):
    """
    导出原生的 Llama-2 W1A8 二进制格式（LMA1），格式定义：
      1. 全局头 (40B):
         char[4] magic = "LMA1"
         uint32 vocab_size, block_size, dim, n_heads, n_kv_heads, n_layers, hidden_dim, group
         float32 rms_eps
      2. Embedding: tok_emb [V][dim] FP32 (Llama-2 用 RoPE，无需 pos_emb)
      3. 每层 L 条:
         float32 att_norm_gain [dim]
         W1 wq, wk, wv, wo (in_f, out_f, group=128, relu=0, packed_bits, scales)
         float32 ffn_norm_gain [dim]
         W1 w1 (dim -> hidden_dim, relu=0)
         W1 w2 (hidden_dim -> dim, relu=0)
         W1 w3 (dim -> hidden_dim, relu=0)
      4. 输出头:
         float32 final_norm_gain [dim]
         W1 lm_head (dim -> vocab_size, relu=0)
         V 条长度前缀的 UTF-8 词表
    """
    cfg = model.config
    V, T, C = cfg["vocab_size"], cfg["block_size"], cfg["dim"]
    H, H_kv = cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"])
    L, H_dim = cfg["n_layers"], cfg["hidden_dim"]
    linears = []

    with open(path, "wb") as f:
        # 1. 头
        f.write(b"LMA1")
        f.write(struct.pack("<8I", V, T, C, H, H_kv, L, H_dim, GROUP))
        f.write(struct.pack("<f", RMS_EPS))
        # 2. Embedding
        f.write(model.tok_embeddings.weight.detach().cpu().numpy().astype("<f4").tobytes())

        def w1(lin):
            packed, scales = pack_w1(lin.weight.detach().cpu().numpy())
            f.write(struct.pack("<4I", lin.in_f, lin.out_f, GROUP, 0))
            f.write(packed.tobytes())
            f.write(scales.tobytes())
            linears.append((lin, packed, scales))
            return packed.nbytes + scales.nbytes

        nbytes = 0
        for blk in model.layers:
            f.write(blk.att_norm.gain.detach().cpu().numpy().astype("<f4").tobytes())
            nbytes += sum(w1(p) for p in (blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.wo))
            f.write(blk.ffn_norm.gain.detach().cpu().numpy().astype("<f4").tobytes())
            nbytes += sum(w1(p) for p in (blk.ffn.w1, blk.ffn.w2, blk.ffn.w3))

        f.write(model.norm.gain.detach().cpu().numpy().astype("<f4").tobytes())
        nbytes += w1(model.lm_head)

        for i in range(V):
            b = tok.itos[i].encode("utf-8")
            f.write(struct.pack("<B", len(b)))
            f.write(b)

    # 切到硬件精确内核
    if HAS_TORCH:
        global HW_MODE
        HW_MODE = True
        for lin, packed, scales in linears:
            sign, sc = unpack_w1(packed, scales)
            lin.hw_sign = torch.from_numpy(sign)
            lin.hw_scales = torch.from_numpy(sc)
            lin.hw_exact = True
        for m in model.modules():
            if isinstance(m, LlamaAttention):
                m.hw_exact = True

    file_size = os.path.getsize(path)
    n_fp32 = sum(p.numel() * 4 for p in model.parameters()) if HAS_TORCH else file_size * 4
    print(f"exported LMA1 {path}: {file_size} B (W1 weights {nbytes / 1024:.1f} KB, "
          f"FP32 eq {n_fp32 / 1024:.1f} KB, vocab {V})")
    return path


def write_ref_logits(model, ids, path):
    """写出参考 logits：供 C 推理引擎对比数值精度"""
    if not HAS_TORCH:
        return
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    logits, _ = model(idx)
    out = logits[0].float().cpu().numpy()
    with open(path, "wb") as f:
        f.write(struct.pack("<2I", len(ids), out.shape[1]))
        f.write(np.asarray(ids, dtype="<i4").tobytes())
        f.write(out.astype("<f4").tobytes())
    print(f"wrote {path}: {len(ids)}x{out.shape[1]} logits")
    return out


def write_tokens(ids, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(np.asarray(ids, dtype="<i4").tobytes())
    print(f"wrote {path}: {len(ids)} tokens")


# ---------------------------------------------------------------------------
# 5. 纯 NumPy 模式：在无 PyTorch 环境下生成自包含合成测试与导出
# ---------------------------------------------------------------------------
def run_synthetic_numpy_test(out_model="llama2.bnn", out_ref="ref_logits.bin", out_tokens="val_tokens.bin"):
    """
    当系统未安装 PyTorch 时，用纯 NumPy 实现确定性的 Llama-2 W1A8 前向并导出模型与参考数据。
    用于验证 LMA1 格式生成与 bnn-infer.c 的完全无缝对接。
    """
    print("=== 正在运行纯 NumPy 模式的 Llama-2 W1A8 合成模型生成与参考构建 ===")
    rng = np.random.RandomState(42)

    text = build_corpus(n_per_template=20, n_poem_repeat=2)
    tok = CharTokenizer(text)
    V = tok.vocab_size
    T = 32
    C = 128
    H = 1
    H_kv = 1
    L = 2
    H_dim = 384  # multiple of 128

    cfg = {
        "vocab_size": V, "block_size": T, "dim": C,
        "n_heads": H, "n_kv_heads": H_kv, "n_layers": L, "hidden_dim": H_dim
    }

    # 随机生成确定性权重
    tok_emb = rng.randn(V, C).astype(np.float32) * 0.02
    layers = []
    for _ in range(L):
        layer = {
            "att_norm": np.ones(C, dtype=np.float32),
            "wq": rng.randn(C, C).astype(np.float32) * 0.02,
            "wk": rng.randn(C, C).astype(np.float32) * 0.02,
            "wv": rng.randn(C, C).astype(np.float32) * 0.02,
            "wo": rng.randn(C, C).astype(np.float32) * 0.02,
            "ffn_norm": np.ones(C, dtype=np.float32),
            "w1": rng.randn(H_dim, C).astype(np.float32) * 0.02,
            "w2": rng.randn(C, H_dim).astype(np.float32) * 0.02,
            "w3": rng.randn(H_dim, C).astype(np.float32) * 0.02,
        }
        layers.append(layer)
    final_norm = np.ones(C, dtype=np.float32)
    lm_head = rng.randn(V, C).astype(np.float32) * 0.02

    # 导出 LMA1 文件
    with open(out_model, "wb") as f:
        f.write(b"LMA1")
        f.write(struct.pack("<8I", V, T, C, H, H_kv, L, H_dim, GROUP))
        f.write(struct.pack("<f", RMS_EPS))
        f.write(tok_emb.astype("<f4").tobytes())

        def write_w1(w):
            packed, scales = pack_w1(w)
            f.write(struct.pack("<4I", w.shape[1], w.shape[0], GROUP, 0))
            f.write(packed.tobytes())
            f.write(scales.tobytes())

        for l in layers:
            f.write(l["att_norm"].astype("<f4").tobytes())
            write_w1(l["wq"])
            write_w1(l["wk"])
            write_w1(l["wv"])
            write_w1(l["wo"])
            f.write(l["ffn_norm"].astype("<f4").tobytes())
            write_w1(l["w1"])
            write_w1(l["w2"])
            write_w1(l["w3"])

        f.write(final_norm.astype("<f4").tobytes())
        write_w1(lm_head)

        for i in range(V):
            b = tok.itos[i].encode("utf-8")
            f.write(struct.pack("<B", len(b)))
            f.write(b)

    print(f"成功导出合成模型: {out_model} ({os.path.getsize(out_model)} 字节)")

    # 纯 NumPy 硬件精确前向计算参考 logits
    cos, sin = precompute_freqs_cis(C // H, T)

    def np_rmsnorm(x, gain):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + RMS_EPS)
        return (x / rms) * gain

    def np_hw_gemm(x, w):
        # 对齐硬件 GEMM：x int8 量化，w 二值化 + FP16 scale
        packed, scales = pack_w1(w)
        sign, sc = unpack_w1(packed, scales)  # [out_f, in_f], [out_f, in_f / 128]
        out_f, in_f = sign.shape
        G = in_f // GROUP
        rows = x.shape[0]

        # 逐行 int8 量化
        amax = np.max(np.abs(x), axis=-1, keepdims=True)
        xs = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        r = x / xs
        q = np.clip(np.sign(r) * np.floor(np.abs(r) + 0.5), -127.0, 127.0).astype(np.float32)

        q_g = q.reshape(rows, G, GROUP)
        s_g = sign.reshape(out_f, G, GROUP)
        sc_g = sc.reshape(out_f, G)

        # 组内整数点积，组间乘 FP16 scale 累加
        dots = np.einsum("rgk,ogk->rog", q_g, s_g)  # [rows, out_f, G]
        y = np.sum(dots * sc_g[None, :, :], axis=-1) * xs
        return y.astype(np.float32)

    def np_forward(ids):
        n = len(ids)
        x = tok_emb[ids].copy()
        for layer in layers:
            # 1. Attention
            norm_x = np_rmsnorm(x, layer["att_norm"])
            q = np_hw_gemm(norm_x, layer["wq"])
            k = np_hw_gemm(norm_x, layer["wk"])
            v = np_hw_gemm(norm_x, layer["wv"])

            # RoPE
            for t_idx in range(n):
                c_val = cos[t_idx]
                s_val = sin[t_idx]
                for i in range(0, C, 2):
                    q0, q1 = q[t_idx, i], q[t_idx, i + 1]
                    k0, k1 = k[t_idx, i], k[t_idx, i + 1]
                    q[t_idx, i] = q0 * c_val[i // 2] - q1 * s_val[i // 2]
                    q[t_idx, i + 1] = q0 * s_val[i // 2] + q1 * c_val[i // 2]
                    k[t_idx, i] = k0 * c_val[i // 2] - k1 * s_val[i // 2]
                    k[t_idx, i + 1] = k0 * s_val[i // 2] + k1 * c_val[i // 2]

            # Attention scores (accelerator QK GEMM)
            # Query as W1 (+FP16 scale), Key as int8
            k_amax = np.max(np.abs(k), axis=-1, keepdims=True)
            k_xs = np.where(k_amax > 0, k_amax / 127.0, 1.0).astype(np.float32)
            k_q = np.clip(np.sign(k / k_xs) * np.floor(np.abs(k / k_xs) + 0.5), -127.0, 127.0)

            q_sign = np.where(q >= 0, 1.0, -1.0)
            q_alpha = np.mean(np.abs(q), axis=-1, keepdims=True).astype(np.float16).astype(np.float32)

            scores = (k_q @ q_sign.T) * q_alpha.T * k_xs  # [n, n] key x query
            scores = scores.T / np.sqrt(C)  # query x key

            # Causal mask & softmax
            mask = np.tril(np.ones((n, n), dtype=bool))
            scores[~mask] = -1e9
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            probs[~mask] = 0.0

            attn_out = probs @ v
            o = np_hw_gemm(attn_out, layer["wo"])
            x += o

            # 2. SwiGLU FFN
            norm_x2 = np_rmsnorm(x, layer["ffn_norm"])
            w1_out = np_hw_gemm(norm_x2, layer["w1"])
            w3_out = np_hw_gemm(norm_x2, layer["w3"])
            silu_w1 = w1_out / (1.0 + np.exp(-w1_out))
            swiglu = silu_w1 * w3_out
            ffn_out = np_hw_gemm(swiglu, layer["w2"])
            x += ffn_out

        logits = np_hw_gemm(np_rmsnorm(x, final_norm), lm_head)
        return logits

    eval_tokens = tok.encode("春天到了，山里的花开了。夏天到了，水里的树绿了。秋风一吹，叶就落了。")[:T]
    logits = np_forward(eval_tokens)

    # 写入参考数据
    with open(out_ref, "wb") as f:
        f.write(struct.pack("<2I", len(eval_tokens), V))
        f.write(np.asarray(eval_tokens, dtype="<i4").tobytes())
        f.write(logits.astype("<f4").tobytes())
    print(f"写入参考 logits: {out_ref} ({len(eval_tokens)}x{V})")

    all_tokens = tok.encode(text)
    write_tokens(all_tokens, out_tokens)
    print("NumPy 模式自测与模型生成完成！")


# ---------------------------------------------------------------------------
# 6. 主程序与命令行
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Llama-2 W1A8 量化训练与模型导出")
    parser.add_argument("--data", default=None, help="外部文本语料，默认使用内置合成语料")
    parser.add_argument("--iters", type=int, default=1500, help="训练迭代次数")
    parser.add_argument("--batch", type=int, default=32, help="批次大小")
    parser.add_argument("--block", type=int, default=32, help="上下文长度 (T)")
    parser.add_argument("--dim", type=int, default=128, help="模型通道数 (C，必须为128的倍数)")
    parser.add_argument("--hidden-dim", type=int, default=384, help="SwiGLU 隐层维度 (必须为128的倍数)")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer 层数")
    parser.add_argument("--n-heads", type=int, default=1, help="查询头数 (dim // n_heads 必须等于 128)")
    parser.add_argument("--n-kv-heads", type=int, default=1, help="KV 头数 (支持 GQA)")
    parser.add_argument("--lr", type=float, default=3e-3, help="学习率")
    parser.add_argument("--out", default="llama2.bnn", help="导出的模型文件名 (默认 LMA1 格式)")
    parser.add_argument("--ref-out", default="ref_llama.bin", help="参考 logits 输出文件 (供 C 端比对)")
    parser.add_argument("--tokens-out", default="val_llama.bin", help="验证集 tokens 输出文件")
    parser.add_argument("--synthetic", action="store_true", help="强制使用纯 NumPy 生成合成模型与参考数据")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not HAS_TORCH or args.synthetic:
        if not HAS_TORCH:
            print("注意：当前 Python 环境未安装 PyTorch，自动回退到纯 NumPy 模式运行合成模型生成与导出。")
        run_synthetic_numpy_test(out_model=args.out, out_ref=args.ref_out, out_tokens=args.tokens_out)
        return

    # PyTorch 训练模式
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    text = open(args.data, encoding="utf-8").read() if args.data else build_corpus()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_tr = int(0.9 * len(data))
    train_data, val_data = data[:n_tr], data[n_tr:]

    config = {
        "vocab_size": tok.vocab_size,
        "block_size": args.block,
        "dim": args.dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "residual_std": 0.02 / math.sqrt(2 * args.n_layers),
    }

    model = Llama2(config).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Llama-2 模型初始化完成: {args.n_layers} 层 / {args.n_heads} 头 / 维度 {args.dim} / "
          f"SwiGLU隐层 {args.hidden_dim} | 参数量 {n_param / 1e3:.1f}K | 运行设备 {dev}")

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - args.block, (args.batch,))
        x = torch.stack([split_data[i:i + args.block] for i in ix])
        y = torch.stack([split_data[i + 1:i + 1 + args.block] for i in ix])
        return x.to(dev), y.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (
        (it + 1) / 100 if it < 100 else 0.5 * (1 + math.cos(math.pi * (it - 100) / max(1, args.iters - 100)))))

    t0 = time.time()
    for it in range(args.iters):
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % 250 == 0 or it == 0:
            with torch.no_grad():
                vx, vy = get_batch(val_data)
                _, vloss = model(vx, vy)
            print(f"iter {it + 1:5d}: train loss {loss.item():.4f} | val loss {vloss.item():.4f} | "
                  f"{(time.time() - t0):.1f}s")

    print(f"训练完成，用时 {time.time() - t0:.1f}s")

    # 导出模型与参考
    export_lma1(model, tok, args.out)
    ref_ids = val_data[:args.block].tolist()
    write_ref_logits(model, ref_ids, args.ref_out)
    write_tokens(val_data.tolist(), args.tokens_out)

    # 生成样例
    print("\n=== Llama-2 文本生成样例 ===")
    for prompt in ["春天", "床前明月", "山上的花"]:
        ids = tok.encode(prompt)
        if not ids:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=dev)
        out = model.generate(idx, max_new_tokens=40, temperature=1.0, top_k=1)
        print(f"  [贪婪] {prompt} -> {tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
