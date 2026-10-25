#!/usr/bin/env python3
"""Qwen3-shaped PyTorch model used for pretraining before Q1_0 export."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen3_gguf import Qwen3Config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def build_rope(head_dim, max_seq, base):
    half = head_dim // 2
    index = torch.arange(half, dtype=torch.float32)
    inv_freq = base ** (-2.0 * index / head_dim)
    positions = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return torch.cat(
        [first * cos - second * sin, first * sin + second * cos], dim=-1
    )


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_mul = self.heads // self.kv_heads
        self.wq = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.kv_size, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.kv_size, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, x, cos, sin):
        batch, seq, _ = x.shape
        q = self.wq(x).view(batch, seq, self.heads, self.head_dim)
        k = self.wk(x).view(batch, seq, self.kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq, self.kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.kv_mul, dim=1)
        v = v.repeat_interleave(self.kv_mul, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(y)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_gate = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.ffn_up = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.ffn_down = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        h = self.ffn_norm(x)
        x = x + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embd = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.output_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embd.weight

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embd.weight
        cos, sin = build_rope(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, tokens):
        _, seq = tokens.shape
        if seq > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        x = self.token_embd(tokens)
        cos = self.rope_cos[:seq][None, None, :, :]
        sin = self.rope_sin[:seq][None, None, :, :]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.output_norm(x))

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.max_position_embeddings :]
            logits = self(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_id is not None and bool((next_token == eos_id).all()):
                break
        return tokens

    def export_state(self):
        state = {
            "token_embd.weight": self.token_embd.weight.detach().cpu().numpy(),
            "output_norm.weight": self.output_norm.weight.detach().cpu().numpy(),
        }
        for index, block in enumerate(self.blocks):
            prefix = f"blk.{index}."
            state[prefix + "attn_norm.weight"] = (
                block.attn_norm.weight.detach().cpu().numpy()
            )
            state[prefix + "attn_q.weight"] = (
                block.attn.wq.weight.detach().cpu().numpy()
            )
            state[prefix + "attn_k.weight"] = (
                block.attn.wk.weight.detach().cpu().numpy()
            )
            state[prefix + "attn_v.weight"] = (
                block.attn.wv.weight.detach().cpu().numpy()
            )
            state[prefix + "attn_output.weight"] = (
                block.attn.wo.weight.detach().cpu().numpy()
            )
            state[prefix + "attn_q_norm.weight"] = (
                block.attn.q_norm.weight.detach().cpu().numpy()
            )
            state[prefix + "attn_k_norm.weight"] = (
                block.attn.k_norm.weight.detach().cpu().numpy()
            )
            state[prefix + "ffn_norm.weight"] = (
                block.ffn_norm.weight.detach().cpu().numpy()
            )
            state[prefix + "ffn_gate.weight"] = (
                block.ffn_gate.weight.detach().cpu().numpy()
            )
            state[prefix + "ffn_up.weight"] = (
                block.ffn_up.weight.detach().cpu().numpy()
            )
            state[prefix + "ffn_down.weight"] = (
                block.ffn_down.weight.detach().cpu().numpy()
            )
        return state

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
