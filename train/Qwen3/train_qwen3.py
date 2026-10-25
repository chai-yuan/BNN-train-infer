#!/usr/bin/env python3
"""Train a small Chinese Qwen3 model on CPU and export a Q1_0 GGUF.

The model is trained in FP32.  Q1_0 packing happens only at export, matching
the "train float, deploy binary" path used by train/MNIST-MLP.
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from qwen3_gguf import Qwen3Config, export_trained, load_tokenizer_json
from qwen3_torch import Qwen3Model

BASE = Path(__file__).resolve().parent
PRESETS = {
    # hidden, layers, ffn, heads, kv_heads
    "tiny": (512, 8, 1536, 4, 2),
    "small": (512, 12, 1536, 4, 2),
    "medium": (768, 8, 2048, 6, 3),
    "large": (768, 12, 2048, 6, 3),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=BASE / "out" / "zh-run")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr-period", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sample-prompt", default="从前，有一个")
    parser.add_argument("--sample-tokens", type=int, default=50)
    return parser.parse_args()


def build_config(args, vocab_size):
    hidden, layers, ffn, heads, kv_heads = PRESETS[args.preset]
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=ffn,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=128,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )


def get_batch(data, batch_size, block_size, generator):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts.tolist()]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in starts.tolist()
        ]
    )
    return x, y


def learning_rate(step, args, period):
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, period - args.warmup)
    progress = min(1.0, progress)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def make_optimizer(model, args):
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def estimate_loss(model, data, args):
    model.eval()
    generator = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.batch, args.seq, generator)
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, model.config.vocab_size), y.view(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, step, args, config, lr_period):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(config),
            "preset": args.preset,
            "data_dir": str(args.data_dir),
            "tokenizer": str(args.tokenizer),
            "lr_period": lr_period,
        },
        path,
    )


def export_model(model, config, tokenizer_path, out_path, preset):
    tokenizer = load_tokenizer_json(tokenizer_path)
    if len(tokenizer.tokens) != config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes differ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_trained(
        config,
        out_path,
        model.export_state(),
        tokenizer,
        name=f"qwen3-zh-{preset}",
    )
    print(f"exported {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)", flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.tokenizer is None:
        args.tokenizer = args.data_dir / "tokenizer-zh-8192.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.tokenizer.is_file():
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        vocab_size = tokenizer.get_vocab_size()
        eos_id = tokenizer.token_to_id("<|endoftext|>")
    else:
        if not args.benchmark:
            raise SystemExit(f"tokenizer not found: {args.tokenizer}")
        tokenizer = None
        vocab_size = 8192
        eos_id = 0
    config = build_config(args, vocab_size)
    model = Qwen3Model(config)
    optimizer = make_optimizer(model, args)
    print(
        f"preset={args.preset} params={model.parameter_count() / 1e6:.2f}M "
        f"vocab={vocab_size} layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} ffn={config.intermediate_size} "
        f"seq={args.seq} batch={args.batch}",
        flush=True,
    )
    print(f"tokenizer={args.tokenizer} eos={eos_id}", flush=True)

    if args.benchmark:
        model.train()
        generator = torch.Generator().manual_seed(args.seed)
        for _ in range(2):
            x, y = get_batch_tokens(vocab_size, args, generator)
            logits = model(x)
            F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        start = time.time()
        tokens = 0
        for _ in range(3):
            x, y = get_batch_tokens(vocab_size, args, generator)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tokens += x.numel()
        elapsed = time.time() - start
        print(
            f"benchmark: {tokens / elapsed:.1f} tokens/s, "
            f"{elapsed / 3:.2f} s/step, {tokens / 3 / args.batch:.0f} tokens/sample",
            flush=True,
        )
        return

    start_step = 0
    lr_period = args.lr_period or args.steps
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        saved_config = checkpoint.get("config")
        if saved_config != dataclasses.asdict(config):
            raise SystemExit("checkpoint config does not match the selected preset")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        if not args.lr_period:
            lr_period = checkpoint.get("lr_period", lr_period)
        print(f"resumed {args.resume} at step {start_step}", flush=True)
    print(f"lr period: {lr_period} steps", flush=True)

    def run_export():
        if args.no_export:
            return
        out_path = args.export or (args.out_dir / "qwen3-zh.gguf")
        export_model(model, config, args.tokenizer, out_path, args.preset)

    if args.export_only:
        if not args.resume:
            raise SystemExit("--export-only requires --resume")
        run_export()
        return

    train_data = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")
    if len(train_data) <= args.seq + 2 or len(val_data) <= args.seq + 2:
        raise SystemExit("data is too short for the requested sequence length")
    print(
        f"data: train={len(train_data)} tokens, val={len(val_data)} tokens",
        flush=True,
    )

    if start_step >= args.steps:
        run_export()
        return

    generator = torch.Generator().manual_seed(args.seed + start_step)
    model.train()
    start_time = time.time()
    last_log_time = start_time
    last_log_step = start_step - 1
    for step in range(start_step, args.steps):
        lr = learning_rate(step, args, lr_period)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch, args.seq, generator)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        (loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            now = time.time()
            steps_done = max(1, step - last_log_step)
            rate = (args.batch * args.seq * steps_done) / max(now - last_log_time, 1e-9)
            print(
                f"step {step} loss {loss.item():.4f} lr {lr:.2e} "
                f"tokens/s {rate:.0f} elapsed {now - start_time:.0f}s",
                flush=True,
            )
            last_log_time = now
            last_log_step = step
        if step > start_step and step % args.eval_every == 0:
            train_loss = estimate_loss(model, train_data, args)
            val_loss = estimate_loss(model, val_data, args)
            print(
                f"eval step {step}: train {train_loss:.4f} val {val_loss:.4f}",
                flush=True,
            )
            prompt_ids = tokenizer.encode(args.sample_prompt).ids
            tokens = torch.tensor([prompt_ids], dtype=torch.long)
            generated = model.generate(tokens, args.sample_tokens, eos_id)
            sample = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            print(f"sample: {sample!r}", flush=True)
            model.train()
        if step > start_step and step % args.save_every == 0:
            save_checkpoint(
                args.out_dir / "latest.pt",
                model,
                optimizer,
                step,
                args,
                config,
                lr_period,
            )
            print(f"saved {args.out_dir / 'latest.pt'}", flush=True)

    save_checkpoint(
        args.out_dir / "final.pt",
        model,
        optimizer,
        args.steps - 1,
        args,
        config,
        lr_period,
    )
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "config": dataclasses.asdict(config),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved {args.out_dir / 'final.pt'}", flush=True)
    run_export()


def get_batch_tokens(vocab_size, args, generator):
    x = torch.randint(0, vocab_size, (args.batch, args.seq), generator=generator)
    y = torch.randint(0, vocab_size, (args.batch, args.seq), generator=generator)
    return x, y


if __name__ == "__main__":
    main()
