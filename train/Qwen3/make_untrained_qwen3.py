#!/usr/bin/env python3
"""Create an untrained Qwen3 model and export it as a Q1_0 GGUF.

The weights are deterministic random signs; no corpus or optimizer is involved
yet.  The resulting file exercises the full v2 Qwen3 load/tokenize/decode path.
"""

import argparse
from pathlib import Path

from qwen3_gguf import Qwen3Config, export_untrained

BASE = Path(__file__).resolve().parent
DEFAULT_OUT = BASE / "out" / "qwen3-tiny-untrained.gguf"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Qwen3Config(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        max_position_embeddings=args.ctx,
    )
    path = export_untrained(config, args.out, seed=args.seed)
    size = path.stat().st_size
    print(f"wrote {path}")
    print(
        f"params={config.parameter_count / 1e6:.2f}M "
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"ffn={config.intermediate_size} kv={config.kv_size} "
        f"vocab={config.vocab_size} ctx={config.max_position_embeddings} "
        f"file={size / 1e6:.2f}MB"
    )


if __name__ == "__main__":
    main()
