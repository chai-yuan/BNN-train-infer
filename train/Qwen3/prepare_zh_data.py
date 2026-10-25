#!/usr/bin/env python3
"""Prepare a Chinese TinyStories subset and an 8k byte-level BPE tokenizer.

The corpus is streamed from Hugging Face, so the full 2.6GB archive is never
written to disk.  Only ``story_zh`` is kept.  The tokenizer uses the same
Qwen2 pre-tokenization split as bnn-sw-v2/tokenizer/tokenizer-bpe.c, then
ByteLevel BPE, so the C tokenizer can reproduce the IDs.
"""

import argparse
import json
import random
import tarfile
import urllib.request
from array import array
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

BASE = Path(__file__).resolve().parent
URL = (
    "https://huggingface.co/datasets/adam89/TinyStoriesChinese/resolve/main/"
    "TinyStories_all_data_zh.tar.gz"
)
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
]
QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def stream_stories(limit):
    request = urllib.request.Request(URL, headers={"User-Agent": "bnn-qwen3-data/1.0"})
    stories = []
    with urllib.request.urlopen(request, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                file = archive.extractfile(member)
                if file is None:
                    continue
                for line in file:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = clean(record.get("story_zh") or record.get("story") or "")
                    if len(text) < 30:
                        continue
                    stories.append(text)
                    if len(stories) % 5000 == 0:
                        print(f"streamed {len(stories)} stories", flush=True)
                    if len(stories) >= limit:
                        return stories
    return stories


def build_tokenizer(train_texts, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(QWEN2_SPLIT), behavior="isolated", invert=False
            ),
            byte_level,
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer, length=len(train_texts))
    return tokenizer


def encode_stories(tokenizer, stories):
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise RuntimeError("tokenizer is missing <|endoftext|>")
    ids = array("H")
    for index, text in enumerate(stories):
        ids.extend(tokenizer.encode(text).ids)
        ids.append(eos_id)
        if (index + 1) % 50000 == 0:
            print(f"tokenized {index + 1} stories, {len(ids)} tokens", flush=True)
    return np.asarray(ids, dtype="<u2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=100000)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-sample", type=int, default=50000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--out-dir", type=Path, default=BASE / "data")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"streaming up to {args.stories} stories from {URL}", flush=True)
    stories = stream_stories(args.stories)
    if len(stories) < 1000:
        raise RuntimeError(f"only received {len(stories)} stories")
    print(f"received {len(stories)} stories, {sum(map(len, stories)) / 1e6:.1f}M chars")

    random.Random(args.seed).shuffle(stories)
    val_count = max(50, int(len(stories) * args.val_ratio))
    val_texts = stories[:val_count]
    train_texts = stories[val_count:]

    tokenizer_texts = train_texts
    if args.tokenizer_sample and len(train_texts) > args.tokenizer_sample:
        tokenizer_texts = random.Random(args.seed).sample(train_texts, args.tokenizer_sample)
    print(
        f"training BPE vocab={args.vocab_size} on {len(tokenizer_texts)} stories",
        flush=True,
    )
    tokenizer = build_tokenizer(tokenizer_texts, args.vocab_size)
    tokenizer_path = args.out_dir / f"tokenizer-zh-{args.vocab_size}.json"
    tokenizer_path.write_text(tokenizer.to_str(), encoding="utf-8")

    train_ids = encode_stories(tokenizer, train_texts)
    val_ids = encode_stories(tokenizer, val_texts)
    train_ids.tofile(args.out_dir / "train.bin")
    val_ids.tofile(args.out_dir / "val.bin")

    sample = "从前，有一个小女孩叫莉莉。"
    sample_ids = tokenizer.encode(sample).ids
    print(f"tokenizer: {tokenizer_path}")
    print(f"vocab={tokenizer.get_vocab_size()} eos={tokenizer.token_to_id('<|endoftext|>')}")
    print(f"train: {len(train_ids)} tokens, val: {len(val_ids)} tokens")
    print(f"sample: {sample!r} -> {sample_ids} -> {tokenizer.decode(sample_ids)!r}")


if __name__ == "__main__":
    main()
