#!/usr/bin/env python3
"""Tokenizer differential test against llama.cpp's llama-tokenize.

The C tokenizer is Qwen2 byte-level BPE with llama.cpp's custom pre-split
implementation.  This test compares token IDs for representative ASCII,
CJK, punctuation, whitespace, code, emoji and Unicode edge cases.
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ID_PATTERN = re.compile(r"^\s*(\d+)\s*->")


PROMPTS = [
    "The capital of France is",
    "春天的首都是",
    "Hello, world! It's 2026-10-24.",
    "  multiple   spaces\n\nand\ttabs  ",
    "def f(x):\n    return x**2\n",
    "emoji 😀🚀 and café naïve",
    "你好，世界！",
    "Qwen3-1.7B: Q1_0 (1 bit) + W1A8.",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "1234567890",
    "a\n\nb\r\nc",
    "don't we're I'll he'd",
    "  leading and trailing  ",
    "中文mixedEnglish123混合",
    "<?xml version='1.0'?><tag attr=\"v\">text</tag>",
    "   ",
    "\n\n\n",
    "\t\t",
    "…—–“”‘’",
    "e\u0301",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢",
    "<|im_end|>",
    "a<|im_start|>b<|endoftext|>",
    "<|vision_start|><|image_pad|><|vision_end|>",
]


def find_llama_tokenize(explicit):
    candidates = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    candidates.append(ROOT.parent / "llama.cpp" / "build" / "bin" / "llama-tokenize")
    found = shutil.which("llama-tokenize")
    if found:
        candidates.append(pathlib.Path(found))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_ours(binary, model, prompt):
    completed = subprocess.run(
        [str(binary), "--tokenize-only", "--prompt", prompt, str(model)],
        cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(f"tokenize-only failed for {prompt!r}:\n{completed.stderr}")
    for line in completed.stdout.splitlines():
        match = re.match(r"^token ids: \d+: ?(.*)$", line)
        if match:
            body = match.group(1).strip()
            return [int(value) for value in body.split()] if body else []
    raise SystemExit(f"no token ids in output for {prompt!r}:\n{completed.stdout}")


def run_llama(tokenize, model, prompt):
    completed = subprocess.run(
        [str(tokenize), "-m", str(model), "-p", prompt], capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(f"llama-tokenize failed for {prompt!r}")
    text = completed.stdout.decode("utf-8", "replace")
    return [int(match.group(1)) for line in text.splitlines()
            if (match := ID_PATTERN.match(line))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("BONSAI_GGUF"))
    parser.add_argument("--binary", default=str(ROOT / "bnn-infer"))
    parser.add_argument("--llama-tokenize")
    args = parser.parse_args()

    if not args.model:
        print("SKIP: set BONSAI_GGUF or pass --model", file=sys.stderr)
        return 0
    if not os.path.exists(args.model):
        print(f"SKIP: model not found: {args.model}", file=sys.stderr)
        return 0
    if not os.path.exists(args.binary):
        raise SystemExit(f"missing {args.binary}; run 'make ACCE=soft' first")
    tokenize = find_llama_tokenize(args.llama_tokenize)
    if not tokenize:
        print("SKIP: llama-tokenize not found; pass --llama-tokenize", file=sys.stderr)
        return 0

    failures = 0
    for prompt in PROMPTS:
        ours = run_ours(args.binary, args.model, prompt)
        reference = run_llama(tokenize, args.model, prompt)
        if ours != reference:
            failures += 1
            print(f"MISMATCH {prompt!r}")
            print(f"  c     : {ours}")
            print(f"  llama : {reference}")
    if failures:
        raise SystemExit(f"{failures}/{len(PROMPTS)} tokenizer prompts mismatched")

    # Chat mode must inject control tokens, not encode their literal text.
    chat = subprocess.run(
        [args.binary, "--tokenize-only", "--chat", "--prompt", "hello", args.model],
        cwd=ROOT, text=True, capture_output=True)
    if chat.returncode != 0:
        raise SystemExit(f"chat tokenization failed:\n{chat.stderr}")
    match = re.search(r"^token ids: \d+: ?(.*)$", chat.stdout, re.MULTILINE)
    if not match or not any(int(value) >= 151643 for value in match.group(1).split()):
        raise SystemExit("chat tokenization did not emit special-token IDs")

    print(f"tokenizer differential test passed: {len(PROMPTS)} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
