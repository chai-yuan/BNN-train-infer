#!/usr/bin/env python3
"""Test Llama-2 W1A8 end-to-end: export -> bnn-infer -> C vs Python reference check.

Run directly, or use the category Makefile:
    make -C tests/llama test
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run(cmd, **kwargs):
    print(f"+ {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)
    if res.returncode != 0:
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        sys.exit(f"Command failed with exit code {res.returncode}")
    return res

def main():
    py = sys.executable
    infer_bin = ROOT / "bnn-infer"
    export_py = ROOT / "train-llama2-export.py"
    if not infer_bin.exists():
        sys.exit(f"Error: {infer_bin} not found. Run make first.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model_path = tmp / "llama2.bnn"
        ref_path = tmp / "ref.bin"
        tokens_path = tmp / "tokens.bin"

        # 1. 导出模型与参考数据
        run([py, str(export_py), "--synthetic",
             "--out", str(model_path),
             "--ref-out", str(ref_path),
             "--tokens-out", str(tokens_path)], cwd=str(ROOT))

        # 2. C 推理引擎 vs Python 参考 logits 比对
        res_ref = run([str(infer_bin), "--profile", "--ref", str(ref_path), str(model_path)], cwd=str(ROOT))
        assert "OK" in res_ref.stdout, f"Logits comparison failed:\n{res_ref.stdout}"
        print("  -> C vs Python logits check: PASSED")

        # 3. 验证集 token 评测
        res_eval = run([str(infer_bin), "--profile", "--eval", str(tokens_path), str(model_path)], cwd=str(ROOT))
        assert "C token eval" in res_eval.stdout, f"Evaluation failed:\n{res_eval.stdout}"
        print("  -> C token evaluation: PASSED")

        # 4. 文本生成 (Full context recompute)
        res_gen = run([str(infer_bin), "--profile", "--prompt", "春天", "--steps", "15", str(model_path)], cwd=str(ROOT))
        assert "generate: 春天" in res_gen.stdout, f"Generation failed:\n{res_gen.stdout}"
        print("  -> C text generation (standard): PASSED")

        # 5. KV Cache 文本生成与输出一致性对比
        res_kv = run([str(infer_bin), "--profile", "--kv-cache", "--prompt", "春天", "--steps", "15", str(model_path)], cwd=str(ROOT))
        assert "generate: 春天" in res_kv.stdout, f"KV generation failed:\n{res_kv.stdout}"
        print("  -> C text generation (KV Cache): PASSED")

        gen_standard = [line for line in res_gen.stdout.splitlines() if line.startswith("generate: ")][0]
        gen_kv = [line for line in res_kv.stdout.splitlines() if line.startswith("generate: ")][0]
        assert gen_standard == gen_kv, f"Mismatch between standard and KV Cache generation!\nStandard: {gen_standard}\nKV Cache: {gen_kv}"
        print(f"  -> Numerical equivalence: PASSED ({gen_kv})")

    print("\nAll Llama-2 tests passed successfully!")

if __name__ == "__main__":
    main()
