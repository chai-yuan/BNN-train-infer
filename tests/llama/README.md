# Llama 端到端测试

## 测试内容

`test_llama.py` 调用 `train-llama2-export.py --synthetic` 生成 Llama-2 合成模型、reference logits 和 tokens，然后验证：

- C 推理结果与 Python reference logits 一致（`bnn-infer --ref`）。
- 验证集 token 评测可以运行（`--eval`）。
- 标准全上下文生成可以运行。
- `--kv-cache` 生成与标准生成的输出完全一致。
- 从导出到 C 推理的完整链路。

测试使用合成权重，不需要训练数据或联网。

## 运行

```bash
make                         # 默认 ACCE=soft
make test
make ACCE=verilator test
```

直接运行前先构建对应后端，例如 `make ACCE=soft`，然后：

```bash
python3 tests/llama/test_llama.py
```

Makefile 会先构建所选后端。测试脚本通过 `sys.executable` 调用导出器，因此可以用 `PYTHON` 指定虚拟环境：

```bash
make PYTHON=/absolute/path/to/.venv/bin/python3
```

## 依赖

- Python 3、NumPy。
- 测试走 `--synthetic` 路径，不需要 PyTorch 或真实训练数据。
- `ACCE=verilator` 还需要 Verilator 和 C++17 编译器。
- 模型、reference 和 tokens 都放在临时目录，测试结束自动清理。
