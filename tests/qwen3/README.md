# Qwen3 / Bonsai Q1_0 差分测试

## 测试内容

### 1. W1A8 前向参考（`test_qwen3.py`）

使用 `tools/bonsai-ref.py` 生成与 C 后端**相同 W1A8 量化语义**的 NumPy 参考 logits，
然后调用 `bnn-infer --ref` 比较：

- 每个位置的 argmax 必须一致；
- 最大相对误差必须小于 CLI 的 `1e-2` 限值。

参考实现按 C 的运算顺序模拟：每行 absmax int8 激活量化、Q1_0 每组 FP16 scale、
分组 FP32 顺序累加、`ffn_down` 的 4096 列分块，以及 RMSNorm/attention 的顺序求和。
当前模型在短序列上可达到 `1e-7` 量级，而不是仅做“能跑”的冒烟测试。

`--generation` 额外验证 `--ids` + KV Cache 的增量生成路径。

### 2. 分词器差分（`test_tokenizer.py`）

用 `bnn-infer --tokenize-only --prompt TEXT` 与 llama.cpp 的 `llama-tokenize`
逐 token ID 比较，覆盖英文、中文、标点、契约缩写、连续空格、换行/制表符、
代码、emoji、组合字符（`e` + U+0301）、数学字母、XML 以及 `<|im_end|>` 等
特殊 token 字面量。当前 24 组测试全部一致。
同时检查 `--chat` 会注入 `<|im_start|>` / `<|im_end|>` / `<think>` 等特殊 token，
而不是把字面量当普通文本编码。

## 运行

模型文件不在仓库中，需要显式指定：

```bash
make ACCE=soft
BONSAI_GGUF=/path/to/Bonsai-1.7B-Q1_0.gguf \
    python3 tests/qwen3/test_qwen3.py --generation

# 只跑分词器差分
BONSAI_GGUF=/path/to/Bonsai-1.7B-Q1_0.gguf \
    python3 tests/qwen3/test_tokenizer.py

# 或通过分类 Makefile
make -C tests/qwen3 test BONSAI_GGUF=/path/to/Bonsai-1.7B-Q1_0.gguf
make -C tests/qwen3 tokenizer BONSAI_GGUF=/path/to/Bonsai-1.7B-Q1_0.gguf
make -C tests/qwen3 generation BONSAI_GGUF=/path/to/Bonsai-1.7B-Q1_0.gguf
```

默认 token IDs 对应 `The capital of France is`（Bonsai 词表：`785 6722 315 9625 374`）。
对其它 Qwen3 词表使用 `--ids` 覆盖。

## 依赖与耗时

- Python 3、NumPy（参考脚本）。
- `llama-tokenize`：优先使用 `../llama.cpp/build/bin/llama-tokenize`，也可用
  `--llama-tokenize` / `LLAMA_TOKENIZE` 指定；找不到时跳过分词器差分。
- 5 token 前向差分约 40 秒：NumPy 参考约 8 秒，C 端约 30 秒（单线程约 6 秒/token）。
- 分词器差分约 10 秒（每个 prompt 重新加载 0.3 秒的模型）。
- 当前仅验证 `ACCE=soft`；RTL 后端结构上可分块执行，但仿真 1.7B 模型不实用。

## 后续

`--eval` 仍需按窗口分配 151669 宽 logits；采样策略、长上下文和 chat 工具调用
尚未实现。分词器与 chat 模板已可用于 `--prompt` 端到端生成。
