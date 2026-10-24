# Qwen3 / GGUF Q1_0 后端（M1–M3）

本后端直接读取 GGUF，在 `bnn-accelerator` 的 W1A8 接口上执行 Qwen3 推理，目标是
`Bonsai-1.7B-Q1_0.gguf`。当前已完成 GGUF 加载、Q1_0 反交织、分块 GEMM、Qwen3 前向、
KV Cache、Qwen2 byte-level BPE 分词器、chat 模板与端到端 `--prompt` 生成。

## 文件

| 文件 | 作用 |
| --- | --- |
| `bnn-sw/gguf.c/.h` | mmap、GGUF v3 头/元数据/张量表解析、边界校验 |
| `bnn-sw/tokenizer-bpe.c/.h` | Qwen2 预切分、byte-level BPE、特殊 token、`BNNE_Encode` 后端 |
| `bnn-sw/unicode-data.h` | 由 `tools/gen-unicode-tables.py` 生成的 Letter/Number/White_Space 表 |
| `bnn-sw/model-qwen3.c` | Q1_0 反交织、BNN_Matrix 分块、Qwen3 前向、KV Cache |
| `tools/bonsai-ref.py` | 同量化语义的 NumPy 参考，生成 `--ref` 文件 |
| `tests/qwen3/` | 前向 logits 差分与分词器差分测试 |

`BNNE_Open` 通过 `GGUF` magic 分派到 `bnne_qwen3_ops`，其余模型不受影响。

## Q1_0 布局

每 128 个权重一个 18 字节块，与 `bnn-accelerator` 的 W1 语义一致：

```text
uint16 d;       // FP16，块内 |w| 均值
uint8  qs[16];  // bit=1 -> +d, bit=0 -> -d; 位序 LSB-first
```

反交织到 `BNN_MatrixDesc`：

```text
weight_bits[o][k/8]   ：第 o 行第 k 位的符号
scale_fp16[o][k/128]  ：对应组的 FP16 scale
```

`token_embd.weight` 同时用于查表和 tied `lm_head`，因此保留一份反交织后的
bits/scales，矩阵创建时由后端再复制。

## 分块

所有 `BNN_Matrix` 保持在 4096 以内，不修改 soft/RTL 后端的能力上限：

| 权重 | 逻辑形状 in → out | 分块 |
| --- | --- | --- |
| q / k / v / o | 2048 → 2048 / 1024 | 1 块 |
| ffn_gate / ffn_up | 2048 → 6144 | out 方向 4096 + 2048 |
| ffn_down | 6144 → 2048 | in 方向 4096 + 2048 |
| lm_head (tied) | 2048 → 151669 | out 方向 37×4096 + 117 |

`ffn_down` 按输入列切分；主机只做一次整行 int8 量化，两个分块共享激活 scale，
输出在 FP32 中相加。统计中的 `logical_macs` 只累计逻辑矩阵乘，不重复计算。

## Qwen3 前向要点

- 每层：`RMSNorm → Q/K/V → 每头 Q/K RMSNorm → NeoX RoPE → GQA attention → Wo → 残差 → SwiGLU → 残差`。
- q/k norm 的 head_dim=128，eps 来自 GGUF（1e-6），在 RoPE 之前。
- RoPE 使用 NeoX 配对 `(j, j+64)`，不是 Llama-2 的 `(j, j+1)`。
- YaRN：`freq_scale=0.25`，`beta_fast=32`，`beta_slow=1`，`n_ctx_orig=8192`，
  `mscale=1+0.1*ln(4)`；相关维度为 `[17, 34]`。
- GQA：16 个 Q 头共享 8 个 KV 头，`kv_mul=2`。
- tied lm_head 复用 `token_embd.weight`。
- 线性层、lm_head 走加速器；RMSNorm、RoPE、attention、softmax、SwiGLU、残差在主机 FP32。
- `BNNE_Run` 从 pos=0 开始逐 token 调用同一核心；`BNNE_StepToken` 要求位置连续，
  `pos=0` 自动清空缓存。

## 分词器

`bnn-sw/tokenizer-bpe.c` 实现 Qwen2 的 GPT-2 byte-level BPE：

- 先按 llama.cpp 的 `unicode_regex_split_custom_qwen2` 做预切分：
  缩写、可选前导字符 + 字母串、单个数字、可选空格 + 标点串、换行/空白；
- 256 字节通过 GPT-2 byte-to-unicode 表映射，再对每段做 BPE；
- merge 表建成 `(left, right) → rank` 哈希，重复 merge 保留第一次出现；
- 合并后的符号查词表；查不到时按 byte token 回退；
- `token_type=3/4` 的 26 个特殊 token（`<|im_start|>`、`<|im_end|>`、
  `<think>` 等）在 `parse_special=1` 时直接输出 ID；CLI 默认按 llama-tokenize
  的行为解析特殊 token 字面量，普通文本不受影响；
- Unicode 字母/数字区间由 `tools/gen-unicode-tables.py` 从 Python 15.1.0 生成，
  与 llama.cpp 的 660 个字母区间、137 个数字区间逐点一致。

公共 API：

```c
BNNE_Status BNNE_Encode(BNNE_Engine engine, const char* text, int parse_special,
                        int32_t* ids, uint32_t capacity, uint32_t* out_count);
```

其他模型没有文本分词器，返回 `BNNE_ENOTSUP`。容量不足返回 `BNNE_ERANGE`。

## 使用

```bash
make ACCE=soft

# 原始文本续写
./bnn-infer --profile --prompt "The capital of France is" --steps 4 \
    Bonsai-1.7B-Q1_0.gguf

# Qwen3 chat 模板（单条 user 消息，空 think 块）
./bnn-infer --profile --chat --prompt "What is the capital of France?" --steps 4 \
    Bonsai-1.7B-Q1_0.gguf

# 只做分词，便于与 llama-tokenize 对拍
./bnn-infer --tokenize-only --prompt "The capital of France is" \
    Bonsai-1.7B-Q1_0.gguf
# token ids: 5: 785 6722 315 9625 374

# 调试时仍可直接传 token IDs
printf '785 6722 315 9625 374\n' > /tmp/ids.txt
./bnn-infer --profile --ids /tmp/ids.txt --steps 4 Bonsai-1.7B-Q1_0.gguf
```

预期原始续写以 `Paris` 开头。`--chat` 使用 GGUF 模板的单条 user 消息形式，
不包含 system/tools；`<|im_start|>`、`<|im_end|>`、`<think>`、`</think>`
作为特殊 token 注入。`--eval` 暂不支持（需要按窗口分配 151669 宽 logits）。

## 数值差分

前向：

```bash
python3 tools/bonsai-ref.py --model Bonsai-1.7B-Q1_0.gguf \
    --ids '785 6722 315 9625 374' --out /tmp/ref-bonsai.bin --topk 3

./bnn-infer --profile --ref /tmp/ref-bonsai.bin Bonsai-1.7B-Q1_0.gguf
# argmax agree 5/5, max relative error ~2e-7
```

参考脚本按 C 的顺序模拟量化与累加，因此短序列能达到 `1e-7` 级一致；位置 0/1
在当前模型上逐位相同。更长的 9-token 解码序列同样通过（`9/9`，`~3.5e-7`）。

分词器：

```bash
BONSAI_GGUF=Bonsai-1.7B-Q1_0.gguf \
    python3 tests/qwen3/test_tokenizer.py
# tokenizer differential test passed: 21 prompts
```

覆盖英文、中文、缩写、连续空格、换行/制表符、代码、emoji、组合字符、
数学字母、XML 和特殊 token 字面量；与 `llama-tokenize` 逐 ID 一致（24 组）。

## 当前限制

- 仅 `ACCE=soft` 实用；RTL 后端结构上可分块，但 1.7B 的 Verilator 仿真不现实。
- 运行期上下文固定为 1024（模型原生 32768），KV Cache 约 235 MB。
- 生成固定贪心；`<|im_end|>`（151645）和 `<|endoftext|>`（151643）都会停止。
- `--chat` 只实现单条 user 消息模板，没有 system/tools/多轮历史。
- `--eval`、采样策略和更长上下文尚未实现。
- 加载约 0.29 s（含分词器哈希表）；单线程约 6.1 s/token
  （1.72B MAC/token，318 次设备调用/token）。速度不是当前目标，未启用 OpenMP/SIMD。

## 中间状态调试

设置 `BNN_QWEN3_DUMP_DIR` 可导出前两个位置、每层的投影/RoPE/attention/残差结果：

```bash
mkdir -p /tmp/qwen-dump
BNN_QWEN3_DUMP_DIR=/tmp/qwen-dump ./bnn-infer --ids /tmp/ids.txt --steps 1 model.gguf
```

`tools/bonsai-ref.py --dump-dir` 会写出同名文件，便于逐层定位数值差异。
该开关只用于调试，不设置时不产生额外文件。
