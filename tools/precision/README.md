# Qwen3 精度对比工具

对比 **C 引擎 W1A8** 与 **Python 全精度激活（W1A32）参考** 的逐位置 logits，
同时用 Python W1A8 作为实现对照，区分“C 端实现错误”和“预期的激活量化损失”。

目录内容：

| 文件 | 作用 |
| --- | --- |
| `dump-logits.c` | 逐 token 调用 C 引擎 `BNNE_Decode`，按固定格式导出 `n / vocab / ids / logits` |
| `Makefile` | 构建 `dump-logits`，同时刷新根工程的 `build/soft/libbnn-engine.a` |
| `compare-qwen3.py` | 跑 C、Python A8、Python A32 三条路径，计算指标并告警 |

## 构建与运行

```bash
make -C tools/precision
python3 tools/precision/compare-qwen3.py \
    --model qwen3-zh-5k.gguf --prompt 'The capital of France is'
```

使用固定 token IDs（避免 tokenizer 差异）：

```bash
python3 tools/precision/compare-qwen3.py \
    --model Qwen3-Bonsai-1.7B-Q1_0.gguf --ids-file ids.txt --json report.json
```

`compare-qwen3.py` 会自动构建缺少的 `dump-logits`。`--skip-a8` 可跳过 Python A8
对照以节省时间；`--ctx N` 传给 C 引擎；`--topk` 控制 top-k 重合率（默认 5）。

单独导出 C 端 logits：

```bash
tools/precision/dump-logits --model qwen3-zh-5k.gguf \
    --prompt '你好' --out c-logits.bin
```

输出格式与 `tools/bonsai-ref.py --out` 一致，均为 little-endian：

```text
uint32 n
uint32 vocab
int32  ids[n]
float32 logits[n * vocab]
```

## 三条对比路径

1. **C W1A8 vs Python W1A8（实现对照）**
   Python `bonsai-ref.py --activation a8` 复现同样的逐行 absmax/127 量化、
   Q1_0 权重、逐组 FP32 累加。它应当与 C 引擎非常接近。
   这里的大偏差才更像实现错误；小偏差可能来自 NumPy/C 数学库以及
   “整段 prefill”与“逐 token 增量解码”的浮点运算顺序差异。
2. **Python W1A8 vs Python A32（预期量化损失）**
   只改变激活精度，保留同一套 Q1_0 权重、FP16 scale 和逐组 FP32 累加。
   这是 W1A8 相对全精度激活的预期差距。
3. **C W1A8 vs Python A32（总差距）**
   用户最关心的数字。若它与第 2 项接近，说明 C 端实现一致，
   差距来自 W1A8 量化；若显著大于第 2 项，则要怀疑 C 端实现。

`--activation a32` 仍然使用 Q1_0 的 ±1 权重和 FP16 scale，只是不做激活量化；
它不是把权重反量化为 FP32 的“全精度模型”。

## 指标

- `max_abs` / `mean_abs` / `rms`：logits 绝对差。
- `rel_l2`：`||candidate - reference|| / ||reference||`。
- `cosine`：整体余弦相似度。
- `top1` / `topK`：逐位置 argmax 一致率和 top-k 集合重合率。
- `KL mean/max`：`KL(reference || candidate)`。
- `PPL`：用当前位置 logits 预测下一个 token 的 NLL/PPL（至少 2 个 token 时计算）。
- worst positions：逐位置 rel-L2 最大的位置，用于定位首次明显分歧。

短序列的 top1/PPL 统计意义有限；做正式评估时应使用 `--ids-file` 提供更长的
代表性 token 序列。

## 告警与退出码

默认阈值：

| 项 | 默认 | 含义 |
| --- | --- | --- |
| C vs Python A8 `max_abs` | 1.5 | 超过则硬错误 |
| C vs Python A8 `rel_l2` | 5% | 超过则硬错误 |
| C vs Python A8 PPL ratio | 偏离 1 超过 2% | 超过则硬错误 |
| C vs Python A32 `rel_l2` | 5% | 超过则警告 |
| C vs Python A32 `top1` | 0.99 | 低于则警告 |
| C vs Python A32 PPL ratio | 1.02 | 高于则警告 |
| C vs Python A32 `rel_l2` | 2% | 超过但未触发警告时给提示 |

退出码：

- `0`：OK（可能带“中等 logit 漂移但 top-1/PPL 正常”的提示）。
- `1`：WARNING，W1A8 与全精度差距过大，或总差距明显大于 Python A8→A32 的预期差距。
- `2`：ERROR，C 与 Python A8 的实现对照失败，或输入/文件/形状错误。

大误差会打印明显的 `!!! WARNING !!!` / `!!! ERROR !!!` 横幅；JSON 报告包含
全部指标、阈值、worst positions、warnings 和 notes。

所有阈值都可以用 `--impl-*` / `--total-*` 参数覆盖，例如：

```bash
python3 tools/precision/compare-qwen3.py --model MODEL.gguf --prompt '...' \
    --total-top1-warn 0.999 --total-rel-l2-warn 0.01
```

## 本地校准示例

以下数字只用于说明量级，换模型、prompt 或编译环境都会变化：

- `qwen3-zh-5k`，prompt `The capital of France is`：
  C vs Python A8 `rel_l2≈1.2e-7`；C vs A32 `rel_l2≈3.0%`、top1 1.0、
  PPL ratio≈0.998 → `OK` + 中等漂移提示。
- `Qwen3-Bonsai-1.7B-Q1_0`，同一 prompt：
  C vs Python A8 `rel_l2≈1.9%`、top1 1.0（数值顺序差异）；
  C vs A32 `rel_l2≈4.3%`、top1 0.8 → `WARNING`，
  且 Python A8→A32 的差距与总差距几乎一致，说明不是 C 实现错误。

## 排查建议

- **ERROR（实现对照失败）**：
  先检查 `dump-logits` 与 `bonsai-ref.py` 的 token IDs 是否完全一致；
  再检查 C 引擎是否仍是预期的 W1A8（F32 输入路径、逐行量化、逐组 FP32 累加）；
  最后检查最近的引擎/参考改动是否同步（例如 K 方向切分、量化下溢规则）。
- **WARNING（总差距大）**：
  比较 Python A8→A32 的指标。若两者接近，这是预期的 W1A8 量化损失；
  若总差距远大于预期，检查 C 端。
- **OK + note**：
  logits 有一定漂移，但 top-1/PPL 仍在阈值内；可结合 worst positions 和
  逐层 dump 判断是否需要更细的精度方案（如 A16）。
