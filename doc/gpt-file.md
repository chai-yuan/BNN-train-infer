# GPT1 文件格式

`GPT1` 是最小 nanoGPT 演示（中文单汉字分词）的部署格式：**W1 权重（±1 + FP16 组 scale）+ int8 激活**。
由 `train-nanoGPT-export.py::export()` 写出，`bnn-sw` 引擎（`BNNE_Open`）读取。

与 `BNN1`（MNIST MLP）的区别：权重仍是 `sign bit + FP16 组 scale`，但去掉了折叠 BN 的
`post_a/post_b`（这里用 RMSNorm，逐 token 归一化无法折叠），并额外带上 embedding、RMSNorm gain 和 UTF-8 词表。

## 约定

| 项 | 说明 |
| --- | --- |
| 字节序 | 全部小端（little-endian） |
| 浮点 | FP16 = IEEE754 half，FP32 = IEEE754 single |
| 对齐 | 无填充，记录紧密排列 |
| 结尾 | 文件末尾不得有冗余字节 |

## 1. 全局头

| 偏移 | 字段 | 类型 | 字节 | 说明 |
| --- | --- | --- | --- | --- |
| 0 | magic | char[4] | 4 | 固定 `"GPT1"` |
| 4 | vocab_size | uint32 | 4 | 词表大小 V（≤4096） |
| 8 | block_size | uint32 | 4 | 上下文长度 T（≤256） |
| 12 | n_embd | uint32 | 4 | 通道数 C，必须是 group 的整数倍 |
| 16 | n_head | uint32 | 4 | 头数 H，必须满足 `C / H == group` |
| 20 | n_layer | uint32 | 4 | 层数 L（≤8） |
| 24 | group | uint32 | 4 | scale 分组大小（128，与加速器一致） |
| 28 | rms_eps | float32 | 4 | RMSNorm 的 eps，C 端必须一致 |

之后依次为：embedding → 每层 → 最终 RMSNorm → lm_head → 词表。

## 2. Embedding（FP32）

| 顺序 | 字段 | 形状 |
| --- | --- | --- |
| 1 | tok_emb | `[V][C]` |
| 2 | pos_emb | `[T][C]` |

## 3. 每层（共 L 条）

| 顺序 | 字段 | 形状 / 说明 |
| --- | --- | --- |
| 1 | ln1_gain | float32 `[C]`，注意力前的 RMSNorm gain |
| 2 | q_proj | W1 层记录，`C → C` |
| 3 | k_proj | W1 层记录，`C → C` |
| 4 | v_proj | W1 层记录，`C → C` |
| 5 | out_proj | W1 层记录，`C → C` |
| 6 | ln2_gain | float32 `[C]`，MLP 前的 RMSNorm gain |
| 7 | mlp.fc | W1 层记录，`C → 4C`，relu=1 |
| 8 | mlp.proj | W1 层记录，`4C → C`，relu=0 |

## 4. 输出头与词表

| 顺序 | 字段 | 形状 / 说明 |
| --- | --- | --- |
| 1 | ln_f_gain | float32 `[C]` |
| 2 | lm_head | W1 层记录，`C → V`，relu=0 |
| 3 | vocab | V 条 `uint8 len` + `len` 字节 UTF-8（一个汉字一个 token） |

## 5. W1 层记录

| 顺序 | 字段 | 类型 | 字节 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | in_f | uint32 | 4 | 输入维度，必须是 group 的整数倍 |
| 2 | out_f | uint32 | 4 | 输出维度（无对齐要求） |
| 3 | group | uint32 | 4 | 必须等于全局 group |
| 4 | relu | uint32 | 4 | 1 = 输出后过 ReLU，0 = 不过 |
| 5 | bits | uint8[] | out_f·in_f/8 | 权重符号位，见 5.1 |
| 6 | scales | fp16[] | out_f·(in_f/group)·2 | 每组一个 scale，见 5.2 |

### 5.1 bits（权重符号位）

| 项 | 说明 |
| --- | --- |
| 布局 | 行主序 `[out_f][in_f/8]`，行内每 8 个权重压 1 字节 |
| 取位 | 第 `k` 个权重 = 字节 `k>>3` 的第 `k&7` 位（LSB 在前） |
| 编码 | bit=1 → 权重 `+1`；bit=0 → 权重 `−1` |

### 5.2 scales（组 scale）

| 项 | 说明 |
| --- | --- |
| 布局 | 行主序 `[out_f][in_f/group]`，即 `scales[o][g]` |
| 取值 | 组内 `|w|` 均值（对 sign 量化 L2 最优），转成 FP16 |
| 作用 | 有效权重 `w = scales[o][g] · sign` |

## 6. 张量语义与推理

层公式（`xq` 为激活量化值，`xs` 为激活 scale）：

```
xq[k] = clip(round(x[k] / xs), -127, 127),  xs = absmax(x) / 127   # 逐 token（逐行）
y[o]  = xs · Σ_g scales[o][g] · Σ_{k∈g} sign[o][k]·xq[k]
y[o]  = relu ? max(y[o], 0) : y[o]
```

| 张量 | 语义 |
| --- | --- |
| xs | 逐层、逐 token 的激活量化 scale（int8 对称） |
| ln*_gain | RMSNorm 的逐通道增益，FP32，在量化之前乘上 |

走加速器的矩阵乘：q/k/v/out_proj、注意力分数、mlp.fc、mlp.proj、lm_head。
注意力分数的加速器视角是 **X = 逐行 int8 的 key，W = ±1(+FP16 组 scale) 的 query**，
输出 `[key][query]` 转置回 `[query][key]` 后乘回 key 的激活 scale，再除以 `sqrt(head_size)`。
softmax / `attn@v` / 残差加 / RMSNorm / embedding 查表保持 FP32。

## 7. 示例（`nanogpt.bnn`，共 134820 B）

| 部分 | 字节 |
| --- | --- |
| 全局头 + tok_emb(113×128) + pos_emb(32×128) | 32 + 57856 + 16384 |
| 2 层 ×（2×gain + 4 个 C→C + C→4C + 4C→C） | 2 × 28928 |
| ln_f + lm_head | 512 + 1858 |
| 词表 113×（1+3） | 452 |

W1 权重合计 56.0 KB，FP32 等价 1667.5 KB。

## 8. 配套文件

| 文件 | 格式 |
| --- | --- |
| `ref_logits.bin` | `uint32 T, uint32 V, int32 ids[T], float32 logits[T][V]` |
| `val_tokens.bin` | `uint32 n, int32 ids[n]` |
