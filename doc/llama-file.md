# LMA1 文件格式

`LMA1` 是基于 **Llama-2 架构（RoPE + SwiGLU + RMSNorm）** 面向硬件加速的部署格式：**W1 权重（±1 + FP16 组 scale）+ int8 激活**。
由 `train-llama2-export.py` 写出，`bnn-sw` 引擎（`BNNE_Open`）与 `bnn-infer` 读取执行。

与 `GPT1`（nanoGPT）的主要区别：
1. **位置编码**：采用 RoPE（旋转位置编码，作用于注意力层的 Q/K 向量），完全去掉了绝对位置编码表 `pos_emb`（节省 `T×C` 参数）；
2. **前馈网络 (FFN)**：采用 Llama-2 标准的 SwiGLU 门控网络（`w1` gate、`w2` down、`w3` up 三个 W1 权重矩阵，激活函数为 $\text{SiLU}(w_1(x)) \odot w_3(x)$），替代 nanoGPT 的 2 矩阵 ReLU MLP；
3. **隐层维度**：支持任意 128 对齐的 `hidden_dim`（通常约为 $\frac{8}{3}C$），突破 nanoGPT 固定的 $4C$ 限制；
4. **多查询注意力 (GQA)**：支持分组查询注意力（`n_kv_heads <= n_heads`）。

## 约定

| 项 | 说明 |
| --- | --- |
| 字节序 | 全部小端（little-endian） |
| 浮点 | FP16 = IEEE754 half，FP32 = IEEE754 single |
| 对齐 | 无填充，记录紧密排列 |
| 结尾 | 文件末尾不得有冗余字节 |

## 1. 全局头 (40 字节)

| 偏移 | 字段 | 类型 | 字节 | 说明 |
| --- | --- | --- | --- | --- |
| 0 | magic | char[4] | 4 | 固定 `"LMA1"` |
| 4 | vocab_size | uint32 | 4 | 词表大小 V（≤65536） |
| 8 | block_size | uint32 | 4 | 上下文长度 T（≤4096） |
| 12 | dim | uint32 | 4 | 通道数 C，必须是 group 的整数倍 |
| 16 | n_heads | uint32 | 4 | 查询头数 H，必须满足 `C / H == group` (128) |
| 20 | n_kv_heads | uint32 | 4 | 键值头数 H_kv，必须满足 `H % H_kv == 0` |
| 24 | n_layers | uint32 | 4 | 层数 L（≤128） |
| 28 | hidden_dim | uint32 | 4 | SwiGLU 隐层维度，必须是 group 的整数倍 |
| 32 | group | uint32 | 4 | scale 分组大小（固定 128，与加速器一致） |
| 36 | rms_eps | float32 | 4 | RMSNorm 的 eps（默认 1e-5） |

之后依次为：embedding → 每层 → 最终 RMSNorm → lm_head → 词表。

## 2. Embedding（FP32）

| 顺序 | 字段 | 形状 | 说明 |
| --- | --- | --- | --- |
| 1 | tok_emb | `[V][C]` | 词表嵌入，FP32 |

*(注意：无 `pos_emb` 表，位置编码通过注意力层的 RoPE 注入)*

## 3. 每层（共 L 条）

| 顺序 | 字段 | 形状 / 说明 |
| --- | --- | --- |
| 1 | att_norm_gain | float32 `[C]`，注意力前的 RMSNorm gain |
| 2 | wq | W1 层记录，`C → C`，relu=0 |
| 3 | wk | W1 层记录，`C → kv_dim`，relu=0（`kv_dim = C * n_kv_heads / n_heads`） |
| 4 | wv | W1 层记录，`C → kv_dim`，relu=0 |
| 5 | wo | W1 层记录，`C → C`，relu=0 |
| 6 | ffn_norm_gain | float32 `[C]`，FFN 前的 RMSNorm gain |
| 7 | w1 | W1 层记录，`C → hidden_dim`，gate_proj，relu=0 |
| 8 | w2 | W1 层记录，`hidden_dim → C`，down_proj，relu=0 |
| 9 | w3 | W1 层记录，`C → hidden_dim`，up_proj，relu=0 |

## 4. 输出头与词表

| 顺序 | 字段 | 形状 / 说明 |
| --- | --- | --- |
| 1 | final_norm_gain | float32 `[C]` |
| 2 | lm_head | W1 层记录，`C → V`，relu=0 |
| 3 | vocab | V 条 `uint8 len` + `len` 字节 UTF-8 |

## 5. W1 层记录

格式与 `GPT1` 完全一致：

| 顺序 | 字段 | 类型 | 字节 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | in_f | uint32 | 4 | 输入维度，必须是 group 的整数倍 |
| 2 | out_f | uint32 | 4 | 输出维度 |
| 3 | group | uint32 | 4 | 必须等于全局 group (128) |
| 4 | relu | uint32 | 4 | 0 = 线性输出（Llama-2 均使用 0） |
| 5 | bits | uint8[] | out_f·in_f/8 | 权重符号位（行主序，LSB 在前） |
| 6 | scales | fp16[] | out_f·(in_f/group)·2 | 每组 FP16 组 scale |

## 6. 加速器与主机分工

- **走硬件加速器的 W1A8 GEMM**：`wq`、`wk`、`wv`、`wo`、注意力分数 $Q \cdot K^T$、`w1`、`w3`、`w2`、`lm_head`；
- **主机端 FP32**：RMSNorm、RoPE 旋转、Softmax、$\text{attn} \cdot V$、SwiGLU 的 $\text{SiLU}(w_1(x)) \odot w_3(x)$ 激活、残差相加与词表查找。

## 7. 配套文件

- `ref_logits.bin`：`uint32 T, uint32 V, int32 ids[T], float32 logits[T][V]`
- `val_tokens.bin`：`uint32 n, int32 ids[n]`
