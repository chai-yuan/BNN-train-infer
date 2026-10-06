# BNN1 文件格式

`.bnn` 是 W1A8 MNIST MLP 的部署格式：**±1 权重 + int8 激活**，权重每 `group` 个共享一个 FP16 scale。
由 `train-MNIST-MLP-export.py::export()` 写出，`bnn-MNIST-MLP.c::load_model()` 读取。

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
| 0 | magic | char[4] | 4 | 固定 `"BNN1"` |
| 4 | n_layers | uint32 | 4 | 层数，`1..8` |

之后依次存放 `n_layers` 条层记录。

## 2. 层记录（每条）

| 顺序 | 字段 | 类型 | 字节 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | in_f | uint32 | 4 | 输入维度 |
| 2 | out_f | uint32 | 4 | 输出维度 |
| 3 | group | uint32 | 4 | scale 分组大小（本例 128） |
| 4 | relu | uint32 | 4 | 末层后是否 ReLU：1 是，0 否 |
| 5 | bits | uint8[] | out_f·in_f/8 | 权重符号位，见 2.1 |
| 6 | scales | fp16[] | out_f·(in_f/group)·2 | 每组一个 scale，见 2.2 |
| 7 | post_a | fp32[] | out_f·4 | 折叠后 BN 缩放 |
| 8 | post_b | fp32[] | out_f·4 | 折叠后 BN 偏置 |

### 2.1 bits（权重符号位）

| 项 | 说明 |
| --- | --- |
| 布局 | 行主序 `[out_f][in_f/8]`，行内每 8 个权重压 1 字节 |
| 取位 | 第 `k` 个权重 = 字节 `k>>3` 的第 `k&7` 位（LSB 在前） |
| 编码 | bit=1 → 权重 `+1`；bit=0 → 权重 `−1` |

### 2.2 scales（组 scale）

| 项 | 说明 |
| --- | --- |
| 布局 | 行主序 `[out_f][in_f/group]`，即 `scales[o][g]` |
| 取值 | 组内 `|w|` 均值（对 sign 量化 L2 最优） |
| 作用 | 有效权重 `w = scales[o][g] · sign` |

## 3. 字段约束（加载校验）

| 字段 | 约束 |
| --- | --- |
| magic | `== "BNN1"` |
| n_layers | `1 ≤ n ≤ 8` |
| in_f / out_f | `1 ≤ v ≤ 4096` |
| group | `group % 8 == 0` 且 `in_f % group == 0` |
| 层间 | `layers[l].in_f == layers[l-1].out_f` |
| 首层 | `in_f ≥ 784`；输入 784 补零到 in_f |
| 末层 | `out_f == 10` |

## 4. 张量语义与推理

层公式（`xq` 为激活量化值，`xs` 为激活 scale）：

```
xq[k] = clip(round(x[k] / xs), -127, 127),  xs = absmax(x) / 127
y[o]  = post_a[o] · ( xs · Σ_g scales[o][g] · Σ_{k∈g} sign[o][k]·xq[k] ) + post_b[o]
y[o]  = relu ? max(y[o], 0) : y[o]
```

| 张量 | 语义 |
| --- | --- |
| post_a | 折叠后的 `γ/√(var+eps)`；末层为 1 |
| post_b | 折叠后的 BN 偏置；末层为分类 bias |
| xs | 逐层、逐样本的激活量化 scale（int8 对称） |

## 5. 示例（`model.bnn`，共 110424 B）

| 层 | in_f | out_f | group | relu | bits | scales | post_a+post_b | 层小计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 896 | 512 | 128 | 1 | 57344 | 7168 | 4096 | 68624 |
| 1 | 512 | 512 | 128 | 1 | 32768 | 4096 | 4096 | 40976 |
| 2 | 512 | 10 | 128 | 0 | 640 | 80 | 80 | 816 |

排版：全局头 8 B + 层头 3×16 B + 数据 90752 + 11344 + 8272 = 110424 B。
