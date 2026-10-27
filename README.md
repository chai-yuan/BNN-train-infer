# BNN 训练与推理加速器原型

本项目探索面向硬件的 **W1A8 二值神经网络推理**：权重为 ±1 符号，
每 128 个权重共享一个 FP16 scale；激活按行量化为 int8。
组内做整数点积，组间用 FP32 累加。矩阵运算统一通过 `bnn-accelerator`。
旧 Verilator 软件后端及 C++ 桥接已移除，当前只保留 C 软件参考后端（`ACCE=soft`）。
`bnn-hw` RTL、独立可选的 Verilator 核心自检和 `tools/wave-deck` 波形工具仍保留；
它们不参与引擎或 CLI 的默认构建。

当前 v2 只保留两条推理闭环：

- `llm`：Qwen3 GGUF（Q1_0）文本生成；
- `classify`：`bnn-mlp` GGUF（MNIST MLP）分类。

旧的 BNN1/BNN2/GPT1/LMA1 格式、旧 `bnn-sw` 引擎和旧 `bnn-infer` CLI 已移除，
需要时仍可从 git 历史恢复。

## 目录

```text
bnn-sw-v2/             v2 引擎：公共 API、GGUF、tokenizer/、bnn-mlp/、qwen3/
bnn-accelerator/       BNN_* API v2 与 soft 软件参考后端
bnn-hw/                SystemVerilog 计算核心、自检与接口说明
bnn-infer-v2.c         命令行入口（llm / classify）
train/MNIST-MLP/       二值 MLP 训练、bnn-mlp GGUF 与测试数据导出
tests/accelerator/     加速器 API 与 soft 参考测试（默认不依赖 Verilator）
tools/                 Unicode 表、Qwen3 NumPy 参考与精度对比、波形 / GGUF 检视工具
```

## 构建

需要 C11 编译器、Make、`ar` 和 POSIX 单调时钟。数值相关编译选项固定为
`-fno-fast-math -ffp-contract=off`。

```bash
make ACCE=soft
# 产物：bnn-infer-v2、build/soft/libbnn-engine.a
```

省略 `ACCE` 时同样使用 `soft`。根 Makefile 只接受 `ACCE=soft`，其他值会显式报错，
不会静默回退，也不调用 Verilator 或 C++ 工具链。
`make clean` 删除 `bnn-infer-v2` 和 `build/soft`；不会清理历史硬件构建目录或用户波形。

## 加速器 API v2

接口、GGUF 布局取舍和后端实现边界详见 [`bnn-accelerator/README.md`](bnn-accelerator/README.md)。
这是破坏性接口更新，调用者与后端必须一起重新编译。

- `BNN_BufferView` 是随任务显式传入的无状态视图：类型、布局与缓冲区范围由本次任务描述，
  不通过 Set/Get 接口在缓冲区上保存可变视图状态。
- `DOT_GROUPS` 输出 `[rows][out_features][groups]` 的 **I16** 分组整数点积，
  不再使用公共 I32 输出；本版要求 G 为 8 的倍数且不超过 256，I8 激活范围 `[-127, 127]`，
  因而组点积绝对值最多为 32512，可由 I16 精确表示。
- I8 输入和 F32 输入的 GEMV/GEMM 是不同操作。I8 路径接收调用方量化后的激活，
  执行整数点积和权重 scale 的 FP32 累加；F32 路径由后端完成逐行 absmax 对称量化、
  分组点积和输出还原，本版固定使用 I8 量化，不提供量化模式选择。
- soft 参考后端实现全部 5 种运算；`supported_ops` 表示完整后端提供的逻辑能力，
  包含相应量化规则，而非硬件单次 tile 限制。
- 引擎使用单行 `GEMV_F32` 路径，主机侧不再做激活量化；超宽输出按
  `max_out_features` 分片，矩阵输入宽度必须落在 `max_in_features` 内。
  未来硬件接入、设备调度与 IPC 设计留待后续，本轮不实现新后端。

公共 DOT 的 I16 布局与旧 RTL 的 `res_data[31:0]` 物理结果协议解耦；本轮不修改 RTL，
也不要求核心结果总线缩窄。

## LLM：Qwen3

```bash
# 原始文本续写
./bnn-infer-v2 llm Qwen3-Bonsai-1.7B-Q1_0.gguf \
    --prompt "The capital of France is" --steps 4

# Qwen3 chat 模板（单条 user 消息，空 think 块）
./bnn-infer-v2 llm Qwen3-Bonsai-1.7B-Q1_0.gguf \
    --chat --prompt "你好" --steps 40

# 按 token ID 输入（空白分隔的十进制文本文件）
./bnn-infer-v2 llm Qwen3-Bonsai-1.7B-Q1_0.gguf --ids ids.txt --ctx 1024
```

参数：

- `--prompt TEXT` 或 `--ids FILE` 二选一；都不给时默认提示词为 `春天`。
- `--steps N` 最大生成 token 数，默认 40。
- `--ctx N` 上下文长度，默认 1024（`n_ctx_train` 为 32768）。
- `--chat` 使用 GGUF 中的 Qwen3 单用户模板；默认是原始续写。

生成固定为贪心；遇到 `<|im_end|>` / `<|endoftext|>` 停止。
`1024` 上下文对应的 KV Cache 约 235 MB。当前 CLI 没有 `--profile`、`--ref`、
`--eval`、采样或多轮/system/tools 对话。

## 分类：bnn-mlp

训练脚本会下载 MNIST 原始 IDX 文件，训练二值 MLP（STE + BatchNorm），
把 BN 折叠进 `post_scale/post_bias`，并导出 GGUF 与部署用测试数据：

```bash
python3 train/MNIST-MLP/train_mnist_mlp.py
# train/MNIST-MLP/out/mnist_mlp.gguf
# train/MNIST-MLP/out/test-inputs.f32
# train/MNIST-MLP/out/test-labels.u8
```

默认 5 epochs、hidden 512。当前导出模型的测试集准确率为 98.40%
（W1A8 模拟 98.39%），1000 条导出测试子集为 98.30%。

```bash
./bnn-infer-v2 classify train/MNIST-MLP/out/mnist_mlp.gguf \
    --input train/MNIST-MLP/out/test-inputs.f32 \
    --labels train/MNIST-MLP/out/test-labels.u8 --batch 64
```

参数：

- `--input FILE` 连续的 little-endian FP32，每样本 `input_features` 个值；
  必须已经按 `(pixel/255 - 0.1307) / 0.3081` 归一化并补零到 896。
- `--labels FILE` 可选，每样本 1 字节；提供时输出准确率。
- `--batch N` 单次 `BNNE_Classify` 的样本数，默认 64。
- `--verbose` 逐样本打印预测类别。

训练依赖 Python 3、NumPy 和 CPU 版 PyTorch；脚本直接下载 raw MNIST，不需要 torchvision。

## 模型格式

GGUF 文件格式及磁盘上的 Q1_0 布局不变。加载层继续负责把 GGUF 数据 repack 为
加速器所需的权重位和 scale 布局；本轮不把 repack 迁移到公共 API 或硬件后端。

### Qwen3

`general.architecture = "qwen3"`，权重为 Q1_0 张量（每 128 个 ±1 权重一个 FP16 scale），
RMSNorm 等为 F32。引擎按 accelerator 的逻辑能力切分矩阵；当前 soft 上限为 8192，
FFN（6144）单块执行，lm_head（151669）按输出分片。KV Cache 按 Context 独立分配。

### bnn-mlp

`general.architecture = "bnn-mlp"`。元数据：

```text
bnn-mlp.block_count / input_features / input_pixels / input_mean / input_std / class_count
bnn-mlp.block.{i}.activation = "relu" | "none"
bnn-mlp.labels（可选字符串数组）
```

张量：

```text
blk.{i}.weight       Q1_0，dims=[in, out]
blk.{i}.post_scale   F32，dims=[out]
blk.{i}.post_bias    F32，dims=[out]
```

前向按层执行 `post_scale * (W·x) + post_bias`，再按元数据选择 ReLU。

## 测试

加速器 API 与 soft 参考测试默认不依赖 Verilator：

```bash
make -C tests/accelerator test
```

硬件自检是独立可选项，需要 Verilator 及其 C++ 构建工具链，不链接软件加速器后端：

```bash
make -C bnn-hw/bnn_core_test test
make -C bnn-hw/bnn_core_test lint
```

旧桥接/context 测试和 `tests/trace` 波形开关测试已随软件后端删除。
`bnn_core_test` 当前没有 `waves` 目标或 VCD 导出逻辑；可用现有 fixture 查看波形，
不能再通过推理 CLI 或已移除的后端生成 VCD。

v2 引擎/模型的自动化回归尚未恢复；LLM 和分类的正确性目前用上面的 CLI 命令手动验证。

## 工具

- `tools/gen-unicode-tables.py`：重新生成 `bnn-sw-v2/tokenizer/unicode-data.h`。
- `tools/bonsai-ref.py`：Qwen3/Q1_0 的独立 NumPy 参考；`--activation a8|a32` 选择激活精度，
  `--topk` 打印参考类别与 logits，`--out` 写出旧版 BNR3 `--ref` 布局（v2 CLI 不读取）。
- `tools/precision/`：Qwen3 C W1A8 与 NumPy A32/A8 的逐位置 logits、top-k、KL、PPL 对比，
  区分实现误差与预期量化损失，并在差距过大时显著告警。
- `tools/wave-deck`：独立 RTL 波形辅助工具；保留示例/fixtures 与字体资源（字体也供 gguf-lens 共用）。
- `tools/gguf-lens`：纯前端 GGUF 结构检视器，查看元数据、张量布局、量化块与原始字节。
- `bnn-hw/README.md`：计算核心握手、完成协议和复位语义。
