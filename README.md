# BNN 训练与推理加速器原型

本项目探索面向硬件的 **W1A8 二值神经网络推理**：权重为 ±1 符号，
每 128 个权重共享一个 FP16 scale；激活按行量化为 int8。
组内做整数点积，组间用 FP32 累加。矩阵运算统一通过 `bnn-accelerator`，
可在 C 软件模拟（`ACCE=soft`）和 Verilator RTL 仿真（`ACCE=verilator`）之间切换。

当前 v2 只保留两条推理闭环：

- `llm`：Qwen3 GGUF（Q1_0）文本生成；
- `classify`：`bnn-mlp` GGUF（MNIST MLP）分类。

旧的 BNN1/BNN2/GPT1/LMA1 格式、旧 `bnn-sw` 引擎和旧 `bnn-infer` CLI 已移除，
需要时仍可从 git 历史恢复。

## 目录

```text
bnn-sw-v2/             v2 引擎：公共 API、GGUF、tokenizer/、bnn-mlp/、qwen3/
bnn-accelerator/       BNN_* 设备接口、soft/Verilator 后端与 RTL 桥接
bnn-hw/                SystemVerilog 计算核心、自检与接口说明
bnn-infer-v2.c         命令行入口（llm / classify）
train/MNIST-MLP/       二值 MLP 训练、bnn-mlp GGUF 与测试数据导出
tests/accelerator/     加速器 API 与 soft/RTL 差分测试
tests/trace/           VCD 波形开关测试
tools/                 Unicode 表生成、Qwen3 NumPy 参考、波形 / GGUF 检视工具
```

## 构建

需要 C11 编译器、Make、`ar` 和 POSIX 单调时钟。数值相关编译选项固定为
`-fno-fast-math -ffp-contract=off`。

```bash
make ACCE=soft
# 产物：bnn-infer-v2、build/soft/libbnn-engine.a
```

Verilator 后端另需 Verilator 和 C++17 编译器：

```bash
make ACCE=verilator -j2
# 产物：bnn-infer-v2、build/verilator/libbnn-engine.a
```

`make clean` 删除 `bnn-infer-v2` 和当前 `ACCE` 的构建目录。

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

### Qwen3

`general.architecture = "qwen3"`，权重为 Q1_0 张量（每 128 个 ±1 权重一个 FP16 scale），
RMSNorm 等为 F32。引擎按 accelerator 能力把 FFN（6144）和 lm_head（151669）
切到每矩阵 4096 以内；KV Cache 按 Context 独立分配。

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

当前保留的自动化测试只覆盖 accelerator 和 RTL 波形，均直接使用公共 `BNN_*` API：

```bash
make -C tests/accelerator test   # soft/RTL 差分、桥接、宿主 context
make -C tests/trace test         # VCD 开关与内容
make -C bnn-hw/bnn_core_test test
```

v2 引擎/模型的自动化回归尚未恢复；LLM 和分类的正确性目前用上面的 CLI 命令手动验证。
`tests/accelerator` 和 `tests/trace` 会自动构建 `ACCE=verilator` 归档。

## 工具

- `tools/gen-unicode-tables.py`：重新生成 `bnn-sw-v2/tokenizer/unicode-data.h`。
- `tools/bonsai-ref.py`：Qwen3/Q1_0 的独立 NumPy 参考；`--topk` 打印参考类别与 logits，
  `--out` 写出旧版 BNR3 `--ref` 文件（v2 CLI 不读取）。
- `tools/wave-deck`：RTL 波形辅助工具。
- `tools/gguf-lens`：纯前端 GGUF 结构检视器，查看元数据、张量布局、量化块与原始字节。
- `bnn-hw/README.md`：计算核心握手、完成协议和复位语义。
