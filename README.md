# BNN 训练与推理加速器原型

本项目探索面向硬件的 **W1A8 二值神经网络推理**：权重符号用 1 bit 表示，激活量化为 int8，
每组权重共享 FP16 scale，组内进行整数点积、组间使用 FP32 累加。
通过 Python 训练与导出、C 推理引擎和软件模拟加速器，评估这种计算方式的精度、性能与适用范围，
为后续硬件接口和架构设计提供依据。

## 当前进度

目前提供 **C 软件模拟后端和 Verilator RTL 仿真后端**，可以让现有推理链路执行 W1A8 计算核心；尚未接入实际 FPGA/ASIC 设备。

- 已支持 MNIST MLP、MNIST CNN、nanoGPT 以及 Llama-2 的训练导出及 C 推理。
- 新增 GGUF/Qwen3 软件后端：可直接加载 `Bonsai-1.7B-Q1_0.gguf` 的 Q1_0 权重，
  已打通 Qwen3 前向、KV Cache、参考 logits 差分、Qwen2 byte-level BPE 分词器、
  chat 模板与端到端 `--prompt` 生成。
- 已将模型统一定义在 `bnn-sw`，使用不透明 `BNNE_Engine` 句柄和 `bnn-infer` 演示入口（支持 GPT1、LMA1 与 GGUF/Qwen3 三类 LLM 格式）。
- 二值矩阵计算统一通过 `bnn-accelerator.h`；可选择 C 软件实现或 Verilator RTL 仿真。
- 已提供 Python logits 正确性校验与分阶段性能统计，不再维护新引擎内的独立参考内核。
- 已通过合成模型数值回归、错误路径、多句柄隔离及内存检查；现有小型 GPT 模型已通过 Python reference 校验。

推理入口围绕两类任务组织：分类（MLP/CNN，输入数据、输出类别）和文本生成（GPT，输入提示、输出文本）；
logits 校验与性能统计是两类任务共用的选项。
真实模型的精度与性能应结合各自的数据集评估；软件后端的时间不能直接代表未来硬件性能。

## 项目结构

```text
train-*-export.py       Python 训练、模型导出和参考数据生成
bnn-infer.c            统一演示入口：分类与文本生成两类任务，附校验与统计
bnn-sw/                BNNE 引擎、模型加载、工作区和前向实现（含 GGUF/Qwen3 后端）
bnn-accelerator/       设备级公共接口、soft/Verilator 后端及 C ABI 仿真桥接
tests/                 分类回归测试（engine / llama / qwen3 / accelerator / trace）
tools/                 GGUF 参考实现等辅助脚本
doc/                   引擎接口及 BNN1、GPT1、LMA1、Qwen3-GGUF 格式说明
bnn-hw/                SystemVerilog 计算核心、核心自检测试及接口说明
```

调用关系：`应用 → BNNE 引擎 → accelerator 接口 → 软件模拟 / 后续硬件后端`。

## 构建与使用

C 端需要支持 C99 的编译器、Make、`ar` 和 POSIX 单调时钟；当前主要在 Linux 上使用。

```bash
make ACCE=soft
```

生成 `./bnn-infer` 和 `build/soft/libbnn-engine.a`。后端在构建时选择，入口没有 `--soft` 参数。

使用当前 SystemVerilog 核心执行整数点积：

```bash
make ACCE=verilator -j2
make -C tests/accelerator test
make -C tests/engine ACCE=verilator test
make -C tests/llama ACCE=verilator test
```

该后端另需 Verilator（本地验证版本 5.032）和 C++17 编译器，产物为 `build/verilator/libbnn-engine.a`。
不需要 cocotb、Python 虚拟环境或动态库加载。每个设备拥有独立模型，C++ 桥接层逐周期驱动命令、数据和结果握手；
DOT_GROUPS 的整数结果直接来自 RTL，GEMV/GEMM 的 scale 和 FP32 组间累加在 C 中完成。
核心在 `cmd_valid && cmd_ready` 的上升沿锁存三个尺寸并启动；`act_data/weight_data` 共用 `in_valid/in_ready`，输入边界由尺寸和握手计数推导，无输入 last。
状态顺序为 `IDLE -> RUN -> DRAIN -> IDLE`，保留 `status_idle/status_busy`；唯一成功完成条件是 `res_valid && res_ready && res_last`，没有独立完成、取消或结果错误通道。
最后结果握手沿不接新命令，返回 IDLE 后下一上升沿才可接收。配置合法性由外层保证，核心仅用仿真 assert 检查非零、整除和 `groups <= 256`，不返回运行时错误码。
`reset` 为高有效同步复位，清空全部锁存配置、计数和数据有效位，置位期间屏蔽所有握手；无取消通知，外层须清理输入源并作废被复位任务的部分输出。详细接口见 [核心说明](bnn-hw/README.md)。
两种后端具有相同的公共 API 和能力范围。仿真耗时不能代表实际硬件性能。

静态归档包含 C 后端、C++ 桥接、RTL 模型及 Verilator 运行库对象。
嵌入程序继续按 C 编译，最终使用 C++ 链接器链接归档及 `-lm -pthread -latomic`。
切换后端时重新执行相应的 `make ACCE=...`，根目录 `bnn-infer` 会重新链接。

### Verilator 波形开关

桥接头文件 `bnn-accelerator/verilator-bnn-bridge.h` 定义 `BNN_RTL_TRACE_DIR_ENV`。
运行时环境变量 `BNN_RTL_TRACE_DIR` 未设置或为空时不导出波形；设为已有可写目录时，
每次创建设备生成一个独立的 `bnn-core-XXXXXX.vcd`，覆盖该设备的复位和全部计算任务。
文件名自动避重，不覆盖旧文件。目录不存在或不可写时，设备创建返回 `BNN_EIO`。
变量在创建设备时读取，运行期间改变不会影响已有设备。

默认构建已包含 `--trace` 支持，开关无需重新编译。VCD 包含计算核心及子模块，时间单位 1ps，仿真时钟周期 10ns。
这个时钟周期仅用于波形显示，不代表综合后的时序能力。每次任务结束或超时恢复后刷新文件，关闭设备时结束文件。

先用很小的计算任务生成波形：

```sh
make -C tests/trace waves
# 脚本会打印两个实际 VCD 路径，任选一个打开：
# gtkwave build/waves/bnn-core-XXXXXX.vcd
```

该测试使用 K=128、N=2、G=128、全 1 激活，两个通道分别全正/全负权重，DOT 结果应为 `[128, -128]`。
测试会检查关闭开关、错误目录、实时刷新、文件避重及两个设备隔离。
`make -C tests/trace test` 执行相同检查，但波形存放在临时目录并自动清理。
加速器差分测试也可以导出波形：`make -C tests/accelerator waves`，默认输出到 `build/waves/accelerator`。

给任意推理命令启用波形：

```sh
mkdir -p build/model-waves
BNN_RTL_TRACE_DIR="$PWD/build/model-waves" ./bnn-infer --profile --ref ref_logits.bin model.bnn
```

建议先调试小型任务，完整模型的 VCD 可能很大。只在这条命令前设置变量即可；不设置该变量的后续运行仍关闭导出。
GTKWave 中先查看 `clock/reset`、`cmd_valid/cmd_ready`、`in_valid/in_ready`、`act_data/weight_data`、
`res_valid/res_ready/res_data/res_group_idx/res_chan_last/res_last` 和 `status_idle/status_busy`，再展开内部流水级。
以最后结果的 `res_valid && res_ready && res_last` 握手沿判断完成，不要只看输入送完或 res_last 置位。

### MNIST MLP / CNN

两个模型使用相同命令，通过模型文件自动区分类型：

```bash
./bnn-infer --profile --ref ref_logits.bin model.bnn \
    data/MNIST/raw/t10k-images-idx3-ubyte \
    data/MNIST/raw/t10k-labels-idx1-ubyte
```

CNN 将模型路径替换为 `model-cnn.bnn`，并使用对应的 reference 文件。
省略 `--ref` 时仅做分类评测；省略 `--profile` 时不采集和显示性能统计。
输入需要解压后的 IDX 文件。reference 必须对应这份测试集的前 N 张图片及其原始顺序。

### LLM (nanoGPT & Llama-2)

```bash
# 仅与 Python logits 比较
./bnn-infer --profile --ref ref_logits.bin nanogpt-lib.bnn

# 贪心生成（默认全上下文前向）
./bnn-infer --profile --prompt '春天' --steps 40 nanogpt-lib.bnn

# 开启 KV Cache 单步自回归生成（获得十倍以上解码加速）
./bnn-infer --profile --kv-cache --prompt '春天' --steps 40 llama2.bnn

# 验证集 next-token 评测：loss、perplexity、top1
./bnn-infer --profile --eval val_tokens.bin llama2.bnn
```

只指定 `--ref` 或 `--eval` 不会自动生成文本；没有指定任何任务时默认从“春天”生成 40 个 token。
模型、reference、tokens 必须配套使用。任何文件、推理或正确性校验失败均返回非零退出码。

### GGUF / Qwen3（Bonsai-1.7B-Q1_0）

`BNNE_Open` 通过 `GGUF` magic 识别 Qwen3 模型，Q1_0 权重直接映射到加速器的 W1
（每组 128 个 ±1 权重 + FP16 scale）。C 端已接入 Qwen2 byte-level BPE 分词器，
可直接用 `--prompt` 生成；`--chat` 使用 GGUF 模板的单条 user 消息与空 think 块：

```bash
# 原始续写
./bnn-infer --profile --prompt "The capital of France is" --steps 4 \
    Bonsai-1.7B-Q1_0.gguf

# Qwen3 chat 模板
./bnn-infer --profile --chat --prompt "What is the capital of France?" --steps 4 \
    Bonsai-1.7B-Q1_0.gguf

# 只分词，便于与 llama-tokenize 对拍
./bnn-infer --tokenize-only --prompt "The capital of France is" \
    Bonsai-1.7B-Q1_0.gguf
# token ids: 5: 785 6722 315 9625 374
```

调试时仍可用 `--ids` 直接传入 token IDs。同时提供 NumPy 参考实现，按与 C 相同的
W1A8 量化语义生成 `--ref` 文件：

```bash
python3 tools/bonsai-ref.py --model Bonsai-1.7B-Q1_0.gguf \
    --ids '785 6722 315 9625 374' --out /tmp/ref-bonsai.bin
./bnn-infer --profile --ref /tmp/ref-bonsai.bin Bonsai-1.7B-Q1_0.gguf
```

分词器在不同文本上与 `llama-tokenize` 逐 ID 一致（见 `tests/qwen3/test_tokenizer.py`）。
详细格式、分块策略与数值差分见 [Qwen3-GGUF 说明](doc/qwen3-gguf.md)。

### 训练与导出

Python 训练需要 `torch`、`numpy`，MNIST 另外需要 `torchvision`。
训练脚本会下载 MNIST；GPT 可使用内置语料或通过 `--data` 指定文本。

```bash
python3 train-MNIST-MLP-export.py --out model.bnn
mv ref_logits.bin ref-mlp.bin

python3 train-MNIST-CNN-export.py --out model-cnn.bnn
mv ref_logits.bin ref-cnn.bin

python3 train-nanoGPT-export.py --out nanogpt.bnn \
    --ref-out ref-gpt.bin --tokens-out val-gpt.bin

python3 train-llama2-export.py --out llama2.bnn \
    --ref-out ref-llama.bin --tokens-out val-llama.bin
```

MLP/CNN 脚本都写入当前目录的 `ref_logits.bin`，应及时另存以免覆盖。
完整训练参数可通过各脚本的 `--help` 查看；GPT 与 C 对比时保留默认量化注意力，勿使用 `--no-attn-quant`。
模型与参考数据通常由本地训练生成，不要求提交到仓库。

## 测试与性能统计

```bash
make -C tests/engine test
make -C tests/engine ACCE=verilator test
make -C tests/llama test
make -C tests/accelerator test
make -C tests/trace test

# Qwen3 差分测试需要外部 GGUF 模型
make -C tests/qwen3 test BONSAI_GGUF=/path/to/Bonsai-1.7B-Q1_0.gguf
```

`tests/` 按类别拆分：`engine/` 覆盖 MLP/CNN/GPT、引擎 API 与 CLI；`llama/` 覆盖 Llama 导出、评测、生成和 KV Cache；
`qwen3/` 用 NumPy 同量化参考差分 GGUF/Qwen3 前向，并与 `llama-tokenize` 对拍分词器；
`accelerator/` 覆盖加速器 API、软/硬件差分、桥接与宿主 context；`trace/` 覆盖可选的 VCD 波形。
engine/llama 的 Makefile 会先构建所选后端；accelerator/trace 固定构建 `ACCE=verilator`。
每个分类目录的 `README.md` 说明了测试内容、依赖、覆盖范围和选项。

测试会在临时目录生成模型与数据，检查三种模型的独立 Python 参考结果、两类任务（分类评测、生成/验证集评测）、
动态序列、输入校验、资源回收、统计和多句柄隔离。engine 测试仅依赖 Python 3 标准库和 C 构建工具；
llama 测试还需要导出器所需依赖（可用 `PYTHON=/absolute/path/to/.venv/bin/python3` 指定虚拟环境）。
这类回归不能替代真实数据集的精度测试。
`make -C tests/engine ACCE=verilator test` 通过相同引擎/CLI 测试验证 RTL 接入。
`make -C tests/accelerator test` 另行比较 soft 与 RTL 后端的整数及 FP32 输出字节，
覆盖三种操作、步长和非对齐偏移、padding 保持、输入校验、资源归属、多设备隔离、桥接超时后恢复及宿主 Verilator 上下文保持。
RTL 共用输入停顿、逐周期随机输出反压、最后结果完成握手和复位恢复测试仍由 `make -C bnn-hw/bnn_core_test test` 独立运行；非法配置通过预期 assert 失败的独立探针验证。

若要只运行某种模型，可以保留合成测试数据后直接调用 CLI，无需训练或下载：

```sh
make ACCE=verilator -j2
make -C tests/engine fixtures FIXTURES=build/verilator-fixtures

# MLP：8 张合成图像及 Python reference
./bnn-infer --profile --ref build/verilator-fixtures/ref.bin \
    build/verilator-fixtures/mnist.bnn \
    build/verilator-fixtures/images.idx build/verilator-fixtures/labels.idx

# CNN：2 张合成图像及 Python reference
./bnn-infer --profile --ref build/verilator-fixtures/cnn-ref.bin \
    build/verilator-fixtures/cnn.bnn \
    build/verilator-fixtures/cnn-images.idx build/verilator-fixtures/cnn-labels.idx

# GPT：logits 校验和 next-token 评测
./bnn-infer --profile --ref build/verilator-fixtures/gpt-logits.bin build/verilator-fixtures/gpt.bnn
./bnn-infer --profile --eval build/verilator-fixtures/gpt-eval.bin build/verilator-fixtures/gpt.bnn

# 核心 RTL：固定随机种子及任务数，日志位于 build/bnn-core-test/
make -C bnn-hw/bnn_core_test test SEEDS=7 RANDOM_CASES=32
```

MLP 合成标签故意放入两个错误类别，正常结果是 6/8；这里应以 reference 校验通过判断数值正确性。
对上述任一 `./bnn-infer` 命令加上 `BNN_RTL_TRACE_DIR=已有目录` 前缀即可记录该模型的波形。

`--profile` 分别报告加载、校验、评测、生成的耗时；推理内部细分量化、传输、设备执行和其余主机工作，
并统计设备调用数、传输字节及有效/补齐后 MAC。MAC 仅计卸载的矩阵运算。
各阶段独立累计，软件模拟中的传输主要是内存复制，不等同真实 DMA。

## 当前限制与后续安排

当前软件设备的 group 为 128，最大输入和输出维度均为 4096。
CNN 按逐组补齐后的尺寸检查；GPT 要求 head size=128、`4*C <= 4096` 且词表不大于 4096。
仓库已有的 `nanogpt-lib.bnn` 可运行；`nanogpt_l3_b48.bnn` 的词表为 6362，当前会返回不支持。
现有 GPT 路径暂不提供矩阵分块、异步调度或多种采样策略，优先保持实现简单。

GGUF/Qwen3 后端在模型层把 6144 维 FFN 和 151669 维 lm_head 分块到 4096 以内，
因此设备能力上限不变。当前限制：

- 运行期上下文固定为 1024；`--eval`、采样策略和更长上下文尚未实现。
- `--chat` 只实现单条 user 消息模板，没有 system/tools/多轮历史。
- 生成固定贪心；`<|im_end|>` 与 `<|endoftext|>` 都会停止。
- 速度不是当前目标：单线程约 6.1 s/token，未启用 OpenMP/SIMD。
- 分块与前向只在 `ACCE=soft` 上验证；RTL 后端结构上可分块执行，但 1.7B 仿真不实用。

后续按以下顺序推进：

1. 按需补齐 `--eval`、采样策略和更长上下文，并评估 chat 的多轮/system/tools 模板。
2. 用真实数据建立精度基线；软件后端速度不作为最终执行指标。
3. 根据测量结果确定硬件的分组、矩阵规模、缓冲及传输能力，再决定是否需要分块等扩展。

嵌入调用、张量与所有权约定见 [BNNE 引擎说明](doc/bnn-engine.md)；
格式说明见 [BNN1](doc/bnn-file.md)、[GPT1](doc/gpt-file.md)、[LMA1](doc/llama-file.md)
和 [Qwen3-GGUF](doc/qwen3-gguf.md)。
