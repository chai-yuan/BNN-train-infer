# BNNE 推理引擎

`bnn-sw/bnn-engine.h` 是唯一公共入口，支持 BNN1 MLP、BNN2 CNN、GPT1 nanoGPT。
引擎负责加载模型、管理设备与工作区、前向执行和统计；`bnn-infer.c` 按两类任务组织演示：
分类（MNIST 评测）与文本生成（GPT 生成/验证集评测），Python reference 校验和性能统计
是两类任务共用的选项。

## 构建

```sh
make ACCE=soft
make ACCE=soft test
```

产物是 `bnn-infer` 与 `build/<ACCE>/libbnn-engine.a`。静态库已包含所选 accelerator。
`ACCE=soft` 的嵌入程序使用 C 编译器链接此库和 `-lm`。
`ACCE=verilator` 另需 Verilator 和 C++17 编译器，将整数点积交给 RTL 仿真，浮点后处理仍在 C 中执行。
该静态库还包含桥接层、生成模型和 Verilator 运行库对象；嵌入程序的 C 源先按 C 编译，
再用 C++ 链接器链接归档和 `-lm -pthread -latomic`，无须直接访问生成头文件。
两种后端均要求 IEEE-754 binary32、POSIX 单调时钟，并禁用 fast-math 和 FMA 收缩。

所有模型的二值矩阵运算统一通过 `bnn-accelerator.h`，在构建时通过 `ACCE=soft` 或 `ACCE=verilator` 选择后端。
新引擎不再提供独立参考内核、backend 选择或运行参数 `--soft`。
正确性由 Python 导出的 logits 校验；FP32 的池化、归一化、softmax、残差等必要主机运算仍保留。

## 演示入口

### MNIST MLP / CNN

```sh
./bnn-infer --profile --ref ref_logits.bin model.bnn \
    t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
```

引擎通过模型 magic 区分 MLP/CNN。CLI 读取 28x28 IDX 图像与标签，按训练时的
`(pixel/255 - 0.1307)/0.3081` 归一化；MLP 额外补零至输入宽度。
省略 `--ref` 时只做分类评测；reference 一律通过 `--ref` 传入。

MNIST reference 格式沿用导出脚本：小端 uint32 `N,10`，随后 FP32 `[N,10]` logits，
对应 IDX 数据的前 N 项，必须与导出 reference 时的图片顺序一致。
校验指标是最大 `abs(actual-reference)/(1+abs(reference))`，容差 `1e-3`。
MLP/CNN 脚本默认使用相同 reference 文件名，使用时应分别保存并配对。

### GPT

```sh
# 仅校验 Python 前向结果，不自动生成文本
./bnn-infer --profile --ref ref_logits.bin nanogpt-lib.bnn

# 最简生成：贪心选择下一 token，每步完整前向，不使用 KV cache
./bnn-infer --profile --prompt '春天' --steps 40 nanogpt-lib.bnn

# 可选验证集评测
./bnn-infer --profile --eval val_tokens.bin nanogpt-lib.bnn
```

GPT reference 格式为小端 uint32 `n,V`，随后 int32 IDs `[n]` 与 FP32 logits `[n,V]`。
校验所有位置的最大相对误差（容差 `1e-2`）及 argmax 一致性。
reference 与验证集一律通过 `--ref` / `--eval` 传入，位置参数只有模型路径本身。

不指定 reference、eval 或生成选项时，默认从“春天”生成 40 个 token。
指定 `--prompt` 或 `--steps` 才会在校验/评测之外额外生成。
生成采用贪心策略，去掉 temperature、top-k 和随机采样参数，保持入口紧凑。
词表匹配使用模型提供的 token 文本，未知 prompt 内容报错。

验证集文件为小端 uint32 `N` 与 int32 IDs `[N]`，报告 next-token loss、perplexity 和 top1。
使用长度为模型上下文 T 的非重叠完整窗口（stride=T），忽略不足窗口的尾部。
Python 原训练脚本使用随机采样窗口，因此其整体评测数值不能直接当作逐项正确性判据。

文件读取、API 调用或数值校验失败均返回非零退出码。

## 公共 API 与所有权

版本为 `BNNE_API_VERSION=2`，删除了首版的 `BNNE_Backend` 与 options.backend 字段。
调用顺序：

```text
OptionsInit → Open → GetModelInfo → GetOutputDesc → Run（可重复）→ Close
                                            ↘ GetStats / ResetStats
```

`Open` 自动识别格式，完成校验、上传固定权重和工作区分配。失败时输出句柄置 NULL 并回收已分配资源。
句柄独占设备、权重和工作区；同一句柄上的调用须串行化。不同句柄没有可变全局工作状态。
`Close(NULL)` 合法；不要用 `Open` 覆盖仍需保留的句柄。

张量连续存放、按元素类型对齐，输入输出内存由调用者管理，`bytes` 表示容量。
通过 `GetOutputDesc` 获取实际输出描述，设置 output.desc 后调用同步 `Run`：

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| MLP | FP32 `[in_features]` | FP32 `[out_features]` |
| CNN | FP32 CHW `[1,28,28]` | FP32 `[out_features]` |
| GPT | int32 `[n]`，`1 <= n <= T` | FP32 `[n,V]` |
| LLaMA | int32 `[n]`，`1 <= n <= T` | FP32 `[n,V]` |

### KV Cache 与自回归单步解码

针对生成式语言模型（GPT 与 LLaMA），引擎提供增量单步解码接口：
- `BNNE_ResetKVCache(engine)`：重置当前模型的 KV Cache 游标。
- `BNNE_StepToken(engine, token, pos, out_logits)`：单步解码。将当前 `token` 在位置 `pos` 的 Key 与 Value 写入缓存，Query 仅与缓存中的历史 Key（位置 `0..pos`）进行注意力计算，输出当前位置的 `out_logits`（长度 `[V]`）。
- CLI 工具 `bnn-infer` 支持 `--kv-cache` 开关。开启后，Prefill 阶段将 Prompt 顺序注入缓存，Decode 阶段进行单步增量计算，将自回归解码算力与带宽开销大幅降低，且与全上下文重计算保持严格数值一致。

GPT `GetModelInfo` 中的 input/output 描述是最大序列长度；实际序列由 `GetOutputDesc` 查询。
IDs 须满足 `0 <= id < V`。`GetToken` 返回 GPT 词表的借用 NUL 结尾 UTF-8 字符串，直到 Close 有效。
引擎不承担图像归一化、字符串分词、采样或数据集读取。

输入类型、形状、容量、对齐、有限值或 token ID 检查失败不修改输出，也不累计统计。
计算错误返回状态码；库不会打印报告或终止进程。
Open 失败用 `StatusString` 获取错误类别，句柄有效时可通过 `LastError` 查看运行详情。

内部保留 `load/prepare/run/destroy` 模型操作表，不构建通用算子图或多级会话体系。
CNN 在模型 group 与设备 group 不同时逐组补齐，要求模型 group 不大于设备 group；
GPT 仍要求 head size 等于模型/设备 group。设备维度和 rows 超限时明确返回不支持，不隐式回退。
当前 soft 设备 group=128、最大输入/输出维度=4096，因此 GPT 的词表须不大于 4096，
且 `4*C <= 4096`；CNN 按补齐后的输入维度检查。
仓库中的 `nanogpt-lib.bnn` 可运行，`nanogpt_l3_b48.bnn` 的词表为 6362，会返回 `BNNE_ENOTSUP`。

## 性能统计与验证

`enable_profile` 默认开启，可关闭以减少计时开销。CLI 仅在 `--profile` 时开启分项统计并打印报告。
统计使用单调墙钟：

- `load_ns`：读取模型、上传固定权重、分配工作区，单独记录，reset 保留。
- `total_ns`：成功 Run 的内部总耗时，不含应用层预处理、采样和误差比较。
- `quantize_ns / transfer_ns / execute_ns / host_ns`：量化打包、接口传输、设备执行、其余主机工作。
- 动态注意力矩阵创建/销毁计入 transfer；host 是 total 减去前三项。
- 设备调用数、传输字节、卸载矩阵运算的有效 MAC 与补齐后执行 MAC（不含主机 FP32 attn@V）。

check、eval、generate 分阶段重置统计，不混合累计。软件设备的传输是内存复制与接口开销，
不能直接等同真实 DMA 性能。CNN 补齐后的执行 MAC 会显著高于有效 MAC。

`make test` 使用 Python 标准库生成临时模型、IDX、tokens 与 reference，检查迁移数值、
多模型句柄隔离、动态序列、错误加载/输入和统计。测试不下载数据、不依赖训练，
真实数据精度仍需使用各自训练脚本导出的模型与数据评估。
