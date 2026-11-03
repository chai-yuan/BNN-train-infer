# BNN v2 仿真与固定回归用例

本目录用 Verilator + DPI-C 驱动 `bnn_core/`：`simple_test_main.cpp` 从两个文件读入
命令流和初始内存镜像，仿真结束后写出 `memory_out.img`，并生成周期级 profile。

```sh
make            # 构建 build/Vbnn_sim_top
make test       # 运行 tests/case 并用 golden 容差校验
make run        # 只运行 tests/case
make check      # 校验上一次运行结果
make gen        # 从 Qwen3-zh-5k GGUF 重新生成 tests/case（需要 numpy）
make clean
```

`make test` / `make check` 只读取已提交的 `tests/case`，不需要模型文件；`make gen` 默认
读取仓库根目录的 `qwen3-zh-5k.gguf`，可用 `MODEL=/path/to/model.gguf` 覆盖。

## 固定用例 `tests/case`

用例来自 Qwen3-zh-5k 的第 0 层 `blk.0.attn_q.weight`（K=512、N=512、G=128，
Q1_0 二值权重 + FP16 scale），M=16 行激活由 token embedding 与 `blk.0` RMSNorm
生成。它通过 bnn-accelerator v2 的 `BNN_OP_GEMM` / `BNN_OP_GEMM_F32` 契约定义三个负载：

| 负载 | 输入 | K | 说明 |
| --- | --- | --- | --- |
| `full_i8` | INT8 | 512 | 4 个 K 块累加，覆盖 4 个 N 块 |
| `tile_i8` | INT8 | 128 | 第一个 K tile |
| `tile_f32` | FP32 | 128 | 同一 tile，输入为 INT8 值乘精确的 `2^-6` |

FP32 输入取 `Q * 2^-6`，因此整行 absmax 量化在硬件与 bnn-accelerator 中都精确还原
同一个 `Q`；`tile_f32` 的输出应等于 `tile_i8 * 2^-6`。v2 RTL 按每个 128 特征
COMPUTE tile 独立量化，K>128 的 FP32 任务不符合 API 的“完整逻辑行量化”语义，
所以 FP32 负载固定为 K=128。

需要强调，per-tile absmax 是**更细的量化粒度**，不等于严格的“更高精度”。当不同 tile
的动态范围差异较大时，它可以降低单个 tile 的量化重构误差；但 RTL 的量化采用向零截断、
scale 除法截断，FP32 计算采用 FTZ + RTZ，均比 bnn-accelerator 的 round-half-away / RNE
更粗。K>128 时它属于另一种块缩放语义，最终 GEMM 结果可能更接近、也可能更远离全行量化
或精确 FP32 结果，不能笼统称为精度更高。K=128 时量化粒度相同；本用例的 FP32 输入是
精确 dyadic 值，两种量化还原出同一个 `Q`，剩余差异仅为 FP32 舍入。

`memory.img` 包含权重、三组输入和预填充 `0xA5` 的输出区；`commands.hex` 是 64 条
64-bit 指令。`golden_*.hex` 是 bnn-accelerator v2 语义下的期望结果（FP32 RNE、
FP16 scale 精确转换、逐组乘积单独舍入、按组递增累加、无 FMA），已与
`bnn-accelerator/soft-bnn-accelerator.c` 逐位交叉验证。

## 容差与基线

`meta.json` 记录地址、负载、golden 文件和容差。校验使用

```text
|got - golden| <= atol + rtol * |golden|
atol = 1e-5, rtol = 1e-6
```

当前 RTL 的 FP32 路径采用 FTZ + RTZ，而 golden 采用 RNE；实测最大相对误差
`1.45e-7`（`full_i8`，2 ulp），`tile_i8` / `tile_f32` 逐位一致。

`baseline_cycles.txt` 记录当前 RTL 的基线总周期（1793055）。`make test`
会打印本次周期数与相对基线的变化，便于后续性能分析；周期变化不会导致失败；`make gen`
不会覆盖该文件。
