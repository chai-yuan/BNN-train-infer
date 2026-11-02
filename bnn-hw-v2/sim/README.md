# BNN v2 Verilator / DPI-C 仿真与验证层

本目录为 `../bnn_core/` 提供独立于具体片外总线（AXI4 等）的 Verilator + DPI-C 仿真环境、命令行运行入口及完整回归测试套件。所有 RTL 与仿真模型均开启 `-Wall --assert` 严格检查，无告警豁免。

## 结构

```text
C++ HostDriver -- cmd_* --> bnn_sim_top / bnn_core_top
                                      |
                              统一读/写 DMA 接口
                                      |
                               bnn_dma_memory.sv
                                      |
                                    DPI-C
                                      |
                     +----------------+----------------+
                     |                                 |
            memory(0): 读源存储器              memory(1): 写目的存储器
              权重 / FP32 / INT8                    结果 / 保护区
```

两个存储器是独立的 C++ 字节数组，即使地址相同也不共享内容。读、写状态机可并行执行；这不是 AXI 总线仿真，也不是片上 Weight/Output SRAM 的替代品。`bnn_core_top` 内部的 Weight SRAM 与 Output Post SRAM 均由可综合 RTL 提供。

| 文件 | 用途 |
| --- | --- |
| `bnn_dma_memory.sv` | ready/valid DMA 适配、可配置延迟/间隔/背压、协议断言与 DPI-C 调用 |
| `include/virtual_memory.hpp`, `virtual_memory.cpp` | 可动态配置容量/基址的两个字节存储器、掩码写、二进制文件加载/导出 |
| `bnn_sim_top.sv` | 将默认配置的 `bnn_core_top` 连接到 `bnn_dma_memory`，C++ 不直接搬运 DMA 信号 |
| `include/host_driver.hpp` | 64-bit ISA 编码、指令提交、完成/错误脉冲检测、分阶段超时控制 |
| `include/simulation.hpp` | 单周期时钟步进、同步高有效复位、可选 VCD 波形导出与 RAII 清理 |
| `main.cpp` | `bnn_core_top` 命令行仿真器（支持命令文件、二进制内存加载/导出与波形生成） |
| `fp_math_test.sv`, `fp_math_test.cpp` | `bnn_fp_math` 算术包组合逻辑穷举/定向/随机参考回归 |
| `core_test.cpp` | `bnn_core_top` 端到端指令、2D 步长/尾掩码、INT8/FP32 量化、跨块累加与在途复位回归 |
| `dma_memory_test.cpp` | 独立驱动 `bnn_dma_memory` 的时序、边界与随机压力回归 |
| `host_driver_test.cpp` | 基于 fake-DUT 的宿主驱动握手与超时回归 |
| `test_protocol_errors.py` | 隔离进程运行故意违反 DMA 协议的负测，检查具体断言致命信息 |
| `test_cli.py` | CLI 参数校验、独立存储、VCD、INT8/FP32 真实指令序列与 `cmd_error` 退出码测试 |

## 构建与运行

依赖：Verilator 5.x、支持 C++17 的编译器、GNU make 与 Python 3（Linux）。无第三方 C++/Python 依赖。C++ 数值测试固定开启 `-fno-fast-math -ffp-contract=off`。

在项目根目录运行：

```sh
# 完整默认回归：宿主驱动 + FP 算术 + 128-bit DMA/负测 + 核心端到端 + CLI 集成测试
make -C bnn-hw-v2/sim test

# 单独运行算术包或核心端到端回归
make -C bnn-hw-v2/sim fp-test
make -C bnn-hw-v2/sim core-test
make -C bnn-hw-v2/sim core-smoke

# 验证 DMA 模型的 64/128/256-bit 数据通路（含 >4 GiB 高地址与协议负测）
make -C bnn-hw-v2/sim test-widths

# 严格 lint：覆盖 DMA、fp_math_test、bnn_core_top（仿真模式 + SYNTHESIS 模式）及 bnn_sim_top
make -C bnn-hw-v2/sim lint
make -C bnn-hw-v2/sim lint-core

# 构建并使用 CLI 仿真器
make -C bnn-hw-v2/sim core
make -C bnn-hw-v2/sim run ARGS="--help"
make -C bnn-hw-v2/sim run ARGS="--idle 32 --trace build/core-idle.vcd"
```

所有编译产物仅生成在 `sim/build/` 下；`make clean` 仅删除该目录。可通过 `VERILATOR=... CXX=... JOBS=...` 覆盖工具链配置。

### 命令文件与二进制内存

命令文件每行一个 64-bit **十六进制**指令，可带 `0x` 前缀，支持空行与 `#` 注释。例如运行单行 INT8 任务（`N=5, M=1`）：

```text
0000000000000001 # 源 stride=0
0000000000000002 # 目的 stride=0
0000100000050010 # 从 0x1000 加载 N=5 权重
0000200002000120 # 从 0x2000 计算 M=1，INT8 直通，覆盖写
0000300000000030 # 写回 0x3000
```

```sh
make -C bnn-hw-v2/sim run ARGS="--commands commands.hex --load 0x1000 weights.bin --load 0x2000 input.bin --dump 0x3000 32 output.bin --trace build/task.vcd"
```

相对文件路径相对于进程当前工作目录（使用 `make -C bnn-hw-v2/sim` 时即 `sim/`）。`--load` 只写读源存储器，`--dump` 只读写目的存储器，均可重复指定。每个存储器默认容量 1 MiB、模拟基址 0，可通过 `--memory-size`、`--read-base`、`--write-base` 调整；模拟地址不是宿主指针。

默认 128-bit DMA 下：5 条权重紧密排列占 90 字节，尾拍读取到 96 字节（每条依次为 2 字节小端 FP16 scale 和 16 字节符号位）；单行 INT8 输入占 128 字节（范围 `[-127, 127]`）；输出占 2 拍共 32 字节，通过尾拍字节掩码仅改写前 5 个 FP32（20 字节），尾部保持原值。软件放置下一块权重时须将其首地址对齐到 16B。

CLI 的地址、大小和超时参数支持十进制或 `0x` 十六进制。`--accept-timeout` 默认 1000 拍，`--timeout` 默认 100000 拍。遇到 `cmd_error`、超时、DPI 越界或文件错误时程序均以非零状态退出，并在结束时检查核心与 DMA 均已回到空闲状态。

## C++ 动态数据接口

可在自定义 C++ 测试中直接驱动 `Vbnn_sim_top`，无需重新编译 RTL：

```cpp
#include "virtual_memory.hpp"
#include "host_driver.hpp"
#include "simulation.hpp"
using namespace bnn::sim;

memory(kReadMemory).configure(1 << 20, 0);          // size, simulated base, fill=0
memory(kWriteMemory).configure(1 << 20, 0, 0xa5);   // sentinel 检查尾部与行间保护字节
memory(kReadMemory).write(0x2000, input_bytes);    // vector<uint8_t>，按小端打包
memory(kReadMemory).load_binary("weights.bin", 0x1000);

Simulation<Vbnn_sim_top> simulation;
simulation.reset();
HostDriver driver(simulation.dut, [&] { simulation.tick(); });
auto result = driver.execute(Instruction::load_weights(0x1000, 5));
// 对每条指令检查 result == CommandResult::done
result = driver.execute(Instruction::compute(0x2000, 1, false, true));
result = driver.execute(Instruction::writeback(0x3000));
auto output_bytes = memory(kWriteMemory).read(0x3000, 32);
```

`Instruction` 提供 `set_source_stride`、`set_destination_stride`、`load_weights`、`compute`、`writeback`，检查 16B 对齐和 `M/N` 的 `1..255` 编码范围；原始非法编码可直接传入 `driver.execute(uint64_t)` 做负测。输出不会自动回流到读源存储器，多层串联时显式复制即可：

```cpp
memory(kReadMemory).write(next_input_address,
                         memory(kWriteMemory).read(output_address, byte_count));
```

## 固定的 DMA 仿真协议

| 项目 | 本环境的约定 |
| --- | --- |
| 时钟与复位 | 上升沿握手，同步高有效 `reset`；`reset` 时所有 `ready` 与 `busy` 为 0 |
| 地址 | 字节地址，必须对齐到每拍字节数（默认核心每拍 16 字节） |
| 长度 | 实际拍数 `1..2^DMA_LEN_WIDTH-1`，不是 AXI `LEN=beats-1`，0 非法 |
| 字节序 | 地址 `addr+i` 对应 `data[8*i +: 8]` |
| 在途数量 | 读、写各最多一笔，无 transaction ID；两个方向独立 |
| 命令 | `valid && ready` 锁存地址/拍数；背压时 `valid` 与全部 payload 保持稳定 |
| 读数据 | `valid` 不依赖 `ready`；未握手不得推进地址、计数、`data` 或 `last` |
| 写数据 | 仅在数据握手沿更新 C++ 内存；`strb[i]` 控制第 `i` 字节；全零 `strb` 合法 |
| `last` | 必须且仅在命令指定的最后一拍为 1，包括单拍事务 |
| 写完成 | 末拍握手后延迟返回单周期 `mem_wr_done`；收到完成前不接下一写命令 |
| 范围 | 命令接收时先检查完整 burst 范围，尾拍即使部分掩码也须映射完整一拍 |
| 复位 | 丢弃未完成读、未接受写拍和待返回 `done`，不回滚已接受写、不清空 C++ 内存 |
| 错误 | 仿真环境协议/越界错误由 DPI 错误信息 + SV `$fatal` 立即终止，不伪造 `cmd_error` |

### 可配置时序

在 CLI 中通过命令行选项设置，在 C++ 测试中直接配置 `dut.cfg_*`（应在事务空闲时修改）：

- `cfg_rd_latency` / `--rd-latency`：默认 2。读命令在沿 `T` 接收，首拍 `valid` 在沿 `T + latency + 1` 后出现。
- `cfg_rd_gap` / `--rd-gap`：默认 0。相邻读拍之间额外插入的无效周期数；0 支持连续逐拍消费。
- `cfg_wr_latency` / `--wr-latency`：默认 2。末拍在沿 `T` 接收，`mem_wr_done` 在沿 `T + latency + 1` 后出现并保持一周期。
- `cfg_cmd_stall_period` / `--cmd-stall-period`：默认 0；`N>0` 时每 `N` 拍拉低一次命令 `ready`，`N=1` 为永久背压。
- `cfg_wr_stall_period` / `--wr-stall-period`：同上，仅控制写数据 `ready`。

## 回归覆盖范围与边界

当前回归套件已覆盖：

- **宿主驱动 (`host_driver_test.cpp`)**：指令编码、对齐/维度异常、背压稳定保持、沿前/沿后/延迟单拍 `done/error`、错误优先级、连续指令与分阶段超时。
- **浮点与量化算术 (`fp_math_test.cpp`)**：全部 65536 种 FP16 转 FP32、INT32 转 FP32、FP32 FTZ/RTZ 乘加（含 Knuth TwoSum 大指数差精确抵消）、除以 127 及定点 INT8 量化阈值边界。
- **DMA 传输与协议负测 (`dma_memory_test.cpp`, `test_protocol_errors.py`)**：64/128/256-bit 位宽、小端序、读写并发、首拍延迟与拍间空泡、稀疏/全零写掩码、保护字节、`>4 GiB` 地址、在途复位及 11 类预期协议违例致命断言。
- **核心端到端与 CLI (`core_test.cpp`, `test_cli.py`)**：非法指令/未对齐/越界/保留标志位/状态违例拒绝（含首次计算前 `WRITEBACK`/`accum`、跨 `M` 累加不兼容、换 `N` 使旧结果失效）、INT8 与 FP32 动态量化 2D 步长读写、尾列字节掩码保护、同 `N` 重载权重跨块累加、在途计算/DMA 复位恢复及 CLI 命令文件执行。

本目录不覆盖外层 AXI4 突发分拆、4KiB 边界处理及 AXI `RRESP/BRESP` 总线错误恢复（留待外层 `bnn_top.sv` 总线适配层验证）。
