# BNN 硬件加速器 v2 (`bnn-hw-v2`)

本目录包含 BNN v2 计算核心的可综合 SystemVerilog RTL 实现（`bnn_core/`）、基于 Verilator + DPI-C 的完整仿真与回归环境（`sim/`），以及待接入片外 AXI4 总线的顶层骨架（`bnn_top.sv`）。

## 目录与模块划分

```text
bnn-hw-v2/
├── bnn_core/                  # 协议无关的 BNN v2 计算核心 RTL
│   ├── bnn_fp_math.sv         # 可综合浮点/量化算术包（FP16->FP32、FTZ/RTZ 乘加、除 127、INT8 量化）
│   ├── bnn_core_top.sv        # 计算核心顶层：组装 6 个功能子模块、Weight SRAM 与读 DMA MUX
│   ├── bnn_top_controller.sv  # 64-bit 单阶段原子指令控制器与全局状态机
│   ├── bnn_weight_loader.sv   # 紧密排列外部权重突发读取、跨拍解包、自动补零写入 Weight SRAM
│   ├── sram_weight_wrapper.sv # Weight SRAM 浅包装（单端口 1RW SRAM 宏，默认 128 x 144-bit）
│   ├── bnn_input_quant_buffer.sv # 2D 步长输入读取、可选 FP32 行动态量化 / INT8 直通缓冲
│   ├── bnn_core_array.sv      # 多通道计算阵列（单行激活锁存、按组读取权重、广播提交与同步收集）
│   ├── bnn_core_unit.sv       # 单通道计算单元（有符号 INT8 二值点积 + FP16 权重缩放 + 可选行反量化）
│   ├── sram_result_wrapper.sv # Result SRAM 浅包装（带字节写掩码的单端口 1RW SRAM 宏，默认 4096 x 128-bit）
│   ├── bnn_output_accum.sv    # 结果 SRAM 访问层：跨 K 块 FP32 覆盖/累加后处理与 session 握手
│   └── bnn_output_writer.sv   # Output SRAM -> 外部内存 2D 步长突发写回与尾列字节掩码生成
├── sim/                       # Verilator + DPI-C 仿真层、CLI 运行器与全套回归测试
└── bnn_top.sv                 # 外层 AXI4 + 128-bit 指令接口顶层封装骨架（尚未接入内部引擎）
```

## 硬件设计 (`bnn_core/`)

### 顶层数据流与存储边界

`bnn_core_top` 对外仅暴露三组与具体片外总线解耦的接口：

1. **宿主机 64-bit 原子命令接口**：`cmd_valid / cmd_inst / cmd_ready / cmd_done / cmd_error / status_busy`。
2. **统一外部内存突发读接口**：`mem_rd_cmd_*` 与 `mem_rd_data_*`（默认 128-bit 数据，长度 `len` 为实际拍数）。
3. **统一外部内存突发写接口**：`mem_wr_cmd_*`、`mem_wr_data_*`（含 16-bit 字节掩码 `strb`）与 `mem_wr_done`。

片上包含两块同步 SRAM。两块存储体都是对 FPGA Block RAM / ASIC 常见**单端口 1RW**（`en / we / addr / wdata / rdata`，同一拍只能读或写）SRAM 宏的**浅包装**：内部只有存储阵列与同步读输出流水，不含控制状态机、不复位清零、不做读写仲裁；文件内以 `VENDOR_SRAM_MACRO_BEGIN/END` 标出可直接替换为厂商宏（RAMB36E1 / altsyncram / ASIC SRAM 编译器输出）的行为模型。

- **Weight SRAM（`sram_weight_wrapper.sv`，默认 `128 × 144-bit`，`READ_LATENCY = 1`）**：最多容纳 `MAX_OUT_CHANS = 128` 条权重（分为 `MAX_WEIGHT_GROUPS = 32` 组，每组 `ODMA_FP32_NUM = 4` 通道）。`OP_LOAD_WEIGHT`（写）与 `OP_COMPUTE`（读）由控制器时分互斥，`bnn_core_top` 用 `en/we/addr` MUX 合流到唯一端口，无仲裁等待、无吞吐损失。
- **Result SRAM（`sram_result_wrapper.sv`，默认 `4096 × 128-bit`，带字节写掩码 `wstrb`）**：最多暂存 `MAX_ROWS × MAX_WEIGHT_GROUPS = 128 × 32` 拍（即 `128 × 128` 个 FP32）分块结果。存储体本身不含逻辑，覆盖/跨块累加与 session 握手由访问层 `bnn_output_accum.sv` 实现并在其中完成单端口合流：累加时 `RECEIVE` 拍发起读、`WRITE_BACK` 拍写回（天然错拍），`OP_WRITEBACK` 期间端口让给 `bnn_output_writer` 顺序读取。
- **DMA 读通道多路复用 (`dma_rd_mux_sel`)**：由控制器在确认在途读事务与子模块完全排空（`children_idle`）后静态切换给 `bnn_weight_loader`（`0`）或 `bnn_input_quant_buffer`（`1`）。

### 数据布局与算术契约 (`bnn_fp_math`)

所有 RTL 模块均不使用浮点厂商 IP、`real` 或 `shortreal`：

- **权重记录格式**：默认 `IDMA_INT8_NUM = 128`、`WEIGHT_WIDTH = 144`（18 字节/通道）。外部内存中 `N` 条权重按小端紧密排列、无通道间填充：低 2 字节为 IEEE 754 binary16 `scale`，随后 16 字节为 128 个二值权重位（`1` 表示 `+1`，`0` 表示 `-1`）。`bnn_weight_loader` 支持任意跨 16B 拍边界解包，并在末尾自动补零至 `ODMA_FP32_NUM (4)` 的整数倍。
- **INT8 直通输入模式 (`input_is_int8 = 1`)**：每行读取 128 个有符号 INT8（128B，软件保证范围 `[-127, 127]`），不执行动态量化，单元关闭行反量化乘法（`idma_dequant_en = 0`）。
- **FP32 行动态量化模式 (`input_is_int8 = 0`)**：每行读取 128 个 FP32（512B）：
  1. 次正规数（`exp == 0`）与 `NaN/Inf`（`exp == 8'hff`）在扫描与量化时视为 `0`，求整行有限正规数最大绝对值 `row_max_abs`。
  2. 若 `row_max_abs == 0`，整行量化输出全 `0` 且 `idma_scale = +0.0`。
  3. 若 `row_max_abs > 0`，反量化系数 `idma_scale = fp32_div127_ftz(row_max_abs)`；各元素通过指数差对齐与尾数定点乘除 `(1.m_i * 127) / (1.m_max << (E_max - E_i))` 向零截断量化至 `[-127, +127]`，避免先求倒数导致的浮点上溢或舍入偏差。
- **点积、缩放与跨块累加**：
  - `bnn_core_unit` 计算精确整数点积 $D = \sum_{k=0}^{127} q_k w_k$，转为 FP32 后乘 `fp16_to_fp32(w_scale)`；若 `idma_dequant_en = 1`，再乘行反量化系数 `idma_scale`。
  - FP16 转 FP32 精确保留 FP16 次正规数；所有 FP32 乘法与 Output SRAM 跨块加法均采用 **输入/输出次正规数冲零（FTZ）+ 有限结果向零截断（RTZ）**，任何 `NaN`（含 `0 × Inf`、`+Inf + -Inf`）统一输出规范静默 NaN `0x7fc00000`。
- **2D 行步长与尾列掩码**：
  - 源/目的基址与行步长均须 16B 对齐；步长为 `0` 时硬件自动按单行整拍空间紧密步进（INT8 输入 128B、FP32 输入 512B、FP32 输出 `ceil(N/4) * 16B`）。
  - `bnn_output_writer` 写回每行尾拍时自动按有效列数 `N` 生成字节写掩码 `strb`，不覆盖尾部无效通道与行间填充区。

## 指令设计

`bnn_top_controller` 采用串行单阶段原子 64-bit 指令集（无内部指令队列）：

```text
  63                              32 31      24 23          16 15           8 7        0
 +----------------------------------+----------+--------------+--------------+----------+
 |     addr / row_stride (32b)      |  flags   |  N (rows_n)  |  M (rows_m)  |  opcode  |
 +----------------------------------+----------+--------------+--------------+----------+
```

| 指令名 | `opcode` | 参数字段 | 功能说明 |
| --- | --- | --- | --- |
| `OP_SET_SRC_STRIDE` | `8'h01` | `[63:32] src_row_stride` | 配置输入相邻行字节步长（16B 对齐，`0` 表示紧密排列），单拍完成 |
| `OP_SET_DST_STRIDE` | `8'h02` | `[63:32] dst_row_stride` | 配置输出相邻行字节步长（16B 对齐，`0` 表示按 `G*16B` 紧密排列），单拍完成 |
| `OP_LOAD_WEIGHT` | `8'h10` | `[63:32] addr`, `[23:16] N` | 从 `addr` 加载 `N`（`1..128`）条权重至 Weight SRAM 并刷新组数 `G = ceil(N/4)` |
| `OP_COMPUTE` | `8'h20` | `[63:32] addr`, `[15:8] M`, `[24] accum`, `[25] int8` | 从 `addr` 读取 `M`（`1..128`）行输入计算 `M×N` 分块，覆盖（`accum=0`）或累加（`accum=1`）至 Output SRAM |
| `OP_WRITEBACK` | `8'h30` | `[63:32] addr` | 按最近一次有效计算的 `(M, N)` 与目的步长将 Output SRAM 结果写回 `addr` |

### 握手、状态合法性与复位约定

1. `cmd_ready` 不依赖 `cmd_valid`；每条握手接收的指令恰好产生一次单周期 `cmd_done` 或 `cmd_error`（二者严格互斥）。
2. 未知 `opcode`、地址/步长未 16B 对齐、`M/N` 为 `0` 或超出容量上限、`OP_COMPUTE` 保留标志位 `[31:26]` 非零、未加载权重时发起 `OP_COMPUTE/OP_WRITEBACK`、无有效结果或 `M` 不一致时发起 `accum=1` 计算，均拒绝执行并返回 1 拍 `cmd_error`，且不破坏此前已保存的合法配置与有效累加结果。
3. **跨块累加生命周期**：以相同 `N` 重新执行 `OP_LOAD_WEIGHT` 会保留 Output SRAM 结果有效位，供沿 `K` 维分块 `OP_COMPUTE(accum=1)` 累加；若新加载的 `N` 发生变化，旧结果立即标记失效，后续必须先执行 `accum=0` 的首块计算方可累加或写回。
4. **同步高有效复位 (`reset`)**：立即终止控制器及所有子模块在途状态、清零步长/维度/有效位并拉低所有对外请求、`ready`、`busy` 与完成脉冲；不逐单元清零片上 SRAM 存储阵列。

### 2D 分块矩阵乘法指令序列示例

计算分块 $A_{11} = B_{11} C_{11}^\top + B_{12} C_{21}^\top$（FP32 动态量化输入，$M=128, N=128, K_{\text{tile}}=128$）：

1. `OP_SET_SRC_STRIDE`：`src_row_stride = K_total * 4`
2. `OP_SET_DST_STRIDE`：`dst_row_stride = N_total * 4`（须 16B 对齐）
3. `OP_LOAD_WEIGHT`：`addr = &C_11, N = 128`
4. `OP_COMPUTE`：`addr = &B_11, M = 128, accum = 0, int8 = 0`
5. `OP_LOAD_WEIGHT`：`addr = &C_21, N = 128`（同 `N`，保留部分和有效）
6. `OP_COMPUTE`：`addr = &B_12, M = 128, accum = 1, int8 = 0`
7. `OP_WRITEBACK`：`addr = &A_11`

## 仿真验证

[`sim/`](sim/README.md) 当前提供固定 Qwen3 W1A8 回归用例：取自 Qwen3-zh-5k 第 0 层
`blk.0.attn_q.weight`（K=512、N=512、G=128），M=16，包含

- `full_i8`：`BNN_OP_GEMM`，K=512，跨 4 个 K 块累加；
- `tile_i8`：`BNN_OP_GEMM`，K=128；
- `tile_f32`：`BNN_OP_GEMM_F32`，K=128，FP32 输入为 INT8 值的精确 `2^-6` 缩放。

`commands.hex` 与 `memory.img` 可直接输入 `simple_test_main.cpp`；`golden_*.hex` 按
bnn-accelerator v2 语义生成，并与 soft 后端逐位交叉验证。校验容差为
`atol=1e-5, rtol=1e-6`，当前 RTL 实测最大相对误差 `1.45e-7`；`baseline_cycles.txt`
记录当前基线总周期，供后续性能对比。

```sh
make -C bnn-hw-v2/sim test    # 构建、运行并校验固定用例
make -C bnn-hw-v2/sim run     # 只运行
make -C bnn-hw-v2/sim check   # 校验上一次运行结果
make -C bnn-hw-v2/sim gen     # 从 Qwen3-zh-5k GGUF 重新生成用例（需要 numpy）
```

详细说明见 [`sim/README.md`](sim/README.md)。注意 v2 RTL 按每个 128 特征 COMPUTE tile
独立量化，K>128 的 FP32 任务不符合 bnn-accelerator 的完整逻辑行量化语义，因此 FP32
负载固定为 K=128。

## 备忘录

- **当前微架构阶段**：`bnn_core/` 当前采用结构清晰的串行基线实现（如 `bnn_core_unit` 逐周期 1 元素点积、`bnn_core_array` 单组串行读权重、`bnn_weight_loader` 逐字节解包、`bnn_input_quant_buffer` 寄存器行缓冲与组合除法、`bnn_fp_math` 全精度 278-bit 定点对齐加法），用于锁定指令集、DMA 边界和端到端数值参考的一致性。
- **后续演进方向**：
  1. **吞吐与面积/时序优化**：将 `bnn_core_unit` 升级为平衡组合/流水加法树，`bnn_core_array` 引入双组权重乒乓预取，`bnn_weight_loader` 改为拍宽并行拼接，`bnn_input_quant_buffer` 用行 SRAM 替代大位宽寄存器 MUX 并多拍化量化除法，收敛 `fp32_add_ftz` 移位位宽。
  2. **外层总线与软件栈对接**：在 `bnn_top.sv` 中实现 AXI4 Master 突发拆分、4KiB 边界处理与总线错误上报，并基于 `sim/` 或设备驱动对接 `bnn-accelerator` v2 后端跑通完整模型推理。
