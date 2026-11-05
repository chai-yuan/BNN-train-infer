# `bnn-hw-v2/sta` — 时序与面积分析

本目录用 **Yosys + ICS55 PDK + OpenSTA** 对 `bnn_core/` 做 post-synthesis 的
Fmax 与面积分析，并自动生成 Markdown 报告。目标是快速回答：

- 当前 RTL 在 55nm H7C 标准单元下的大致 Fmax 是多少；
- 最差 corner 是哪一条路径（startpoint / endpoint / slack）；
- 综合后的标准单元数、DFF 数、组合逻辑和面积；
- 改 RTL 或换库后，Fmax / 面积如何变化。

## 依赖

本机已安装到 `~/.local/bin`：

```sh
yosys   0.70
sv2v    v0.0.13
sta     OpenSTA 3.1.0
```

OpenSTA 和 sv2v 是本次新安装的；`yosys` 沿用原来的。`python3` 用于生成报告。
PDK 默认路径为 `/home/charain/project/icsprout55-pdk`，可用 `PDK=...` 覆盖。

若换机器，安装方式：

```sh
# sv2v：下载 sv2v-Linux.zip，把 sv2v 放到 ~/.local/bin
# OpenSTA：从 https://github.com/parallaxsw/OpenSTA 构建
#   cmake -S OpenSTA -B OpenSTA/build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_INSTALL_PREFIX=$HOME/.local -DUSE_TCL_READLINE=OFF
#   cmake --build OpenSTA/build -j
#   cmake --install OpenSTA/build
# 构建依赖：tcl-dev、swig、bison、flex、eigen3、cudd、cmake、g++
```

## 快速开始

```sh
cd bnn-hw-v2/sta

make            # 综合 + 所有默认 corner + 报告（默认 H7CR RVT）
make report     # 同上
make synth      # 只做 sv2v + Yosys 映射，生成 build/ 下的网表和 synth.log
make sta        # 只跑 OpenSTA（复用已有网表）
make summarize  # 只重新生成报告（用 RUN=latest 读取最近一次运行）
make clean      # 删除 build/
make distclean  # 删除 build/ 和 reports/
```

常用覆盖：

```sh
# 换成 LVT（更快、漏电更高）
make FAMILY=H7CL

# 只跑最差 slow corner，缩短 STA 时间
make CORNERS="ss_rcworst_1p08_125"

# 换 probe 时钟周期 / 不确定度 / 报告路径数
make CLK_PERIOD=80 CLK_UNC=0.2 TOP_N=10

# 指定报告目录，便于对比
make RUN=before-rtl-change
```

每次运行输出到 `reports/<RUN>/`：

```text
reports/<RUN>/
├── report.md            # 汇总报告（Fmax、面积、热点路径）
├── summary.csv          # 机器可读的 corner 汇总
├── sta_<corner>.json    # OpenSTA JSON（完整路径数据）
├── sta_<corner>.rpt     # 人读的 full_clock_expanded 路径
└── sta_<corner>.log     # OpenSTA 诊断日志
```

综合中间产物在 `build/`：`<TOP>_conv.v`（sv2v 输出）、`<TOP>_sta.v`（网表）、
`synth.log`（含 `stat -liberty` 面积）。

## 流程说明

1. **sv2v**：Yosys 0.70 不能解析 `bnn_fp_math` 的 SV package，先转换为
   Verilog-2005。默认定义 `SYNTHESIS` 和 `TARGET_ASIC`。
2. **Yosys**：`synth -flatten`、DFF 映射、ABC 映射到 ICS55 H7C 标准单元。
   ABC 加 `-constr`（`set_driving_cell`/`set_load`）启用 buffer/upsize，
   否则高扇出控制网会被映射成单格驱动上千负载，产生假的 100ns+ 路径。
3. **面积**：`stat -liberty` 输出标准单元数、DFF 数、`Chip area`。
4. **OpenSTA**：每个 corner 一个 `sta` 进程，约束 `clock` 端口、周期和 clock
   uncertainty；输出 JSON + full 报告。
5. **summarize.py**：解析 `synth.log` 和 JSON，生成 `report.md` / `summary.csv`。

## 关键变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PDK` | `/home/charain/project/icsprout55-pdk` | PDK 根目录 |
| `TOP` | `bnn_core_top` | 综合顶层 |
| `FAMILY` | `H7CR` | `H7CR` RVT / `H7CL` LVT / `H7CH` HVT |
| `MAP_CORNER` | `ss_rcworst_1p08_125` | ABC 映射使用的 Liberty corner |
| `CORNERS` | typ + 3 个 ss | STA 扫描的 corner 后缀 |
| `CLK_PORT` | `clock` | 时钟端口名 |
| `CLK_PERIOD` | `100.0` | probe 周期（ns），用于从 slack 外推 Fmax |
| `CLK_UNC` | `0.1` | clock uncertainty（ns） |
| `TOP_N` | `10` | 报告中列出的关键路径数 |
| `IO_DELAY` | 空 | 可选输入/输出 delay（ns） |
| `OUT_LOAD` | 空 | 可选输出负载（pF） |
| `RUN` | 时间戳 | 报告目录名 |

Fmax 的计算：

```text
Fmax = 1 / (CLK_PERIOD - slack)
```

`slack` 已包含 clock uncertainty、library setup time 和（可选的）IO delay。
`CLK_PERIOD` 只是 probe 值；一般设为明显大于关键路径的值即可。

## Corner 选择

Setup（Fmax）应至少覆盖：

- `typ_tt_1p2_25`：典型，用于 nominal 参考；
- `ss_cworst_1p08_m40`：慢管、低电压、-40°C；
- `ss_rcworst_1p08_125`：慢管、低电压、125°C（默认最差）；
- `ss_rcworst_1p2_m40`：慢管、1.2V、-40°C。

最终 signoff 取最差 corner。Hold 检查需要另跑 FF corner；本脚本目前只做 setup。

## Baseline（2026-11-05，H7CR，`TARGET_ASIC` SRAM 占位）

| Corner | Slack @100ns | Fmax |
| --- | ---: | ---: |
| `typ_tt_1p2_25` | 80.16 ns | 50.4 MHz |
| `ss_rcworst_1p2_m40` | 75.82 ns | 41.4 MHz |
| `ss_cworst_1p08_m40` | 63.49 ns | 27.4 MHz |
| `ss_rcworst_1p08_125` | 63.03 ns | **27.0 MHz** |

- 关键路径：`u_input_quant_buffer.sample_index` → `row_storage` 的 128 选 1
  MUX → `quantize_int8_ftz` 的 31-bit 除法 → `quantized_row`。
- 逻辑面积约 **292,198 µm²**，137,788 个标准单元，13,934 个 DFF。
- 这些数字会随 RTL、`FAMILY`、`MAP_CORNER` 和综合脚本变化；请以本次运行生成的
  `reports/<RUN>/report.md` 为准。`reports/latest` 指向最近一次报告。

## 重要限制

- **Post-synthesis only**：理想时钟、没有布线延迟、CTS、OCV。真实 Fmax 会更低。
- **SRAM 是占位模型**：`TARGET_ASIC` 分支只是一个寄存器，不是 128×144 /
  4096×128 的真实 SRAM。真实 SRAM 宏会引入 clk→Q、setup 和额外面积；请用 ICS55
  SRAM Compiler 生成 `.lib/.lef/.gds` 后替换 wrapper，再跑本流程。
- **面积只含逻辑**：SRAM、IO、电源、时钟树面积不在其中。
- **不是 signoff**：需要 signoff 时用 OpenROAD/OpenSTA 做 place/CTS/route 后的
  多 corner STA，并加入 SPEF、OCV/AOCV、真实 SDC。
- OpenSTA 的 JSON 报告和 `report_checks > file` 需要较新的 OpenSTA；本目录按
  OpenSTA 3.1.0 编写。
