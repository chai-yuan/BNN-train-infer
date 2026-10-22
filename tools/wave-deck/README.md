# WAVE/DECK

BNN 顶层握手流观测台。流式读取 Verilator 导出的 VCD，只保留 `bnn_core_top`
对外暴露的信号，把仿真还原成**握手级别的数据流动**，不绘制逐周期波形。

界面风格沿用 Drop-Deploy 的像素机柜语言，并改为日间纸白 / 墨绿配色：
任务卡、数据流画布、握手密度时间线、事件检视和性能统计。

## 快速开始

```sh
cd tools/wave-deck
make serve
# 浏览器打开 http://127.0.0.1:8765/
```

点击「载入示例」可以直接查看随仓库附带的两任务小波形；也可以把任意 `.vcd`
拖入页面，或点击「载入 VCD」。端口被占用时用 `PORT=9000 make serve`。

直接双击 `index.html` 也能解析拖入的 VCD；只有「载入示例」需要通过 HTTP 读取
`samples/demo.vcd`。

## 生成波形

在仓库根目录：

```sh
# 小型专项 DOT/GEMV 波形，输出 build/waves/*.vcd
make -C tests/trace waves

# 加速器差分测试波形，输出 build/waves/accelerator/*.vcd
make -C tests/accelerator waves

# 任意推理命令
mkdir -p build/model-waves
BNN_RTL_TRACE_DIR="$PWD/build/model-waves" ./bnn-infer --profile model.bnn
```

`BNN_RTL_TRACE_DIR` 指向已有可写目录时，每个设备生成一个独立 VCD。完整模型
波形可能很大；先用小型任务调试。

## 观测内容

| 事务 | 触发条件 | 携带信息 |
| --- | --- | --- |
| `CMD` | `cmd_valid && cmd_ready` | k_blocks、group_div、out_chans，推导 K/groups/N 和合法性 |
| `IN` | `act_valid && act_ready` 与 `weight_valid && weight_ready` | 通道号、通道内拍号、32 路 int8 激活、32 位权重、last 标志 |
| `RES` | `res_valid && res_ready` | int32 结果、通道号、组号、`res_chan_last`、`res_last` |
| `DONE` | `done_valid && done_ready` | 状态码、成功/配置错误/中止/协议错误 |

一次 CMD 握手开启一个任务，DONE 握手关闭任务。工具会校验：

- 输入拍数是否符合 `N × k_blocks`；
- 输入 `last` 是否落在通道最后一拍；
- 结果数是否符合 `N × groups`；
- `res_group_idx` 是否连续、`res_chan_last` 与组边界是否一致；
- 成功任务的两路计数是否与命令参数一致；
- 复位中断、运行中中止、孤儿握手、任务未完成等异常。

这些检查只在顶层可见信号上完成，不读取内部状态或参考模型。

### 采样语义

Verilator 在时钟上升沿的同一时间戳里会同时写入时钟跳变和触发器的新值。
如果直接读该时间戳的末态，会看到握手完成后的 `ready=0`，从而漏掉握手。
本工具保留**跳变前已稳定的快照**，用它对 `valid/ready/payload` 判定握手，
再推进到当前时间戳。这个细节也由测试覆盖。

## 界面

- **任务 / COMMANDS**：按 CMD 握手切分；点击聚焦任务，时间线自动缩放到任务窗口。
- **数据流 / HANDSHAKE FLOW**：DRIVER → CORE → SINK 的 CMD、ACT、WGT、RES、DONE
  五条通道；播放时握手包沿通道移动，核心面板显示当前任务的计数和进度。
- **时间线 / DENSITY**：每行是一条通道的握手密度，不是逐周期波形；`BUSY` 显示
  `status_busy` 占空段。滚轮缩放，双击或 `FIT` 适应当前任务/全波形。
- **检视 / INSPECT**：`握手` 页给出事件表与负载（激活 lane、权重 bit、结果）；
  `统计` 页给出全局和当前任务的性能/协议摘要。
- **跟随**：播放时事件列表自动跟随；手动滚动会关闭跟随，点按钮可重新打开。

快捷键：`空格` 播放/暂停，`←/→` 上一个/下一个握手，`Home/End` 首个/末个握手。

## 性能摘要

全局：

- 上升沿数、非复位周期、忙碌周期与占空比；
- CMD、输入、结果、完成握手数量；
- 各通道 `valid && !ready` 停顿计数；
- 时钟周期与波形时长（来自 `$timescale`）。

单任务：

- 周期范围、输入/结果首事件偏移、结果排空滞后；
- 输入拍密度、结果密度、通道完成数；
- 忙周期、完成通知等待周期、停顿计数；
- 期望/实际握手数和协议提示。

## 文件结构

```text
tools/wave-deck/
  index.html
  style.css
  js/vcd.js       流式 VCD 解析、作用域选择、上升沿采样
  js/analyze.js   握手事件、任务模型、校验和统计
  js/gfx.js       像素画布工具
  js/flow.js      数据流画布
  js/timeline.js  握手密度时间线
  js/main.js      状态、交互、列表/检视/统计
  samples/demo.vcd
  test/analyze.test.js
```

解析器与界面解耦：`analyzeVcd(file)` 返回纯数据结构 `trace`，其中包含
`header`、`tasks`、`events`、`markers`、`warnings`、`summary`。
后续接入更多顶层模块或增加性能图表时，可以复用解析/任务层，只替换或增加视图。

## 边界

- 仅支持 VCD；FST 需先转换。
- 仅读取选中作用域的直接变量；当前自动优先 `bnn_core_top`，没有该模块名时选择
  握手接口最完整的顶层作用域。
- 当前要求五组 `valid/ready` 通道都存在；其他模块接口需要增加适配层。
- 事件明细上限为 50 万条；超过后任务计数和统计仍然准确，但输入/结果明细会省略。
- VCD 中若没有时钟上升沿、缺少握手端口或头部损坏，会直接报错。
- 时间线是握手密度和任务窗口，不是门级/周期级时序波形。

## 测试

```sh
cd tools/wave-deck
make test
```

测试覆盖：不规则分块流式解析、作用域自动选择、预跳变握手采样、任务参数/计数、
激活 lane 解码、复位期间屏蔽握手、非法配置完成、运行中 abort，以及示例波形计数。
字体使用 Fusion Pixel（SIL Open Font License 1.1，许可证见
`assets/fonts/OFL-fusion-pixel.txt`）。
