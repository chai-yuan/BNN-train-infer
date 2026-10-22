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
| `IN` | `in_valid && in_ready` | 同时消费 act_data/weight_data；通道号、通道内拍号与通道末拍由命令及配对输入计数推导 |
| `RES` | `res_valid && res_ready` | int32 结果、通道号、组号、`res_chan_last`、`res_last` |
| `END`（推导事件） | `res_valid && res_ready && res_last` | 与末 RES 同周期生成的成功完成记录；不是独立完成通道 |

一次 CMD 握手锁存配置并开启任务；首对输入最早在下一上升沿接收。末结果握手是
唯一成功完成条件：`res_last=1` 但 `res_ready=0` 时任务仍在运行；末结果交付同沿
不能接收 CMD，下一上升沿可以接收。`status_idle/status_busy` 保留用于状态观测。
reset 取消任务并清空配置、计数和反压检查状态，不生成完成通知。

工具会校验：

- 配对输入拍数是否符合 `N × k_blocks`，按 `k_blocks` 推导每通道边界；
- 结果数是否符合 `N × groups`；
- `res_group_idx` 是否连续，`res_chan_last` / `res_last` 是否与通道 / 任务边界一致；
- CMD、IN、RES 在 `valid && !ready` 后，直到接收沿的 valid 和全部负载是否保持稳定；
- CMD / 首对输入同沿、末结果 / 下一命令同沿、复位取消、孤儿握手、任务未完成等情况。

非法配置由外层保证不提交，RTL 仿真 assert；分析器只对错误波形给出诊断，
不虚构配置错误响应。新接口没有 `ctrl_abort`、`done_*`、`res_error`，也没有
独立 ACT/WGT 的 valid/ready 或物理 `act_last/weight_last`。
这些检查只在顶层可见信号上完成，不读取内部状态或参考模型。

### 采样语义

Verilator 在时钟上升沿的同一时间戳里会同时写入时钟跳变和触发器的新值。
如果直接读该时间戳的末态，会看到握手完成后的 `ready=0`，从而漏掉握手。
本工具保留**跳变前已稳定的快照**，用它对 `valid/ready/payload` 判定握手，
再推进到当前时间戳。这个细节也由测试覆盖。

## 界面

- **任务 / COMMANDS**：按 CMD 握手切分；点击聚焦任务，时间线自动缩放到任务窗口。
- **数据流 / HANDSHAKE FLOW**：DRIVER → CORE → SINK 的命令、配对输入、结果及推导完成事件；
  ACT/WGT 分开展示同一个 IN 握手的两路负载；`END*` 仅标注末结果交付，不绘制独立完成通道。
- **时间线 / DENSITY**：各行显示通道握手密度或推导事件（`END*`），不是逐周期波形；`BUSY` 显示
  `status_busy` 占空段。滚轮缩放，双击或 `FIT` 适应当前任务/全波形。
- **检视 / INSPECT**：`握手` 页给出事件表与负载（激活 lane、权重 bit、结果）；
  `统计` 页给出全局和当前任务的性能/协议摘要。
- **跟随**：播放时事件列表自动跟随；手动滚动会关闭跟随，点按钮可重新打开。

快捷键：`空格` 播放/暂停，`←/→` 上一个/下一个握手，`Home/End` 首个/末个握手。

## 性能摘要

全局：

- 上升沿数、非复位周期、忙碌周期与占空比；
- CMD、配对输入、结果握手数量，以及由末结果握手推导的完成任务数；
- 各通道 `valid && !ready` 停顿计数；
- 时钟周期与波形时长（来自 `$timescale`）。

单任务：

- 周期范围、输入/结果首事件偏移、结果排空滞后；
- 输入拍密度、结果密度、通道完成数；
- 忙周期、末结果反压等待周期、停顿计数（配对输入只计一次）；
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
### 数据 schema 与 UI 兼容

- 输入事件仍为 `type: 'in'`，新增 `inH: true`；`actH === wgtH === inH`。
  `summary.inHandshakes`、`actHandshakes`、`weightHandshakes` 都是配对输入握手数。
- 新增 `ev.channelLast`；兼容字段 `actLast` / `weightLast` 均等于它，来源标注为
  `boundarySource: 'cmd_k_blocks + input count'`，不是物理端口。没有对应任务时边界为 `null`。
- 保留 `type: 'done'`，其 `derived: true`、`source: 'res_valid && res_ready && res_last'`。
  它紧随末 RES 事件、与其同周期；没有独立 DONE 反压。
  `task.doneCycle === task.lastResultCycle`，`task.completionSource` 和
  `trace.completionSource` 标明相同推导来源；成功 `statusCode: 0` 是软件兼容值，
  `doneStatusRaw: null`，不对应物理状态码。
- `summary.completed === summary.doneEvents`：实际交付末结果的任务数；`ok` 为其中
  通过分析器校验的任务，`failed` 为校验失败或被新命令覆盖的任务。
  `canceled` 为 reset 取消数，单独统计，不计入 `failed`；`unfinished` 为波形结束仍未完成数。
- 新增全局 / 任务 `stalls.in`；兼容 `stalls.act` / `stalls.weight` 均为同一输入停顿数，
  `task.stallTotal = stalls.in + stalls.res`，避免重复计算。
  全局 `stalls.done` 固定为 0；`task.doneWaitCycles` 表示末结果被反压的周期数。
- reset 任务保留 `status: 'reset'`，`doneCycle` / `completionSource` 为 `null`，没有 DONE 事件。
  不再产生配置错误、abort 或协议错误的硬件完成状态；协议问题记录在 `issues` / `warnings`。

`samples/demo.vcd` 来自当前已编译新 RTL，通过公共 BNN API 执行两个 K128/N2/G128
任务（DOT_GROUPS、GEMV）。28 个上升沿，16 次配对输入、4 次结果握手，两个任务
分别在周期 14 / 27 完成；原始结果为 `[128, -128]`，无合成的旧端口。

## 边界

- 仅支持 VCD；FST 需先转换。
- 仅读取选中作用域的直接变量；当前自动优先 `bnn_core_top`，没有该模块名时选择
  握手接口最完整的顶层作用域。
- 要求 CMD、IN、RES 三组 `valid/ready`、clock/reset、命令参数、status_idle/status_busy
  及全部输入/结果负载端口存在；缺失时直接报错，不适配旧顶层端口。
- 事件明细上限为 50 万条；超过后任务计数和统计仍然准确，但输入/结果明细会省略。
- VCD 中若没有时钟上升沿、缺少握手端口或头部损坏，会直接报错。
- 时间线是握手密度和任务窗口，不是门级/周期级时序波形。

## 测试

```sh
cd tools/wave-deck
make test
```

测试覆盖：不规则分块流式解析、作用域自动选择、预跳变握手采样、共用输入与推导边界、
三通道反压稳定性、末结果阻塞不提前完成、下一拍命令、reset 取消与清理、缺失新端口、
事件上限下的完成统计、非法配置诊断，以及真实新 RTL 示例的计数与 lane 解码。
字体使用 Fusion Pixel（SIL Open Font License 1.1，许可证见
`assets/fonts/OFL-fusion-pixel.txt`）。
