# VCD 波形导出测试

## 测试内容

通过公共 `BNN_*` API 驱动 Verilator 后端，验证 `BNN_RTL_TRACE_DIR` 波形开关：

- 未设置或为空：不生成任何文件。
- 目录不存在或不可写：`BNN_DeviceOpen` 返回 `BNN_EIO`，不留下 VCD 或目录。
- 正常开启：每个设备生成一个唯一的 `bnn-core-XXXXXX.vcd`，覆盖复位和该设备的全部任务。
- 任务结束后 flush；关闭设备后波形不再变化。
- 两个设备隔离、文件名避重、目录中已有文件保持不变。
- 检查 VCD 层级、1 ps 时间单位、10 ns 仿真时钟和结果内容。

## 运行

```bash
make                 # 临时波形，测试结束自动清理
make test
make waves           # 保留到 build/waves
make waves OUTPUT_DIR=build/my-waves
```

直接运行（需要先构建 `ACCE=verilator` 归档）：

```bash
python3 tests/trace/test_verilator_trace.py --output-dir build/waves
```

## 依赖

- 必须使用 `ACCE=verilator` 归档；Makefile 会自动构建。
- 需要 Verilator、C99/C++17 编译器和 Python 3。
- 测试任务很小（`K=128, N=2, G=128`），适合生成可读的调试波形。

## 波形观察

先查看 `clock/reset`、`cmd_valid/cmd_ready`、`in_valid/in_ready`、`act_data/weight_data`、
`res_valid/res_ready/res_data/res_group_idx/res_chan_last/res_last` 和 `status_idle/status_busy`。
激活与权重共用输入握手，无独立输入握手或输入 last；边界由命令尺寸和已握手拍数推导。
`cmd_valid && cmd_ready` 在上升沿锁存三个尺寸并启动任务，状态依次为 `IDLE -> RUN -> DRAIN -> IDLE`。
唯一成功完成事件是 `res_valid && res_ready && res_last`；最后结果阻塞时必须保持，不能仅以 res_last 置位判断完成。
最后结果握手的同一沿不能接收下一命令，返回 IDLE 后下一上升沿才可接新命令。

`reset` 为高有效同步复位，清空全部锁存配置、计数和数据有效位；置位期间屏蔽所有握手。
复位丢弃任务但无取消通知，外层须清理旧输入并作废已接收的部分输出。
配置合法性由外层保证；核心仅仿真 assert 检查非零、整除和 `groups <= 256`，不提供运行时错误码。
非法配置的 assert 失败探针由核心测试覆盖，不属于本目录的波形开关测试。

本目录脚本使用保留的 `clock`、`reset`、`cmd_valid`、`res_data` 解析波形，无旧端口依赖；
这些检查覆盖 trace 生命周期和结果内容，不替代核心逐周期握手与反压测试。
