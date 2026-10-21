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
