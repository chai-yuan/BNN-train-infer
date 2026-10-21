# 加速器 API 与软/硬件差分测试

## 测试内容

这组测试绕过 BNNE 引擎和模型加载，直接针对 `bnn-accelerator` 层：

- `test_bnn_accelerator.c`：用独立标量 oracle 校验公共 `BNN_*` API；同一个 C 测试对象分别链接 soft 实现和 Verilator 归档，再逐字节比较 int32/FP32 输出。
- `test_verilator_bridge.cpp`：直接测试桥接 ABI，覆盖超时恢复、`INT8_MIN`、参数校验、多模型隔离。
- `test_verilator_context.cpp`：测试嵌入宿主 Verilator 程序时 context、TLS 和时间的保存/恢复。

差分测试不是只把两个后端互相比：每个 runner 先和独立 oracle 对齐，然后才比较两份输出 dump。

## 运行

```bash
make                 # 构建 verilator 并跑全部测试
make test
make waves           # 同上，并把 VCD 保留到 build/waves/accelerator
make waves OUTPUT_DIR=build/my-waves
```

直接运行（需要先构建 `ACCE=verilator` 归档）：

```bash
python3 tests/accelerator/test_accelerator.py \
    --trace-dir build/waves/accelerator
```

## 波形导出

默认不导出波形。使用 `--trace-dir DIR` 或环境变量 `BNN_RTL_TRACE_DIR` 开启；目录不存在会自动创建。

- 每个仿真实例生成一个 `bnn-core-XXXXXX.vcd`。
- public API、bridge、context 三段测试都会产生波形，所以一次完整运行会得到多个 VCD；脚本会按阶段打印新增文件路径。
- 已存在的 VCD 不会被删除或覆盖。
- soft runner 不产生波形；波形只来自 Verilator 后端。

## 覆盖范围

- 形状：`K = 128 / 384 / 4096`，`N = 1 / 3 / 17`，`M = 1 / 3`。
- 操作：`DOT_GROUPS`、`GEMV`、`GEMM`。
- 布局：packed / padded、非零 offset、非对齐 stride，并检查输入与输出 padding 不被改写。
- 数值：FP16 scale 覆盖正规数、次正规数、±0、最大有限值。
- 错误路径：逻辑输入中的 `-128`、非法 job/matrix、NaN/Inf scale、缓冲区越界、资源归属和多设备隔离。
- 桥接专项：超时恢复、`INT8_MIN`、多模型交错与销毁。

## 依赖

- 必须使用 `ACCE=verilator` 归档；Makefile 会自动构建。
- 需要 Verilator、C99/C++17 编译器和 Python 3。
- 测试的编译产物和临时 dump 在临时目录，波形只在指定 `--trace-dir` 下保留。
