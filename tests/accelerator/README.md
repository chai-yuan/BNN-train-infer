# 加速器 API v2 独立 soft 回归测试

这组测试绕过 BNNE 引擎和模型加载，直接针对 `bnn-accelerator` 公共 API。
`test_bnn_accelerator.c` 使用独立标量 oracle：逐项累加有符号权重乘积，
并用字面量 FP16/FP32 对照值校验 scale，不复用 soft 的点积或半精度转换实现。

Python 默认直接编译 C runner 与 `bnn-accelerator/soft-bnn-accelerator.c`，
然后在临时目录运行；不需要 RTL、Verilator、静态归档或仓库根目录的 build。
这里不实现新后端，也不执行 Verilator bridge/context 测试。

## 运行

从仓库根目录：

```bash
python3 tests/accelerator/test_accelerator.py
make -C tests/accelerator       # 默认目标即 test
make -C tests/accelerator test
```

在本目录可以直接运行 `make` 或 `make test`。支持 compiler wrapper 和额外编译/链接参数：

```bash
python3 tests/accelerator/test_accelerator.py --cc 'ccache gcc'
python3 tests/accelerator/test_accelerator.py \
    --cflags='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
make -C tests/accelerator test CC='ccache gcc' \
    CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

`--cc`、`--cflags` 默认分别读取环境变量 `CC`、`CFLAGS`。
测试与仓库根构建统一使用 C11。`-std=c11`、`-fno-fast-math`、`-ffp-contract=off`
始终放在用户参数之后，避免用户 flags 启用 fast-math / FMA 或改变语言标准而破坏 oracle。
编译器及运行器的单次超时为 300 秒；编译产物运行后自动清理，不导出波形。
C runner 仍可独立运行，接受可选输出文件参数，记录成功任务的原始 int16/FP32 结果。

## v2 覆盖范围

- 形状：保留 `K = 128 / 384 / 4096 / 8192`、`N = 1 / 3 / 17`、`M = 1 / 3` 的 sweep。
- 已实现操作：I8 输入的 `DOT_GROUPS`、`GEMV`、`GEMM`，输入由调用者预先量化；
  F32 输入的 `GEMV_F32`、`GEMM_F32`，由后端按头文件规则逐行量化。
  I8 逻辑输入范围为 `[-127, 127]`；逻辑输入出现 `-128` 返回 `BNN_ERANGE`，padding 中的 `-128` 合法。
- 输出：`DOT_GROUPS` 为 `[M][N][K/G]` 的 `int16_t`；GEMV/GEMM 输出 FP32。
  保留独立 oracle 逐字节校验，并专门覆盖 `G=128` 时的点积极值 `±16256`。
- F32 输入：`GEMV_F32 / GEMM_F32` 使用独立 oracle 覆盖 absmax 对称量化、
  `a / 127` 下溢回退到最小正次正规数、半整数远离零舍入、零行、normal/subnormal 输入、
  非对齐 view 与 padding 中的 NaN；逻辑输入含 NaN/Inf 返回 `BNN_ERANGE` 且输出不变，
  每个拒绝后都执行一次成功恢复。
- typed views：`BNN_BufferView` 的 type、字节 offset、字节 row_stride；packed / padded、
  非零且非对齐的 offset/stride，以及同一对 buffer 上不同类型和布局的 view 交错运行互不影响。
  输入与输出必须是不同 buffer；检查输入、输出前后缀及 padding 不被改写；F32 view
  同样覆盖非对齐 offset/stride。
- 类型校验：五种操作按各自输入/输出类型检查；错误类型和未知类型返回 `BNN_EINVAL`，
  未知运算返回 `BNN_ENOTSUP`；`GEMV_F32` 的 `rows != 1` 也返回 `BNN_EINVAL`；
  拒绝任务不会改写输出。
- scale：保留 FP16 正规数、次正规数、±0、最大有限值；拒绝 NaN/Inf。
- group size：接口要求 `G` 是 8 的倍数且 `G <= 256`，非法值返回 `BNN_EINVAL`；
  soft 仍仅支持 `G=128`，完整有效的 `G=64 / 256` 描述返回 `BNN_ENOTSUP`。
- caps：API 版本为 2，soft 的 `max_in_features` / `max_out_features` 为 8192；
  `supported_ops` 包含 `BNN_CAP_GEMV`、`BNN_CAP_GEMM`、`BNN_CAP_DOT_GROUPS`、
  `BNN_CAP_GEMV_F32`、`BNN_CAP_GEMM_F32`。
- 保留错误 job/matrix、stride/offset 溢出与越界、数据上传拷贝、buffer 传输、资源归属、
  busy、重复执行、多设备隔离与完整销毁后重开等回归。

## F32 输入路径

`BNN_OP_GEMV_F32` / `BNN_OP_GEMM_F32` 已由 soft 实现并在 caps 中声明。
后端对每个完整逻辑输入行做 absmax 对称量化，`a == 0` 时 `s = 1`；
`a / 127` 在 binary32 中下溢为零时使用 `FLT_TRUE_MIN`；
量化使用半整数远离零舍入并 clamp 到 `[-127, 127]`；输出为 `Z * s`。
输入含 NaN/Inf 时返回 `BNN_ERANGE`，且由于先完成整批输入检查，输出保持不变。

测试用与后端不同的舍入实现（`modff`）和标量点积 oracle 逐字节比较 F32 输出，
并覆盖 normal/subnormal、scale 下溢、半整数 tie、全零行、非对齐 view、
padding 中非有限值以及拒绝后的恢复。

## 依赖

仅需 Python 3、C11 编译器及标准 C 数学库（`-lm`）。无需第三方 Python 包或 C++ 编译器。
