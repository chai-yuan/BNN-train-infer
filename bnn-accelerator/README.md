# Accelerator API v2

公共接口见 `bnn-accelerator.h`。当前 soft 参考后端实现接口声明的全部 5 种运算：
I8 输入的 `GEMV / GEMM / DOT_GROUPS`，以及 F32 输入的 `GEMV_F32 / GEMM_F32`。
本轮不新增硬件后端或 IPC transport。旧 Verilator 软件后端、C++ bridge、
bridge/context 专项及 `tests/trace` 已移除；旧 RTL 与独立自检保留作参考。

## 边界与兼容性

- `BNN_API_VERSION = 2`，是破坏性接口变更，调用者与后端必须一起重新编译。
- Buffer 是无类型字节存储，`BNN_BufferView` 是按值传给 Job 的借用描述。
  修改视图不会改变 Buffer、另一份视图或已经构造的 Job；视图不延长资源生命周期。
- View 提供 `buffer / type / offset / row_stride`，形状由 Matrix、Job 和 operation 确定。
  offset/stride 单位均为字节，允许非对齐；stride=0 表示紧密排列。
- 不提供可变的 SetBufferView/GetBufferView 状态，不增加通用 Tensor 或异步框架。
- 旧 `job.input` 句柄改为 `job.input.buffer`，旧 `input_offset/input_row_stride`
  改为 `job.input.offset/job.input.row_stride`，output 同理；必须填写输入输出类型。
- 旧 caps 的 `supports_*` 改为 `supported_ops` 位集合，F32 运算能力包含其约定的量化规则。
  不单独提供量化模式或能力字段，不要把操作枚举值直接当能力位。
- `DOT_GROUPS` 从 I32 改为 I16，需要同时修改容量、步长及读回元素类型。

例如现有 I8 GEMV 的任务构造为：

```c
BNN_Job job = {0};
job.operation = BNN_OP_GEMV;
job.matrix = matrix;
job.input.buffer = input_buffer;
job.input.type = BNN_TYPE_I8;
job.output.buffer = output_buffer;
job.output.type = BNN_TYPE_F32;
job.rows = 1;
/* 偏移和行步长均为 0，表示从缓冲区起点开始紧密排列。 */
BNN_Status status = BNN_Execute(device, &job);
```

输入、输出必须使用不同 Buffer。修改一份 View 可以复用同一 allocation 的其他区域，
但不能据此执行原地运算。参数检查、容量检查和输入值检查失败不能修改输出；
执行成功也不能改写 padding。其余所有权、并发和数值契约见头文件。

## 精度、量化与 INT16

`GEMV/GEMM` 保留 I8 输入、F32 输出，激活范围仍为 `[-127,127]`。
它们不会应用激活 scale；调用者仍负责还原。

`*_F32` 只描述外部输入输出类型，不表示直接执行原始 FP32 激活的矩阵乘法。
本版固定使用 I8 对称量化，不提供量化模式选择，也不为短期内不使用的精度预留枚举和能力位。
如果以后确实需要 W1A4/W1A16，再明确各自的数值契约和接口扩展，不能由后端静默改变精度。

F32 运算按每个**完整逻辑输入行**做 absmax 对称量化，量化到最近整数，半整数远离零；
零行使用 scale=1，极小 scale 的处理和非有限输入的错误语义见头文件。
后端内部 K 分片必须复用整行的量化结果。上层自行拆 Matrix 再调用 F32 操作时，
每片已经是独立逻辑矩阵，因此不能假定其等价于整行量化；需要保持原语义时继续走 I8 路径。

本版权重 scale 的组大小 G 必须为 8 的倍数且不超过 256：
`|sum(q*w)| <= 256*127 = 32512`，所以 I8 `DOT_GROUPS` 的 I16 结果不会溢出。
内部运算允许更宽，禁止 wrap/saturate。未来更高激活位宽不应套用此 I8 的 I16 界限；
F32 运算的内部累加宽度与公共 I8 `DOT_GROUPS` 输出类型是两回事。

soft 声明 `GEMV / GEMM / DOT_GROUPS / GEMV_F32 / GEMM_F32`，G=128，逻辑矩阵尺寸上限 8192。
F32 路径在后端内完成逐行 absmax 量化、分组整数点积和 `Z * s` 还原：
`a / 127` 下溢为零时使用最小正 binary32 次正规数，半整数远离零舍入，输入含 NaN/Inf 返回
`BNN_ERANGE`，并在写输出前完成全部行的输入检查。v2 引擎使用单行 `GEMV_F32`，
主机侧不再量化；超过 `max_in_features` 的矩阵由引擎拒绝，当前模型均落在 8192 上限内。

## GGUF 不需要与运行时布局完全一致

保留现有 Q1_0 磁盘格式，不需要重导出或迁移模型：

| 层 | 布局与职责 |
| --- | --- |
| GGUF Q1_0 | 每 128 个权重一个 18 字节块：little-endian FP16 scale + 16 字节符号位；scale 与 bits 交错 |
| `bnne_q1_0_read()` | 解码磁盘字节序，将交错块拆为 bits 和 scale 两个数组 |
| `BNN_MatrixDesc` | 行优先 `[out][in]` packed bits，加 `[out][in/G]` 主机 uint16_t FP16 位模式 |
| 后端 | 根据设备需要上传、分片、重排或缓存，不把私有布局传给模型加载层 |

GGUF 的 `dims=[in,out]` 按最快变化维度在前描述，同一逻辑矩阵的 C 行优先表示是
`[out][in]`；这不是一次数学转置。两者的符号位含义保持一致，bit 1 为 +1，bit 0 为 -1。

磁盘编码服务模型存储，公共描述服务计算语义，设备布局服务带宽和计算复用。
强制它们一致会把运行时锁死到某种 GGUF 块编码，并不能自动实现零拷贝。
未来若测量证明 repack 成本显著，再单独设计带显式编码标记的导入接口，
不要改变 `weight_bits` 等现有字段的含义。

## 逻辑能力与未来设备接入

caps 表示完整后端对调用者提供的能力，包括内部 tiling 和必要的软件后处理，
不是裸 RTL 指令集或单次命令的限制。后端必须保留规定的量化域、FP32 累加顺序和完成语义，
不能为了适配 tile 改变结果。当前参考实现的尺寸限制不代表未来硬件的必须限制。

未来可以把硬件进程/仿真器放在后端的 transport 之后；公共 API 不预设寄存器、
共享内存、socket、周期步进或 Verilator context。C 句柄、指针、`size_t` 和原生结构体
不是 IPC wire format，不能直接 memcpy 给另一进程作为设备协议。
任务完成必须意味着结果已写回且可读，不仅是计算核心接收了最后一个结果。
具体消息协议、资源 ID、传输、超时与恢复机制留待硬件设计阶段。

## 回归

```sh
make -C tests/accelerator test
```

默认只需要 Python 3 和 C 编译器，不依赖 Verilator。测试使用独立整数/FP32 与
F32 量化 oracle，保留资源、布局、错误路径覆盖，并检查类型、View 复用、INT16 极值、
F32 量化边界与 NaN/Inf 拒绝。
