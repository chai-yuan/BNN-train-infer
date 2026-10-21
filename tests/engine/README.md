# Engine / CLI 回归测试

## 测试内容

- `test_mlp.py` 用 Python 标准库实现独立的 MLP/CNN/GPT 参考前向，生成临时 `.bnn` 模型、IDX 数据、tokens 和 reference logits。
- 编译并运行 `test_bnn_engine.c`，覆盖 BNNE 公共 API：模型信息、动态 shape、tokens、统计、错误加载/输入、资源回收和多句柄隔离。
- 用根目录的 `bnn-infer` 验证分类、CNN、GPT 的 reference 校验、`--eval`、`--profile` 阶段选择和生成行为。
- `fixtures` 目标只生成测试数据，不运行测试。

这组测试经过 BNNE 引擎和 CLI，不直接测试 `bnn-accelerator`；按接口层测试请使用 `tests/accelerator/`。

## 运行

```bash
make                         # 默认 ACCE=soft，等价 make test
make test
make ACCE=verilator test
make api                     # 只运行引擎 API runner，跳过 CLI
make fixtures                # 生成到 build/fixtures
make fixtures FIXTURES=build/verilator-fixtures
```

`make` 会先调用顶层构建所选后端，因此需要根目录 `bnn-infer` 和 `build/<ACCE>/libbnn-engine.a`。
`FIXTURES` 的相对路径按仓库根解析；`PYTHON`、`CC`、`CXX` 可在命令行覆盖。

## 依赖

- Python 3 标准库、C99 编译器、Make 和 `ar`。
- `ACCE=verilator` 还需要 Verilator 和 C++17 编译器。
- 所有模型、reference 和临时可执行文件都在临时目录，测试结束自动清理。
