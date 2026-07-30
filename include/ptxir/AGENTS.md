# include/ptxir — PTXIR 公共 API

## 公共头文件修改协议

修改 `ptxir_serialization.h` 中的公共 API 签名时，必须：

1. 同步更新 `src/ptxir/ptxir_serialization.cpp` 实现
2. 同步更新测试文件 `tests/unit/test_ptxir_serialization.cpp`
3. 同步更新 `src/ptx_ir/AGENTS.md` 中的 API 引用

## 交叉引用

- `src/ptx_ir/AGENTS.md` — PTXIR 序列化内部实现
- `src/ptx_ir/ptxir_writer.cpp` / `ptxir_reader.cpp` — 序列化实现
- `tests/unit/test_ptxir_serialization.cpp` — Roundtrip 测试