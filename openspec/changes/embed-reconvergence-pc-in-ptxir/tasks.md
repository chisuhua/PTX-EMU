## 1. PTXIR 格式扩展：S_BAR 序列化 reconvergence_pc + 版本升级

- [x] 1.1 更新 `include/ptx_ir/ptxir_format.h`：`PTXIR_VERSION` 2→3，`BARRIER_ENCODED_SIZE` 增加 `sizeof(int32_t)`
- [x] 1.2 修改 `src/ptx_ir/ptxir_writer.cpp`：`write_barrier()` 追加 `write_i32(out_, instr.reconvergence_pc)`
- [x] 1.3 修改 `src/ptx_ir/ptxir_reader.cpp`：`read_instruction()` S_BAR case 在 v3 时读取 `reconvergence_pc`（v2 跳过，保持 `-1`）
- [x] 1.4 修改 `src/ptx_ir/ptxir_reader.cpp`：`read_header()` 版本检查 `if (version_ != 1 && version_ != 2)` 扩展为接受 v3（`version_ != 1 && version_ != 2 && version_ != 3`）
- [x] 1.5 更新测试 `tests/unit/test_ptxir_serialization.cpp`：
  - **T1**：新增 `Roundtrip: BarrierInstr reconvergence_pc=42` — 设置非默认值 `reconvergence_pc=42`，序列化/反序列化后断言 `out.reconvergence_pc == 42`（现有测试只检查 `barId`，不检查 `reconvergence_pc`）
  - **T2**：新增 `Roundtrip: BarrierInstr barId=nullopt` — 验证 `barId=std::nullopt` 的 roundtrip（sentinel -1 路径）
  - **T11**：新增 `PTXIR header version=3 accepted` — 构造 v3 header 字节，验证 `read_header()` 不抛异常
  - **T12**：新增 `BARRIER_ENCODED_SIZE constant` — `static_assert` 或编译时检查 `ptxir_encoding::BARRIER_ENCODED_SIZE == sizeof(uint16_t) + sizeof(int32_t) + sizeof(int32_t)`
- [x] 1.6 验证：`cd build && cmake -S .. -B . && cmake --build . --target test_ptxir_serialization && ctest -R "unit_ptxir_serialization" --output-on-failure`

## 2. generate_ptxir() 嵌入 CFG 计算

- [x] 2.1 修改 `src/ptxir/ptxir_serialization.cpp`：`generate_ptxir()` 在 `serialize_statements()` 前插入 CFG 计算逻辑：
  - 构造 `label2pc` 映射
  - 调用 `CFGBuilder::build()` → `CFGBuilder::computePostDominators()`
  - 填充 `kernel->kernelStatements` 中所有 S_BRA 和 S_BAR 的 `reconvergence_pc`
- [x] 2.2 新增测试 `tests/unit/test_ptxir_serialization.cpp`：
  - **T4**：`PTXIR v3 file: load_ptxir(apply_cfg=false) returns non-default reconvergence_pc` — 构造 v3 格式二进制（含 S_BAR reconvergence_pc），验证反序列化后值正确且不调用 CFG
  - **T5**：`PTXIR v3 file: load_ptxir(apply_cfg=true) == load_ptxir(apply_cfg=false)` — 验证嵌入值与 CFG 重算值一致
- [x] 2.3 验证：`cd build && cmake --build .` 编译通过
- [x] 2.4 验证：`ctest --output-on-failure` 全量测试无回归（旧 PTXIR v2 文件兼容）

## 3. 加载路径优化验证

- [x] 3.1 验证 `load_ptxir()` 默认 `apply_cfg=false` 行为：
  - 确认 v3 PTXIR 文件加载后 S_BRA 和 S_BAR 的 `reconvergence_pc` 正确
  - 确认 `apply_cfg=true` 时旧 v2 文件仍可回退重建
- [x] 3.2 运行 `./scripts/sanity.sh` 确认无回归
- [x] 3.3 运行 `./scripts/regression.sh` 确认全量回归通过