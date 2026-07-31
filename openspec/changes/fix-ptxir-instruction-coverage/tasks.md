## 1. Reader GENERIC_INSTR 全覆盖（Phase 1）

- [ ] 1.1 扩展 `src/ptx_ir/ptxir_reader.cpp` 的 `read_instruction()` GENERIC_INSTR case 组：从 8 个 enum（S_MOV/S_ADD/S_SUB/S_MUL/S_LD/S_ST/S_SETP/S_CVT）扩展为全部 53 个 GENERIC_INSTR enum（加 S_CVTA/S_PRMT/S_ISSPACEP/S_MAPA/S_ALLOCA/S_MUL24/S_DIV/S_REM/S_MIN/S_MAX/S_NEG/S_ABS/S_MAD/S_MAD24/S_FMA/S_ADDC/S_SUBC/S_SAD/S_COPYSIGN/S_TESTP/S_TANH/S_AND/S_OR/S_XOR/S_NOT/S_SHL/S_SHR/S_SHF/S_BFE/S_LOP3/S_SET/S_SELP/S_SLCT/S_CNOT/S_SIN/S_COS/S_LG2/S_EX2/S_RCP/S_RSQRT/S_SQRT/S_POPC/S_CLZ/S_ACTIVEMASK/S_ST_BULK）
- [ ] 1.2 补齐其他缺失 enum：`S_BRX` → S_BRA case 组（BranchInstr）；`S_TRAP/S_BRK/S_BRKPT` → S_EXIT/S_RET case 组（VoidInstr）
- [ ] 1.3 更新测试 `tests/unit/test_ptxir_serialization.cpp`：新增代表性 GENERIC enum roundtrip 测试（至少覆盖 S_CVTA、S_FMA、S_DIV、S_SIN、S_AND、S_POPC、S_MUL24、S_ACTIVEMASK）+ S_BRX + S_TRAP/S_BRK/S_BRKPT roundtrip 测试
- [ ] 1.4 验证：`cd build && cmake --build . --target unit_ptxir_serialization && ./bin/tests/unit_ptxir_serialization` 全绿

## 2. Tcgen05Instr 序列化（Phase 2）

- [ ] 2.1 修改 `src/ptx_ir/ptxir_writer.cpp`：新增 `write_tcgen05(const Tcgen05Instr&)`（write_qualifiers + write_operands），并在 `write_instruction()` if-constexpr 链注册 `std::is_same_v<T, Tcgen05Instr>` 分支
- [ ] 2.2 修改 `src/ptx_ir/ptxir_reader.cpp`：新增 `S_TCGEN05_*` 11 个 enum 的 case 组，重建 `Tcgen05Instr`（qualifiers + operands），`op_kind` 从 stmt.type 1:1 派生
- [ ] 2.3 更新测试 `tests/unit/test_ptxir_serialization.cpp`：新增 `Roundtrip: Tcgen05Instr (S_TCGEN05_MMA)` — 构造含 qualifiers + operands 的 Tcgen05Instr，roundtrip 后断言 type/op_kind/qualifiers 正确
- [ ] 2.4 验证：`cd build && cmake --build . --target unit_ptxir_serialization && ./bin/tests/unit_ptxir_serialization` 全绿

## 3. 全 enum + 真实 kernel roundtrip + 回归（Phase 3）

- [ ] 3.1 新增全 enum roundtrip 测试：遍历 `ptx_op.def` 全部 106 个 enum，为每个构造代表性 StatementContext 并 roundtrip（X-Macro 展开或逐一手写），断言类型保持且不抛异常
- [ ] 3.2 新增真实 kernel roundtrip 测试：使用 `tests/ptx/test_divergence_sync_standalone.ptx`（含 cvta）或 `bench/cute/` fixture，验证 `generate_ptxir() → load_ptxir()` 不抛异常且语句非空
- [ ] 3.3 运行 `cd build && ctest --output-on-failure` 全量测试无回归
- [ ] 3.4 运行 `./scripts/sanity.sh` 和 `./scripts/regression.sh` 确认全量回归通过
