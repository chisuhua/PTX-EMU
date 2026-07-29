# refactor-ptxir-writer - Tasks

## 1. Phase 0: 准备工作

- [ ] 1.1 MUST 运行 `wc -l src/ptx_ir/ptxir_writer.cpp` 记录基线行数（应为 360）
- [ ] 1.2 MUST 运行 `ctest -R ptxir --output-on-failure` 记录基线测试结果（全绿）
- [ ] 1.3 MUST 运行 `./tests/ptx/test_all_ptx.sh` 记录基线 PTX 语法测试结果
- [ ] 1.4 MUST 生成基线 .ptxir 文件用于二进制一致性对比

## 2. Phase 1: 提取公共 helpers - qualifiers + operands（45 min）

- [ ] 2.1 MUST 在 `include/ptx_ir/ptxir_writer.h` 添加 private 方法声明：
  - `void write_qualifiers(const std::vector<Qualifier>& qualifiers);`
  - `void write_operands(const std::vector<OperandContext>& operands, bool write_imm);`
- [ ] 2.2 MUST 在 `ptxir_writer.cpp` 实现 `write_qualifiers()`：提取 qualifier size + 遍历写入逻辑
- [ ] 2.3 MUST 在 `ptxir_writer.cpp` 实现 `write_operands()`：提取 operand 遍历 + REG/IMM/other 分发逻辑，`write_imm` 控制是否处理 IMM
- [ ] 2.4 MUST 替换 31 个 variant 分支中的 qualifier/operand 遍历循环为 `write_qualifiers()` / `write_operands()` 调用
- [ ] 2.5 MUST 验证：`cmake --build build` 通过
- [ ] 2.6 MUST 验证：`ctest -R ptxir --output-on-failure` 全绿
- [ ] 2.7 MUST 验证：`./tests/ptx/test_all_ptx.sh` 全绿
- [ ] 2.8 MUST 验证：新生成的 .ptxir 与基线 byte-identical
- [ ] 2.9 git commit -m "refactor(ptxir_writer): extract write_qualifiers/write_operands helpers"

## 3. Phase 2: 提取 per-type helper 方法（60 min）

- [ ] 3.1 MUST 在 `include/ptx_ir/ptxir_writer.h` 添加按类别分组的 private helper 声明：
  - Control flow: `write_branch`, `write_label`, `write_void`, `write_abi_directive`
  - Barrier/Sync: `write_barrier`, `write_bar_warp_sync`, `write_membar`, `write_fence`, `write_redux_sync`, `write_mbarrier`
  - Generic/Declaration: `write_generic`, `write_declaration`, `write_predicate_prefix`
  - Warp collective: `write_vote`, `write_shfl`
  - Memory/Atomic: `write_atom`, `write_texture`, `write_surface`, `write_reduction`, `write_prefetch`, `write_cp_async`
  - Misc: `write_pragma`, `write_dollar_name`, `write_call`
- [ ] 3.2 MUST 在 `ptxir_writer.cpp` 实现每个 helper 方法（从原 variant 分支体移动，使用 Phase 1 的公共 helpers）
- [ ] 3.3 MUST 将 `write_instruction()` 各 `if constexpr` 分支体替换为单行 helper 调用
- [ ] 3.4 MUST 验证：`cmake --build build` 通过
- [ ] 3.5 MUST 验证：`ctest -R ptxir --output-on-failure` 全绿
- [ ] 3.6 MUST 验证：`./tests/ptx/test_all_ptx.sh` 全绿
- [ ] 3.7 MUST 验证：新生成的 .ptxir 与基线 byte-identical
- [ ] 3.8 git commit -m "refactor(ptxir_writer): extract 31 variant branches to per-type helpers"

## 4. Phase 3: 最终验证（15 min）

- [ ] 4.1 MUST 验证：`write_instruction()` 函数 < 50 行（仅分发逻辑）
- [ ] 4.2 MUST 验证：`ctest --output-on-failure` 全绿
- [ ] 4.3 MUST 验证：`./tests/ptx/test_all_ptx.sh` 全绿
- [ ] 4.4 MUST 验证：新生成的 .ptxir 与基线 byte-identical
- [ ] 4.5 SHOULD 验证：每个指令类别的序列化函数可通过 round-trip 测试间接验证

## 5. 应用阶段

- [ ] 5.1 MUST 运行 `openspec validate refactor-ptxir-writer --strict`
- [ ] 5.2 MUST 通过所有验证后 archive 此 change

## 验收

- `write_instruction()` < 50 行（从 232 行）
- PTXIR round-trip 测试全绿（write->read 无损）
- 所有现有 PTXIR 测试通过
- 二进制输出与拆分前 byte-identical
- 每个指令类别的序列化函数可独立测试（通过 round-trip 间接覆盖）

## 关键约束（MUST/MUST NOT）

- MUST 保持 PTXIR 二进制格式 byte-identical（reader 兼容性）
- MUST NOT 改变 write_u16/write_u32/write_string 等底层写入函数
- MUST NOT 修改 PTXIR reader 端
- MUST NOT 改变二进制格式（field order / size / endianness）
- MUST NOT 修改 `ptx_op.def` X-Macro 定义
- SHOULD 按指令类别分组到独立 section
