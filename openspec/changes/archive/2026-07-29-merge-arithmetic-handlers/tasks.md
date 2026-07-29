# merge-arithmetic-handlers - Tasks

## Task List

### Phase 1: 提取共享 Helper（45 min）

- [ ] 1.1 MUST 在 `arithmetic.cpp` 顶部匿名 namespace 中实现 `extract_binary` helper：
  ```cpp
  namespace {
      struct BinaryOps {
          void* dst; void* src1; void* src2;
          int bytes; bool is_float; bool is_signed;
      };
      BinaryOps extract_binary(void** operands, const std::vector<Qualifier>& quals);
  }
  ```
- [ ] 1.2 MUST 实现 `apply_float` 模板（消除 f16/f32/f64 switch 重复）
- [ ] 1.3 MUST 实现 `apply_int` 模板（消除 s8/s16/s32/s64 switch 重复）
- [ ] 1.4 MUST 实现 `update_cc` helper（条件码更新逻辑提取）
- [ ] 1.5 MUST 实现 `f16_binary_op` helper（f16->f32 计算->f16 转换）
- [ ] 1.6 MUST 编译验证：`cmake --build build` 通过（helper 语法正确）

### Phase 2: 迁移基础算术 Handler（30 min）

- [ ] 2.1 MUST 迁移 `AddHandler` 使用 helper（删除重复类型解析 + memcpy）
- [ ] 2.2 MUST 迁移 `SubHandler` 使用 helper
- [ ] 2.3 MUST 迁移 `NegHandler` 使用 helper
- [ ] 2.4 MUST 迁移 `AbsHandler` 使用 helper
- [ ] 2.5 MUST 删除 `arithmetic.cpp` 中 ~350 行注释掉的旧代码（line 9-109 + 散落注释）
- [ ] 2.6 MUST 编译验证：`cmake --build build` 通过

### Phase 3: 迁移乘除运算 Handler（30 min）

- [ ] 3.1 MUST 将 `arithmetic_muldiv.cpp` 的 6 个 handler 迁移到 `arithmetic.cpp` Group 2 区域
- [ ] 3.2 MUST 迁移 `MulHandler` 使用 helper
- [ ] 3.3 MUST 迁移 `DivHandler` 使用 helper
- [ ] 3.4 MUST 迁移 `MadHandler` 使用 helper
- [ ] 3.5 MUST 迁移 `MinHandler`, `MaxHandler`, `RemHandler` 使用 helper
- [ ] 3.6 MUST 编译验证：`cmake --build build` 通过

### Phase 4: 迁移扩展算术 Handler（30 min）

- [ ] 4.1 MUST 将 `arithmetic_ext.cpp` 的 5 个 handler 迁移到 `arithmetic.cpp` Group 3 区域
- [ ] 4.2 MUST 迁移 `AddcHandler` 使用 helper（含 CC 更新）
- [ ] 4.3 MUST 迁移 `SubcHandler` 使用 helper（含 CC 更新）
- [ ] 4.4 MUST 迁移 `Mul24Handler`, `Mad24Handler` 使用 helper
- [ ] 4.5 MUST 迁移 `FmaHandler` 使用 helper
- [ ] 4.6 MUST 编译验证：`cmake --build build` 通过

### Phase 5: 合并文件 + CMake 更新（15 min）

- [ ] 5.1 MUST 删除 `src/ptxsim/instructions/arithmetic_ext.cpp`
- [ ] 5.2 MUST 删除 `src/ptxsim/instructions/arithmetic_muldiv.cpp`
- [ ] 5.3 MUST 更新 `src/CMakeLists.txt`：移除 `ptxsim/instructions/arithmetic_muldiv.cpp`（line 125）
- [ ] 5.4 MUST 更新 `src/CMakeLists.txt`：移除 `ptxsim/instructions/arithmetic_ext.cpp`（line 144）
- [ ] 5.5 MUST 编译验证：`cmake --build build` 通过

### Phase 6: 验证（30 min）

- [ ] 6.1 MUST 全量测试：`cd build && ctest --output-on-failure`
- [ ] 6.2 MUST 验证算术 unit 测试：`cd build && ctest -R "unit_.*(add|sub|mul|div|mad|fma|min|max|rem|neg|abs)" --output-on-failure`
- [ ] 6.3 MUST 验证算术 integration 测试：`cd build && ctest -R "integration_.*(add|sub|mul|div|mad|fma|min|max|rem|neg|abs)" --output-on-failure`
- [ ] 6.4 MUST 验证行数减少 ≥ 15%：`wc -l src/ptxsim/instructions/arithmetic.cpp` 结果 ≤ 1472 行
- [ ] 6.5 MUST 验证文件删除：确认 `arithmetic_ext.cpp` 和 `arithmetic_muldiv.cpp` 不存在

### Phase 7: 提交

- [ ] 7.1 git commit -m "refactor(arithmetic): merge 3 handler files into unified arithmetic.cpp with shared helpers"
- [ ] 7.2 MUST 运行 `openspec validate merge-arithmetic-handlers --strict`
- [ ] 7.3 MUST 通过所有验证后 archive 此 change

## 验收

- 合并后 `arithmetic.cpp` 总行数 ≤ 1472 行（减少 ≥ 15%）
- `arithmetic_ext.cpp` 和 `arithmetic_muldiv.cpp` 已删除
- `src/CMakeLists.txt` 已更新（移除 2 行源文件引用）
- 所有算术指令测试通过（unit + integration）
- `ctest` 全量全绿
- `ptx_op.def` X-Macro 注册不变
- 任何算术指令的计算结果不变

## 关键约束（MUST/MUST NOT）

- MUST 保持所有算术指令的计算结果不变
- MUST 保持 `ptx_op.def` X-Macro 注册不变
- MUST 保持 `instruction_factory.cpp` 分派逻辑不变
- SHOULD 按运算类型分组（基础/乘除/扩展）而非按文件大小拆分
- MUST NOT 修改测试文件
- MUST NOT 修改 `instruction_handlers.h` 中的 handler 类声明
- MUST NOT 修改 `utils/arithmetic_utils.h`（已有文件，不混淆）
