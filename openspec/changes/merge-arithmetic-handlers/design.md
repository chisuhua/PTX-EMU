# merge-arithmetic-handlers - Design

## Overview

将 3 个算术 handler 文件（`arithmetic.cpp` 478 行、`arithmetic_ext.cpp` 764 行、`arithmetic_muldiv.cpp` 490 行，共 1732 行）合并为单一 `arithmetic.cpp`，提取共享 operand 处理逻辑为 helper 函数，统一类型分派，删除大量注释掉的旧代码。合并后预期总行数减少 ≥ 15%（目标 ≤ 1472 行）。

当前状态：15 个 handler 分布在 3 文件中，每个 handler 独立实现 `getBytes()` + `is_float` + `is_signed` 类型解析、`memcpy` operand 读写、f16->f32 转换、`.cc` 条件码更新等重复逻辑。

## Design Decisions

### 决策 1: 合并策略 - 按运算类型分组

**选择**: 合并为单一 `arithmetic.cpp`，内部按运算类型分组组织：

```cpp
// arithmetic.cpp 结构（合并后）
// ============ Shared Helpers ============
// extract_binary_operands(), apply_float_op(), apply_int_op(),
// update_cc_flags(), f16_op_dispatch()

// ============ Group 1: Add/Sub/Neg/Abs (基础算术) ============
// AddHandler, SubHandler, NegHandler, AbsHandler

// ============ Group 2: Mul/Div/Mad/Min/Max/Rem (乘除运算) ============
// MulHandler, DivHandler, MadHandler, MinHandler, MaxHandler, RemHandler

// ============ Group 3: Addc/Subc/Mul24/Mad24/Fma (扩展算术) ============
// AddcHandler, SubcHandler, Mul24Handler, Mad24Handler, FmaHandler
```

**理由**:
- improvement 要求"按运算类型分组而非按文件大小拆分"
- 单文件便于 handler 间共享 helper，减少重复
- 分组注释保持代码组织清晰

**替代方案**:
- A. 保持 3 文件但提取共享 helper 到 `arithmetic_helpers.h` -> 仍然多文件管理复杂
- B. 按 handler 数量均分 -> 违反"按运算类型分组"要求
- C. **采用**: 单文件 + 分组注释 + 共享 helper

### 决策 2: 共享 Helper 提取

**选择**: 提取以下 5 个 helper 函数到 `arithmetic.cpp` 文件顶部（或 `utils/arithmetic_helpers.h`）：

1. `extract_binary_operands(operands, qualifiers)` -> 返回 `{dst, src1, src2, bytes, is_float, is_signed}`
2. `apply_float_binary_op<T>(src1, src2, op)` -> 浮点二元运算模板（消除 f16/f32/f64 switch 重复）
3. `apply_int_binary_op<T>(src1, src2, is_signed, op)` -> 整数二元运算模板
4. `update_cc_flags(context, result, a, b, is_signed, bytes)` -> 条件码更新逻辑
5. `f16_binary_op(h1, h2, op)` -> f16->f32 计算 -> f16 转换

**理由**:
- 这 5 个模式在 15 个 handler 中重复出现
- 模板化消除 `switch(bytes)` 重复
- f16 转换逻辑在 Mul、Fma、Add 等至少 5 处重复

**实现伪码**:
```cpp
namespace arith_helpers {
    struct BinaryOps {
        void* dst; void* src1; void* src2;
        int bytes; bool is_float; bool is_signed;
    };
    BinaryOps extract_binary(void** operands, const std::vector<Qualifier>& quals);

    template<typename Op>
    void apply_float(void* dst, void* s1, void* s2, int bytes, Op op);

    template<typename Op>
    void apply_int(void* dst, void* s1, void* s2, int bytes, bool is_signed, Op op);

    void update_cc(ThreadContext* ctx, int64_t result, int64_t a, int64_t b,
                  bool is_signed, int bytes);
}
```

### 决策 3: 注释代码清理

**选择**: 删除 `arithmetic.cpp` 中 ~350 行注释掉的旧 `process_binary_arithmetic` 模板代码（line 9-109 + 其他散落注释）

**理由**:
- 注释代码无实际作用，增加文件体积和阅读干扰
- Git 历史保留完整旧代码，可随时恢复
- 合并目标包括"总代码行数减少 ≥ 15%"

### 决策 4: Helper 放置位置

**选择**: 优先放在 `arithmetic.cpp` 文件内匿名 namespace；如需跨文件共享则放 `utils/arithmetic_helpers.h`

**理由**:
- 合并后仅 1 个文件使用这些 helper，无需单独头文件
- 匿名 namespace 避免链接器符号冲突
- 如未来其他 handler 需要复用，再提取到头文件

**替代方案**:
- A. 直接放 `utils/arithmetic_utils.h` -> 当前已有此文件但内容不同，避免混淆
- B. 新建 `utils/arithmetic_helpers.h` -> 过度工程化，当前无跨文件需求
- C. **采用**: 文件内匿名 namespace

### 决策 5: CMakeLists.txt 更新

**选择**: 从 `src/CMakeLists.txt` 移除 `arithmetic_ext.cpp` 和 `arithmetic_muldiv.cpp` 的引用

**理由**:
- 文件删除后必须更新 CMake，否则构建失败
- 当前引用位置：line 124 (`arithmetic.cpp`)、line 125 (`arithmetic_muldiv.cpp`)、line 144 (`arithmetic_ext.cpp`)
- 合并后保留 line 124 的 `arithmetic.cpp`，移除 line 125 和 144

## Implementation Plan

### Phase 1: 提取共享 Helper（45 min）
1. 在 `arithmetic.cpp` 顶部匿名 namespace 中实现 5 个 helper 函数
2. 编译验证 helper 语法正确
3. 暂不修改 handler，确保旧代码仍编译

### Phase 2: 迁移 Handler 到使用 Helper（60 min）
1. 逐个迁移 15 个 handler 使用新 helper（按组：基础 -> 乘除 -> 扩展）
2. 每迁移一组后编译验证
3. 迁移过程中删除 `arithmetic.cpp` 的注释代码

### Phase 3: 合并文件（30 min）
1. 将 `arithmetic_ext.cpp` 的 5 个 handler 迁移到 `arithmetic.cpp` Group 3
2. 将 `arithmetic_muldiv.cpp` 的 6 个 handler 迁移到 `arithmetic.cpp` Group 2
3. 删除 `arithmetic_ext.cpp` 和 `arithmetic_muldiv.cpp`
4. 更新 `src/CMakeLists.txt`（移除 2 行）

### Phase 4: 验证（30 min）
1. 全量编译 + ctest 验证
2. 确认行数减少 ≥ 15%
3. 确认所有算术指令行为不变

## Testing Strategy

| 测试场景 | 命令 | 预期 |
|---------|------|------|
| 编译 | `cmake --build build` | 编译通过，无警告 |
| 算术 unit | `ctest -R "unit_.*add\|unit_.*sub\|unit_.*mul\|unit_.*div\|unit_.*mad\|unit_.*fma\|unit_.*min\|unit_.*max\|unit_.*rem\|unit_.*neg\|unit_.*abs"` | 全绿 |
| 算术 integration | `ctest -R "integration_.*add\|integration_.*sub\|integration_.*mul\|..."` | 全绿 |
| 全量回归 | `ctest --output-on-failure` | 全绿 |
| 行数减少 | `wc -l arithmetic.cpp` | ≤ 1472 行（1732 × 0.85） |

### 行为不变性验证

合并前后对以下指令执行 golden value 对比：
- `add.u32`, `add.s32`, `add.f32`, `add.f16`, `add.cc.u32`
- `sub.*`, `mul.*`, `div.*`, `mad.*`, `fma.*`
- `neg.*`, `abs.*`, `min.*`, `max.*`, `rem.*`
- `addc.*`, `subc.*`, `mul24.*`, `mad24.*`

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| Helper 模板类型推导失败 | 编译错误 | 逐 handler 迁移，每步编译验证 |
| 行为微妙变化（如溢出处理） | 计算结果不一致 | 分组迁移 + 每组后运行测试 |
| 文件过大影响可读性 | 维护困难 | 分组注释 + helper 抽取减少重复；预期 ≤ 1472 行 |
| 删除注释代码丢失参考 | 无实际影响 | Git 历史保留完整旧代码 |

## Open Questions

1. **helper 是否应该放 `utils/arithmetic_helpers.h` 以便未来复用？**
   - 当前无跨文件需求，匿名 namespace 足够
   - 如未来 bitwise.cpp 等需要复用，再提取

2. **是否将 `arithmetic_utils.h` 中已有函数也整合？**
   - Out Scope：不修改 `utils/arithmetic_utils.h`，仅新增 helper

## 关联文档

- `improvements/merge-arithmetic-handlers.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-3`：原债务条目
- `src/ptxsim/instructions/AGENTS.md`：指令 handler 架构
- `include/ptx_ir/ptx_op.def`：X-Macro 注册（不变）
