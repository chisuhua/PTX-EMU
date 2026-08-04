## Context

`include/ptxsim/testing/memory_test_utils.h` 当前 18 个 include（与原基线一致）。原 change `2026-07-29-reduce-memory-test-utils-includes` archive 后 tasks.md 全勾选但代码未 apply。

当前 18 个 include 分类：
- 第三方 1 个：`catch_amalgamated.hpp`（不可精简）
- 项目头文件 8 个：`ptxsim/cta_context.h`, `ptxsim/warp_context.h`, `ptxsim/sm_context.h`, `ptxsim/instruction_factory.h`, `ptx_ir/operand_context.h`, `ptx_ir/statement_context.h`, `memory/resource_manager.h`, `register/register_bank_manager.h`
- 标准库 9 个：`<algorithm>`, `<cstdint>`, `<cstdlib>`, `<map>`, `<memory>`, `<string>`, `<vector>` 等

## Goals / Non-Goals

**Goals:**
- include 数量 18 → ≤12（净减 ≥6，验收 30%）
- 测试编译时间缩短
- 所有使用 memory_test_utils.h 的测试文件编译通过

**Non-Goals:**
- 改 memory_test_utils.h 的任何函数签名
- 影响使用该头文件的测试文件行为
- 引入新头文件

## Decisions

1. **优先标准库 include 移到 .cpp**：9 个标准库 include 中仅 .cpp 使用的 → 移到 .cpp
   - Rationale: 标准库 include 改动面最小
   - Alternatives considered: 全部前向声明 → 标准库无前向声明概念

2. **项目头文件前向声明**：8 个项目头文件中，指针/引用类型用前向声明；inline 函数体内调用方法的类型必须保留完整 include
   - Rationale: 测试工具头文件的 inline 函数需完整类型支持；其他可前向声明化
   - Alternatives considered: 全 #include → 无收益

3. **Catch2 测试宏依赖保留**：`catch_amalgamated.hpp` 必须保留（Catch2 框架强制依赖）

## Risks / Trade-offs

- [Risk] inline 函数体内调用方法需完整类型，前向声明化失败 → Mitigation: 分析每个 inline 函数的依赖图，标注依赖完整类型的 include
- [Risk] Catch2 测试 ASSERT_* 宏展开需要特定头文件 → Mitigation: catch_amalgamated.hpp 永不删除

## Migration Plan

1. 列出所有 18 个 include 的使用方式
2. 移除仅 .cpp 使用的标准库 include
3. 对项目头文件添加前向声明并移除可精简的 #include
4. 每步 ctest -L unit 验证
5. Rollback: git revert

## Open Questions

- 无