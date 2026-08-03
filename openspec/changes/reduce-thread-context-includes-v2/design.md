## Context

`include/ptxsim/thread_context.h` 当前 25 个 include（基线 21），其中：
- 标准库 11 个（`<iostream>`, `<any>`, `<array>`, `<map>`, `<memory>`, `<stack>`, `<string>`, `<unordered_map>`, `<vector>` 等）
- 项目头文件 14 个

原 change `2026-07-29-reduce-thread-context-includes` archive 后 tasks.md 全勾选但代码未 apply，需重做。最新 commit `edb9302e refactor(ptxsim): document forward declarations in thread_context.h` 已部分文档化 forward declaration 区但未实际精简 include。

## Goals / Non-Goals

**Goals:**
- include 数量 25 → ≤15（净减 ≥10，验收 30%）
- 标准库 include 优先前向声明化（仅 .cpp 使用）
- 编译通过且无新增 warning
- 修改被前向声明的头文件不触发依赖者重编译

**Non-Goals:**
- 改 ThreadContext public API
- 删除实际需要的 include
- 引入新头文件

## Decisions

1. **优先标准库 include 移到 .cpp**：11 个标准库 include 中，仅在头文件 inline 函数体内使用的 → .cpp；其余需先分析
   - Rationale: 标准库 include 改动面最小，无 ABI 影响
   - Alternatives considered: 全前向声明化 → 标准库无"前向声明"概念

2. **项目头文件前向声明**：14 个项目头文件中，仅以指针/引用参数出现的类型用前向声明；值类型成员必须保留完整 include
   - Rationale: lessons-learned §1 跨模块状态翻译站点保护：值类型不能被前向声明替代
   - Alternatives considered: 全部 #include → 无收益

3. **集中 forward declaration 区**：头文件顶部添加 `// --- Forward declarations ---` 注释块
   - Rationale: 可读性 + 与 commit `edb9302e` 已建立的模式一致
   - Alternatives considered: 散落在各处 → 难维护

## Risks / Trade-offs

- [Risk] 误把值类型 include 改为前向声明 → 编译错误或不完整类型 → Mitigation: 行级 diff 审计；逐个移除 include 后立即 ctest
- [Risk] inline 函数体内使用的方法调用需完整类型，前向声明不够 → Mitigation: Phase 2.5 明确标注 inline 函数体依赖

## Migration Plan

1. 分析所有 25 个 include 的使用方式（值类型成员 / 指针引用参数 / inline 函数体）
2. 移除非必要 include（先标准库，再项目头文件）
3. 每移除一个 include 立即 ctest 验证
4. 最终 include 数 ≤15 + 无 warning
5. Rollback: 单 include 回滚

## Open Questions

- 无