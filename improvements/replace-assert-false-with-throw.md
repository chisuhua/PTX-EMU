# replace-assert-false-with-throw

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-21
**阶段**: default | **分类**: core-impl
**类型**: refactor

## 架构依据

- `src/ptx_ir/statement_context.cpp:19` 存在 `assert(false && "Unknown StatementType")`
- `assert(false)` 在 Release 构建中被 NDEBUG 消除，导致静默执行到未定义行为路径
- `include/ptxsim/ptx_exceptions.h:5` 注释明确说明："提供运行时错误报告机制，**替代 assert(false)** 和 TODO 注释"——项目已有既定替代方案

## 范围

- **In Scope**:
  - 将 `statement_context.cpp:19` 的 `assert(false && "Unknown StatementType")` 替换为 `throw UnsupportedInstructionException(...)` 或等效异常
  - 全项目扫描确认无其他 `assert(false)` 残留
- **Out Scope**:
  - 不修改 assert 的正常使用（非 false 触发的 assert 保留）
  - 不改变函数签名
  - 不引入新异常类型（复用 ptx_exceptions.h 已有类型）

## 关键场景

- GIVEN Release 构建, WHEN 遇到未知 StatementType, THEN 抛出异常而非静默继续
- GIVEN Debug 构建, WHEN 遇到未知 StatementType, THEN 行为与之前一致（异常替代 assert 崩溃）

## 技术约束

- MUST 复用 `include/ptxsim/ptx_exceptions.h` 中已有异常类型
- MUST NOT 保留任何 `assert(false)` 路径
- SHOULD 在异常消息中包含 StatementType 的数值

## 验收标准

- `grep -rn "assert(false" src/ include/` 返回 0 结果
- Debug + Release 构建均编译通过
- 相关单元测试通过（statement_context 类型转换测试）
