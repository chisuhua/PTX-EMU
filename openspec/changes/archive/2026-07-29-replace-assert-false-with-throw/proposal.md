# replace-assert-false-with-throw - Proposal

## Why

`src/ptx_ir/statement_context.cpp:19` 存在 `assert(false && "Unknown StatementType")`。
`assert(false)` 在 Release 构建（`-DNDEBUG`）中被编译器完全消除，导致执行流静默
落入 `default` 分支后的未定义行为路径——返回 `"invalid"` 字符串但调用方无感知。

`include/ptxsim/ptx_exceptions.h:5` 注释明确说明："提供运行时错误报告机制，
**替代 assert(false) 和 TODO 注释**"——项目已有既定替代方案。

全项目 grep 确认仅此 1 处 `assert(false)` 残留。

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-21`

## What Changes

- **替换** `statement_context.cpp:19` 的 `assert(false && "Unknown StatementType")`
  为 `throw PtxEmuException(...)`，在异常消息中包含 `StatementType` 的数值
- **确认** 全项目无其他 `assert(false)` 残留（已验证仅 1 处）
- **保留** 正常 `assert` 使用（非 `false` 触发的断言不受影响）

## Capabilities

### New Capabilities

（无新增能力——这是防御性编程改进，不引入新功能）

### Modified Capabilities

- `statement-context-toString`: `S2s()` 函数在遇到未知 `StatementType` 时从
  静默返回 `"invalid"` 改为抛出 `PtxEmuException`，确保 Debug 和 Release 构建行为一致

## Impact

**受影响代码**：
- `src/ptx_ir/statement_context.cpp`（1 处修改：line 19 的 `assert(false)` → `throw`）

**不受影响**：
- `include/ptxsim/ptx_exceptions.h`（复用已有 `PtxEmuException`，不新增异常类型）
- 其他所有 `assert(condition)` 的正常使用
- 函数签名不变
- 编译选项不变

**依赖**：
- 无前置 change 依赖，可独立执行

**工时**: 0.5h（修改 + 验证）

## Design-Time Checklist

- [ ] 确认 `PtxEmuException` 构造函数签名（`const std::string& message`,
      `PtxEmuErrorCode error_code = INTERNAL_ERROR`）
- [ ] 确认 `statement_context.cpp` 已 include 或可 include `ptx_exceptions.h`
- [ ] 确认 `S2s()` 的调用方不会因异常传播导致资源泄漏（无裸指针持有）
- [ ] 确认全项目 `assert(false` grep 结果仅此 1 处
