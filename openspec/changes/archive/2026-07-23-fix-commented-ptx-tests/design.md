## Context

Debt Audit 2026-07-02 §P0-C3 报告 `tests/unit/CMakeLists.txt` 中 7 个 PTX 单元测试被注释掉（line 432-472）。这些测试覆盖了 integer、float、extended、bitwise、cvt、ld_st、cvta 七类基础 PTX 指令，在原 `reference/` 目录迁移过程中未完成恢复。

## Goals / Non-Goals

**Goals:**
- 恢复 7 个被注释的 `add_catch_test` 注册
- 更新测试代码以匹配当前 API（StatementContext、ptxsim::testing 命名空间等）
- 全部 7 个测试绿色通过
- ctest 标签恢复（`unit;ptx;integer` 等）

**Non-Goals:**
- 不新增额外的测试用例
- 不修改被测试的生产代码
- 不重构测试框架

## Decisions

| # | 决策 | 依据 |
|---|------|------|
| 1 | 保留原测试文件内容骨架，仅适配当前 API | 最小化改动范围，专注恢复而非重写 |
| 2 | 使用 `ptxsim::testing` 命名空间工具（`make_*`、`step_warp`、`setup_pred`） | 当前 integration 测试标准模式 |
| 3 | 不迁移至 integration/ 目录 | 原注册在 `tests/unit/CMakeLists.txt`，属直接单元测试范畴 |

## Migration Plan

1. 检查原测试文件是否存在于 `tests/reference/` 或 `tests/unit/` 下对应源文件
2. 恢复 CMakeLists.txt 中的 `add_catch_test` 行
3. 修复编译错误（适配当前 API 签名）
4. 逐个验证测试通过
5. 运行 `./scripts/sanity.sh` 全量验证无回归

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|---------|
| 测试引用已删除的函数/API | 提前 grep 确认被调用 API 仍存在 |
| 测试与当前 instruction 行为不一致 | 逐个验证失败原因，区分"测试需适配"vs"模拟器行为变更" |
| 恢复后某测试失败但非本 change 责任 | 标记为已知问题，commit message 说明