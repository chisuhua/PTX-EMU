# strengthen-pod-tests - Proposal

## Why

`tests/unit/` 中与 context 相关的测试文件偏浅，仅验证"对象可构造"，不验证行为正确性：

| 文件 | 行数 | 问题 |
|------|------|------|
| `tests/unit/warp/test_warp_context.cpp` | 98 行 | 仅验证构造 + 基本 setter |
| `tests/unit/warp/test_sm_context.cpp` | 126 行 | 仅验证构造 |
| `tests/unit/ptx/test_cvt_context.cpp` | 274 行 | 部分行为但边界不足 |
| `tests/unit/cpptlm/test_smcontext_injection.cpp` | 375 行 | 相对完整但缺错误路径 |

原始审计指出 `tests/unit/contexts/` 下曾有 7 个 <50 行 POD 测试（目录已重组，但浅测试模式仍存在）。浅测试无法捕获：
- 状态转换正确性（如 active_mask 的 overwrite vs OR 语义）
- 边界条件（如 0 线程 warp、全活跃 warp）
- 错误路径（如非法 lane index、空指针操作）

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-6`

## What Changes

- **深化** 现有 context 测试，补充行为验证（状态转换、边界条件、错误路径）
- **新增** `execute_warp_instruction` 驱动的集成验证（test-coverage-enforcer 要求）
- **确保** 每个测试文件 ≥ 50 行有效断言

## Capabilities

### New Capabilities
- `unit-context-behavioral-testing`: context 对象行为测试（状态转换、边界、错误路径）
- `unit-context-integration-bridge`: unit 测试与 `execute_warp_instruction` 集成测试的对应关系

### Modified Capabilities
- `unit-warp-context-test`: `test_warp_context.cpp` 从构造验证扩展为行为验证
- `unit-sm-context-test`: `test_sm_context.cpp` 从构造验证扩展为行为验证
- `unit-cvt-context-test`: `test_cvt_context.cpp` 补充边界条件测试
- `unit-smcontext-injection-test`: `test_smcontext_injection.cpp` 补充错误路径测试

## Impact

**受影响代码**：
- `tests/unit/warp/test_warp_context.cpp`（补充行为断言）
- `tests/unit/warp/test_sm_context.cpp`（补充行为断言）
- `tests/unit/ptx/test_cvt_context.cpp`（补充边界条件）
- `tests/unit/cpptlm/test_smcontext_injection.cpp`（补充错误路径）
- `tests/integration/` 对应目录（新增 `execute_warp_instruction` 集成测试，如缺失）

**不受影响**：
- `src/ptxsim/` 被测源码（不修改）
- 现有 integration/e2e 测试覆盖的场景（不重复）
- 测试目录结构（不新建）

**依赖**：
- 无前置 change 依赖，可独立执行
- test-coverage-enforcer 技能：新增 unit 测试必须有对应 integration 测试

**工时**: 2-3h（4 个文件深化 + 对应集成测试）

## Design-Time Checklist

- [ ] 确认每个 context 测试文件的现有断言数量和覆盖范围
- [ ] 确认每个新增 unit 测试场景是否有对应 integration 测试（test-coverage-enforcer）
- [ ] 确认不重复已有 integration/e2e 测试覆盖的场景
- [ ] 确认 Catch2 测试标签格式 `<type>;<subject>`
- [ ] 确认每个文件最终 ≥ 50 行有效断言
