# expand-e2e-divergence-coverage - Proposal

## Why

`tests/e2e/divergence/test_divergence.cu`（333 行，20 个 TEST_CASE/SECTION）已覆盖 8 个 divergence kernel + 1 个 barrier kernel，但 SIMT v2.0 收敛验证（Phase 7）后需要更多边界场景覆盖：

- 当前仅 1 个非 barrier E2E 场景被原始审计标记为不足（`barrier_sync` 标记为"已知限制"）
- 缺少深层嵌套 divergence（3+ 层）的 E2E 验证
- 缺少 divergence + memory access 交叉场景
- 缺少非 32 线程倍数 CTA 的 divergence 边界
- 缺少 warp 内 non-uniform branch + 立即 reconvergence 场景

这些场景缺失导致 SIMT stack 的 push/pop 边界行为未被充分验证，回归风险高。

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-24`

## What Changes

- **新增** 3-5 个 divergence E2E kernel 测试场景：
  - warp 内 non-uniform branch + 立即 reconvergence
  - 深层嵌套 divergence（3+ 层）
  - divergence + memory access 交叉
  - 非 32 线程倍数 CTA 的 divergence 边界
- **修复或文档化** `barrier_sync` 的"已知限制"标记

## Capabilities

### New Capabilities
- `e2e-divergence-nonuniform-reconvergence`: non-uniform branch 后立即 reconvergence 的 E2E 验证
- `e2e-divergence-deep-nesting`: 3+ 层嵌套 divergence 的 E2E 验证
- `e2e-divergence-memory-interaction`: divergence 与 memory access 交叉的 E2E 验证
- `e2e-divergence-non32-threads`: 非 32 线程倍数 CTA 的 divergence 边界验证

### Modified Capabilities
- `e2e-divergence-test-suite`: `test_divergence.cu` 新增 ≥ 3 个 kernel + TEST_CASE

## Impact

**受影响代码**：
- `tests/e2e/divergence/test_divergence.cu`（新增 kernel 函数 + TEST_CASE）

**不受影响**：
- `src/ptxsim/` 源码（不修改 SIMT stack 实现）
- 现有 8 个 divergence kernel 的行为
- `ptx_op.def` 或任何指令 handler

**依赖**：
- 无前置 change 依赖，可独立执行
- 依赖现有 SIMT v2.0 stack 正确性（如有 bug 将被新测试暴露）

**工时**: 1-2h（编写 CUDA kernel + 验证）

## Design-Time Checklist

- [ ] 确认新 kernel 编译为 PTX 后不触发未实现指令
- [ ] 确认每个 kernel 的预期输出与 SIMT stack reconvergence 语义一致
- [ ] 确认 `barrier_sync` "已知限制"的最终结论（修复 or 文档化）
- [ ] 确认新 kernel 不与现有 8 个 kernel 行为重叠
- [ ] 确认 CMakeLists.txt 注册新测试目标（如需）
