# expand-e2e-divergence-coverage

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-24
**阶段**: default | **分类**: core-test
**类型**: test

## 架构依据

- `tests/e2e/divergence/test_divergence.cu` 333 行，含 **20 个 TEST_CASE/SECTION**
- 已有 8 个 divergence kernel + 1 个 barrier kernel：
  - `divergence_if_else`, `divergence_multi_path`, `divergence_nested_if`, `divergence_loop_if`
  - `divergence_uneven_loop`, `divergence_mixed`, `divergence_reduction`, `divergence_barrier_sync`
- 但仅 1 个非 barrier E2E 场景被原始审计标记为不足（`barrier_sync` 标记为"已知限制"）
- SIMT v2 收敛验证（Phase 7）后，需要更多边界场景覆盖

## 范围

- **In Scope**:
  - 添加 3-5 个新 divergence E2E kernel：
    - warp 内 non-uniform branch + 立即 reconvergence
    - 深层嵌套 divergence（3+ 层）
    - divergence + memory access 交叉
    - 非 32 线程倍数 CTA 的 divergence 边界
  - 修复或移除 `barrier_sync` 的"已知限制"标记
- **Out Scope**:
  - 不修改 SIMT stack 实现
  - 不改变现有 8 个 kernel 的行为
  - 不添加 performance benchmark

## 关键场景

- GIVEN 新 divergence kernel, WHEN nvcc 编译 + 模拟执行, THEN 输出与预期一致
- GIVEN 深层嵌套 divergence, WHEN SIMT stack push/pop, THEN reconvergence 点正确

## 技术约束

- MUST 使用 CUDA C++ 编写（.cu 文件）
- MUST 通过 nvcc 编译为 PTX
- MUST 通过 fake libcudart.so 拦截执行
- SHOULD 遵循现有 test_divergence.cu 的测试模式

## 验收标准

- 新增 ≥ 3 个 divergence E2E 测试
- 所有 E2E 测试通过（ctest -L e2e）
- `barrier_sync` 的"已知限制"状态有明确结论（修复或文档化）
