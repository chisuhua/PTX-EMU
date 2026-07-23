## Context

ClusterContext 模块自 2026-07-04 `implement-wmma-tensor-core-phase-0-infra` 归档以来，完整实现 + 测试通过，但**生产代码零调用**。本 change 按用户决策"接入 tcgen05"，采用最小 opt-in 接入路径。

### 探索发现（per explore agent 调研）

| 状态 | 文件:行 | 描述 |
|------|---------|------|
| ✅ 实现完整 | `src/ptxsim/cluster/cluster_context.{h,cpp}` | ClusterContext + cta_cluster_arrive/wait |
| ✅ API 暴露 | `include/ptxsim/cta_context.h:109-122` | init_cluster_context/cluster_context/has_cluster_context |
| ❌ 0 生产调用 | 全代码库 | init_cluster_context/cta_cluster_arrive/wait 均 0 调用 |
| ❌ 未接入 tcgen05 | `wmma.cpp:502-553` | commit/wait handler 忽略 cluster_context |
| ⚠️ KernelContext 已有字段 | `include/ptx_ir/kernel_context.h` | usesClusterScope / clusterDimX（未消费） |

### 接入策略：opt-in 模式（避免破坏现有测试）

```cpp
// 修改后的 tcgen05.commit (wmma.cpp)
void WmmaHandler::processTcgen05Commit(/* ... */) {
  // 现有逻辑：cta->tc_queue().commit()
  cta->tc_queue().commit(/* ... */);
  
  // 新增 opt-in cluster sync
  if (cta->has_cluster_context()) {
    PTX_DEBUG_EMU("tcgen05.commit: cluster arrive cta=%d", cta->get_id());
    cta->cluster_context().cta_cluster_arrive(cta->get_id());
  }
}
```

**关键**: `has_cluster_context()` 返回 false 时跳过（现有 cta_group::1 测试无影响）

### 决策矩阵

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **A. opt-in (has_cluster_context check)** | ✅ 不破坏现有测试 ✅ 增量接入 | ⚠️ 多 1 个 branch | ✅ **采纳** |
| B. 强制要求 cluster context | 简洁 | ❌ 破坏所有现有 tcgen05 测试 | ❌ 拒绝 |
| C. 删除 ClusterContext | 简单 | ❌ 违反用户决策 | ❌ 拒绝 |
| D. 完整接入（cudaFuncSetAttribute + bar.cluster + multi-CTA） | 完整 cluster 支持 | ❌ 需 6-10h 工作量 | ❌ 延期到 ADR-0018 |

### Metis Review

本 change 经过 explore agent 完整调研 + 风险评估（R1 死锁 / R2 CTA ID / R3 测试回退），属 lessons-learned Checklist H 范畴。决策已锁定为方案 A。

## Goals / Non-Goals

### Goals

1. **修改 `src/ptxsim/core/gpu_context.cpp`**：在 CTA 创建后添加 opt-in `init_cluster_context` 调用（基于 `KernelContext::usesClusterScope`）
2. **修改 `src/ptxsim/instructions/wmma.cpp::processTcgen05Commit`**（line 502-526）：添加 opt-in cluster arrive
3. **修改 `src/ptxsim/instructions/wmma.cpp::processTcgen05Wait`**（line 528-553）：添加 opt-in cluster wait
4. **新增 oracle 测试** `tests/unit/cluster/test_cluster_tcgen05_integration.cpp`：
   - 测试 1: 启用 cluster context → arrive + wait 正常调用
   - 测试 2: 禁用 cluster context → arrive/wait 跳过（opt-in 行为）
5. **同步 2 个文档**：ADR-0016 + post-tcgen05-roadmap

### Non-Goals（明确排除）

1. ❌ **`cudaFuncSetAttribute` cudart 拦截**：延期到 ADR-0018 change（需 cudaLaunchKernel 改造）
2. ❌ **`bar.cluster` PTX handler**：延期到 ADR-0018 change
3. ❌ **`cta_group::2` tcgen05 路径**：延期到 ADR-0018 change
4. ❌ **分布式 SMEM 支持**：ADR-0018 scope
5. ❌ **删除 ClusterContext**：违反用户决策（接入而非删除）

## Decisions

### Decision 1: opt-in 模式（has_cluster_context check）

**Choice**: opt-in

**Rationale**：
- R3 风险（现有测试回退）必须避免
- opt-in 模式允许现有 cta_group::1 测试无修改通过
- 新增 oracle 测试验证 cluster 路径
- 增量接入，未来可演进为强制模式

### Decision 2: init_cluster_context 调用点

**Choice**: `gpu_context.cpp` CTA 创建后立即调用（基于 `KernelContext::usesClusterScope` flag）

**Rationale**：
- 用户调用 `cudaLaunchKernel` 时 kernel 信息已加载（`KernelContext`）
- 在 GPUContext 创建 CTA 时读取 cluster 配置
- 避免运行时延迟

### Decision 3: 测试策略

**Choice**: 新增 oracle 测试 + 现有测试零修改

**Rationale**：
- 现有 `tests/unit/wmma/test_wmma.cpp` 等 cta_group::1 测试覆盖主路径
- 新增 `test_cluster_tcgen05_integration.cpp` 验证 cluster 路径
- 双重覆盖：现有测试无回归 + 新测试验证新行为

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **R1: cluster_wait 死锁** | 🟡 中 | (1) 确保 cluster_wait 在 warp lock 外调用 (2) oracle 测试验证单 CTA 路径无死锁 |
| **R2: cta_id 不匹配** | 🟢 低 | (1) 使用 `cta->get_id()` 标准接口 (2) oracle 测试验证 |
| **R3: 现有 tcgen05 测试回退** | 🟢 低（opt-in 模式）| (1) `has_cluster_context()` check 跳过未初始化路径 (2) 全 ctest PASS |
| **R4: gpu_context.cpp init_cluster_context 调用时机错误** | 🟡 中 | (1) 仅在 `KernelContext::usesClusterScope=true` 时调用 (2) 默认 false（保持现有行为）|
| **R5: oracle 测试覆盖不足** | 🟢 低 | (1) 至少 2 个测试用例 (2) baseline diff 确保无回归 |

## Migration Plan

### Phase 0: Artifacts Git-Tracking + Baseline

```bash
git checkout -b feature/wire-cluster-context-to-tcgen05
git add openspec/changes/wire-cluster-context-to-tcgen05/
git commit -m "docs(openspec): add wire-cluster-context-to-tcgen05 artifacts"
```

### Phase 1: Oracle 测试先行（Fix #1 — Red Phase）

```bash
git worktree add .worktrees/cluster-tcgen05-impl feature/wire-cluster-context-to-tcgen05
cd .worktrees/cluster-tcgen05-impl

# 1.1 创建 tests/unit/cluster/test_cluster_tcgen05_integration.cpp
# 包含 2 个测试用例：
#   - test_cluster_tcgen05_arrive_when_initialized
#   - test_cluster_tcgen05_skipped_when_not_initialized

# 1.2 注册到 tests/unit/cluster/CMakeLists.txt
# add_catch_test(unit_cluster_tcgen05_integration
#     test_cluster_tcgen05_integration.cpp
# )
# set_tests_properties(unit_cluster_tcgen05_integration PROPERTIES LABELS "unit;cluster;tcgen05")

# 1.3 验证测试失败（Red Phase）：ctest -R unit_cluster_tcgen05_integration -V
# 期望: 编译失败（test_cluster_tcgen05_integration.cpp 未实现）

# Commit
git commit -am "test(cluster): add failing oracle test for ClusterContext tcgen05 integration (Fix #1)

Per Checklists A/D/TDD Red Phase:
- test_cluster_tcgen05_integration.cpp: 2 scenarios
  - cluster context initialized → arrive/wait called
  - cluster context not initialized → opt-in skip
- Registered as unit_cluster_tcgen05_integration (label: unit;cluster;tcgen05)

Test expected to fail compilation at this Phase (TDD Red Phase)."
```

### Phase 2: 接入 ClusterContext（Fix #2 — Green Phase）

```bash
# 2.1 修改 include/ptxsim/cta_context.h
# 验证 get_id() 方法存在（如果没有需添加）

# 2.2 修改 src/ptxsim/core/gpu_context.cpp
# 在 CTA 创建后（line 188-191）：
#   if (kernel_ctx.usesClusterScope) {
#     cta->init_cluster_context(blockIdx.x, kernel_ctx.clusterDimX);
#   }

# 2.3 修改 src/ptxsim/instructions/wmma.cpp::processTcgen05Commit
# 在 cta->tc_queue().commit() 之后添加：
#   if (cta->has_cluster_context()) {
#     PTX_DEBUG_EMU("tcgen05.commit: cluster arrive cta=%d", cta->get_id());
#     cta->cluster_context().cta_cluster_arrive(cta->get_id());
#   }

# 2.4 修改 src/ptxsim/instructions/wmma.cpp::processTcgen05Wait
# 在 cta->tc_queue().wait() 之后添加：
#   if (cta->has_cluster_context()) {
#     PTX_DEBUG_EMU("tcgen05.wait: cluster wait cta=%d", cta->get_id());
#     cta->cluster_context().cta_cluster_wait(cta->get_id());
#   }

# 2.5 验证 oracle 测试通过（Green Phase）
ctest -R unit_cluster_tcgen05_integration -V
# 期望: PASS

# 2.6 验证全测试无回归
ctest --output-on-failure
./tests/ptx/test_all_ptx.sh
./scripts/sanity.sh --quick

# Commit
git commit -am "feat(cluster): wire ClusterContext into tcgen05 commit/wait (Fix #2)

Changes:
- src/ptxsim/core/gpu_context.cpp: opt-in init_cluster_context when KernelContext::usesClusterScope
- src/ptxsim/instructions/wmma.cpp:
  - processTcgen05Commit: add cluster arrive (opt-in via has_cluster_context)
  - processTcgen05Wait: add cluster wait (opt-in via has_cluster_context)

Strategy: opt-in pattern preserves existing cta_group::1 tests (no regression).
New behavior enables ClusterContext infrastructure for future ADR-0018 cta_group::2 work.

Oracle test: tests/unit/cluster/test_cluster_tcgen05_integration.cpp (2 scenarios).
Full ctest PASS, sanity --quick PASS, ptx syntax tests PASS.

Per lessons-learned Checklists E/F/H.
Refs: archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/ (ClusterContext infra)
"
```

### Phase 3: 文档同步（Fix #3）

```bash
# 3.1 更新 docs/adr/ADR-0016-blackwell-only-tcgen05.md
# 添加 §2026-07-06 cluster 接入状态

# 3.2 更新 docs/dev-process/post-tcgen05-roadmap.md
# 标注 F0 (ClusterContext 接入) 完成

# Commit
git commit -am "docs(cluster): sync ADR-0016 + roadmap post-Fix #2 (Fix #3)

Per lessons-learned Checklist I + §21:"
```

### Phase 4: Archive

```bash
openspec archive wire-cluster-context-to-tcgen05 --yes
git checkout main
git merge --no-ff feature/wire-cluster-context-to-tcgen05
```

### Rollback Strategy

```bash
# 任何 Phase 失败立即 revert
git revert HEAD
cmake --build build
ctest --output-on-failure
```

## Open Questions

### OQ-1: CTA ID 接口

**Question**: `CTAContext::get_id()` 方法是否存在？

**Status**: Phase 2.1 验证。如不存在需添加 `CTAContext::cta_id_t get_id() const`。

### OQ-2: KernelContext cluster 字段

**Question**: `KernelContext::usesClusterScope` 和 `clusterDimX` 字段是否已定义？

**Status**: 已确认存在（per explore agent 报告）。Phase 2.2 验证访问。

### OQ-3: cta_cluster_wait 阻塞语义

**Question**: 当 cluster context 存在但 num_ctas=1 时，cta_cluster_wait 是否会立即返回？

**Status**: ClusterContext 实现确认：`arrived_set_.size() == num_ctas_` 立即成立（自己 arrive 后），无需 cv.wait 阻塞。无需特殊处理。