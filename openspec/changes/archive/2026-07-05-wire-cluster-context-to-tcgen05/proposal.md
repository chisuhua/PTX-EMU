## Why

`ClusterContext` 模块（`src/ptxsim/cluster/cluster_context.{h,cpp}`）是 ADR-0016 Phase 0.3 基础设施，已完整实现 + 单元测试 + 集成测试 100% 通过，但**从未被生产代码调用**：

```bash
$ grep -rn "init_cluster_context\|cta_cluster_arrive\|cta_cluster_wait" src/ include/ \
    | grep -v "cluster/cluster_context\.\(h\|cpp\)\|cta_context\.\(h\|cpp\)\|test_cluster"
# 0 结果
```

`CTAContext::init_cluster_context()`、`cluster_context()`、`has_cluster_context()` 在生产代码中**零调用**。

### tcgen05 集成点（已识别）

`/workspace/project/PTX-EMU/src/ptxsim/instructions/wmma.cpp` 中的 tcgen05 handler 当前访问 `cta->tmem()` / `cta->tc_queue()` / `cta->tma_descriptor_store()` 但**未调用** `cta->cluster_context()`。具体接入点：

| Handler | 行号 | 当前实现 | 接入 cluster_context 后的语义 |
|---|---|---|---|
| `tcgen05.mma` | 361-420 | `cta->tmem()` 本地访问 | 不变（每个 CTA 本地数据） |
| `tcgen05.ld` | 423-461 | `cta->tma_descriptor_store()` 本地 | 不变 |
| `tcgen05.st` | 463-500 | `cta->tma_descriptor_store()` 本地 | 不变 |
| **`tcgen05.commit`** | **502-526** | `cta->tc_queue()` 入队 | **+ cta_cluster_arrive(cta_id) opt-in** |
| **`tcgen05.wait`** | **528-553** | `cta->tc_queue()` 等待 | **+ cta_cluster_wait(cta_id) opt-in** |

接入 commit/wait 是**最小集成路径**：
- `commit` 表示"本 CTA 已完成本地 commit，等待 peer"
- `wait` 表示"等所有 peer 都完成 commit"

### 关键约束（避免破坏现有测试）

- **R3 from explore agent**: "现有测试回退 — 所有 tcgen05 测试使用单 CTA 上下文，没有 init_cluster_context()。添加 has_cluster_context() 检入会破坏它们。"

**解决**: 使用 `if (cta->has_cluster_context()) { ... }` opt-in 模式 — 仅在 cluster context 已初始化时才调用。现有测试（无 cluster context）继续通过。

### 接入 vs 删除

按用户决策"接入 tcgen05"，**采用接入方案**：
- ✅ 启用 `cta_group::2` 路径基础（虽然 `cta_group::2` handler 本身延期到 ADR-0018）
- ✅ `init_cluster_context()` 至少有一个调用点（gpu_context.cpp）
- ✅ `cta_cluster_arrive/wait` 至少有一个调用点（tcgen05.commit/wait）
- ✅ ClusterContext 不再是 100% 死代码
- ⚠️ `cudaFuncSetAttribute` 拦截 + `bar.cluster` handler 延期到独立 ADR-0018 change（本 change 不触及）

## What Changes

- **修改 `src/ptxsim/core/gpu_context.cpp`**：在 CTA 创建后添加 opt-in `init_cluster_context` 调用（基于 `KernelContext::usesClusterScope`）
- **修改 `src/ptxsim/instructions/wmma.cpp`**：
  - `tcgen05.commit` handler：添加 `if (cta->has_cluster_context()) { ... arrive() ... }` (after line 523)
  - `tcgen05.wait` handler：添加 `if (cta->has_cluster_context()) { ... wait() ... }` (after line 550)
- **修改 `include/ptx_ir/kernel_context.h`（如有需要）**：确保 `usesClusterScope` 字段被 GPUContext 读取
- **新增 oracle 测试 `tests/unit/cluster/test_cluster_tcgen05_integration.cpp`**：验证接入路径
- **修改 `tests/unit/wmma/test_wmma.cpp`（如有）**：确保现有 cta_group::1 测试无回归

**BREAKING**: 无 — 使用 opt-in 模式（`has_cluster_context()` check），现有测试无变化

## Capabilities

### New Capabilities

- `cluster-context-tcgen05-wiring`: ClusterContext 接入 tcgen05 commit/wait handlers（opt-in 模式）

### Modified Capabilities

- `wmma-tensor-core`: 增加 cluster-aware commit/wait 语义（条件性）
- `docs-discoverability`: 文档说明 ClusterContext 接入状态

## Impact

**受影响的代码/文件**：

| 文件 | 改动 | 影响 |
|------|------|------|
| `src/ptxsim/core/gpu_context.cpp` | 添加 init_cluster_context opt-in 调用 | ~10 行 |
| `src/ptxsim/instructions/wmma.cpp` | commit/wait 增加 has_cluster_context check | ~10 行 |
| `tests/unit/cluster/test_cluster_tcgen05_integration.cpp` | 新增 oracle 测试 | 新文件 |
| `docs/adr/ADR-0016-blackwell-only-tcgen05.md` | 同步 cluster 接入状态 | ~5 行 |
| `docs/dev-process/post-tcgen05-roadmap.md` | 标注本 change 完成 F0 | ~3 行 |

**受影响的 ADR**：
- ADR-0016：cluster arrive/wait 接入 tcgen05（Phase 0.3 完整化）
- ADR-0018（未实施）：multi-CTA 集群同步延期到独立 change

**测试覆盖**：
- ✅ 现有 tcgen05 测试（`cta_group::1`，无 cluster context）继续通过（opt-in 模式）
- ✅ 新增 oracle 测试 `test_cluster_tcgen05_integration.cpp` 验证 cluster 路径
- ✅ `./scripts/sanity.sh --quick` 验证 0 回归

**回归风险**：
- 🟢 低：opt-in 模式 + oracle 测试双重保护
- 🟡 中：gpu_context.cpp init_cluster_context 调用路径需谨慎（仅在 `usesClusterScope=true` 时）

**Lessons-learned 集成**：
- ✅ Checklist E（artifacts 必 tracked）
- ✅ Checklist F（git verify）
- ✅ Checklist H（pre-impl review）：已通过 explore agent 完成（含 R3 风险缓解）
- ✅ Checklist G（lifecycle）

**关联 change**：
- `archive/2026-07-04-implement-wmma-tensor-core-tcgen05/` — tcgen05 主体已实施
- `archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/` — ClusterContext 基础设施
- 未来 ADR-0018 change — multi-CTA 集群同步（bar.cluster + cudaFuncSetAttribute 拦截）