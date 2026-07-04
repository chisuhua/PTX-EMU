# Phase 0: Blackwell tcgen05 Infrastructure (TMA + TMEM + Cluster + Async Queue)

> **架构决策**: 本 change scope 由 [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) 锁定。
>
> **本 change 仅交付 Phase 0（基础设施 9 commits）**；Phase 1-3（tcgen05.mma/ld/st/commit/wait/GEMM e2e）由独立 OpenSpec change `implement-wmma-tensor-core-tcgen05` 交付，本 change archive 后 propose。
>
> **前置依赖**:
> - `replace-silent-stub-failures` (archived 2026-07-04)：建立 `WmmaHandler` 抛 `UnsupportedInstructionException` 合约
> - ADR-0016（Accepted 2026-07-04）：架构依据
> - `UnsupportedInstructionException` 基础设施已就绪

## Why

本 change 是 `implement-wmma-tensor-core-tcgen05` (Phase 1-3) 的**硬前置基础设施**。

`tcgen05.mma` 需要 4 个 Blackwell 新范式基础设施，pre-Blackwell 不存在：

1. **TMA descriptors**：per-CTA tensor descriptor 表，配合 `tcgen05.ld/st`
2. **Tensor Memory (TMEM)**：per-CTA 的 256 slot × 128 byte = 32KB 存储层
3. **Cluster mode**：`cta_cluster_arrive`/`wait` 同步原语（`cta_group::1` 不需要 distributed_smem，留给 `cta_group::2`）
4. **Async tensor core queue**：per-CTA commit-group counter + wait-aware 调度

**不先建基础设施就写 handler 的后果**：
- workarounds 会被后期架构返工删除（ptx-lessons-learned §3）
- 跨模块状态翻译审计缺失 → lessons-learned §1 的 #1 bug 模式
- 集成测试无法隔离 → 任何 regression 都污染整个 feature

## What Changes

### Phase 0.1：TMA descriptors (Fix #5)
- `src/ptxsim/memory/tma_descriptor.{h,cpp}`（~800-1200 LoC）
- 解析 `cuda::tma::desc` 二进制布局（TensorMap header + swizzle + strides）
- 提供 `tma_descriptor::load(...)` / `store(...)` 抽象
- 拦截 fake `cudaMemcpy` 时识别 descriptor 拷贝
- ≥10 swizzle/stride 组合的单测

### Phase 0.2：Tensor Memory (TMEM) (Fix #6)
- `src/ptxsim/memory/tmem.{h,cpp}`（~600-800 LoC）
- per-CTA 的新存储层，pre-Blackwell 没有
- 256 slot × 128 byte / slot = 32 KB per CTA
- 与 shared memory 平行，不互通

### Phase 0.3：Cluster mode (Fix #7, simplified)
- `src/ptxsim/cluster/cluster_context.{h,cpp}`（~300-400 LoC）
- `cta_cluster_arrive()` / `cta_cluster_wait()` 同步原语
- **Deferred**: distributed_smem（when cta_group::2 needed; Oracle review 简化，从 800-1200 削减至 arrive/wait only）

### Phase 0.4：Async tensor core queue (Fix #8)
- `src/ptxsim/async/tc_queue.{h,cpp}`（~800-1200 LoC）
- per-CTA 命令队列（commit-group counter）
- `tc_queue.commit(group_id)` / `tc_queue.wait(group_id)` 同步原语
- 与现有 `WarpState` 集成（独立抽象层，不冲突 per-thread PC，per ADR-0012）
- **关键审计**：commit 前必须跑 `state-modification-audit` skill（lessons-learned §1）

### Phase 0.5：逐子系统集成到 CTAContext (Fix #9a, #9b, #9c, #9d)
- 4 个微 commit，每个只集成一个子系统引用
- 修改 `src/ptxsim/core/cta_context.{h,cpp}`
- **Revert unit = 整体 Phase 0.5**（per design.md Decision: 0.5.1-0.5.4 因 `TcQueue::enqueue_mma()` 写 TMEM 不可独立 revert）

## Non-Goals

### 显式拒绝（ADR-0016 锁定，留待后续 change 处理）

- **不实现 `tcgen05.mma.*`**（在 `implement-wmma-tensor-core-tcgen05` change Phase 1）
- **不实现 `tcgen05.ld` / `tcgen05.st`**（在 Phase 1-3 change Phase 2）
- **不实现 `tcgen05.commit` / `tcgen05.wait`**（在 Phase 1-3 change Phase 2）
- **不实现 GEMM e2e kernel**（在 Phase 1-3 change Phase 3）
- **不修改 `tensor.cpp` → `wmma.cpp` rename**（在 Phase 1-3 change Phase 1.1）
- **不修改 AGENTS.md**（在 Phase 1-3 change Phase 3.3-3.4）

### 本 change 范围内锁定

- **pre-Blackwell WMMA 行为不变**：仍抛 `UnsupportedInstructionException`（per `replace-silent-stub-failures`）
- **stub-explicit-failure spec**：本 change **无 delta**（行为不变）
- **不支持 distributed_smem**（Phase 0.3 仅 arrive/wait）
- **TMA host API 拦截**：Phase 0 用 fake descriptor（手工填值），host API 拦截（候选 ADR-0017）
- **sm_120 sparse / FP4 / mxfp8**：留待后续 change

## Goals

1. TMA descriptor 解析单元测试覆盖 ≥ 10 种典型 swizzle/stride 组合
2. TMEM 单元测试验证 256 slot × 128 byte 容量 + 读写一致性 + cross-CTA 隔离
3. cluster mode 单元测试验证 cta_cluster_arrive/wait 正确性
4. async queue 单元测试验证 commit-group counter + wait-aware 调度
5. state-modification-audit 输出：`commit_group_counter` / `is_blocked` 写点 ⊆ `design.md §Decision 7 Implementation contract`
6. 4 个子系统 + 4 个集成 commit 全部独立可构建（部分集成整体 revert 接受）
7. baseline worktree 对比：无新增 FAIL

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| TMA descriptor 解析与 NVIDIA 实际二进制布局不匹配 | **Critical** (Oracle review) | Phase 0.1 用 NVIDIA PTX ISA §9.7 TensorMap 字段定义手工构造 descriptor 字节；标注为 unverified-against-hardware |
| TcQueue → WarpState 状态翻译 bugs (lessons-learned §1) | **Critical** (#1 failure mode in codebase) | design.md Decision 7 复用 `BAR_SYNC` + `BarrierModule::release_warp_barrier`；tasks.md Phase 0.4.5 强制 `state-modification-audit` skill |
| cluster mode + 现有 CTAContext 集成复杂 | High | Phase 0.3 子系统先 unit test 验证隔离行为；Phase 0.5.3 集成测试 |
| async queue 与 WarpState 集成产生 invariant 冲突 | High | TcQueue 是独立抽象层（WarpState references but does not embed），per design.md Decision 2 |
| Phase 0 工程量大（~3000-5000 LoC，9 commits） | Medium | 拆为 4 standalone (0.1-0.4) + 4 micro (0.5.1-0.5.4) commits；cluster 简化（800-1200 → 300-400 LoC） |
| Phase 0.5 micro-commit 不可独立 revert | Medium | tasks.md Phase 0.5 顶部已说明：revert unit = 整体 Phase 0.5（4 commits 整体回退至 0.4.7 后状态） |
| cute header sm_100 编译性不明确 | Medium | Phase 0 前手工 `nvcc -ptx` spike 验证 `bench/cute/include/`；失败则 propose `fix-cute-sm100-headers`（design.md Open Question #5） |
| TMA host API 拦截策略不明确 | Medium | Phase 0 用 fake descriptor；ADR-0017 候选后续单独决策 |

## Capabilities

### New Capabilities

- `wmma-tensor-core-infrastructure`: Blackwell `tcgen05.*` infrastructure 基础设施
  (TMA + TMEM + cluster + async queue)。本 change 仅交付此 capability 的基础设施层；
  feature 层（mma/ld/st/commit/wait）由 `implement-wmma-tensor-core-tcgen05` 交付。

### Modified Capabilities

无行为变化的 capability 修改。本 change 不修改 `wmma-tensor-core` 或 `stub-explicit-failure`
spec，因为 Phase 0 不改变 WmmaHandler 行为。

## Quality Gates (Phase 0 → Phase 1-3 入口)

> **Gate 全部通过才能 propose Phase 1-3 change 进入 Active 状态**

| Gate | 命令 | 阈值 | 原理 |
|------|------|------|------|
| **G1** 回归测试 | `ctest -L "unit;integration;e2e" 2>&1 \| grep -c "^FAILED"` | `== 0` | 不打破现有 165 ctest |
| **G2** baseline diff | `diff <(.worktrees/fix-pre-p0-baseline/build/ctest) (build/ctest)` | 0 new FAIL | lessons-learned §4 基线 worktree 对比 |
| **G3** state-modification-audit | `state-modification-audit` skill 输出 | `commit_group_counter` / `is_blocked` 写点 ⊆ `design.md §Decision 7` 声明集合 | lessons-learned §1 强制审计 |
| **G4** artifacts tracked | `git ls-files openspec/changes/implement-wmma-tensor-core-phase-0-infra/` | 非空 | lessons-learned §6 / Checklist E |
| **G5** Oracle re-review TMA | (manual) 对照 NVIDIA PTX ISA §9.7 TensorMap | TMA 解析逻辑站得住脚 | Critical risk #1 独有 |
| **G6** sanity 全套 | `./scripts/sanity.sh` | 0 unexpected FAIL | 含 PTX 语法测试 |
| **G7** cute spike | `nvcc -ptx` 验证 `bench/cute/include/` 编译 | exit 0 | design.md Open Question #5 |

## Impact

**新建文件**:
- `src/ptxsim/memory/tma_descriptor.{h,cpp}` (Phase 0.1)
- `src/ptxsim/memory/tmem.{h,cpp}` (Phase 0.2)
- `src/ptxsim/cluster/cluster_context.{h,cpp}` (Phase 0.3)
- `src/ptxsim/async/tc_queue.{h,cpp}` (Phase 0.4)
- `tests/unit/memory/test_tma_descriptor.cpp` (Phase 0.1)
- `tests/unit/memory/test_tmem.cpp` (Phase 0.2)
- `tests/unit/cluster/test_cluster_mode.cpp` (Phase 0.3)
- `tests/unit/async/test_tc_queue.cpp` (Phase 0.4)
- `tests/integration/{tma,tmem,cluster,async}/test_*.cpp` (Phase 0.5)

**修改文件**:
- `src/ptxsim/core/cta_context.{h,cpp}` (Phase 0.5: 4 micro-commits 加成员引用)
- `src/CMakeLists.txt` (新增 4 source)
- `tests/unit/CMakeLists.txt` + `tests/integration/CMakeLists.txt` (新增 test 注册)
- `docs/architecture/sm90_100.md` (Phase 0 起引用 ADR-0016)

**影响范围**:
- 现有 165 ctest (无变化 — Phase 0 单元测试独立于现有测试)
- cute_rmsnorm 等 e2e 测试（无变化 — 不依赖 WMMA 也不依赖新基础设施）
- Multi-PTX warning (`PTX_WARN_EMU` Fix #3)（无变化）
- WmmaHandler behavior（**不变** — Phase 0 仅添加基础设施，不改 handler）
