# Design: Blackwell tcgen05 Infrastructure (Phase 0)

> **架构依据**: [ADR-0016](../../../../docs/adr/0016-blackwell-only-tcgen05.md)
> **前置 change**: `replace-silent-stub-failures` (archived 2026-07-04)
> **后续 change**: `implement-wmma-tensor-core-tcgen05` (per `Ref:` after archive)

## Context

Blackwell `tcgen05.*` 指令集 (sm_100/sm_120) 需要 4 个新基础设施：

1. **TMA descriptors**：per-CTA tensor descriptor 解析与存储
2. **Tensor Memory (TMEM)**：per-CTA 256 slot × 128 byte 存储层
3. **Cluster mode**：`arrive`/`wait` 同步原语（distributed_smem 推迟至 `cta_group::2`）
4. **Async tensor core queue**：per-CTA commit-group counter + wait-aware 调度

这些是 pre-Blackwell 不存在的抽象层。Phase 1-3 的 `tcgen05.mma/ld/st/commit/wait` handler
**完全无法工作**直到此基础设施就绪。

## Goals / Non-Goals

**Goals**:
1. TMA descriptor 解析 + 存储，≥ 10 swizzle/stride 组合单测
2. TMEM 256 slot × 128 byte 存储 + 读写一致性 + cross-CTA 隔离
3. Cluster arrive/wait 同步原语
4. Async tc_queue commit-group counter + wait-aware 调度（复用 BAR_SYNC 路径）
5. 4 子系统独立单元测试 + 集成测试通过
6. baseline worktree 对比 0 新增 FAIL
7. state-modification-audit 通过

**Non-Goals**:
- `tcgen05.mma/ld/st/commit/wait` handler 实现（Phase 1-3 change 范围）
- `tensor.cpp` → `wmma.cpp` rename（Phase 1-3 change 范围）
- AGENTS.md 同步（Phase 1-3 change 范围）
- GEMM e2e kernel（Phase 1-3 change 范围）
- pre-Blackwell WMMA 行为变更（per ADR-0016：永久抛异常）
- sm_120 sparse / FP4 / mxfp8（后续 change）
- distributed_smem（deferred，when cta_group::2 needed）
- TMA host API (`cuda::tma::create_tensor_map`) 拦截（候选 ADR-0017）

## Decisions

### Decision 1: Phase 0 = 4 子系统 + 4 集成 commit

**Context**: `tcgen05.mma` is unusable without its supporting infrastructure.
Attempting instruction handler before infrastructure would require extensive
workarounds that get thrown away later.

**Choice**: Phase 0 builds all 4 infrastructure subsystems before any
`tcgen05` handler is written. Each subsystem is one independent commit
（0.1/0.2/0.3/0.4）。集成步骤 0.5 拆为 4 micro-commit（0.5.1/0.5.2/0.5.3/0.5.4）。

**Rationale**:
- `ptx-lessons-learned` §3：每个 commit 独立可 revert
- 基础设施 commits 是干净的 architectural units — 各自可单元测试隔离
- Future sm_120 / FP4 work builds on same infrastructure

**Alternatives considered**:
- ❌ 与 instruction handler 同步建基础设施：blurs ownership, hard to revert
- ❌ Skip 基础设施（fake everything）：产生错误结果

### Decision 2: Async queue 是独立抽象层，不嵌入 WarpState

**Context**: `WarpState.threads[i].pc` (ADR-0012) is per-thread synchronous.
tcgen05's commit-group is per-CTA async. Two different abstractions.

**Choice**: 构建 `async::tc_queue::TcQueue` 作为 CTA-level subsystem, `WarpState` 引用但不嵌入。

**Rationale**:
- 避免 invasive WarpState refactor
- `ptx-lessons-learned` §1：TcQueue 有自己的状态（commit-group counter, pending operations），需明确翻译到 scheduler-visible 状态
- Future async instructions (e.g. `cp.async.bulk`) 可共用同一队列

**Alternatives considered**:
- ❌ 嵌入 WarpState：scope creep, hard to revert
- ❌ Per-warp queue：与 CTA-level tcgen05 semantics 冲突

### Decision 3: TMA descriptor 从原始字节解析，不从 CUDA API 构造

**Context**: `cuda::tma::create_tensor_map` is a host API. Descriptor is built
by host and copied to device memory, where `tcgen05.ld` reads it. PTX-EMU
needs to intercept this copy to capture the descriptor.

**Choice**: Phase 0 支持 "fake descriptor" 模式 — test/host utility 直接构造
descriptor bytes 存入 fake device memory。`cuda::tma::create_tensor_map` 拦截
defer 到 ADR-0017。

**Rationale**:
- 限定 Phase 0 scope
- 避免 cuda runtime API 模拟复杂度
- Real descriptor interception 是独立决策

**Alternatives considered**:
- ❌ Phase 0 全 API 拦截：scope creep
- ❌ Reject 所有 `cuda::tma::*` calls：blocks cutlass users

### Decision 4: Cluster mode — 仅 arrive/wait，defer distributed_smem

**Context**: Hopper cluster (sm_90) 引入 8-CTA clusters 与 distributed shared
memory。Blackwell 扩展此。PTX-EMU 当前 `Hopper (sm_90+) | cluster 抽象未实现`
per 根 AGENTS.md。

然而 `tcgen05.mma.cta_group::1`（Phase 1-3 目标）在 single CTA 内操作 —
**不需要** distributed shared memory。仅 `cta_group::2`（2x1SM cluster，future Phase）
需要 cross-CTA smem access。

**Choice**: Phase 0.3 仅实现 `ClusterContext` 与 `arrive`/`wait` 原语。
`distributed_smem` defer 到 `cta_group::2` 真正需要时（独立 change 或 Phase expansion）。
这削减 ~400-600 LoC 从 Phase 0。

**Rationale**:
- Oracle review 确认：cute `mma_sm100_umma.hpp` cta_group::1 patterns 单 CTA
  操作（no cross-CTA memory access）
- `bar.sync` 已处理 single-CTA barriers（ADR-0008）；cluster arrive/wait
  是扩展而非替换
- Defer distributed_smem 让 Phase 0.3 < 400 LoC vs 800-1200，加速 Phase 0 交付 ~20%

**Alternatives considered**:
- ❌ 当前实现 full distributed_smem：~600 LoC 浪费（代码在 cta_group::2 之前不 exercise）
- ❌ Skip cluster 整个：tcgen05.wait 仍需 CTA-level 同步 for proxy flag；arrive/wait 是最小实现

### Decision 6: 测试用 synthetic fixtures，不用真 NVIDIA tools

**Context**: Phase 0 基础设施在 nvcc/cuobjdump 能产出有意义 test inputs 之下
（no PTX, no compiled kernels）。

**Choice**: 单元测试用 synthetic TMA descriptors（手工字节）、synthetic TMEM patterns、
synthetic cluster configurations。集成测试从 Phase 1-3（phase-1-3 change）开始
用真实 PTX 驱动模拟器。

**Rationale**:
- Phase 0 在 PTX 接口之下 — synthetic 是唯一选项
- 手工 fixtures 比 golden-file tests catch 更多 edge cases

**Alternatives considered**:
- ❌ 等 cuobjdump 支持：blocks Phase 0 indefinitely
- ❌ Skip Phase 0 单元测试：regressions 只在 Phase 1-3 被捕获

### Decision 7: tcgen05.wait blocking 复用 barrier 基础设施 via BAR_SYNC state translation

> **本 change 范围 (Phase 0)**：定义 Decision 7 的 framework（即下面 TcQueue::commit/wait 的 contract + cross-module audit points）。Phase 0 实施 `TcQueue` 类并暴露 `commit()` / `wait()` 接口契约，但不挂接任何 `tcgen05.commit/wait` handler 调用。
>
> **Phase 1-3 change 范围**：实现 `tcgen05.commit` / `tcgen05.wait` PTX handler，调用本决策定义的 `TcQueue::commit()` / `wait()`，由此真正触发 Decision 7 "commit → release_warp_barrier" 翻译链的执行。

**Context**: `TcQueue::wait(group_id)` 必须 block calling warp 直到 commit-group
counter 达到 `group_id`。这要求 warp 进入 scheduler 能识别的 blocked 状态，并在
counter 前进时 reawaken。`ptx-lessons-learned` §1 警告 cross-module state
translation（TcQueue state → WarpState blocking）是 codebase 中最容易 bug 的模式。

**Choice**: Phase 0.4 在 TcQueue 接口层预留 `wait()` 框架 + 显式 implementation
contract：

```cpp
// TcQueue::wait(group_id):
//   1. If commit_group_counter >= group_id → no-op (already done).
//   2. Else: call context->get_warp_context()->set_warp_state(
//            BAR_SYNC, /* all lanes blocked at current PC */);
//      The scheduler sees is_blocked=true and skips this warp.
//   3. Store wait group_id in TcQueue's pending_waiters list.

// TcQueue::commit(group_id):
//   1. commit_group_counter = max(counter, group_id).
//   2. For each pending waiter with waited_group <= counter:
//        BarrierModule::release_warp_barrier(waiter.warp_id, reconvergence_pc)
//        → sets is_blocked=false, active_mask back to full.
```

**Rationale**:
- 不发明新 blocking 机制 — `BAR_SYNC` + `is_blocked` 已充分测试 (commit `f033312`, `migrate-bar-warp-sync`)
- `ptx-lessons-learned` §1 明确要求审计 set_state → sync_to_warp_state 翻译链。
  复用 BAR_SYNC 让现有翻译自动正确
- `tcgen05.wait` 在 PTX 层面与 `bar.warp.sync` 对 waiting warp 语义相同：
  block 直到条件满足，然后 advance PC

**Cross-module audit** (Phase 0.4 commit 前强制 `state-modification-audit` skill)：
- `TcQueue::commit_group_counter` — 所有 writers 是 `TcQueue::commit()`。
  Scheduler 不直接读；只读 `WarpState.is_blocked`。
- `WarpState.threads[i].is_blocked` — 由 `TcQueue::wait()` 中的
  `set_warp_state(BAR_SYNC)` 设置；由 `BarrierModule::release_warp_barrier()`
  清零（当 TcQueue 调用）。

**Alternatives considered**:
- ❌ New `TC_WAIT` exec state：需修改 scheduler state machine, untested path,
  违反 lessons-learned §1（"现有 state 覆盖语义时不发明新 state"）
- ❌ Spin-wait (busy loop in handler)：浪费 cycles, 打破 scheduler model
- ❌ Defer 到 Phase 2.2 讨论：在无翻译计划下实施 TcQueue，是 lessons-learned §1 bug 的根因

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| TMA descriptor 二进制布局与 NVIDIA 不匹配 | **Critical** (无硬件交叉验证) | 用 NVIDIA PTX ISA §9.7 TensorMap 字段手工构造 descriptor bytes；header 标注 "unverified against hardware" |
| TcQueue→WarpState state translation bugs (ptx-lessons-learned §1) | **Critical** (cross-module translation is #1 failure mode) | Decision 7 复用 BAR_SYNC + BarrierModule 路径；Phase 0.4.5 强制 `state-modification-audit` skill 执行；所有 `commit_group_counter` writers AND `is_blocked` consumers 必须审计 |
| cluster mode + 现有 CTAContext 集成复杂 | High | Phase 0.3 子系统先 unit test 隔离行为；Phase 0.5.3 集成测试 |
| Phase 0 工程量大 | Medium | 9 commits (4 standalone + 4 micro + 1 artifacts tracked): TMA/TMEM/cluster/async queue + 4 个逐子系统集成；cluster 简化 (800-1200 → 300-400 LoC) |
| 现有 165 ctest regression | Low | Phase 0 添加新测试，不修改现有 handler paths |
| TMA host API 拦截策略 | Medium | ADR-0017 候选；Phase 0 用 fake descriptor |
| cute header sm_100 编译性 | Medium | Phase 0 前手工 spike；失败则 propose `fix-cute-sm100-headers` (Open Question #5) |

## Migration Plan

**9 commits** (4 standalone + 4 micro + 1 artifacts FIRST per lessons-learned §6):

```
Phase 0.0 (artifacts tracked, MUST FIRST per lessons-learned §6):
  Commit 0.0: docs(openspec): track implement-wmma-tensor-core-phase-0-infra artifacts
              [git add openspec/changes/implement-wmma-tensor-core-phase-0-infra/]

Phase 0.1: feat(memory): TMA descriptor parser (Fix #5) [独立可 revert]
Phase 0.2: feat(memory): per-CTA TMEM (Fix #6) [独立可 revert]
Phase 0.3: feat(sim): cluster arrive/wait (Fix #7, simplified) [独立可 revert]
Phase 0.4: feat(async): tc_queue commit-group + wait-aware scheduling (Fix #8)
           [独立可 revert；前必须 state-modification-audit]

Phase 0.5: 4 micro commits [整体 revert unit; 不可独立 revert 任何 micro]
  0.5.1: feat(sim): integrate TMA descriptor store with CTAContext (Fix #9a)
  0.5.2: feat(sim): integrate TMEM with CTAContext (Fix #9b)
  0.5.3: feat(sim): integrate cluster context with CTAContext (Fix #9c)
  0.5.4: feat(sim): integrate TcQueue with CTAContext (Fix #9d)
```

**Revert unit 澄清**:
- 0.1, 0.2, 0.3, 0.4 独立可 revert（每子系统 self-contained；`cta_context.cpp` 未被触碰）
- 0.5.1~0.5.4 **不可独立 revert** — 每个加 `CTAContext` 成员引用，且
  `TcQueue::enqueue_mma()` 写 `Tmem` slots, 创建跨子系统依赖
- **Revert unit for Phase 0.5 = 所有 4 commits 整体**
  (`git revert <0.5.1-sha>..<0.5.4-sha>`)
- Phase 0.5 失败处理：任何子系统 bug → 整体 revert 4 commits to 0.4.7 后状态，
  不单独 revert

## Open Questions

1. **TMA host API 拦截策略** — separate ADR-0017 in future.
2. **distributed_smem 模拟策略** — separate ADR-0018; only when cta_group::2 needed.
3. **async tensor core queue 与现有 WarpState 集成模式** — separate ADR-0019; if
   Phase 0.4+0.5 implementation reveals gaps.
4. **sm_120 sparse / FP4 / mxfp8** — separate changes per feature.
5. **cute header sm_100 编译性** — Phase 0 前手工 `nvcc -ptx` spike 验证
   `bench/cute/include/` 编译；失败则 propose `fix-cute-sm100-headers` change
   (per ptx-lessons-learned Checklist G "新建 fix-* change")

   **Spike 结果 (2026-07-04, Gate G7)**:
   ```bash
   # Spike #1: 仅 include cute/arch/mma_sm100_umma.hpp
   nvcc -arch=sm_100 -ptx -I bench/cute/include /tmp/spike_tcgen05.cu
   # EXIT=0, /tmp/spike1.ptx 1205 bytes

   # Spike #2: cute_rmsnorm_debug.cu (uses sm_100 per earlier grep)
   nvcc -arch=sm_100 -ptx --expt-relaxed-constexpr -I bench/cute/include \
        bench/cute/cute_rmsnorm_debug.cu
   # EXIT=0, /tmp/spike2.ptx 6669 bytes
   # 注意: cute_rmsnorm_debug 用 tiled_copy 而非 tcgen05,
   # 但 baseline build 已编译,证明 sm_100 headers 语法有效

   # Baseline (b7d48ca pre-split) full build also succeeds:
   # build/bin/{cute_hello_col_major,cute_rmsnorm,cute_rmsnorm_debug} all built
   ```

   **Gate G7 决议**: **PASS** — cute headers 编译 sm_100 有效, Phase 3 e2e
   `test_blackwell_gemm.cu` (cute tcgen05 style) 可推进。无需 propose `fix-cute-sm100-headers` change。

## Impact

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `src/ptxsim/memory/tma_descriptor.{h,cpp}` | 新建 | TMA descriptor 解析与存储 |
| `src/ptxsim/memory/tmem.{h,cpp}` | 新建 | per-CTA Tensor Memory |
| `src/ptxsim/cluster/cluster_context.{h,cpp}` | 新建 | cluster mode arrive/wait 原语 |
| `src/ptxsim/async/tc_queue.{h,cpp}` | 新建 | async tensor core queue 框架 |
| `src/ptxsim/core/cta_context.{h,cpp}` | 修改 (Phase 0.5) | 集成 4 子系统成员引用 |
| `src/CMakeLists.txt` | 修改 (Phase 0) | 新增 4 source |
| `tests/unit/CMakeLists.txt` | 修改 (Phase 0) | 新增 4 unit test |
| `tests/integration/CMakeLists.txt` | 修改 (Phase 0.5) | 新增 4 integration test |
| `docs/architecture/sm90_100.md` | 修改 (Phase 0 起) | 引用 ADR-0016 |

## 相关 ADR

- **ADR-0016**：本 design 的依据 (Blackwell-only vision)
- **ADR-0012** (per-thread-pc)：独立抽象层；TcQueue 不冲突 per-thread PC
- **ADR-0008** (barrier-semantics)：cluster mode 扩展 `bar.sync` 而非替换（TcQueue
  wait 复用 BAR_SYNC state translation per Decision 7）
- **未来 ADR-0017 候选**：TMA host API 拦截策略
- **未来 ADR-0018 候选**：cluster mode 的 distributed shared memory 模拟策略
- **未来 ADR-0019 候选**：async tensor core queue 与现有 WarpState 集成模式
- **未来 fix-* 候选**：`fix-cute-sm100-headers` (条件: Open Question #5 spike 失败)
