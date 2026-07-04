# Design: Implement Blackwell tcgen05 (skip pre-Blackwell WMMA)

> **架构依据**: [ADR-0016](../../docs/adr/0016-blackwell-only-tcgen05.md) —
> skip pre-Blackwell WMMA, only implement Blackwell tcgen05, with TMA / cluster /
> TMEM / async queue infrastructure as hard prerequisites.

## Context

`replace-silent-stub-failures` (archived 2026-07-04) made `WmmaHandler` throw
`UnsupportedInstructionException` on any wmma.* instruction. This change implements
the real Blackwell path so modern kernels can run, while keeping the
explicit-failure contract for pre-Blackwell variants (per ADR-0016).

The fundamental challenge: Blackwell's `tcgen05.*` instructions operate on
Tensor Memory (TMEM), which is loaded via TMA descriptors and committed via
async commit groups. This is a different execution model than per-thread PC
(ADR-0012) or even async `wgmma` — it requires new infrastructure before
any instruction handler can be written.

## Goals / Non-Goals

**Goals** (Phase 0-3):
1. TMA descriptor parse/store/load with ≥ 10 swizzle/stride combinations tested
2. TMEM 256-slot × 128-byte storage with read/write consistency
3. cluster mode: distributed shared memory + arrive/wait primitives
4. async tensor core queue: commit-group counter + wait-aware scheduling
5. `tcgen05.mma.cta_group::1.kind::f16` real fragment arithmetic
6. `tcgen05.ld/st` + `commit/wait` async flow
7. cutlass 3.x GEMM e2e kernel passes
8. AGENTS.md / spec publish

**Non-Goals** (per ADR-0016):
- Any pre-Blackwell WMMA instruction (`wmma.mma.sync`, `wgmma.async`, `mma.sync`)
- sm_120 sparse / FP4 / mxfp8 variants (separate change per feature)
- Performance parity (functional correctness only)
- SASS-level semantics
- TMA host API (`cuda::tma::create_tensor_map`) interception (separate ADR-0017)

## Decisions

### Decision 1: Phase 0 = TMA + TMEM + cluster + async queue infrastructure

**Context**: `tcgen05.mma` is unusable without its supporting infrastructure.
Attempting instruction handler before infrastructure would require extensive
workarounds that get thrown away later.

**Choice**: Phase 0 builds all 4 infrastructure subsystems before any
`tcgen05` handler is written. Each subsystem is one independent commit.

**Rationale**:
- `ptx-lessons-learned` §3 (分 Phase commit): each Phase independently revertible
- Infrastructure commits are clean architectural units — TMA / TMEM / cluster
  / async queue can be unit-tested in isolation
- Future sm_120 / FP4 work lands on top of the same infrastructure

**Alternatives considered**:
- ❌ Build infrastructure alongside instruction handler: blurs ownership, hard to revert
- ❌ Skip infrastructure (fake everything): produces wrong results

### Decision 2: Async queue is a separate abstraction layer, not integrated into WarpState

**Context**: `WarpState.threads[i].pc` (ADR-0012) is per-thread synchronous.
tcgen05's commit-group is per-CTA async. Two different abstractions.

**Choice**: Build `async::tc_queue::TcQueue` as a CTA-level subsystem that
`WarpState` references but does not embed.

**Rationale**:
- Avoids invasive WarpState refactor
- `ptx-lessons-learned` §1 (跨模块状态翻译): TcQueue has its own state
  (commit-group counter, pending operations); these need explicit
  translation to whatever scheduler-visible state
- Future async instructions (e.g. `cp.async.bulk`) can share the same queue

**Alternatives considered**:
- ❌ Embed in WarpState: scope creep, hard to revert
- ❌ Per-warp queue: cuts against CTA-level tcgen05 semantics

### Decision 3: TMA descriptor is parsed from raw bytes, not constructed from cuda API

**Context**: `cuda::tma::create_tensor_map` is a host API. The descriptor
is built by the host and copied to device memory, where `tcgen05.ld` reads
it. PTX-EMU needs to intercept this copy to capture the descriptor.

**Choice**: Phase 0 supports "fake descriptor" mode: a test/host utility
constructs descriptor bytes directly and stores them in fake device memory.
The `cuda::tma::create_tensor_map` interception is deferred to ADR-0017.

**Rationale**:
- Keeps Phase 0 scope bounded
- Avoids cuda runtime API simulation complexity
- Real descriptor interception is a separate decision

**Alternatives considered**:
- ❌ Full API interception in Phase 0: scope creep
- ❌ Reject all `cuda::tma::*` calls: blocks cutlass users

### Decision 4: cluster mode — arrive/wait only, defer distributed_smem

**Context**: Hopper cluster (sm_90) introduced 8-CTA clusters with distributed
shared memory access. Blackwell extends this. PTX-EMU currently has
`Hopper (sm_90+) | cluster 抽象未实现` per root AGENTS.md.

However, `tcgen05.mma.cta_group::1` (Phase 1 target) operates within a
single CTA — it does NOT need distributed shared memory. Only
`cta_group::2` (2x1SM cluster, future Phase) requires cross-CTA smem access.

**Choice**: Phase 0.3 implements only `ClusterContext` with
`arrive`/`wait` primitives. `distributed_smem` is deferred to when
`cta_group::2` is actually needed (separate change or Phase expansion).
This cuts ~400-600 LoC from Phase 0.

**Rationale**:
- Oracle review confirmed: cute `mma_sm100_umma.hpp` cta_group::1
  patterns operate single-CTA (no cross-CTA memory access)
- `bar.sync` already handles single-CTA barriers (ADR-0008); cluster
  arrive/wait are an extension, not a replacement
- Deferred distributed_smem keeps Phase 0.3 under 400 LoC instead of
  800-1200, accelerating Phase 0 delivery by ~20%

**Alternatives considered**:
- ❌ Implement full distributed_smem now: wastes ~600 LoC on code
  nothing exercises until cta_group::2
- ❌ Skip cluster entirely: tcgen05.wait still needs CTA-level
  synchronization for the proxy flag; arrive/wait is the minimum

### Decision 5: Throw on divergent wmma, async wait on divergent tcgen05

**Context**: pre-Blackwell `wmma.mma.sync` is synchronous — divergent warps
can be detected at instruction fetch and rejected. `tcgen05.mma` is async
— divergent warp at fetch is fine; the divergence manifests at
`tcgen05.wait`.

**Choice**:
- pre-Blackwell: throw `ExecutionStateException` at fetch if
  `active_mask != 0xFFFFFFFF` (matches original proposal)
- Blackwell tcgen05: no throw at fetch; `tcgen05.wait` blocks until all
  async ops in the commit group complete; lanes that didn't issue an mma
  still wait correctly

**Rationale**:
- Matches hardware semantics (Blackwell async model is fundamentally
  different from sync wmma)
- Preserves `replace-silent-stub-failures` contract for pre-Blackwell

**Alternatives considered**:
- ❌ Always throw on divergence: rejects valid Blackwell code
- ❌ Always async wait: wrong for pre-Blackwell

### Decision 6: Tests use synthetic fixtures, not real NVIDIA tools

**Context**: Phase 0 infrastructure is below the level where nvcc/cuobjdump
can produce meaningful test inputs (no PTX, no compiled kernels).

**Choice**: Unit tests use synthetic TMA descriptors (handcrafted bytes),
synthetic TMEM patterns, synthetic cluster configurations. Integration tests
start in Phase 1+ where real PTX can drive the simulator.

**Rationale**:
- Phase 0 is below the PTX interface — synthetic is the only option
- Handcrafted fixtures catch more edge cases than golden-file tests

**Alternatives considered**:
- ❌ Wait for cuobjdump support: blocks Phase 0 indefinitely
- ❌ Skip Phase 0 unit tests: regressions will only be caught in Phase 3

### Decision 7: tcgen05.wait blocking reuses barrier infrastructure via BAR_SYNC state translation

**Context**: `TcQueue::wait(group_id)` must block the calling warp until
the commit-group counter reaches `group_id`. This requires the warp to
enter a blocked state that the scheduler recognizes, and to be re-awakened
when the counter advances. `ptx-lessons-learned` §1 warns that
cross-module state translation (TcQueue state → WarpState blocking) is the
single most bug-prone pattern in the codebase.

**Choice**: `TcQueue::wait()` calls `WarpContext::set_warp_state(BAR_SYNC)`
to mark the warp as blocked, reusing the existing barrier infrastructure
(`BarrierModule` in `src/ptxsim/barrier/barrier_module.cpp`). When
`TcQueue::commit(group_id)` advances the counter past the waited-for
group, it calls `BarrierModule::release_warp_barrier(...)` to wake the
blocked warp at the wait instruction's PC.

**Rationale**:
- Avoids inventing a new blocking mechanism — `BAR_SYNC` + `is_blocked`
  is already well-tested (commit `f033312`, `migrate-bar-warp-sync`).
- `ptx-lessons-learned` §1 explicitly requires auditing the set_state →
  sync_to_warp_state translation chain. Reusing BAR_SYNC means the
  existing translation is automatically correct.
- `tcgen05.wait` at the PTX level is semantically the same as `bar.warp.sync`
  for the waiting warp: block until condition met, then advance PC.

**Implementation contract**:
```cpp
// In TcQueue::wait(group_id):
//   1. If commit_group_counter >= group_id → no-op (already done).
//   2. Else: call context->get_warp_context()->set_warp_state(
//            BAR_SYNC, /* all lanes blocked at current PC */);
//      The scheduler sees is_blocked=true and skips this warp.
//   3. Store wait group_id in TcQueue's pending_waiters list.

// In TcQueue::commit(group_id):
//   1. commit_group_counter = max(counter, group_id).
//   2. For each pending waiter with waited_group <= counter:
//        BarrierModule::release_warp_barrier(waiter.warp_id, reconvergence_pc)
//        → sets is_blocked=false, active_mask back to full.
```

**Cross-module audit** (must run `state-modification-audit` skill):
- `TcQueue::commit_group_counter` — all writers are `TcQueue::commit()`.
  Scheduler never reads this directly; it only sees `WarpState.is_blocked`.
- `WarpState.threads[i].is_blocked` — set by `set_warp_state(BAR_SYNC)` in
  `TcQueue::wait()`, cleared by `BarrierModule::release_warp_barrier()`
  when TcQueue calls it.
- Consumers of `is_blocked`: `sm_context.cpp scheduler` → reads it to
  decide whether to skip the warp. No new consumers introduced.
- **Audit checklist** (tasks.md Phase 0.4.5): grep all writers of both
  variables; confirm translation chain is one-way and consistent.

**Alternatives considered**:
- ❌ New `TC_WAIT` exec state: requires modifying scheduler state machine,
  untested path, violates lessons-learned §1 ("don't invent new state
  when existing state covers the semantics").
- ❌ Spin-wait (busy loop in handler): wastes cycles, breaks the scheduler
  model.
- ❌ Defer to Phase 2.2 discussion: risks implementing TcQueue without
  a translation plan, which is the root cause of lessons-learned §1 bugs.

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| TMA descriptor binary layout mismatch with NVIDIA | **Critical** (Oracle review: no HW to cross-validate) | Use NVIDIA PTX ISA §9.7 TensorMap field definitions to handcraft descriptor bytes; annotate header as "unverified against hardware". Phase 3 e2e failure triggers descriptor parse audit |
| Fragment layout 解读错误无硬件交叉验证 | **Critical** (Oracle review: "hand-computed reference" correctness is actually self-consistency) | Annotate every fragment element's lane→(row,col) mapping with source PTX ISA §9.7.13 line references in wmma.cpp comments. Accept this risk: correctness = self-consistency until real hardware is available |
| TcQueue→WarpState state translation bugs (ptx-lessons-learned §1) | **Critical** (Oracle review: cross-module translation is the #1 failure mode in this codebase) | Decision 7 requires reusing BAR_SYNC + BarrierModule path. Tasks.md Phase 0.4.5 mandates state-modification-audit skill execution before commit. All commit_group_counter writers AND is_blocked consumers must be audited |
| cluster mode + existing CTAContext integration bugs | High | Phase 0 unit tests verify isolation; cluster integration test in Phase 0.5.3 |
| cute template sm_100 fallback to sm_90 wgmma | Medium | Out of scope — pre-Blackwell throw is expected per ADR-0016 |
| Phase 0 工程量超估 | Medium | 9 commits for Phase 0 (Oracle: was 5 → 9 = 4 standalone + 4 micro + 1 artifacts); each commit self-contained; cluster simplified from 800-1200 → 300-400 LoC via deferred distributed_smem |
| cute_rmsnorm future upgrade triggers Phase 0-2 dependency | Low | cute_rmsnorm upgrade blocked until Phase 0-2 done |
| Existing 165 ctest regression risk | Low | Phase 0 adds new tests, doesn't modify existing handler paths |
| sm_120 sparse requires cta_group::2 | Low | Phase 2 reserves `cta_group::2` extension point |
| TMA host API interception strategy unclear | Medium | ADR-0017 deferred; Phase 0 uses fake descriptor |

## Migration Plan

**4 Phases, 14 commits total** (Oracle review C2 fix, 2026-07-04):
- Phase 0 = 8 implementation commits (0.1–0.4 + 0.5.1–0.5.4) + 1 artifacts commit = 9
- Phase 1 = 2 commits (1.1 + 1.2)
- Phase 2 = 2 commits (2.1 + 2.2)
- Phase 3 = 1 commit (3.1)
- Phase 4 (merge/archive) not counted as implementation commit

```
Phase 0: Infrastructure (8 implementation commits + 1 artifacts commit, ~3000-5000 LoC)
  Commit 0.1: feat(memory): TMA descriptor parser + storage
  Commit 0.2: feat(memory): Tensor Memory (TMEM) per-CTA storage
  Commit 0.3: feat(sim): cluster arrive/wait (simplified—no distributed smem)
  Commit 0.4: feat(async): tc_queue commit-group + wait-aware scheduling
  Commit 0.5.1: feat(sim): integrate TMA descriptor store with CTAContext
  Commit 0.5.2: feat(sim): integrate TMEM with CTAContext
  Commit 0.5.3: feat(sim): integrate cluster context with CTAContext
  Commit 0.5.4: feat(sim): integrate TcQueue with CTAContext
  [Plus 1 artifacts commit: docs(openspec) tracked before code, per
   ptx-lessons-learned experience 6 / Checklist E]

Phase 1: tcgen05.mma (2 commits, ~500-800 LoC)
  Commit 1.1: feat(wmma): implement tcgen05.mma fragment arithmetic
  Commit 1.2: test(wmma): integration test (read TMEM directly, no commit/wait)

Phase 2: tcgen05.ld/st + commit/wait (2 commits, ~600-1000 LoC)
  Commit 2.1: feat(wmma): tcgen05.ld/st with TMA + TMEM integration
  Commit 2.2: feat(wmma): tcgen05.commit/wait async flow

Phase 3: e2e + AGENTS + spec publish (1 commit, ~300-500 LoC)
  Commit 3.1: docs+test: e2e GEMM kernel + AGENTS sync + spec publish
```

**Revert unit (Oracle Q2 fix, 2026-07-04)**:
- Commits 0.1, 0.2, 0.3, 0.4 are independently revertible (each subsystem
  is self-contained; `cta_context.cpp` is not yet touched).
- Commits 0.5.1–0.5.4 are **NOT** independently revertible — they each
  add a `CTAContext` member reference, and `TcQueue::enqueue_mma()` writes
  to `Tmem` slots, creating cross-subsystem dependencies. **Revert unit
  for Phase 0.5 = all 4 commits together** (`git revert <0.5.1-sha>..<0.5.4-sha>`).
- Commits 1.1, 1.2, 2.1, 2.2, 3.1 are independently revertible.
- The artifacts commit (Phase 0 start) is independently revertible.

**Oracle review note**: Phase 1.2 integration test originally claimed to
verify "mma + commit + wait" sequence, but commit/wait is Phase 2.2.
Fixed to verify mma correctness by reading TMEM slots directly.

## Open Questions

1. **TMA host API interception strategy** — separate ADR-0017 in future.
2. **sm_120 sparse / FP4 / mxfp8** — separate changes per feature.
3. **cute_rmsnorm upgrade to tcgen05** — blocked until Phase 0-2 done; track in
   follow-up issue.
4. **async queue priority vs scheduler** — does TcQueue compete with
   WarpState for cycle resources, or run on dedicated cycles? Need real
   Blackwell hardware data to calibrate.

## 影响范围

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `src/ptxsim/memory/tma_descriptor.{h,cpp}` | 新建 | TMA descriptor 解析与存储 |
| `src/ptxsim/memory/tmem.{h,cpp}` | 新建 | per-CTA Tensor Memory |
| `src/ptxsim/cluster/{h,cpp}` | 新建 | cluster mode + 分布式 shared memory |
| `src/ptxsim/async/tc_queue.{h,cpp}` | 新建 | async tensor core queue |
| `src/ptxsim/instructions/wmma.cpp` | 修改 (Phase 1-2) | 抛异常 → tcgen05 实现 |
| `src/ptxsim/core/cta_context.{h,cpp}` | 修改 (Phase 0.5) | 集成 cluster + TcQueue |
| `src/ptxsim/instructions/AGENTS.md` | 修改 (Phase 3) | KNOWN STUBS 移除 WMMA 条目 |
| 根 `AGENTS.md` | 修改 (Phase 3) | 已知限制表 WMMA 条目更新 |
| `docs/architecture/sm90_100.md` | 修改 (Phase 0 起) | 引用 ADR-0016 |
| `src/CMakeLists.txt` | 修改 (Phase 0) | 新增 4 source |
| `tests/unit/CMakeLists.txt` | 修改 (Phase 0-1) | 新增 4 unit test |

## 相关 ADR

- **ADR-0016**：本 design 的依据（Blackwell-only vision）
- **ADR-0012**（per-thread-pc）：独立抽象层；TcQueue 不冲突 per-thread PC
- **ADR-0014**（independent-thread-scheduling）：正交；ITS 是 warp 内多路径调度
- **ADR-0008**（barrier-semantics）：cluster mode 扩展 `bar.sync` 而非替换
- **未来 ADR-0017 候选**：TMA host API 拦截策略
- **未来 ADR-0018 候选**：cluster mode 的 distributed shared memory 模拟策略
- **未来 ADR-0019 候选**：async tensor core queue 与现有 WarpState 集成模式