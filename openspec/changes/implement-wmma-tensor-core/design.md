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

### Decision 4: cluster mode = distributed shared memory across N CTAs

**Context**: Hopper cluster (sm_90) introduced 8-CTA clusters with distributed
shared memory access. Blackwell extends this. PTX-EMU currently has
`Hopper (sm_90+) | cluster 抽象未实现` per root AGENTS.md.

**Choice**: Build `cluster::ClusterContext` per `CTAContext` that owns a
`distributed_smem` view across 1-8 CTAs. `cta_cluster_arrive` / `cta_cluster_wait`
synchronize across the cluster.

**Rationale**:
- Matches NVIDIA hardware model
- `bar.sync` already handles single-CTA barriers (ADR-0008); cluster
  barriers are an extension, not a replacement

**Alternatives considered**:
- ❌ Implement only single-CTA cluster (skip distributed smem): defeats the purpose
- ❌ Refactor existing CTAContext to cluster: invasive, breaks other tests

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

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| TMA descriptor binary layout mismatch with NVIDIA | High | Use cuobjdump-extracted real descriptors as ground truth; cross-validate against NVIDIA PTX ISA §9.7 |
| cluster mode + existing CTAContext integration bugs | High | Phase 0 unit tests verify isolation; cluster integration test in Phase 1 |
| async queue state translation bugs (ptx-lessons-learned §1) | High | Use `state-modification-audit` skill before each Phase commit; verify all consumers of `commit_group_counter` see consistent state |
| TMA host API interception unclear | Medium | ADR-0017 deferred; Phase 0 uses fake descriptor |
| Cute template sm_100 fallback to sm_90 wgmma | Medium | Out of scope — pre-Blackwell throw is expected per ADR-0016 |
| cute_rmsnorm future upgrade triggers Phase 0-2 dependency | Low | cute_rmsnorm upgrade blocked until Phase 0-2 done |
| 5 Phase 0 commits × ~1000 LoC each = large merge surface | Medium | Each commit is self-contained; revert one doesn't break others |
| Existing 165 ctest regression risk | Low | Phase 0 adds new tests, doesn't modify existing handler paths |
| sm_120 sparse requires cta_group::2 | Low | Phase 2 reserves `cta_group::2` extension point |

## Migration Plan

**4 Phases, 9 commits total**:

```
Phase 0: Infrastructure (5 commits, ~3000-5000 LoC)
  Commit 0.1: feat(memory): TMA descriptor parser + storage
  Commit 0.2: feat(memory): Tensor Memory (TMEM) per-CTA storage
  Commit 0.3: feat(sim): cluster mode + distributed shared memory
  Commit 0.4: feat(async): tc_queue commit-group + wait-aware scheduling
  Commit 0.5: feat(sim): integrate TMA+TMEM+cluster+queue with CTAContext

Phase 1: tcgen05.mma (2 commits, ~500-800 LoC)
  Commit 1.1: feat(wmma): implement tcgen05.mma fragment arithmetic
  Commit 1.2: test(wmma): integration test for mma + commit sequence

Phase 2: tcgen05.ld/st + commit/wait (2 commits, ~600-1000 LoC)
  Commit 2.1: feat(wmma): tcgen05.ld/st with TMA + TMEM integration
  Commit 2.2: feat(wmma): tcgen05.commit/wait async flow

Phase 3: e2e + AGENTS + spec publish (1 commit, ~300-500 LoC)
  Commit 3.1: docs+test: e2e GEMM kernel + AGENTS sync + spec publish
```

Each commit independently revertible (per ptx-lessons-learned §3).

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