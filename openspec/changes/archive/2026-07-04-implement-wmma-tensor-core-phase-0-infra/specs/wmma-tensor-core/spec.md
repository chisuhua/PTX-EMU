## ADDED Requirements

> **Scope**: Blackwell `tcgen05.*` 基础设施 (Phase 0)，per [ADR-0016](../../../../docs/adr/0016-blackwell-only-tcgen05.md).
> 本 spec 仅覆盖 4 个基础设施子系统 (TMA + TMEM + cluster + async queue);
> `tcgen05.mma/ld/st/commit/wait` handler 实现与 GEMM e2e kernel 由
> `implement-wmma-tensor-core-tcgen05` change 交付。
>
> pre-Blackwell WMMA 仍抛 `UnsupportedInstructionException` per
> `replace-silent-stub-failures` 合约。

### Requirement: TMA-Descriptor-Parse-And-Store MUST

The system SHALL provide a `TmaDescriptor` type that parses raw TMA
descriptor bytes (per NVIDIA PTX ISA §9.7 TensorMap layout) and stores
them in a per-CTA descriptor table. The parser MUST handle at least
10 distinct swizzle/stride combinations covering f16, bf16, tf32, f32,
f8, and f4 dtypes.

#### Scenario: descriptor-parse-roundtrip
- **WHEN** a handcrafted TMA descriptor byte sequence (header + swizzle
  + stride + dtype) is passed to `parse_descriptor_bytes(...)`
- **THEN** the returned `TmaDescriptor` carries the same dtype, swizzle,
  and stride values
- **AND** the descriptor can be stored in the per-CTA `TmaDescriptorStore`
  and retrieved by id

#### Scenario: descriptor-bad-magic-rejected
- **WHEN** a byte sequence with an invalid TensorMap magic number is parsed
- **THEN** `parse_descriptor_bytes` returns an error
- **AND** no descriptor is added to the store

### Requirement: TMEM-PerCTA-Storage MUST

The system SHALL provide a `Tmem` type implementing per-CTA Tensor Memory
(256 slots × 128 bytes = 32 KB per CTA), matching NVIDIA Blackwell
hardware layout. The store SHALL support read/write/clear operations
and SHALL be independent from shared memory.

#### Scenario: tmem-write-read-consistency
- **WHEN** `Tmem::write(slot=5, bytes=0xABCD...)` is called
- **THEN** `Tmem::read(slot=5)` returns `0xABCD...` byte-identical

#### Scenario: tmem-isolation-across-ctas
- **WHEN** two CTAs each have their own `Tmem`
- **THEN** writes from CTA-1 are NOT visible to CTA-2
- **AND** TMEM is independent from shared memory (writes to TMEM do not
  affect shared memory and vice versa)

#### Scenario: tmem-capacity-enforced
- **WHEN** `Tmem::write(slot=300, ...)` is called (out of bounds)
- **THEN** an error is raised
- **AND** no partial write occurs

### Requirement: Cluster-Mode-Arrive-Wait MUST

The system SHALL provide a `ClusterContext` with `cta_cluster_arrive()`
and `cta_cluster_wait()` synchronization primitives that block until
all CTAs in the cluster have arrived.

> **Oracle review fix (Q3, 2026-07-04)**: 原 Requirement `Cluster-Mode-Distributed-SMEM`
> 描述了 `distributed_smem.read()` 但 `design.md:Decision 4` **明确 deferred**
> distributed_smem 到 `cta_group::2` 阶段（候选 ADR-0018）。本 change Phase 0.3
> **仅**实现 arrive/wait 同步原语；distributed shared memory 留待后续 change。
>
> **Spec 范围**: arrive/wait 同步。distributed_smem scenarios 已移除。

#### Scenario: cluster-arrive-wait-sync
- **WHEN** CTA-0 calls `cta_cluster_arrive()` and CTA-1 calls `cta_cluster_arrive()`
- **THEN** both calls return once each CTA's arrive count is reached
- **AND** `cta_cluster_wait()` blocks on each CTA until all peer CTAs
  have arrived

### Requirement: Async-TcQueue-Commit-Wait MUST

The system SHALL provide a `TcQueue` per-CTA async tensor core queue
with commit-group counter and wait-aware scheduling. The queue SHALL
support `commit(group_id)`, `wait(group_id)`, and `enqueue_mma(...)`
operations, where `wait` blocks until `commit_group_counter >= group_id`.

The wait blocking mechanism **MUST** reuse the `BAR_SYNC` exec state via
`WarpState::set_warp_state(BAR_SYNC)` and `BarrierModule::release_warp_barrier(...)`
per design.md Decision 7. **MUST NOT** introduce a new `TC_WAIT` exec state.

> **本 change (Phase 0.4) 仅交付 TcQueue 接口框架与 unit-level 单元测试。**
> 实际的 `tcgen05.wait` 调用 TcQueue::wait 的端到端集成（commit 触发 release_warp_barrier）
> 由 Phase 1-3 change (`implement-wmma-tensor-core-tcgen05`) 交付。

#### Scenario: commit-wait-ordering
- **WHEN** warp-0 calls `tc_queue.commit(group=1)` then `tc_queue.commit(group=2)`
- **AND** another warp calls `tc_queue.wait(group=2)`
- **THEN** the wait blocks until both commit operations have completed
- **AND** after wait, the queue is in a state where group-2 mmas are visible

#### Scenario: wait-without-matching-commit-blocks
- **WHEN** a warp calls `tc_queue.wait(group=5)` but no commit to group-5
  has occurred
- **THEN** the wait blocks indefinitely (or returns a deadline error after
  a configurable timeout)

#### Scenario: enqueue-mma-not-committed-not-visible
- **WHEN** warp-0 calls `tc_queue.enqueue_mma(...)` without commit
- **THEN** a peer warp's `wait()` does NOT see the result
- **AND** the result is only visible after `commit(group=N)` + `wait(group=N)`
  sequence

#### Scenario: wait-uses-existing-BAR-SYNC-state-not-new-state
- **WHEN** `TcQueue::wait(group=N)` blocks the calling warp
- **THEN** the blocked warp's `WarpState.is_blocked == true` is set via
  `set_warp_state(BAR_SYNC)` (NOT a new `TC_WAIT` state)
- **AND** `BarrierModule::release_warp_barrier(...)` is the only path
  that clears `is_blocked` for waiters
- **AND** `grep -rn "is_blocked\s*=" src/ptxsim/` shows no writers
  outside the documented set (TcQueue::wait + BarrierModule::release)
