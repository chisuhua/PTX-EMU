## ADDED Requirements

> **Scope**: Blackwell `tcgen05.*` only, per [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md).
> pre-Blackwell WMMA remains `UnsupportedInstructionException` per
> `replace-silent-stub-failures` contract.

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

### Requirement: Cluster-Mode-Distributed-SMEM MUST

The system SHALL provide a `ClusterContext` implementing distributed
shared memory across 1-8 CTAs in a Hopper/Blackwell cluster. The
context SHALL support `cta_cluster_arrive()` and `cta_cluster_wait()`
synchronization primitives that block until all CTAs in the cluster
have arrived.

#### Scenario: cluster-arrive-wait-sync
- **WHEN** CTA-0 calls `cta_cluster_arrive()` and CTA-1 calls `cta_cluster_arrive()`
- **THEN** both calls return once each CTA's arrive count is reached
- **AND** `cta_cluster_wait()` blocks on each CTA until all peer CTAs
  have arrived

#### Scenario: cluster-distributed-smem-access
- **WHEN** CTA-0 writes to its shared memory at offset 100
- **AND** CTA-1 calls `distributed_smem.read(cta=0, offset=100)` (assuming
  same cluster)
- **THEN** CTA-1 sees the value written by CTA-0

#### Scenario: cluster-no-cross-cta-write-without-arrive
- **WHEN** CTA-1 attempts `distributed_smem.read(cta=0, ...)` without
  calling `cta_cluster_arrive()` first
- **THEN** the read throws (synchronization required)

### Requirement: Async-TcQueue-Commit-Wait MUST

The system SHALL provide a `TcQueue` per-CTA async tensor core queue
with commit-group counter and wait-aware scheduling. The queue SHALL
support `commit(group_id)`, `wait(group_id)`, and `enqueue_mma(...)`
operations, where `wait` blocks until `commit_group_counter >= group_id`.

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

### Requirement: Tcgen05-Mma-Fragment-Arithmetic MUST

The system SHALL implement `tcgen05.mma.cta_group::1.kind::f16` real
fragment arithmetic in `src/ptxsim/instructions/wmma.cpp`. The handler
SHALL read A and B fragments from TMEM via TMA, perform the matrix
multiplication, and write the result to TMEM, delegating to `TcQueue`
for commit-group management. `include/ptxsim/utils/half_utils.h`
MUST be reused for f16 ↔ f32 conversion.

#### Scenario: full-fragment-correctness
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked with
  `tcgen05.mma.cta_group::1.kind::f16` qualifiers on a uniform warp
  loaded with deterministic A, B fragments and zero C accumulator
- **THEN** all output fragment elements equal the hand-computed reference
  (per NVIDIA PTX ISA §9.7.13 Blackwell fragment layout)
- **AND** no element is left uninitialized

#### Scenario: tcgen05-mma-divergent-warp-async-wait
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked on a
  divergent warp (`active_mask != 0xFFFFFFFF`)
- **THEN** the handler does NOT throw at fetch (async semantics)
- **AND** `TcQueue::wait(group=N)` synchronizes correctly regardless of
  which lanes issued the original mma

#### Scenario: half-utils-reuse-not-reimplemented
- **WHEN** `wmma.cpp` needs f16 ↔ f32 conversion
- **THEN** `include/ptxsim/utils/half_utils.h::f16_to_f32` is used
- **AND** no inline conversion logic is duplicated in `wmma.cpp`

### Requirement: Tcgen05-Ldst-CommitWait-Full-Flow MUST

The system SHALL implement `tcgen05.ld`, `tcgen05.st`, `tcgen05.commit`,
and `tcgen05.wait` instructions, integrating with the Phase 0
infrastructure (TMA + TMEM + TcQueue). A complete mma sequence
(ld → mma → commit → wait → st) SHALL execute end-to-end on TMEM.

#### Scenario: ld-mma-commit-wait-st-roundtrip
- **WHEN** a warp executes the full sequence:
  - `tcgen05.ld` (load A/B from TMA descriptor into TMEM)
  - `tcgen05.mma` (compute fragment)
  - `tcgen05.commit`
  - `tcgen05.wait`
  - `tcgen05.st` (store result from TMEM via TMA descriptor)
- **THEN** the final memory state matches the hand-computed reference
  for all fragment elements

#### Scenario: commit-group-counter-increments
- **WHEN** `tcgen05.commit` is invoked three times with group ids 1, 2, 3
- **THEN** `TcQueue::commit_group_counter` reaches 3
- **AND** `wait(2)` returns after commit-2 completes (not requiring
  commit-3 to complete)

### Requirement: Blackwell-GEMM-E2E-Kernel-Passes MUST

SHALL produce correct output for a Blackwell `tcgen05` matrix-multiply
kernel. The e2e test MUST verify correctness via host-side comparison.

A type-3 e2e test (compiled CUDA kernel → PTX extraction → simulator
execution → host-side verification) SHALL demonstrate that the
implemented Blackwell tcgen05 path produces correct output for a
small matrix-multiply.

#### Scenario: small-matmul-correctness
- **WHEN** a 16×16 GEMM kernel is run with deterministic f16 inputs and
  zero accumulator, target sm_100
- **THEN** the e2e test verifies `C[i][j] == sum_k A[i][k] * B[k][j]`
  for all `i,j ∈ [0,16)²` within f32 rounding tolerance

### Requirement: File-Renamed-To-wmma-cpp MUST

`src/ptxsim/instructions/wmma.cpp` SHALL be the file name (already
renamed from `tensor.cpp` in `replace-silent-stub-failures` follow-up,
or renamed in Phase 1.1 of this change). The class name `WmmaHandler`
and X-Macro registration are unchanged.

#### Scenario: rename-builds-and-tests-pass
- **WHEN** the project is built
- **THEN** `cmake --build build` succeeds with no source-level errors
- **AND** `ctest -L "unit;integration;e2e"` passes with no regression
- **AND** `grep -rn "tensor.cpp" src/CMakeLists.txt` does not return
  the old filename

### Requirement: AGENTS-Sync-Blackwell-Implemented MUST

SHALL synchronize AGENTS.md documentation to reflect the new Blackwell
`tcgen05` implementation. The pre-Blackwell throw behavior SHALL be
documented as permanent per ADR-0016.

The KNOWN STUBS section in `src/ptxsim/instructions/AGENTS.md` and
the "已知限制" table in the root `AGENTS.md` SHALL be updated to
describe the new Blackwell tcgen05 implementation. The pre-Blackwell
throw behavior SHALL be documented as permanent per ADR-0016.

#### Scenario: agents-md-reflects-blackwell-impl
- **WHEN** reading `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS section
- **THEN** the WMMA entry describes Blackwell tcgen05 as implemented
- **AND** references ADR-0016 for pre-Blackwell throw policy

#### Scenario: root-agents-md-known-limitations-sync
- **WHEN** reading the root `AGENTS.md` "已知限制" table
- **THEN** the WMMA entry states "Blackwell tcgen05 已实现；
  pre-Blackwell 永久抛异常（ADR-0016）"

> **Oracle review fix (2026-07)**: The `Stub-Explicit-Failure-WMMA-Permanent`
> requirement originally here was a near-duplicate of
> `stub-explicit-failure/spec.md:Stub-Explicit-Failure-Permanent-Policy`.
> Removed — one policy SHALL have one canonical location
> (`stub-explicit-failure` spec). The `wmma-tensor-core` spec is the
> "how Blackwell works" spec; `stub-explicit-failure` is the "what
> pre-Blackwell does" policy spec.