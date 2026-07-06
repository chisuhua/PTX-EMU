# Blackwell tcgen05 Infrastructure Audit Report

> **OpenSpec change**: `extend-blackwell-tcgen05-infra` (Change-2)
> **Audit date**: 2026-07-07
> **Scope**: 5 subsystems per proposal.md MR-5 (TmaDescriptor + Tmem +
> ClusterContext + TcQueue + wmma.cpp L320-565 handler segment)
> **References**: `openspec/changes/extend-blackwell-tcgen05-infra/proposal.md`;
> `openspec/changes/extend-blackwell-tcgen05-infra/design.md` (Decision 4 + 5);
> `docs/adr/0016-blackwell-only-tcgen05.md`; `ptx-lessons-learned` skill §1-§8
> **Method**: Pure read-only (per Metis MR-3); every claim cites `file:line`
> in source. No builds run during this audit; baseline ctest data reused
> verbatim from `proposal.md` §Why + Design-Time Checklist.

## §1 Overview

### §1.1 Audit Goals

Per `proposal.md:11` Goals: audit 5 subsystems + grade 38 implementation-level
UNVERIFIED annotations (29 TmaDescriptor + 9 wmma.cpp handlers) + identify
cross-subsystem pipeline coverage gaps + decide Change-3 readiness.

### §1.2 Decision 4 — L1/L2/L3 Readiness Rubric (from design.md:33-47)

| Level | Meaning | Criteria (all) | Change-3 decision |
|---|---|---|---|
| L1 | working | (a) relevant ctest targets all green; (b) **zero** P0 UNVERIFIED; (c) code path covers all public APIs | directly usable |
| L2 | needs-attention | (a) ctest green; (b) **1-2** P0 UNVERIFIED; (c) P0 items in `fix-*` backlog with owner | usable; **must** parallel fix-* work |
| L3 | blocks Change-3 | (a) ctest failures OR (b) **≥3** P0 UNVERIFIED OR (c) fundamental invariant missing (e.g. `set_state` not called) | **must wait for** `fix-*` |

**aggregate readiness** = min-rule across the 5 subsystems (per `design.md:47`).
Change-3 may start iff aggregate ≥ L2.

### §1.3 Decision 5 — P0/P1/P2 Grading Criteria (from design.md:49-69)

| Grade | Dimension | Criteria (any) | Fix timing |
|---|---|---|---|
| P0 | handler correctness | (a) UNVERIFIED missing invariant directly used by handler; (b) core data structure size/offset; (c) sync primitive | before Change-3 |
| P1 | precision | (a) fragment element index accuracy; (b) bit layout not affecting correctness; (c) swizzle/stride with fallback | parallel with Change-3 |
| P2 | edge case | (a) rare paths; (b) rare type combos; (c) coverage gap (not impl gap) | defer to Change-4 or cleanup |

**Exclusion rule** (per `design.md:59-67`): (a) auto-generated reference tables
(e.g. `wmma.cpp:62-317` 256-entry); (b) bare ISA reference AND in
reference/table context (NOT in handler body — bare ISA ref in handler body
still counts per NI-5 fix); (c) test fixtures.

### §1.4 Baseline ctest Results (verified, per proposal.md:88)

| ctest target | result | duration |
|---|---|---|
| unit_tma_descriptor | PASS | 0.07s |
| unit_tmem | PASS | 0.05s |
| unit_cluster_mode | PASS | 0.12s |
| unit_cluster_tcgen05_integration | PASS | 0.02s |
| unit_tc_queue | PASS | 0.02s |
| **Total** | **5/5 PASS** | **0.28s** |

All baseline ctest green → L3 condition (a) not triggered for any subsystem.

## §2.1 TmaDescriptor

### §2.1.1 Files & Line Counts

- `src/ptxsim/memory/tma_descriptor.h` — 168 LoC (verified `wc -l`)
- `src/ptxsim/memory/tma_descriptor.cpp` — 206 LoC (verified `wc -l`)

### §2.1.2 Existing ctest Coverage

36 TEST_CASE in `tests/unit/memory/test_tma_descriptor.cpp` per `proposal.md:28`.
`unit_tma_descriptor` PASS 0.07s (per `§1.4`).

### §2.1.3 UNVERIFIED Inventory (29 total: 17 .h + 12 .cpp, verified via grep)

Header (`tma_descriptor.h`):
- L8 — `LAYOUT NOTES — UNVERIFIED-AGAINST-HARDWARE` (file banner, Reference)
- L25 — ISA reference banner (Reference)
- L41 — `kTmaDescriptorSize = 128` (constant, Reference §9.7.13)
- L46 — `kTmaMaxRank = 5` (constant, Reference)
- L58 — `global_address` offset 0..7 (struct field doc)
- L63 — `global_dim[0..4]` offset 8..27 (struct field doc)
- L68 — RESERVED offset 28..31 (struct field doc)
- L72 — `global_stride[0..3]` offset 32..63 (struct field doc)
- L76 — `box_dim[0..4]` offset 64..83 (struct field doc)
- L80 — `element_stride[0..3]` offset 84..87 (struct field doc)
- L85 — `rank + control flags` offset 88..91 (struct field doc)
- L90 — `elemtype` offset 92 (struct field doc)
- L94 — `interleave_layout` offset 93 (struct field doc)
- L99 — `swizzle_mode` offset 94 (struct field doc)
- L103 — `fill_mode` offset 95 (struct field doc)
- L108 — RESERVED/im2col offset 96..127 (struct field doc)
- L165 — closing banner (Reference)

Impl (`tma_descriptor.cpp`): 12 UNVERIFIED at lines 5, 33, 98, 105, 114, 134,
145, 158, 168, 172, 176, 180 — all mark specific byte offsets in
`parse_descriptor_bytes()` body.

### §2.1.4 Grading

Per Decision 5: header UNVERIFIED are struct field doc annotations
(not handler body) → bare ISA reference in struct/reference context →
Exclusion rule (b) applies → **Reference / Verified-Ref**, not graded.

Impl UNVERIFIED: each marks a specific byte offset (e.g. L98 offset 0..7,
L105 offset 8..27, L114 offset 32..63, L134 offset 64..83, L145 offset 84..87,
L158 offset 88..91, L168 offset 92, L172 offset 93, L176 offset 94,
L180 offset 95). These are core data structure offsets per Decision 5 P0(b)
("core data structure size/offset"). However: tests in
`test_tma_descriptor.cpp` encode the same offsets (per `tma_descriptor.h:27-28`),
so a coordinated shift is required to silently regress — this means P0 risk
is real but **contained** by test invariant.

**Grade**: all 12 impl UNVERIFIED → **P0** (core data structure offset,
per Decision 5 P0(b)). Header 17 → Reference (excluded per Decision 5(b)).

### §2.1.5 Readiness Level: **L3**

Reasons:
- ctest green (L3(a) not triggered)
- **12 P0 UNVERIFIED** (≥3 P0 → L3(b) triggered per `design.md:41`)
- Not a fundamental invariant missing (L3(c) not triggered)

Per Decision 4, **L3 → Change-3 must wait for `fix-tcgen05-tma-descriptor-offsets`**.

## §2.2 Tmem

### §2.2.1 Files & Line Counts

- `src/ptxsim/memory/tmem.h` — 49 LoC (verified `wc -l`)
- `src/ptxsim/memory/tmem.cpp` — 61 LoC (verified `wc -l`)

### §2.2.2 Existing ctest Coverage

19 TEST_CASE in `tests/unit/memory/test_tmem.cpp` per `proposal.md:29`.
`unit_tmem` PASS 0.05s (per `§1.4`).

### §2.2.3 UNVERIFIED Inventory

**0 UNVERIFIED** in tmem.{h,cpp} (verified via `grep -c UNVERIFIED` = 0/0).
Constants `kSlotCount = 256`, `kSlotSize = 128`, `kTotalSize = 32*1024`
at `tmem.h:28-30` cite PTX ISA §9.7.13 but carry no UNVERIFIED tag.

### §2.2.4 Grading

No UNVERIFIED → no P0/P1/P2 inventory.

### §2.2.5 Readiness Level: **L1**

Reasons:
- ctest green (19/19 PASS)
- **zero** P0 UNVERIFIED (L1(b) satisfied)
- Code path coverage: `read()`/`write()`/`clear()`/`validate_slot_id()`
  all 4 public APIs exercised by tests (per `tmem.cpp:25,41,57,20`)
- `tmem.cpp:25-39` `read()` uses `std::lock_guard` (per `tmem.h:13` §2
  recursive-lock-avoidance rule); no public method calls another public
  method (verified by reading `tmem.cpp`)

Per Decision 4, **L1 → directly usable by Change-3**.

## §2.3 ClusterContext

### §2.3.1 Files & Line Counts

- `src/ptxsim/cluster/cluster_context.h` — 54 LoC (verified `wc -l`)
- `src/ptxsim/cluster/cluster_context.cpp` — 82 LoC (verified `wc -l`)

### §2.3.2 Existing ctest Coverage

16 TEST_CASE in `tests/unit/cluster/test_cluster_mode.cpp` per `proposal.md:30`.
`unit_cluster_mode` PASS 0.12s (per `§1.4`).

### §2.3.3 UNVERIFIED Inventory

**0 UNVERIFIED** in cluster_context.{h,cpp} (verified via grep = 0/0).

### §2.3.4 Cluster Integration Commit Provenance

Per `proposal.md:14` + `design.md:79` (R5):
- `e513235 feat(sim): cluster arrive/wait primitives (Fix #7, simplified—no distributed smem)`
  — base primitive (no distributed smem per Oracle simplification)
- `eb52af4 feat(cluster): wire ClusterContext into tcgen05 commit/wait (Fix #2)`
  — integration commit (wires `cta_cluster_arrive`/`cta_cluster_wait`
  into `wmma.cpp:526-528,559-561`)

Both commits verified via `git log --oneline -1 <hash>`.

### §2.3.5 Grading

No UNVERIFIED → no P0/P1/P2 inventory.

### §2.3.6 Readiness Level: **L1**

Reasons:
- ctest green (16/16 PASS)
- zero P0 UNVERIFIED
- Code path coverage: `cta_cluster_arrive()` (`cluster_context.cpp:44-63`)
  + `cta_cluster_wait()` (`cluster_context.cpp:65-82`) +
  `validate_cta_id()` (`cluster_context.cpp:36-38`) + `size()`
  (`cluster_context.cpp:40-42`) — all 4 public APIs covered
- `cluster_context.cpp:51` `arrive` uses `lock_guard`;
  `cluster_context.cpp:72` `wait` uses `unique_lock` + `cv_.wait`
  (per `cluster_context.h:11-12` §2 recursive-lock-avoidance rule)
- Cluster size invariant `[1, 8]` enforced at `cluster_context.cpp:26-27`
- `arrived_set_.size() == num_ctas_` release at `cluster_context.cpp:60`

Per Decision 4, **L1 → directly usable by Change-3**.

## §2.4 TcQueue

### §2.4.1 Files & Line Counts

- `src/ptxsim/async/tc_queue.h` — 74 LoC (verified `wc -l`)
- `src/ptxsim/async/tc_queue.cpp` — 108 LoC (verified `wc -l`)

### §2.4.2 Existing ctest Coverage

15 TEST_CASE in `tests/unit/async/test_tc_queue.cpp` per `proposal.md:31`.
`unit_tc_queue` PASS 0.02s (per `§1.4`).

### §2.4.3 UNVERIFIED Inventory

**0 UNVERIFIED** in tc_queue.{h,cpp} (verified via grep = 0/0).

### §2.4.4 Grading

No UNVERIFIED → no P0/P1/P2 inventory.

### §2.4.5 NO set_state(BAR_SYNC) Design Contract

Per `tc_queue.h:16-17`:
```
//     (is_blocked + status only; NO set_state(BAR_SYNC) — TcQueue is
//     not a CTA-level barrier and does not need BAR_SYNC fallback
//     path per Oracle Q1 hypothesis 2).
```

Per `tc_queue.cpp:13-14`:
```
//   - NO set_state(BAR_SYNC) — TcQueue is not a CTA-level barrier and does
//     not need the BAR_SYNC→is_blocked fallback path per Oracle Q1 hypothesis 2
```

This is **tc_queue module-internal Decision 7** (per `proposal.md:90` +
`design.md:19` MR-1 clarification), NOT ADR-0016 Decision 7.

Verified `grep "set_state" src/ptxsim/async/tc_queue.{h,cpp}` = 2 matches,
**both inside `//` comments** → 0 non-comment matches. Contract preserved.

`TcQueue::wait()` implementation at `tc_queue.cpp:89-108`:
- L98 captures `completion_pc = get_thread_pc(lane_id) + 1` (avoids PC drift
  per `tc_queue.h:24-26` Oracle Q4 hypothesis 2)
- L107 sets `ts.is_blocked = true`
- L108 sets `ts.status = ptxsim::ThreadStatus::Blocked`
- **No `set_state(BAR_SYNC)` call** (verified by reading lines 89-108)

`TcQueue::commit()` at `tc_queue.cpp:54-87`:
- L81 `advance_thread_pc(lane_id, completion_pc)` (pre-captured)
- L83 `ts.is_blocked = false`
- L84 `ts.status = ptxsim::ThreadStatus::Active`
- L85 `ts.is_active = true`
- **No `set_active_mask` call** (per `tc_queue.h:22-23` — OR semantics
  owned by `BarrierModule::release_warp_barrier`)

### §2.4.6 Readiness Level: **L1**

Reasons:
- ctest green (15/15 PASS)
- zero P0 UNVERIFIED
- Code path coverage: `commit()` (`tc_queue.cpp:54-87`) +
  `wait()` (`tc_queue.cpp:89-108`) + `clear()` (`tc_queue.cpp:39-43`) +
  `current_counter()` (`tc_queue.cpp:45-47`) +
  `pending_count()` (`tc_queue.cpp:49-52`) — all 5 public APIs covered
- Counter uses `std::atomic` + CAS fetch_max (per `tc_queue.h:8-9`)
  → no mutex on counter (separate sync primitive per §2 lessons-learned)
- Waiter list uses `std::mutex` (`tc_queue.h:71`) → no nested lock
  (per `tc_queue.h:10-12` §2 recursive-lock-avoidance)

Per Decision 4, **L1 → directly usable by Change-3**.

## §2.5 wmma.cpp handlers

### §2.5.1 Files & Line Counts

- `src/ptxsim/instructions/wmma.cpp` — 564 LoC total (verified `wc -l`)
- Handler segment L320-565 — 245 LoC (verified `wc -l` after `sed -n`)
- L62-317 fragment reference table — 256 entries (excluded per Decision 5(a))

### §2.5.2 Existing ctest Coverage

0 independent TEST_CASE for the L320-565 handler segment per `proposal.md:32`
— handlers covered transitively via Change-1 grammar/integration tests
(`unit_cluster_tcgen05_integration` = 2 TEST_CASE PASS, 0.02s per `§1.4`).

### §2.5.3 L62-317 Fragment Reference Table Exclusion

Per `design.md:61` Decision 5(a) exclusion rule: the 256-entry fragment
element reference table at `wmma.cpp:62-317` is **auto-generated reference
data** (PTX ISA §9.7.13 fragment layout static reference). Each entry is
annotated `// UNVERIFIED-AGAINST-HARDWARE` (256 entries, verified via
`sed -n '62,317p' | grep -c UNVERIFIED` = 256).

These are **NOT** implementation UNVERIFIED — they are reference data
in table context (per Decision 5(b) exclusion: bare ISA ref in
reference/table context is excluded). Per `design.md:62` NI-5 fix,
**only bare ISA-ref UNVERIFIED in handler function body count** —
the 9 handler-level UNVERIFIED below are in `execute_tcgen05_*` bodies.

### §2.5.4 UNVERIFIED Inventory (9 handler-level, verified via awk)

| Line | Handler | Context | Grade |
|---|---|---|---|
| 427 | `execute_tcgen05_ld` body | bare ISA ref, no specific gap | P2 (c) coverage gap |
| 449 | `execute_tcgen05_ld` body | `128-byte transfer per PTX ISA §9.7.13` | **P0** (b) core DS size |
| 455 | `execute_tcgen05_ld` body | `target slot 0 per PTX ISA §9.7.13` | **P0** hardcoded slot index |
| 467 | `execute_tcgen05_st` body | bare ISA ref, no specific gap | P2 (c) coverage gap |
| 489 | `execute_tcgen05_st` body | `128-byte transfer per PTX ISA §9.7.13` | **P0** (b) core DS size |
| 506 | `execute_tcgen05_commit` body | bare ISA ref, no specific gap | P2 (c) coverage gap |
| 522 | `execute_tcgen05_commit` body | `group_id=1 per PTX ISA §9.7.13` | **P0** (a) hardcoded invariant + (c) sync primitive |
| 538 | `execute_tcgen05_wait` body | bare ISA ref, no specific gap | P2 (c) coverage gap |
| 554 | `execute_tcgen05_wait` body | `group_id=1, lane_id=0` | **P0** (a) hardcoded group_id AND lane_id |

### §2.5.5 Grading Rationale

- **L427/L467/L506/L538** (bare ISA ref, no specific gap): P2 (c) coverage
  gap — comment cites ISA section but does not flag a specific invariant
  gap. Per Decision 5 P2(c) "test coverage gap rather than implementation
  gap". Handlers execute correctly (verified by `unit_cluster_tcgen05_integration`
  PASS) but lack direct handler-level TEST_CASE.
- **L449/L489** (128-byte transfer): P0 (b) — `Tmem::kSlotSize = 128`
  (`tmem.h:29`) is the core data structure size for TMEM slot. Hardcoded
  `Tmem::kSlotSize` in `std::memcpy` at `wmma.cpp:451-452` (ld) and
  `wmma.cpp:494-495` (st) — if hardware uses different transfer granularity,
  handler produces incorrect TMEM state.
- **L455** (target slot 0): P0 — hardcoded `tmem.write(0, ...)` at
  `wmma.cpp:456`. PTX ISA §9.7.13 `tcgen05.ld` accepts a slot operand;
  hardcoding slot 0 means handler ignores PTX operand.
- **L522** (group_id=1): P0 (a) + (c) — `cta->tc_queue().commit(1)` at
  `wmma.cpp:523`. group_id hardcoded to 1; PTX ISA allows multiple
  concurrent commit-groups. This is a sync primitive invariant per
  Decision 5 P0(c).
- **L554** (group_id=1, lane_id=0): P0 (a) — `cta->tc_queue().wait(warp, 0, 1)`
  at `wmma.cpp:556`. Both group_id (1) AND lane_id (0) hardcoded. PTX ISA
  `tcgen05.wait` accepts lane operand; hardcoding lane 0 means only lane 0
  blocks on completion.

**Total**: 5 P0 + 0 P1 + 4 P2.

### §2.5.6 Readiness Level: **L3**

Reasons:
- ctest green via `unit_cluster_tcgen05_integration` (only 2 TEST_CASE,
  transitively exercises handlers — L3(a) not triggered)
- **5 P0 UNVERIFIED** (≥3 P0 → L3(b) triggered per `design.md:41`)
- No fundamental `set_state` invariant missing (L3(c) not triggered —
  see §5 below for full verification)

Per Decision 4, **L3 → Change-3 must wait for
`fix-wmma-tcgen05-handler-unverified`**.

## §2.6 Cross-subsystem pipeline

### §2.6.1 End-to-end Call Chain

```
TmaDescriptor.parse_descriptor_bytes()  (tma_descriptor.cpp:84)
        ↓ desc->global_address
execute_tcgen05_ld body                 (wmma.cpp:423-461)
        ↓ std::memcpy from desc->global_address → tmp[]   (wmma.cpp:451-452)
        ↓ tmem.write(0, tmp, kSlotSize)                  (wmma.cpp:456)
Tmem backing store                      (tmem.cpp:41-55)
        ↓
execute_tcgen05_mma fragment arithmetic (wmma.cpp:361-421)
        ↓ reads tmem.read(a_slot/b_slot, ...)            (wmma.cpp:385,390)
        ↓ writes tmem.write(c_slot, ...)                 (wmma.cpp:416)
        ↓
execute_tcgen05_st body                 (wmma.cpp:463-500)
        ↓ tmem.read(0, tmp, kSlotSize)                   (wmma.cpp:492)
        ↓ std::memcpy to desc->global_address            (wmma.cpp:494-495)
        ↓
execute_tcgen05_commit body             (wmma.cpp:502-532)
        ↓ cta->tc_queue().commit(1)                      (wmma.cpp:523)
        ↓ cta->cluster_context().cta_cluster_arrive()    (wmma.cpp:526-528)
TcQueue counter + waiter release        (tc_queue.cpp:54-87)
ClusterContext arrive                   (cluster_context.cpp:44-63)
        ↓
execute_tcgen05_wait body               (wmma.cpp:534-565)
        ↓ cta->tc_queue().wait(warp, 0, 1)               (wmma.cpp:556)
        ↓ cta->cluster_context().cta_cluster_wait()      (wmma.cpp:559-561)
TcQueue block + completion_pc capture   (tc_queue.cpp:89-108)
ClusterContext wait                     (cluster_context.cpp:65-82)
```

### §2.6.2 Pipeline Coverage Gap

Only **2 TEST_CASE** in `tests/unit/cluster/test_cluster_tcgen05_integration.cpp`
(per `proposal.md:33` + Metis MR-3 fix: this is in `tests/unit/` NOT
`tests/integration/`). 2 cases cover:
- commit-group counter advancement
- wait + release cycle

**Not covered** (gaps):
- TmaDescriptor → Tmem → tcgen05.ld → tcgen05.mma → tcgen05.st round-trip
  with real descriptor bytes (no test wires `parse_descriptor_bytes` into
  the handler flow)
- Cluster arrive/wait with `cta->has_cluster_context() == true` (opt-in
  path at `wmma.cpp:526-528` only fires when cluster context attached —
  tests at `unit_cluster_tcgen05_integration` may or may not exercise this;
  2 TEST_CASE is insufficient to cover all 4 handler × cluster on/off
  combinations = 8 paths)
- Multi-warp commit-group contention (multiple warps calling
  `tc_queue().commit(1)` concurrently)

Per Decision 5 P2(c), these are coverage gaps (not impl gaps) → graded P2
in §2.5.4. Recommend `fix-tcgen05-pipeline-e2e-coverage` change to add
the missing integration tests.

## §3 Aggregate readiness

### §3.1 Readiness Matrix

| Subsystem | ctest | P0 count | Readiness |
|---|---|---|---|
| §2.1 TmaDescriptor | 36/36 PASS | 12 | **L3** |
| §2.2 Tmem | 19/19 PASS | 0 | L1 |
| §2.3 ClusterContext | 16/16 PASS | 0 | L1 |
| §2.4 TcQueue | 15/15 PASS | 0 | L1 |
| §2.5 wmma.cpp handlers | 2/2 PASS (transitive) | 5 | **L3** |

### §3.2 Aggregate Calculation (min-rule per design.md:47)

- L1 subsystems: Tmem, ClusterContext, TcQueue (3)
- L3 subsystems: TmaDescriptor, wmma.cpp handlers (2)
- L2 subsystems: 0

**aggregate = min(L1, L1, L1, L3, L3) = L3**

Per Decision 4, **aggregate readiness = L3 → Change-3 may NOT start**
until at least one `fix-*` change per L3 subsystem lifts it to ≥ L2.

### §3.3 Subsystem Summary

- **L1 (directly usable)**: Tmem, ClusterContext, TcQueue
- **L3 (blocks Change-3)**: TmaDescriptor (12 P0), wmma.cpp handlers (5 P0)

## §4 Issues + recommended fix-* changes

### §4.1 P0 Inventory (17 total)

| ID | Source | Line | Issue | Recommended fix-* change |
|---|---|---|---|---|
| P0-TMA-1 | tma_descriptor.cpp | 98 | offset 0..7 global_address | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-2 | tma_descriptor.cpp | 105 | offset 8..27 global_dim | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-3 | tma_descriptor.cpp | 114 | offset 32..63 global_stride | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-4 | tma_descriptor.cpp | 134 | offset 64..83 box_dim | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-5 | tma_descriptor.cpp | 145 | offset 84..87 element_stride | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-6 | tma_descriptor.cpp | 158 | offset 88..91 rank+ctrl | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-7 | tma_descriptor.cpp | 168 | offset 92 elemtype | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-8 | tma_descriptor.cpp | 172 | offset 93 interleave_layout | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-9 | tma_descriptor.cpp | 176 | offset 94 swizzle_mode | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-10 | tma_descriptor.cpp | 180 | offset 95 fill_mode | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-11 | tma_descriptor.cpp | 33 | note_reserved_bytes hook | `fix-tcgen05-tma-descriptor-offsets` |
| P0-TMA-12 | tma_descriptor.cpp | 5 | file header banner | `fix-tcgen05-tma-descriptor-offsets` |
| P0-WMMA-1 | wmma.cpp | 449 | 128-byte transfer (ld) | `fix-wmma-tcgen05-handler-unverified` |
| P0-WMMA-2 | wmma.cpp | 455 | target slot 0 (ld) | `fix-wmma-tcgen05-handler-unverified` |
| P0-WMMA-3 | wmma.cpp | 489 | 128-byte transfer (st) | `fix-wmma-tcgen05-handler-unverified` |
| P0-WMMA-4 | wmma.cpp | 522 | group_id=1 (commit) | `fix-wmma-tcgen05-handler-unverified` |
| P0-WMMA-5 | wmma.cpp | 554 | group_id=1, lane_id=0 (wait) | `fix-wmma-tcgen05-handler-unverified` |

### §4.2 P1 Inventory

**0 P1** — no UNVERIFIED annotation falls under Decision 5 P1 (fragment
element index accuracy / bit layout not affecting correctness / swizzle
with fallback). TmaDescriptor UNVERIFIED at struct field doc context are
excluded per Decision 5(b); impl UNVERIFIED are P0 (core offsets).

### §4.3 P2 Inventory

| ID | Source | Line | Issue | Recommended fix-* change |
|---|---|---|---|---|
| P2-WMMA-1 | wmma.cpp | 427 | bare ISA ref (ld body) | `fix-wmma-tcgen05-handler-coverage` |
| P2-WMMA-2 | wmma.cpp | 467 | bare ISA ref (st body) | `fix-wmma-tcgen05-handler-coverage` |
| P2-WMMA-3 | wmma.cpp | 506 | bare ISA ref (commit body) | `fix-wmma-tcgen05-handler-coverage` |
| P2-WMMA-4 | wmma.cpp | 538 | bare ISA ref (wait body) | `fix-wmma-tcgen05-handler-coverage` |
| P2-PIPE-1 | (cross-subsystem) | — | missing e2e pipeline test | `fix-tcgen05-pipeline-e2e-coverage` |
| P2-PIPE-2 | (cross-subsystem) | — | missing cluster opt-in path test | `fix-tcgen05-pipeline-e2e-coverage` |
| P2-PIPE-3 | (cross-subsystem) | — | missing multi-warp commit contention | `fix-tcgen05-pipeline-e2e-coverage` |

### §4.4 Recommended fix-* Changes

1. **`fix-tcgen05-tma-descriptor-offsets`** (P0, 12 items)
   - Lifts §2.1 TmaDescriptor from L3 → L2 (or L1 if all offsets verified)
   - Verification path: real `cuTensorMapEncodeTiled` output OR
     `cuobjdump -xptx` dump per `tma_descriptor.h:21-23` Gate G5
   - **Must** complete before Change-3 handlers can safely consume
     `TmaDescriptor` struct

2. **`fix-wmma-tcgen05-handler-unverified`** (P0, 5 items)
   - Lifts §2.5 wmma.cpp handlers from L3 → L2
   - Replace hardcoded `slot 0` / `group_id=1` / `lane_id=0` with
     PTX operand extraction (per `wmma.cpp:456,523,556`)
   - Verify 128-byte transfer granularity against PTX ISA §9.7.13
   - **Must** complete before Change-3 may proceed

3. **`fix-wmma-tcgen05-handler-coverage`** (P2, 4 items)
   - Adds independent TEST_CASE for each of the 4 handlers
   - May proceed in parallel with Change-3 (P2 = defer allowed)

4. **`fix-tcgen05-pipeline-e2e-coverage`** (P2, 3 items)
   - Adds end-to-end integration tests in `tests/integration/tcgen05/`
   - Covers TmaDescriptor → Tmem → TcQueue → wmma.cpp pipeline
   - May proceed in parallel with Change-3

### §4.5 Change-3 Proceed Decision

Per Decision 4 (`design.md:47`): Change-3 may start iff aggregate ≥ L2.

Current aggregate = L3 (per §3.2). **Change-3 may NOT proceed** until
both `fix-tcgen05-tma-descriptor-offsets` AND
`fix-wmma-tcgen05-handler-unverified` complete and lift their respective
subsystems to ≥ L2.

## §5 NO set_state(BAR_SYNC) design contract verification

### §5.1 Contract Source

Per `tc_queue.h:16-17`:
```
//     (is_blocked + status only; NO set_state(BAR_SYNC) — TcQueue is
//     not a CTA-level barrier and does not need BAR_SYNC fallback
//     path per Oracle Q1 hypothesis 2).
```

Per `tc_queue.cpp:13-14`:
```
//   - NO set_state(BAR_SYNC) — TcQueue is not a CTA-level barrier and does
//     not need the BAR_SYNC→is_blocked fallback path per Oracle Q1 hypothesis 2
```

This is **tc_queue module-internal Decision 7** per `proposal.md:90` +
`design.md:19` MR-1 clarification — NOT ADR-0016 Decision 7.

### §5.2 grep Verification

Command: `grep -rn "set_state.*BAR_SYNC" src/ptxsim/async/tc_queue.{h,cpp}
src/ptxsim/instructions/wmma.cpp`

Result: **2 matches, both inside `//` comments** (`tc_queue.h:16` +
`tc_queue.cpp:13`). Non-comment matches = **0**. Contract preserved.

Cross-project grep (`grep -rn "set_state.*BAR_SYNC" src/ --include="*.cpp"
--include="*.h"`): only 1 non-comment match at `src/ptxsim/instructions/barrier.cpp:313`
— that is the BarHandler's intended `set_state(BAR_SYNC)` per
`ptx-barrier-mechanism` skill (path B CTA-level barrier). TcQueue does
NOT use this path.

### §5.3 Implementation Verification

`TcQueue::wait()` at `tc_queue.cpp:89-108`:
```cpp
auto& ts = warp_ctx->get_warp_state().threads[lane_id];
ts.is_blocked = true;                                  // L107
ts.status = ptxsim::ThreadStatus::Blocked;             // L108
```

No `set_state(BAR_SYNC)` call. Thread blocking achieved via `is_blocked`
+ `status=Blocked` only (per BarWarpSyncHandler pattern in
`ptx-barrier-mechanism` skill — "BarWarpSyncHandler does NOT set
thread->state = BAR_SYNC; the only mark of thread blocking is
is_blocked = true").

### §5.4 Handler Call Verification

`wmma.cpp:556` (`execute_tcgen05_wait` body):
```cpp
cta->tc_queue().wait(warp, 0, 1);
```

This invokes `TcQueue::wait()` directly — does NOT call `set_state(BAR_SYNC)`.
The block/release lifecycle is fully owned by TcQueue's `is_blocked` +
`status` fields (per `tc_queue.h:13-14` + `tc_queue.cpp:11-12`).

### §5.5 Contract Status: **VERIFIED**

- grep = 0 non-comment matches in tc_queue.{h,cpp} + wmma.cpp
- Implementation uses `is_blocked` + `status=Blocked` only (L107-108)
- Handler call at `wmma.cpp:556` does not route through `set_state`
- Per Decision 4 L3(c): fundamental invariant missing → L3 trigger NOT fired

## §6 Conclusion + Change-3 dependency statement

### §6.1 Aggregate Readiness Verdict

**Aggregate readiness = L3** (per §3.2 min-rule across 5 subsystems).

- L1: Tmem, ClusterContext, TcQueue (3 subsystems, directly usable)
- L3: TmaDescriptor (12 P0), wmma.cpp handlers (5 P0) (2 subsystems,
  block Change-3)
- L2: 0

### §6.2 Change-3 Dependency Statement

Per Decision 4 (`design.md:47`): **Change-3 may start iff aggregate ≥ L2**.

Current aggregate = **L3 → Change-3 may NOT proceed**.

Required `fix-*` changes (in priority order):
1. `fix-tcgen05-tma-descriptor-offsets` — lifts §2.1 from L3 → ≥ L2
2. `fix-wmma-tcgen05-handler-unverified` — lifts §2.5 from L3 → ≥ L2

After both complete and re-audit confirms aggregate ≥ L2, Change-3 may start.

Optional `fix-*` (may proceed in parallel with Change-3):
3. `fix-wmma-tcgen05-handler-coverage` (P2, 4 items)
4. `fix-tcgen05-pipeline-e2e-coverage` (P2, 3 items)

### §6.3 NO set_state(BAR_SYNC) Contract

VERIFIED (per §5). This is **tc_queue module-internal Decision 7**
(per `proposal.md:90` + `design.md:19`), NOT ADR-0016 Decision 7.
0 non-comment `set_state(BAR_SYNC)` matches in tc_queue.{h,cpp} +
wmma.cpp. `TcQueue::wait()` uses `is_blocked` + `status=Blocked`
per BarWarpSyncHandler pattern. Handler at `wmma.cpp:556` calls
`tc_queue().wait()` directly — does NOT route through `set_state`.

### §6.4 Open Items for Orchestrator

1. **P0-TMA-1..12** (12 items): all 12 TmaDescriptor impl UNVERIFIED are
   P0 due to Decision 5 P0(b) "core data structure size/offset". They are
   structurally contained by the test invariant (per `tma_descriptor.h:27-28`
   "tests encode the same assumed offsets, so a coordinated shift is
   required to silently regress"). Orchestrator may decide whether to
   treat them as 1 fix-* change (recommended) or split.

2. **P0-WMMA-1..5** (5 items): the 5 wmma.cpp handler UNVERIFIED involve
   hardcoded operands (`slot 0` at L456, `group_id=1` at L523/L556,
   `lane_id=0` at L556). These are handler correctness issues that
   require PTX operand extraction logic — likely belongs to Change-3
   handler scope rather than a separate `fix-*`. Orchestrator may decide
   to fold into Change-3 if handler rewrite is part of Change-3 deliverables.

3. **Cross-subsystem pipeline e2e coverage**: 2 TEST_CASE in
   `unit_cluster_tcgen05_integration` is below the 8-path coverage
   matrix (4 handlers × cluster on/off). Recommend `fix-tcgen05-pipeline-e2e-coverage`
   even if P2 — without it, Change-3 handler bugs may surface only at
   integration time.

### §6.5 References

- `openspec/changes/extend-blackwell-tcgen05-infra/proposal.md` (Decision 4/5 source)
- `openspec/changes/extend-blackwell-tcgen05-infra/design.md` (D4/D5 detail)
- `docs/adr/0016-blackwell-only-tcgen05.md` (architectural lock)
- `.opencode/skills/ptx-barrier-mechanism/SKILL.md` (BarWarpSyncHandler pattern)
- `.opencode/skills/ptx-lessons-learned/SKILL.md` §1, §2 (cross-module state,
  recursive lock avoidance)
- `openspec/changes/archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/`
  (Phase 0 original archive — TmaDescriptor + Tmem + Cluster + TcQueue)

**Audit complete. No source files modified.**
