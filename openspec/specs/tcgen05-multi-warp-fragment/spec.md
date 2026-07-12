# tcgen05-multi-warp-fragment Specification

## Purpose
TBD - created by archiving change fix-tcgen05-multi-warp-fragment. Update Purpose after archive.
## Requirements
### Requirement: tcgen05 fragment helper SHALL support multi-warp C slot offset (Oracle C4 fix)

The system SHALL modify `tcgen05_fragment_mma_f16` helper to accept an `int warp_id` parameter (located between `Tmem& tmem` and `bool accumulate = false`). The C slot computation SHALL be `c_slot = warp_id * 32 + 64 + lane_id`, providing each warp with an independent C slot range `[warp_id*32+64 : warp_id*32+95]`. A/B slots `[0..63]` SHALL remain shared input fragments, matching the FlashAttention FA3 producer-consumer model where Q tile is per-warp and K tile is shared via `tcgen05.cp`.

#### Scenario: single-warp backward compatibility (warp_id = 0)

- **WHEN** `tcgen05_fragment_mma_f16(tmem, /*warp_id=*/0, /*accumulate=*/false)` is called
- **THEN** `c_slot` is computed as `0 * 32 + 64 + lane_id == 64 + lane_id` (equivalent to pre-C4-fix behavior)
- **AND** all existing tests pass without modification (no `warp_id` change in callers passing `0`)

#### Scenario: 2-warp C slot isolation

- **WHEN** two warps (warp_id 0 and warp_id 1) each call `tcgen05_fragment_mma_f16(tmem, warp_id, false)` with distinct lane values
- **THEN** warp 0 writes C to TMEM slots `[64..95]` (per `warp_id * 32 + 64 + lane_id`)
- **AND** warp 1 writes C to TMEM slots `[96..127]` (per `warp_id * 32 + 64 + lane_id`)
- **AND** no slot overlap occurs (each warp owns 32 unique slots)
- **AND** reading warp 0's C slot yields the value warp 0 wrote (no cross-warp contamination)

#### Scenario: 4-warp C slot layout

- **WHEN** four warps (warp_id 0..3) each call `tcgen05_fragment_mma_f16(tmem, warp_id, false)`
- **THEN** warp 0 owns slots `[64..95]`, warp 1 owns `[96..127]`, warp 2 owns `[128..159]`, warp 3 owns `[160..191]`
- **AND** total C slot usage is 128 slots (16KB out of Tmem `kTotalSize` = 32KB, leaving 16KB for A/B and system) per design.md D4 capacity table

#### Scenario: caller passes warp->get_warp_id()

- **WHEN** `processTcgen05Mma` (in `src/ptxsim/instructions/tcgen05.cpp`) dispatches a Tcgen05Instr
- **THEN** it invokes `tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), /*accumulate=*/false)` (per Oracle Q4 Option a, design.md D1)
- **AND** warp_id is fetched from `WarpContext::get_warp_id()` (existing public API, see `tcgen05_alloc.cpp:68,92,147,191` for prior uses)

#### Scenario: invalid warp_id throws clear exception

- **WHEN** `tcgen05_fragment_mma_f16` is called with `warp_id < 0`
- **THEN** the helper throws `std::invalid_argument` with message containing "warp_id must be >= 0"
- **AND** no TMEM write occurs
- **AND** this guards against accidental unsigned-cast or signed/unsigned mismatch bugs in callers (per Oracle Risk R6)

#### Scenario: A/B slot layout unchanged (shared input)

- **WHEN** warp 0 calls `tcgen05_fragment_mma_f16(tmem, 0, false)` then warp 1 calls the same with warp_id=1
- **THEN** A slots read by warp 0 are `[lane_id * 2]` and B slots are `[lane_id * 2 + 1]` for both warps (shared input)
- **AND** C slots are partitioned per warp (per Oracle Q4 Option a, design.md D2 minimal fix)
- **NOTE**: per-warp A/B partitioning is a P2 follow-up if future kernels require it (decision D2 trade-off)

#### Scenario: accumulate=true path composes with multi-warp layout

- **WHEN** two warps each call `tcgen05_fragment_mma_f16(tmem, warp_id, /*accumulate=*/true)` twice
- **THEN** warp 0's 2nd mma reads C slot `64 + lane_id` (warp 0's range) and accumulates with `f32_to_f32(existing)`
- **AND** warp 1's 2nd mma reads C slot `96 + lane_id` (warp 1's range) and accumulates independently
- **AND** cross-warp accumulation does not occur (each warp accumulates within its own slot range)
- **NOTE**: `accumulate=true` parameter is added by sister change `fix-tcgen05-mma-accumulator-and-f32-storage`; this scenario verifies the two parameters compose correctly

