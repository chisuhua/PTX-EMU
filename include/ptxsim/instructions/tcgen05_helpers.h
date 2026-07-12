#pragma once
// =============================================================================
// tcgen05_helpers.h — shared fragment-arithmetic helpers (Phase 2.5).
//
// Extracted from src/ptxsim/instructions/tcgen05.cpp:312-375 by
// `feat(tcgen05): extract fragment_mma_f16 helper` (Oracle 2026-07-08
// Q4-recommendation, pre-Phase 3 refactor per implement-tcgen05-handlers-extended).
//
// Per ptx-lessons-learned §6: helpers are declared directly in `ptxsim`
// namespace (NOT anonymous namespace), so unit/integration tests can
// reach them through forward declarations or direct calls.
//
// Usage:
//   processTcgen05Mma(ctx, instr)         // regular mma path
//     └── tcgen05_fragment_mma_f16(tmem)  // shared kernel
//   processTcgen05Mma(ctx, instr_with_ws) // ws routing (Phase 3)
//     └── tcgen05_fragment_mma_f16(tmem)  // same shared kernel
//
// Fragment layout (multi-warp aware, post C4 fix):
//   A slots: [0..63]      (shared input fragments, lane_id * 2)
//   B slots: [0..63]      (shared input fragments, lane_id * 2 + 1)
//   C slots: [64..95]     (per-warp owned, warp_id * 32 + 64 + lane_id)
//   warp 0: [64..95], warp 1: [96..127], warp 2: [128..159], warp 3: [160..191]
//   Each warp owns 32 unique slots; A/B remain shared input.
//   - A fragment: 8 rows × 8 cols (64 f16 elements)
//   - B fragment: 8 rows × 4 cols (32 f16 elements) — note ROWS shared with A
//   - C fragment: 8 rows × 4 cols (32 f32 elements per lane, 128 bytes fills slot).
//   - C output: 32 f32 elements per lane (128 bytes, fills slot completely).
//     Storage format changed from f16 in fix-tcgen05-mma-accumulator-and-f32-storage
//     Phase 2 commit (Oracle H2 fix per PTX ISA §9.7.16).
//   - Accumulation: C[i][j] = sum_k A[i][k] * B[k][j], f16↔f32 round-trip
// =============================================================================

#include "ptxsim/memory/tmem.h"

namespace ptxsim {

// 32-lane × 8×4 f16 fragment multiply-accumulate.
// Reads A from TMEM slots [0..63] and B from TMEM slots [0..63] (interleaved
// with A on a per-lane basis), writes C to TMEM slots
// [warp_id * 32 + 64 .. warp_id * 32 + 95].
//
// warp_id: per-warp slot offset to prevent multi-warp C slot conflict.
//          - 0 = single-warp mode (backward compatible)
//          - N = warp N owns C slots [N*32+64 : N*32+95]
//          - A/B slots [0..63] remain shared input fragments.
//          - Caller MUST pass warp->get_warp_id() (or 0 for single-warp code).
//          - Throws std::invalid_argument if warp_id < 0.
//
// When accumulate=false (default): C slot is initialized to zero before
//   the mma sum, matching pre-H1 overwrite behavior.
// When accumulate=true: existing C slot values (f32) are read back,
//   accumulated in f32, and written back in f32 (Phase 2 H2).
//
// Caller must ensure:
//   - TMEM is allocated (via TmemAllocator or direct write)
//   - Caller has validated scope (Q3-A: Q_F16 + Q_TCGEN_CTA_GROUP for ws path)
//   - Exclusive access to Tmem slots for the duration of this call.
//     Each Tmem::read / Tmem::write call holds Tmem::mu_ independently,
//     but the read-compute-write sequence here is NOT atomic. Concurrent
//     modification between read and write would cause TOCTOU corruption.
//     Currently safe because the SM scheduler runs one warp at a time
//     (sequential execution); if the simulator ever evolves to multi-warp
//     concurrency, callers must add a higher-level lock.
//     [SINGLE-WARP ASSUMPTION] — per lessons-learned §28.
//
// UNVERIFIED-AGAINST-HARDWARE: ws-specific weight-stationary layout transform
// is NOT applied — the simulator uses identical fragment arithmetic for both
// regular mma and mma.ws (deferred per tasks.md §4).
void tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id,
                              bool accumulate = false);

} // namespace ptxsim