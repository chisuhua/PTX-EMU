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
// Fragment layout (hardcoded per tcgen05.cpp:334-336):
//   - A input:  TMEM slots [0..63],   a_slot = lane_id * 2
//   - B input:  TMEM slots [0..63],   b_slot = lane_id * 2 + 1
//   - C output: TMEM slots [64..95],  c_slot = 64 + lane_id
//   - A fragment: 8 rows × 8 cols (64 f16 elements)
//   - B fragment: 8 rows × 4 cols (32 f16 elements) — note ROWS shared with A
//   - C fragment: 8 rows × 4 cols (32 f16 elements)
//   - Accumulation: C[i][j] = sum_k A[i][k] * B[k][j], f16↔f32 round-trip
// =============================================================================

#include "ptxsim/memory/tmem.h"

namespace ptxsim {

// 32-lane × 8×4 f16 fragment multiply-accumulate.
// Reads A from TMEM slots [0..63] and B from TMEM slots [0..63] (interleaved
// with A on a per-lane basis), writes C to TMEM slots [64..95].
//
// Caller must ensure:
//   - TMEM is allocated (via TmemAllocator or direct write)
//   - Caller has validated scope (Q3-A: Q_F16 + Q_TCGEN_CTA_GROUP for ws path)
//
// UNVERIFIED-AGAINST-HARDWARE: ws-specific weight-stationary layout transform
// is NOT applied here — single-warp simplification (both regular mma and mma.ws
// use identical fragment arithmetic in this simulator). See
// implement-tcgen05-handlers-extended/tasks.md §4 for future work.
void tcgen05_fragment_mma_f16(Tmem& tmem);

} // namespace ptxsim