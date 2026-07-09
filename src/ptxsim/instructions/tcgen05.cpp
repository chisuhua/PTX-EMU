// src/ptxsim/instructions/tcgen05.cpp
// Blackwell tcgen05.* instruction handlers (ADR-0016, Phase 1-3).
//
// Extracted from wmma.cpp: per-op handlers for tcgen05.{mma,ld,st,commit,wait}.
// Dispatch is via Tcgen05OpKind (parsed by visitTcgen05Inst from the grammar's
// single tcgen05Inst rule), NOT via qualifier-based detection — see ADR-0016.
//
// Per ptx-lessons-learned: no anonymous namespace around helpers; helpers are
// declared static at file scope or placed directly in ptxsim namespace.
//
// All fragment-element UNVERIFIED-AGAINST-HARDWARE annotations (256 total,
// 8 lanes × 8 rows × 4 cols) are preserved verbatim from wmma.cpp lines
// 62-317, with the PTX ISA section corrected from §9.7.13 to §9.7.16 (the
// tcgen05-specific section per ADR-0016).

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/instructions/tcgen05.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/async/tc_queue.h"
#include "ptxsim/utils/half_utils.h"
#include "utils/logger.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace ptxsim {

// ---------------------------------------------------------------------------
// 256 per-fragment-element UNVERIFIED-AGAINST-HARDWARE annotations.
// 8 lanes × 8 rows × 4 cols = 256 elements; one annotation per element,
// preserved verbatim from the original wmma.cpp implementation (lines
// 62-317). Section reference corrected to §9.7.16 (tcgen05-specific) per
// ADR-0016 — the original §9.7.13 reference tracked the shared TMA/TMEM
// infrastructure section, not the tcgen05 MMA semantics section.
// ---------------------------------------------------------------------------
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][3] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][0] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][1] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][2] PTX ISA §9.7.16
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][3] PTX ISA §9.7.16

// ---------------------------------------------------------------------------
// processTcgen05Mma — 32 lane × 8x4 f16 matrix multiply (PTX ISA §9.7.16).
//
// Per-lane fragment layout (extracted from wmma.cpp:374-420):
//   - A input:  TMEM slots [0..63],   a_slot = lane_id * 2
//   - B input:  TMEM slots [0..63],   b_slot = lane_id * 2 + 1
//   - C output: TMEM slots [64..95],  c_slot = 64 + lane_id
//   - A fragment: 8 rows × 8 cols (64 f16 elements)
//   - B fragment: 8 rows × 4 cols (32 f16 elements) — note ROWS shared with A
//   - C fragment: 8 rows × 4 cols (32 f16 elements)
//   - Accumulation: C[i][j] = sum_k A[i][k] * B[k][j], f16↔f32 round-trip
//
// Dispatch is via instr.op_kind == Tcgen05OpKind::MMA (NOT qualifier-based).
// ---------------------------------------------------------------------------
void processTcgen05Mma(ThreadContext* context, const Tcgen05Instr& instr) {
    (void)instr;  // op_kind already validated by caller dispatch

    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.mma: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.mma", "tcgen05.mma requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.mma: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.mma", "tcgen05.mma requires an active CTAContext");
    }

    Tmem& tmem = cta->tmem();
    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        size_t a_slot = static_cast<size_t>(lane_id) * 2;
        size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1;
        size_t c_slot = static_cast<size_t>(64) + static_cast<size_t>(lane_id);

        std::array<uint8_t, Tmem::kSlotSize> a_buf{};
        tmem.read(a_slot, a_buf.data(), Tmem::kSlotSize);
        const uint16_t* a_raw =
            reinterpret_cast<const uint16_t*>(a_buf.data());

        std::array<uint8_t, Tmem::kSlotSize> b_buf{};
        tmem.read(b_slot, b_buf.data(), Tmem::kSlotSize);
        const uint16_t* b_raw =
            reinterpret_cast<const uint16_t*>(b_buf.data());

        float a_flat[ROWS * COLS_A];
        float b_flat[ROWS * COLS_B];
        for (int k = 0; k < ROWS * COLS_A; ++k)
            a_flat[k] = f16_to_f32(a_raw[k]);
        for (int k = 0; k < ROWS * COLS_B; ++k)
            b_flat[k] = f16_to_f32(b_raw[k]);

        std::array<uint16_t, ROWS * COLS_B> c_frag{};
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS_B; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < COLS_A; ++k) {
                    sum += a_flat[i * COLS_A + k] *
                           b_flat[k * COLS_B + j];
                }
                c_frag[i * COLS_B + j] = f32_to_f16(sum);
            }
        }

        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        std::memcpy(c_buf.data(), c_frag.data(),
                    c_frag.size() * sizeof(uint16_t));
        tmem.write(c_slot, c_buf.data(), Tmem::kSlotSize);
    }

    PTX_DEBUG_EMU("tcgen05.mma.cta_group::1.kind::f16 executed "
                  "(32 lanes x 8x4 fragments)");
}

// ---------------------------------------------------------------------------
// processTcgen05Ld — 128-byte load from TMA descriptor to TMEM
// (PTX ISA §9.7.16). Extracted from wmma.cpp:423-461.
//
// Layout: TmaDescriptorStore at cta->tma_descriptor_store().load(0)
//         128-byte transfer (Tmem::kSlotSize) desc->global_address → TMEM slot 0
// ---------------------------------------------------------------------------
void processTcgen05Ld(ThreadContext* context, const Tcgen05Instr& instr) {
    (void)instr;  // op_kind already validated by caller dispatch
    // PTX ISA §9.7.16: tcgen05.ld — load from TMA descriptor to TMEM
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.ld: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.ld", "tcgen05.ld requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.ld: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.ld", "tcgen05.ld requires an active CTAContext");
    }

    TmaDescriptorStore& desc_store = cta->tma_descriptor_store();
    const TmaDescriptor* desc = desc_store.load(0);
    if (!desc) {
        PTX_ERROR_EMU("tcgen05.ld: no TMA descriptor found for cta_id=0");
        throw UnsupportedInstructionException(
            "tcgen05.ld", "tcgen05.ld requires a TMA descriptor");
    }

    // UNVERIFIED-AGAINST-HARDWARE — 128-byte transfer per PTX ISA §9.7.16
    uint8_t tmp[Tmem::kSlotSize];
    std::memcpy(tmp, reinterpret_cast<const void*>(desc->global_address),
                Tmem::kSlotSize);

    Tmem& tmem = cta->tmem();
    // UNVERIFIED-AGAINST-HARDWARE — target slot 0 per PTX ISA §9.7.16
    tmem.write(0, tmp, Tmem::kSlotSize);

    PTX_DEBUG_EMU("tcgen05.ld: TMA desc global=0x%016lx → TMEM slot 0 "
                  "(%zu bytes)",
                  desc->global_address, Tmem::kSlotSize);
}

// ---------------------------------------------------------------------------
// processTcgen05St — 128-byte store from TMEM to TMA descriptor
// (PTX ISA §9.7.16). Extracted from wmma.cpp:463-500.
//
// Layout: TmaDescriptorStore at cta->tma_descriptor_store().load(0)
//         128-byte transfer (Tmem::kSlotSize) TMEM slot 0 → desc->global_address
// ---------------------------------------------------------------------------
void processTcgen05St(ThreadContext* context, const Tcgen05Instr& instr) {
    (void)instr;  // op_kind already validated by caller dispatch
    // PTX ISA §9.7.16: tcgen05.st — store from TMEM to TMA descriptor
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.st: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.st", "tcgen05.st requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.st: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.st", "tcgen05.st requires an active CTAContext");
    }

    TmaDescriptorStore& desc_store = cta->tma_descriptor_store();
    const TmaDescriptor* desc = desc_store.load(0);
    if (!desc) {
        PTX_ERROR_EMU("tcgen05.st: no TMA descriptor found for cta_id=0");
        throw UnsupportedInstructionException(
            "tcgen05.st", "tcgen05.st requires a TMA descriptor");
    }

    // UNVERIFIED-AGAINST-HARDWARE — 128-byte transfer per PTX ISA §9.7.16
    uint8_t tmp[Tmem::kSlotSize];
    Tmem& tmem = cta->tmem();
    tmem.read(0, tmp, Tmem::kSlotSize);

    std::memcpy(reinterpret_cast<void*>(desc->global_address),
                tmp, Tmem::kSlotSize);

    PTX_DEBUG_EMU("tcgen05.st: TMEM slot 0 → TMA desc global=0x%016lx "
                  "(%zu bytes)",
                  desc->global_address, Tmem::kSlotSize);
}

// ---------------------------------------------------------------------------
// processTcgen05Commit — tc_queue commit + cluster arrive
// (PTX ISA §9.7.16). Extracted from wmma.cpp:502-532.
//
// Per ADR-0016 Phase 0.3: cluster arrive is opt-in via has_cluster_context().
// ---------------------------------------------------------------------------
void processTcgen05Commit(ThreadContext* context, const Tcgen05Instr& instr) {
    (void)instr;  // op_kind already validated by caller dispatch
    // PTX ISA §9.7.16: tcgen05.commit — commit async tensor ops
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.commit: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.commit",
            "tcgen05.commit requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.commit: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.commit",
            "tcgen05.commit requires an active CTAContext");
    }

    // UNVERIFIED-AGAINST-HARDWARE — group_id=1 per PTX ISA §9.7.16
    cta->tc_queue().commit(1);

    // Wire ClusterContext: opt-in cluster arrive (per ADR-0016 Phase 0.3)
    if (cta->has_cluster_context()) {
        PTX_DEBUG_EMU("tcgen05.commit: cluster arrive cta_id=%d", cta->blockIdx.x);
        cta->cluster_context().cta_cluster_arrive(cta->blockIdx.x);
    }

    PTX_DEBUG_EMU("tcgen05.commit: group_id=1 committed");
}

// ---------------------------------------------------------------------------
// processTcgen05Wait — tc_queue wait + cluster wait
// (PTX ISA §9.7.16). Extracted from wmma.cpp:534-565.
//
// Per ADR-0016 Phase 0.3: cluster wait is opt-in via has_cluster_context().
// ---------------------------------------------------------------------------
void processTcgen05Wait(ThreadContext* context, const Tcgen05Instr& instr) {
    (void)instr;  // op_kind already validated by caller dispatch
    // PTX ISA §9.7.16: tcgen05.wait — wait for async tensor ops completion
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.wait: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "tcgen05.wait",
            "tcgen05.wait requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.wait: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "tcgen05.wait",
            "tcgen05.wait requires an active CTAContext");
    }

    // UNVERIFIED-AGAINST-HARDWARE — group_id=1, lane_id=0
    // per PTX ISA §9.7.16
    cta->tc_queue().wait(warp, 0, 1);

    // Wire ClusterContext: opt-in cluster wait (per ADR-0016 Phase 0.3)
    if (cta->has_cluster_context()) {
        PTX_DEBUG_EMU("tcgen05.wait: cluster wait cta_id=%d", cta->blockIdx.x);
        cta->cluster_context().cta_cluster_wait(cta->blockIdx.x);
    }

    PTX_DEBUG_EMU("tcgen05.wait: waiting on group_id=1 for lane 0");
}

// Tcgen05Handler::processTcgen05Operation — dispatches on instr.op_kind
// to the 5 per-op free functions (kept for backward compat with
// fix-tcgen05-test-coverage-gaps dead-code coverage test). Deferred
// op_kinds (ALLOC/DEALLOC/CP/MMA_WS/FENCE) throw to surface them
// until implement-tcgen05-handlers-extended lands.
}  // namespace ptxsim

// Tcgen05Handler is in global namespace (per instruction_handlers.h
// X-Macro factory registration pattern).
void Tcgen05Handler::processTcgen05Operation(
    ThreadContext *context, void **operands,
    const std::vector<Qualifier> &qualifiers,
    const Tcgen05Instr &instr) {
    (void)operands;
    (void)qualifiers;

    switch (instr.op_kind) {
    case Tcgen05OpKind::MMA:
        ptxsim::processTcgen05Mma(context, instr);
        break;
    case Tcgen05OpKind::LD:
        ptxsim::processTcgen05Ld(context, instr);
        break;
    case Tcgen05OpKind::ST:
        ptxsim::processTcgen05St(context, instr);
        break;
    case Tcgen05OpKind::COMMIT:
        ptxsim::processTcgen05Commit(context, instr);
        break;
    case Tcgen05OpKind::WAIT:
        ptxsim::processTcgen05Wait(context, instr);
        break;
    case Tcgen05OpKind::ALLOC:
        ptxsim::processTcgen05Alloc(context, instr);
        break;
    case Tcgen05OpKind::DEALLOC:
        ptxsim::processTcgen05Dealloc(context, instr);
        break;
    case Tcgen05OpKind::RELINQUISH:
        ptxsim::processTcgen05Relinquish(context, instr);
        break;
    case Tcgen05OpKind::CP:
        ptxsim::processTcgen05Cp(context, instr);
        break;
    case Tcgen05OpKind::MMA_WS:
    case Tcgen05OpKind::FENCE:
        throw UnsupportedInstructionException(
            "tcgen05.*",
            "op_kind " + std::to_string(static_cast<int>(instr.op_kind)) +
            " not yet implemented (per ADR-0016, deferred but wired; "
            "see implement-tcgen05-handlers-extended)");
        break;
    }
}
