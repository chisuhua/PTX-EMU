#pragma once
// =============================================================================
// WARNING: These functions are currently DEAD CODE.
//
// `processTcgen05Mma`, `processTcgen05Ld`, `processTcgen05St`,
// `processTcgen05Commit`, `processTcgen05Wait` are implemented in
// src/ptxsim/instructions/tcgen05.cpp (commits df6dde7 + tcgen05 cleanup)
// but are NOT registered in the dispatch table.
//
// S_TCGEN05_* StatementType enums are intentionally excluded from the
// X-Macro loop in ptx_op.def (see ptx_op.def:129-136 explanatory comment
// and ptx_types.h:23-27 inline comment). InstructionFactory::initialize()
// only registers handlers from ptx_op.def, so get_handler(S_TCGEN05_*)
// returns nullptr, and ThreadContext::execute_thread_instruction() falls
// through to the "No handler found" path at thread_context.cpp:142-146,
// which sets state = EXIT.
//
// DEAD-CODE-NOTICE: This header exists solely to support dead-code
// coverage tests until the dispatch issue is resolved in the separate
// `fix-tcgen05-handler-dispatch` change. See:
//   - openspec/changes/fix-tcgen05-test-coverage-gaps/design.md (D4)
//   - ADR-0016 §2026-07-04 (deferred dispatch wiring)
//   - openspec/changes/fix-tcgen05-handler-dispatch/proposal.md
//
// =============================================================================

#include "ptx_ir/statement_context.h" // Tcgen05Instr + Tcgen05OpKind
#include "ptxsim/thread_context.h"

namespace ptxsim {

// Forward declarations only — definitions live in
// src/ptxsim/instructions/tcgen05.cpp
void processTcgen05Mma(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);
void processTcgen05Ld(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);
void processTcgen05St(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);
void processTcgen05Commit(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);
void processTcgen05Wait(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);

// Phase 1 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q1-A/Q2-A):
// 3 alloc-family handlers (alloc/dealloc/relinquish_alloc_permit).
// Definitions live in src/ptxsim/instructions/tcgen05_alloc.cpp.
void processTcgen05Alloc(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);
void processTcgen05Dealloc(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);
void processTcgen05Relinquish(ThreadContext *context,
                              const ptxemu::ir::Tcgen05Instr &instr);

// Phase 2 (Oracle Q4-B/Q2-A): tcgen05.cp — shared memory → TMEM copy.
// Definition lives in src/ptxsim/instructions/tcgen05_cp.cpp.
void processTcgen05Cp(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);

// Phase 4 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q6-B):
// tcgen05.fence no-op marker (design D8). Definition lives in
// src/ptxsim/instructions/tcgen05_fence.cpp.
void processTcgen05Fence(ThreadContext *context, const ptxemu::ir::Tcgen05Instr &instr);

// Phase 2 helpers for tcgen05.cp. Exposed for unit tests; the public
// entry point remains `processTcgen05Cp`.
[[noreturn]] void throw_cta_group_2(const char *instr_name);
uint32_t extract_smem_offset_placeholder(const ptxemu::ir::Tcgen05Instr &instr);

} // namespace ptxsim