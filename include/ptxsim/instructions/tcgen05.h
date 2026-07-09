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

#include "ptxsim/thread_context.h"
#include "ptx_ir/statement_context.h"  // Tcgen05Instr + Tcgen05OpKind

namespace ptxsim {

// Forward declarations only — definitions live in src/ptxsim/instructions/tcgen05.cpp
void processTcgen05Mma(ThreadContext* context, const Tcgen05Instr& instr);
void processTcgen05Ld(ThreadContext* context, const Tcgen05Instr& instr);
void processTcgen05St(ThreadContext* context, const Tcgen05Instr& instr);
void processTcgen05Commit(ThreadContext* context, const Tcgen05Instr& instr);
void processTcgen05Wait(ThreadContext* context, const Tcgen05Instr& instr);

// Phase 1 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q1-A/Q2-A):
// 3 alloc-family handlers (alloc/dealloc/relinquish_alloc_permit).
// Definitions live in src/ptxsim/instructions/tcgen05_alloc.cpp.
void processTcgen05Alloc(ThreadContext* context, const Tcgen05Instr& instr);
void processTcgen05Dealloc(ThreadContext* context, const Tcgen05Instr& instr);
void processTcgen05Relinquish(ThreadContext* context, const Tcgen05Instr& instr);

}  // namespace ptxsim