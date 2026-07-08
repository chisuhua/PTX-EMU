#include "ptxsim/instruction_handlers.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "utils/logger.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Declaration handlers (for .reg, .const, etc.)
#define IMPLEMENT_DECLARATION_HANDLER(Name) \
    void Name##Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        DeclarationHandler::ExecPipe(context, stmt); \
    }

// Simple handlers (labels, pragmas, dollar names, membar, fence, etc.)
#define IMPLEMENT_SIMPLE_HANDLER(Name) \
    void Name##Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        SimpleHandler::ExecPipe(context, stmt); \
    }

// Void handlers (ret, exit, trap, etc.)
#define IMPLEMENT_VOID_HANDLER(Name) \
    void Name##Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        VoidHandler::ExecPipe(context, stmt); \
    } \
    __attribute__((weak)) void Name##Handler::processOperation(ThreadContext *context, StatementContext &stmt) { \
        /* Default implementation does nothing */ \
        (void)context; \
        (void)stmt; \
    };

// Branch handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_BRANCH_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::executeBranch(ThreadContext *context, const BranchInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    };

// Barrier handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_BARRIER_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::executeBarrier(ThreadContext *context, const BarrierInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    };

// Call handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_CALL_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::executeCall(ThreadContext *context, const CallInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    }; \
    __attribute__((weak)) void Name##Handler::handlePrintf(ThreadContext *context, const CallInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    }; \
    __attribute__((weak)) void Name##Handler::parseAndPrintFormat(ThreadContext *context, const std::string &format, \
                                           const std::vector<void *> &args) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)format; \
        (void)args; \
        return; \
    };

// Generic instruction handlers (add, ld, st, mov, etc.)
// These are implemented in separate .cpp files
#define IMPLEMENT_GENERIC_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::processOperation(ThreadContext *context, void **operands, \
                                        const std::vector<Qualifier> &qualifiers, \
                                        const std::vector<char> *operand_is_immediate) { \
        /* Implementation is in separate .cpp file - check if comparison.cpp etc. is linked */ \
        (void)context; \
        (void)operands; \
        (void)qualifiers; \
        (void)operand_is_immediate; \
    };

// Atomic instruction handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_ATOM_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::processAtomicOperation(ThreadContext *context, void **operands, \
                                              const std::vector<Qualifier> &qualifiers, \
                                              const std::vector<char> *operand_is_immediate) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)operands; \
        (void)qualifiers; \
        (void)operand_is_immediate; \
        return; \
    };

// WMMA instruction handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_WMMA_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::processWmmaOperation(ThreadContext *context, void **operands, \
                                            const std::vector<Qualifier> &qualifiers) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)operands; \
        (void)qualifiers; \
        return; \
    };

// CP_ASYNC handler (currently treated as simple, but can be extended)
#define IMPLEMENT_CP_ASYNC_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::executeAsyncCopy(ThreadContext *context, const CpAsyncInstr &instr) { \
        PTX_DEBUG_EMU("Enqueuing async copy: dst=%p, src=%p, size=%d", \
                      instr.operands[0].operand_phy_addr, \
                      instr.operands[1].operand_phy_addr, \
                      *(int*)instr.operands[2].operand_phy_addr); \
        /* TODO: integrate with async copy engine */ \
        return; \
    };

// All other instruction types map to SimpleHandler
// T1-4: membar handler — no-op + logging in single-threaded SC memory model.
// Per docs/superpowers/plans/2026-06-22-phase2-critical-debt.md Task 4: membar's
// memory barrier semantics are implicit in PC advancement; no barrier API needed.
#define IMPLEMENT_MEMBAR_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        PTX_DEBUG_EMU("membar handler: no-op in single-threaded SC model"); \
        SimpleHandler::ExecPipe(context, stmt); \
    }

// WARP_BARRIER handlers - use GenericPipelineHandler pattern
#define IMPLEMENT_WARP_BARRIER_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::processOperation(ThreadContext *context, void **operands, \
                                        const std::vector<Qualifier> &qualifiers, \
                                        const std::vector<char> *operand_is_immediate) { \
        /* Implementation is in src/ptxsim/instructions/barrier.cpp */ \
        (void)context; \
        (void)operands; \
        (void)qualifiers; \
        (void)operand_is_immediate; \
        return; \
    };

#define IMPLEMENT_FENCE_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        PTX_DEBUG_EMU("fence handler: no-op in single-threaded SC model"); \
        SimpleHandler::ExecPipe(context, stmt); \
    }
#define IMPLEMENT_REDUX_INSTR_HANDLER(Name)      IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_MBARRIER_INSTR_HANDLER(Name)   IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_PREDICATE_PREFIX_HANDLER(Name) IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_VOTE_INSTR_HANDLER(Name)       IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_SHFL_INSTR_HANDLER(Name)       IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_TEXTURE_INSTR_HANDLER(Name)    IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_SURFACE_INSTR_HANDLER(Name)    IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_REDUCTION_INSTR_HANDLER(Name)  IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_PREFETCH_INSTR_HANDLER(Name)   IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_ASYNC_STORE_HANDLER(Name)      IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_ASYNC_REDUCE_HANDLER(Name)     IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_TCGEN_INSTR_HANDLER(Name)      IMPLEMENT_SIMPLE_HANDLER(Name)
// Tcgen05: 11 S_TCGEN05_* enums all dispatch to Tcgen05Handler
// (processTcgen05Operation is virtual, dispatches on instr.op_kind).
// The strong definition lives in src/ptxsim/instructions/tcgen05.cpp;
// here we provide a weak stub that throws UnsupportedInstructionException
// for any tcgen05.* that isn't yet implemented (per ADR-0016 Deferred-but-Wired).
// The X-Macro expansion below is a no-op (single Tcgen05Handler class,
// already declared in instruction_handlers.h).
#define IMPLEMENT_TCGEN05_INSTR_HANDLER(Name)  /* no-op: see tcgen05.cpp */
#define IMPLEMENT_TENSORMAP_INSTR_HANDLER(Name)  IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_ABI_DIRECTIVE_HANDLER(Name)    IMPLEMENT_SIMPLE_HANDLER(Name)

// Add implementations for operand handlers (they are declaration handlers)
#define IMPLEMENT_OPERAND_REG_HANDLER(Name)     IMPLEMENT_DECLARATION_HANDLER(Name)
#define IMPLEMENT_OPERAND_CONST_HANDLER(Name)   IMPLEMENT_DECLARATION_HANDLER(Name)
#define IMPLEMENT_OPERAND_MEMORY_HANDLER(Name)  IMPLEMENT_DECLARATION_HANDLER(Name)
#define IMPLEMENT_SIMPLE_NAME_HANDLER(Name)     IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_LABEL_INSTR_HANDLER(Name)     IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_SIMPLE_STRING_HANDLER(Name)   IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_VOID_INSTR_HANDLER(Name)      IMPLEMENT_VOID_HANDLER(Name)

// Generate all handler implementations from ptx_op.def
// TCGEN05_INSTR is skipped — the single Tcgen05Handler class is defined
// in instruction_handlers.h and its processTcgen05Operation is implemented
// in tcgen05.cpp (X-Macro expansion would re-define the function 11 times).
#undef X
#define X(enum_val, op_name, op_str, op_count, struct_kind, instr_kind) \
    IMPLEMENT_##struct_kind##_HANDLER(op_str)
#include "ptx_ir/ptx_op.def"
#undef X

// Tcgen05 handler implementation: single class, dispatches by op_kind.
// The strong implementation lives in src/ptxsim/instructions/tcgen05.cpp
// (and is compiled into libptxsim via libptxsim_instrs). Here we provide
// a weak stub that throws UnsupportedInstructionException for op_kinds
// that are not yet implemented in tcgen05.cpp (per ADR-0016 §C5 fix #1).
__attribute__((weak)) void Tcgen05Handler::processTcgen05Operation(
    ThreadContext *context, void **operands,
    const std::vector<Qualifier> &qualifiers, const Tcgen05Instr &instr) {
    (void)context; (void)operands; (void)qualifiers; (void)instr;
    throw UnsupportedInstructionException(
        "tcgen05.*", "stub: real implementation in tcgen05.cpp");
}
