#include "ptxsim/instruction_handlers.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
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
    void Name##_Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        DeclarationHandler::ExecPipe(context, stmt); \
    }

// Simple handlers (labels, pragmas, dollar names, membar, fence, etc.)
#define IMPLEMENT_SIMPLE_HANDLER(Name) \
    void Name##_Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        SimpleHandler::ExecPipe(context, stmt); \
    }

// Void handlers (ret, exit, trap, etc.)
#define IMPLEMENT_VOID_HANDLER(Name) \
    void Name##_Handler::ExecPipe(ThreadContext *context, StatementContext &stmt) { \
        VoidHandler::ExecPipe(context, stmt); \
    } \
    __attribute__((weak)) void Name##_Handler::processOperation(ThreadContext *context, StatementContext &stmt) { \
        /* Default implementation does nothing */ \
        (void)context; \
        (void)stmt; \
    };

// Branch handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_BRANCH_HANDLER(Name) \
    __attribute__((weak)) void Name##_Handler::executeBranch(ThreadContext *context, const BranchInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    };

// Barrier handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_BARRIER_HANDLER(Name) \
    __attribute__((weak)) void Name##_Handler::executeBarrier(ThreadContext *context, const BarrierInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    };

// Call handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_CALL_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##_Handler::executeCall(ThreadContext *context, const CallInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    }; \
    __attribute__((weak)) void Name##_Handler::handlePrintf(ThreadContext *context, const CallInstr &instr) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)instr; \
        return; \
    }; \
    __attribute__((weak)) void Name##_Handler::parseAndPrintFormat(ThreadContext *context, const std::string &format, \
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
    __attribute__((weak)) void Name##_Handler::processOperation(ThreadContext *context, void **operands, \
                                        const std::vector<Qualifier> &qualifiers) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)operands; \
        (void)qualifiers; \
        return; \
    };

// Atomic instruction handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_ATOM_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##_Handler::processAtomicOperation(ThreadContext *context, void **operands, \
                                              const std::vector<Qualifier> &qualifiers) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)operands; \
        (void)qualifiers; \
        return; \
    };

// WMMA instruction handlers
// These are implemented in separate .cpp files
#define IMPLEMENT_WMMA_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##_Handler::processWmmaOperation(ThreadContext *context, void **operands, \
                                            const std::vector<Qualifier> &qualifiers) { \
        /* Implementation is in separate .cpp file */ \
        (void)context; \
        (void)operands; \
        (void)qualifiers; \
        return; \
    };

// CP_ASYNC handler (currently treated as simple, but can be extended)
#define IMPLEMENT_CP_ASYNC_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##_Handler::executeAsyncCopy(ThreadContext *context, const CpAsyncInstr &instr) { \
        PTX_DEBUG_EMU("Enqueuing async copy: dst=%p, src=%p, size=%d", \
                      instr.operands[0].operand_phy_addr, \
                      instr.operands[1].operand_phy_addr, \
                      *(int*)instr.operands[2].operand_phy_addr); \
        /* TODO: integrate with async copy engine */ \
        return; \
    };

// All other instruction types map to SimpleHandler
#define IMPLEMENT_MEMBAR_INSTR_HANDLER(Name)     IMPLEMENT_SIMPLE_HANDLER(Name)
#define IMPLEMENT_FENCE_INSTR_HANDLER(Name)      IMPLEMENT_SIMPLE_HANDLER(Name)
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
#undef X
#define X(enum_val, type_name, str, op_count, struct_kind) \
    IMPLEMENT_##struct_kind##_HANDLER(type_name)
#include "ptx_ir/ptx_op.def"
#undef X
