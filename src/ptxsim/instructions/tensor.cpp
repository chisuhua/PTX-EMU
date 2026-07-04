#include "memory/hardware_memory_manager.h" // 确保包含 MemoryManager
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "utils/logger.h"
#include <cmath>

// WMMA / Tensor Core stub. Throws UnsupportedInstructionException
// (error_code auto-set to UNSUPPORTED_INSTRUCTION) instead of silently
// no-op'ing, so dst register is not corrupted with uninitialized data.
// Real implementation tracked by `implement-wmma-tensor-core` change.
// Message prefix "wmma." required by stub-explicit-failure spec for log
// filtering. Do not "simplify" back to a silent no-op.
void WmmaHandler::processWmmaOperation(ThreadContext *context, void **operands,
                                        const std::vector<Qualifier> &qualifiers) {
    (void)context;
    (void)operands;
    PTX_ERROR_EMU("WMMA / Tensor Core instruction not implemented "
                  "(qualifiers=%zu) - see implement-wmma-tensor-core",
                  qualifiers.size());
    throw UnsupportedInstructionException(
        "wmma.*",
        "Tensor Core not yet implemented in ptx-emu (ref: c5 Fix #1)");
}
