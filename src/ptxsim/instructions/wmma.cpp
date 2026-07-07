#include "memory/hardware_memory_manager.h"
#include "memory/hardware_memory_manager.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/async/tc_queue.h"
#include "ptxsim/utils/half_utils.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "utils/logger.h"
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

void WmmaHandler::processWmmaOperation(ThreadContext *context, void **operands,
                                        const std::vector<Qualifier> &qualifiers) {
    (void)context;
    (void)operands;
    (void)qualifiers;

    PTX_ERROR_EMU("WMMA / Tensor Core instruction not implemented "
                  "(qualifiers=%zu) - see implement-wmma-tensor-core",
                  qualifiers.size());
    throw UnsupportedInstructionException(
        "wmma.*",
        "Tensor Core not yet implemented in ptx-emu (ref: c5 Fix #1)");
}