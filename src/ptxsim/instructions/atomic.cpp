#include "memory/hardware_memory_manager.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "ptx_ir/ptx_types.h"
#include <cstring>
#include <cstdint>

void AtomHandler::processAtomicOperation(ThreadContext *context, void **operands,
                                 const std::vector<Qualifier> &qualifiers,
                                 const std::vector<char> *operand_is_immediate) {
    // Operands (collected in ptx_visitor_atom.cpp after Task 8 fix):
    //   operands[0] = dst register address (write old value here)
    //   operands[1] = memory address (host pointer to atomic location)
    //   operands[2] = src value address (register or immediate buffer)
    void *dst = operands[0];
    void *addr = operands[1];
    void *src = operands[2];

    if (!dst || !addr || !src) {
        // Silent no-op matches StHandler::processOperation contract.
        return;
    }

    // Data size and memory space from qualifiers (e.g., .u32, .global)
    size_t data_size = getBytes(qualifiers);
    if (data_size == 0) {
        // No data type qualifier found → cannot infer size; bail safely.
        return;
    }
    MemorySpace space = getAddressSpace(qualifiers);

    // Identify atomic opcode (one of Q_{ADD,AND,OR,XOR,INC,DEC,EXCH,MIN,MAX,CAS}_ATOM)
    Qualifier atom_op = Qualifier::Q_UNKNOWN;
    for (auto q : qualifiers) {
        switch (q) {
        case Qualifier::Q_ADD_ATOM:
        case Qualifier::Q_AND_ATOM:
        case Qualifier::Q_OR_ATOM:
        case Qualifier::Q_XOR_ATOM:
        case Qualifier::Q_INC_ATOM:
        case Qualifier::Q_DEC_ATOM:
        case Qualifier::Q_EXCH_ATOM:
        case Qualifier::Q_MIN_ATOM:
        case Qualifier::Q_MAX_ATOM:
            atom_op = q;
            break;
        default:
            continue;
        }
        if (atom_op != Qualifier::Q_UNKNOWN) break;
    }

    // CAS is out-of-scope (per Must NOT list).
    if (atom_op == Qualifier::Q_UNKNOWN) {
        return;
    }

    // Read-modify-write. The PTX-EMU execution model serializes warp-level
    // dispatch (see src/ptxsim/core/sm_context.cpp:225-260), so a plain
    // load → compute → store sequence produces correct results without a
    // hardware-style CAS loop. Real atomicity (relaxed ordering only, per
    // Must NOT list) is not modeled.
    uint64_t old_val = 0;
    HardwareMemoryManager::instance().access(addr, &old_val, data_size,
                                             /*is_write=*/false, space);

    uint64_t src_val = 0;
    std::memcpy(&src_val, src, data_size);

    uint64_t new_val = old_val;
    switch (atom_op) {
    case Qualifier::Q_ADD_ATOM:
        new_val = old_val + src_val;
        break;
    case Qualifier::Q_AND_ATOM:
        new_val = old_val & src_val;
        break;
    case Qualifier::Q_OR_ATOM:
        new_val = old_val | src_val;
        break;
    case Qualifier::Q_XOR_ATOM:
        new_val = old_val ^ src_val;
        break;
    case Qualifier::Q_EXCH_ATOM:
        new_val = src_val;
        break;
    case Qualifier::Q_MIN_ATOM:
        new_val = (old_val < src_val) ? old_val : src_val;
        break;
    case Qualifier::Q_MAX_ATOM:
        new_val = (old_val > src_val) ? old_val : src_val;
        break;
    case Qualifier::Q_INC_ATOM:
        // Wrap to 0 when old_val >= src_val (matches PTX ISA 8.7.4 atom.inc)
        new_val = (old_val >= src_val) ? 0u : (old_val + 1u);
        break;
    case Qualifier::Q_DEC_ATOM:
        // Clamp at src_val when old_val == 0 or old_val > src_val
        new_val = (old_val == 0 || old_val > src_val) ? src_val : (old_val - 1u);
        break;
    default:
        // Should not reach here (UNKNOWN handled above).
        new_val = old_val;
        break;
    }

    HardwareMemoryManager::instance().access(addr, &new_val, data_size,
                                             /*is_write=*/true, space);

    // PTX semantics: dst receives the OLD value at the memory location.
    // AtomicPipelineHandler::commitResults() will publish dst into the
    // register bank; we just write the value here.
    std::memcpy(dst, &old_val, data_size);
}