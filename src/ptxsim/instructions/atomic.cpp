#include "memory/hardware_memory_manager.h"
#include "ptxsim/atomic/atomic_mutex.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "ptx_ir/ptx_types.h"
#include <cstring>
#include <cstdint>

void AtomHandler::processAtomicOperation(ThreadContext *context, void **operands,
                                 const std::vector<ptxemu::ir::Qualifier> &qualifiers,
                                 const std::vector<char> *operand_is_immediate) {
    // Operands (collected in ptx_visitor_atom.cpp):
    //   operands[0] = dst register address (write old value here)
    //   operands[1] = memory address (host pointer to atomic location)
    //   operands[2] = src value address (register or immediate buffer)
    //                 OR for CAS: compare value (first non-dst RuleContext operand)
    //   operands[3] = for CAS only: val (second non-dst RuleContext operand)
    void *dst = operands[0];
    void *addr = operands[1];

    if (!dst || !addr) {
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
    ptxemu::ir::Qualifier atom_op = ptxemu::ir::Qualifier::Q_UNKNOWN;
    for (auto q : qualifiers) {
        switch (q) {
        case ptxemu::ir::Qualifier::Q_ADD_ATOM:
        case ptxemu::ir::Qualifier::Q_AND_ATOM:
        case ptxemu::ir::Qualifier::Q_OR_ATOM:
        case ptxemu::ir::Qualifier::Q_XOR_ATOM:
        case ptxemu::ir::Qualifier::Q_INC_ATOM:
        case ptxemu::ir::Qualifier::Q_DEC_ATOM:
        case ptxemu::ir::Qualifier::Q_EXCH_ATOM:
        case ptxemu::ir::Qualifier::Q_MIN_ATOM:
        case ptxemu::ir::Qualifier::Q_MAX_ATOM:
        case ptxemu::ir::Qualifier::Q_CAS_ATOM:
            atom_op = q;
            break;
        default:
            continue;
        }
        if (atom_op != ptxemu::ir::Qualifier::Q_UNKNOWN) break;
    }

    if (atom_op == ptxemu::ir::Qualifier::Q_UNKNOWN) {
        return;
    }

    // Cross-warp atomicity (Phase 2 of implement-atomic-cas-and-true-atomicity):
    // acquire the global atomic mutex around the read-modify-write sequence.
    // Lock-order proof vs other ptxsim mutexes (audit §MR-5):
    //   - HardwareMemoryManager::mutex_ is acquired INSIDE .access() which is
    //     called from this scope; this mutex is the OUTER lock for that call.
    //   - CTABarrier::mutex_ is acquired only by barrier handlers, which do
    //     not invoke atomic handlers under that lock; the two mutexes are
    //     therefore never held simultaneously.
    //   - No public method on AtomicLockGuard re-locks the same mutex
    //     (std::mutex is non-recursive), matching the cta_barrier.cpp:47
    //     pattern documented in ptx-lessons-learned §2.
    ptxsim::AtomicLockGuard atomic_guard(ptxsim::global_atomic_mutex());

    // CAS path: read-modify-compare-write. PTX-EMU's warp-level scheduler
    // (sm_context.cpp:225-260) serializes per-warp dispatch; the cross-warp
    // mutex above bridges concurrent warps.
    if (atom_op == ptxemu::ir::Qualifier::Q_CAS_ATOM) {
        void *cmp_buf = operands[2];
        void *val_buf = operands[3];
        if (!cmp_buf || !val_buf) {
            return;
        }

        uint64_t old_val = 0;
        HardwareMemoryManager::instance().access(addr, &old_val, data_size,
                                                 /*is_write=*/false, space);

        uint64_t cmp_val = 0;
        uint64_t new_val = 0;
        std::memcpy(&cmp_val, cmp_buf, data_size);
        std::memcpy(&new_val, val_buf, data_size);

        // Compare-and-swap: write new_val only when loaded value equals cmp_val.
        if (old_val == cmp_val) {
            HardwareMemoryManager::instance().access(addr, &new_val, data_size,
                                                     /*is_write=*/true, space);
        }

        // PTX ISA semantics: dst always receives the originally loaded value.
        std::memcpy(dst, &old_val, data_size);
        return;
    }

    // Non-CAS read-modify-write path (add, and, or, xor, exch, min, max, inc, dec).
    void *src = operands[2];
    if (!src) {
        return;
    }

    uint64_t old_val = 0;
    HardwareMemoryManager::instance().access(addr, &old_val, data_size,
                                             /*is_write=*/false, space);

    uint64_t src_val = 0;
    std::memcpy(&src_val, src, data_size);

    uint64_t new_val = old_val;
    switch (atom_op) {
    case ptxemu::ir::Qualifier::Q_ADD_ATOM:
        new_val = old_val + src_val;
        break;
    case ptxemu::ir::Qualifier::Q_AND_ATOM:
        new_val = old_val & src_val;
        break;
    case ptxemu::ir::Qualifier::Q_OR_ATOM:
        new_val = old_val | src_val;
        break;
    case ptxemu::ir::Qualifier::Q_XOR_ATOM:
        new_val = old_val ^ src_val;
        break;
    case ptxemu::ir::Qualifier::Q_EXCH_ATOM:
        new_val = src_val;
        break;
    case ptxemu::ir::Qualifier::Q_MIN_ATOM:
        new_val = (old_val < src_val) ? old_val : src_val;
        break;
    case ptxemu::ir::Qualifier::Q_MAX_ATOM:
        new_val = (old_val > src_val) ? old_val : src_val;
        break;
    case ptxemu::ir::Qualifier::Q_INC_ATOM:
        // Wrap to 0 when old_val >= src_val (matches PTX ISA 8.7.4 atom.inc)
        new_val = (old_val >= src_val) ? 0u : (old_val + 1u);
        break;
    case ptxemu::ir::Qualifier::Q_DEC_ATOM:
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