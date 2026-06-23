#ifndef PTXSIM_CONTEXTS_MEMORY_REF_H
#define PTXSIM_CONTEXTS_MEMORY_REF_H

class WarpContext;
class CTAContext;

namespace ptxsim {
namespace contexts {

/**
 * @brief Memory reference POD: per-thread pointers to shared/local memory
 *        and to the parent contexts (warp/CTA).
 *
 * @details Groups the per-thread memory spaces (shared / local) and the
 *          back-pointers to the parent contexts (warp / CTA) needed for
 *          memory operations and CTA-level coordination. Pure data —
 *          no methods.
 *
 * @author PTX-EMU Team (T2-3 god-class split)
 * @date 2026-06-24
 */
struct MemoryPod {
    // Shared memory base address (per-CTA, but each thread holds a pointer)
    void *shared_mem_space = nullptr;

    // Local memory base address (per-thread)
    void *local_mem_space = nullptr;

    // Back-pointers to parent contexts (set during init / add_thread)
    WarpContext *warp_context_ = nullptr;
    CTAContext *cta_context_ = nullptr;
};

}  // namespace contexts
}  // namespace ptxsim

#endif  // PTXSIM_CONTEXTS_MEMORY_REF_H