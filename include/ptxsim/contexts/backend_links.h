#ifndef PTXSIM_CONTEXTS_BACKEND_LINKS_H
#define PTXSIM_CONTEXTS_BACKEND_LINKS_H

#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_context.h"
#include "register/register_bank_manager.h"
#include <memory>
#include <vector>

class SMContext;
class CTAContext;

namespace ptxsim {
namespace contexts {

/**
 * @brief Backend links POD: per-warp references to backend systems
 *        (register bank, parent contexts, threads, SIMT stack).
 *
 * @details Groups the per-warp back-pointers to backend systems: the
 *          shared register-bank manager, the parent SM/CTA contexts, the
 *          owned threads (unique_ptr<ThreadContext>[]), and the SIMT
 *          control-flow stack. Pure data — no methods.
 *
 * @author PTX-EMU Team (T2-3 god-class split)
 * @date 2026-06-24
 */
struct BackendLinksPod {
    // Register bank manager (shared with all threads in the warp)
    std::shared_ptr<RegisterBankManager> register_bank_manager_;

    // Back-pointers to parent contexts
    SMContext *sm_context_ = nullptr;
    CTAContext *cta_context_ = nullptr;

    // Owned threads (unique_ptr ownership = warp owns thread lifetime)
    std::vector<std::unique_ptr<ThreadContext>> threads;

    // SIMT control-flow stack (per-warp)
    ptxsim::SIMTStack simt_stack;

    // Single-step mode flag
    bool single_step_mode = false;
};

}  // namespace contexts
}  // namespace ptxsim

#endif  // PTXSIM_CONTEXTS_BACKEND_LINKS_H