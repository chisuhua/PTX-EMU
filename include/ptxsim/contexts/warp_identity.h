#ifndef PTXSIM_CONTEXTS_WARP_IDENTITY_H
#define PTXSIM_CONTEXTS_WARP_IDENTITY_H

namespace ptxsim {
namespace contexts {

/**
 * @brief Warp identity POD: per-warp identifier fields.
 *
 * @details Groups the per-warp identity fields: logical warp_id, physical
 *          hardware warp_id, physical block (CTA) id, and the legacy
 *          warp-level PC (kept for backward compat; the per-thread PC in
 *          WarpState is the authoritative source). Pure data — no methods.
 *
 * @author PTX-EMU Team (T2-3 god-class split)
 * @date 2026-06-24
 */
struct WarpIdentityPod {
    // Logical warp ID (within a CTA: 0..num_warps-1)
    int warp_id = 0;

    // Physical hardware warp ID (across all warps in the SM)
    int physical_warp_id = 0;

    // Physical block (CTA) ID
    int physical_block_id = 0;

    // Warp-level PC (legacy backward-compat field;
    // authoritative source is warp_state.threads[i].pc)
    int pc = 0;
};

}  // namespace contexts
}  // namespace ptxsim

#endif  // PTXSIM_CONTEXTS_WARP_IDENTITY_H