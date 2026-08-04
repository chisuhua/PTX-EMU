#ifndef PTXSM_SIM_BLOCK_DISPATCH_H
#define PTXSM_SIM_BLOCK_DISPATCH_H

#include <cstddef>
#include <memory>

class SMContext;
class CTAContext;

// CTA admission / pending-queue / resource-release helpers extracted from
// SMContext (god-class-refactor-sm-context C-2 Phase 3). SMContext friends
// this Access class for direct private-member access (the block-dispatch
// members touch ~15 private fields — a friend class is the minimal-complexity
// boundary; see sm_context_reconvergence/sm_context_cpptlm_inject for the
// no-friend alternative used when only public collaborators are needed).
namespace sm_block_dispatch {

class Access {
public:
    static bool add_block(SMContext &ctx, std::unique_ptr<CTAContext> block);
    static void try_admit_pending_blocks(SMContext &ctx);
    static void cleanup_finished_blocks(SMContext &ctx);
    static void free_shared_memory(SMContext &ctx, CTAContext *block);
    static bool reserve_resources(SMContext &ctx, size_t shared_mem_size,
                                  int warp_count);
    static void release_resources(SMContext &ctx, int reservation_id);
};

}  // namespace sm_block_dispatch

#endif
