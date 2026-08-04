#ifndef PTXSM_SIM_WARP_LIFECYCLE_H
#define PTXSM_SIM_WARP_LIFECYCLE_H

#include <vector>

class SMContext;

// Warp registration / retirement / active-count helpers extracted from
// SMContext (god-class-refactor-sm-context C-2 Phase 4). SMContext friends
// this Access class for direct private-member access.
namespace sm_warp_lifecycle {

class Access {
public:
    static void update_state(SMContext &ctx);
    static int select_next_group(SMContext &ctx,
                                 const std::vector<int> &active_lanes);
    static void suspend_and_switch(SMContext &ctx, int current_group,
                                   int next_group);
    static int get_active_warps_count(const SMContext &ctx);
    static int get_active_threads_count(const SMContext &ctx);
};

}  // namespace sm_warp_lifecycle

#endif
