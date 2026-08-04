#include "ptxsim/core/sm_warp_lifecycle.h"

#include "ptxsim/core/sm_block_dispatch.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_scheduler.h"
#include "memory/shared_memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// sm_warp_lifecycle::Access — warp registration / retirement / active-count
// helpers extracted from SMContext (god-class-refactor-sm-context C-2
// Phase 4). SMContext friend-declares this Access class for direct
// private-member access. Bodies are line-level diffs of the originals from
// sm_context.cpp:441-449, 451-459, 465-505, 658-682, 683-689
// (lessons-learned §1).

namespace sm_warp_lifecycle {

void Access::update_state(SMContext &ctx) {
    // 更新warp调度器状态
    ctx.warp_scheduler->update_state();

    // 检查整体SM状态
    bool has_active_warps = false;
    auto it = ctx.warps.begin();
    while (it != ctx.warps.end()) {
        auto warp = it->get();
        if (warp && !warp->is_finished()) {
            has_active_warps = true;
            it++;
        } else {
            // 从warp调度器中移除warp
            ctx.warp_scheduler->remove_warp(warp);

            auto physical_block_id = warp->get_physical_block_id();
            ctx.physical_block_warp_counts[physical_block_id]--;
            it = ctx.warps.erase(it);
        }
    }

    // 清理已完成的blocks（释放共享内存）
    sm_block_dispatch::Access::cleanup_finished_blocks(ctx);

    // 检查是否有正在管理的blocks
    bool has_managed_blocks = !ctx.managed_blocks.empty();

    if (!has_active_warps && !has_managed_blocks) {
        ctx.sm_state = EXIT;
    } else {
        ctx.sm_state = RUN;
    }

    // 更新统计信息
    ctx.stats_.active_warps = ctx.warps.size();
    ctx.stats_.active_threads = Access::get_active_threads_count(ctx);
    if (ctx.shared_mem_manager_) {
        ctx.stats_.allocated_shared_mem =
            ctx.shared_mem_manager_->get_allocated_size();
    }
}

int Access::select_next_group(SMContext &ctx,
                              const std::vector<int> &active_lanes) {
    // With multiple active paths, select based on divergence mode
    if (active_lanes.size() <= 1) {
        return 0; // No divergence, use first group
    }

    switch (ctx.divergence_mode_) {
    case ptxsim::DivergenceExecutionMode::Sequential:
        // Execute groups in order - just return first for now
        return 0;

    case ptxsim::DivergenceExecutionMode::Interleaved:
        // Use round-robin or similar to switch dynamically
        return 0; // Could implement round-robin counter per warp

    case ptxsim::DivergenceExecutionMode::ShortestFirst:
        // Estimate path length and execute shortest first
        // For now, fall through to sequential
        return 0;

    default:
        return 0;
    }
}

void Access::suspend_and_switch(SMContext &ctx, int current_group,
                                int next_group) {
    // Suspend current group and switch to next_group
    // This is a placeholder for future blocking implementation (Phase 3)
    // For now, we just proceed with the next group selection
    PTX_DEBUG_EMU("SM %d: Suspend group %d, switch to group %d", ctx.sm_id_,
                  current_group, next_group);
}

int Access::get_active_warps_count(const SMContext &ctx) {
    int count = 0;
    for (const auto &warp : ctx.warps) {
        if (warp && warp->is_active()) {
            count++;
        }
    }
    return count;
}

int Access::get_active_threads_count(const SMContext &ctx) {
    int count = 0;
    for (const auto &warp : ctx.warps) {
        if (warp) {
            count += warp->get_active_count();
        }
    }
    return count;
}

}  // namespace sm_warp_lifecycle
