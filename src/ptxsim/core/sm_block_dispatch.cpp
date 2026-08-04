#include "ptxsim/core/sm_block_dispatch.h"

#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_scheduler.h"
#include "memory/shared_memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <vector>

// sm_block_dispatch::Access — CTA admission / pending-queue / resource-release
// helpers extracted from SMContext (god-class-refactor-sm-context C-2 Phase 3).
// SMContext friend-declares this Access class (include/ptxsim/sm_context.h) for
// direct private-member access. Public SMContext methods become one-line
// forwarders to these statics. Bodies are line-level diffs of the originals
// from sm_context.cpp:130-204, 206-258, 628-643, 645-665, 667-689, 691-695
// (lessons-learned §1). Cross-helper calls to other extracted members route
// through these statics recursively; the `update_state()` call inside
// add_block still routes through the public SMContext forwarder (which
// forwards to the real member until Task 3 extracts it).

namespace sm_block_dispatch {

bool Access::add_block(SMContext &ctx, std::unique_ptr<CTAContext> block) {
    // 1. 计算资源需求
    size_t required_shared_mem = block->get_shared_memory_requirement();
    int required_warps = block->get_warp_count();

    // BUG-SM-ADMISSION-OVERFLOW: 拒绝"绝对无法 fit"的块
    // 单 block 所需资源 > SM 总容量 → 直接失败(原语义)
    // 否则进 pending_blocks_,等待资源释放
    if (required_warps > ctx.max_warps_per_sm) {
        PTX_DEBUG_EMU(
            "Block requires %d warps > SM max %d — cannot ever fit, dropping",
            required_warps, ctx.max_warps_per_sm);
        return false;
    }
    if (required_shared_mem > ctx.max_shared_mem) {
        PTX_DEBUG_EMU("Block requires %zu shared mem > SM max %zu — cannot "
                      "ever fit, dropping",
                      required_shared_mem, ctx.max_shared_mem);
        return false;
    }

    // 2. 分配reservation_id并设置到CTAContext
    int reservation_id = ctx.current_reservation_id_++;
    block->set_reservation_id(reservation_id);

    // 3. 检查资源是否足够
    if (!Access::reserve_resources(ctx, required_shared_mem, required_warps)) {
        // BUG-SM-ADMISSION-OVERFLOW fix: 不丢弃,进 pending 队列
        // 资源释放后由 try_admit_pending_blocks() 重新 admit
        PTX_DEBUG_EMU(
            "Block queued in pending (SM full): shared_mem=%zu, warps=%d, "
            "pending_count=%zu",
            required_shared_mem, required_warps, ctx.pending_blocks_.size() + 1);
        ctx.pending_blocks_.push_back(std::move(block));
        return true;
    }

    void *shared_mem_space = nullptr;
    if (required_shared_mem > 0 && ctx.shared_mem_manager_) {
        shared_mem_space = ctx.shared_mem_manager_->allocate(
            required_shared_mem, block->get_reservation_id());
        if (!shared_mem_space) {
            Access::release_resources(ctx, block->get_reservation_id());
            PTX_DEBUG_EMU(
                "Failed to allocate shared memory of size %zu for block %d",
                required_shared_mem, block->get_reservation_id());
            return false;
        }
    }

    block->build_shared_memory_symbol_table(shared_mem_space);
    ctx.allocated_shared_mem += required_shared_mem;

    int physical_block_id = ctx.next_physical_block_id++;
    ctx.physical_block_warp_counts[physical_block_id] = required_warps;
    ctx.managed_blocks.insert({physical_block_id, std::move(block)});

    auto block_warps = ctx.managed_blocks[physical_block_id]->release_warps();
    for (auto &warp : block_warps) {
        warp->set_physical_block_id(physical_block_id);
        warp->set_physical_warp_id(ctx.next_physical_warp_id++);
        warp->set_sm_context(&ctx);
        ctx.warps.push_back(std::move(warp));
        ctx.warp_scheduler->add_warp(ctx.warps.back().get());
    }

    // 更新状态
    ctx.update_state();

    PTX_DEBUG_EMU("Successfully added block with %zu shared memory bytes, "
                  "%d warps to SM %d",
                  required_shared_mem, required_warps, ctx.sm_id_);

    return true;
}

void Access::try_admit_pending_blocks(SMContext &ctx) {
    // FIFO admit:队首 block 资源能 fit 就 admit,继续检查下一个
    // 直到队首 block 资源 fit 失败(说明当前 SM 已满)或队列空
    while (!ctx.pending_blocks_.empty()) {
        auto &front_block = ctx.pending_blocks_.front();
        size_t req_smem = front_block->get_shared_memory_requirement();
        int req_warps = front_block->get_warp_count();

        if (!Access::reserve_resources(ctx, req_smem, req_warps)) {
            // 资源仍不足(被其他 admitted block 占用),停止
            // 下一个 cleanup_finished_blocks() 释放资源后再试
            return;
        }

        // 资源够,admit 这个 block
        std::unique_ptr<CTAContext> block = std::move(ctx.pending_blocks_.front());
        ctx.pending_blocks_.pop_front();

        // 后续逻辑与 add_block 主路径完全相同
        int reservation_id = block->get_reservation_id();
        void *shared_mem_space = nullptr;
        if (req_smem > 0 && ctx.shared_mem_manager_) {
            shared_mem_space =
                ctx.shared_mem_manager_->allocate(req_smem, reservation_id);
            if (!shared_mem_space) {
                Access::release_resources(ctx, reservation_id);
                PTX_DEBUG_EMU(
                    "try_admit_pending: smem alloc failed for pending block");
                continue;
            }
        }

        block->build_shared_memory_symbol_table(shared_mem_space);
        ctx.allocated_shared_mem += req_smem;

        int physical_block_id = ctx.next_physical_block_id++;
        ctx.physical_block_warp_counts[physical_block_id] = req_warps;
        ctx.managed_blocks.insert({physical_block_id, std::move(block)});

        auto block_warps = ctx.managed_blocks[physical_block_id]->release_warps();
        for (auto &warp : block_warps) {
            warp->set_physical_block_id(physical_block_id);
            warp->set_physical_warp_id(ctx.next_physical_warp_id++);
            warp->set_sm_context(&ctx);
            ctx.warps.push_back(std::move(warp));
            ctx.warp_scheduler->add_warp(ctx.warps.back().get());
        }

        PTX_DEBUG_EMU("try_admit_pending: admitted block with %zu smem, "
                      "%d warps; %zu still pending",
                      req_smem, req_warps, ctx.pending_blocks_.size());
    }
}

void Access::cleanup_finished_blocks(SMContext &ctx) {
    auto it = ctx.managed_blocks.begin();
    while (it != ctx.managed_blocks.end()) {
        auto physical_block_id = it->first;
        auto block = it->second.get();
        if (ctx.physical_block_warp_counts[physical_block_id] == 0) {
            Access::free_shared_memory(ctx, it->second.get());
            ctx.physical_block_warp_counts.erase(physical_block_id);
            it = ctx.managed_blocks.erase(it);
        } else {
            ++it;
        }
    }
    // BUG-SM-ADMISSION-OVERFLOW: 资源刚释放,尝试重灌 pending
    Access::try_admit_pending_blocks(ctx);
}

void Access::free_shared_memory(SMContext &ctx, CTAContext *block) {
    // 释放共享内存
    if (block->sharedMemSpace != nullptr && ctx.shared_mem_manager_) {
        size_t shared_mem_size =
            block->get_shared_memory_requirement(); // 获取要释放的内存大小

        ctx.shared_mem_manager_->deallocate(block->sharedMemSpace,
                                            block->get_reservation_id());

        // 更新本地统计 - 减去释放的内存大小
        if (ctx.allocated_shared_mem >= shared_mem_size) {
            ctx.allocated_shared_mem -= shared_mem_size;
        } else {
            // 防止下溢出，理论上不应该发生
            ctx.allocated_shared_mem = 0;
        }

        // 重置block的共享内存指针
        const_cast<void *&>(block->sharedMemSpace) = nullptr;
    }
}

bool Access::reserve_resources(SMContext &ctx, size_t shared_mem_size,
                               int warp_count) {
    if (!ctx.shared_mem_manager_) {
        PTX_DEBUG_EMU("Shared memory manager not initialized");
        return false;
    }

    // 检查共享内存是否足够
    if (ctx.shared_mem_manager_->get_available_size() < shared_mem_size) {
        PTX_DEBUG_EMU(
            "Insufficient shared memory: requested %zu, available %zu",
            shared_mem_size, ctx.shared_mem_manager_->get_available_size());
        return false;
    }

    // 检查warp数量是否足够
    if (static_cast<int>(ctx.warps.size()) + warp_count > ctx.max_warps_per_sm) {
        PTX_DEBUG_EMU("Insufficient warps: current %zu, requested %d, max %d",
                      ctx.warps.size(), warp_count, ctx.max_warps_per_sm);
        return false;
    }

    return true;
}

void Access::release_resources(SMContext &ctx, int reservation_id) {
    // 在实际实现中，这会释放为特定块预留的资源
    // 但现在我们使用共享内存管理器来处理资源释放
    PTX_DEBUG_EMU("Releasing resources for reservation_id %d", reservation_id);
}

}  // namespace sm_block_dispatch
