#include "ptxsim/sm_context.h"
#include "ptxsim/cta_context.h"
#include <algorithm>
#include <cassert>

SMContext::SMContext(int max_warps, int max_threads_per_sm, size_t shared_mem_size)
    : max_warps_per_sm(max_warps), max_threads_per_sm(max_threads_per_sm), 
      max_shared_mem(shared_mem_size), allocated_shared_mem(0), 
      current_thread_count(0), sm_state(RUN) {
    // 初始化warp调度器为轮询调度器
    warp_scheduler = std::make_unique<RoundRobinWarpScheduler>();
}

void SMContext::init(Dim3& gridDim, Dim3& blockDim, 
                     std::vector<StatementContext>& statements,
                     std::map<std::string, PtxInterpreter::Symtable*>& name2Sym,
                     std::map<std::string, int> &label2pc) {
    this->gridDim = gridDim;
}

bool SMContext::add_block(CTAContext* block) {
    // 检查资源是否足够
    int block_threads = block->BlockDim.x * block->BlockDim.y * block->BlockDim.z;
    
    if (current_thread_count + block_threads > max_threads_per_sm) {
        return false;  // 超出线程数限制
    }
    
    if (static_cast<int>(warps.size()) + block->warpNum > max_warps_per_sm) {
        return false;  // 超出warp数限制
    }
    
    // 尝试分配共享内存
    if (!allocate_shared_memory(block)) {
        return false;  // 共享内存不足
    }
    
    // 获取CTAContext中的warp所有权
    auto block_warps = block->release_warps();
    
    // 将warp添加到SM并注册到调度器
    for (auto& block_warp : block_warps) {
        if (block_warp) {  // 确保warp不为空
            warps.push_back(std::move(block_warp));
            warp_scheduler->add_warp(warps.back().get());
        }
    }
    
    // 将块添加到管理列表中（转移所有权）
    std::unique_ptr<CTAContext> managed_block(block);
    managed_blocks.push_back(std::move(managed_block));
    
    current_thread_count += block_threads;
    
    return true;
}

EXE_STATE SMContext::exe_once() {
    if (sm_state != RUN) {
        return sm_state;
    }
    
    // 检查是否所有warp都已完成
    if (warp_scheduler->all_warps_finished()) {
        sm_state = EXIT;
        return sm_state;
    }
    
    // 调度下一个warp执行
    WarpContext* next_warp = warp_scheduler->schedule_next();
    if (next_warp) {
        // 获取当前warp中第一个活跃线程的PC作为指令来源
        ThreadContext* firstActiveThread = nullptr;
        StatementContext* currentStmt = nullptr;
        
        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext* thread = next_warp->get_thread(lane);
            if (thread && thread->is_active() && !thread->is_exited()) {
                firstActiveThread = thread;
                // 使用安全的PC检查
                if (thread->is_valid_pc()) {
                    currentStmt = thread->get_current_statement();
                    break; // 找到指令后跳出
                }
            }
        }
        
        if (currentStmt) {
            // 执行warp指令
            next_warp->execute_warp_instruction(*currentStmt);
        }
    }
    
    // 检查同步操作 - 现在我们直接检查warp中的线程状态
    // 如果所有线程都在barrier状态，则释放所有barrier状态的线程
    bool all_at_barrier = true;
    for (const auto& warp : warps) {
        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext* thread = warp->get_thread(lane);
            if (thread) {
                // 在检查线程状态前，先确保线程的PC是有效的
                // 如果线程已退出，则不需要检查其他状态
                if (!thread->is_at_barrier() && !thread->is_exited()) {
                    all_at_barrier = false;
                    break;
                }
            }
        }
        if (!all_at_barrier) break;
    }
    
    if (all_at_barrier) {
        // 所有线程都在屏障处等待，释放屏障
        for (auto& warp : warps) {
            for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
                ThreadContext* thread = warp->get_thread(lane);
                if (thread && thread->is_at_barrier()) {
                    thread->set_state(RUN);
                }
            }
        }
    }
    
    // 更新状态
    update_state();
    
    return sm_state;
}

bool SMContext::is_idle() const {
    return warp_scheduler->all_warps_finished();
}

int SMContext::get_active_warps_count() const {
    int count = 0;
    for (const auto& warp : warps) {
        if (warp && warp->is_active()) {
            count++;
        }
    }
    return count;
}

int SMContext::get_active_threads_count() const {
    int count = 0;
    for (const auto& warp : warps) {
        if (warp) {
            count += warp->get_active_count();
        }
    }
    return count;
}

void SMContext::set_warp_scheduler(std::unique_ptr<WarpScheduler> scheduler) {
    warp_scheduler = std::move(scheduler);
}

void SMContext::update_state() {
    // 更新warp调度器状态
    warp_scheduler->update_state();
    
    // 检查整体SM状态
    bool has_active_warps = false;
    for (const auto& warp : warps) {
        if (warp && !warp->is_finished()) {
            has_active_warps = true;
            break;
        }
    }
    
    if (!has_active_warps) {
        sm_state = EXIT;
    } else {
        sm_state = RUN;
    }
}

bool SMContext::allocate_shared_memory(CTAContext* block) {
    // 计算block所需的共享内存大小
    size_t required_shared_mem = block->sharedMemBytes;
    
    if (allocated_shared_mem + required_shared_mem > max_shared_mem) {
        return false;  // 共享内存不足
    }
    
    allocated_shared_mem += required_shared_mem;
    return true;
}

void SMContext::free_shared_memory(CTAContext* block) {
    // 释放共享内存
    size_t freed_mem = block->sharedMemBytes;
    if (allocated_shared_mem >= freed_mem) {
        allocated_shared_mem -= freed_mem;
    }
}