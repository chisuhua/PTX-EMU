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
                     std::map<std::string, int>& label2pc) {
    this->gridDim = gridDim;
}

bool SMContext::add_block(CTAContext* block) {
    // 检查资源是否足够
    int block_threads = block->BlockDim.x * block->BlockDim.y * block->BlockDim.z;
    
    if (current_thread_count + block_threads > max_threads_per_sm) {
        return false;  // 超出线程数限制
    }
    
    // 尝试分配共享内存
    if (!allocate_shared_memory(block)) {
        return false;  // 共享内存不足
    }
    
    // 添加块到SM
    blocks.push_back(block);
    
    // 为块初始化warp（如果block已经包含warp，则直接添加）
    // 如果block的warp尚未初始化，则在这里初始化
    if (block->warps.empty()) {
        init_warps_for_block(block);
    } else {
        // 如果block已经有warp，直接添加到SM的warp列表
        for (auto& warp : block->warps) {
            warps.push_back(std::make_unique<WarpContext>(*warp));  // 复制warp
        }
    }
    
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
                if (thread->get_pc() < thread->statements->size()) {
                    currentStmt = &(*thread->statements)[thread->get_pc()];
                    break; // 找到指令后跳出
                }
            }
        }
        
        if (currentStmt) {
            // 执行warp指令
            next_warp->execute_warp_instruction(*currentStmt);
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

void SMContext::init_warps_for_block(CTAContext* block) {
    // 如果block已经初始化了warp，直接将它们添加到SM
    for (auto& block_warp : block->warps) {
        auto new_warp = std::make_unique<WarpContext>();
        // 复制warp的内容（这是一个简化实现）
        // 在实际实现中，可能需要更复杂的warp复制逻辑
        new_warp->set_warp_id(block_warp->get_warp_id());
        // 复制线程等信息
        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext* thread = block_warp->get_thread(lane);
            new_warp->add_thread(thread, lane);
        }
        warps.push_back(std::move(new_warp));
        warp_scheduler->add_warp(warps.back().get());
    }
}

void SMContext::update_state() {
    // 更新warp调度器状态
    warp_scheduler->update_state();
    
    // 检查整体SM状态
    if (warp_scheduler->all_warps_finished()) {
        sm_state = EXIT;
    }
}

bool SMContext::allocate_shared_memory(CTAContext* block) {
    // 简化实现：假设block中没有显式的共享内存需求
    // 在实际实现中，需要计算block所需的共享内存大小
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