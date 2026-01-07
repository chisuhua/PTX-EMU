#include "ptxsim/sm_context.h"
#include "memory/memory_manager.h" // 添加MemoryManager头文件
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "utils/logger.h" // 添加logger头文件
#include <algorithm>
#include <cassert>

SMContext::SMContext(int max_warps, int max_threads_per_sm,
                     size_t shared_mem_size)
    : max_warps_per_sm(max_warps), max_threads_per_sm(max_threads_per_sm),
      max_shared_mem(shared_mem_size), allocated_shared_mem(0),
      current_thread_count(0), sm_state(EXE_STATE::IDLE),
      next_physical_block_id(0), next_physical_warp_id(0) {
    // 初始化warp调度器
    warp_scheduler = std::make_unique<WarpScheduler>();
}

SMContext::~SMContext() {
    // 在SMContext销毁时，需要释放所有warp中的共享内存
    // 由于warp持有ThreadContext，而ThreadContext可能通过指针访问共享内存
    // 但是共享内存空间本身是由SMContext分配的，需要在这里处理

    // 遍历所有warp，找到它们所属的CTAContext，并释放共享内存
    // 但是由于warp已经从CTAContext转移过来，我们需要一种方式来追踪共享内存
    // 当前的实现中，sharedMemSpace是通过build_shared_memory_symbol_table设置到CTAContext的
    // 但在add_block后，CTAContext的warp被转移了，CTAContext本身可能没有被保存

    // 对于当前的架构，当CTA执行完成时，应该调用free_shared_memory来释放内存
    // 在SMContext销毁时，如果还有未释放的共享内存，发出警告
    if (allocated_shared_mem > 0) {
        PTX_DEBUG_EMU("Warning: SMContext destroyed with %zu bytes of "
                      "allocated shared memory",
                      allocated_shared_mem);
    }
}

void SMContext::init(Dim3 &gridDim, Dim3 &blockDim,
                     std::vector<StatementContext> &statements,
                     std::map<std::string, Symtable *> &name2Sym,
                     std::map<std::string, int> &label2pc) {
    this->gridDim = gridDim;
}

bool SMContext::add_block(CTAContext *block) {
    // 检查资源是否足够
    if (warps.size() + block->warpNum > max_warps_per_sm) {
        return false; // 超过SM最大warp数限制
    }

    if (allocated_shared_mem + block->sharedMemBytes > max_shared_mem) {
        return false; // 超过共享内存限制
    }

    // 分配共享内存
    if (!allocate_shared_memory(block)) {
        return false; // 内存分配失败
    }
    // 为CTAContext分配共享内存空间
    void *shared_mem_space = nullptr;
    if (block->sharedMemBytes > 0) {
        shared_mem_space =
            MemoryManager::instance().malloc_shared(block->sharedMemBytes);
        if (shared_mem_space == nullptr) {
            PTX_DEBUG_EMU(
                "Failed to allocate SHARED memory of size %zu for block",
                block->sharedMemBytes);
            // 释放已分配的共享内存
            free_shared_memory(block);
            return false;
        }

        PTX_DEBUG_EMU("Allocated SHARED memory of size %zu at %p for block",
                      block->sharedMemBytes, shared_mem_space);
    }

    // 调用CTAContext的启动函数来构建共享内存符号表，传入分配好的内存空间
    block->build_shared_memory_symbol_table(shared_mem_space);

    // 分配物理ID并记录这个块的warp总数
    int physical_block_id = next_physical_block_id++;
    physical_block_warp_counts[physical_block_id] = block->warpNum;

    // 将块添加到管理列表
    managed_blocks.emplace_back(
        {physical_block_id, std::unique_ptr<CTAContext>(block)});

    // 获取CTAContext中的warp所有权
    auto block_warps = block->release_warps();

    // 将warp添加到SM的warp列表中
    for (auto &warp : block_warps) {
        // 设置物理warp ID
        warp->set_physical_warp_id(next_physical_warp_id++);
        // 添加到SM的warp列表
        warps.push_back(std::move(warp));
    }

    // 更新SM状态
    update_state();

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
    WarpContext *next_warp = warp_scheduler->schedule_next();
    if (next_warp) {
        // 获取当前warp中第一个活跃线程的PC作为指令来源
        ThreadContext *firstActiveThread = nullptr;
        StatementContext *currentStmt = nullptr;

        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext *thread = next_warp->get_thread(lane);
            if (thread && thread->is_active() && !thread->is_exited()) {
                firstActiveThread = thread;
                // 使用安全的PC检查
                if (thread->is_valid_pc()) {
                    currentStmt = thread->get_current_statement();
                    break; // 找到指令后跳出
                }
                assert(false);
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
    for (const auto &warp : warps) {
        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext *thread = warp->get_thread(lane);
            if (thread) {
                // 在检查线程状态前，先确保线程的PC是有效的
                // 如果线程已退出，则不需要检查其他状态
                if (!thread->is_at_barrier() && !thread->is_exited()) {
                    all_at_barrier = false;
                    break;
                }
            }
        }
        if (!all_at_barrier)
            break;
    }

    if (all_at_barrier) {
        // 所有线程都在屏障处等待，释放屏障
        for (auto &warp : warps) {
            for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
                ThreadContext *thread = warp->get_thread(lane);
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

bool SMContext::is_idle() const { return warp_scheduler->all_warps_finished(); }

int SMContext::get_active_warps_count() const {
    int count = 0;
    for (const auto &warp : warps) {
        if (warp && warp->is_active()) {
            count++;
        }
    }
    return count;
}

int SMContext::get_active_threads_count() const {
    int count = 0;
    for (const auto &warp : warps) {
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
    auto it = warps.begin();
    while (it != warps.end()) {
        auto warp = *it;
        if (warp && !warp->is_finished()) {
            has_active_warps = true;
            it++;
        } else {
            auto physical_block_id = warp->get_physical_block_id();
            physical_block_warp_counts[physical_block_id]--;
            it = warps.erase(it);
        }
    }

    if (!has_active_warps) {
        sm_state = EXIT;
    } else {
        sm_state = RUN;
    }
}

void SMContext::cleanup_finished_blocks() {
    // 检查每个managed_block，看其相关的warp是否都已完成
    auto it = managed_blocks.begin();
    while (it != managed_blocks.end()) {
        auto physical_block_id = it->first;
        auto block = it->second.get();
        if (physical_block_warp_counts[physical_block_id] == 0) {
            // 释放这个块的共享内存
            free_shared_memory(it->get());

            physical_block_warp_counts.erase(physical_block_id);
            // 从managed_blocks中移除这个块
            it = managed_blocks.erase(it);

        } else {
            ++it;
        }
    }
}

bool SMContext::allocate_shared_memory(CTAContext *block) {
    // 计算block所需的共享内存大小
    size_t required_shared_mem = block->sharedMemBytes;

    if (allocated_shared_mem + required_shared_mem > max_shared_mem) {
        return false; // 共享内存不足
    }

    // 分配共享内存并构建符号表
    // 为了实现这个功能，我们需要修改CTAContext的初始化过程
    // 使CTAContext能够接收一个外部分配的共享内存地址
    // 并在SMContext中调用build_shared_memory_symbol_table来构建符号表
    allocated_shared_mem += required_shared_mem;
    return true;
}

void SMContext::free_shared_memory(CTAContext *block) {
    // 释放共享内存
    if (block->sharedMemSpace != nullptr) {
        MemoryManager::instance().free_shared(block->sharedMemSpace);
        allocated_shared_mem -= block->sharedMemBytes;
        // 重置block的共享内存指针
        const_cast<void *&>(block->sharedMemSpace) = nullptr;
    }
}