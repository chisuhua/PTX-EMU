#include "ptxsim/sm_context.h"
// #include "memory/memory_manager.h"        // 添加MemoryManager头文件
#include "memory/resource_manager.h"      // 添加ResourceManager头文件
#include "memory/shared_memory_manager.h" // 添加SharedMemoryManager头文件
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/ptx_config.h"     // 添加ptx_config头文件
#include "ptxsim/warp_scheduler.h" // 添加warp调度器头文件
#include "utils/logger.h"          // 添加logger头文件
#include <algorithm>
#include <cassert>
#include <set>

SMContext::SMContext(int max_warps, int max_threads_per_sm,
                     size_t shared_mem_size, int sm_id)
    : max_warps_per_sm(max_warps), max_threads_per_sm(max_threads_per_sm),
      max_shared_mem(shared_mem_size), allocated_shared_mem(0),
      current_thread_count(0), sm_state(IDLE), next_physical_block_id(0),
      next_physical_warp_id(0), shared_mem_manager_(nullptr),
      current_reservation_id_(0), sm_id_(sm_id) {
    // 初始化warp调度器，使用RoundRobinWarpScheduler具体实现
    warp_scheduler = std::make_unique<RoundRobinWarpScheduler>();

    // 初始化资源统计
    stats_ = {0, max_shared_mem, 0, max_warps, 0, max_threads_per_sm};

    // 获取共享内存管理器
    shared_mem_manager_ =
        ResourceManager::instance().get_shared_memory_manager(sm_id);
    if (!shared_mem_manager_) {
        PTX_DEBUG_EMU("Failed to get shared memory manager for SM %d", sm_id);
    }
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

void SMContext::init() {
    // 现在初始化逻辑在构造函数中完成
    // 这里可以放置其他初始化逻辑
}

bool SMContext::add_block(std::unique_ptr<CTAContext> block) {
    // 1. 计算资源需求
    size_t required_shared_mem = block->get_shared_memory_requirement();
    int required_warps = block->get_warp_count();

    // 2. 检查资源是否足够
    if (!reserve_resources(required_shared_mem, required_warps)) {
        PTX_DEBUG_EMU("Failed to reserve resources for block: "
                      "shared_mem=%zu, warps=%d",
                      required_shared_mem, required_warps);
        return false; // 资源不足
    }

    // 3. 分配共享内存
    void *shared_mem_space = nullptr;
    if (required_shared_mem > 0 && shared_mem_manager_) {
        shared_mem_space = shared_mem_manager_->allocate(
            required_shared_mem, block->get_reservation_id());
        if (!shared_mem_space) {
            // 分配失败，释放预留
            release_resources(block->get_reservation_id());
            PTX_DEBUG_EMU(
                "Failed to allocate shared memory of size %zu for block %d",
                required_shared_mem, block->get_reservation_id());
            return false;
        }
    }

    // 4. 构建共享内存符号表
    block->build_shared_memory_symbol_table(shared_mem_space);

    // 5. 更新已分配的共享内存统计
    allocated_shared_mem += required_shared_mem;

    // 6. 分配物理ID并记录块信息
    int physical_block_id = next_physical_block_id++;
    physical_block_warp_counts[physical_block_id] = required_warps;

    // 7. 添加到管理列表 - 直接使用unique_ptr
    managed_blocks.insert({physical_block_id, std::move(block)});

    // 8. 获取warp所有权并添加到SM
    auto block_warps = managed_blocks[physical_block_id]->release_warps();
    for (auto &warp : block_warps) {
        warp->set_physical_block_id(physical_block_id);
        warp->set_physical_warp_id(next_physical_warp_id++);
        // 设置SMContext指针
        warp->set_sm_context(this);
        warps.push_back(std::move(warp));
        warp_scheduler->add_warp(warps.back().get());
    }

    // 更新状态
    update_state();

    PTX_DEBUG_EMU("Successfully added block with %zu shared memory bytes, "
                  "%d warps to SM %d",
                  required_shared_mem, required_warps, sm_id_);

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

    // 检查barrier等待的线程，如果有线程在等待barrier且已满足同步条件，需要更新它们的状态
    for (auto &[barId, waiting_threads] : barrier_waiting_threads) {
        if (!waiting_threads.empty()) {
            // 找到这些等待线程所属的block，检查是否所有线程都已到达barrier
            ThreadContext *sample_thread = *waiting_threads.begin();
            if (sample_thread && sample_thread->get_warp_context()) {
                int physical_block_id =
                    sample_thread->get_warp_context()->get_physical_block_id();
                if (physical_block_id != -1) {
                    auto block_it = managed_blocks.find(physical_block_id);
                    if (block_it != managed_blocks.end()) {
                        CTAContext *cta_ctx = block_it->second.get();
                        int total_threads_in_block =
                            cta_ctx->get_thread_count();

                        // 检查是否所有线程都已经到达barrier
                        if (waiting_threads.size() >=
                            static_cast<size_t>(total_threads_in_block)) {
                            // 所有线程都到达了barrier，释放所有等待的线程
                            PTX_DEBUG_EMU("All threads reached barrier %d, "
                                          "releasing %zu threads",
                                          barId, waiting_threads.size());

                            // 设置所有等待线程的状态为RUN
                            for (auto waiting_thread : waiting_threads) {
                                waiting_thread->set_state(RUN);
                            }

                            // 清空barrier等待队列
                            waiting_threads.clear();
                        }
                    }
                }
            }
        }
    }

    // 调度下一个warp执行
    WarpContext *next_warp = warp_scheduler->schedule_next();
    if (next_warp) {
        // 设置warp为被调度状态
        next_warp->set_scheduled(true);

        // 检查warp中是否有活跃线程处于barrier状态
        bool has_barrier_threads = false;
        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext *thread = next_warp->get_thread(lane);
            if (thread && thread->is_at_barrier()) {
                has_barrier_threads = true;
                break;
            }
        }

        // 如果warp中有线程在barrier等待，则跳过该warp的执行
        if (!has_barrier_threads) {

            // 获取当前warp中第一个活跃线程的PC作为指令来源
            ThreadContext *firstActiveThread = nullptr;
            StatementContext *currentStmt = nullptr;

            for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
                ThreadContext *thread = next_warp->get_thread(lane);
                if (thread && thread->is_active() && !thread->is_exited() &&
                    !thread->is_at_barrier()) {
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
                // 从DebugConfig单例获取warp跟踪配置
                if (ptxsim::DebugConfig::get().is_trace_warp_enabled()) {
                    print_warp_status(
                        next_warp); // 在执行指令前打印被调度warp的状态
                }

                // 执行warp指令
                next_warp->execute_warp_instruction(*currentStmt);
            }
        }

        // 执行完后取消warp的被调度状态
        next_warp->set_scheduled(false);
    }

    // 更新状态
    update_state();

    // 从DebugConfig单例获取warp跟踪配置，如果需要打印所有warp状态
    if (ptxsim::DebugConfig::get().is_trace_warp_enabled()) {
        print_warp_status(); // 打印所有warp的状态
    }

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
        auto warp = it->get();
        if (warp && !warp->is_finished()) {
            has_active_warps = true;
            it++;
        } else {
            // 从warp调度器中移除warp
            warp_scheduler->remove_warp(warp);

            auto physical_block_id = warp->get_physical_block_id();
            physical_block_warp_counts[physical_block_id]--;
            it = warps.erase(it);
        }
    }

    // 检查是否有正在管理的blocks
    bool has_managed_blocks = !managed_blocks.empty();

    if (!has_active_warps && !has_managed_blocks) {
        sm_state = EXIT;
    } else {
        sm_state = RUN;
    }

    // 更新统计信息
    stats_.active_warps = warps.size();
    stats_.active_threads = get_active_threads_count();
    if (shared_mem_manager_) {
        stats_.allocated_shared_mem = shared_mem_manager_->get_allocated_size();
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
            free_shared_memory(it->second.get());

            physical_block_warp_counts.erase(physical_block_id);
            // 从managed_blocks中移除这个块
            it = managed_blocks.erase(it);

        } else {
            ++it;
        }
    }
}

void SMContext::free_shared_memory(CTAContext *block) {
    // 释放共享内存
    if (block->sharedMemSpace != nullptr && shared_mem_manager_) {
        size_t shared_mem_size =
            block->get_shared_memory_requirement(); // 获取要释放的内存大小

        shared_mem_manager_->deallocate(block->sharedMemSpace,
                                        block->get_reservation_id());

        // 更新本地统计 - 减去释放的内存大小
        if (allocated_shared_mem >= shared_mem_size) {
            allocated_shared_mem -= shared_mem_size;
        } else {
            // 防止下溢出，理论上不应该发生
            allocated_shared_mem = 0;
        }

        // 重置block的共享内存指针
        const_cast<void *&>(block->sharedMemSpace) = nullptr;
    }
}

bool SMContext::reserve_resources(size_t shared_mem_size, int warp_count) {
    if (!shared_mem_manager_) {
        PTX_DEBUG_EMU("Shared memory manager not initialized");
        return false;
    }

    // 检查共享内存是否足够
    if (shared_mem_manager_->get_available_size() < shared_mem_size) {
        PTX_DEBUG_EMU(
            "Insufficient shared memory: requested %zu, available %zu",
            shared_mem_size, shared_mem_manager_->get_available_size());
        return false;
    }

    // 检查warp数量是否足够
    if (static_cast<int>(warps.size()) + warp_count > max_warps_per_sm) {
        PTX_DEBUG_EMU("Insufficient warps: current %zu, requested %d, max %d",
                      warps.size(), warp_count, max_warps_per_sm);
        return false;
    }

    // 分配预留ID
    int reservation_id = current_reservation_id_++;

    // 更新CTAContext的预留ID
    // 注意：block参数是CTAContext指针，但我们不能直接访问其reservation_id_成员
    // 因为这是私有成员，所以我们需要在调用reserve_resources之前设置reservation_id
    return true;
}

void SMContext::release_resources(int reservation_id) {
    // 在实际实现中，这会释放为特定块预留的资源
    // 但现在我们使用共享内存管理器来处理资源释放
    PTX_DEBUG_EMU("Releasing resources for reservation_id %d", reservation_id);
}

SMContext::ResourceStats SMContext::get_resource_stats() const {
    return stats_;
}

void SMContext::print_resource_usage() const {
    PTX_DEBUG_EMU("=== SM %p Resource Usage ===", this);
    PTX_DEBUG_EMU("Shared Memory: %zu/%zu (%.2f%%)",
                  stats_.allocated_shared_mem, stats_.max_shared_mem,
                  stats_.max_shared_mem > 0
                      ? 100.0 * stats_.allocated_shared_mem /
                            stats_.max_shared_mem
                      : 0.0);
    PTX_DEBUG_EMU(
        "Warps: %d/%d (%.2f%%)", stats_.active_warps, stats_.max_warps,
        stats_.max_warps > 0 ? 100.0 * stats_.active_warps / stats_.max_warps
                             : 0.0);
    PTX_DEBUG_EMU("Threads: %d/%d (%.2f%%)", stats_.active_threads,
                  stats_.max_threads,
                  stats_.max_threads > 0
                      ? 100.0 * stats_.active_threads / stats_.max_threads
                      : 0.0);
    PTX_DEBUG_EMU("========================");
}

bool SMContext::synchronize_barrier(int barId, ThreadContext *thread) {
    // 获取线程所在的物理block ID
    int physical_block_id =
        thread->get_warp_context()
            ? thread->get_warp_context()->get_physical_block_id()
            : -1;

    if (physical_block_id == -1) {
        PTX_DEBUG_EMU("Error: Could not determine physical block ID for thread "
                      "at barrier");
        return false;
    }

    // 获取该block的CTAContext来获取block维度信息
    auto block_it = managed_blocks.find(physical_block_id);
    if (block_it == managed_blocks.end()) {
        PTX_DEBUG_EMU("Error: Could not find block %d for barrier sync",
                      physical_block_id);
        return false;
    }

    CTAContext *cta_ctx = block_it->second.get();
    int total_threads_in_block = cta_ctx->get_thread_count();

    // 更新该barrier的线程计数
    barrier_thread_counts[barId] = total_threads_in_block;

    // 将当前线程加入到barrier等待队列
    barrier_waiting_threads[barId].insert(thread);

    PTX_DEBUG_EMU("Thread in block %d waiting at barrier %d, %zu threads "
                  "waiting, need %d",
                  physical_block_id, barId,
                  barrier_waiting_threads[barId].size(),
                  total_threads_in_block);

    // 检查是否所有线程都已经到达barrier
    if (barrier_waiting_threads[barId].size() >=
        static_cast<size_t>(barrier_thread_counts[barId])) {
        // 所有线程都到达了barrier，释放所有等待的线程
        PTX_DEBUG_EMU("All threads reached barrier %d, releasing %zu threads",
                      barId, barrier_waiting_threads[barId].size());

        // 设置所有等待线程的状态为RUN
        for (auto waiting_thread : barrier_waiting_threads[barId]) {
            waiting_thread->set_state(RUN);
        }

        // 清空barrier等待队列
        barrier_waiting_threads[barId].clear();

        return true; // 表示同步完成
    }

    // 线程设置为等待状态
    thread->set_state(BAR_SYNC);

    return false; // 表示线程还在等待
}

void SMContext::print_warp_status() const {
    PTX_DEBUG_EMU("=== SM %d All %zu Warps Status ===", sm_id_, warps.size());

    for (size_t i = 0; i < warps.size(); ++i) {
        const auto &warp = warps[i];
        if (warp) {
            print_warp_status(warp.get(), false); // 调用带参数版本
        }
    }
}

void SMContext::print_warp_status(const WarpContext *warp,
                                  bool print_sm_id) const {
    if (!warp) {
        PTX_DEBUG_EMU("Warp is null, cannot print status");
        return;
    }

    if (print_sm_id)
        PTX_DEBUG_EMU("--- SM %d Warp Status ---", sm_id_);

    int active_count = warp->get_active_count();
    bool is_finished = warp->is_finished();
    bool is_all_exited = warp->is_all_threads_exited();
    int warp_id = warp->get_warp_id();
    bool is_scheduled = warp->is_scheduled(); // 获取调度状态

    PTX_DEBUG_EMU("Warp ID=%d, Active Threads=%d, IsFinished=%s, "
                  "AllExited=%s, Scheduled=%s",
                  warp_id, active_count, is_finished ? "Yes" : "No",
                  is_all_exited ? "Yes" : "No", is_scheduled ? "Yes" : "No");

    // 按PC值分组，记录每个PC对应的lane及其状态
    std::map<int, std::array<char, WarpContext::WARP_SIZE>> pc_to_lanes;
    std::map<int, std::string> pc_to_instruction;

    for (int lane = 0; lane < WarpContext::WARP_SIZE; ++lane) {
        ThreadContext *thread = warp->get_thread(lane);

        if (thread) {
            int pc = thread->get_pc();

            // 获取线程状态字符
            char state_char;
            EXE_STATE state = thread->get_state();
            switch (state) {
            case RUN:
                state_char = 'R';
                break;
            case EXIT:
                state_char = 'E';
                break;
            case BAR_SYNC:
                state_char = 'S';
                break;
            default:
                state_char = 'U';
                break;
            }

            // 将该lane的状态加入对应的PC组
            pc_to_lanes[pc][lane] = state_char;

            // 获取当前PC对应的指令文本
            if (pc_to_instruction.find(pc) == pc_to_instruction.end()) {
                StatementContext *stmt = thread->get_current_statement();
                if (stmt != nullptr) {
                    pc_to_instruction[pc] = stmt->instructionText;
                } else {
                    pc_to_instruction[pc] = "<no_instruction>";
                }
            }
        } else {
            // 如果线程不存在，标记为未知，但仍然要记录其位置
            // 因为我们仍需在每个PC组中为这个lane显示'-'
            for (auto &[pc, lanes] : pc_to_lanes) {
                lanes[lane] = '-';
            }
        }
    }

    // 为每个不同的PC值打印一行信息
    for (const auto &[pc, lanes] : pc_to_lanes) {
        std::string lane_states = "";
        for (int lane = 0; lane < WarpContext::WARP_SIZE; ++lane) {
            if (lanes[lane] != '\0') {
                lane_states += lanes[lane];
            } else {
                // 如果此lane的PC与此PC不匹配，则显示'-'
                lane_states += '-';
            }
        }

        PTX_DEBUG_EMU("  PC[0x%x]: %s | Lane States: %s", pc,
                      pc_to_instruction[pc].c_str(), lane_states.c_str());
    }
}
