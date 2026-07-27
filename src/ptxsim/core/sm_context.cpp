#include "ptxsim/sm_context.h"
#include "sm_context_reconvergence.h"
#include "sm_context_cpptlm_inject.h"
// #include "memory/memory_manager.h"        // 添加MemoryManager头文件
#include "memory/resource_manager.h"          // 添加ResourceManager头文件
#include "memory/shared_memory_manager.h"     // 添加SharedMemoryManager头文件
#include "ptx_ir/instruction_latency_table.h" // Phase 8.B PTX-6 (ADR-0020)
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/register_analyzer.h"    // Phase 8.B PTX-6 (ADR-0020)
#include "ptxsim/warp_scheduler.h"       // 添加warp调度器头文件
#include "ptxsim/warp_trace_formatter.h" // 添加ptx_config头文件
#include "utils/logger.h"                // 添加logger头文件
#include <algorithm>
#include <cassert>
#include <cmath> // Phase 8.B PTX-6 — for std::ceil(double→uint32_t)
#include <set>

namespace {

// === Phase 8.B PTX-6: exe_once() 3-step injection helpers (ADR-0020) ===
// File-local helpers — kept in anonymous namespace to avoid ABI exposure.
// Step A and Step C use scoreboard; both respect nullptr semantics
// (nullptr injector = byte-identical to pre-injection behavior).
// Step B has been extracted to SMContext::step_b_set_blocked_cycles
// (public static) for direct unit testability.

// Step A: Scoreboard hazard check. Returns true if execution may proceed,
// false if RAW hazard detected or scoreboard full (caller should goto
// warp_done).
inline bool step_a_scoreboard_check(IScoreboard *scoreboard, WarpContext *warp,
                                    const StatementContext &stmt) {
    if (!scoreboard)
        return true; // nullptr = skip injection
    scoreboard->tick();
    if (!scoreboard->has_free_entry())
        return false;
    auto dest_regs = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    auto warp_id = static_cast<uint32_t>(warp->get_physical_warp_id());
    std::vector<uint32_t> allocated;
    for (auto reg_id : dest_regs) {
        // Skip duplicate reg_ids to prevent double-release on rollback
        // (e.g., instructions that reference the same register multiple times).
        if (std::find(allocated.begin(), allocated.end(), reg_id) !=
            allocated.end())
            continue;
        if (!scoreboard->allocate(reg_id, warp_id)) {
            for (auto prev : allocated)
                scoreboard->release(prev, warp_id);
            return false;
        }
        allocated.push_back(reg_id);
    }
    return true;
}

// Step B has been extracted to SMContext::step_b_set_blocked_cycles (public
// static, see sm_context.h) for direct unit testability. The two call sites
// in exe_once() now invoke SMContext::step_b_set_blocked_cycles.

// Step C: Scoreboard release after successful execution. Caller MUST guard
// with warp_executed flag to prevent releasing unallocated entries when
// Step A failed.
inline void step_c_release_scoreboard(IScoreboard *scoreboard,
                                      WarpContext *warp,
                                      const StatementContext &stmt) {
    if (!scoreboard)
        return; // nullptr = skip injection
    auto dest_regs = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    auto warp_id = static_cast<uint32_t>(warp->get_physical_warp_id());
    for (auto reg_id : dest_regs)
        scoreboard->release(reg_id, warp_id);
}

/// Dest register extraction wrapper — aligns with design.md §7.2 helper list.
/// Delegates to RegisterAnalyzer::get_dest_registers_as_ids for the actual
/// StatementContext variant traversal.
inline std::vector<uint32_t> get_dest_registers(const StatementContext &stmt) {
    return RegisterAnalyzer::get_dest_registers_as_ids(stmt);
}

} // anonymous namespace

SMContext::SMContext(int max_warps, int max_threads_per_sm,
                     size_t shared_mem_size, int sm_id)
    : max_warps_per_sm(max_warps), max_threads_per_sm(max_threads_per_sm),
      max_shared_mem(shared_mem_size), allocated_shared_mem(0),
      current_thread_count(0), sm_state(IDLE), next_physical_block_id(0),
      next_physical_warp_id(0), shared_mem_manager_(nullptr),
      current_reservation_id_(0), sm_id_(sm_id), cycle_counter_(0) {
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

    // BUG-SM-ADMISSION-OVERFLOW: 拒绝"绝对无法 fit"的块
    // 单 block 所需资源 > SM 总容量 → 直接失败(原语义)
    // 否则进 pending_blocks_,等待资源释放
    if (required_warps > max_warps_per_sm) {
        PTX_DEBUG_EMU(
            "Block requires %d warps > SM max %d — cannot ever fit, dropping",
            required_warps, max_warps_per_sm);
        return false;
    }
    if (required_shared_mem > max_shared_mem) {
        PTX_DEBUG_EMU("Block requires %zu shared mem > SM max %zu — cannot "
                      "ever fit, dropping",
                      required_shared_mem, max_shared_mem);
        return false;
    }

    // 2. 分配reservation_id并设置到CTAContext
    int reservation_id = current_reservation_id_++;
    block->set_reservation_id(reservation_id);

    // 3. 检查资源是否足够
    if (!reserve_resources(required_shared_mem, required_warps)) {
        // BUG-SM-ADMISSION-OVERFLOW fix: 不丢弃,进 pending 队列
        // 资源释放后由 try_admit_pending_blocks() 重新 admit
        PTX_DEBUG_EMU(
            "Block queued in pending (SM full): shared_mem=%zu, warps=%d, "
            "pending_count=%zu",
            required_shared_mem, required_warps, pending_blocks_.size() + 1);
        pending_blocks_.push_back(std::move(block));
        return true;
    }

    void *shared_mem_space = nullptr;
    if (required_shared_mem > 0 && shared_mem_manager_) {
        shared_mem_space = shared_mem_manager_->allocate(
            required_shared_mem, block->get_reservation_id());
        if (!shared_mem_space) {
            release_resources(block->get_reservation_id());
            PTX_DEBUG_EMU(
                "Failed to allocate shared memory of size %zu for block %d",
                required_shared_mem, block->get_reservation_id());
            return false;
        }
    }

    block->build_shared_memory_symbol_table(shared_mem_space);
    allocated_shared_mem += required_shared_mem;

    int physical_block_id = next_physical_block_id++;
    physical_block_warp_counts[physical_block_id] = required_warps;
    managed_blocks.insert({physical_block_id, std::move(block)});

    auto block_warps = managed_blocks[physical_block_id]->release_warps();
    for (auto &warp : block_warps) {
        warp->set_physical_block_id(physical_block_id);
        warp->set_physical_warp_id(next_physical_warp_id++);
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

void SMContext::try_admit_pending_blocks() {
    // FIFO admit:队首 block 资源能 fit 就 admit,继续检查下一个
    // 直到队首 block 资源 fit 失败(说明当前 SM 已满)或队列空
    while (!pending_blocks_.empty()) {
        auto &front_block = pending_blocks_.front();
        size_t req_smem = front_block->get_shared_memory_requirement();
        int req_warps = front_block->get_warp_count();

        if (!reserve_resources(req_smem, req_warps)) {
            // 资源仍不足(被其他 admitted block 占用),停止
            // 下一个 cleanup_finished_blocks() 释放资源后再试
            return;
        }

        // 资源够,admit 这个 block
        std::unique_ptr<CTAContext> block = std::move(pending_blocks_.front());
        pending_blocks_.pop_front();

        // 后续逻辑与 add_block 主路径完全相同
        int reservation_id = block->get_reservation_id();
        void *shared_mem_space = nullptr;
        if (req_smem > 0 && shared_mem_manager_) {
            shared_mem_space =
                shared_mem_manager_->allocate(req_smem, reservation_id);
            if (!shared_mem_space) {
                release_resources(reservation_id);
                PTX_DEBUG_EMU(
                    "try_admit_pending: smem alloc failed for pending block");
                continue;
            }
        }

        block->build_shared_memory_symbol_table(shared_mem_space);
        allocated_shared_mem += req_smem;

        int physical_block_id = next_physical_block_id++;
        physical_block_warp_counts[physical_block_id] = req_warps;
        managed_blocks.insert({physical_block_id, std::move(block)});

        auto block_warps = managed_blocks[physical_block_id]->release_warps();
        for (auto &warp : block_warps) {
            warp->set_physical_block_id(physical_block_id);
            warp->set_physical_warp_id(next_physical_warp_id++);
            warp->set_sm_context(this);
            warps.push_back(std::move(warp));
            warp_scheduler->add_warp(warps.back().get());
        }

        PTX_DEBUG_EMU("try_admit_pending: admitted block with %zu smem, "
                      "%d warps; %zu still pending",
                      req_smem, req_warps, pending_blocks_.size());
    }
}

// === CppTLM D1-Full injection helpers (Phase 8.B PTX-6, ADR-0020) ===
// Public static for testability (tests/unit/sm/test_exe_once_helpers.cpp).

bool SMContext::is_tensor_core_instruction(const StatementContext &stmt) {
    return stmt.type >= StatementType::S_TCGEN05_ALLOC &&
           stmt.type <= StatementType::S_TCGEN05_FENCE;
}

PipelineId
SMContext::map_instruction_to_pipeline(const StatementContext &stmt) {
    if (is_tensor_core_instruction(stmt)) {
        return PipelineId::P4_TC;
    }
    if (stmt.type == StatementType::S_LD || stmt.type == StatementType::S_ST ||
        stmt.type == StatementType::S_ATOM) {
        return PipelineId::P3_LSU;
    }
    // SFU (Special Function Unit) instructions → P2_SFU
    switch (stmt.type) {
    case StatementType::S_SIN:
    case StatementType::S_COS:
    case StatementType::S_LG2:
    case StatementType::S_EX2:
    case StatementType::S_RCP:
    case StatementType::S_RSQRT:
    case StatementType::S_SQRT:
    case StatementType::S_TANH:
        return PipelineId::P2_SFU;
    default:
        break;
    }
    // FP64 instructions → P1_FP64 (detect by .f64 qualifier)
    if (std::holds_alternative<GenericInstr>(stmt.data)) {
        const auto &instr = std::get<GenericInstr>(stmt.data);
        for (const auto &q : instr.qualifiers) {
            if (q == Qualifier::Q_F64)
                return PipelineId::P1_FP64;
        }
    }
    // Default: integer/FP32 → P0_INT_FP32 (V_SIMD for vector/SIMD ops not
    // distinguishable from StatementType alone — map to P0 as baseline.)
    return PipelineId::P0_INT_FP32;
}

TcPrecision
SMContext::map_instruction_to_tc_precision(const StatementContext &stmt) {
    if (std::holds_alternative<GenericInstr>(stmt.data)) {
        const auto &instr = std::get<GenericInstr>(stmt.data);
        for (const auto &q : instr.qualifiers) {
            switch (q) {
            case Qualifier::Q_F16:
                return TcPrecision::FP16;
            case Qualifier::Q_BF16:
                return TcPrecision::BF16;
            case Qualifier::Q_TCGEN_TF32:
                return TcPrecision::TF32;
            case Qualifier::Q_F8:
                return TcPrecision::FP8;
            default:
                continue;
            }
        }
    }
    return TcPrecision::FP16;
}

void SMContext::step_b_set_blocked_cycles(IPipelineLatencyProvider *pipeline,
                                          ITensorCoreTiming *tc,
                                          WarpContext *warp,
                                          const StatementContext &stmt) {
    sm_cpptlm_inject::step_b_set_blocked_cycles(pipeline, tc, warp, stmt);
}

EXE_STATE SMContext::exe_once() {
    cycle_counter_++; // 递增周期计数器
    if (sm_state != RUN) {
        return sm_state;
    }

    // 检查是否所有warp都已完成
    if (warp_scheduler->all_warps_finished()) {
        sm_state = EXIT;
        return sm_state;
    }

    // Decrement blocked_cycles_remaining for ALL warps BEFORE scheduling
    // (B4.1 Bug #2 + #3: must run every tick, even for warps not yet selected,
    // so that newly-unblocked lanes become schedulable in the SAME tick).
    for (auto &w : warps) {
        if (!w)
            continue;
        WarpContext::decrement_blocked_cycles(w->get_warp_state());
    }

    // 【BUG-001 Fix #1】After blocked-decrement, recalculate active_count for
    // each warp. decrement_blocked_cycles() directly sets is_active=true on
    // unblocked lanes, but this bypasses WarpContext::active_count which is
    // only updated by update_active_mask(). Without this fix, active_count
    // stays at 0, is_active() returns false, and the warp scheduler skips
    // the warp forever — causing a hang on any kernel with ld.global.
    for (auto &w : warps) {
        if (w)
            w->update_active_mask();
    }

    // 调度下一个warp执行
    WarpContext *next_warp = warp_scheduler->schedule_next();
    if (next_warp) {
        // 设置warp为被调度状态
        next_warp->set_scheduled(true);
        bool warp_executed = false;

        // [Divergent Execution Fix] Execute instructions for all unique PC
        // groups Fast path: if all schedulable lanes share the same PC, use the
        // old path
        auto lanes_by_pc = next_warp->get_lanes_by_pc();

        if (lanes_by_pc.size() == 1) {
            // Fast path: non-divergent, all lanes at same PC
            auto it = lanes_by_pc.begin();
            int target_pc = it->first;
            const auto &lanes = it->second;
            int sample_lane = lanes[0];
            ThreadContext *sample_thread = next_warp->get_thread(sample_lane);

            if (sample_thread) {
                if (target_pc >= 0 &&
                    target_pc <
                        static_cast<int>(sample_thread->statements_size())) {
                    StatementContext *stmt =
                        sample_thread->get_statement_at(target_pc);
                    if (stmt) {
                        // 【Phase 8.B PTX-6】Step A: Scoreboard hazard check
                        if (!step_a_scoreboard_check(scoreboard_, next_warp,
                                                     *stmt))
                            goto warp_done;
                        if (ptxsim::DebugConfig::get()
                                .is_trace_warp_enabled()) {
                            if (ptxsim::DebugConfig::get()
                                    .is_trace_cycle_enabled()) {
                                PTX_DEBUG_EMU(
                                    "%s",
                                    ptxsim::WarpTraceFormatter::
                                        format_instruction(
                                            cycle_counter_, sm_id_,
                                            next_warp->get_warp_id(), target_pc,
                                            stmt->instructionText,
                                            next_warp->get_exec_mask())
                                            .c_str());
                            } else {
                                print_warp_status(next_warp);
                            }
                        }
                        next_warp->execute_warp_instruction(*stmt, target_pc);
                        warp_executed = true;
                        // 【Phase 8.B PTX-6】Step B: Latency query →
                        // set_blocked_cycles_for_active (MUST run AFTER
                        // execute: the instruction must execute before threads
                        // are blocked, otherwise is_lane_active() returns false
                        // and skip all lanes.)
                        try {
                            SMContext::step_b_set_blocked_cycles(
                                pipeline_provider_, tensor_core_timing_,
                                next_warp, *stmt);
                        } catch (...) {
                            // Scoreboard rollback on exception: Step A
                            // allocated slots that Step C would release. If
                            // Step B throws, we must release them now.
                            if (scoreboard_)
                                step_c_release_scoreboard(scoreboard_,
                                                          next_warp, *stmt);
                            throw;
                        }
                        // 【Phase 8.B PTX-6】Step C: Scoreboard release (gated
                        // by warp_executed to prevent releasing unallocated
                        // entries on Step A failure)
                        step_c_release_scoreboard(scoreboard_, next_warp,
                                                  *stmt);
                        sm_reconvergence::drain_simt_and_update_active(
                            next_warp);
                    }
                }
            }
        } else if (!lanes_by_pc.empty()) {
            int pc = -1;
            const std::vector<int> *selected_lanes = nullptr;
            bool found_non_blocked = false;

            auto &ws = next_warp->get_warp_state();
            for (const auto &[candidate_pc, candidate_lanes] : lanes_by_pc) {
                bool all_non_blocked = true;
                for (int lane : candidate_lanes) {
                    if (ws.threads[lane].is_blocked) {
                        all_non_blocked = false;
                        break;
                    }
                }
                if (all_non_blocked) {
                    pc = candidate_pc;
                    selected_lanes = &candidate_lanes;
                    found_non_blocked = true;
                    break;
                }
            }

            if (!found_non_blocked) {
                auto it = lanes_by_pc.begin();
                pc = it->first;
                selected_lanes = &it->second;
            }

            const auto &lanes = *selected_lanes;

            // 构建当前执行 group 的真实 lane mask
            uint32_t current_exec_mask = 0;
            for (int lane : lanes) {
                current_exec_mask |= (1u << lane);
            }

            int sample_lane = lanes[0];
            ThreadContext *sample_thread = next_warp->get_thread(sample_lane);

            if (sample_thread && pc >= 0 &&
                pc < sample_thread->statements_size()) {
                StatementContext *stmt = sample_thread->get_statement_at(pc);

                if (stmt) {
                    // 【Phase 8.B PTX-6】Step A: Scoreboard hazard check
                    if (!step_a_scoreboard_check(scoreboard_, next_warp, *stmt))
                        goto warp_done;
                    if (ptxsim::DebugConfig::get().is_trace_warp_enabled()) {
                        if (ptxsim::DebugConfig::get()
                                .is_trace_cycle_enabled()) {
                            PTX_DEBUG_EMU(
                                "%s",
                                ptxsim::WarpTraceFormatter::format_instruction(
                                    cycle_counter_, sm_id_,
                                    next_warp->get_warp_id(), pc,
                                    stmt->instructionText, current_exec_mask)
                                    .c_str());
                        } else {
                            print_warp_status(next_warp);
                        }
                    }

                    next_warp->execute_warp_instruction(*stmt, pc);
                    warp_executed = true;
                    // 【Phase 8.B PTX-6】Step B: Latency query →
                    // set_blocked_cycles_for_active (MUST run AFTER execute —
                    // same constraint as fast path.)
                    try {
                        SMContext::step_b_set_blocked_cycles(
                            pipeline_provider_, tensor_core_timing_, next_warp,
                            *stmt);
                    } catch (...) {
                        if (scoreboard_)
                            step_c_release_scoreboard(scoreboard_, next_warp,
                                                      *stmt);
                        throw;
                    }
                    // 【Phase 8.B PTX-6】Step C: Scoreboard release (gated by
                    // warp_executed)
                    step_c_release_scoreboard(scoreboard_, next_warp, *stmt);

                    // Check SIMT stack reconvergence after every instruction
                    sm_reconvergence::drain_simt_and_update_active(
                        next_warp);
                }
            }

            // Log divergence info if enabled
            if (ptxsim::DebugConfig::get().is_trace_divergence_enabled() &&
                lanes_by_pc.size() > 1) {
                PTX_DEBUG_EMU(
                    "%s",
                    ptxsim::WarpTraceFormatter::format_divergence(lanes_by_pc)
                        .c_str());
            }
        }

        // 执行完后取消warp的被调度状态
        // 【Phase 8.B PTX-6】warp_done label: target of `goto warp_done` from
        // Step A failure (must be BEFORE set_scheduled(false) per Oracle
        // 2026-07-17 BUG-1 fix)
    warp_done:
        next_warp->set_scheduled(false);
    }

    // 更新状态
    update_state();

    // 从DebugConfig单例获取warp跟踪配置，仅在非cycle模式时打印完整状态
    if (ptxsim::DebugConfig::get().is_trace_warp_enabled() &&
        !ptxsim::DebugConfig::get().is_trace_cycle_enabled()) {
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

    // 清理已完成的blocks（释放共享内存）
    cleanup_finished_blocks();

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
    auto it = managed_blocks.begin();
    while (it != managed_blocks.end()) {
        auto physical_block_id = it->first;
        auto block = it->second.get();
        if (physical_block_warp_counts[physical_block_id] == 0) {
            free_shared_memory(it->second.get());
            physical_block_warp_counts.erase(physical_block_id);
            it = managed_blocks.erase(it);
        } else {
            ++it;
        }
    }
    // BUG-SM-ADMISSION-OVERFLOW: 资源刚释放,尝试重灌 pending
    try_admit_pending_blocks();
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

void SMContext::set_divergence_execution_mode(
    ptxsim::DivergenceExecutionMode mode) {
    divergence_mode_ = mode;
    PTX_DEBUG_EMU("SM %d: Set divergence execution mode to %s", sm_id_,
                  ptxsim::divergence_execution_mode_to_string(mode));
}

ptxsim::DivergenceExecutionMode
SMContext::get_divergence_execution_mode() const {
    return divergence_mode_;
}

int SMContext::select_next_group(const std::vector<int> &active_lanes) {
    // With multiple active paths, select based on divergence mode
    if (active_lanes.size() <= 1) {
        return 0; // No divergence, use first group
    }

    switch (divergence_mode_) {
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

void SMContext::suspend_and_switch(int current_group, int next_group) {
    // Suspend current group and switch to next_group
    // This is a placeholder for future blocking implementation (Phase 3)
    // For now, we just proceed with the next group selection
    PTX_DEBUG_EMU("SM %d: Suspend group %d, switch to group %d", sm_id_,
                  current_group, next_group);
}
