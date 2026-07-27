#include "ptxsim/warp_context.h"
#include "warp_context_active_mask.h"
#include "warp_context_simt.h"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_trace.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_trace_formatter.h"
#include <algorithm>
#include <cassert>
#include <cstring>

void WarpContext::handle_branch(const std::string &predicate,
                                bool predicate_negated, int target_pc,
                                int reconvergence_pc, int current_inst_pc) {
    assert(
        current_inst_pc >= 0 &&
        "handle_branch: current_inst_pc must be non-negative, got default -1");
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;

    for (int i = 0; i < 32; i++) {
        if (!warp_state.threads[i].is_active)
            continue;
        if (warp_state.threads[i].pc != current_inst_pc)
            continue;

        bool should_branch = true;

        if (!predicate.empty()) {
            std::string pred_name = predicate;
            if (!pred_name.empty() && pred_name[0] == '%') {
                pred_name = pred_name.substr(1);
            }

            if (register_bank_manager_) {
                void *reg_addr =
                    register_bank_manager_->get_register(pred_name, warp_id, i);
                if (reg_addr) {
                    uint8_t pred_value = *static_cast<uint8_t *>(reg_addr);
                    bool pred_bool = (pred_value != 0);
                    should_branch = predicate_negated ? !pred_bool : pred_bool;
                }
            }
        }

        if (should_branch) {
            taken_mask |= (1u << i);
        } else {
            not_taken_mask |= (1u << i);
        }
    }

    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    int fallthrough_pc = current_inst_pc + 1;

    if (is_divergent) {
        ptxsim::SIMTStackEntry entry;
        entry.branch_pc = current_inst_pc;
        entry.reconvergence_pc = reconvergence_pc;
        entry.active_mask = taken_mask;
        entry.return_mask = warp_state.exec_mask;
        entry.return_pc = reconvergence_pc;

        simt_stack.push(entry);

        // SIMT栈push跟踪
        if (ptxsim::DebugConfig::get().is_trace_simt_stack_enabled() &&
            sm_context_) {
            PTX_DEBUG_EMU("%s", ptxsim::WarpTraceFormatter::format_simt_push(
                                    sm_context_->get_cycle_count(),
                                    sm_context_->get_sm_id(), warp_id, entry,
                                    taken_mask, simt_stack)
                                    .c_str());
        }

        for (int i = 0; i < 32; i++) {
            if (taken_mask & (1u << i)) {
                warp_state.threads[i].pc = target_pc;
                warp_state.threads[i].next_pc = target_pc;
            } else if (not_taken_mask & (1u << i)) {
                warp_state.threads[i].pc = fallthrough_pc;
                warp_state.threads[i].next_pc = fallthrough_pc;
            }
        }

        warp_state.exec_mask = taken_mask;
    } else {
        int next_pc = (taken_mask != 0) ? target_pc : fallthrough_pc;

        for (int i = 0; i < 32; i++) {
            if (warp_state.threads[i].is_active &&
                warp_state.threads[i].pc == current_inst_pc) {
                warp_state.threads[i].pc = next_pc;
                warp_state.threads[i].next_pc = next_pc;
            }
        }
    }
}

void WarpContext::advance_thread_pc(int lane_id, int new_pc) {
    if (lane_id < 0 || lane_id >= WARP_SIZE)
        return;
    warp_state.threads[lane_id].pc = new_pc;
    warp_state.threads[lane_id].next_pc = new_pc;
}

bool WarpContext::check_reconvergence() {
    return warp_simt::check_reconvergence(this);
}

// BUG-DISPATCH-GATE-LANE0-SKIP (fix): only block lanes that belong to the
// top entry's divergence group (return_mask). Lanes outside return_mask are
// on an unrelated path (or have already converged past reconv_pc) and must
// continue executing. Without this guard, the gate incorrectly blocks any
// lane sitting at reconv_pc, including those on unrelated paths — this was
// the root cause of cute_rmsnorm's lane 0 st.shared being skipped (lanes
// 1-31 had stale @%p10 back-edge entries whose return_mask did not include
// lane 0, but the gate blocked lane 0 anyway because lane 0's PC ==
// entry.reconvergence_pc).
bool WarpContext::check_and_block_at_reconvergence_point(
    int target_pc, std::vector<int> &blocked_lanes) {
    blocked_lanes.clear();
    if (simt_stack.empty()) {
        return false;
    }

    const ptxsim::SIMTStackEntry &top = simt_stack.top();
    int reconv_pc = top.reconvergence_pc;
    if (target_pc != reconv_pc) {
        return false;
    }

    int lanes_not_at_reconv = 0;
    int lanes_at_reconv = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!(top.return_mask & (1u << i)))
            continue;
        if (warp_state.threads[i].is_exited)
            continue;
        if ((int)warp_state.threads[i].pc == reconv_pc) {
            lanes_at_reconv++;
        } else {
            lanes_not_at_reconv++;
        }
    }

    // Block every lane at the reconvergence point so the last arrival
    // cannot execute past it before check_reconvergence() pops the entry.
    if (lanes_at_reconv == 0)
        return false;

    for (int i = 0; i < WARP_SIZE; i++) {
        // Block all lanes in return_mask that sit at reconv_pc. This includes
        // both the "taken" (active_mask) lanes and the "fallthrough" lanes —
        // both must wait until the entry pops before advancing past the
        // convergence point. Without this, fallthrough lanes can execute
        // instructions at reconv_pc before the taken lanes have converged,
        // causing out-of-order execution at the reconvergence point.
        if (!(top.return_mask & (1u << i)))
            continue;
        if (!warp_state.threads[i].is_exited &&
            (int)warp_state.threads[i].pc == reconv_pc &&
            !warp_state.threads[i].is_blocked) {
            warp_state.threads[i].is_blocked = true;
            blocked_lanes.push_back(i);
        }
    }
    return !blocked_lanes.empty();
}

WarpContext::WarpContext()
    : active_count(0), warp_id(-1), single_step_mode(false),
      divergence_detected(false), sm_context_(nullptr), simt_stack() {
    for (int i = 0; i < WARP_SIZE; i++) {
        warp_thread_ids[i] = -1;
        active_mask[i] = true;
        warp_state.threads[i].pc = 0;
        warp_state.threads[i].next_pc = 0;
        warp_state.threads[i].is_active = false;
        warp_state.threads[i].is_exited = false;
        warp_state.threads[i].is_blocked = false;
        warp_state.threads[i].status = ptxsim::ThreadStatus::Active;
    }
    warp_state.exec_mask = 0xFFFFFFFF;
}

void WarpContext::add_thread(std::unique_ptr<ThreadContext> thread,
                             int lane_id) {
    if (lane_id >= 0 && lane_id < WARP_SIZE) {
        threads.resize(
            std::max(threads.size(), static_cast<size_t>(lane_id + 1)));
        threads[lane_id] = std::move(thread);

        if (threads[lane_id]) {
            // 设置warp_context_指针
            threads[lane_id]->set_warp_context(this);

            warp_thread_ids[lane_id] =
                threads[lane_id]->ThreadIdx.x +
                threads[lane_id]->ThreadIdx.y * threads[lane_id]->BlockDim.x +
                threads[lane_id]->ThreadIdx.z * threads[lane_id]->BlockDim.x *
                    threads[lane_id]->BlockDim.y;

            warp_state.threads[lane_id].is_active = true;
            active_mask[lane_id] = true;
            active_count++;

            // T2-3 A4b: Mirror LaneMaskPod fields only. backend_links_
            // .threads is left empty until A4c consolidates thread
            // ownership from WarpContext::threads to BackendLinksPod.
            lane_mask_.warp_thread_ids[lane_id] = warp_thread_ids[lane_id];
            lane_mask_.active_mask[lane_id] = active_mask[lane_id];
            lane_mask_.active_count = active_count;
        } else {
            warp_thread_ids[lane_id] = -1;
            lane_mask_.warp_thread_ids[lane_id] = -1;
        }
    }
}

void WarpContext::execute_warp_instruction(StatementContext &stmt,
                                           int target_pc) {
    std::vector<int> blocked_lanes;
    if (check_and_block_at_reconvergence_point(target_pc, blocked_lanes)) {
        // 在 update_active_mask 之前获取 lanes，否则被阻塞的线程会被过滤掉
        auto current_lanes_before_block = get_lanes_by_pc();
        update_active_mask();
        // 汇聚点调试输出：有线程到达汇聚点但仍有分歧路径未到达
        if (ptxsim::DebugConfig::get().is_trace_convergence_enabled() &&
            sm_context_) {
            auto current_lanes = current_lanes_before_block;
            if (current_lanes.size() > 1) {
                // 找出下一条非阻塞的调度路径（跳过汇聚点自身的 PC 组）
                int next_pc = -1;
                uint32_t next_mask = 0;
                for (const auto &[candidate_pc, candidate_lanes] :
                     current_lanes) {
                    if (candidate_pc == target_pc)
                        continue; // 跳过汇聚点（已阻塞）
                    bool has_non_blocked = false;
                    for (int lane : candidate_lanes) {
                        if (!warp_state.threads[lane].is_blocked) {
                            has_non_blocked = true;
                            break;
                        }
                    }
                    if (has_non_blocked) {
                        next_pc = candidate_pc;
                        for (int lane : candidate_lanes) {
                            if (!warp_state.threads[lane].is_blocked)
                                next_mask |= (1u << lane);
                        }
                        break;
                    }
                }
                // 构建剩余分歧路径（排除汇聚点 PC 组）
                std::map<int, std::vector<int>> remaining_lanes;
                for (const auto &[pc_val, lane_list] : current_lanes) {
                    if (pc_val != target_pc) {
                        remaining_lanes[pc_val] = lane_list;
                    }
                }
                if (!remaining_lanes.empty() && next_pc >= 0) {
                    PTX_DEBUG_EMU("%s", ptxsim::WarpTraceFormatter::
                                            format_convergence_remaining(
                                                target_pc, remaining_lanes,
                                                next_pc, next_mask)
                                                .c_str());
                }
            }
        }
        return;
    }
    // Snapshot lanes to process BEFORE any handler runs.
    std::vector<int> lanes_to_process;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i >= (int)threads.size() || threads[i] == nullptr)
            continue;
        bool lane_active = is_lane_active(i);
        bool blocked_at_barrier = (threads[i]->get_state() == BAR_SYNC);
        if ((!lane_active && !blocked_at_barrier) ||
            warp_state.threads[i].pc != static_cast<uint32_t>(target_pc))
            continue;
        lanes_to_process.push_back(i);
    }

    for (int i : lanes_to_process) {
        ThreadContext *thread = threads[i].get();

        thread->sync_from_warp_state();

        // Re-check PC after sync: a previous lane's divergent branch
        // handling (e.g. bra_pred) may have moved this lane's PC away
        // from target_pc. The snapshot pattern would otherwise re-execute
        // the divergence, double-jumping lanes.
        if (warp_state.threads[i].pc != static_cast<uint32_t>(target_pc)) {
            thread->sync_to_warp_state();
            continue;
        }

        // Skip already-exited lanes: warp-level handlers (ret) mark ALL
        // lanes as exited, but sync_to_warp_state would otherwise re-run
        // the handler on each lane, double-advancing PC.
        if (thread->get_state() == EXIT) {
            thread->sync_to_warp_state();
            continue;
        }

        if (thread->get_state() == BAR_SYNC) {
            if (cta_context_ != nullptr) {
                // Scan all warp barrier slots for any incomplete barrier.
                // With 16 slots (ADR-0008), the active barrier can be on any slot.
                auto& bm = cta_context_->get_barrier_module();
                bool any_wbar_incomplete = false;
                for (int i = 0; i < ptxsim::MAX_WARP_BARRIERS; ++i) {
                    auto* wbar = bm.get_warp_barrier(i);
                    if (wbar && wbar->is_initialized() && !wbar->is_complete()) {
                        any_wbar_incomplete = true;
                        break;
                    }
                }

                if (any_wbar_incomplete) {
                    PTX_WARN_EMU("Fallback CTA sync: lane %d, wbar incomplete",
                                 thread->lane_id_);
                    cta_context_->get_barrier_module().arrive_at_cta_barrier(
                        thread->bar_id, thread);
                }
            }
            thread->sync_to_warp_state();
            continue;
        }

        thread->execute_thread_instruction();
        thread->sync_to_warp_state();

        if (ptxsim::ExecutionTracer::is_enabled()) {
            ptxsim::ExecutionTracer::record(i, warp_state.threads[i].pc,
                                            stmt.instructionText);
        }
    }

    update_active_mask();
}

void WarpContext::update_active_mask() {
    warp_active_mask::update_active_mask(this);
}

void WarpContext::set_active_mask(int lane_id, bool active) {
    warp_active_mask::set_active_mask_lane(this, lane_id, active);
}

bool WarpContext::is_finished() const {
    // A warp is finished when ALL threads have exited.
    // Blocked threads (is_blocked) are NOT finished — they will
    // resume when blocked_cycles_remaining drains or a barrier releases them.
    return active_count == 0 && is_all_threads_exited();
}

bool WarpContext::is_warp_ready_to_fetch() const {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!warp_state.threads[i].is_active)
            continue;
        if (warp_state.threads[i].pc != warp_state.threads[i].next_pc) {
            return false;
        }
    }
    return true;
}

bool WarpContext::is_all_threads_exited() const {
    // 检查warp中的所有线程是否都已退出
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            if (!threads[i]->is_exited()) {
                // 如果有任何一个线程还没有退出，则warp尚未完成
                return false;
            }
        }
    }
    return true;
}

void WarpContext::sync_threads() {
    // 在真正的硬件模拟中，这里会实现warp级同步
    // 目前我们简单地确保所有活跃线程都执行到相同的PC
}

void WarpContext::reset() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            threads[i]->reset();
            active_mask[i] = true;
            warp_state.threads[i].is_active = true;
            active_count++;
        } else {
            active_mask[i] = false;
            warp_state.threads[i].is_active = false;
        }
    }
    divergence_detected = false;
}

uint32_t WarpContext::get_active_mask() const {
    return warp_active_mask::get_active_mask_u32(this);
}

void WarpContext::set_active_mask(uint32_t mask) {
    warp_active_mask::set_active_mask_u32(this, mask);
}

std::map<int, std::vector<int>> WarpContext::get_lanes_by_pc() const {
    std::map<int, std::vector<int>> pc_to_lanes;

    for (int lane = 0; lane < WARP_SIZE; lane++) {
        if (lane < (int)threads.size() && threads[lane] != nullptr &&
            warp_state.threads[lane].is_active &&
            !warp_state.threads[lane].is_exited) {
            int pc = warp_state.threads[lane].pc;
            pc_to_lanes[pc].push_back(lane);
        }
    }

    return pc_to_lanes;
}

std::vector<int> WarpContext::get_unique_pcs() const {
    std::vector<int> pcs;
    auto lanes_by_pc = get_lanes_by_pc();

    for (const auto &[pc, lanes] : lanes_by_pc) {
        pcs.push_back(pc);
    }

    return pcs;
}

void WarpContext::force_reconvergence_at_barrier(int barrier_pc) {
    // 不主动推进线程PC —— 让调度器自然选择非阻塞的PC执行
    // 屏障处理器会在 divergence 路径中阻塞当前线程（set is_blocked=true），
    // 调度器随后会跳过有阻塞线程的PC组，选择其他PC执行。
    // 当所有线程都到达屏障后，wbar 完成并释放所有线程。
    //
    // 注意：不能推进线程PC，否则会跳过 shared memory store 等关键指令。
    // 注释掉的代码（advance_thread_pc）曾导致 E2E 测试中共享内存数据丢失。
}

void WarpContext::decrement_blocked_cycles(ptxsim::WarpState &ws) {
    // (B4.1 Bug #2 + #3: must run every tick, even for warps not yet selected,
    // so that newly-unblocked lanes become schedulable in the SAME tick).
    for (auto &thread : ws.threads) {
        if (thread.is_blocked && thread.blocked_cycles_remaining > 0) {
            thread.blocked_cycles_remaining--;
            if (thread.blocked_cycles_remaining == 0) {
                thread.is_blocked = false;
                if (!thread.is_exited &&
                    thread.status == ptxsim::ThreadStatus::Active) {
                    thread.is_active = true;
                }
            }
        }
    }
}

void WarpContext::set_blocked_cycles_for_active(uint32_t cycles) {
    for (auto &thread : warp_state.threads) {
        // Skip threads that are already blocked (barrier, previous instruction
        // latency, etc.) and threads at barrier (status == Blocked) to prevent
        // interaction between blocked_cycles and barrier state machine.
        if (thread.is_active && !thread.is_blocked &&
            thread.status != ptxsim::ThreadStatus::Blocked) {
            thread.blocked_cycles_remaining = cycles;
            thread.is_blocked = true;
        }
    }
    // T2-1 contract: keep active_mask[] / active_count synchronized with
    // warp_state mutations so scheduler sees the change immediately.
    update_active_mask();
}
