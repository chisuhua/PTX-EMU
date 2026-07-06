#include "ptxsim/simt_pc_manager.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include <cassert>

// ── PC accessors ───────────────────────────────────────────────────────

int SimtPcManager::get_pc() const {
    if (!warp_context_)
        return 0;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32)
        return 0;
    return warp_context_->get_warp_state().threads[lane].pc;
}

void SimtPcManager::set_pc(int new_pc) {
    if (!warp_context_)
        return;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32)
        return;
    warp_context_->get_warp_state().threads[lane].pc = new_pc;
    warp_context_->get_warp_state().threads[lane].next_pc = new_pc;
}

int SimtPcManager::get_next_pc() const {
    if (!warp_context_)
        return 0;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32)
        return 0;
    return warp_context_->get_warp_state().threads[lane].next_pc;
}

void SimtPcManager::set_next_pc(int new_next_pc) {
    if (!warp_context_)
        return;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32)
        return;
    warp_context_->get_warp_state().threads[lane].next_pc = new_next_pc;
}

void SimtPcManager::commit_pc() { set_pc(get_next_pc()); }

// ── Bidirectional warp_state sync ──────────────────────────────────────

void SimtPcManager::sync_from_warp_state() {
    if (!warp_context_)
        return;

    int lane = lane_id_;
    if (lane < 0 || lane >= WarpContext::WARP_SIZE)
        return;

    ptxsim::ThreadState &thread_state =
        warp_context_->get_warp_state().threads[lane];

    // PC is read directly via get_pc()/get_next_pc() — no sync needed
    // sync_to_warp_state() keeps next_pc consistent

    switch (thread_state.status) {
    case ptxsim::ThreadStatus::Active:
        state_ = RUN;
        break;
    case ptxsim::ThreadStatus::Blocked:
        state_ = BAR_SYNC;
        break;
    case ptxsim::ThreadStatus::Exited:
        state_ = EXIT;
        break;
    case ptxsim::ThreadStatus::Yielded:
        state_ = RUN;
        break;
    }
}

void SimtPcManager::sync_to_warp_state() {
    if (!warp_context_)
        return;

    int lane = lane_id_;
    if (lane < 0 || lane >= WarpContext::WARP_SIZE)
        return;

    ptxsim::ThreadState &thread_state =
        warp_context_->get_warp_state().threads[lane];

    // If thread is already waiting at a barrier (is_blocked=true or
    // status=Blocked), only sync next_pc — do NOT overwrite blocked status.
    // Note: barrier release goes through set_state(RUN) + clearing is_blocked
    // BEFORE calling this function (see BarrierModule::arrive_at_cta_barrier).
    bool already_blocked =
        (thread_state.is_blocked ||
         thread_state.status == ptxsim::ThreadStatus::Blocked);

    // Barrier completion handlers may directly update warp_state via
    // warp_ctx->advance_thread_pc(). Here we only sync ThreadContext's
    // own next_pc state upward.
    thread_state.next_pc = get_next_pc();

    // already_blocked guard: only sync next_pc, preserve blocked state
    if (already_blocked) {
        return;
    }

    // Translate EXE_STATE → ThreadStatus
    switch (state_) {
    case RUN:
        thread_state.status = ptxsim::ThreadStatus::Active;
        thread_state.is_blocked = false;
        thread_state.is_active = true;
        break;
    case BAR_SYNC:
        thread_state.status = ptxsim::ThreadStatus::Blocked;
        thread_state.is_blocked = true;
        break;
    case EXIT:
        thread_state.status = ptxsim::ThreadStatus::Exited;
        thread_state.is_exited = true;
        thread_state.is_active = false;
        thread_state.is_blocked = false;
        break;
    default:
        break;
    }
}

// ── PC validation and statement query ──────────────────────────────────

bool SimtPcManager::is_valid_pc() const {
    int pc = get_pc();
    return statements_ != nullptr && pc >= 0 &&
           pc < static_cast<int>(statements_->size());
}

bool SimtPcManager::is_valid_pc(int p) const {
    return statements_ != nullptr && p >= 0 &&
           p < static_cast<int>(statements_->size());
}

int SimtPcManager::statements_size() const {
    return statements_ ? static_cast<int>(statements_->size()) : 0;
}

StatementContext *SimtPcManager::get_statement_at(int p) {
    if (statements_ != nullptr && p >= 0 &&
        p < static_cast<int>(statements_->size())) {
        return &(*statements_)[p];
    }
    return nullptr;
}

StatementContext *SimtPcManager::get_current_statement() {
    int pc = get_pc();
    if (statements_ != nullptr && pc >= 0 &&
        pc < static_cast<int>(statements_->size())) {
        return &(*statements_)[pc];
    }
    return nullptr;
}