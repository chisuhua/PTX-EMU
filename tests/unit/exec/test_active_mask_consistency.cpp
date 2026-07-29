#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

#include <memory>

using namespace ptxsim;

namespace {
StatementContext make_nop_stmt() {
    StatementContext stmt;
    stmt.type = S_MOV;
    GenericInstr instr;
    stmt.data = instr;
    return stmt;
}

void add_thread(WarpContext &warp, int lane, bool is_exited = false) {
    auto thread = std::make_unique<ThreadContext>();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {(uint32_t)lane, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts;
    stmts.push_back(make_nop_stmt());
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                 label2pc, nullptr, nullptr);
    thread->set_state(is_exited ? EXIT : RUN);
    warp.add_thread(std::move(thread), lane);
}

void init_full_warp(WarpContext &warp) {
    for (int i = 0; i < 32; i++) {
        add_thread(warp, i);
    }
}
} // namespace

TEST_CASE("J1: default active_mask matches exec_mask", "[active_mask]") {
    WarpContext warp;
    init_full_warp(warp);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("J2: active_mask unchanged during divergence", "[active_mask]") {
    WarpContext warp;
    init_full_warp(warp);
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J3: thread exit updates active_mask", "[active_mask]") {
    WarpContext warp;
    init_full_warp(warp);

    // 退出前8个线程
    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
        warp.get_thread(i)->set_state(EXIT);
    }
    warp.update_active_mask();

    uint32_t mask = warp.get_active_mask();
    REQUIRE((mask & 0x000000FF) == 0);
    REQUIRE((mask & 0xFFFFFF00) == 0xFFFFFF00);
}

TEST_CASE("J4: active_mask consistent after convergence", "[active_mask]") {
    WarpContext warp;
    init_full_warp(warp);
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    for (int i = 0; i < 32; i++)
        warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J5: active_count matches active_mask bits", "[active_mask]") {
    WarpContext warp;
    init_full_warp(warp);

    // 前16个线程退出
    for (int i = 0; i < 16; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
        warp.get_thread(i)->set_state(EXIT);
    }
    // 接下来8个阻塞在屏障
    for (int i = 16; i < 24; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    warp.update_active_mask();

    // 还剩24-31共8个活跃线程
    REQUIRE(warp.get_active_count() == 8);
}

TEST_CASE("J6: update_active_mask syncs is_active to warp_state",
          "[active_mask][issue-004]") {
    WarpContext warp;
    init_full_warp(warp);

    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].is_active = true;
    }

    warp.update_active_mask();

    for (int i = 0; i < 8; i++) {
        REQUIRE(warp.get_warp_state().threads[i].is_active == false);
    }
}

TEST_CASE(
    "J7: update_active_mask keeps is_lane_active and is_schedulable consistent",
    "[active_mask][issue-004]") {
    WarpContext warp;
    init_full_warp(warp);

    // 让前8个线程退出
    for (int i = 0; i < 8; i++) {
        warp.get_thread(i)->set_state(EXIT);
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
    }

    // update_active_mask 后，active_mask[] 和 warp_state.is_active 应一致
    warp.update_active_mask();

    for (int i = 0; i < 32; i++) {
        bool from_mask = warp.is_lane_active(i);
        bool from_state = warp.get_warp_state().threads[i].is_active;
        REQUIRE(from_mask == from_state);
    }
}

TEST_CASE("J8: sync_to_warp_state RUN sets is_active=true after barrier",
          "[active_mask][issue-004]") {
    WarpContext warp;
    init_full_warp(warp);

    int lane = 0;

    warp.get_warp_state().threads[lane].is_active = false;
    warp.get_thread(lane)->set_state(BAR_SYNC);
    warp.get_thread(lane)->sync_to_warp_state();

    // 【修复】遵循 sync_to_warp_state 契约：caller 必须在 set_state(RUN)
    // 前显式清 is_blocked 和 status 参考 sm_context.cpp:609-610
    // 的生产调用方模式（必须同时清两者，否则 already_blocked 仍为 true）
    warp.get_warp_state().threads[lane].is_blocked = false;
    warp.get_warp_state().threads[lane].status = ptxsim::ThreadStatus::Active;
    warp.get_thread(lane)->set_state(RUN);
    warp.get_thread(lane)->sync_to_warp_state();

    REQUIRE(warp.get_warp_state().threads[lane].is_active == true);
}

// ============================================================================
// B4.1 — Scheduler Blocked-Finish Cascade Bug Regression Tests
// ----------------------------------------------------------------------------
// Locks down the contract that is_finished() must NOT return true while
// threads are still blocked (is_blocked==true) waiting for memory latency
// or barrier release. A blocked warp MUST continue to receive scheduler
// ticks; otherwise ld.global post-load latency will destroy the warp before
// it can reach bar.sync and complete the compute loop.
// ============================================================================

TEST_CASE("J9: blocked warp is not finished", "[active_mask][b4-1]") {
    // B4.1 Bug #1: is_finished() returns true when active_count==0,
    // even if some threads are still blocked (is_blocked==true).
    // That causes the scheduler to destroy the warp mid-execution.
    WarpContext warp;
    init_full_warp(warp);

    // All 32 threads are blocked on ld.global post-load latency.
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].blocked_cycles_remaining = 5;
        warp.get_warp_state().threads[i].is_active = false;
    }

    // update_active_mask() will set active_count = 0 because all threads
    // are blocked. The fix must ensure is_finished() stays false.
    warp.update_active_mask();

    REQUIRE(warp.get_active_count() == 0);
    REQUIRE(warp.is_finished() == false);
}

TEST_CASE("J10: mixed exited+blocked warp is not finished",
          "[active_mask][b4-1]") {
    // B4.1 Bug #1: a warp with some exited and some blocked threads must
    // not be considered finished — the blocked threads still need to
    // resume and reach bar.sync. Destroying the warp early produces
    // all-zero kernel output (observed in simpleGEMM / simpleCONV / bitonic).
    WarpContext warp;
    init_full_warp(warp);

    // First 16 threads: exited (e.g. divergent branch leading to `ret`).
    for (int i = 0; i < 16; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
        warp.get_thread(i)->set_state(EXIT);
    }

    // Remaining 16 threads: blocked (e.g. waiting on ld.global latency).
    for (int i = 16; i < 32; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].blocked_cycles_remaining = 5;
        warp.get_warp_state().threads[i].is_active = false;
    }

    warp.update_active_mask();

    // All threads are non-active, but the warp is NOT finished because
    // 16 threads are merely blocked and have not yet exited.
    REQUIRE(warp.get_active_count() == 0);
    REQUIRE(warp.is_finished() == false);
}

// ============================================================================
// T2-1 Task 2: is_lane_active() delegation to is_lane_schedulable() (ISSUE-005)
// ----------------------------------------------------------------------------
// is_lane_active() must read from the authoritative source
// (warp_state.threads[i].is_schedulable()), NOT from the derived active_mask[].
// This guarantees that any direct mutation of warp_state (e.g., barrier release
// via set_pc + set_state(RUN)) is immediately reflected in
// is_lane_active() without waiting for the next update_active_mask() cycle.
// ============================================================================

TEST_CASE(
    "J11: is_lane_active() delegates to warp_state without update_active_mask",
    "[active_mask][issue-005]") {
    WarpContext warp;
    init_full_warp(warp);

    // Mutate warp_state directly. Do NOT call update_active_mask() — we want to
    // prove is_lane_active() reflects warp_state immediately.
    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].is_active = false;
    }

    for (int i = 0; i < 8; i++) {
        REQUIRE(warp.is_lane_active(i) == false);
        REQUIRE(warp.is_lane_schedulable(i) == false);
    }
    for (int i = 8; i < 32; i++) {
        REQUIRE(warp.is_lane_active(i) == true);
    }
    for (int i = 0; i < 32; i++) {
        REQUIRE(warp.is_lane_active(i) == warp.is_lane_schedulable(i));
    }
}

TEST_CASE("J12: is_lane_active() reflects exited flag from warp_state",
          "[active_mask][issue-005]") {
    WarpContext warp;
    init_full_warp(warp);

    for (int i = 0; i < 16; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
    }

    for (int i = 0; i < 16; i++) {
        REQUIRE(warp.is_lane_active(i) == false);
    }
    for (int i = 16; i < 32; i++) {
        REQUIRE(warp.is_lane_active(i) == true);
    }
}
