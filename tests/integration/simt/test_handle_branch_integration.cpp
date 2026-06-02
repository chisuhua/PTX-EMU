#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

static void init_warp_threads(WarpContext& warp) {
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 10;
        warp.get_warp_state().threads[i].next_pc = 10;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
    }
    warp.get_warp_state().exec_mask = 0xFFFFFFFF;
}

TEST_CASE("D3: divergent branch pushes SIMT stack", "[branch][simt][divergence]") {
    WarpContext warp;
    init_warp_threads(warp);

    warp.handle_branch("", false, 20, 30, 10);

    REQUIRE(warp.get_simt_stack().empty() == true);
    for (int i = 0; i < 32; i++) {
        REQUIRE(warp.get_thread_pc(i) == 20);
    }
}

TEST_CASE("D5: convergence after divergence restores state", "[branch][simt][convergence]") {
    WarpContext warp;
    init_warp_threads(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 20);
    for (int i = 16; i < 32; i++) warp.set_thread_pc(i, 11);
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_simt_stack().depth() == 1);
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("D4: nested divergence maintains stack order", "[branch][simt][nested]") {
    WarpContext warp;
    init_warp_threads(warp);

    SIMTStackEntry outer;
    outer.branch_pc = 10;
    outer.reconvergence_pc = 50;
    outer.active_mask = 0x0000FFFF;
    outer.return_mask = 0xFFFFFFFF;
    outer.return_pc = 50;
    warp.get_simt_stack().push(outer);
    warp.set_exec_mask(0x0000FFFF);

    SIMTStackEntry inner;
    inner.branch_pc = 20;
    inner.reconvergence_pc = 40;
    inner.active_mask = 0x000000FF;
    inner.return_mask = 0x0000FFFF;
    inner.return_pc = 40;
    warp.get_simt_stack().push(inner);
    warp.set_exec_mask(0x000000FF);

    REQUIRE(warp.get_simt_stack().depth() == 2);
    REQUIRE(warp.get_simt_stack().top().branch_pc == 20);
}

TEST_CASE("D1: non-divergent all taken branch", "[branch][simt]") {
    WarpContext warp;
    init_warp_threads(warp);
    warp.handle_branch("", false, 20, 30, 10);
    REQUIRE(warp.get_simt_stack().empty() == true);
    for (int i = 0; i < 32; i++) {
        REQUIRE(warp.get_thread_pc(i) == 20);
    }
}

TEST_CASE("D2: non-divergent none taken (all fallthrough)", "[branch][simt]") {
    WarpContext warp;
    init_warp_threads(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 32; i++) {
        warp.set_thread_pc(i, 11);
    }
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}
