#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

static void setup_full_warp(WarpContext& warp) {
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

TEST_CASE("I1: full divergence-convergence cycle", "[integration]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 20);
    for (int i = 16; i < 32; i++) warp.set_thread_pc(i, 11);

    REQUIRE(warp.get_simt_stack().depth() == 1);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I2: nested branches with multiple levels", "[integration]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry l1, l2, l3;
    l1.branch_pc = 10; l1.reconvergence_pc = 50;
    l1.active_mask = 0x0000FFFF; l1.return_mask = 0xFFFFFFFF; l1.return_pc = 50;
    l2.branch_pc = 20; l2.reconvergence_pc = 40;
    l2.active_mask = 0x000000FF; l2.return_mask = 0x0000FFFF; l2.return_pc = 40;
    l3.branch_pc = 25; l3.reconvergence_pc = 35;
    l3.active_mask = 0x0000000F; l3.return_mask = 0x000000FF; l3.return_pc = 35;

    warp.get_simt_stack().push(l1);
    warp.get_simt_stack().push(l2);
    warp.get_simt_stack().push(l3);
    warp.set_exec_mask(0x0000000F);

    REQUIRE(warp.get_simt_stack().depth() == 3);

    for (int i = 0; i < 8; i++) warp.set_thread_pc(i, 35);
    warp.check_reconvergence();
    REQUIRE(warp.get_simt_stack().depth() == 2);
    REQUIRE(warp.get_exec_mask() == 0x000000FF);

    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 40);
    warp.check_reconvergence();
    REQUIRE(warp.get_simt_stack().depth() == 1);
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 50);
    warp.check_reconvergence();
    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I3: branch + barrier combination", "[integration]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I4: convergence with thread exits", "[integration][exit]") {
    WarpContext warp;
    setup_full_warp(warp);

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
        warp.get_warp_state().threads[i].pc = 0;
    }
    for (int i = 8; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 30;
    }

    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("I5: scheduler skips unfinished warp", "[integration][scheduler]") {
    WarpContext warp;
    setup_full_warp(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 20;
    REQUIRE(warp.is_warp_ready_to_fetch() == false);

    warp.get_warp_state().threads[0].pc = 20;
    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}
