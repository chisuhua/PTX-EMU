#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

TEST_CASE("E1: barrier after reconvergence", "[barrier][simt]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 10;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
    }

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("E2: barrier inside divergent path", "[barrier][simt][divergence]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 10;
        warp.get_warp_state().threads[i].is_active = true;
    }

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    REQUIRE(warp.get_simt_stack().depth() == 1);
}

TEST_CASE("E3: barrier completion triggers convergence check", "[barrier][simt]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 10;
        warp.get_warp_state().threads[i].is_active = true;
    }

    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
}

TEST_CASE("E4: wbar and simt_stack independent cleanup", "[barrier][simt]") {
    WarpContext warp;
    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    REQUIRE(warp.get_simt_stack().depth() == 1);

    warp.get_cta_context()->get_barrier_module().get_warp_barrier(0)->reset();
    REQUIRE(warp.get_simt_stack().depth() == 1);
}
