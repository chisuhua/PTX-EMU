#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

TEST_CASE("J1: default active_mask matches exec_mask", "[active_mask]") {
    WarpContext warp;
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("J2: active_mask unchanged during divergence", "[active_mask]") {
    WarpContext warp;
    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J3: thread exit updates active_mask", "[active_mask]") {
    WarpContext warp;
    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    warp.update_active_mask();
    uint32_t mask = warp.get_active_mask();
    REQUIRE((mask & 0x000000FF) == 0);
    REQUIRE((mask & 0xFFFFFF00) == 0xFFFFFF00);
}

TEST_CASE("J4: active_mask consistent after convergence", "[active_mask]") {
    WarpContext warp;
    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J5: active_count matches active_mask bits", "[active_mask]") {
    WarpContext warp;
    for (int i = 0; i < 16; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    for (int i = 16; i < 24; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    warp.update_active_mask();
    REQUIRE(warp.get_active_count() == 8);
}
