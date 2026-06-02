#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

TEST_CASE("H1: sync_from_warp_state reads PC correctly", "[sync]") {
    WarpContext warp;
    warp.set_thread_pc(0, 15);
    warp.get_warp_state().threads[0].status = ThreadStatus::Active;
    REQUIRE(warp.get_thread_pc(0) == 15);
    REQUIRE(warp.get_warp_state().threads[0].status == ThreadStatus::Active);
}

TEST_CASE("H2: sync_to_warp_state preserves PC", "[sync]") {
    WarpContext warp;
    warp.get_warp_state().threads[0].next_pc = 20;
    warp.get_warp_state().threads[0].status = ThreadStatus::Active;
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 20);
}

TEST_CASE("H3: branch PC not overwritten by sync", "[sync][branch]") {
    WarpContext warp;
    warp.set_thread_pc(0, 20);
    warp.get_warp_state().threads[0].next_pc = 20;
    REQUIRE(warp.get_thread_pc(0) == 20);
}

TEST_CASE("H4: force_set_pc for barrier completion", "[sync][barrier]") {
    WarpContext warp;
    warp.set_thread_pc(0, 10);
    warp.set_thread_pc(0, 30);
    REQUIRE(warp.get_thread_pc(0) == 30);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 30);
}

TEST_CASE("H5: exited thread state sync", "[sync][exit]") {
    WarpContext warp;
    warp.get_warp_state().threads[0].is_exited = true;
    warp.get_warp_state().threads[0].is_active = false;
    REQUIRE(warp.get_warp_state().threads[0].is_exited == true);
    REQUIRE(warp.get_warp_state().threads[0].is_active == false);
}

TEST_CASE("H6: bidirectional sync consistency", "[sync]") {
    WarpContext warp;
    warp.set_thread_pc(0, 10);
    REQUIRE(warp.get_thread_pc(0) == 10);
    warp.set_thread_pc(0, 20);
    REQUIRE(warp.get_thread_pc(0) == 20);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 20);
}
