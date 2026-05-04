#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"

using namespace ptxsim;

TEST_CASE("G1: advance_thread_pc updates both sources", "[pc][unified]") {
    WarpContext warp;
    warp.advance_thread_pc(5, 20);
    REQUIRE(warp.get_thread_pc(5) == 20);
    REQUIRE(warp.get_warp_state().threads[5].next_pc == 20);
}

TEST_CASE("G2: advance_all_threads only advances active", "[pc][unified]") {
    WarpContext warp;
    warp.get_warp_state().threads[0].is_active = false;
    warp.get_warp_state().threads[1].is_active = false;
    warp.advance_all_threads(30);
    REQUIRE(warp.get_thread_pc(2) == 30);
    REQUIRE(warp.get_thread_pc(0) == 0);
}

TEST_CASE("G3: advance_thread_pc out-of-bounds safety", "[pc][safety]") {
    WarpContext warp;
    warp.advance_thread_pc(-1, 10);
    warp.advance_thread_pc(32, 10);
    warp.advance_thread_pc(0, 42);
    REQUIRE(warp.get_thread_pc(0) == 42);
}

TEST_CASE("G4: multiple advance_thread_pc calls accumulate", "[pc]") {
    WarpContext warp;
    warp.advance_thread_pc(0, 10);
    warp.advance_thread_pc(0, 20);
    warp.advance_thread_pc(0, 30);
    REQUIRE(warp.get_thread_pc(0) == 30);
}

TEST_CASE("G5: pc consistency after advance_all_threads", "[pc]") {
    WarpContext warp;
    warp.advance_all_threads(42);
    for (int i = 0; i < 32; i++) {
        if (warp.get_warp_state().threads[i].is_active) {
            REQUIRE(warp.get_thread_pc(i) == 42);
        }
    }
}
