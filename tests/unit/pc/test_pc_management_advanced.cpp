#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"

using namespace ptxsim;

TEST_CASE("G1: advance_thread_pc updates both sources", "[pc][unified]") {
    WarpContext warp;
    warp.advance_thread_pc(5, 20);
    REQUIRE(warp.get_thread_pc(5) == 20);
    REQUIRE(warp.get_warp_state().threads[5].next_pc == 20);
}

TEST_CASE("G2: advance_thread_pc out-of-bounds safety", "[pc][safety]") {
    WarpContext warp;
    warp.advance_thread_pc(-1, 10);
    warp.advance_thread_pc(32, 10);
    warp.advance_thread_pc(0, 42);
    REQUIRE(warp.get_thread_pc(0) == 42);
}

TEST_CASE("G3: multiple advance_thread_pc calls accumulate", "[pc]") {
    WarpContext warp;
    warp.advance_thread_pc(0, 10);
    warp.advance_thread_pc(0, 20);
    warp.advance_thread_pc(0, 30);
    REQUIRE(warp.get_thread_pc(0) == 30);
}