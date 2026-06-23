#include "catch_amalgamated.hpp"
#include "ptxsim/thread_state.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

TEST_CASE("C1: default initialization", "[warp_state]") {
    WarpState ws;
    for (int i = 0; i < 32; i++) {
        REQUIRE(ws.threads[i].pc == 0);
        REQUIRE(ws.threads[i].is_active == true);
        REQUIRE(ws.threads[i].is_exited == false);
        REQUIRE(ws.threads[i].is_blocked == false);
    }
    REQUIRE(ws.exec_mask == 0xFFFFFFFF);
    REQUIRE(ws.current_wbar_id == -1);
}

TEST_CASE("C2: reset restores defaults", "[warp_state]") {
    WarpState ws;
    ws.exec_mask = 0x0000FFFF;
    ws.current_wbar_id = 2;
    ws.threads[0].is_exited = true;
    ws.threads[0].is_active = false;
    ws.threads[0].pc = 99;

    ws.reset();

    REQUIRE(ws.exec_mask == 0xFFFFFFFF);
    REQUIRE(ws.current_wbar_id == -1);
    REQUIRE(ws.threads[0].is_exited == false);
    REQUIRE(ws.threads[0].is_active == true);
    REQUIRE(ws.threads[0].pc == 0);
}

TEST_CASE("C3: count_active_lanes", "[warp_state]") {
    WarpState ws;
    for (int i = 0; i < 16; i++)
        ws.threads[i].is_active = false;
    REQUIRE(ws.count_active_lanes() == 16);
    for (int i = 16; i < 21; i++)
        ws.threads[i].is_exited = true;
    REQUIRE(ws.count_active_lanes() == 11);
}

TEST_CASE("C4: count_schedulable_lanes", "[warp_state]") {
    WarpState ws;
    for (int i = 0; i < 5; i++)
        ws.threads[i].is_blocked = true;
    for (int i = 5; i < 10; i++) {
        ws.threads[i].is_exited = true;
        ws.threads[i].is_active = false;
    }
    for (int i = 10; i < 12; i++)
        ws.threads[i].is_active = false;
    REQUIRE(ws.count_schedulable_lanes() == 20);
}

TEST_CASE("C5: is_all_exited", "[warp_state]") {
    WarpState ws;
    REQUIRE(ws.is_all_exited() == false);
    for (int i = 0; i < 31; i++)
        ws.threads[i].is_exited = true;
    REQUIRE(ws.is_all_exited() == false);
    ws.threads[31].is_exited = true;
    REQUIRE(ws.is_all_exited() == true);
}

TEST_CASE("C6: has_schedulable_threads", "[warp_state]") {
    WarpState ws;
    REQUIRE(ws.has_schedulable_threads() == true);
    for (int i = 0; i < 32; i++) {
        ws.threads[i].is_blocked = true;
        ws.threads[i].is_active = false;
    }
    REQUIRE(ws.has_schedulable_threads() == false);
}

// A2: thread_predicates and warp_pc were removed (truly dead code - 0
// production references). The C7 thread_predicates test was therefore
// deleted. wbars[] / current_wbar_id remain deprecated (see warp_state.h
// TODO comments) because barrier.cpp:145-263 still uses them.