#include "catch_amalgamated.hpp"
#include "ptxsim/bsync_state.h"

using namespace ptxsim;

TEST_CASE("BsyncManager bssy creates new barrier state", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0xF);
    REQUIRE(mgr.get_state(1) != nullptr);
    REQUIRE(mgr.get_state(1)->total_threads == 4);
}

TEST_CASE("BsyncManager bsync marks thread as waiting", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0xF);
    bool result = mgr.bsync(1, 0, 100);
    REQUIRE(result == true);
    REQUIRE(mgr.is_waiting(1, 0) == true);
}

TEST_CASE("BsyncManager check_release returns true when all arrived", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0x3);
    mgr.bsync(1, 0, 100);
    mgr.bsync(1, 1, 100);
    REQUIRE(mgr.check_release(1) == true);
}

TEST_CASE("BsyncManager release clears waiting mask", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0x3);
    mgr.bsync(1, 0, 100);
    mgr.release(1);
    REQUIRE(mgr.get_state(1)->is_released == true);
}

TEST_CASE("BsyncManager bssy with multiple threads", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(5, 0xFF);
    auto* state = mgr.get_state(5);
    REQUIRE(state != nullptr);
    REQUIRE(state->total_threads == 8);
    REQUIRE(state->waiting_threads_mask == 0);
    REQUIRE(state->is_released == false);
}

TEST_CASE("BsyncManager is_waiting returns false for non-waiting lane", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0xF);
    mgr.bsync(1, 0, 100);
    REQUIRE(mgr.is_waiting(1, 1) == false);
    REQUIRE(mgr.is_waiting(1, 2) == false);
}

TEST_CASE("BsyncManager get_waiting_mask reflects arrived threads", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0xF);
    mgr.bsync(1, 0, 100);
    mgr.bsync(1, 2, 100);
    REQUIRE(mgr.get_waiting_mask(1) == 0x5);
}

TEST_CASE("BsyncManager check_release returns false when not all arrived", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0xF);
    mgr.bsync(1, 0, 100);
    REQUIRE(mgr.check_release(1) == false);
}

TEST_CASE("BsyncManager cleanup removes released barriers", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0x3);
    mgr.bsync(1, 0, 100);
    mgr.bsync(1, 1, 100);
    mgr.release(1);
    REQUIRE(mgr.size() == 1);
    mgr.cleanup();
    REQUIRE(mgr.size() == 0);
}

TEST_CASE("BsyncManager reset clears all barriers", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    mgr.bssy(1, 0xF);
    mgr.bssy(2, 0x3);
    REQUIRE(mgr.size() == 2);
    mgr.reset();
    REQUIRE(mgr.size() == 0);
}

TEST_CASE("BsyncManager bsync returns false for non-existent barrier", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    bool result = mgr.bsync(999, 0, 100);
    REQUIRE(result == false);
}

TEST_CASE("BsyncManager get_state returns nullptr for non-existent barrier", "[bsync][bsync_manager]") {
    BsyncManager mgr;
    REQUIRE(mgr.get_state(999) == nullptr);
}
