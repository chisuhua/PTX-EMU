// tests/integration/async/test_tc_queue_with_cta_context.cpp
// Phase 0.5.4 (Fix #9d): CTAContext TcQueue integration test.
//
// Verifies that CTAContext exposes a per-CTA TcQueue via the tc_queue()
// accessor, that default-init sets counter=0, commit is monotonic, clear
// resets state, and independent CTAContext instances have isolated queues.
//
// TcQueue API (from c0fa43f src/ptxsim/async/tc_queue.h):
//   - commit(group_id)  — monotonic CAS fetch_max
//   - wait(warp, lane, group_id) — blocks lane, stores (pc+1) completion
//   - clear()           — reset counter + pending waiters
//   - current_counter() — current commit group counter
//   - pending_count()   — number of pending waiters

#include "catch_amalgamated.hpp"

#include "ptxsim/async/tc_queue.h"
#include "ptxsim/cta_context.h"

TEST_CASE("Fix #9d: CTAContext TcQueue integration — default constructor",
          "[integration][tc_queue][cta]") {
    CTAContext cta;

    SECTION("Default-init counter is zero") {
        REQUIRE(cta.tc_queue().current_counter() == 0);
    }

    SECTION("Default-init pending count is zero") {
        REQUIRE(cta.tc_queue().pending_count() == 0);
    }
}

TEST_CASE("Fix #9d: CTAContext TcQueue integration — commit monotonic",
          "[integration][tc_queue][cta]") {
    CTAContext cta;

    cta.tc_queue().commit(5);
    REQUIRE(cta.tc_queue().current_counter() == 5);

    cta.tc_queue().commit(3);
    REQUIRE(cta.tc_queue().current_counter() == 5);

    cta.tc_queue().commit(10);
    REQUIRE(cta.tc_queue().current_counter() == 10);
}

TEST_CASE("Fix #9d: CTAContext TcQueue integration — clear",
          "[integration][tc_queue][cta]") {
    CTAContext cta;

    cta.tc_queue().commit(10);
    REQUIRE(cta.tc_queue().current_counter() == 10);

    cta.tc_queue().clear();
    REQUIRE(cta.tc_queue().current_counter() == 0);
    REQUIRE(cta.tc_queue().pending_count() == 0);

    cta.tc_queue().commit(3);
    REQUIRE(cta.tc_queue().current_counter() == 3);
}

TEST_CASE("Fix #9d: Two CTAContext instances have independent TcQueue",
          "[integration][tc_queue][cta][isolation]") {
    CTAContext cta1;
    CTAContext cta2;

    cta1.tc_queue().commit(10);
    REQUIRE(cta1.tc_queue().current_counter() == 10);
    REQUIRE(cta2.tc_queue().current_counter() == 0);

    cta2.tc_queue().commit(20);
    REQUIRE(cta1.tc_queue().current_counter() == 10);
    REQUIRE(cta2.tc_queue().current_counter() == 20);

    cta1.tc_queue().clear();
    REQUIRE(cta1.tc_queue().current_counter() == 0);
    REQUIRE(cta2.tc_queue().current_counter() == 20);
}

TEST_CASE("Fix #9d: CTAContext TcQueue integration — const accessor",
          "[integration][tc_queue][cta][const]") {
    CTAContext cta;
    cta.tc_queue().commit(7);

    const CTAContext& const_cta = cta;
    REQUIRE(const_cta.tc_queue().current_counter() == 7);
    REQUIRE(const_cta.tc_queue().pending_count() == 0);
}