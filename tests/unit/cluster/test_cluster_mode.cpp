// tests/unit/cluster/test_cluster_mode.cpp
// Phase 0.3 (Fix #7): cluster arrive/wait synchronization primitives unit tests.
//
// ≥10 TEST_CASEs cover constructor validation, single-CTA arrive/wait,
// multi-CTA blocking synchronization, duplicate arrive rejection,
// wait-before-arrive rejection, cross-cluster isolation.
// Consumed by cta_group::1 (Phase 1-3) — no distributed_smem per Oracle simplification.

#include "catch_amalgamated.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include "ptxsim/cluster/cluster_context.h"

using cta_id_t = ClusterContext::cta_id_t;
using cluster_size_t = ClusterContext::cluster_size_t;

// ---------------------------------------------------------------------------
// 1. construct_cluster_size_1
// ---------------------------------------------------------------------------
TEST_CASE("construct_cluster_size_1", "[cluster][construct]") {
    ClusterContext cluster(0, 1);
    REQUIRE(cluster.size() == 1);
}

// ---------------------------------------------------------------------------
// 2. construct_cluster_size_8
// ---------------------------------------------------------------------------
TEST_CASE("construct_cluster_size_8", "[cluster][construct]") {
    ClusterContext cluster(0, 8);
    REQUIRE(cluster.size() == 8);
}

// ---------------------------------------------------------------------------
// 3. construct_invalid_size_zero_throws
// ---------------------------------------------------------------------------
TEST_CASE("construct_invalid_size_zero_throws", "[cluster][construct][error]") {
    REQUIRE_THROWS_AS(ClusterContext(0, 0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 4. construct_invalid_size_9_throws
// ---------------------------------------------------------------------------
TEST_CASE("construct_invalid_size_9_throws", "[cluster][construct][error]") {
    REQUIRE_THROWS_AS(ClusterContext(0, 9), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 5. construct_invalid_root_id_8_throws
// ---------------------------------------------------------------------------
TEST_CASE("construct_invalid_root_id_8_throws", "[cluster][construct][error]") {
    REQUIRE_THROWS_AS(ClusterContext(8, 4), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 6. validate_cta_id_in_range
// ---------------------------------------------------------------------------
TEST_CASE("validate_cta_id_in_range", "[cluster][validate]") {
    ClusterContext cluster(0, 4);
    REQUIRE(cluster.validate_cta_id(0));
    REQUIRE(cluster.validate_cta_id(3));
    REQUIRE_FALSE(cluster.validate_cta_id(4));
    REQUIRE_FALSE(cluster.validate_cta_id(7));
}

// ---------------------------------------------------------------------------
// 7. arrive_then_wait_single_cta_immediate
// ---------------------------------------------------------------------------
TEST_CASE("arrive_then_wait_single_cta_immediate", "[cluster][sync]") {
    ClusterContext cluster(0, 1);
    cluster.cta_cluster_arrive(0);
    cluster.cta_cluster_wait(0);
}

// ---------------------------------------------------------------------------
// 8. arrive_multiple_peer_ctas_wait_blocks_until_all
// ---------------------------------------------------------------------------
TEST_CASE("arrive_multiple_peer_ctas_wait_blocks_until_all", "[cluster][sync][blocking]") {
    ClusterContext cluster(0, 4);

    cluster.cta_cluster_arrive(0);
    cluster.cta_cluster_arrive(1);
    cluster.cta_cluster_arrive(2);

    bool thread_wait_done = false;
    std::thread waiter([&]() {
        cluster.cta_cluster_wait(0);
        thread_wait_done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE_FALSE(thread_wait_done);

    cluster.cta_cluster_arrive(3);

    waiter.join();
    REQUIRE(thread_wait_done);

    cluster.cta_cluster_wait(3);
}

// ---------------------------------------------------------------------------
// 9. wait_before_arrive_throws
// ---------------------------------------------------------------------------
TEST_CASE("wait_before_arrive_throws", "[cluster][sync][error]") {
    ClusterContext cluster(0, 2);
    REQUIRE_THROWS_AS(cluster.cta_cluster_wait(0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 10. arrive_invalid_cta_id_throws
// ---------------------------------------------------------------------------
TEST_CASE("arrive_invalid_cta_id_throws", "[cluster][sync][error]") {
    ClusterContext cluster(0, 2);
    REQUIRE_THROWS_AS(cluster.cta_cluster_arrive(2), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 11. wait_invalid_cta_id_throws
// ---------------------------------------------------------------------------
TEST_CASE("wait_invalid_cta_id_throws", "[cluster][sync][error]") {
    ClusterContext cluster(0, 2);
    cluster.cta_cluster_arrive(0);
    REQUIRE_THROWS_AS(cluster.cta_cluster_wait(2), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 12. multiple_waits_after_all_arrived_succeeds
// ---------------------------------------------------------------------------
TEST_CASE("multiple_waits_after_all_arrived_succeeds", "[cluster][sync]") {
    ClusterContext cluster(0, 2);
    cluster.cta_cluster_arrive(0);
    cluster.cta_cluster_arrive(1);
    cluster.cta_cluster_wait(0);
    cluster.cta_cluster_wait(1);
}

// ---------------------------------------------------------------------------
// 13. duplicate_arrive_throws
// ---------------------------------------------------------------------------
TEST_CASE("duplicate_arrive_throws", "[cluster][sync][error]") {
    ClusterContext cluster(0, 2);
    cluster.cta_cluster_arrive(0);
    REQUIRE_THROWS_AS(cluster.cta_cluster_arrive(0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 14. cross_cluster_isolation
// ---------------------------------------------------------------------------
TEST_CASE("cross_cluster_isolation", "[cluster][sync][isolation]") {
    ClusterContext cluster_a(0, 4);
    ClusterContext cluster_b(0, 3);

    cluster_a.cta_cluster_arrive(0);
    cluster_a.cta_cluster_arrive(1);
    cluster_a.cta_cluster_arrive(2);

    cluster_b.cta_cluster_arrive(0);
    cluster_b.cta_cluster_arrive(1);

    bool cluster_b_done = false;
    std::thread waiter_b([&]() {
        cluster_b.cta_cluster_wait(0);
        cluster_b_done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // cluster_a fully arriving should NOT unblock cluster_b's wait
    cluster_a.cta_cluster_arrive(3);
    cluster_a.cta_cluster_wait(0);
    REQUIRE_FALSE(cluster_b_done);

    // Now fully arrive cluster_b
    cluster_b.cta_cluster_arrive(2);
    waiter_b.join();
    REQUIRE(cluster_b_done);

    cluster_b.cta_cluster_wait(2);
}

// ---------------------------------------------------------------------------
// 15. arrive_and_wait_all_ctas_cluster_size_8
// ---------------------------------------------------------------------------
TEST_CASE("arrive_and_wait_all_ctas_cluster_size_8", "[cluster][sync][max]") {
    ClusterContext cluster(0, 8);
    for (cta_id_t i = 0; i < 8; ++i) {
        cluster.cta_cluster_arrive(i);
    }
    for (cta_id_t i = 0; i < 8; ++i) {
        cluster.cta_cluster_wait(i);
    }
}