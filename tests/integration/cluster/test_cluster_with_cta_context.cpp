// tests/integration/cluster/test_cluster_with_cta_context.cpp
// Phase 0.5.3 (Fix #9c): CTAContext cluster integration test.
//
// Verifies that CTAContext exposes a per-CTA ClusterContext reference via
// init_cluster_context() + cluster_context() accessor. ClusterContext has
// an explicit constructor (root_id, num_ctas) unlike TmaDescriptorStore/Tmem
// (which have default ctors), so lazy-init via std::optional is required.
//
// Test coverage:
//   - Single-CTA cluster arrive/wait through CTAContext accessor
//   - has_cluster_context() pre-condition check (false before init)
//   - Access before init throws (std::bad_optional_access)
//   - Multi-CTA blocking synchronization via std::thread
//   - Const accessor correctness
//   - Full cluster size-8 arrive/wait

#include "catch_amalgamated.hpp"

#include <cstdint>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <thread>

#include "ptxsim/cluster/cluster_context.h"
#include "ptxsim/cta_context.h"

using cta_id_t = ClusterContext::cta_id_t;
using cluster_size_t = ClusterContext::cluster_size_t;

// ---------------------------------------------------------------------------
// 1. Single-CTA cluster arrive/wait through CTAContext accessor
// ---------------------------------------------------------------------------
TEST_CASE("Fix #9c: Single-CTA cluster arrive/wait through CTAContext",
          "[integration][cluster][cta]") {
    CTAContext cta;
    REQUIRE_FALSE(cta.has_cluster_context());

    cta.init_cluster_context(0, 1);
    REQUIRE(cta.has_cluster_context());

    ClusterContext& cl = cta.cluster_context();
    REQUIRE(cl.size() == 1);

    cl.cta_cluster_arrive(0);
    cl.cta_cluster_wait(0);
}

// ---------------------------------------------------------------------------
// 2. has_cluster_context() pre-condition check
// ---------------------------------------------------------------------------
TEST_CASE("Fix #9c: has_cluster_context pre-condition check",
          "[integration][cluster][cta]") {
    CTAContext cta;
    REQUIRE_FALSE(cta.has_cluster_context());

    cta.init_cluster_context(0, 3);
    REQUIRE(cta.has_cluster_context());

    ClusterContext& cl = cta.cluster_context();
    REQUIRE(cl.size() == 3);
    REQUIRE(cl.validate_cta_id(0));
    REQUIRE(cl.validate_cta_id(2));
    REQUIRE_FALSE(cl.validate_cta_id(3));
}

// ---------------------------------------------------------------------------
// 3. Access uninitialized cluster_context throws
// ---------------------------------------------------------------------------
TEST_CASE("Fix #9c: Access uninitialized cluster_context throws",
          "[integration][cluster][cta][error]") {
    CTAContext cta;
    REQUIRE_FALSE(cta.has_cluster_context());
    REQUIRE_THROWS_AS(cta.cluster_context(), std::bad_optional_access);
}

// ---------------------------------------------------------------------------
// 4. Multi-CTA blocking synchronization via single CTAContext
//    (simulating one CTA's view of a multi-CTA cluster via arrive/wait)
// ---------------------------------------------------------------------------
TEST_CASE("Fix #9c: Multi-CTA blocking arrive/wait through CTAContext",
          "[integration][cluster][cta][blocking]") {
    CTAContext cta;
    cta.init_cluster_context(0, 4);
    ClusterContext& cl = cta.cluster_context();

    cl.cta_cluster_arrive(0);
    cl.cta_cluster_arrive(1);
    cl.cta_cluster_arrive(2);

    bool thread_wait_done = false;
    std::thread waiter([&]() {
        cl.cta_cluster_wait(0);
        thread_wait_done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE_FALSE(thread_wait_done);

    cl.cta_cluster_arrive(3);

    waiter.join();
    REQUIRE(thread_wait_done);

    cl.cta_cluster_wait(3);
}

// ---------------------------------------------------------------------------
// 5. Const accessor correctness
// ---------------------------------------------------------------------------
TEST_CASE("Fix #9c: const cluster_context accessor",
          "[integration][cluster][cta][const]") {
    CTAContext cta;
    cta.init_cluster_context(0, 1);
    cta.cluster_context().cta_cluster_arrive(0);

    const CTAContext& const_cta = cta;
    const ClusterContext& const_cl = const_cta.cluster_context();

    REQUIRE(const_cl.size() == 1);
    REQUIRE(const_cl.validate_cta_id(0));
    REQUIRE_FALSE(const_cl.validate_cta_id(1));
}

// ---------------------------------------------------------------------------
// 6. Cluster size 8: full-scale arrive/wait through CTAContext
// ---------------------------------------------------------------------------
TEST_CASE("Fix #9c: Full cluster size-8 arrive/wait through CTAContext",
          "[integration][cluster][cta][max]") {
    CTAContext cta;
    cta.init_cluster_context(0, 8);
    ClusterContext& cl = cta.cluster_context();

    for (cta_id_t i = 0; i < 8; ++i) {
        cl.cta_cluster_arrive(i);
    }
    for (cta_id_t i = 0; i < 8; ++i) {
        cl.cta_cluster_wait(i);
    }
}
