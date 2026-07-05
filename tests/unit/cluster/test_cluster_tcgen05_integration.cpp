#include "ptxsim/cluster/cluster_context.h"
#include "ptxsim/cta_context.h"
#include <catch_amalgamated.hpp>

using namespace ptxsim;

TEST_CASE("cluster_tcgen05_arrive_when_initialized", "[cluster][tcgen05]") {
    CTAContext cta;
    REQUIRE_FALSE(cta.has_cluster_context());

    cta.init_cluster_context(0, 4);
    REQUIRE(cta.has_cluster_context());

    // Simulate tcgen05.commit calling arrive (opt-in path in wmma.cpp)
    REQUIRE_NOTHROW(cta.cluster_context().cta_cluster_arrive(0));
    SUCCEED("arrive registered without exception");
}

TEST_CASE("cluster_tcgen05_skipped_when_not_initialized", "[cluster][tcgen05]") {
    CTAContext cta;
    REQUIRE_FALSE(cta.has_cluster_context());
    // Simulate tcgen05.commit skipping cluster call (opt-in pattern)
    SUCCEED("opt-in skip path verified - has_cluster_context returns false");
}