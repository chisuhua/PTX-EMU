// Unit tests for T2-3 POD split — memory_ref.h
// Verifies that MemoryPod aggregates per-thread memory-space and back-pointer
// fields without behavior.

#include "ptxsim/contexts/memory_ref.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::MemoryPod;

TEST_CASE("MemoryPod: default memory pointers are nullptr",
          "[contexts][pod][memory_ref]") {
    MemoryPod pod;
    REQUIRE(pod.shared_mem_space == nullptr);
    REQUIRE(pod.local_mem_space == nullptr);
}

TEST_CASE("MemoryPod: default back-pointers are nullptr",
          "[contexts][pod][memory_ref]") {
    MemoryPod pod;
    REQUIRE(pod.warp_context_ == nullptr);
    REQUIRE(pod.cta_context_ == nullptr);
}

TEST_CASE("MemoryPod: memory-space pointers can be assigned",
          "[contexts][pod][memory_ref]") {
    MemoryPod pod;
    int dummy_shared = 0;
    int dummy_local = 0;
    pod.shared_mem_space = &dummy_shared;
    pod.local_mem_space = &dummy_local;
    REQUIRE(pod.shared_mem_space == &dummy_shared);
    REQUIRE(pod.local_mem_space == &dummy_local);
}