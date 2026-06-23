// Unit tests for T2-3 POD split — backend_links.h
// Verifies that BackendLinksPod aggregates per-warp backend-link fields
// (register bank, parent contexts, threads, SIMT stack) without behavior.

#include "ptxsim/contexts/backend_links.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::BackendLinksPod;

TEST_CASE("BackendLinksPod: default register_bank_manager_ is null",
          "[contexts][pod][backend_links]") {
    BackendLinksPod pod;
    REQUIRE(pod.register_bank_manager_ == nullptr);
}

TEST_CASE("BackendLinksPod: default back-pointers are null",
          "[contexts][pod][backend_links]") {
    BackendLinksPod pod;
    REQUIRE(pod.sm_context_ == nullptr);
    REQUIRE(pod.cta_context_ == nullptr);
}

TEST_CASE("BackendLinksPod: default threads vector is empty",
          "[contexts][pod][backend_links]") {
    BackendLinksPod pod;
    REQUIRE(pod.threads.empty());
}

TEST_CASE("BackendLinksPod: default single_step_mode is false",
          "[contexts][pod][backend_links]") {
    BackendLinksPod pod;
    REQUIRE_FALSE(pod.single_step_mode);
}

TEST_CASE("BackendLinksPod: single_step_mode is assignable",
          "[contexts][pod][backend_links]") {
    BackendLinksPod pod;
    pod.single_step_mode = true;
    REQUIRE(pod.single_step_mode);
}