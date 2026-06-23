// Unit tests for T2-3 POD split — program_ref.h
// Verifies that ProgramRefPod aggregates per-thread program-state references
// (statements, symbol tables, label map, call stack) without behavior.

#include "ptxsim/contexts/program_ref.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::ProgramRefPod;

TEST_CASE("ProgramRefPod: default pointers are nullptr",
          "[contexts][pod][program_ref]") {
    ProgramRefPod pod;
    REQUIRE(pod.statements == nullptr);
    REQUIRE(pod.name2Sym == nullptr);
    REQUIRE(pod.name2Share == nullptr);
}

TEST_CASE("ProgramRefPod: default label2pc is empty",
          "[contexts][pod][program_ref]") {
    ProgramRefPod pod;
    REQUIRE(pod.label2pc.empty());
}

TEST_CASE("ProgramRefPod: default call_stack is empty",
          "[contexts][pod][program_ref]") {
    ProgramRefPod pod;
    REQUIRE(pod.call_stack.empty());
}

TEST_CASE("ProgramRefPod: label2pc can be populated",
          "[contexts][pod][program_ref]") {
    ProgramRefPod pod;
    pod.label2pc["L1"] = 5;
    pod.label2pc["L2"] = 10;
    REQUIRE(pod.label2pc["L1"] == 5);
    REQUIRE(pod.label2pc["L2"] == 10);
}

TEST_CASE("ProgramRefPod: call_stack can push/pop",
          "[contexts][pod][program_ref]") {
    ProgramRefPod pod;
    pod.call_stack.push(7);
    pod.call_stack.push(13);
    REQUIRE(pod.call_stack.top() == 13);
    pod.call_stack.pop();
    REQUIRE(pod.call_stack.top() == 7);
}