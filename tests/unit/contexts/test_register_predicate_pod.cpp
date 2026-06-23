// Unit tests for T2-3 POD split — register_predicate.h
// Verifies that RegisterPredicatePod aggregates per-thread register-bank and
// operand-collection state without behavior.

#include "ptxsim/contexts/register_predicate.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::RegisterPredicatePod;

TEST_CASE("RegisterPredicatePod: default register_bank_manager_ is null",
          "[contexts][pod][register_predicate]") {
    RegisterPredicatePod pod;
    REQUIRE(pod.register_bank_manager_ == nullptr);
}

TEST_CASE("RegisterPredicatePod: default operand collections are empty",
          "[contexts][pod][register_predicate]") {
    RegisterPredicatePod pod;
    REQUIRE(pod.operand_collected.empty());
    REQUIRE(pod.operand_is_immediate_.empty());
    REQUIRE(pod.vecOp_phy_addrs.empty());
}

TEST_CASE("RegisterPredicatePod: default dst_operand_reg_name_ is empty",
          "[contexts][pod][register_predicate]") {
    RegisterPredicatePod pod;
    REQUIRE(pod.dst_operand_reg_name_.empty());
}

TEST_CASE("RegisterPredicatePod: can push into operand collection",
          "[contexts][pod][register_predicate]") {
    RegisterPredicatePod pod;
    int dummy = 0;
    pod.operand_collected.push_back(&dummy);
    pod.operand_is_immediate_.push_back(1);

    REQUIRE(pod.operand_collected.size() == 1);
    REQUIRE(pod.operand_is_immediate_.size() == 1);
    REQUIRE(pod.operand_is_immediate_[0] == 1);
}