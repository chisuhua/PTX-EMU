/**
 * PTX-6 TDD: compile-only smoke test for memory_test_utils.h.
 *
 * Locks the header-visibility contract for reduce-memory-test-utils-includes-v2:
 * the header must remain self-sufficient after include reduction.
 * Zero behavior change; signature visibility is the only assertion.
 *
 * Ref: openspec/changes/reduce-memory-test-utils-includes-v2/design.md
 */
#include "catch_amalgamated.hpp"
#include "ptxsim/testing/memory_test_utils.h"

namespace ptu = ptxsim::testing;

TEST_CASE("memory_test_utils.h: factory init signature is reachable",
          "[unit][memory][include_smoke]") {
    // Signature lock: take a function pointer (do not invoke — invoking
    // triggers global state mutation, which violates Type 1 unit boundaries).
    auto fp = &ptu::init_instruction_factory_once;
    REQUIRE(fp != nullptr);
}

TEST_CASE("memory_test_utils.h: shared/local decl helpers compile",
          "[unit][memory][include_smoke]") {
    auto shared = ptu::make_shared_decl("buf", 16);
    CHECK(shared.type == S_SHARED);

    auto local = ptu::make_local_decl("lbuf", 8);
    CHECK(local.type == S_LOCAL);
}

TEST_CASE("memory_test_utils.h: addr-based ld/st helpers compile",
          "[unit][memory][include_smoke]") {
    auto st_shared = ptu::make_st_shared_addr("buf", "%r1", "%r2");
    CHECK(st_shared.type == S_ST);

    auto st_local = ptu::make_st_local_addr("lbuf", "%r1", "%r2");
    CHECK(st_local.type == S_ST);

    auto ld_shared = ptu::make_ld_shared_addr("%rd", "buf", "%r1");
    CHECK(ld_shared.type == S_LD);

    auto ld_local = ptu::make_ld_local_addr("%rd", "lbuf", "%r1");
    CHECK(ld_local.type == S_LD);
}

TEST_CASE("memory_test_utils.h: qualifier-overloaded helpers compile",
          "[unit][memory][include_smoke]") {
    auto ld_q = ptu::make_ld_shared_addr("%rd", "buf", "%r1", Qualifier::Q_B16);
    CHECK(ld_q.type == S_LD);

    auto st_q = ptu::make_st_shared_addr("buf", "%r1", "%r2", Qualifier::Q_B16);
    CHECK(st_q.type == S_ST);

    auto shared_q = ptu::make_shared_decl("buf16", 8, Qualifier::Q_B16);
    CHECK(shared_q.type == S_SHARED);
}

TEST_CASE("memory_test_utils.h: vector ld/st helpers compile",
          "[unit][memory][include_smoke]") {
    auto ld_v2 = ptu::make_ld_shared_addr_v2("%r1", "%r2", "buf", "%off");
    CHECK(ld_v2.type == S_LD);

    auto st_v2 = ptu::make_st_shared_addr_v2("buf", "%off", "%r1", "%r2");
    CHECK(st_v2.type == S_ST);

    auto ld_v4 = ptu::make_ld_shared_addr_v4("%r1", "%r2", "%r3", "%r4", "buf", "%off");
    CHECK(ld_v4.type == S_LD);

    auto st_v4 = ptu::make_st_shared_addr_v4("buf", "%off", "%r1", "%r2", "%r3", "%r4");
    CHECK(st_v4.type == S_ST);
}

TEST_CASE("memory_test_utils.h: setp comparison helpers compile",
          "[unit][memory][include_smoke]") {
    CHECK(ptu::make_setp_eq("%p", "%a", "%b").type == S_SETP);
    CHECK(ptu::make_setp_ne("%p", "%a", "%b").type == S_SETP);
    CHECK(ptu::make_setp_gt("%p", "%a", "%b").type == S_SETP);
    CHECK(ptu::make_setp_ge("%p", "%a", "%b").type == S_SETP);
    CHECK(ptu::make_setp_le("%p", "%a", "%b").type == S_SETP);

    CHECK(ptu::make_setp_eq_imm("%p", "%a", 0).type == S_SETP);
    CHECK(ptu::make_setp_ne_imm("%p", "%a", 0).type == S_SETP);
    CHECK(ptu::make_setp_lt_imm("%p", "%a", 0).type == S_SETP);
    CHECK(ptu::make_setp_gt_imm("%p", "%a", 0).type == S_SETP);
    CHECK(ptu::make_setp_le_imm("%p", "%a", 0).type == S_SETP);
    CHECK(ptu::make_setp_ge_imm("%p", "%a", 0).type == S_SETP);
}

TEST_CASE("memory_test_utils.h: mov_b64 vec src helper compiles",
          "[unit][memory][include_smoke]") {
    auto mov = ptu::make_mov_b64_vec_src("%rd", {"%r1", "%r2"});
    CHECK(mov.type == S_MOV);
}