// test_is_global_space_walk.cpp
// =============================================================================
// Unit test: is_global_space() traverses entire qualifier list (Lessons Learned #1)
//
// Verifies that getAddressSpace() traverses ALL qualifiers to find the
// memory space qualifier (Q_GLOBAL, Q_SHARED, Q_LOCAL, etc.).
// Based on spec: cpptlm-d1-full "is_global_space() 遍历整个 qualifier 列表"
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"

TEST_CASE("getAddressSpace: GLOBAL qualifier in various positions", "[unit][qualifier][global_space]") {
    SECTION("GLOBAL alone") {
        std::vector<Qualifier> q = {Qualifier::Q_GLOBAL};
        REQUIRE(getAddressSpace(q) == MemorySpace::GLOBAL);
    }

    SECTION("GLOBAL after type qualifier") {
        std::vector<Qualifier> q = {Qualifier::Q_F32, Qualifier::Q_GLOBAL};
        REQUIRE(getAddressSpace(q) == MemorySpace::GLOBAL);
    }

    SECTION("GLOBAL before type qualifier") {
        std::vector<Qualifier> q = {Qualifier::Q_GLOBAL, Qualifier::Q_B32};
        REQUIRE(getAddressSpace(q) == MemorySpace::GLOBAL);
    }
}

TEST_CASE("getAddressSpace: SHARED qualifier", "[unit][qualifier][global_space]") {
    std::vector<Qualifier> q = {Qualifier::Q_F32, Qualifier::Q_SHARED};
    REQUIRE(getAddressSpace(q) == MemorySpace::SHARED);
}

TEST_CASE("getAddressSpace: LOCAL qualifier", "[unit][qualifier][global_space]") {
    std::vector<Qualifier> q = {Qualifier::Q_S32, Qualifier::Q_LOCAL};
    REQUIRE(getAddressSpace(q) == MemorySpace::LOCAL);
}

TEST_CASE("getAddressSpace: CONST qualifier", "[unit][qualifier][global_space]") {
    std::vector<Qualifier> q = {Qualifier::Q_CONST, Qualifier::Q_B32};
    REQUIRE(getAddressSpace(q) == MemorySpace::CONST);
}

TEST_CASE("getAddressSpace: PARAM qualifier", "[unit][qualifier][global_space]") {
    std::vector<Qualifier> q = {Qualifier::Q_PARAM, Qualifier::Q_B64};
    REQUIRE(getAddressSpace(q) == MemorySpace::PARAM);
}

TEST_CASE("getAddressSpace: no space qualifier defaults to GLOBAL", "[unit][qualifier][global_space]") {
    std::vector<Qualifier> q = {Qualifier::Q_F32, Qualifier::Q_RN};
    REQUIRE(getAddressSpace(q) == MemorySpace::GLOBAL);
}

TEST_CASE("getAddressSpace: empty qualifier list defaults to GLOBAL", "[unit][qualifier][global_space]") {
    std::vector<Qualifier> q;
    REQUIRE(getAddressSpace(q) == MemorySpace::GLOBAL);
}

TEST_CASE("getAddressSpace: traverses all qualifiers (Lessons Learned #1)", "[unit][qualifier][global_space][lessons]") {
    // Lessons Learned #1: 跨模块间接状态翻译 — 迁移函数时漏掉看似冗余的
    // qualifier 遍历检查。验证 getAddressSpace 遍历整个 qualifier 列表。
    std::vector<Qualifier> mixed = {
        Qualifier::Q_F32, Qualifier::Q_RN,
        Qualifier::Q_GLOBAL, Qualifier::Q_B32
    };
    REQUIRE(getAddressSpace(mixed) == MemorySpace::GLOBAL);

    // SHARED qualifier found via full traversal
    std::vector<Qualifier> mixed_shared = {
        Qualifier::Q_S32, Qualifier::Q_WIDE, Qualifier::Q_SHARED
    };
    REQUIRE(getAddressSpace(mixed_shared) == MemorySpace::SHARED);
}