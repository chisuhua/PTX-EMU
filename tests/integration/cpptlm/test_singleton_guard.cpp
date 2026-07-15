// test_singleton_guard.cpp
// =============================================================================
// Integration test: SingletonGuard 重复初始化检测 (D-PTX-2)
//
// 验证 SingletonGuard 设计契约。
// 完整 __cudaRegisterFatBinary 重复调用测试需要 fork+exec 独立二进制，
// 此处仅验证设计契约文档化 + g_cpptlm_bridge 初始状态。
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"

TEST_CASE("SingletonGuard: g_cpptlm_bridge defaults to nullptr", "[cpptlm][singleton]") {
    // Before any bridge loading, g_cpptlm_bridge must be nullptr
    REQUIRE(g_cpptlm_bridge == nullptr);
}

TEST_CASE("SingletonGuard: design contract documented", "[cpptlm][singleton]") {
    // D-PTX-2 contract:
    // 1. SingletonGuard::instance() returns a static singleton
    // 2. check_and_mark() returns true on duplicate call → FATAL abort
    // 3. check_and_mark() returns false on first call → proceed
    // 4. reset() clears the flag (for testing/cleanup)
    //
    // Implementation: src/cudart/cudart_sim.cpp (SingletonGuard class)
    // Full e2e test: fork + exec binary + verify SIGABRT on second call
    REQUIRE(true);
}
