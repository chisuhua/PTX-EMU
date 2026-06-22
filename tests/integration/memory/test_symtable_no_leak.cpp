// test_symtable_no_leak.cpp
// =============================================================================
// Integration test (类型二): 验证 Symtable 跨 kernel launch 不累积泄漏
//
// TDD RED PHASE (Phase 2 Task 1.1):
//   验证 name2Sym / name2Share / name2Local 的 value 类型已迁移到
//   std::unique_ptr<Symtable>，即源代码中已无裸 `new Symtable()` 持有。
//
// 验证手段:
//   1. 编译期 static_assert:
//        KernelLaunchRequest::name2Sym 的 value 类型必须是
//        std::unique_ptr<Symtable>（否则 Symtable* 容器析构时不 delete → 泄漏）
//   2. 编译期 static_assert:
//        CTAContext::name2Share / name2Local 的 value 类型必须是
//        std::unique_ptr<Symtable>
//
// 注：真正的泄漏检测由 ASan 在 Sub-task 1.5 验证（必须为 0）。
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/cta_context.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/thread_context.h"

#include <map>
#include <memory>
#include <string>
#include <type_traits>

namespace {

// 编译期验证 value 类型为 std::unique_ptr<Symtable>。
template <typename T>
struct is_unique_ptr_to_symtable : std::false_type {};

template <>
struct is_unique_ptr_to_symtable<std::unique_ptr<Symtable>> : std::true_type {};

} // namespace

TEST_CASE("KernelLaunchRequest::name2Sym uses unique_ptr<Symtable> (compile-time)",
          "[integration][memory][symtable][leak]") {
    // name2Sym = std::shared_ptr<std::map<std::string, V>>
    //   V 必须为 std::unique_ptr<Symtable>。
    //   否则 owning map 析构时不 delete 其 Symtable* 值 → 每次 kernel launch 泄漏。
    //
    // 验证经过 KernelLaunchRequest::name2Sym（public, GPU 分发使用）。
    using SharedMapType = decltype(std::declval<KernelLaunchRequest>().name2Sym);
    using MapType       = typename SharedMapType::element_type;
    using ValueType     = typename MapType::value_type::second_type;

    STATIC_REQUIRE(is_unique_ptr_to_symtable<ValueType>::value);
}

TEST_CASE("CTAContext::name2Share uses unique_ptr<Symtable> (compile-time)",
          "[integration][memory][symtable][leak]") {
    using ValueType =
        decltype(std::declval<CTAContext>().name2Share)::value_type::second_type;
    STATIC_REQUIRE(is_unique_ptr_to_symtable<ValueType>::value);
}

TEST_CASE("CTAContext::name2Local uses unique_ptr<Symtable> (compile-time)",
          "[integration][memory][symtable][leak]") {
    using ValueType =
        decltype(std::declval<CTAContext>().name2Local)::value_type::second_type;
    STATIC_REQUIRE(is_unique_ptr_to_symtable<ValueType>::value);
}
