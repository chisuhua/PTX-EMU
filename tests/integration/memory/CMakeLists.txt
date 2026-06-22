# tests/integration/memory/CMakeLists.txt
# Type-2 integration test: Symtable leak detection (Phase 2 Task 1.1)
#
# Verifies (via STATIC_REQUIRE compile-time assertions) that GPUContext::name2Sym,
# CTAContext::name2Share, and CTAContext::name2Local hold std::unique_ptr<Symtable>
# rather than raw Symtable* pointers, ensuring RAII-based memory management
# and zero leaks per kernel launch.

add_catch_test(integration_symtable_no_leak
    test_symtable_no_leak.cpp
)
set_tests_properties(integration_symtable_no_leak PROPERTIES
    LABELS "integration;memory;symtable;leak")
