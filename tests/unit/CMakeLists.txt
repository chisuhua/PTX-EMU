# tests/unit/CMakeLists.txt
# Type 1: 直接单元测试
# 直接实例化核心数据结构/类，调用方法验证行为，不经过指令执行流程

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_ARCHITECTURES "100")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep --no-compress")

set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/tests)

# ============================================================================
# Barrier 单元测试
# ============================================================================
add_catch_test(unit_barrier_module
    barrier/test_barrier_module.cpp
)
set_tests_properties(unit_barrier_module PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_warp_barrier
    barrier/test_warp_barrier.cpp
)
set_tests_properties(unit_warp_barrier PROPERTIES LABELS "unit;wbar")

add_catch_test(unit_barrier_scenarios
    barrier/test_barrier_scenarios.cpp
)
set_tests_properties(unit_barrier_scenarios PROPERTIES LABELS "unit;barrier;regression")

add_catch_test(unit_barrier_reconvergence
    barrier/test_barrier_reconvergence.cpp
)
set_tests_properties(unit_barrier_reconvergence PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_barrier_interaction
    barrier/test_barrier_interaction_integrated.cpp
)
set_tests_properties(unit_barrier_interaction PROPERTIES LABELS "unit;barrier")

# test_barrier_verification.cpp has pre-existing build error (uses <catch2/catch.hpp>)
# Not built in main branch either
# add_catch_test(unit_barrier_verification
#     barrier/test_barrier_verification.cpp
# )

# ============================================================================
# SIMT 单元测试
# ============================================================================
add_catch_test(unit_simt_stack_entry
    simt/test_simt_stack_entry.cpp
)
set_tests_properties(unit_simt_stack_entry PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_simt_stack_catch2
    simt/test_simt_stack_catch2.cpp
)
set_tests_properties(unit_simt_stack_catch2 PROPERTIES LABELS "unit;simt_stack;simt_entry")

add_catch_test(unit_simt_thread_pc
    simt/test_simt_thread_pc.cpp
)
set_tests_properties(unit_simt_thread_pc PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_handle_branch_two_level_divergence
    simt/test_handle_branch_two_level_divergence.cpp
)
set_tests_properties(unit_handle_branch_two_level_divergence PROPERTIES LABELS "unit;simt;divergence")

# ============================================================================
# Warp 单元测试
# 注: test_warp_context.cpp, test_warp_scheduler.cpp, test_sm_context.cpp
#     是 standalone 测试 (有 main 函数), 不用 add_catch_test
# ============================================================================
add_catch_test(unit_warp_state
    warp/test_warp_state.cpp
)
set_tests_properties(unit_warp_state PROPERTIES LABELS "unit;warp;state;simt_entry")

# ============================================================================
# Memory 单元测试
# 注: 原始 CMakeLists.txt 中 test_memory_manager.cpp 实际上未被编译
#     (add_catch_test 的参数解析bug, test_memory_manager.cpp 被当作 test name)
#     只有 test_parse_immediate.cpp 被编译。保留此行为。
# ============================================================================
add_catch_test(unit_memory_manager_legacy
    common/test_parse_immediate.cpp
)
set_tests_properties(unit_memory_manager_legacy PROPERTIES LABELS "unit;memory;parser")

add_catch_test(unit_memory_bounds
    memory/test_memory_bounds.cpp
)
set_tests_properties(unit_memory_bounds PROPERTIES LABELS "unit;memory")

# ============================================================================
# Exec 单元测试
# ============================================================================
add_catch_test(unit_exec_mask
    exec/test_exec_mask.cpp
)
set_tests_properties(unit_exec_mask PROPERTIES LABELS "unit;exec_mask;simt_entry")

add_catch_test(unit_active_mask_consistency
    exec/test_active_mask_consistency.cpp
)
set_tests_properties(unit_active_mask_consistency PROPERTIES LABELS "unit;active_mask")

add_catch_test(unit_exec_integration_h1_h4
    exec/test_exec_integration_h1_h4.cpp
)
set_tests_properties(unit_exec_integration_h1_h4 PROPERTIES LABELS "unit;exec")

add_catch_test(unit_exec_layer_e1_e3
    exec/test_exec_layer_e1_e3.cpp
)
set_tests_properties(unit_exec_layer_e1_e3 PROPERTIES LABELS "unit;exec")

# ============================================================================
# PC 单元测试
# ============================================================================
add_catch_test(unit_pc_management
    pc/test_pc_management.cpp
)
set_tests_properties(unit_pc_management PROPERTIES LABELS "unit;pc;scheduler")

add_catch_test(unit_pc_management_advanced
    pc/test_pc_management_advanced.cpp
)
set_tests_properties(unit_pc_management_advanced PROPERTIES LABELS "unit;simt_entry")

# ============================================================================
# Sync 单元测试
# ============================================================================
add_catch_test(unit_sync_mechanism
    sync/test_sync_mechanism.cpp
)
set_tests_properties(unit_sync_mechanism PROPERTIES LABELS "unit;sync;simt_entry")

add_catch_test(unit_barrier_active_mask_preserved
    sync/test_barrier_active_mask_preserved.cpp
)
set_tests_properties(unit_barrier_active_mask_preserved PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_barrier_pc_overwrite
    sync/test_barrier_pc_overwrite.cpp
)
set_tests_properties(unit_barrier_pc_overwrite PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_bsync_state
    sync/test_bsync_state.cpp
)
set_tests_properties(unit_bsync_state PROPERTIES LABELS "unit;bsync")

# ============================================================================
# Common 单元测试
# ============================================================================
# test_cc_register.cpp has pre-existing build errors (subc_handler undeclared)
# Not built in main branch either
# add_catch_test(unit_cc_register
#     common/test_cc_register.cpp
# )

add_catch_test(unit_specific_bugs_unit
    common/test_specific_bugs_unit.cpp
)
set_tests_properties(unit_specific_bugs_unit PROPERTIES LABELS "unit;regression")

add_catch_test(unit_scheduler_config
    common/test_scheduler_config.cpp
)
set_tests_properties(unit_scheduler_config PROPERTIES LABELS "unit;scheduler")

add_catch_test(unit_parse_immediate
    common/test_parse_immediate.cpp
)
set_tests_properties(unit_parse_immediate PROPERTIES LABELS "unit;parser")

add_catch_test(unit_warp_trace_formatter
    common/test_warp_trace_formatter.cpp
)
set_tests_properties(unit_warp_trace_formatter PROPERTIES LABELS "unit;warp_trace_formatter")

# ============================================================================
# PTX 指令单元测试
# ============================================================================
add_catch_test(unit_ptx_integer
    ptx/test_ptx_integer.cu
    ptx/ptx_integer_arith.cu
)
set_tests_properties(unit_ptx_integer PROPERTIES LABELS "unit;ptx;integer")

add_catch_test(unit_ptx_float
    ptx/test_ptx_float.cu
    ptx/ptx_float_arith.cu
)
set_tests_properties(unit_ptx_float PROPERTIES LABELS "unit;ptx;float")

add_catch_test(unit_ptx_extended
    ptx/test_ptx_extended.cu
    ptx/ptx_extended_prec.cu
)
set_tests_properties(unit_ptx_extended PROPERTIES LABELS "unit;ptx;integer")

add_catch_test(unit_ptx_bitwise
    ptx/test_ptx_bitwise.cu
    ptx/ptx_bitwise_shift.cu
)
set_tests_properties(unit_ptx_bitwise PROPERTIES LABELS "unit;ptx;bitwise")

add_catch_test(unit_ptx_cvt
    ptx/test_ptx_cvt.cu
    ptx/ptx_cvt_arith.cu
)
set_tests_properties(unit_ptx_cvt PROPERTIES LABELS "unit;ptx;cvt")

add_catch_test(unit_ptx_ld_st
    ptx/test_ptx_ld_st.cu
    ptx/ptx_ld_st.cu
)
set_tests_properties(unit_ptx_ld_st PROPERTIES LABELS "unit;ptx;ld_st")

add_catch_test(unit_ptx_cvta
    ptx/test_ptx_cvta.cu
    ptx/ptx_cvta.cu
)
set_tests_properties(unit_ptx_cvta PROPERTIES LABELS "unit;ptx;cvta")
