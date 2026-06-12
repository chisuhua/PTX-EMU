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

add_catch_test(unit_barrier_scenarios_integrated
    barrier/test_barrier_scenarios_integrated.cpp
)
set_tests_properties(unit_barrier_scenarios_integrated PROPERTIES LABELS "unit;barrier;regression")

# P2-1 (re-enabled 2026-06-07): was disabled per KNOWN_ISSUES.md §D1.2
# (SIMTStackEntry/simt_stack not in scope). Fixed by adding the missing
# #include "ptxsim/simt_stack.h" to test_barrier_verification.cpp.
add_catch_test(unit_barrier_verification
    barrier/test_barrier_verification.cpp
)
set_tests_properties(unit_barrier_verification PROPERTIES LABELS "unit;barrier;fence")

# ============================================================================
# SIMT 单元测试
# ============================================================================
add_catch_test(unit_simt_stack_entry
    simt/test_simt_stack_entry.cpp
)
set_tests_properties(unit_simt_stack_entry PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_simt_stack_entry_integrated
    simt/test_simt_stack_entry_integrated.cpp
)
set_tests_properties(unit_simt_stack_entry_integrated PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_simt_stack_catch2
    simt/test_simt_stack_catch2.cpp
)
set_tests_properties(unit_simt_stack_catch2 PROPERTIES LABELS "unit;simt_stack;simt_entry")

add_catch_test(unit_simt_thread_pc
    simt/test_simt_thread_pc.cpp
)
set_tests_properties(unit_simt_thread_pc PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_simt_integration
    simt/test_simt_integration.cpp
)
set_tests_properties(unit_simt_integration PROPERTIES LABELS "unit;simt")

add_catch_test(unit_handle_branch_two_level_divergence
    simt/test_handle_branch_two_level_divergence.cpp
)
set_tests_properties(unit_handle_branch_two_level_divergence PROPERTIES LABELS "unit;simt;divergence")

add_catch_test(unit_handle_branch
    simt/test_handle_branch_integration.cpp
)
set_tests_properties(unit_handle_branch PROPERTIES LABELS "unit;branch;simt_entry")

add_catch_test(unit_barrier_simt
    simt/test_barrier_simt_integration.cpp
)
set_tests_properties(unit_barrier_simt PROPERTIES LABELS "unit;barrier;simt")

# ============================================================================
# Warp 单元测试
# 注: test_warp_context.cpp, test_warp_scheduler.cpp, test_sm_context.cpp
#     是 standalone 测试 (有 main 函数), 不用 add_catch_test
# ============================================================================
add_catch_test(unit_warp_state
    warp/test_warp_state.cpp
)
set_tests_properties(unit_warp_state PROPERTIES LABELS "unit;warp;state;simt_entry")

# BUG-001: active_count staleness after WarpContext::decrement_blocked_cycles()
# drains blocked cycles. RED phase: this test MUST FAIL on unpatched code
# (active_count remains 0 → scheduler skips a schedulable warp → hang).
add_catch_test(unit_warp_blocked_decrement
    warp/test_warp_blocked_decrement.cpp
)
set_tests_properties(unit_warp_blocked_decrement PROPERTIES LABELS "unit;warp;regression;BUG-LDGLOBAL-HANG")

# ============================================================================
# Memory 单元测试
# 注: 原始 CMakeLists.txt 中 test_memory_manager.cpp 实际上未被编译
#     (legacy bug: 同一个 test_parse_immediate.cpp 被编译两次 — 一次作为
#      unit_memory_manager_legacy, 另一次作为 unit_parse_immediate)。
#     修复: 拆分为两个独立目标, 各用正确的源文件。
# ============================================================================
add_catch_test(unit_memory_manager
    memory/test_memory_manager.cpp
)
set_tests_properties(unit_memory_manager PROPERTIES LABELS "unit;memory")

add_catch_test(unit_memory_bounds
    memory/test_memory_bounds.cpp
)
set_tests_properties(unit_memory_bounds PROPERTIES LABELS "unit;memory")

# ============================================================================
# Memory Regression Tests (BUG-ST-OVERREAD, BUG-NAME2SHARE)
# RED PHASE: These tests must FAIL on unpatched code.
# ============================================================================
add_catch_test(unit_st_handler_data_size
    memory/test_st_handler_data_size.cpp
)
set_tests_properties(unit_st_handler_data_size PROPERTIES LABELS "unit;memory;regression;BUG-ST-OVERREAD")

add_catch_test(unit_get_memory_addr_shared
    memory/test_get_memory_addr_shared.cpp
)
set_tests_properties(unit_get_memory_addr_shared PROPERTIES LABELS "unit;memory;regression;BUG-NAME2SHARE")

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
# Exec Regression Tests (BUG-BRAPRED)
# RED PHASE: These tests must FAIL on unpatched code.
# ============================================================================
add_catch_test(unit_bra_pred_divergent
    exec/test_bra_pred_divergent.cpp
)
set_tests_properties(unit_bra_pred_divergent PROPERTIES LABELS "unit;exec;regression;BUG-BRAPRED")

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

add_catch_test(unit_syncthreads_direction
    sync/test_syncthreads_direction.cpp
)
set_tests_properties(unit_syncthreads_direction PROPERTIES LABELS "unit;syncthreads")

add_catch_test(unit_syncthreads_test3_repro
    sync/test_syncthreads_test3_repro.cpp
)
set_tests_properties(unit_syncthreads_test3_repro PROPERTIES LABELS "unit;syncthreads;regression")

# ============================================================================
# Logger 单元测试
# ============================================================================
add_catch_test(unit_logger_cudart_component
    logger/test_logger_cudart_component.cpp
)
set_tests_properties(unit_logger_cudart_component PROPERTIES LABELS "unit;logger;cudart")

# ============================================================================
# Common 单元测试
# ============================================================================
# P2-3 (re-enabled 2026-06-07): was disabled per KNOWN_ISSUES.md §D1.2
# (subc_handler undeclared + non-Catch2 format). Fixed by rewriting
# to Catch2 TEST_CASE format with corrected processOperation (camelCase).
add_catch_test(unit_cc_register
    common/test_cc_register.cpp
)
set_tests_properties(unit_cc_register PROPERTIES LABELS "unit;cc;addc;subc")

add_catch_test(unit_specific_bugs_unit
    common/test_specific_bugs_unit.cpp
)
set_tests_properties(unit_specific_bugs_unit PROPERTIES LABELS "unit;regression")

add_catch_test(unit_instruction_latency_table
    common/test_instruction_latency_table.cpp
)
set_tests_properties(unit_instruction_latency_table PROPERTIES LABELS "unit;ptx;latency")

add_catch_test(unit_instruction_latency_config_load
    common/test_instruction_latency_config_load.cpp
)
set_tests_properties(unit_instruction_latency_config_load PROPERTIES LABELS "unit;ptx;latency")

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
# Divergence 单元测试
# ============================================================================
add_catch_test(unit_divergence_sync_standalone_integrated
    divergence/test_divergence_sync_standalone_integrated.cpp
)
set_tests_properties(unit_divergence_sync_standalone_integrated PROPERTIES LABELS "unit;divergence;barrier_sync")

# ============================================================================
# PTX 指令单元测试
# ---------------------------------------------------------------------------
# A2 cleanup (2026-06): the 7 tests below all `<<<1, 1>>>` real GPU kernels
# and verify NVIDIA's PTX semantics — they do NOT exercise the PTX-EMU
# simulator. They were moved to tests/reference/ptx_builtin/ as a
# historical "semantic reference". Simulator-driven equivalents live in
# tests/integration/ptx/ (see integration_ptx_ld_st_shared and
# integration_ptx_integer_arith). Disabled per Oracle A2 recommendation.
# ============================================================================
# add_catch_test(unit_ptx_integer
#     ptx/test_ptx_integer.cu
#     ptx/ptx_integer_arith.cu
# )
# set_tests_properties(unit_ptx_integer PROPERTIES LABELS "unit;ptx;integer")
#
# add_catch_test(unit_ptx_float
#     ptx/test_ptx_float.cu
#     ptx/ptx_float_arith.cu
# )
# set_tests_properties(unit_ptx_float PROPERTIES LABELS "unit;ptx;float")
#
# add_catch_test(unit_ptx_extended
#     ptx/test_ptx_extended.cu
#     ptx/ptx_extended_prec.cu
# )
# set_tests_properties(unit_ptx_extended PROPERTIES LABELS "unit;ptx;integer")
#
# add_catch_test(unit_ptx_bitwise
#     ptx/test_ptx_bitwise.cu
#     ptx/ptx_bitwise_shift.cu
# )
# set_tests_properties(unit_ptx_bitwise PROPERTIES LABELS "unit;ptx;bitwise")
#
# add_catch_test(unit_ptx_cvt
#     ptx/test_ptx_cvt.cu
#     ptx/ptx_cvt_arith.cu
# )
# set_tests_properties(unit_ptx_cvt PROPERTIES LABELS "unit;ptx;cvt")
#
# add_catch_test(unit_ptx_ld_st
#     ptx/test_ptx_ld_st.cu
#     ptx/ptx_ld_st.cu
# )
# set_tests_properties(unit_ptx_ld_st PROPERTIES LABELS "unit;ptx;ld_st")
#
# add_catch_test(unit_ptx_cvta
#     ptx/test_ptx_cvta.cu
#     ptx/ptx_cvta.cu
# )
# set_tests_properties(unit_ptx_cvta PROPERTIES LABELS "unit;ptx;cvta")
