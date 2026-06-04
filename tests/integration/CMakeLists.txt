# tests/integration/CMakeLists.txt
# Type 2: 指令序列集成测试
# 使用 StatementContext + execute_warp_instruction() 验证指令执行

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_ARCHITECTURES "100")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep --no-compress")

set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/tests)

# ============================================================================
# Barrier 集成测试
# ============================================================================
add_catch_test(integration_barrier_module
    barrier/test_barrier_module_integrated.cpp
)
set_tests_properties(integration_barrier_module PROPERTIES LABELS "integration;barrier")

add_catch_test(integration_barrier_scenarios
    barrier/test_barrier_scenarios_integrated.cpp
)
set_tests_properties(integration_barrier_scenarios PROPERTIES LABELS "integration;barrier;regression")

add_catch_test(integration_barrier_verification
    barrier/test_barrier_verification_integrated.cpp
)
set_tests_properties(integration_barrier_verification PROPERTIES LABELS "integration;barrier")

add_catch_test(integration_barrier_divergence_scheduling
    barrier/test_barrier_divergence_scheduling.cpp
)
set_tests_properties(integration_barrier_divergence_scheduling PROPERTIES LABELS "integration;barrier;divergence")

add_catch_test(integration_warp_barrier
    barrier/test_warp_barrier_integrated.cpp
)
set_tests_properties(integration_warp_barrier PROPERTIES LABELS "integration;wbar;regression")

# ============================================================================
# SIMT 集成测试
# ============================================================================
add_catch_test(integration_simt_stack_entry
    simt/test_simt_stack_entry_integrated.cpp
)
set_tests_properties(integration_simt_stack_entry PROPERTIES LABELS "integration;simt_stack")

add_catch_test(integration_simt_thread_pc
    simt/test_simt_thread_pc_integrated.cpp
)
set_tests_properties(integration_simt_thread_pc PROPERTIES LABELS "integration;thread_pc")

# ============================================================================
# Divergence 集成测试
# ============================================================================
add_catch_test(integration_divergence_sync_convergence
    divergence/test_divergence_sync_convergence.cpp
)
set_tests_properties(integration_divergence_sync_convergence PROPERTIES LABELS "integration;divergence;convergence")

add_catch_test(integration_divergence_sync_standalone
    divergence/test_divergence_sync_standalone_integrated.cpp
)
set_tests_properties(integration_divergence_sync_standalone PROPERTIES LABELS "integration;divergence;barrier_sync")

add_catch_test(integration_nested_divergence
    divergence/test_nested_divergence.cpp
)
set_tests_properties(integration_nested_divergence PROPERTIES LABELS "integration;divergence;nested")

add_catch_test(integration_post_barrier_divergence
    divergence/test_post_barrier_divergence.cpp
)
set_tests_properties(integration_post_barrier_divergence PROPERTIES LABELS "integration;divergence;barrier")

# ============================================================================
# Exec 集成测试
# ============================================================================
add_catch_test(integration_warp_state
    exec/test_warp_state_integrated.cpp
)
set_tests_properties(integration_warp_state PROPERTIES LABELS "integration;warp_state")

add_catch_test(integration_ptx_lane_verification
    exec/test_ptx_lane_verification.cpp
)
target_compile_definitions(integration_ptx_lane_verification PRIVATE
    TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
set_tests_properties(integration_ptx_lane_verification PROPERTIES LABELS "integration;ptx_lane_verification")

# ============================================================================
# PC 集成测试
# ============================================================================
add_catch_test(integration_pc_management
    pc/test_pc_management_integrated.cpp
)
set_tests_properties(integration_pc_management PROPERTIES LABELS "integration;pc")

# ============================================================================
# Sync 集成测试
# ============================================================================
add_catch_test(integration_sync_mechanism
    sync/test_sync_mechanism_integrated.cpp
)
set_tests_properties(integration_sync_mechanism PROPERTIES LABELS "integration;sync")

add_catch_test(integration_syncthreads_full_pipeline
    sync/test_syncthreads_full_pipeline.cpp
)
set_tests_properties(integration_syncthreads_full_pipeline PROPERTIES LABELS "integration;syncthreads")

# test_spinlock_simulation.cu has pre-existing build errors
# Not built in main branch
# add_catch_test(integration_spinlock_simulation
#     sync/test_spinlock_simulation.cu
# )

# ============================================================================
# CFG 集成测试
# ============================================================================
# test_cfg_analysis.cpp has pre-existing build errors (uses <catch2/catch.hpp>)
# Not built in main branch
# add_catch_test(integration_cfg_analysis
#     cfg/test_cfg_analysis.cpp
# )

# ============================================================================
# Register 集成测试
# ============================================================================
# test_register_bank_subwarp.cpp has pre-existing build errors
# Not built in main branch
# add_catch_test(integration_register_bank_subwarp
#     register/test_register_bank_subwarp.cpp
# )
