# tests/e2e/CMakeLists.txt
# Type 3: CUDA Kernel E2E 测试
# 编译真实 CUDA kernel，通过 cuobjdump 提取 PTX，完整执行

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_ARCHITECTURES "100")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep --no-compress")

set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/tests)

# ============================================================================
# Kernel E2E 测试 - 完整 kernel 通过 cudaLaunchKernel 触发
# ============================================================================
add_catch_test(e2e_test3_cfg_full
    kernel/test_test3_cfg_full.cpp
)
set_tests_properties(e2e_test3_cfg_full PROPERTIES LABELS "e2e;cfg")

add_catch_test(e2e_barrier_warp_sync
    kernel/test_barrier_warp_sync.cu
)
set_tests_properties(e2e_barrier_warp_sync PROPERTIES LABELS "e2e;barrier")

add_catch_test(e2e_ldglobal_simple
    kernel/test_ldglobal_simple.cu
)
set_tests_properties(e2e_ldglobal_simple PROPERTIES LABELS "e2e;ldglobal")

# ============================================================================
# Divergence E2E 测试 - 完整 warp divergence kernel 验证
# ============================================================================
add_catch_test(e2e_divergence
    divergence/test_divergence.cu
    divergence/divergence_kernels.cu
)
set_tests_properties(e2e_divergence PROPERTIES LABELS "e2e;divergence;warp")

add_catch_test(e2e_divergence_sync
    divergence/test_divergence_sync.cu
)
set_tests_properties(e2e_divergence_sync PROPERTIES LABELS "e2e;divergence;barrier_sync")

# ============================================================================
# Shared Memory E2E 测试 - 共享内存 kernel 验证
# ============================================================================
add_catch_test(e2e_shared_memory_dynamic
    kernel/test_shared_memory_dynamic.cu
)
set_tests_properties(e2e_shared_memory_dynamic PROPERTIES LABELS "e2e;shared_memory;dynamic")

add_catch_test(e2e_shared_memory_multi_buffer
    kernel/test_shared_memory_multi_buffer.cu
)
set_tests_properties(e2e_shared_memory_multi_buffer PROPERTIES LABELS "e2e;shared_memory;multi_buffer")

add_catch_test(e2e_shared_memory_divergence_visibility
    kernel/test_shared_memory_divergence_visibility.cu
)
set_tests_properties(e2e_shared_memory_divergence_visibility PROPERTIES LABELS "e2e;shared_memory;divergence;visibility")

# ============================================================================
# Blackwell tcgen05 GEMM E2E 测试 - 16x16 float kernel (sm_100)
# ============================================================================
add_catch_test(e2e_blackwell_gemm
    kernel/test_blackwell_gemm.cu
)
set_tests_properties(e2e_blackwell_gemm PROPERTIES LABELS "e2e;tcgen05;gemm;sm100")
target_include_directories(e2e_blackwell_gemm PRIVATE
    ${PROJECT_SOURCE_DIR}/bench/cute/include
)

# ============================================================================
# tcgen05.mma GEMM E2E — explicit 5 tcgen05 instructions + f32 fallback
# (ADR-0016, fix-tcgen05-test-coverage-gaps, Priority 3 fallback path)
# ============================================================================
add_catch_test(e2e_tcgen05_mma_gemm
    kernel/test_tcgen05_mma_gemm.cu
)
set_tests_properties(e2e_tcgen05_mma_gemm PROPERTIES
    LABELS "e2e;kernel;tcgen05;gemm;mma;sm100")

# ============================================================================
# tcgen05.cp SMEM → TMEM E2E — Priority 3 f32 fallback
# (ADR-0016, tcgen05-cp-test-coverage-and-exception-cleanup, Phase 3)
# ptxas 13.0 does NOT support `tcgen05.cp` on .target sm_100
# (same constraint as e2e_tcgen05_mma_gemm). Mirrors the Priority 3
# fallback: pure CUDA C++ that performs the same logical SMEM → "TMEM"
# copy via __shared__ memory, source-grep oracle on `tcgen05.cp`.
# ============================================================================
add_catch_test(e2e_tcgen05_cp
    kernel/test_tcgen05_cp.cu
)
set_tests_properties(e2e_tcgen05_cp PROPERTIES
    LABELS "e2e;kernel;tcgen05;cp;sm100")
