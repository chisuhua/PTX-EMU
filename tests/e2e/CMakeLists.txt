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
# Kernel E2E 测试 - 完整 kernel 通过 cudaLaunchKernel 触发 (kept at root per tasks.md 4.11)
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
# Shared Memory E2E 测试 - 共享内存 kernel 验证 (kept at root per tasks.md 4.11)
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
# FU-5: FlashAttention mini-kernel E2E test (kept at root per tasks.md 4.11)
# ============================================================================
add_catch_test(e2e_flashattention_mini
    kernel/test_flashattention_mini.cu
)
set_tests_properties(e2e_flashattention_mini PROPERTIES
    LABELS "e2e;kernel;tcgen05;flashattention;sm100;FU-5")

# ============================================================================
# Path-organized e2e subdirectories (fix-path-coverage-gaps Phase 4)
# ============================================================================
add_subdirectory(path_1A_legacy_ptx)
add_subdirectory(path_1B_ptxir_fatbinary)
add_subdirectory(path_1C_driver_api)
add_subdirectory(path_2D_image_executor)
