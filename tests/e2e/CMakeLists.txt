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
