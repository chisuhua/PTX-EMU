add_catch_test(e2e_image_executor_output test_image_executor_output.cpp)
set_tests_properties(e2e_image_executor_output PROPERTIES
    LABELS "e2e;path_2D" TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Phase 3.1 (path_2D): synchronous completion regression.
# Uses CUDA runtime APIs to allocate device buffers + trigger
# __cudaRegisterFatBinary (which sets up g_gpu_context), then calls
# ptxemu_image_load/execute directly to verify the kernel's write is
# observable before the API returns.
find_package(CUDAToolkit REQUIRED)
add_catch_test(e2e_image_executor_synchronous
    test_image_executor_synchronous.cpp
)
target_link_libraries(e2e_image_executor_synchronous PRIVATE CUDA::cudart ptxemu_device)
set_tests_properties(e2e_image_executor_synchronous PROPERTIES
    LABELS "e2e;path_2D;cpptlm_module;synchronous"
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENVIRONMENT "PTXIR_FIXTURE_DIR=${CMAKE_SOURCE_DIR}/tests/ptxir/fixtures;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib"
)
