add_catch_test(e2e_cuda_driver_exec test_cuda_driver_exec.cpp)
target_include_directories(e2e_cuda_driver_exec PRIVATE /workspace/project/opt/cuda/include)
target_link_libraries(e2e_cuda_driver_exec PRIVATE cudart)
set_tests_properties(e2e_cuda_driver_exec PROPERTIES
    LABELS "e2e;path_1C" TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_command(TARGET e2e_cuda_driver_exec PRE_BUILD
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/build_ptxir_blob.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vec_add.cu
)
