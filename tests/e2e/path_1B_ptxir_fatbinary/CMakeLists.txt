add_catch_test(e2e_ptxir_fatbinary_exec
    test_ptxir_fatbinary_exec.cpp
)
set_tests_properties(e2e_ptxir_fatbinary_exec PROPERTIES
    LABELS "e2e;path_1B"
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENVIRONMENT "PTXIR_MODE=auto;PATH="
)
add_custom_command(TARGET e2e_ptxir_fatbinary_exec PRE_BUILD
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_standalone.sh
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/path_1B_kernels.cu
)

# Format-level PTXIR-Embedded CUBIN test (moved from tests/e2e/kernel/)
add_catch_test(e2e_ptxir_cubin_embed test_ptxir_cubin_embed.cpp)
set_tests_properties(e2e_ptxir_cubin_embed PROPERTIES
    LABELS "e2e;path_1B;ptxir;tools;nvcc;cuobjdump"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
