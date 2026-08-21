add_catch_test(integration_ptxir_cubin_loader
    test_ptxir_cubin_loader.cpp
)
set_tests_properties(integration_ptxir_cubin_loader PROPERTIES LABELS "integration;cudart;ptxir_dispatch")
target_link_libraries(integration_ptxir_cubin_loader PRIVATE ptxir_writer ptxir_reader)

add_catch_test(integration_in_memory_mutation
    test_in_memory_mutation.cpp
)
set_tests_properties(integration_in_memory_mutation PROPERTIES LABELS "integration;cudart;mutation")
target_link_libraries(integration_in_memory_mutation PRIVATE ptxir_writer ptxir_reader ptxemu_device)

add_catch_test(integration_cuda_driver_api
    test_cuda_driver_api.cpp
)
set_tests_properties(integration_cuda_driver_api PROPERTIES LABELS "integration;cudart;driver")
target_link_libraries(integration_cuda_driver_api PRIVATE ptxir_writer ptxir_reader)

add_catch_test(integration_libptxemu_device
    test_libptxemu_device.cpp
)
set_tests_properties(integration_libptxemu_device PROPERTIES LABELS "integration;cudart;abi")
target_link_libraries(integration_libptxemu_device PRIVATE ptxemu_device ptxir_writer ptxir_reader)
target_compile_definitions(integration_libptxemu_device PRIVATE
    TEST_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/ptxir/fixtures")

# Sync-only runtime contract: cudaStreamSynchronize(nullptr) returns cudaSuccess
# immediately (per cudart-sync-only-runtime/spec.md Stream lifecycle scenario).
# Created per Metis 2026-08-21 MUST-RESOLVE #5 (Phase 2a replacement for
# test_stream_sync_loop.cpp).
add_catch_test(integration_sync_only_immediate
    test_sync_only_immediate.cpp
)
set_tests_properties(integration_sync_only_immediate PROPERTIES
    LABELS "integration;cudart;sync")
