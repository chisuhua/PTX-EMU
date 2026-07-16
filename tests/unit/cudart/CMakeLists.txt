# tests/unit/cudart/CMakeLists.txt
# CUDA Runtime 拦截单元测试
#
# NOTE: This file is currently NOT included by the build (tests are registered
# in tests/unit/CMakeLists.txt instead). Kept for future migration reference.
# The unit_cuda_stream_handle + unit_stream_destroy tests are registered in
# tests/unit/CMakeLists.txt §"CUDA Runtime" block.

add_catch_test(unit_cuda_stream_handle
    cudart/test_cuda_stream_handle.cpp
)
set_tests_properties(unit_cuda_stream_handle PROPERTIES LABELS "unit;cudart;stream")

# B3: cudaStreamDestroy UB delete + g_active_streams cleanup
add_catch_test(unit_stream_destroy
    cudart/test_stream_destroy.cpp
)
set_tests_properties(unit_stream_destroy PROPERTIES LABELS "unit;cudart;stream;destroy")