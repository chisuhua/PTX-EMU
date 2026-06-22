# tests/unit/cudart/CMakeLists.txt
# CUDA Runtime 拦截单元测试

add_catch_test(unit_cuda_stream_handle
    cudart/test_cuda_stream_handle.cpp
)
set_tests_properties(unit_cuda_stream_handle PROPERTIES LABELS "unit;cudart;stream")