# test_extern_shared - CMake build configuration
add_executable(test_extern_shared test_extern_shared.cu)
target_link_libraries(test_extern_shared PRIVATE cudart ptxsim ptx_parser)
