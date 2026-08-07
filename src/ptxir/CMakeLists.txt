add_library(ptxir STATIC
    ptxir_serialization.cpp
)
target_link_libraries(ptxir PUBLIC ptxir_writer ptxir_reader ptx_parser)
target_include_directories(ptxir PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(ptxir PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(ptxir GenerateParser)
