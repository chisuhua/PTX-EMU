add_library(ptxir STATIC
    ptxir_serialization.cpp
)
target_link_libraries(ptxir PUBLIC ptxir_writer ptxir_reader)
target_include_directories(ptxir PUBLIC ${PROJECT_SOURCE_DIR}/include)
