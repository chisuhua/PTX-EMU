add_library(ptxir_writer STATIC
    ptxir_writer.cpp
)
target_link_libraries(ptxir_writer PUBLIC ptx_ir)
target_include_directories(ptxir_writer PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(ptxir_reader STATIC
    ptxir_reader.cpp
)
target_link_libraries(ptxir_reader PUBLIC ptx_ir)
target_include_directories(ptxir_reader PUBLIC ${PROJECT_SOURCE_DIR}/include)
