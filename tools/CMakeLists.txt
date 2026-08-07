add_executable(ptxir_embed ptxir_embed.cpp)
target_link_libraries(ptxir_embed PRIVATE cudart OpenSSL::Crypto)
target_include_directories(ptxir_embed PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(ptxir_extract ptxir_extract.cpp)
target_link_libraries(ptxir_extract PRIVATE cudart OpenSSL::Crypto)
target_include_directories(ptxir_extract PRIVATE ${PROJECT_SOURCE_DIR}/include)
