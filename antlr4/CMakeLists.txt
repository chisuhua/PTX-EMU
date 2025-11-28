cmake_minimum_required(VERSION 3.14)
project(antlr4_cpp_runtime)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(run/usr/local/lib)

if(ANTLR_CPP_RUNTIME)
    install(TARGETS antlr4 DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    install(FILES antlr4.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()