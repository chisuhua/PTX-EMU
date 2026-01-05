cmake_minimum_required(VERSION 3.15)

project(PTX-EMU LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 是否使用详细日志的选项
option(USE_DETAILED_LOGGING "Use detailed logging with file location info" OFF)

if(USE_DETAILED_LOGGING)
    add_compile_definitions(PTXSIM_USE_DETAILED_LOGGING)
endif()

# 检查CUDA_PATH环境变量
if(NOT DEFINED ENV{CUDA_PATH})
    # message(FATAL_ERROR "CUDA_PATH environment variable is not set. Please set it to your CUDA installation path.")
    message(ERROR "CUDA_PATH environment variable is not set. Please set it to your CUDA installation path.")
    set(CUDA_PATH /user/local/cuda)
endif()

set(CUDA_PATH $ENV{CUDA_PATH})

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 查找所需的包
find_package(Java COMPONENTS Runtime REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 设置CUDA架构
set(DEFAULT_CUDA_ARCH "80" CACHE STRING "Default CUDA architecture")

# 添加ANTLR依赖
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/antlr4/antlr-4.11.1-complete.jar)
set(ANTLR4_RUNTIME_SOURCE_DIR ${CMAKE_SOURCE_DIR}/antlr4/antlr4-cpp-runtime-4.11.1-source)
set(ANTLR4_GENERATED_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/antlr4_generated_src)

# 创建目录
file(MAKE_DIRECTORY ${ANTLR4_GENERATED_SRC_DIR})

# 构建ANTLR4 C++运行时
set(ANTLR4_INCLUDE_DIR ${ANTLR4_RUNTIME_SOURCE_DIR}/runtime/src)
set(ANTLR4_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/antlr4_runtime)
add_subdirectory(${ANTLR4_RUNTIME_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/antlr4_runtime)

# 包含ANTLR运行时头文件
include_directories(${ANTLR4_INCLUDE_DIR})
include_directories(${ANTLR4_GENERATED_SRC_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/json)

# 设置要链接的ANTLR4库
set(ANTLR4_LIBRARY antlr4_shared)

# 添加src子目录
add_subdirectory(src)

# 添加tests子目录
enable_testing()
add_subdirectory(tests)

# 添加安装规则
install(DIRECTORY lib/ 
        DESTINATION lib
        FILES_MATCHING PATTERN "*.so*")

install(DIRECTORY bin/ 
        DESTINATION bin
        FILES_MATCHING PATTERN "*")
