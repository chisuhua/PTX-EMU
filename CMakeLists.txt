cmake_minimum_required(VERSION 3.15)

# ==============================
# 🔧 关键：在 project() 前设置 CUDA 标准并启用语言
# ==============================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
enable_language(CUDA)

# ==============================
# 📦 项目定义（包含 CXX 和 CUDA）
# ==============================
project(PTX-EMU LANGUAGES CXX CUDA)

# ==============================
# 🛠️ 全局设置
# ==============================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 日志选项
option(USE_DETAILED_LOGGING "Use detailed logging with file location info" OFF)
if(USE_DETAILED_LOGGING)
    add_compile_definitions(PTXSIM_USE_DETAILED_LOGGING)
endif()

# 输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ==============================
# 🔍 查找依赖
# ==============================
find_package(Java COMPONENTS Runtime REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA 架构（可选）
set(DEFAULT_CUDA_ARCH "100" CACHE STRING "Default CUDA architecture")

# ==============================
# 🐜 ANTLR4 配置
# ==============================
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/antlr4/antlr-4.11.1-complete.jar)
set(ANTLR4_RUNTIME_SOURCE_DIR ${CMAKE_SOURCE_DIR}/antlr4/antlr4-cpp-runtime-4.13.1-source/runtime/Cpp)
set(ANTLR4_GENERATED_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/antlr4_generated_src)

file(MAKE_DIRECTORY ${ANTLR4_GENERATED_SRC_DIR})

# 🔥 关键：禁用 ANTLR4 的测试和 demo，避免下载 googletest
set(ANTLR4_BUILD_TESTS OFF CACHE BOOL "Disable ANTLR4 tests" FORCE)
set(ANTLR_BUILD_CPP_TESTS OFF CACHE BOOL "Disable ANTLR4 tests" FORCE)
set(WITH_DEMO OFF CACHE BOOL "Disable ANTLR4 demo" FORCE)

# 构建 ANTLR4 运行时
add_subdirectory(
    ${ANTLR4_RUNTIME_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/antlr4_runtime
)

# 包含头文件（临时方案，建议后续改用 target_include_directories）
include_directories(${ANTLR4_RUNTIME_SOURCE_DIR}/runtime/src)
include_directories(${ANTLR4_GENERATED_SRC_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/json)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/inipp)

set(ANTLR4_LIBRARY antlr4_shared)

# ==============================
# 📁 子目录
# ==============================
add_subdirectory(src)

# 启用测试（但确保 tests/CMakeLists.txt 不再 enable_language(CUDA)）
enable_testing()
add_subdirectory(tests)

# 可选 benchmark
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/cute/CMakeLists.txt)
    add_subdirectory(bench/cute)
endif()

# ==============================
# 📤 安装规则
# ==============================
install(DIRECTORY lib/
        DESTINATION lib
        FILES_MATCHING PATTERN "*.so*")

install(DIRECTORY bin/
        DESTINATION bin
        FILES_MATCHING PATTERN "*")