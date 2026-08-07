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

# Enable compile_commands.json for LSP/clang-tidy/IDE tooling (T0-1)
# MUST be set after project() but BEFORE add_subdirectory() so all targets
# are captured (setting it after add_subdirectory() misses targets).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_WERROR "Treat compiler warnings as errors" OFF)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

if(ENABLE_WERROR)
    add_compile_options(-Werror)
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
find_package(OpenSSL REQUIRED)

# 查找 cuobjdump 工具路径
find_program(CUOBJDUMP_EXECUTABLE cuobjdump HINTS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIRS} ENV CUDA_PATH ENV PATH)
if(CUOBJDUMP_EXECUTABLE)
    message(STATUS "Found cuobjdump: ${CUOBJDUMP_EXECUTABLE}")
    add_compile_definitions(CUOBJDUMP_PATH="${CUOBJDUMP_EXECUTABLE}")
else()
    message(WARNING "cuobjdump not found - PTX extraction from CUDA binaries will fail")
endif()

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER "ccache")
    set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "ccache")
endif()

# CUDA 架构（可选）
set(DEFAULT_CUDA_ARCH "100" CACHE STRING "Default CUDA architecture")

# ==============================
# 🐜 ANTLR4 配置
# ==============================
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/antlr4/antlr-4.13.2-complete.jar)
set(ANTLR4_RUNTIME_SOURCE_DIR ${CMAKE_SOURCE_DIR}/antlr4/antlr4-cpp-runtime-4.13.2-source)
set(ANTLR4_GENERATED_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/antlr4_generated_src)

file(MAKE_DIRECTORY ${ANTLR4_GENERATED_SRC_DIR})

# 🔥 关键：禁用 ANTLR4 的测试和 demo，避免下载 googletest
set(ANTLR_BUILD_CPP_TESTS OFF CACHE BOOL "Disable ANTLR4 C++ tests" FORCE)
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
add_subdirectory(tools)

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

# ==============================
# 🔗 CppTLM Bridge 集成
# ==============================
# CppTLM 始终链接，协同仿真通过环境变量控制：
#   默认（无环境变量）: 同步执行（g_cpptlm_bridge == nullptr）
#   EMU_COSIM=1:        协同仿真（StubBridge auto-attach + auto-advance）
set(CPPTLM_SOURCE_DIR "/workspace/project/CppTLM" CACHE PATH "Local CppTLM repository path")

if(NOT EXISTS "${CPPTLM_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "CppTLM not found at ${CPPTLM_SOURCE_DIR}. "
                        "Set CPPT LM_SOURCE_DIR to the correct path.")
endif()

# 构建 CppTLM 作为子项目（强制 PIC：cpptlm_core 将被链接到 libcudart.so）
set(_prev_pic ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${CPPTLM_SOURCE_DIR} ${CMAKE_BINARY_DIR}/cpptlm)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_prev_pic})

# 链接 CppTLM 到 cudart 目标
# --whole-archive: 强制纳入 cpptlm_core 所有对象，确保 CppTLM 的
# cpptlm_set_driver 强定义覆盖 PTX-EMU 的 __attribute__((weak)) 版本
# NOTE: -Wl,-u,cpptlm_set_driver 不可行（weak 符号在对象文件中已解析，
#       链接器不会搜 archive 找强定义），必须用 --whole-archive
add_dependencies(cudart cpptlm_core)
target_link_libraries(cudart
    -Wl,--whole-archive cpptlm_core -Wl,--no-whole-archive)
include_directories(${CPPTLM_SOURCE_DIR}/include)
target_compile_definitions(cudart PRIVATE BUILD_LIB_CPPTLM_CUDART)

message(STATUS "CppTLM bridge: always linked (runtime: EMU_COSIM=1 to enable co-sim)")
