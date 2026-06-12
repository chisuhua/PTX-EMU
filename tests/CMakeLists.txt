# tests/CMakeLists.txt
# Top-level CMake configuration for PTX-EMU test suite
#
# 测试分类：
#   unit/        - Type 1: 直接单元测试
#   integration/ - Type 2: 指令序列集成测试
#   e2e/         - Type 3: CUDA Kernel E2E 测试
#   ptx/         - PTX 解析/语法专项测试
#   instructions/ - 指令测试（已存在）
#   ptxir/       - PTXIR 序列化（已存在）
#   common/      - 通用工具（已存在）
#   archive/     - 历史归档

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)       # 全局强制PTX模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF) # 禁用分离编译

set(CMAKE_CUDA_COMPILE_OPTIONS_PTX "")
set(CMAKE_CUDA_COMPILE_OPTIONS_ARCH "")
set(CMAKE_CUDA_COMPILE_OPTIONS_CODE "")

# 简化架构设置处理 - 现代CMake方式
if(NOT DEFAULT_CUDA_ARCH)
    # 默认使用虚拟架构
    set(DEFAULT_CUDA_ARCH "100")
endif()

# 设置CUDA编译选项，保留中间PTX文件，禁用压缩
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep --no-compress")

# 设置为只使用sm_100架构（虚拟架构）
set(CMAKE_CUDA_ARCHITECTURES "100")

# 启用测试支持
enable_testing()
# 启用 CUDA 语言支持

# Catch2 单头文件路径
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 定义创建单个 Catch2 测试的函数
function(add_catch_test TEST_NAME)
    # 参数：测试名称（不含后缀），其余为该测试的源文件（不含 catch_amalgamated.cpp）
    set(SOURCES ${ARGN})

    # 如果 TEST_NAME 是文件路径，提取不带扩展名的基本文件名作为目标名称
    get_filename_component(TEST_TARGET_NAME ${TEST_NAME} NAME_WE)

    # 自动包含 catch_amalgamated.cpp
    list(APPEND SOURCES
    ${CATCH_INCLUDE_DIR}/catch_amalgamated.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
    )

    # 创建可执行文件，使用处理后的目标名称
    add_executable(${TEST_TARGET_NAME} ${SOURCES})

    # 包含目录
    target_include_directories(${TEST_TARGET_NAME} PRIVATE
        ${CATCH_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )

    # 链接我们自己构建的 cudart 库和 ANTLR 库
    target_link_directories(${TEST_TARGET_NAME} PRIVATE
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        ${CMAKE_SOURCE_DIR}/lib
        )

    # 链接库
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE
        ptxsim
        cudart
        antlr4_shared
        ptxir_writer
        -Wl,--as-needed  # 恢复默认行为
        -ldl -lpthread
    )

    # 设置测试目标
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 添加测试
    add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})
endfunction()

# 定义创建单个独立测试的函数（不使用Catch2）
function(add_standalone_test TEST_NAME)
    # 参数：测试名称，其余为该测试的源文件
    set(SOURCES ${TEST_NAME} ${ARGN})

    get_filename_component(TEST_TARGET_NAME ${TEST_NAME} NAME_WE)

    # 自动包含 catch_amalgamated.cpp
    list(APPEND SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
    )

    # 创建可执行文件
    add_executable(${TEST_TARGET_NAME} ${SOURCES})

    # 包含目录
    target_include_directories(${TEST_TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${ANTLR4_INCLUDE_DIR}
        ${ANTLR4_GENERATED_SRC_DIR}
    )

    # 链接必要的库
    target_link_libraries(${TEST_TARGET_NAME}
        ptx_ir
        ptx_parser
        cudart
        ptxsim
        antlr4_shared
        pthread
    )

    # 设置运行时路径
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_SOURCE_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    # 注册为 CTest 测试
    add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})

    set_tests_properties(${TEST_TARGET_NAME} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS"
        FAIL_REGULAR_EXPRESSION "Error|Failed"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endfunction()

# ============================================================================
# 测试子目录（按三类组织）
# ============================================================================

# Type 1: 直接单元测试
add_subdirectory(unit)
add_subdirectory(unit/testing)

# Type 2: 指令序列集成测试
add_subdirectory(integration)

# Type 3: CUDA Kernel E2E 测试
add_subdirectory(e2e)

# ============================================================================
# 保留的特殊测试
# ============================================================================

# Warp standalone 单元测试 (有 main 函数)
add_standalone_test(unit/warp/test_warp_context.cpp)
add_standalone_test(unit/warp/test_warp_scheduler.cpp)
add_standalone_test(unit/warp/test_sm_context.cpp)

# Addc/Subc standalone 单元测试 (有 main 函数)
add_standalone_test(unit/exec/test_addc_subc_handler.cpp)

# Printf standalone E2E 测试 (有 main 函数)
add_standalone_test(e2e/kernel/test_printf.cu)
# test_printf uses debug logs that contain "Failed" but tests actually pass
# Custom configuration to look for completion message instead
set_tests_properties(test_printf PROPERTIES
    PASS_REGULAR_EXPRESSION "Printf test completed"
    FAIL_REGULAR_EXPRESSION "^Test.*FAILED"
)

# PTX 解析验证（不使用 Catch2）
add_standalone_test(ptx/parser/test-ptx.cpp)
# test_wmma.cpp has pre-existing build errors (StatementContext::WMMA not defined)
# Disabled in main branch as well
# add_standalone_test(ptx/parser/test_wmma.cpp)

# Warp divergence E2E 测试已迁移到 tests/e2e/divergence/（见 tests/e2e/CMakeLists.txt）

# 已有指令测试
add_standalone_test(instructions/test_ptx_bra.cu)

# PTX 语法测试（使用 ANTLR 解析）
# Note: tests/ptx/test_cfg_builder.cpp removed (Phase 5.1 stub, no real assertions;
# coverage provided by test_cfg_edge_cases.cpp and tests/integration/cfg/test_cfg_analysis.cpp)
add_standalone_test(ptx/test_cfg_edge_cases.cpp)
# test_cfg_benchmark.cpp moved to tests/integration/cfg/integration_cfg_benchmark.cpp
# (commit ab55e06 naming convention: integration_* prefix; classified as integration test
#  per AGENTS.md test type classification since it exercises CFG subsystem integration)
add_standalone_test(ptx/test_simt_stack_integration.cpp)

# ============================================================================
# Mini/basic/extra 基准测试（使用 bench/ 下的 .cu 文件）
# ============================================================================

# 定义 mini tests (基础测试集)
set(PTX_MINI_TESTS
    dummy dummy-args dummy-add dummy-float dummy-grid dummy-mul dummy-sub dummy-condition
    dummy-long dummy-sieve dummy-share dummy-loop dummy-ldglobal)

set(PTX_BASIC_TESTS
    simpleGEMM-int simpleGEMM-float simpleGEMM-double
    simpleCONV-int simpleCONV-float simpleCONV-double 2Dentropy
    aligned-types all-pairs-distance bitonic bfs )
    # aligned-types all-pairs-distance bitonic bfs backprop RAY cfd)

set(PTX_SYNC_TESTS
    test_syncthreads
    test_warp_divergence
    test_shared_memory
    test_divergence_sync_standalone)

# 定义全部测试 (包含基础测试和其他复杂测试)
set(PTX_ALL_TESTS
    ${PTX_MINI_TESTS}
    ${PTX_SYNC_TESTS}
    ${PTX_BASIC_TESTS}
    ${PTX_MMA_TESTS})


# 创建方便的自定义目标来运行不同组的测试
add_custom_target(minitest
    COMMAND ${CMAKE_CTEST_COMMAND} -L mini
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running mini tests")

add_custom_target(alltest
    COMMAND ${CMAKE_CTEST_COMMAND} -L mini -L full
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests")


# 设置 CMP0104 策略以避免警告
cmake_policy(SET CMP0104 NEW)


# 为每个测试创建 CMake 编译目标
foreach(TEST_NAME ${PTX_ALL_TESTS})
    # 检查测试源文件是否存在
    if(EXISTS ${CMAKE_SOURCE_DIR}/bench/${TEST_NAME}/${TEST_NAME}.cu)
        set(CUDA_INCLUDE_DIRS ${CUDA_PATH}/include)
    else()
        message(FATAL_ERROR "Test file does not exist: ${CMAKE_SOURCE_DIR}/bench/${TEST_NAME}/${TEST_NAME}.cu")
    endif()

    set(OBJECT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cu.o")
    add_custom_command(
        OUTPUT ${OBJECT_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER}
            -cudart=shared
            -g -std=c++17
            -arch=compute_100
            -code=compute_100
            -I${PROJECT_SOURCE_DIR}/include
            -I${CUDA_INCLUDE_DIRS}
            -x cu
            -c ${CMAKE_SOURCE_DIR}/bench/${TEST_NAME}/${TEST_NAME}.cu
            -o ${OBJECT_FILE}
        DEPENDS ${CMAKE_SOURCE_DIR}/bench/${TEST_NAME}/${TEST_NAME}.cu
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
    # 创建可执行文件
    add_executable(${TEST_NAME} "")

    target_sources(${TEST_NAME} PRIVATE ${OBJECT_FILE})

    set_target_properties(${TEST_NAME} PROPERTIES
        LINKER_LANGUAGE CXX  # 必须显式指定!
        SUFFIX ""            # 防止添加.a后缀
        OUTPUT_NAME "${TEST_NAME}"
        PREFIX ""  # 防止添加lib前缀
    )

    # 包含 CUDA 头文件
    target_include_directories(${TEST_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

    # 链接我们自己构建的 cudart 库和 ANTLR 库
    target_link_directories(${TEST_NAME} PRIVATE
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        ${CMAKE_SOURCE_DIR}/lib
            )

    target_link_libraries(${TEST_NAME} PRIVATE
            -Wl,--no-as-needed
            cudart
            antlr4_shared
            -Wl,--as-needed  # 恢复默认行为
            -ldl -lpthread
            )

    # 设置 RPATH，确保运行时能找到我们自己的库
    set_target_properties(${TEST_NAME} PROPERTIES
            INSTALL_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
            BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
            BUILD_WITH_INSTALL_RPATH TRUE
            SKIP_BUILD_RPATH FALSE
            INSTALL_RPATH_USE_LINK_PATH TRUE
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

    add_dependencies(${TEST_NAME} cudart antlr4_shared)  # 确保库先构建

    add_custom_target(${TEST_NAME}_obj DEPENDS ${OBJECT_FILE})
    add_dependencies(${TEST_NAME} ${TEST_NAME}_obj)

    # 添加测试
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # 设置工作目录和GPU配置环境变量
    list(FIND PTX_BASIC_TESTS ${TEST_NAME} IS_BASIC_TEST)
    if(NOT IS_BASIC_TEST EQUAL -1)
        # 基础测试需要A100配置和工作目录为项目根目录
        set_tests_properties(${TEST_NAME} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            ENVIRONMENT "PTX_EMU_GPU_CONFIG=ampere_a100.json;PTX_EMU_CONFIG=perf_config.ini;PTX_EMU_GEMM_TEST_MODE=small;PTX_EMU_CONV_TEST_MODE=small")
    endif()

    # 设置属性，标记哪些是基础测试
    list(FIND PTX_MINI_TESTS ${TEST_NAME} IS_MINI_TEST)
    if(NOT IS_MINI_TEST EQUAL -1)
        # 标记为基础测试
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "mini")
    endif()
    list(FIND PTX_BASIC_TESTS ${TEST_NAME} IS_BASIC_TEST)
    if(NOT IS_BASIC_TEST EQUAL -1)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "basic")
    endif()
    list(FIND PTX_MMA_TESTS ${TEST_NAME} IS_MMA_TEST)
    if(NOT IS_MMA_TEST EQUAL -1)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "mma")
    endif()
endforeach()

# Add external test subdirectories
add_subdirectory(../bench/test_extern_shared CMakeFiles.test_extern_shared.dir)

# ============================================================================
# 通用工具（不是测试，但提供 build target）
# ============================================================================

# PTX Lane PC Trace Printer (common utility)
add_executable(ptx_lane_printer
    common/ptx_lane_printer.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)
target_include_directories(ptx_lane_printer PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
target_link_directories(ptx_lane_printer PRIVATE
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    ${CMAKE_SOURCE_DIR}/lib
)
target_link_libraries(ptx_lane_printer PRIVATE
    ptxsim
    cudart
    antlr4_shared
    -Wl,--as-needed
    -ldl -lpthread
)
set_target_properties(ptx_lane_printer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
