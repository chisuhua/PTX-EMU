# tests/CMakeLists.txt

# 启用测试支持
enable_testing()

# Catch2 单头文件路径
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 定义创建单个 Catch2 测试的函数
function(add_catch_test TEST_NAME)
    # 参数：测试名称，其余为该测试的源文件（不含 catch_amalgamated.cpp）
    set(SOURCES ${ARGN})

    # 自动包含 catch_amalgamated.cpp
    list(APPEND SOURCES ${CATCH_INCLUDE_DIR}/catch_amalgamated.cpp)

    # 创建可执行文件
    add_executable(${TEST_NAME} ${SOURCES})

    # 包含目录
    target_include_directories(${TEST_NAME} PRIVATE
        ${CATCH_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
    )

    # 链接库（如果存在）
    if(TARGET cpphdl)
        target_link_libraries(${TEST_NAME} cpphdl)
    endif()

    # 注册为 CTest 测试
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# ================================
# 方案一：为每个 test_xxx.cpp 创建独立测试（推荐）
# ================================

# 列出所有独立的测试源文件（每个文件一个测试）
set(SINGLE_TEST_SOURCES
#TODO add files
)

# 为每个源文件创建一个独立的测试
foreach(TEST_SRC ${SINGLE_TEST_SOURCES})
    # 提取测试名，如 test_reg.cpp -> test_reg
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_catch_test(${TEST_NAME} ${TEST_SRC})
endforeach()

# ================================
# 方案二：特殊测试（多文件组合）
# ================================

# test_bitvector 需要多个源文件
#add_catch_test(test_bitvector
# source cpp list
#)

# ================================
# 添加基于 CUDA 的测试（直接使用 CMake 编译）
# ================================

# 定义 mini tests (基础测试集)
set(PTX_MINI_TESTS 
    dummy dummy-add dummy-float dummy-grid dummy-mul dummy-sub dummy-condition
    dummy-long dummy-sieve dummy-share)

# 定义全部测试 (包含基础测试和其他复杂测试)
set(PTX_ALL_TESTS 
    ${PTX_MINI_TESTS} 
    simpleGEMM-int simpleGEMM-float simpleGEMM-double
    simpleCONV-int simpleCONV-float simpleCONV-double 2Dentropy
    aligned-types all-pairs-distance bitonic bfs backprop RAY cfd)


# 创建方便的自定义目标来运行不同组的测试
add_custom_target(minitest
    COMMAND ${CMAKE_CTEST_COMMAND} -L mini
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running mini tests")

add_custom_target(alltest
    COMMAND ${CMAKE_CTEST_COMMAND} -L mini -L full
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests")

# 启用 CUDA 语言支持
enable_language(CUDA)

# 设置 CMP0104 策略以避免警告
cmake_policy(SET CMP0104 NEW)

# 简化架构设置处理 - 现代CMake方式
if(NOT DEFAULT_CUDA_ARCH)
    # 默认使用Ampere架构
    set(DEFAULT_CUDA_ARCH "80")
endif()


# 为每个测试创建 CMake 编译目标
foreach(TEST_NAME ${PTX_ALL_TESTS})
    # 检查测试源文件是否存在
    if(EXISTS ${CMAKE_SOURCE_DIR}/bench/${TEST_NAME}/${TEST_NAME}.cu)
        # 创建可执行文件
        add_executable(${TEST_NAME} 
            ${CMAKE_SOURCE_DIR}/bench/${TEST_NAME}/${TEST_NAME}.cu)
        
        # 关键修改1: 使用现代CMake CUDA架构设置
        # 这会同时生成PTX (compute_XX)和二进制代码(sm_XX)
        set_target_properties(${TEST_NAME} PROPERTIES
            CUDA_ARCHITECTURES "${DEFAULT_CUDA_ARCH}-real"
            CUDA_STANDARD 14
            CUDA_STANDARD_REQUIRED ON)

        target_compile_options(${TEST_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                "-gencode=arch=compute_${DEFAULT_CUDA_ARCH},code=compute_${DEFAULT_CUDA_ARCH}"
                "-gencode=arch=compute_${DEFAULT_CUDA_ARCH},code=sm_${DEFAULT_CUDA_ARCH}"
            >)

        # 关键修改2: 移除冲突的手动-arch设置和分离编译
        # 只有在确实需要设备代码链接时才启用分离编译
        # set_target_properties(${TEST_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        
        # 关键修改3: 移除冲突的-arch编译选项
        # target_compile_options(${TEST_NAME} PRIVATE 
        #     $<$<COMPILE_LANGUAGE:CUDA>: -arch=${REAL_CUDA_ARCH}>)
        
        # 包含 CUDA 头文件
        target_include_directories(${TEST_NAME} PRIVATE ${CUDA_PATH}/include)
        
        # 链接我们自己构建的 cudart 库和 ANTLR 库
        target_link_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/lib)
        target_link_libraries(${TEST_NAME} 
            cudart
            antlr4_shared)
            
        # 设置 RPATH，确保运行时能找到我们自己的库
        set_target_properties(${TEST_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_SOURCE_DIR}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE)
        
        # 设置输出目录与主项目一致
        set_target_properties(${TEST_NAME} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        
        # 添加测试
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        
        # 设置属性，标记哪些是基础测试
        list(FIND PTX_MINI_TESTS ${TEST_NAME} IS_MINI_TEST)
        if(NOT IS_MINI_TEST EQUAL -1)
            # 标记为基础测试
            set_tests_properties(${TEST_NAME} PROPERTIES LABELS "mini")
        else()
            # 标记为完整测试
            set_tests_properties(${TEST_NAME} PROPERTIES LABELS "full")
        endif()
    endif()
endforeach()
