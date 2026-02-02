# tests/CMakeLists.txt
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)       # 全局强制PTX模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF) # 禁用分离编译

set(CMAKE_CUDA_COMPILE_OPTIONS_PTX "")
set(CMAKE_CUDA_COMPILE_OPTIONS_ARCH "")
set(CMAKE_CUDA_COMPILE_OPTIONS_CODE "")
# set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "")
# set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "")
# set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "")
# set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")

# 简化架构设置处理 - 现代CMake方式
if(NOT DEFAULT_CUDA_ARCH)
    # 默认使用虚拟架构
    set(DEFAULT_CUDA_ARCH "100")
endif()

# 设置CUDA编译选项，保留中间PTX文件
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep")

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
    )
endfunction()

add_catch_test(
    test_memory_manager.cpp
    test_parse_immediate.cpp
)

# ================================
# Catch2 PTX 指令测试套件（按类别拆分）
# ================================

# 整数算术测试
add_catch_test(test_ptx_integer
    test_ptx_integer.cu
    ptx_integer_arith.cu
)
set_tests_properties(test_ptx_integer PROPERTIES LABELS "ptx;integer")

# 浮点算术测试
add_catch_test(test_ptx_float
    test_ptx_float.cu
    ptx_float_arith.cu
)
set_tests_properties(test_ptx_float PROPERTIES LABELS "ptx;float")

# 扩展精度整数测试
add_catch_test(test_ptx_extended
    test_ptx_extended.cu
    ptx_extended_prec.cu
)
set_tests_properties(test_ptx_extended PROPERTIES LABELS "ptx;integer")

# 逻辑与移位测试
add_catch_test(test_ptx_bitwise
    test_ptx_bitwise.cu
    ptx_bitwise_shift.cu
)
set_tests_properties(test_ptx_bitwise PROPERTIES LABELS "ptx;bitwise")

# 类型转换测试
add_catch_test(test_ptx_cvt
    test_ptx_cvt.cu
    ptx_cvt_arith.cu
)
set_tests_properties(test_ptx_cvt PROPERTIES LABELS "ptx;cvt")

# 内存访问测试
add_catch_test(test_ptx_ld_st
    test_ptx_ld_st.cu
    ptx_ld_st.cu
)
set_tests_properties(test_ptx_ld_st PROPERTIES LABELS "ptx;ld_st")

# 地址转换测试
add_catch_test(test_ptx_cvta
    test_ptx_cvta.cu
    ptx_cvta.cu
)
set_tests_properties(test_ptx_cvta PROPERTIES LABELS "ptx;cvta")

# （保留其他非 PTX 的 catch 测试，如将来有）
# add_catch_test(test_another test_another.cpp)


# ================================
# 添加独立测试（不使用Catch2）
# ================================

# add_standalone_test(test-ptx test-ptx.cpp)
# add_standalone_test(test-wmma test_wmma.cpp)
# add_standalone_test(test-cc-register test_cc_register.cpp)
# add_standalone_test(test_ptx_bra test_ptx_bra.cu)
# add_standalone_test(warp/test_warp_context.cpp)
# add_standalone_test(warp/test_warp_scheduler.cpp)
# add_standalone_test(warp/test_sm_context.cpp)

add_standalone_test(test-ptx.cpp)
add_standalone_test(test_printf.cu)
add_standalone_test(test_wmma.cpp)
add_standalone_test(test_cc_register.cpp)
add_standalone_test(instructions/test_ptx_bra.cu)
add_standalone_test(warp/test_warp_context.cpp)
add_standalone_test(warp/test_warp_scheduler.cpp)
add_standalone_test(warp/test_sm_context.cpp)


# ================================
# 添加基于 CUDA 的测试（直接使用 CMake 编译）
# ================================

# 定义 mini tests (基础测试集)
set(PTX_MINI_TESTS 
    dummy dummy-args dummy-add dummy-float dummy-grid dummy-mul dummy-sub dummy-condition
    dummy-long dummy-sieve dummy-share dummy-loop)

set(PTX_BASIC_TESTS 
    simpleGEMM-int simpleGEMM-float simpleGEMM-double
    simpleCONV-int simpleCONV-float simpleCONV-double 2Dentropy
    aligned-types all-pairs-distance bitonic bfs )
    # aligned-types all-pairs-distance bitonic bfs backprop RAY cfd)

set(PTX_EXTRA_TESTS
    dummy-wmma)

# 定义全部测试 (包含基础测试和其他复杂测试)
set(PTX_ALL_TESTS 
    ${PTX_MINI_TESTS} 
    ${PTX_BASIC_TESTS}
    ${PTX_EXTRA_TESTS}
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
    list(FIND PTX_EXTRA_TESTS ${TEST_NAME} IS_EXTRA_TEST)
    if(NOT IS_EXTRA_TEST EQUAL -1)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "extra")
    endif()
    list(FIND PTX_MMA_TESTS ${TEST_NAME} IS_MMA_TEST)
    if(NOT IS_MMA_TEST EQUAL -1)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "mma")
    endif()
endforeach()