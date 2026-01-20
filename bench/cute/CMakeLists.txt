# bench/cute/CMakeLists.txt
# 用于构建cute子目录下的CUDA测试

# 设置CMAKE_CUDA_RUNTIME_LIBRARY
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)       # 全局强制PTX模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF) # 禁用分离编译

# 设置CUDA编译标志，保留中间PTX文件
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep")

# 获取当前目录下所有.cu文件
file(GLOB CUTE_CUDA_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# 为每个CUDA文件创建测试
foreach(CUDA_FILE ${CUTE_CUDA_FILES})
    # 获取文件名（不含扩展名）
    get_filename_component(TEST_NAME ${CUDA_FILE} NAME_WE)
    
    # 设置完整测试名称
    set(FULL_TEST_NAME "${TEST_NAME}")
    
    # 设置对象文件
    set(OBJECT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cu.o")
    
    # 添加自定义命令来编译CUDA文件为对象文件
    add_custom_command(
        OUTPUT ${OBJECT_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER}
            -cudart=shared
            -g -std=c++17
            -arch=compute_100
            -code=compute_100
            -I${PROJECT_SOURCE_DIR}/include
            -I${CUDA_PATH}/include
            -I${PROJECT_SOURCE_DIR}/bench/cute/include
            -x cu
            -c ${CUDA_FILE}
            -o ${OBJECT_FILE}
        DEPENDS ${CUDA_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
    
    # 创建可执行文件
    add_executable(${FULL_TEST_NAME} "")
    
    target_sources(${FULL_TEST_NAME} PRIVATE ${OBJECT_FILE}) 
    
    set_target_properties(${FULL_TEST_NAME} PROPERTIES
        LINKER_LANGUAGE CXX  # 必须显式指定!
        SUFFIX ""            # 防止添加.a后缀
        OUTPUT_NAME "${FULL_TEST_NAME}"
        PREFIX ""  # 防止添加lib前缀
    )
           
    # 包含 CUDA 头文件
    target_include_directories(${FULL_TEST_NAME} PRIVATE ${CUDA_PATH}/include  ${PROJECT_SOURCE_DIR}/bench/cute/include)
    
    # 链接我们自己构建的 cudart 库和 ANTLR 库
    target_link_directories(${FULL_TEST_NAME} PRIVATE 
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        ${CMAKE_SOURCE_DIR}/lib
    )
    
    target_link_libraries(${FULL_TEST_NAME} PRIVATE
        -Wl,--no-as-needed
        cudart
        antlr4_shared
        -Wl,--as-needed  # 恢复默认行为
        -ldl -lpthread
    )
    
    # 设置 RPATH，确保运行时能找到我们自己的库
    set_target_properties(${FULL_TEST_NAME} PROPERTIES
        INSTALL_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH TRUE            
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
    
    add_dependencies(${FULL_TEST_NAME} cudart antlr4_shared)  # 确保库先构建
    
    add_custom_target(${FULL_TEST_NAME}_obj DEPENDS ${OBJECT_FILE})
    add_dependencies(${FULL_TEST_NAME} ${FULL_TEST_NAME}_obj)
    
    # 添加测试
    add_test(NAME ${FULL_TEST_NAME} COMMAND ${FULL_TEST_NAME})
    
    # 标记为cute测试
    set_tests_properties(${FULL_TEST_NAME} PROPERTIES LABELS "cute")
endforeach()