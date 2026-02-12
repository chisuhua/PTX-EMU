# 生成ANTLR解析器
find_program(ANTLR4_CC grammars-v4/ANTLRv4Parser.g4 JAR_FILE ${ANTLR_EXECUTABLE})

# 定义ANTLR生成的源文件列表
set(ANTLR4_GENERATED_SOURCES
    ${ANTLR4_GENERATED_SRC_DIR}/ptxLexer.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxLexer.h
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParser.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParser.h
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserVisitor.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserVisitor.h
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserBaseVisitor.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserBaseVisitor.h
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserListener.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserListener.h
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserBaseListener.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserBaseListener.h
)

# 定义ANTLR生成目标 - 再生成语法分析器
add_custom_command(
    OUTPUT ${ANTLR4_GENERATED_SOURCES}
    COMMAND java -Xss500M -Xmx500M -cp "${ANTLR_EXECUTABLE}" org.antlr.v4.Tool
        -Dlanguage=Cpp ptxLexer.g4 ptxParser.g4
        -listener -visitor -package ptxparser
        -o ${ANTLR4_GENERATED_SRC_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/grammar/ptxLexer.g4 ${CMAKE_CURRENT_SOURCE_DIR}/grammar/ptxParser.g4
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/grammar
    COMMENT "Generating parser from ANTLR grammar file..."
    VERBATIM
)

add_custom_target(GenerateParser
    DEPENDS 
        ${ANTLR4_GENERATED_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/grammar/ptxLexer.g4
        ${CMAKE_CURRENT_SOURCE_DIR}/grammar/ptxParser.g4
)

# 设置源文件（包含ANTLR生成的源文件）
set(SOURCES
    cudart/ptx_interpreter.cpp
    cudart/cuda_driver.cpp  # 添加新的驱动内存管理器
    cudart/simple_memory_allocator.cpp
    cudart/cudart_sim.cpp
    ${ANTLR4_GENERATED_SOURCES}
)

# 头文件目录
include_directories(${CUDA_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ANTLR4_GENERATED_SRC_DIR})

add_library(ptx_ir SHARED
    ptx_ir/ptx_types.cpp
    ptx_ir/operand_context.cpp
    ptx_ir/statement_context.cpp
)
target_include_directories(ptx_ir PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(ptxsim SHARED
    utils/logger.cpp
    ptxsim/core/cta_context.cpp
    ptxsim/core/thread_context.cpp
    ptxsim/core/warp_context.cpp
    ptxsim/core/warp_scheduler.cpp
    ptxsim/core/sm_context.cpp
    ptxsim/core/gpu_context.cpp
    ptxsim/core/active_warp_manager.cpp
    register/condition_code_register.cpp
    register/register_bank_manager.cpp
    memory/hardware_memory_manager.cpp  # 添加新的硬件内存管理器
    memory/simple_memory.cpp
    memory/shared_memory_manager.cpp
    memory/resource_manager.cpp
    ptxsim/debug/ptx_config.cpp
    ptxsim/debug/ptx_debugger.cpp
    ptxsim/instruction_factory.cpp
    ptxsim/instruction_base.cpp
    ptxsim/instruction_handlers.cpp
    ptxsim/register_analyzer.cpp
    ptxsim/utils/qualifier_utils.cpp
    ptxsim/utils/type_utils.cpp
    ptxsim/utils/half_utils.cpp
    ptxsim/instructions/arithmetic.cpp
    ptxsim/instructions/arithmetic_muldiv.cpp
    ptxsim/instructions/bitwise.cpp
    ptxsim/instructions/math.cpp
    ptxsim/instructions/memory.cpp
    ptxsim/instructions/control.cpp
    ptxsim/instructions/comparison.cpp
    ptxsim/instructions/atomic.cpp
    ptxsim/instructions/tensor.cpp
    ptxsim/instructions/arithmetic_ext.cpp
    ptxsim/instructions/arithmetic_conversion.cpp
    ptxsim/instructions/data_transfer.cpp
    ptxsim/instructions/call.cpp
)
target_link_libraries(ptxsim ptx_ir)

add_library(ptx_parser SHARED
    ptx_parser/ptx_visitor.cpp
)
target_link_libraries(ptx_parser ptx_ir)
add_dependencies(ptx_parser GenerateParser)

# 创建共享库
add_library(cudart SHARED ${SOURCES})
target_link_libraries(cudart ptx_ir ptx_parser ptxsim)
target_link_libraries(cudart ptx_ir ptxsim)
add_dependencies(cudart GenerateParser)
add_dependencies(cudart ptx_parser)
add_dependencies(cudart ptxsim)

# 标记ANTLR生成的源文件为生成文件
set_source_files_properties(${ANTLR4_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# 设置库属性
set_target_properties(cudart PROPERTIES
    VERSION 12.0
    SOVERSION 12
    LIBRARY_OUTPUT_NAME "cudart"
)

# 链接库
target_link_libraries(cudart ${ANTLR4_LIBRARY} pthread)

# 确保CUDA包含目录正确设置
target_include_directories(cudart PRIVATE ${CUDA_PATH}/include ${CUDAToolkit_INCLUDE_DIRS})

# 创建版本脚本
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/linux-so-version.txt
    ${CMAKE_CURRENT_BINARY_DIR}/linux-so-version.txt
    COPYONLY
)

# 应用版本脚本
target_link_options(cudart PRIVATE 
    "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/linux-so-version.txt"
)

# 添加安装规则
install(TARGETS cudart
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

# 创建符号链接，将构建生成的库文件链接到项目根目录的lib目录
add_custom_command(TARGET cudart POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/lib/libcudart.so*
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libcudart.so
        ${CMAKE_SOURCE_DIR}/lib/libcudart.so
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libcudart.so.12.0
        ${CMAKE_SOURCE_DIR}/lib/libcudart.so.12.0
    COMMENT "Creating symlinks to libcudart.so in project root lib directory"
)

# 添加test-ptx可执行文件目标
