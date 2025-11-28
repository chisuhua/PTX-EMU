# 生成ANTLR解析器
find_program(ANTLR4_CC grammars-v4/ANTLRv4Parser.g4 JAR_FILE ${ANTLR_EXECUTABLE})

# 定义ANTLR生成的源文件列表
set(ANTLR4_GENERATED_SOURCES
    ${ANTLR4_GENERATED_SRC_DIR}/ptxLexer.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParser.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserVisitor.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserBaseVisitor.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserListener.cpp
    ${ANTLR4_GENERATED_SRC_DIR}/ptxParserBaseListener.cpp
)

# 定义ANTLR生成目标
add_custom_command(
    OUTPUT ${ANTLR4_GENERATED_SOURCES}
    COMMAND java -Xss500M -Xmx500M -cp "${ANTLR_EXECUTABLE}" org.antlr.v4.Tool 
        -Dlanguage=Cpp ptxLexer.g4 ptxParser.g4 
        -listener -visitor -package ptxparser 
        -o ${ANTLR4_GENERATED_SRC_DIR}
    DEPENDS ptxLexer.g4 ptxParser.g4
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating parser from ANTLR grammar files..."
    VERBATIM
)

add_custom_target(GenerateParser DEPENDS ${ANTLR4_GENERATED_SOURCES})

# 设置源文件（包含ANTLR生成的源文件）
set(SOURCES
    PTXEMU.cpp
    ${ANTLR4_GENERATED_SOURCES}
)

# 头文件目录
include_directories(${CUDA_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 创建共享库
add_library(cudart SHARED ${SOURCES})
add_dependencies(cudart GenerateParser)

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
target_include_directories(cudart PRIVATE ${CUDA_PATH}/include)

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