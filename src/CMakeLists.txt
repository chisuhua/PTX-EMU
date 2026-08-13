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
# Auto-discover cudart sources via GLOB. CONFIGURE_DEPENDS ensures cmake
# re-runs glob when new files appear.
file(GLOB SOURCES CONFIGURE_DEPENDS
    cudart/*.cpp
    cudart/cpptlm_bridge/*.cpp
    utils/cubin_utils.cpp
    ptxsim/atomic/atomic_mutex.cpp  # Phase 2 of implement-atomic-cas-and-true-atomicity
)
# ADR-0029 §D5: ptxemu_image_* ABI lives ONLY in libptxemu_device.so.
# Remove cpptlm_module.cpp from the cudart library GLOB so it does not leak
# into libcudart.so (verified via integration_phase0_byte_identical_gates Gate 1).
# All consumers (tests/unit/cudart/* + tests/integration/cpptlm/*) explicitly
# link ptxemu_device per openspec/specs/libptxemu-abi-freeze/spec.md.
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cudart/cpptlm_module.cpp")
set(SOURCES ${SOURCES} ${ANTLR4_GENERATED_SOURCES})

# 头文件目录
include_directories(${CUDA_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ANTLR4_GENERATED_SRC_DIR})

add_library(ptx_ir SHARED
    ptx_ir/ptx_types.cpp
    ptx_ir/operand_context.cpp
    ptx_ir/statement_context.cpp
    ptx_ir/ptx_syntax_utils.cpp
    ptx_ir/instruction_latency_table.cpp
)
target_include_directories(ptx_ir PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(ptxir)

add_library(ptxsim SHARED
    utils/logger.cpp
    utils/ptx_lane_verification.cpp
    ptxsim/core/cta_context.cpp
    ptxsim/core/thread_context.cpp
    ptxsim/core/simt_pc_manager.cpp
    ptxsim/core/register_access_layer.cpp
    ptxsim/core/warp_context.cpp
    ptxsim/core/warp_context_active_mask.cpp
    ptxsim/core/warp_context_simt.cpp
    ptxsim/core/warp_context_dispatch.cpp
    ptxsim/core/execution_tracer.cpp
    ptxsim/core/warp_scheduler.cpp
    ptxsim/core/sm_context.cpp
    ptxsim/core/sm_context_reconvergence.cpp
    ptxsim/core/sm_context_cpptlm_inject.cpp
    ptxsim/core/sm_block_dispatch.cpp
    ptxsim/core/sm_warp_lifecycle.cpp
    ptxsim/core/gpu_context.cpp
    register/condition_code_register.cpp
    register/register_bank_manager.cpp
    memory/hardware_memory_manager.cpp  # 添加新的硬件内存管理器
    memory/simple_memory.cpp
    memory/shared_memory_manager.cpp
    memory/resource_manager.cpp
    # Phase 0.1 (Fix #5): Blackwell TMA TensorMap descriptor parser.
    # Lives in ptxsim/memory/ (NEW dir for Phase 0 infra, distinct from
    # legacy src/memory/). Consumed by tcgen05.ld in Phase 1-3.
    ptxsim/memory/tma_descriptor.cpp
    # Phase 0.2 (Fix #6): Blackwell per-CTA Tensor Memory (TMEM).
    # 256 slot × 128 byte = 32 KB per CTA, parallel to shared memory.
    # Consumed by tcgen05.* in Phase 1-3.
    ptxsim/memory/tmem.cpp
    # Phase 1 of implement-tcgen05-handlers-extended (Oracle Q1-A): TMEM
    # slot allocator on top of Tmem, providing alloc/dealloc semantics
    # required by tcgen05.alloc / tcgen05.dealloc.
    ptxsim/memory/tmem_allocator.cpp
    # Phase 0.3 (Fix #7): cluster mode arrive/wait synchronization.
    # 1-8 CTA cluster identifier + arrive/wait primitives for cta_group::1.
    # NO distributed_smem (DEFERRED to cta_group::2 per Oracle simplification).
    # Consumed by tcgen05.* in Phase 1-3.
    ptxsim/cluster/cluster_context.cpp
    # Phase 0.4 (Fix #8): Blackwell per-CTA async tensor core command queue.
    # std::atomic commit-group counter + per-warp pending_waiters for
    # tcgen05.* async commit/wait primitives.
    # Consumed by tcgen05.* in Phase 1-3.
    ptxsim/async/tc_queue.cpp
    ptxsim/debug/ptx_config.cpp
    ptxsim/debug/warp_trace_formatter.cpp
    ptxsim/core/simt_stack.cpp
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
    ptxsim/instructions/tcgen05.cpp
    # Phase 1 of implement-tcgen05-handlers-extended (Oracle Q1-A/Q2-A):
    # 3 alloc-family handlers (alloc/dealloc/relinquish_alloc_permit).
    ptxsim/instructions/tcgen05_alloc.cpp
    # Phase 2 (Oracle Q4-B/Q2-A): tcgen05.cp (smem → tmem copy).
    ptxsim/instructions/tcgen05_cp.cpp
    # Phase 2.5 (Oracle 2026-07-08 Q4-recommendation, pre-Phase 3 refactor):
    # extracted fragment_mma_f16 helper shared by regular mma + ws routing.
    ptxsim/instructions/tcgen05_helpers.cpp
    # Phase 4 of implement-tcgen05-handlers-extended (Oracle Q6-B):
    # tcgen05.fence no-op marker (design D8).
    ptxsim/instructions/tcgen05_fence.cpp
    ptxsim/instructions/arithmetic_ext.cpp
    ptxsim/instructions/cvt/cvt_helpers.cpp
    ptxsim/instructions/cvt/cvt_strategy.cpp
    ptxsim/instructions/cvt/cvt_float_to_float.cpp
    ptxsim/instructions/cvt/cvt_int_to_float.cpp
    ptxsim/instructions/cvt/cvt_float_to_int.cpp
    ptxsim/instructions/cvt/cvt_int_to_int.cpp
    ptxsim/instructions/data_transfer.cpp
    ptxsim/instructions/call.cpp
    ptxsim/instructions/barrier.cpp
    ptxsim/barrier/warp_barrier.cpp
    ptxsim/barrier/cta_barrier.cpp
    ptxsim/barrier/barrier_module.cpp
)
target_link_libraries(ptxsim ptx_ir)

add_library(ptx_parser SHARED
    ptx_parser/ptx_visitor.cpp
    ptx_parser/cfg_builder.cpp
)
target_link_libraries(ptx_parser ptx_ir)
add_dependencies(ptx_parser GenerateParser)

add_subdirectory(ptx_ir)

# 创建共享库
add_library(cudart SHARED ${SOURCES})
target_link_libraries(cudart ptx_ir ptx_parser ptxsim ptxir)
add_dependencies(cudart GenerateParser)
add_dependencies(cudart ptx_parser)
add_dependencies(cudart ptxsim)
add_dependencies(cudart ptxir)

# Phase 1: PTX-EMU Image Executor shared library (ADR-0029 §D5)
add_library(ptxemu_device SHARED
    cudart/cpptlm_module.cpp
)
target_link_libraries(ptxemu_device PUBLIC ptxsim ptx_ir ptxir)
target_include_directories(ptxemu_device PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(ptxemu_device PROPERTIES
    VERSION 12.0
    SOVERSION 12
    POSITION_INDEPENDENT_CODE ON
)
install(TARGETS ptxemu_device LIBRARY DESTINATION lib)

add_custom_command(TARGET ptxemu_device POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libptxemu_device.so
        ${CMAKE_SOURCE_DIR}/lib/libptxemu_device.so
    COMMENT "Creating symlink to libptxemu_device.so in project root lib directory"
)

# 标记ANTLR生成的源文件为生成文件
set_source_files_properties(${ANTLR4_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# 设置库属性
set_target_properties(cudart PROPERTIES
    VERSION 12.0
    SOVERSION 12
    LIBRARY_OUTPUT_NAME "cudart"
)

# 链接库
target_link_libraries(cudart ${ANTLR4_LIBRARY} pthread OpenSSL::Crypto)
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
