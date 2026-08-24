# ptxemu-core-library Specification

## Purpose
TBD - created by archiving change ptxemu-public-device-api. Update Purpose after archive.
## Requirements
### Requirement: `ptxemu_core` CMake 库目标 MUST 可被 `add_subdirectory` 消费

`add_library(ptxemu_core STATIC ...)` MUST 在 PTX-EMU 仓 root CMakeLists.txt 中定义, 且消费方 MUST 能通过 `add_subdirectory(external/PTX-EMU)` 链接到 `ptxemu_core`。

#### Scenario: 消费方 add_subdirectory 零修改即可链接
- **WHEN** CppTLM 仓 `add_subdirectory(external/PTX-EMU)` + `target_link_libraries(cpptlm_core PUBLIC ptxemu_core)`
- **THEN** 无 undefined reference 错误, 无 missing include 错误

### Requirement: PUBLIC/PRIVATE include 拆分

`ptxemu_core` MUST 通过 `target_include_directories` 强制拆分:
- **PUBLIC**: `${CMAKE_CURRENT_SOURCE_DIR}/include/ptxemu` (CppTLM 可见)
- **PRIVATE**: `${CMAKE_CURRENT_SOURCE_DIR}/include/ptx_ir` 等内部头 (Phase 1 净化后变 PUBLIC) + `${CMAKE_CURRENT_SOURCE_DIR}/include/ptxir` + `${CMAKE_CURRENT_SOURCE_DIR}/include/ptxsim` + `${CMAKE_CURRENT_SOURCE_DIR}/src/ptxsim` + `${CMAKE_CURRENT_SOURCE_DIR}/src/cudart` (仅内部)

#### Scenario: 内部头对 CppTLM 不可见
- **WHEN** CppTLM 编译 TU 中 `find . -name '*.cc'` 列出所有 `#include`
- **THEN** 0 处引用 `${PTXEMU_SRC}/include/ptxsim/` 或 `${PTXEMU_SRC}/include/ptx_ir/statement_context.h`(直接引用, 而非通过 `ptxemu/ir/statement.h` 转发)

### Requirement: `option(PTXEMU_BUILD_TESTING OFF)` 默认值

PTX-EMU root CMakeLists.txt MUST 在顶部声明 `option(PTXEMU_BUILD_TESTING "Build PTX-EMU tests" OFF)`。PTX-EMU 自身顶层构建时默认 ON,被 `add_subdirectory` 消费时默认 OFF。

#### Scenario: CppTLM 消费时不触发 PTX-EMU 测试构建
- **WHEN** `add_subdirectory(external/PTX-EMU)` 时 `PTXEMU_BUILD_TESTING` 未设置 (默认 OFF)
- **THEN** PTX-EMU 的 `tests/` 子目录不被处理, 无 `add_test()` 注册

### Requirement: `if(PROJECT_IS_TOP_LEVEL)` 隔离模式

PTX-EMU root CMakeLists.txt MUST 使用 `if(PROJECT_IS_TOP_LEVEL OR PTXEMU_BUILD_TESTING)` 包裹 `enable_testing() + add_subdirectory(tests)`。PTX-EMU 顶层构建 (独立) 时自动启用测试;被消费时仅在 `PTXEMU_BUILD_TESTING=ON` 时启用。

#### Scenario: PROJECT_IS_TOP_LEVEL 自动检测
- **WHEN** PTX-EMU 顶层 cmake build (`-S . -B build`)
- **THEN** `PROJECT_IS_TOP_LEVEL=TRUE` (CMake 内置变量), `enable_testing()` 自动调用

#### Scenario: add_subdirectory 消费时不启用测试
- **WHEN** CppTLM `add_subdirectory(external/PTX-EMU)` (PROJECT_IS_TOP_LEVEL 仍为 CppTLM=TRUE)
- **THEN** PTX-EMU 的 `enable_testing()` 与 `add_subdirectory(tests)` 跳过

### Requirement: install 规则

PTX-EMU MUST 提供 `install(TARGETS ptxemu_core EXPORT ptxemu_core_targets ARCHIVE DESTINATION lib INCLUDES DESTINATION include)` 规则 (HSK-8 spec §3 Risk 4 Mitigation)。

#### Scenario: CppTLM 安装模式兼容
- **WHEN** CppTLM 仓启用 `install` 模式 (即使本次不实施)
- **THEN** PTX-EMU `ptxemu_core` 库随 CppTLM install 一并安装 (CMake 链式 install)

