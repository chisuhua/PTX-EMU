# ptx-debugger-cleanup Specification

## Purpose
PTXDebugger 整个模块（311 LOC + 14 LOC stub）零生产调用方，已删除。

## Requirements
### Requirement: PTXDebugger-Module-Deleted MUST

PTXDebugger 整个模块（包含 header + cpp + debug/ 目录）已被删除。`libptxsim.a` 不再包含 PTXDebugger 符号。

#### Scenario: Header-Removed
- **WHEN** searching for `ptxsim/ptx_debugger.h`
- **THEN** 文件不存在

#### Scenario: Debug-Directory-Removed
- **WHEN** `ls src/ptxsim/debug/` is executed
- **THEN** 目录不存在

#### Scenario: Zero-Regression
- **WHEN** ctest runs after deletion
- **THEN** 100% PASS（零回归）