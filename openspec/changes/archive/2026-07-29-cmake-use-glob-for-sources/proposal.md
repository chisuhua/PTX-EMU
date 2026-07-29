# cmake-use-glob-for-sources - Proposal

## Why

`src/CMakeLists.txt` 手动维护 **77 个 .cpp 文件**列表（跨 4 个目标）：
- `SOURCES` (cudart): 7 个文件
- `ptx_ir` 库: 5 个文件
- `ptxsim` 库: 63 个文件
- `ptx_parser` 库: 2 个文件

问题：
- 新增源文件需手动添加到 `CMakeLists.txt`，容易遗漏导致链接错误
- 维护成本随文件数量线性增长
- 文件列表注释（如 `# Phase 0.1 (Fix #5)`）与实际文件耦合，移动文件时需同步修改

`file(GLOB ... CONFIGURE_DEPENDS)` 可自动发现新文件，配合 `CONFIGURE_DEPENDS` 实现增量检测，消除手动维护负担。

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-9`

## What Changes

- **替换** `src/CMakeLists.txt` 中 4 处手动 `set(SOURCES ...)` / `add_library(... explicit list)` 为 `file(GLOB ... CONFIGURE_DEPENDS)`
- **保留** `src/ptx_ir/CMakeLists.txt` 和 `src/ptxir/CMakeLists.txt` 中的手动列表（文件数 ≤ 1，GLOB 无收益）
- **不修改** `tests/CMakeLists.txt`（测试文件通常需显式控制）
- **保留** ANTLR 生成源文件的手动列表（这些是生成文件，不应 GLOB）

## Capabilities

### New Capabilities
- `cmake-glob-sources`: 自动发现 src/ 下 .cpp 文件，无需手动维护 CMakeLists.txt 源文件列表

### Modified Capabilities
- `cmake-build-config`: src/CMakeLists.txt 从手动源文件列表改为 GLOB 自动发现

## Impact

**受影响代码**：
- `src/CMakeLists.txt`（4 处源文件列表替换为 GLOB）

**不受影响**：
- `src/ptx_ir/CMakeLists.txt`（2 个 target 各 1 个文件，保持手动）
- `src/ptxir/CMakeLists.txt`（1 个文件，保持手动）
- `tests/CMakeLists.txt`（测试文件保持显式控制）
- `CMakeLists.txt`（根，不涉及源文件列表）
- 编译选项、链接配置不变
- `build.sh` / `env.sh` 不受影响

**依赖**：
- 无前置 change 依赖，可独立执行
- 建议在 `add-cmake-options` 之后执行（可选，无硬依赖）

**工时**: 0.5-1h（CMake 配置变更 + 全量验证）
