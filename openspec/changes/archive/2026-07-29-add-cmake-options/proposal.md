# add-cmake-options - Proposal

## Why

根 `CMakeLists.txt` 仅有 **1 个 cmake option**（`USE_DETAILED_LOGGING`）。缺少常用开发选项：

- **ASAN** (AddressSanitizer)：模拟器有并发 warp 调度 + 共享内存模拟，内存安全检测尤为重要
- **UBSAN** (UndefinedBehaviorSanitizer)：C++20 代码中大量 reinterpret_cast / variant 访问，UB 检测有价值
- **WERROR**：将编译警告视为错误，防止警告累积导致真正问题被淹没

缺少这些选项导致：
- 开发者无法通过 CMake 标准方式启用 sanitizer，需手动传编译器 flag
- CI 无法标准化内存安全检测流程
- 新贡献者不知道有哪些可用构建选项

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-10`

## What Changes

- **新增** `ENABLE_ASAN` option - AddressSanitizer 支持（默认 OFF）
- **新增** `ENABLE_UBSAN` option - UndefinedBehaviorSanitizer 支持（默认 OFF）
- **新增** `ENABLE_WERROR` option - 将警告视为错误（默认 OFF）
- **可选新增** `BUILD_TESTS` / `BUILD_BENCH` 开关 - 控制测试和基准构建

所有新选项默认 OFF，不破坏现有构建行为。

## Capabilities

### New Capabilities
- `cmake-asan-option`: 通过 `-DENABLE_ASAN=ON` 启用 AddressSanitizer
- `cmake-ubsan-option`: 通过 `-DENABLE_UBSAN=ON` 启用 UndefinedBehaviorSanitizer
- `cmake-werror-option`: 通过 `-DENABLE_WERROR=ON` 将编译警告视为错误

### Modified Capabilities
- `cmake-build-config`: 根 CMakeLists.txt 新增 3+ 个 option 定义及条件编译逻辑

## Impact

**受影响代码**：
- `CMakeLists.txt`（根，新增 option 定义 + sanitizer flag 逻辑）

**不受影响**：
- `src/CMakeLists.txt`（子目录，不直接修改）
- `tests/CMakeLists.txt`
- 源代码文件（无代码变更）
- 默认构建行为（所有新选项默认 OFF）

**依赖**：
- 无前置 change 依赖，可独立执行
- ASAN 和 UBSAN 可同时启用（需正确拼接 flag）

**工时**: 0.5-1h（纯 CMake 配置变更）
