# add-cmake-options - Design

## Overview

在根 `CMakeLists.txt` 中添加 ASAN/UBSAN/WERROR 三个开发构建选项，使开发者可通过标准 CMake 选项启用 sanitizer 和严格警告检查。所有选项默认 OFF，不影响现有构建行为。

当前状态：根 `CMakeLists.txt` 在 line 34 仅有 `option(USE_DETAILED_LOGGING ...)`。sanitizer 逻辑应紧跟此 option 定义块之后。

## Design Decisions

### 决策 1: Option 定义位置

**选择**: 在根 `CMakeLists.txt` 的 `USE_DETAILED_LOGGING` option 之后（约 line 37-38），`CMAKE_EXPORT_COMPILE_COMMANDS` 之前添加新 option 块

**理由**:
- 所有 option 集中管理，符合现有风格
- 在 `add_subdirectory(src)` 之前定义，确保所有子目标继承 flag
- 开发者通过 `cmake -L` 可在统一位置查看所有选项

### 决策 2: Sanitizer flag 拼接策略

**选择**: 使用 `add_compile_options()` + `add_link_options()` 全局应用 flag

**理由**:
- 全局应用确保所有目标（cudart、ptxsim、ptx_ir、ptx_parser、tests）均受影响
- `add_compile_options()` 在 `add_subdirectory()` 之前调用，子目录自动继承

**实现伪码**:
```cmake
# Sanitizers
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_WERROR "Treat warnings as errors" OFF)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

# ASAN + UBSAN 可同时启用（flag 自动叠加）

if(ENABLE_WERROR)
    add_compile_options(-Werror)
endif()
```

**替代方案**:
- A. 按目标设置 `target_compile_options()` -> 需在每个 target 重复，遗漏风险高
- B. 使用 `CMAKE_CXX_FLAGS` 变量 -> 不推荐（CMake 官方建议用 `add_compile_options`）
- C. **采用**: 全局 `add_compile_options()` + `add_link_options()`

### 决策 3: ASAN + UBSAN 共存

**选择**: 两个 option 独立检查，flag 自动叠加

**理由**:
- GCC/Clang 支持 `-fsanitize=address,undefined` 同时启用
- 独立检查使开发者可单独启用任一 sanitizer
- flag 叠加由编译器自动处理，无需额外逻辑

**验证**: `cmake -DENABLE_ASAN=ON -DENABLE_UBSAN=ON ..` 应生成包含两个 sanitizer 的编译命令

### 决策 4: 是否添加 BUILD_TESTS / BUILD_BENCH

**选择**: 作为可选任务，优先级低于三个核心 option

**理由**:
- 当前 `enable_testing()` 和 `add_subdirectory(tests)` 是无条件执行
- 添加 `BUILD_TESTS` 开关可加速纯库构建（跳过测试编译）
- 但这改变了默认行为（需默认 ON），增加了复杂度
- 决定：列为可选任务（SHOULD），不阻塞核心验收

### 决策 5: WERROR 与 CUDA 编译器

**选择**: WERROR 仅应用于 CXX 编译器，不应用于 CUDA 编译器（nvcc）

**理由**:
- nvcc 警告与 host 编译器不同，且部分 nvcc 警告无法消除
- `-Werror` 传给 nvcc 需用 `--Werror` 或 `-Xcompiler -Werror`
- 为简化，WERROR 仅影响 CXX 代码

**实现**:
```cmake
if(ENABLE_WERROR)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
```

## Implementation Plan

### Phase 1: 添加 option 定义和 sanitizer 逻辑
1. 在 `CMakeLists.txt` 的 `USE_DETAILED_LOGGING` 之后添加 3 个 option 定义
2. 添加 `if(ENABLE_ASAN)` / `if(ENABLE_UBSAN)` / `if(ENABLE_WERROR)` 条件块
3. 验证: `cmake -L ..` 显示所有新 option

### Phase 2: 验证各选项独立工作
1. `cmake -DENABLE_ASAN=ON ..` 构建 + 运行测试
2. `cmake -DENABLE_UBSAN=ON ..` 构建 + 运行测试
3. `cmake -DENABLE_WERROR=ON ..` 构建验证（可能发现现有警告）

### Phase 3: 验证选项组合
1. `cmake -DENABLE_ASAN=ON -DENABLE_UBSAN=ON ..` 构建 + 运行测试
2. 默认构建（无选项）验证行为不变

## Testing Strategy

| 测试场景 | 命令 | 预期 |
|---------|------|------|
| 默认构建 | `cmake -S . -B build && cmake --build build` | 行为与变更前一致 |
| ASAN 构建 | `cmake -DENABLE_ASAN=ON -S . -B build-asan && cmake --build build-asan` | 构建通过，测试可运行 |
| UBSAN 构建 | `cmake -DENABLE_UBSAN=ON -S . -B build-ubsan && cmake --build build-ubsan` | 构建通过，测试可运行 |
| ASAN+UBSAN | `cmake -DENABLE_ASAN=ON -DENABLE_UBSAN=ON -S . -B build-san && cmake --build build-san` | 构建通过，测试可运行 |
| WERROR 构建 | `cmake -DENABLE_WERROR=ON -S . -B build-werror && cmake --build build-werror` | 构建通过（或暴露已有警告） |
| 选项可见性 | `cmake -L ..` | 所有 4+ option 可见（含 USE_DETAILED_LOGGING） |

### ASAN 运行验证

```bash
cmake -DENABLE_ASAN=ON -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug
cmake --build build-asan
cd build-asan && ctest --output-on-failure
# ASAN 报告内存错误（如有），否则正常退出
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| ASAN 与 CUDA 代码冲突 | nvcc 编译的 .cu 文件可能不兼容 ASAN | ASAN flag 仅应用于 CXX；如需 CUDA ASAN 需额外配置 |
| WERROR 暴露大量已有警告 | 构建失败 | WERROR 默认 OFF；开发者按需启用；暴露的警告可后续修复 |
| sanitizer 运行时性能下降 | 测试变慢 | 仅在 Debug + sanitizer 构建中使用，不影响 Release |
| CppTLM 子项目继承 sanitizer flag | CppTLM 构建可能失败 | `add_compile_options` 在 `add_subdirectory(CppTLM)` 之前设置，会继承；如有问题可用 `target_compile_options` 替代 |

## Open Questions

1. **是否为 ASAN 构建自动设置 Debug 构建类型？**
   - 推荐：NO（开发者应显式选择 `-DCMAKE_BUILD_TYPE=Debug`）
   - 决定：不自动设置，文档中推荐搭配 Debug 使用

2. **是否添加 `ENABLE_TSAN` (ThreadSanitizer)？**
   - 推荐：暂不添加（ASAN 已覆盖大部分内存问题，TSAN 与 ASAN 互斥）
   - 决定：Out Scope，未来可扩展

## 关联文档

- `improvements/add-cmake-options.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-10`：原债务条目
- `CMakeLists.txt`：当前根 CMake 配置（line 34: 唯一 option）
