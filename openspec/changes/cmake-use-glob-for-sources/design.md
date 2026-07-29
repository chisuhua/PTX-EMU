# cmake-use-glob-for-sources - Design

## Overview

将 `src/CMakeLists.txt` 中 4 处手动维护的 .cpp 文件列表替换为 `file(GLOB ... CONFIGURE_DEPENDS)` 自动发现机制，使新增源文件无需修改 CMakeLists.txt 即可被自动纳入编译。

## Context

当前 `src/CMakeLists.txt` 包含 4 处手动源文件列表：

| 目标 | 行号 | 文件数 | 内容 |
|------|------|--------|------|
| `SOURCES` (cudart) | 41-50 | 7 | cudart/ + utils/ + atomic/ + ANTLR 生成 |
| `ptx_ir` 库 | 58-64 | 5 | ptx_ir/ 下类型实现 |
| `ptxsim` 库 | 69-157 | 63 | ptxsim/ 全部子目录 |
| `ptx_parser` 库 | 160-163 | 2 | ptx_parser/ visitor + cfg |

手动维护问题：新增文件（如 `ptx_visitor_tcgen05.cpp`、`tcgen05_alloc.cpp` 等）需手动添加到对应列表，遗漏导致链接错误。

## Design Decisions

### 决策 1: GLOB 范围 - 按库目标分组 GLOB

**选择**: 每个库目标使用独立的 `file(GLOB ...)` 匹配其对应目录

**GLOB 模式**:
```cmake
# cudart SOURCES (cudart/ + utils/ + atomic/)
file(GLOB CUDART_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/cudart/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cudart/cpptlm_bridge/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ptxsim/atomic/*.cpp
)
set(SOURCES ${CUDART_SOURCES} ${ANTLR4_GENERATED_SOURCES})

# ptx_ir 库
file(GLOB PTX_IR_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/ptx_ir/*.cpp
)

# ptxsim 库
file(GLOB PTXSIM_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/ptxsim/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ptxsim/**/*.cpp
)
# 或使用 RECURSE: file(GLOB_RECURSE PTXSIM_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ptxsim/*.cpp)

# ptx_parser 库
file(GLOB PTX_PARSER_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/ptx_parser/*.cpp
)
```

**理由**:
- 按库目标分组匹配，避免文件被错误纳入不相关的库
- `CONFIGURE_DEPENDS` 确保 CMake 在增量构建时检查文件变化
- 非递归 `GLOB` 更精确（不意外包含深层子目录文件）

**替代方案**:
- A. 单一 `GLOB_RECURSE` 匹配所有 `src/**/*.cpp` -> 无法区分哪些文件属于哪个库
- B. 保持手动列表 -> 维护成本持续增长
- C. **采用**: 按库目标的独立 GLOB

### 决策 2: GLOB vs GLOB_RECURSE

**选择**: ptxsim 使用 `GLOB_RECURSE`（有深层子目录），其他使用 `GLOB`

**理由**:
- `ptxsim` 目录结构深（`ptxsim/instructions/cvt/`、`ptxsim/memory/`、`ptxsim/barrier/` 等），需递归
- `cudart/`、`ptx_ir/`、`ptx_parser/` 目录浅（仅一层），非递归即可
- `cudart/cpptlm_bridge/` 子目录需单独 GLOB（1 个文件）

**实现**:
```cmake
# ptxsim 需递归
file(GLOB_RECURSE PTXSIM_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/ptxsim/*.cpp
)
```

### 决策 3: CONFIGURE_DEPENDS 的必要性

**选择**: 所有 GLOB 调用必须使用 `CONFIGURE_DEPENDS`

**理由**:
- CMake 官方文档推荐：使用 GLOB 时搭配 `CONFIGURE_DEPENDS` 确保 CMake 检测文件增删
- 不加 `CONFIGURE_DEPENDS` 的 GLOB 只在 CMake 首次配置时扫描文件，后续新增文件不会被检测
- `CONFIGURE_DEPENDS` 会在每次构建时检查文件列表变化，有微小性能开销但确保正确性

### 决策 4: 排除不应编译的文件

**选择**: 目前无需排除（所有 src/ 下的 .cpp 都应被编译）

**理由**:
- 当前 94 个 .cpp 文件全部纳入编译，无需要排除的文件
- 如未来有需要排除的文件，可用 `list(FILTER ...)` 或 `list(REMOVE_ITEM ...)` 处理

**预留**: 在 GLOB 后添加注释说明排除策略：
```cmake
# 如需排除特定文件，使用:
# list(FILTER PTXSIM_SOURCES EXCLUDE REGEX "pattern")
```

### 决策 5: 不修改 tests/CMakeLists.txt

**选择**: tests 目录保持手动列表

**理由**:
- 测试文件通常需显式控制（某些测试需特定编译选项或链接依赖）
- 测试文件命名需精确匹配 ctest 注册名
- 测试目录 CMakeLists.txt 结构更复杂（有条件编译、CUDA 测试等）
- 改动风险高于 src/ 目录

### 决策 6: 不修改 src/ptx_ir/CMakeLists.txt 和 src/ptxir/CMakeLists.txt

**选择**: 子目录 CMakeLists.txt 中的 1-2 个文件列表保持手动

**理由**:
- `src/ptx_ir/CMakeLists.txt`: `ptxir_writer` (1 文件) + `ptxir_reader` (1 文件) -> GLOB 无收益
- `src/ptxir/CMakeLists.txt`: `ptxir_serialization` (1 文件) -> GLOB 无收益
- 这些列表极短且极少变化

## Implementation Plan

### Phase 1: 替换 4 处源文件列表为 GLOB
1. 替换 `SOURCES` (cudart) 列表为 GLOB + 保留 ANTLR 生成文件
2. 替换 `ptx_ir` 库文件列表为 GLOB
3. 替换 `ptxsim` 库文件列表为 GLOB_RECURSE
4. 替换 `ptx_parser` 库文件列表为 GLOB
5. 验证: `cmake -S . -B build && cmake --build build` 通过

### Phase 2: 全量验证
1. `ctest` 全绿
2. 验证新增文件无需修改 CMakeLists.txt 即可编译（创建临时测试文件）
3. 验证 `CONFIGURE_DEPENDS` 增量检测工作

## Testing Strategy

| 测试场景 | 命令 | 预期 |
|---------|------|------|
| 全量编译 | `cmake -S . -B build && cmake --build build` | 通过，无链接错误 |
| 全量测试 | `cd build && ctest --output-on-failure` | 全绿 |
| GLOB 正确性 | 对比 GLOB 结果与 git 跟踪文件列表 | 一致 |
| 新增文件检测 | 创建临时 .cpp 文件，增量构建 | 自动纳入编译 |
| CONFIGURE_DEPENDS | 删除 .cpp 文件，增量构建 | 自动移除引用 |

### GLOB 结果验证

```bash
# 对比 GLOB 结果与 git 跟踪文件
cmake -S . -B build -DCMAKE_VERBOSE_MAKEFILE=ON
# 检查 build.compile_commands.json 中包含的源文件
grep -o 'src/[^"]*\.cpp' build/compile_commands.json | sort -u > globbed.txt
git ls-files 'src/**/*.cpp' 'src/*.cpp' | sort > tracked.txt
diff globbed.txt tracked.txt  # 应无差异（除 ANTLR 生成文件）
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| GLOB 意外包含不该编译的文件 | 链接错误或重复符号 | 当前无此情况；预留 `list(FILTER ...)` 排除策略 |
| CONFIGURE_DEPENDS 增量构建开销 | 构建稍慢 | 微小开销（文件列表检查），可接受 |
| 文件未加入 git 导致 GLOB 漏文件 | 链接缺失 | CI 检查 GLOB 结果与 git 跟踪文件一致性 |
| GLOB 顺序不确定 | 链接顺序变化（通常无影响） | C++ 链接顺序对 .so 无影响（符号在运行时解析） |
| CMake 版本兼容性 | CONFIGURE_DEPENDS 需 CMake 3.12+ | 项目要求 CMake 3.15+，满足条件 |

## Open Questions

1. **是否在 CI 中添加 GLOB 一致性检查？**
   - 推荐：YES（可选，通过脚本对比 GLOB 结果与 git 跟踪文件）
   - 决定：列为 SHOULD 任务，不阻塞核心验收

2. **是否同时迁移 src/ptx_ir/CMakeLists.txt 中的列表？**
   - 推荐：NO（1-2 个文件，GLOB 无收益）
   - 决定：明确划入 Out Scope

## 关联文档

- `improvements/cmake-use-glob-for-sources.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-9`：原债务条目
- `src/CMakeLists.txt`：当前手动源文件列表（77 个 .cpp）
- [CMake file(GLOB) 文档](https://cmake.org/cmake/help/latest/command/file.html#glob)
