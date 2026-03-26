# PTX-EMU Agent Instructions

> **开发流程**: 本项目遵循 [文档驱动开发流程](~/.config/opencode/docs/dev-process/README.md)  
> **快速参考**: [quick-reference.md](~/.config/opencode/docs/dev-process/quick-reference.md)  
> **核心规则**: 
> - 🚫 NO IMPLEMENTATION WITHOUT APPROVED DESIGN
> - 🚫 NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST

---

## 🎯 项目技能自动触发

### 全局技能（~/.config/opencode/skills/）

| 技能 | 触发词 | 适用场景 |
|------|--------|---------|
| **git-commit-review** | `/commit-review` | 审查最近 N 个 commit 的代码变更 |

### 项目技能位置

**项目技能目录**: `docs/skills/`

| 技能 | 文件 | 触发方式 |
|------|------|---------|
| **ptx-grammar-modification** | `docs/skills/ptx-grammar-modification.md` | 自动触发（见下方 PTX 语法修改流程） |
| **ptx-debug** | `docs/skills/ptx-debug/SKILL.md` | 关键词触发或手动加载 |

### ptx-debug 技能触发关键词

当用户提到以下关键词时，**自动加载 ptx-debug 技能**：

**问题类型触发**:
- "测试失败", "ctest 不通过", "单元测试失败"
- "程序崩溃", "segfault", "SIGSEGV", "core dumped"
- "内存错误", "非法访问", "越界", "内存泄漏"
- "指令错误", "执行结果不对", "结果不对"
- "性能慢", "性能优化", "benchmark"
- "调试这个", "分析一下问题", "查看日志"

**场景触发**:
- 运行测试后失败
- 查看日志文件
- 使用调试配置
- 分析内存访问
- 跟踪指令执行

**触发示例**:
```
用户："test_memory 测试失败了"
→ 自动加载 ptx-debug 技能
→ 选择 debug_config.ini
→ 运行测试收集日志

用户："程序崩溃了，帮我分析"
→ 自动加载 ptx-debug 技能
→ 选择 verbose_trace.ini
→ 收集崩溃前日志

用户："这个内存访问有问题"
→ 自动加载 ptx-debug 技能
→ 选择 memory_debug.ini
→ 跟踪内存操作
```

### 手动加载技能

**明确要求加载技能**:
```
"请加载 ptx-debug 技能分析这个问题"
"使用 ptx-debug 技能调试这个测试"
```

**使用 skill 工具**:
```bash
# 在对话中要求
skill name="ptx-debug"
```

---

## 🔍 LSP 自动触发规则 (C++/CUDA 分析)

**当分析 C++/CUDA 文件时 (.cpp, .h, .cu, .cuh)**:

### 自动触发场景

| 场景 | 自动调用工具 | 说明 |
|------|------------|------|
| 读取文件后 | `lsp_diagnostics` | 检查编译错误/警告 |
| 询问"有哪些函数/类" | `lsp_symbols` | 获取符号列表 |
| 询问"X 在哪里定义" | `lsp_goto_definition` | 跳转到定义 |
| 询问"X 在哪里使用" | `lsp_find_references` | 查找所有引用 |
| 重命名变量/函数 | `lsp_prepare_rename` → `lsp_rename` | 安全重命名 |

### 工作流程示例

```
1. 用户："看一下 src/cudart/cudart_sim.cpp"
   → read(filePath="...")
   → 自动触发 → lsp_diagnostics(filePath="...")

2. 用户："这个文件有哪些函数？"
   → 自动触发 → lsp_symbols(filePath="...", scope="document")

3. 用户："GPUContext 在哪里定义？"
   → 自动触发 → lsp_goto_definition(filePath="...", line=N, character=N)
```

### 降级策略

如果 LSP 不可用 (超时/未响应):
1. 使用 `grep` 文本搜索
2. 使用 `ast_grep` AST 搜索
3. 使用 `glob` 文件匹配

### 预热提示

首次使用或 clangd 重启后，LSP 可能需要 30-60 秒预热。
如遇超时，等待后重试或使用降级策略。

---

## 🛑 PTX 语法修改流程（最高优先级）

> **⚠️ 重要**: 本流程**优先于所有其他操作**。违反此流程 = 违反核心规则 🛫

### 触发条件（满足任一即🛫停止）

**当任务涉及 PTX 语法解析问题时，必须首先执行以下步骤**：

| 触发场景 | 关键词/错误 | 立即行动 |
|---------|------------|---------|
| **用户请求修复解析错误** | "PTX 解析错误", "语法错误", "ANTLR 错误" | 🛫 → 加载技能 → 运行测试 |
| **ANTLR 解析错误** | `no viable alternative at input` | 🛫 → 加载技能 → 运行测试 |
| **意外 Token** | `mismatched input 'X' expecting Y` | 🛫 → 加载技能 → 运行测试 |
| **修改语法文件** | 改动 `src/grammar/*.g4` | 🛫 → 加载技能 → 运行测试 |
| **解析阶段崩溃** | `Segmentation fault` 在 parser 阶段 | 🛫 → 加载技能 → 运行测试 |

### 📋 强制检查清单（执行前必读）

**在写任何代码之前，必须按顺序完成以下步骤**：

```
□ 步骤 1: 加载技能
   → 加载项目技能：ptx-grammar-modification
   → 位置：docs/skills/ptx-grammar-modification.md

□ 步骤 2: 阅读文档
   → 阅读 docs/ptx/ 对应章节（了解 PTX 语法规范）
    
□ 步骤 3: 运行基线测试
   → 执行 ./tests/ptx/test_all_ptx.sh（不是 ctest！）
   → 记录当前失败状态

□ 步骤 4: 准备测试用例
   → 如果有真实 binary，使用 cuobjdump -xptx 提取 PTX
   → 复制到 tests/ptx/ 目录

□ 步骤 5: 修复语法
   → 修改 .g4 文件
   → cmake --build build --target GenerateParser

□ 步骤 6: 验证通过
   → ./tests/ptx/test_all_ptx.sh 必须全部通过
   → 否则回到步骤 3
```

### ❌ 禁止行为

- 🛫 **禁止**使用 `ctest` 代替 `./tests/ptx/test_all_ptx.sh`
- 🛫 **禁止**在未阅读 docs/ptx/ 文档前修改语法
- 🛫 **禁止**在未添加测试用例前修复语法
- 🛫 **禁止**在测试未全部通过前标记任务完成

### ✅ 正确示例

```
用户："列出所有因为 PTX 解析错误的单元测试，并一一修复"

正确流程：
1. 🛫 识别为 PTX 语法问题
2. 加载技能：ptx-grammar-modification（docs/skills/）
3. 运行 ./tests/ptx/test_all_ptx.sh 确定失败用例
4. 对每个失败用例：分析错误 → 添加测试 → 修复语法 → 验证
5. 确保所有测试通过后才交付
```

**完整流程文档**: [docs/skills/ptx-grammar-modification.md](docs/skills/ptx-grammar-modification.md)
**项目技能总览**: [docs/skills/README.md](docs/skills/README.md)

---

## Build Commands

```bash
# Setup environment (required before building)
. env.sh

# Configure and build (Release)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Or use the build script (does env setup automatically)
./build.sh

# Build specific target
cmake --build build --target cudart
cmake --build build --target ptxsim

# Debug build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Test Commands

```bash
# Run all tests (from build directory)
cd build && ctest

# Run specific test by name
ctest -R test_memory_manager

# Run tests with labels
ctest -L mini      # Mini tests
ctest -L ptx       # PTX instruction tests

# Run with verbose output
ctest -V

# Run single benchmark test (from project root)
make -C build dummy
make -C build RAY
```

## Lint/Format

```bash
# Format code with clang-format
find src include -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

## Code Style Guidelines

### Formatting
- **Style**: LLVM-based (see `.clang-format`)
- **Indent**: 4 spaces, no tabs
- **Line limit**: 80 columns
- **Braces**: Attach style (no newline before braces)
- **Short functions**: Allow on single line
- Use `clang-format` to auto-format before committing

### Naming Conventions
- **Files**: snake_case (e.g., `ptx_parser.cpp`, `instruction_handlers.h`)
- **Functions**: camelCase for most; snake_case for PTX-specific handlers
- **Classes/Structs**: PascalCase (e.g., `GPUContext`, `ThreadContext`)
- **Variables**: camelCase (e.g., `gridDim`, `threadIdx`)
- **Constants/Enums**: UPPER_SNAKE_CASE or enum class with PascalCase
- **Member variables**: Same as variables (no special prefix)
- **PTX instructions**: lowercase (e.g., `mov`, `add`, `ld`, `st`)

### Types & Includes
- **Standard**: C++20 (CUDA code uses C++17)
- **Headers**: Use `#ifndef`/`#define`/`#endif` guards
- **Include order**:
  1. Generated ANTLR headers (if needed)
  2. Project headers (e.g., `"ptxsim/..."`)
  3. Standard library (e.g., `<vector>`, `<string>`)
- Use forward declarations when possible to reduce includes

### Error Handling
- Use assertions (`assert()`) for internal invariants
- Return error codes for recoverable errors
- Use logging macros: `PTX_ERROR()`, `PTX_WARN()`, `PTX_INFO()`
- Fatal errors: print message and exit or throw

## Project Structure

- **src/ptx_ir/**: IR types and semantic context
- **src/ptx_parser/**: ANTLR-based PTX parser (PtxListener)
- **src/ptxsim/**: Execution engine (GPU/SM/CTA/Warp/Thread context)
- **src/ptxsim/instructions/**: PTX instruction implementations
- **src/ptxsim/core/**: Core execution logic
- **src/ptxsim/memory/**: Memory abstractions
- **src/ptxsim/register/**: Register abstractions
- **src/cudart/**: CUDA runtime API replacement (fake libcudart.so)
- **src/grammar/**: ANTLR4 grammar files (ptxLexer.g4, ptxParser.g4)
- **include/**: Public headers
- **tests/**: Catch2 + CUDA PTX tests (forces PTX compilation mode)
- **bench/**: Benchmark programs
- **configs/**: GPU architecture JSON configs and debug INI files

## Key Conventions

### Adding PTX Instructions
1. Update `include/ptx_ir/ptx_op.def` (X-Macro pattern)
2. Implement handler in `src/ptxsim/instructions/`
3. Update grammar in `src/grammar/ptxParser.g4` if needed
4. Regenerate parser: `cmake --build build --target GenerateParser`

### X-Macro Pattern
```cpp
#define X(name, ...) process_##name(__VA_ARGS__);
#include "ptx_op.def"
#undef X
```

### Testing
- Tests use Catch2 framework
- Run specific test: `ctest -R test_name -V`
- Labels: `ctest -L mini` (mini tests), `ctest -L ptx` (PTX instruction tests)
- CUDA files use `.cu` extension, link against fake `libcudart.so`

### Adding CUDA API
- Add implementation in `src/cudart/` directory
- Ensure function signature matches CUDA runtime API
- Rebuild `cudart` target

## Architecture Overview

- **PTX Simulator**: C++/CUDA emulator in `src/` (ptx_ir/, ptxsim/, cudart/)
- **Parser**: ANTLR4-based (v4.13.1), grammar in `src/grammar/`
- **Execution hierarchy**: GPUContext → SMContext → CTAContext → WarpContext → ThreadContext
- **ANTLR runtime**: antlr4/antlr4-cpp-runtime-4.13.1-source

## Debugging & Logging

- Configured via INI files: `configs/config.ini` or `configs/debug_config.ini`
- Component logs: `emu`, `exec`, `mem`, `reg`, `thread`, `func`
- Log levels: trace, debug, info, warning, error, fatal
- See `docs/debugging_guide.md` for detailed setup

## Common Workflows

```bash
# Full rebuild after major changes
. env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build

# Run single test with verbose output
cd build && ctest -R test_name -V

# Rebuild specific target (faster iteration)
cmake --build build --target ptxsim

# Regenerate ANTLR parser (after grammar changes)
cmake --build build --target GenerateParser

# Run benchmark
make -C build RAY

# Run PTX syntax tests (from project root)
./tests/ptx/test_all_ptx.sh
```

## PTX Grammar Modification

**For detailed workflow**: See [docs/skills/ptx-grammar-modification.md](docs/skills/ptx-grammar-modification.md)

**Quick reference**:
1. **Before changes**: Read corresponding chapter in `docs/ptx/`
2. **Verify**: Run `./tests/ptx/test_all_ptx.sh`
3. **Debug** (if tests fail): Extract PTX with `cuobjdump`, add test case, fix parser

**Core principle**: Test-driven development. Read docs first, then fix.

## Important Files

| File | Description |
|------|-------------|
| `include/ptx_ir/ptx_op.def` | Instruction definitions (X-Macro) |
| `src/grammar/ptxLexer.g4` / `ptxParser.g4` | ANTLR grammar |
| `src/cudart/cudart_sim.cpp` | Main CUDA runtime entry point |
| `src/ptxsim/instruction_handlers.h` | Instruction handler declarations |
| `configs/*.json` | GPU architecture configs (ampere_a100.json, hopper_h100.json) |

## Reference Documentation

- `docs/gpgpu_arch.md` - GPU architecture details
- `docs/debugging_guide.md` - Debugging and logging setup
- `docs/arch.md` - System architecture
- `docs/sm90_100.md` - Hopper/Blackwell GPU specifics
- `docs/skills/ptx-grammar-modification.md` - PTX grammar modification skill (TDD workflow)



## ANTI-PATTERNS (THIS PROJECT)

### Critical Limitations (Silent Failures)
- **WMMA/Tensor Core**: Instructions are parsed but implementations are empty stubs — silently do nothing
- **Atomic operations**: No actual atomicity guarantees — stubs return immediately
- **Function calls in PTX**: Call logic not fully implemented
- **Multi-PTX cubins**: Only first PTX extracted (FIXME in ptx_parser.cpp:59)

### Architecture Constraints
- **Hopper (sm_90+) NOT supported**: Thread block cluster abstraction missing
- **Tensor Core (wmma, mma) NOT implemented**: Stubs only
- **Event/Stream APIs**: Fake implementations that log but don't synchronize

### Development Gotchas
- `assert(false)` in multiple places — crashes on unhandled code paths
- TODO/FIXME comments indicate incomplete implementations
- PTX opcode parsing in `ptx_visitor.cpp` has many unimplemented paths

### Safe Assumptions
- Basic PTX arithmetic/logic instructions work
- Memory operations (ld/st) work for global/shared/local
- Control flow (bra, ret) works
- Ampere (sm_80) and earlier architectures supported