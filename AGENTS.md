# PTX-EMU Agent Instructions

> **PTX 模拟器**: C++20/CUDA PTX 模拟器，ANTLR4 解析 PTX，fake libcudart.so 拦截 CUDA runtime
> **C++ 标准**: C++20（含 CUDA 代码）
> **ANTLR 版本**: 4.11.1（antlr-4.11.1-complete.jar）
> **Generated**: 2026-05-05 | **Commit**: baa8c4e | **Branch**: main

---

## 🛑 PTX 语法修改流程（最高优先级）

**当检测到 ANTLR 解析错误时，必须先运行测试分类，再行动。**

### 错误分类（先运行测试！）

```bash
# 1. 运行测试获取错误
cd build && ctest -R <test_name> -V 2>&1 | tail -50

# 2. 分类错误
echo <输出> | grep -E "missing|mismatched|no viable|extraneous|ANTLR"
# → 有输出 = ANTLR 解析错误 → 走语法修复流程
# → 无输出 = 运行时错误   → 走 ptx-debug 流程
```

### 语法修复检查清单

```
□ 1. 加载技能：.opencode/skills/ptx-grammar-modification/SKILL.md
□ 2. 阅读 docs/ptx/ 对应章节
□ 3. 运行基线：./tests/ptx/test_all_ptx.sh（不是 ctest！）
□ 4. cuobjdump -xptx 提取真实 PTX → 复制到 tests/ptx/
□ 5. 修改 .g4 → cmake --build build --target GenerateParser
□ 6. ./tests/ptx/test_all_ptx.sh 全部通过才能交付
```

### ❌ 禁止

- 用 `ctest` 代替 `./tests/ptx/test_all_ptx.sh`
- 未读 docs/ptx/ 就改语法
- 未加测试用例就修语法
- 测试未全通过就标完成
- 手动编辑 `build/antlr4_generated_src/` 中的生成文件

---

## 🎯 技能触发

### 全局技能 (~/.config/opencode/skills/)

| 触发关键词 | 技能 |
|-----------|------|
| `segfault`, `SIGSEGV`, `core dumped`, `死锁`, `内存泄漏` | cpp-debug |
| `CMakeLists.txt`, `link error`, `undefined reference` | cmake-manage |
| `模块依赖`, `架构图` | cpp-architecture |
| `现代 C++`, `智能指针` | cpp-modernize |
| CUDA/kernel/nsys/ncu/cuda-gdb | cuda-ptx |

### 项目技能 (.opencode/skills/)

| 触发场景 | 技能文件 |
|---------|---------|
| ANTLR 解析错误 | `.opencode/skills/ptx-grammar-modification/SKILL.md` |
| 运行时错误/SegFault | `.opencode/skills/ptx-debug/SKILL.md` |
| 屏障机制问题 | `.opencode/skills/ptx-barrier-mechanism/SKILL.md` |
| 指令执行/PC 问题 | `.opencode/skills/ptx-instruction-pipeline/SKILL.md` |
| PTX 加载慢/序列化 | `.opencode/skills/ptxir-serialization/SKILL.md` |
| 咨询 Oracle | `.opencode/skills/oracle-prompting/SKILL.md` |
| 测试回归定位 | `.opencode/skills/regression-bisect/SKILL.md` |
| 状态值异常审计 | `.opencode/skills/state-modification-audit/SKILL.md` |
| 生成 PTX 测试 | `.opencode/skills/three-mode-testing/SKILL.md` |

### 技能调用关系

```
ptx-debug (入口)
  ├─ regression-bisect (测试回归 → 找 root cause)
  │   ├─ state-modification-audit (值被覆盖 → 交叉引用)
  │   └─ oracle-prompting (咨询 Oracle → 防幻觉)
  ├─ ptx-instruction-pipeline (PC/ExecPipe 问题)
  │   └─ ptx-barrier-mechanism (屏障问题)
  ├─ ptx-grammar-modification (ANTLR 解析错误)
  └─ cpp-debug (C++ 崩溃/内存)
```

### ⚠️ 测试失败决策树

```
用户："修复 test_X"
    ↓
先运行测试，看错误输出
    ↓
ANTLR 解析错误？──→ ptx-grammar-modification
SegFault/崩溃？   ──→ ptx-debug + cpp-debug
逻辑错误/断言？   ──→ ptx-debug
编译/链接错误？   ──→ cmake-manage + cpp-debug
```

---

## 构建

```bash
# 1. 设置环境（必须！设置 CUDA_PATH、CLASSPATH、LD_LIBRARY_PATH）
. env.sh

# 2. 配置 + 构建（Release）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# 快捷方式
./build.sh          # Debug 模式，自动 . env.sh

# 构建特定目标
cmake --build build --target cudart     # fake libcudart.so
cmake --build build --target ptxsim     # PTX 模拟引擎
cmake --build build --target ptx_parser # PTX 解析器
cmake --build build --target ptxir      # PTXIR 序列化库（ptxir_writer + ptxir_reader）

# 重生成 ANTLR 解析器（改 .g4 后）
cmake --build build --target GenerateParser
```

**依赖**: CMake ≥ 3.15, CUDA Toolkit, GCC, Java, ccache（可选自动启用）

**输出目录**: `build/bin/` (可执行文件), `build/lib/` (共享库)
`libcudart.so` 自动软链到项目根目录 `lib/`

---

## 测试

```bash
# 全部测试
cd build && ctest

# 按标签分组
ctest -L mini      # Mini 基础测试
ctest -L ptx       # PTX 指令测试

# 单个测试（verbose）
ctest -R test_memory_manager -V

# 运行单个 benchmark（项目根目录）
make -C build RAY

# PTX 语法全量测试（非 ctest！）
./tests/ptx/test_all_ptx.sh
```

**测试框架**: Catch2（`tests/catch_amalgamated.hpp`）
**CUDA 测试编译**: `-keep` 保留中间 PTX，`sm_100` 虚拟架构

---

## TDD 开发流程

**强制执行**: 所有功能实现和缺陷修复必须遵循 TDD 三阶段流程。

### 三阶段流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. 测试先行  │ → │  2. 实现代码  │ → │  3. 验证     │
│  Write Test │    │  Implement  │    │  Sanity     │
└─────────────┘    └─────────────┘    └─────────────┘
```

| 阶段 | 操作 | 验证方式 |
|------|------|---------|
| **1. 测试先行** | 编写失败的测试用例 | `./scripts/sanity.sh --quick` 确认失败 |
| **2. 实现代码** | 编写通过测试的实现 | `./scripts/sanity.sh --quick` 确认通过 |
| **3. 验证** | 运行完整 sanity 检查 | `./scripts/sanity.sh` 无回归 |

### Sanity 脚本用法

```bash
# 完整 sanity 检查（推荐开发时使用）
./scripts/sanity.sh

# 快速检查（仅关键 bug 修复）
./scripts/sanity.sh --quick

# 仅 PTX 语法测试
./scripts/sanity.sh --ptx

# 详细输出
./scripts/sanity.sh --verbose

# 运行特定标签
cd build && ctest -L "exec_mask|simt_entry" -V
```

### 测试标签速查

| 标签 | 覆盖范围 | 实际测试示例 |
|------|---------|-------------|
| `exec_mask` | BUG-001 exec_mask 恢复 | `test_exec_mask` |
| `simt_entry` | BUG-002 SIMT stack exit 处理 | `test_simt_stack_entry` |
| `active_mask` | ISSUE-004 active_mask 一致性 | `test_active_mask_consistency` |
| `barrier` | 屏障同步、reconvergence | `test_warp_barrier_extended`, `test_barrier_reconvergence` |
| `ptx;integer/float/bitwise/cvt/ld_st` | PTX 指令 | `test_ptx_integer`, `test_ptx_float`, `test_ptx_ld_st` |
| `memory` | 内存分配、边界检查 | `test_memory_manager`, `test_memory_bounds` |
| `integration` | 端到端集成 | `test_simt_integration`, `test_barrier_simt_integration` |

### ❌ 禁止

- 未写测试就实现功能
- 测试未失败就实现（Red 阶段必须有失败）
- 提交前不运行 `./scripts/sanity.sh`
- 用 `ctest` 代替 `./tests/ptx/test_all_ptx.sh` 做 PTX 语法测试

---

## 代码风格

- **格式化**: clang-format（`.clang-format`, BasedOnStyle=LLVM, IndentWidth=4, ColumnLimit=80）
- **命名**: 文件 snake_case | 函数 camelCase | 类 PascalCase | 变量 camelCase
- **PTX 指令**: 全小写（`mov`, `add`, `ld`, `st`）
- **头文件**: `#ifndef`/`#define`/`#endif` 守卫
- **提交前**: `clang-format -i <file>`

---

## 架构速览

### 执行层次

```
GPUContext (全局内存, SM 列表)
  └── SMContext (资源, warp 调度器, 屏障)
        └── CTAContext (warps, shared/local 内存)
              └── WarpContext (32 线程, 活跃掩码, 分歧)
                    └── ThreadContext (寄存器, 条件码, PC)
```

### 执行流

1. `__cudaRegisterFatBinary` → `cuobjdump` 提取 PTX → ANTLR4 解析 → 填充 PtxContext
2. `cudaLaunchKernel` → PtxInterpreter → 构建符号表 → 提交 KernelLaunchRequest
3. GPUContext → SMContext → CTAContext → WarpContext 分发
4. ThreadContext::execute_thread_instruction() → InstructionFactory 分发

### 核心目录

| 目录 | 内容 |
|------|------|
| `src/ptx_ir/` | IR 类型、操作数/语句上下文 |
| `src/ptxir/` | PTXIR 序列化库（ptxir_writer + ptxir_reader） |
| `src/ptx_parser/` | PTXVisitor, CFGBuilder |
| `src/ptxsim/core/` | GPU/SM/CTA/Warp/Thread 上下文 |
| `src/ptxsim/instructions/` | PTX 指令实现 |
| `src/ptxsim/memory/` | 内存抽象 (SimpleMemory, SharedMemoryManager) |
| `src/ptxsim/register/` | 寄存器抽象 |
| `src/cudart/` | fake libcudart.so (CUDA runtime 替代) |
| `src/grammar/` | ptxLexer.g4, ptxParser.g4 |
| `configs/` | GPU 架构 JSON + 调试 INI |
| `include/` | 公共头文件 |

---

## 添加 PTX 指令

1. 更新 `include/ptx_ir/ptx_op.def`（X-Macro 模式）
2. 在 `src/ptxsim/instructions/` 实现 handler
3. 如需，更新 `src/grammar/ptxParser.g4`
4. `cmake --build build --target GenerateParser`
5. 添加测试

### X-Macro 模式

```cpp
#define X(name, ...) process_##name(__VA_ARGS__);
#include "ptx_op.def"
#undef X
```

---

## 调试与日志

```bash
# 使用调试脚本选择配置
./scripts/debug-run.sh {配置名} ./build/bin/程序

# 可用配置：debug, verbose, memory, instruction, perf
```

配置文件：`configs/config.ini` 或 `configs/debug_config.ini`
组件: emu, exec, mem, reg, thread, func
级别: trace, debug, info, warning, error, fatal

详见 `docs/debugging_guide.md`

---

## 已知限制

| 类别 | 状态 |
|------|------|
| WMMA/Tensor Core (wmma, mma) | 解析但空实现 (stub) |
| Atomic 操作 | 无真正原子性 (stub) |
| Hopper (sm_90+) | cluster 抽象未实现 |
| Event/Stream API | fake 返回（不同步） |
| 函数调用 | 未完全实现 |
| Multi-PTX cubins | 仅提取第一个 PTX (ptx_parser.cpp:59 FIXME) |
| `assert(false)` | 多处 → 遇未处理代码路径会崩溃 |

### 安全假设

- 基本算术/逻辑指令 ✓
- 内存操作 (ld/st): global/shared/local ✓
- 控制流 (bra, ret) ✓
- Ampere (sm_80) 及更早架构 ✓

---

## 重要文件

| 文件 | 说明 |
|------|------|
| `include/ptx_ir/ptx_op.def` | PTX 指令定义 (X-Macro) |
| `src/grammar/ptxLexer.g4` / `ptxParser.g4` | ANTLR 语法 |
| `src/cudart/cudart_sim.cpp` | CUDA runtime 入口 |
| `src/ptxsim/InstructionHandlers.cpp` | 指令实现 |
| `configs/ampere_a100.json` | 默认 GPU 架构配置 |
| `include/ptxir/ptxir_serialization.h` | PTXIR 序列化 API |
| `tests/ptx/test_all_ptx.sh` | PTX 语法全量测试 |

## 参考文档

- `docs/gpgpu_arch.md` - GPU 执行架构
- `docs/ptx-emu_arch.md` - 系统架构
- `docs/debugging_guide.md` - 调试与日志
- `docs/sm90_100.md` - Hopper/Blackwell 架构
- `.opencode/skills/README.md` - 技能索引与调用关系
