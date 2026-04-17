# PTX-EMU Agent Instructions

> **PTX 模拟器**: C++20/CUDA PTX 模拟器，ANTLR4 解析 PTX，fake libcudart.so 拦截 CUDA runtime
> **C++ 标准**: C++20（含 CUDA 代码）
> **ANTLR 版本**: 4.11.1（antlr-4.11.1-complete.jar）

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
□ 1. 加载技能：docs/skills/ptx-grammar-modification.md
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

### 项目技能 (docs/skills/)

| 触发 | 文件 |
|------|------|
| ANTLR 解析错误 | `ptx-grammar-modification.md` |
| 运行时错误/SegFault | `ptx-debug/SKILL.md` |

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

## 代码风格

- **格式化**: clang-format（`.clang-format`, BasedOnStyle=LLVM, 4 空格缩进, 80 列）
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

```ini
# configs/config.ini 或 configs/debug_config.ini
# 组件: emu, exec, mem, reg, thread, func
# 级别: trace, debug, info, warning, error, fatal
```

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
| `tests/ptx/test_all_ptx.sh` | PTX 语法全量测试 |

## 参考文档

- `docs/gpgpu_arch.md` - GPU 执行架构
- `docs/ptx-emu_arch.md` - 系统架构
- `docs/debugging_guide.md` - 调试与日志
- `docs/sm90_100.md` - Hopper/Blackwell 架构
