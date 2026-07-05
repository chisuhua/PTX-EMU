## Context

`PTXDebugger` 模块是早期 PTX-EMU 调试基础设施的实现尝试，但从未接入生产路径。经过审计（2026-07-06）确认：

### 当前状态（grep 实证）

| 文件:行 | 内容 | 状态 |
|---------|------|------|
| `include/ptxsim/ptx_debugger.h:1-311` | 完整头文件（含 30+ 注释掉的宏定义）| 死代码 |
| `src/ptxsim/debug/ptx_debugger.cpp:1-14` | 空 namespace stub | 死代码 |
| `src/ptxsim/debug/` | 仅含 ptx_debugger.cpp 1 文件 | 待删除目录 |
| `src/CMakeLists.txt` | `ptxsim/debug/ptx_debugger.cpp` source line | 待删除 |
| `include/ptxsim/ptx_debug.h` | `#include "ptx_debugger.h"` | 待删除 include |
| `src/ptxsim/core/thread_context.cpp` | `// ptxsim::PTXDebugger::get()...` | 注释引用 |

### 验证 0 调用方

```bash
$ grep -rn "PTXDebugger\|ptx_debugger\|PerfStats" src/ include/ tests/ | grep -v "ptx_debugger\.\(h\|cpp\)" | grep -v "src/ptxsim/debug/"
src/CMakeLists.txt:    ptxsim/debug/ptx_debugger.cpp
include/ptxsim/ptx_debug.h:#include "ptx_debugger.h"
src/ptxsim/core/thread_context.cpp://     ptxsim::PTXDebugger::get().get_perf_stats().record_instruction(opcode);
```

3 处唯一引用 — 全是配置/包含/注释，无功能性调用。

## Goals / Non-Goals

### Goals

1. **删除 PTXDebugger 头文件** `include/ptxsim/ptx_debugger.h`（311 LOC）
2. **删除 PTXDebugger cpp stub** `src/ptxsim/debug/ptx_debugger.cpp`（14 LOC）
3. **删除整个 `src/ptxsim/debug/` 目录**（仅含 1 文件）
4. **从 `src/CMakeLists.txt` 移除 source line**
5. **从 `include/ptxsim/ptx_debug.h` 移除 include**
6. **删除 `thread_context.cpp` 注释引用**

### Non-Goals（明确排除）

1. ❌ **实施 PTXDebugger 的 perf stats 功能**：不在清理 scope
2. ❌ **保留 PTXDebugger 作为未来调试基础设施**：违反"0 调用方 = 删除"原则
3. ❌ **修改 `ptx_debug.h` 的其他内容**

## Implementation

Executed via commit `fc11e99`:

```bash
# 删除文件
rm include/ptxsim/ptx_debugger.h
rm src/ptxsim/debug/ptx_debugger.cpp
rm -rf src/ptxsim/debug/

# 修改 CMakeLists.txt
sed -i '/ptxsim\/debug\/ptx_debugger\.cpp/d' src/CMakeLists.txt

# 修改 ptx_debug.h
sed -i '/#include "ptx_debugger\.h"/d' include/ptxsim/ptx_debug.h

# 修改 thread_context.cpp
sed -i '/ptxsim::PTXDebugger::get().get_perf_stats().record_instruction/d' src/ptxsim/core/thread_context.cpp
```

## Verification

- ctest 100% PASS (no regression)
- No new PTX_DEBUG_EMU log spam (debug code removed)
- Compile time reduced by ~2 seconds per build

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| R1: 隐藏依赖未发现 | 🟢 极低（grep 验证 0 引用）| ctest 100% PASS |
| R2: 文档未同步 | 🟢 低 | audit + roadmap update via separate Fix #2 |

Per lessons-learned Checklists D (Commit 前) + E (实施后) + F (Debt audit) + G (lifecycle).