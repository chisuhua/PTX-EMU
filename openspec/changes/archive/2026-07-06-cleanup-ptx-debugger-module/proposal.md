## Why

`PTXDebugger` 整个模块（`include/ptxsim/ptx_debugger.h` 311 LOC + `src/ptxsim/debug/ptx_debugger.cpp` 14 LOC 空 stub）经过审计已确认 **0 生产调用方**：

- `include/ptxsim/ptx_debugger.h` 头文件中绝大部分代码（30+ 宏定义）已注释掉
- `src/ptxsim/debug/ptx_debugger.cpp` 仅含 namespace 声明 + 注释
- `src/ptxsim/core/thread_context.cpp:154` 唯一引用也是注释掉的 `// ptxsim::PTXDebugger::get().get_perf_stats().record_instruction(opcode);`
- `include/ptxsim/ptx_debug.h:8` 间接 include 这个死模块

清理此模块消除 ~325 LOC 死代码 + 减少编译时间 + 避免后续维护者误以为这是活跃子系统。

## What Changes

- **删除 `include/ptxsim/ptx_debugger.h`** （311 LOC）
- **删除 `src/ptxsim/debug/ptx_debugger.cpp`** （14 LOC）
- **删除 `src/ptxsim/debug/` 整个目录**（仅含此 1 文件）
- **修改 `src/CMakeLists.txt`**：从 SOURCES 列表移除 `ptxsim/debug/ptx_debugger.cpp`
- **修改 `include/ptxsim/ptx_debug.h`**：移除 `#include "ptx_debugger.h"`（如 ptx_debug.h 不再需要 PTXDebugger）
- **修改 `src/ptxsim/core/thread_context.cpp`**：删除注释掉的 `// ptxsim::PTXDebugger::get()...` 行

**BREAKING**: 无 — 0 调用方意味着无外部依赖。

## Capabilities

### New Capabilities

- `ptx-debugger-cleanup`: 删除 PTXDebugger 死代码模块（~325 LOC）

### Modified Capabilities

无 — 不影响任何 spec 级行为。

## Impact

**受影响的代码/文件**：

| 文件 | 改动 | 影响 |
|------|------|------|
| `include/ptxsim/ptx_debugger.h` | 删除 | 311 LOC |
| `src/ptxsim/debug/ptx_debugger.cpp` | 删除 | 14 LOC |
| `src/ptxsim/debug/` | 删除整个目录 | （仅含 ptx_debugger.cpp） |
| `src/CMakeLists.txt` | 移除 source line | 1 行 |
| `include/ptxsim/ptx_debug.h` | 移除 include | 1 行 |
| `src/ptxsim/core/thread_context.cpp` | 删除注释 | 1 行 |

**受影响的 ADR**：
- 无直接 ADR 影响（无架构决策变更）

**测试覆盖**：
- 现有测试无回归（grep 验证 0 调用方）
- `./scripts/sanity.sh --quick` 验证编译通过 + ctest PASS

**回归风险**：
- 🟢 极低：0 调用方意味着删除无行为影响

**Lessons-learned 集成**：
- ✅ Checklist E（artifacts 必 tracked）
- ✅ Checklist F（git verify）
- ✅ Checklist H（pre-impl review）
- ✅ Checklist G（lifecycle）

## Implementation

Executed via commit `fc11e99`:
- 5 files changed, 329 deletions(-)
- delete mode 100644 include/ptxsim/ptx_debugger.h
- delete mode 100644 src/ptxsim/debug/ptx_debugger.cpp

ctest 100% PASS, zero regression.

Per lessons-learned Checklists E/F.
Refs: docs/audits/debt-audit-2026-07-02.md §2 P0-C2 (PTXDebugger).