# Tasks: add-instruction-latency-model

## 任务清单

### Task 1: 定义指令 Latency 表 ✅
**时间**: 30 分钟
**验证**: 编译通过

MUST:
- [x] 在 `include/ptx_ir/` 创建 `instruction_latency.h` 和 `instruction_latency_table.h`
- [x] 定义常见指令的 latency 值（ld.global=100, mul=4, etc.）
- [x] 查询层：getLatency(StatementType) 旁路表（避开 ptx_op.def X-Macro）

**状态**: 已完成 - 采用旁路表（`instruction_latency_table.cpp`）代替 X-Macro 扩展，
避免影响 5 处现有 X-Macro 展开点（ptx_types.h, statement_context.cpp, ptx_parser.h,
ptx_visiter.h, instruction_handlers.h, ptx_parser.cpp）。

### Task 2: 修改 ThreadContext 添加 is_blocked ✅
**时间**: 30 分钟
**验证**: 编译通过 + 测试通过

MUST:
- [x] `ThreadState::is_blocked` 已存在于 `thread_state.h:42`
- [x] `is_schedulable()` 已正确使用 `!is_blocked` 检查（含 `blocked_cycles_remaining > 0`）
- [x] barrier 释放时清除 blocked 状态
- [x] sm_context.cpp 递减 `blocked_cycles_remaining` 周期计数器

**结果**: `is_blocked` 状态已存在，无需额外实现（commit `abb45e0` 已完成）。

### Task 3: 修改 exe_once() 调度器 ✅
**时间**: 45 分钟
**验证**: 单元测试通过

MUST:
- [x] 在 `exe_once()` 中检测 blocked 状态
- [x] 实现 "选择最低 PC 的非 blocked 组" 逻辑
- [x] 如果所有组 blocked，选择 Lowest PC（被动等待）

**结果**: 调度器集成在 commit `6811c4d` 之后，`sm_context.cpp:242-268` 已实现
"选最低 PC + 单 cycle 一组"，`is_schedulable()` 已纳入 blocked 检查。

### Task 4: 实现 ld.global 长延迟处理 ✅
**时间**: 30 分钟
**验证**: 相关测试通过

NOTE:
- [x] `ld.global` 执行后标记 lane 为 blocked（`LdHandler::processOperation`）
- [x] 实现 cycle 级别的 blocked 状态递减（`sm_context.cpp:348-357`）
- [x] 测试内存加载指令的 blocking 行为（`integration_ptx_ld_st_latency_table`）

**结果**: commit `2b9d803` 实现了 `LdHandler` 中的 blocked 标记，集成到 `getLatency` 旁路表。

### Task 5: 运行完整性检查 ✅
**时间**: 15 分钟
**验证**: `./scripts/sanity.sh --quick`

NOTE:
- [x] 执行 sanity 检查（subset: PTX 测试 33/33 通过，无回归）
- [x] 任何 Pre-P0 失败是**预存在**的，与本次实现无关（详细见 docs/developer-guide/KNOWN_ISSUES.md §Pre-P0b/c）

**结果**: 4 个新 latency 测试全绿（unit + integration），PTX 语法测试 33/33 通过。
Pre-P0 baseline red 中的 e2e_shared_memory_* / cute_* 失败是**预存在**问题
（不属本次范围），文档化在 KNOWN_ISSUES.md。

---

## 归档说明

- 全部 5 个 task 实质完成
- 唯一设计变更：采用旁路表代替 ptx_op.def X-Macro 扩展（理由见 Task 1 状态）
- 后续 follow-up（不属本 change）：
  - 实现 `divergence_execution_mode` 调度策略（Interleaved / ShortestFirst）
  - 解决 `cute_*` kernel 输出全零的运行时问题
