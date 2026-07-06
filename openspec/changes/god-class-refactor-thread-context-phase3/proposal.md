## Why

Phase 1+2 of `god-class-refactor-thread-context` (archived at `archive/2026-07-06-god-class-refactor-thread-context/`) reduced `ThreadContext` from 884 → 471 lines by extracting `SimtPcManager` (PC + state) and `RegisterAccessLayer` (register lookup). Phase 3.1 (MemoryAccessor) and Phase 3.2 (InstructionPipeline) were cancelled during implementation due to deep coupling with `ThreadContext`'s public state.

This change picks up the remaining ~280 lines of `ThreadContext` that still mix memory address resolution, control flow orchestration, and operand collection into one class. Two technical blockers from the previous phase must be resolved first:

1. `shared_mem_space` and `local_mem_space` are public data members on `ThreadContext`, settable from external code (CTAContext, etc.). MemoryAccessor copies diverged from this state in the previous attempt, causing 9 test regressions. **Required precondition**: convert these to non-public members or `std::shared_ptr` references that MemoryAccessor owns.
2. `handler->ExecPipe(this, statement)` requires `this` to be `ThreadContext*`. Extracting `_execute_once()` into `InstructionPipeline` would break all 40+ handler signatures. **Required precondition**: change handler signature to accept both `ThreadContext*` and `InstructionPipeline*` (or a new context interface), starting with a single handler as prototype.

## What Changes

- **Phase 3.1 (~3h)**: Extract `MemoryAccessor` after converting `shared_mem_space`/`local_mem_space` to private members with setter methods
  - Migrate `get_memory_addr()`, `mov_data()`, `mov()`, `initialize_shared_memory()`, `set_local_memory_space()`, related PODs
  - Members: `shared_mem_space`, `local_mem_space`, `name2Sym`, `name2Share`, `cta_context_`, `SHMEMADDR`
- **Phase 3.2 (~4h)**: Refactor handler signature, then extract `InstructionPipeline`
  - First sub-step: change base handler `ExecPipe(ThreadContext*, ...)` to `ExecPipe(ThreadContext*, InstructionPipeline*, ...)` — 40+ file changes but mechanical
  - Migrate `_execute_once()`, `execute_thread_instruction()`, `collect_operands()`, `commit_operand()`, `init()`, `reset()`, `clear_temporaries()`, `dump_state()`, `prepare_breakpoint_context()`, `trace_status()`, `print_instruction_status()`
  - Members: `operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`, `dst_operand_reg_name_`, `call_stack`
- **Phase 3.3 (~1h)**: Delete legacy PODs, add ADR-0017
  - Delete `exec_state_`, `reg_pred_`, `memory_`, `program_ref_` PODs from `ThreadContext`
  - Add `docs/adr/0017-pc-management-extraction.md` recording 3-Phase decision history
- Each Phase independent commit + independent revert
- **Final target**: `thread_context.cpp` ≤ 250 lines (pure delegation hub)

## Capabilities

### New Capabilities
- `memory-access-extraction`: Phase 3.1 `MemoryAccessor` class — encapsulates per-thread memory address resolution and data movement, owns shared/local memory spaces and symbol table references
- `control-flow-extraction`: Phase 3.2 `InstructionPipeline` class — encapsulates per-instruction execution orchestration and operand collection, owns operand buffers (`operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`)
- `handler-signature-refactor`: Phase 3.2 precondition — base handler `ExecPipe` accepts `InstructionPipeline*` alongside `ThreadContext*` for forwarding

### Modified Capabilities
<!-- No existing spec-level requirements are changing; this is a pure refactor. -->

## Impact

| 受影响组件 | 影响类型 |
|-----------|---------|
| `src/ptxsim/core/thread_context.cpp` (471 → ~200 行) | 加 ~50 行内联委托，删 200+ 行实现 |
| `include/ptxsim/thread_context.h` (~312 → ~250 行) | 删 public `shared_mem_space`/`local_mem_space`，加 setter |
| `src/ptxsim/instructions/*.cpp` (40+ 文件) | Phase 3.2 handler 签名变更（机械改动） |
| `include/ptxsim/instruction_handlers.h` | 基类 `ExecPipe` 签名变更 |
| 新增 `include/ptxsim/memory_accessor.h` + `.cpp` | Phase 3.1 提取 (~200 行) |
| 新增 `include/ptxsim/instruction_pipeline.h` + `.cpp` | Phase 3.2 提取 (~300 行) |
| 新增 `docs/adr/0017-pc-management-extraction.md` | 决策记录 |
| 测试 | 已有 174 个测试覆盖所有路径，预期无需新测试 |

## Prerequisites from Previous Phase

Before implementing Phase 3.1:
- [ ] `shared_mem_space` and `local_mem_space` must be made private on `ThreadContext`, with public setter methods that also update `MemoryAccessor`
- [ ] `name2Sym`, `name2Share`, `cta_context_` similar treatment (may be deferred if getter-only access is sufficient)

Before implementing Phase 3.2:
- [ ] Handler base class `ExecPipe` signature updated to accept `(ThreadContext*, InstructionPipeline*, ...)`
- [ ] All 40+ handler implementations updated (mechanical change: pass through pipeline)
- [ ] At least one handler tested in isolation to confirm signature works

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- [ ] 列出 Phase 3.1 迁移的 set_* / commit_* 调用 (set_shared_memory_space, etc.)
- [ ] 行级 diff 计划：`get_memory_addr()` + `mov_data()` + `mov()` + `initialize_shared_memory()`
- [ ] 行级 diff 计划：`_execute_once()` + `execute_thread_instruction()` + `collect_operands()`

### 多 Phase 推进
- [ ] Phase 拆分方案 + 独立 commit 粒度已说明
- [ ] 基线 worktree 命令已记录 (`git worktree add .worktrees/baseline-pre-c3-phase1 HEAD~1`)
- [ ] 失败处理策略（revert 该 Phase 单独 commit）

### Prerequisite 隔离
- [ ] Phase 3.1 prerequisites（前置条件：public → private 转换）单独 commit
- [ ] Phase 3.2 prerequisites（handler 签名变更）单独 commit
- [ ] 确认 prerequisite commit 各自通过所有测试

### 文档同步
- [ ] `src/ptxsim/core/AGENTS.md`: 新增 `MemoryAccessor` + `InstructionPipeline` 条目
- [ ] ADR-0017 追加：Phase 3 历程 + 取消教训
- [ ] `tasks.md` Phase 状态变更已说明

## References

- Archived predecessor: `openspec/changes/archive/2026-07-06-god-class-refactor-thread-context/`
- Predecessor commits: `9ce8e93`, `3082e80`, `bc1a0aa`, `3fa5e5f`, `a2d7ab0`
- Predecessor learnings: 5 Metis MUST-RESOLVE + 3 Oracle gaps all applied in Phase 1+2
- Debt audit: `docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 (P1)
- Roadmap: `docs/roadmap/post-phase3-debt-roadmap.md` §1.2 C-1
- Lessons-learned integration: §1 (cross-module state translation), §3 (multi-Phase commit), §7 (pre-impl Metis review)

