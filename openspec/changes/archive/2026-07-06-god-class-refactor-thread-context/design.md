## Context

`ThreadContext` 当前是一个 884 行的 god class（`src/ptxsim/core/thread_context.cpp`），头文件 320 行（`include/ptxsim/thread_context.h`），包含 22 个 `#include`。该类覆盖 4 个正交的子系统：

| 子系统 | 关键方法 | 行数估 | 耦合点 |
|--------|---------|--------|--------|
| **SIMT 栈/PC 管理** | `get_pc()`, `set_pc()`, `commit_pc()`, `sync_from_warp_state()`, `sync_to_warp_state()`, `is_active()`, `is_exited()`, `is_at_barrier()`, `set_state()`, `get_state()`, `reset()` | ~150 行 | `warp_context_` (WarpState), `EXE_STATE state` |
| **寄存器访问** | `acquire_register()`, `register_bank_manager_`, `cc_reg`, `get_condition_codes()`, `set_condition_codes()`, `get_dst_operand_reg_name()` | ~60 行 | `RegisterBankManager`, `ConditionCodeRegister` |
| **内存访问** | `acquire_operand()`, `get_memory_addr()`, `initialize_shared_memory()`, `set_local_memory_space()`, `mov()`, `mov_data()` | ~300 行 | `shared_mem_space`, `name2Share`, `name2Sym`, `cta_context_`, `SHMEMADDR` |
| **控制流/执行编排** | `_execute_once()`, `execute_thread_instruction()`, `init()`, `collect_operands()`, `commit_operand()`, `dump_state()`, `prepare_breakpoint_context()`, `trace_status()`, `print_instruction_status()`, `clear_temporaries()` | ~370 行 | 所有其他子系统 |

当前已有 T2-3 A3a 的 POD 结构（`exec_state_`, `reg_pred_`, `memory_`, `program_ref_`）在类末尾声明，但未形成独立的类封装。

### 约束
- **零外部 API 变更**：`WarpContext`、`CTAContext` 及所有 PTX instruction handler 调用 `ThreadContext` 方法的方式不能改变
- **跨模块状态翻译保持**：`sync_to_warp_state(RUN)` → `is_blocked=false` 的语义必须保留（lessons-learned §1）
- **Per-thread PC 权威源不变**：`WarpState.threads[i].pc` 保持为唯一权威源（AGENTS.md DUAL STATE MECHANISM）
- **Phase 必须独立可 revert**（lessons-learned §3）

## Goals / Non-Goals

**Goals:**
- Phase 1：提取 `SimtPcManager` 类，封装所有 PC/state 管理，`ThreadContext` 通过组合持有 `std::unique_ptr<SimtPcManager>`
- Phase 2：提取 `RegisterAccessLayer` 类，封装寄存器查找与条件码
- Phase 3：提取内存访问 + 控制流模块
- 每个 Phase 后 `thread_context.cpp` 行数递减、所有已有测试通过

**Non-Goals:**
- 不改变 `WarpState` 或 `WarpContext` 的数据结构
- 不改动 `RegisterBankManager` 或任何 instruction handler
- 不新增功能行为（纯结构重构）
- 不修改 `thread_context.h` 的公开 API 签名（方法签名保留，内部转发到新类）
- 不在 Phase 1 中处理寄存器/内存/控制流的提取

## Decisions

### Decision 1: Phase 1 提取 `SimtPcManager` 而非更细粒度拆分

**选择**：一个统一的 `SimtPcManager` 类封装 PC 管理 + 执行状态管理。

**理由**：PC（`get_pc()`/`set_pc()`/`commit_pc()`）和执行状态（`is_active()`/`is_exited()`/`is_at_barrier()`/`set_state()`/`get_state()`）在同一 `sync_to_warp_state()`/`sync_from_warp_state()` 中耦合。分开会导致两个类间循环依赖。

**构造顺序约束（MR-5）**：`SimtPcManager` 必须在 `warp_id_` / `lane_id_` 计算之后（`init()` line 55-58）构造。如果构造提前到这些计算之前，`SimtPcManager::lane_id_` 将未初始化，导致 `sync_to_warp_state()` 访问错误的 thread state。

**替代方案**：
- `SimtStackView` + `ThreadStateMachine`：会导致 `sync_to_warp_state()` 需要同时修改两个对象
- 仅提取 PC 不提取状态：`is_blocked`→`BAR_SYNC` 翻译仍留在 ThreadContext，不解决核心耦合

### Decision 2: 委托模式保持 `ThreadContext` 的向后兼容 API

**选择**：`ThreadContext` 保留所有现有 public 方法签名，内部转发到 `simt_pc_mgr_->get_pc()` 等。

**理由**：`execute_thread_instruction()` 中 `handler->ExecPipe(this, statement)` 将 `ThreadContext*` 传递给 instruction handler，handler 调用 `context->get_pc()` 等。不改 API 签名意味着零 handler 改动（impact 归零）。

**替代方案**：
- handler 接收 `SimtPcManager*` → 需修改所有 handler + X-Macro 分发签名（40+ 文件，高风险）
- 引入 virtual interface → 虚函数开销影响每条指令执行

### Decision 3: Phase 1 不移除 `call_stack` 和 `bar_id`

**选择**：`call_stack` 和 `bar_id` 留在 `ThreadContext` 直到 Phase 3，但 `SimtPcManager` 可以访问（通过构造注入指针）。

**理由**：`call_stack` 是控制流功能（Phase 3），`bar_id` 是屏障/同步功能。过早迁移引入不必要的跨模块耦合。

### Decision 4: 基线 worktree 使用 `HEAD~1` 作为基线 commit（Phase 0 commit 之后）

**选择**：Phase 0（artifacts commit）之后执行 `git worktree add .worktrees/baseline-pre-c1-phase1 HEAD~1`。

**理由**：Phase 0 commit 之后，`HEAD` = artifacts commit，`HEAD~1` = pre-change baseline。如果在 Phase 0 之前建立 worktree，应使用 `HEAD` 而非 `HEAD~1`（MR-3）。

**注意事项**：
- 如果 main 在此 change 期间有其他 merge，需更新 baseline commit
- 每个 Phase commit 后立即 `ctest` 对比基线

## Risks / Trade-offs

| Risk | 概率 | 缓解措施 |
|------|------|---------|
| `sync_to_warp_state()` 中 `already_blocked` 检查被遗漏 | 中 | lessons-learned §1 跨模块状态翻译：行级 diff `sync_to_warp_state` 的 `is_blocked`/`status==Blocked` 逻辑（line 807-817）。tasks.md 1.4.1 显式列出完整行级清单。 |
| 新类通过 `warp_context_` 裸指针访问 WarpState → use-after-free | 低 | `SimtPcManager` 生命周期在 `ThreadContext` 内；`ThreadContext` 生命周期在 `WarpContext` 内，由 `WarpContext` 保证 |
| 委托转发引入额外函数调用开销 | 低（inlineable） | 所有 PC getter/setter 是简单转发，编译器会 inline（`-O2`）。如果性能回归，使用 `__attribute__((always_inline))` |
| Phase 1 commit 后测试回归，revert 后混入 Phase 2 改动 | 中 | lessons-learned §3：每个 Phase = 独立 commit，revert 该 commit 即可回退 Phase 1 |
| `exec_state_` POD 与新提取类不一致（MR-2） | **高** | Phase 1 `init()` 中 `exec_state_.state` 从 `simt_pc_mgr_->get_state()` 回填。Phase 3 移除 `exec_state_.state` 字段。tasks.md 1.3.2 显式任务。 |
| `set_warp_context()` 不会扇出到 `SimtPcManager`（MR-4） | **中**（`warp_context.cpp:236` 在生产代码中调用 `threads[lane_id]->set_warp_context(this)`） | 在 `ThreadContext::set_warp_context()` 中添加 `simt_pc_mgr_->set_warp_context(warp_ctx)` 扇出。tasks.md 1.3.1 显式任务。 |
| `init()` 中 `SimtPcManager` 构造早于 `lane_id_` 计算（MR-5） | 低（但后果严重） | design.md Decision 1 已文档化构造顺序约束。tasks.md 1.3.2 添加行号注释标记正确构造位置。 |
| `call.cpp:15` 裸字段 `context->state = EXIT` 编译失败（MR-1） | **确定** | tasks.md 1.2.4 在迁移前修复此 handler：`context->state = EXIT` → `context->set_state(EXIT)`。其余 handler 仅通过方法访问，零改动。 |

## Migration Plan

### Phase 1: SIMT PC/State 提取（~3h, Tier 2 合格）

1. 建立基线 worktree
2. 新建 `src/ptxsim/core/simt_pc_manager.h` 和 `simt_pc_manager.cpp`
3. 将以下方法从 `ThreadContext` 迁移到 `SimtPcManager`：
   - `get_pc()`, `set_pc()`, `get_next_pc()`, `set_next_pc()`, `commit_pc()`
   - `sync_from_warp_state()`, `sync_to_warp_state()`
   - `get_state()`, `set_state()`, `is_active()`, `is_exited()`, `is_at_barrier()`
   - `is_valid_pc()`, `is_valid_pc(int)`, `statements_size()`, `get_statement_at()`, `get_current_statement()`
4. `ThreadContext` 保留上述方法为转发委托
5. 新增类型一单元测试 `unit_simt_pc_manager`
6. 运行类型一/二/三全部测试 → 全部通过
7. 独立 commit

### Phase 2: 寄存器访问层提取（~4h）

1. 新建 `src/ptxsim/core/register_access_layer.h` 和 `.cpp`
2. 提取 `acquire_register()`, `register_bank_manager_`, `cc_reg`, 条件码方法
3. 保留 `ThreadContext` 委托
4. 新增类型一单元测试
5. 回归测试 → 独立 commit

### Phase 3: 内存+控制流提取（~3h, 跨季度）

1. 提取内存访问方法到独立类
2. 提取控制流编排方法
3. `ThreadContext` 行数目标 ~200 行
4. 新增 `docs/adr/ADR-0017-pc-management-extraction.md`

### Rollback 策略

每个 Phase 独立 revert：
```bash
git revert <phase-commit>
```
回退后所有测试仍需通过（验证 baseline 等价性）。

## Open Questions

- Q1: `SimtPcManager` 的 `sync_to_warp_state()` 是否需要调用 `notify_pc_changed()` 回调？当前 `ThreadContext` 无此回调，但未来 `WarpScheduler` 可能需感知 PC 变更。
- Q2: Phase 3 是否应引入 `InstructionPipeline` 类替换 `_execute_once()` 的直接 `handler->ExecPipe(this, statement)` 调用？或保持简单委托？
- Q3: 是否需要 ADR-0017 来记录 "why extract PC management into `SimtPcManager`"? 建议：Phase 1 交付后追加。
- Q4: Phase 1 `exec_state_` POD 中 `state` 字段应回填（同步读取 `SimtPcManager::get_state()`）还是直接移除？**已决议**: 回填（`exec_state_.state = simt_pc_mgr_->get_state()`），Phase 3 移除（MR-2）。

## Metis Pre-Implementation Review

**Session**: `ses_0ca442a15ffeTGnLE1vdxlOw45`（2026-07-06）
**Decision**: ⚠️ CONDITIONAL — 5 MUST-RESOLVE 已全部解决
**Resolved**: MR-1 (call.cpp:15 裸字段访问), MR-2 (exec_state_ POD 同步), MR-3 (baseline worktree 引用), MR-4 (set_warp_context 扇出), MR-5 (init 构造顺序)