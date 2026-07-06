# Tasks: god-class-refactor-thread-context

> **Ref**: `docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 (P1)
> **Ref**: `docs/roadmap/post-phase3-debt-roadmap.md` §1.2 + §3.3
> **Lessons-Learned**: Checklists A+B+D + §1 (cross-module state translation) + §3 (multi-Phase commit) + §4 (baseline worktree) + §6 (artifacts-first)

---

## 0. Artifacts 提交（Phase 0 — 先于实施）

- [ ] 0.1 `git add openspec/changes/god-class-refactor-thread-context/`
- [ ] 0.2 验证 artifacts 已 tracked：`git ls-files openspec/changes/god-class-refactor-thread-context/` 不应为空
- [ ] 0.3 `git commit -m "docs(openspec): add god-class-refactor-thread-context artifacts (Phase 1 proposal)

Metis pre-impl review applied.
Refs: lessons-learned §6, §20
Refs: debt-audit-2026-07-02.md §2.2 C-1
"` 

---

## 1. Phase 1 — SIMT PC/State 提取 (`SimtPcManager`, ~3h, Tier 2 合格)

**目标**: `thread_context.cpp` 从 884 行降至 ~735 行（新增 `simt_pc_manager.cpp` ~120 行 + `simt_pc_manager.h` ~60 行）

### 1.1 建立 baseline

- [ ] 1.1.1 建立基线 worktree：`git worktree add .worktrees/baseline-pre-c1-phase1 HEAD~1`
- [ ] 1.1.2 baseline 全量编译：`cd .worktrees/baseline-pre-c1-phase1 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
- [ ] 1.1.3 基线测试通过：`cd .worktrees/baseline-pre-c1-phase1/build && ctest --output-on-failure` — 记录通过/失败数量
- [ ] 1.1.4 记录 baseline ctest 结果到 `tasks.md` 本小节下方

### 1.2 创建 SimtPcManager 类

- [ ] 1.2.1 新建 `include/ptxsim/simt_pc_manager.h`：声明 `SimtPcManager` 类
  - 成员：`WarpContext* warp_context_`, `int lane_id_`, `EXE_STATE state_`, `std::vector<StatementContext>* statements_`（通过构造注入）
  - 公开方法：`get_pc()`, `set_pc(int)`, `get_next_pc()`, `set_next_pc(int)`, `commit_pc()`, `get_state()`, `set_state(EXE_STATE)`, `is_active()`, `is_exited()`, `is_at_barrier()`, `sync_from_warp_state()`, `sync_to_warp_state()`, `is_valid_pc()`, `is_valid_pc(int)`, `statements_size()`, `get_statement_at(int)`, `get_current_statement()`
- [ ] 1.2.2 新建 `src/ptxsim/core/simt_pc_manager.cpp`：实现上述方法，从 `thread_context.cpp` 逐行迁移（保持逻辑不变）
- [ ] 1.2.3 更新 `src/ptxsim/core/CMakeLists.txt`：添加 `simt_pc_manager.cpp` 到 `SOURCES`

### 1.3 迁移 ThreadContext 方法为委托

- [ ] 1.3.1 修改 `include/ptxsim/thread_context.h`：
  - 删除 `state` 成员变量，改为持有 `std::unique_ptr<SimtPcManager> simt_pc_mgr_`
  - 所有 PC/state 相关方法改为内联委托：`int get_pc() const { return simt_pc_mgr_->get_pc(); }`
  - 保留所有其他成员和方法不变
- [ ] 1.3.2 修改 `src/ptxsim/core/thread_context.cpp`：
  - 删除 17 个已迁移到 `SimtPcManager` 的方法实现
  - 修改 `init()`：创建 `SimtPcManager` 实例并传递依赖
  - 修改 `reset()`：委托到 `simt_pc_mgr_->set_pc(0)` 等
  - 修改 `_execute_once()`：`get_pc()` / `set_next_pc()` / `commit_pc()` 自动通过委托生效
- [ ] 1.3.3 `#include "ptxsim/simt_pc_manager.h"` 添加到 `thread_context.h`

### 1.4 验证（Checklist A: 函数迁移完整性）

- [ ] 1.4.1 行级 diff `sync_to_warp_state` 原实现 vs 新实现 — 验证 `already_blocked` guard 保留
- [ ] 1.4.2 行级 diff `sync_from_warp_state` 原实现 vs 新实现 — 验证 `ThreadStatus → EXE_STATE` 翻译不变
- [ ] 1.4.3 验证 `set_pc()` 同时写入 `pc` 和 `next_pc` 的行为保留
- [ ] 1.4.4 编译验证：`cmake --build build -j$(nproc)` 无错误

### 1.5 测试覆盖（必跑）

- [ ] 1.5.1 类型一（单元）：ctest 全部单元测试通过
  ```bash
  cd build && ctest -L "unit" --output-on-failure
  ```
- [ ] 1.5.2 类型二（集成）：ctest 全部集成测试通过（含 SIMT/barrier/divergence）
  ```bash
  cd build && ctest -L "integration" --output-on-failure
  ```
- [ ] 1.5.3 类型三（E2E）：ctest 全部 E2E 测试通过
  ```bash
  cd build && ctest -L "e2e" --output-on-failure
  ```
- [ ] 1.5.4 对比基线 worktree ctest 结果 — 无回归（通过数 ≥ baseline 通过数）
- [ ] 1.5.5 如果任何测试回归 → **立即 revert Phase 1 commit**（lessons-learned §3）

### 1.6 Phase 1 提交

- [ ] 1.6.1 新增类型一单元测试 `tests/unit/core/test_simt_pc_manager.cpp`（验证 `SimtPcManager` 独立行为）
- [ ] 1.6.2 更新 `tests/unit/core/CMakeLists.txt`：添加 `unit_simt_pc_manager` ctest
- [ ] 1.6.3 `git add` 所有已修改/新增文件
- [ ] 1.6.4 commit：
  ```
  refactor(core): extract SimtPcManager from ThreadContext (Phase 1)

  Extracts PC management and execution state from 884-line god class
  ThreadContext into standalone SimtPcManager class (~120 new lines).

  - Migrated 17 methods: get_pc/set_pc/commit_pc/sync_from_warp_state/etc.
  - ThreadContext retains all public API signatures as delegation wrappers
  - Zero handler changes required (backward-compatible delegation)
  - Added unit_simt_pc_manager test (type-1)

  Fix #1: SimtPcManager class extraction
  Refs: lessons-learned §1 (cross-module state translation verified)
  Refs: lessons-learned §3 (single-Phase commit + revert strategy)
  Refs: debt-audit-2026-07-02.md §2.2 C-1
  ```

---

## 2. Phase 2 — 寄存器访问层提取 (`RegisterAccessLayer`, ~4h)

**目标**: `thread_context.cpp` 从 ~735 行降至 ~675 行（新增 `register_access_layer.cpp` ~80 行 + `.h` ~50 行）

### 2.1 建立 Phase 2 baseline

- [ ] 2.1.1 确认 Phase 1 commit 已通过所有测试（不跑则 Phase 2 基线 = Phase 1 终态）
- [ ] 2.1.2 运行 `cd build && ctest --output-on-failure` 记录 Phase 2 起始状态

### 2.2 创建 RegisterAccessLayer 类

- [ ] 2.2.1 新建 `include/ptxsim/register_access_layer.h`：声明 `RegisterAccessLayer` 类
  - 成员：`std::shared_ptr<RegisterBankManager> register_bank_manager_`, `int warp_id_`, `int lane_id_`, `ConditionCodeRegister cc_reg_`
  - 公开方法：`acquire_register(const RegOperand&, std::vector<Qualifier>)`, `get_register_bank_manager()`, `set_register_bank_manager()`, `get_condition_codes()`, `set_condition_codes()`
- [ ] 2.2.2 新建 `src/ptxsim/core/register_access_layer.cpp`：实现上述方法，从 `thread_context.cpp` 逐行迁移
- [ ] 2.2.3 更新 `src/ptxsim/core/CMakeLists.txt`：添加 `register_access_layer.cpp`

### 2.3 迁移方法为委托

- [ ] 2.3.1 修改 `include/ptxsim/thread_context.h`：持有 `std::unique_ptr<RegisterAccessLayer> reg_access_`
- [ ] 2.3.2 寄存器方法改为内联委托
- [ ] 2.3.3 修改 `src/ptxsim/core/thread_context.cpp`：删除已迁移方法、修改 `init()` 创建 `RegisterAccessLayer`

### 2.4 验证与测试

- [ ] 2.4.1 编译 + 全部 ctest（类型一/二/三）通过
- [ ] 2.4.2 对比 Phase 1 基线 — 无回归
- [ ] 2.4.3 新增类型一单元测试 `tests/unit/core/test_register_access_layer.cpp`

### 2.5 Phase 2 提交

- [ ] 2.5.1 `git add` + commit：
  ```
  refactor(core): extract RegisterAccessLayer from ThreadContext (Phase 2)

  Extracts register access and condition-code management from ThreadContext
  into standalone RegisterAccessLayer class.

  Fix #2: RegisterAccessLayer class extraction
  Refs: debt-audit-2026-07-02.md §2.2 C-1
  ```

---

## 3. Phase 3 — 内存访问 + 控制流提取（~3h, 跨季度）

**目标**: `thread_context.cpp` 从 ~675 行降至 ~200 行（`ThreadContext` 成为纯委托编排层）

### 3.1 提取内存访问

- [ ] 3.1.1 新建 `include/ptxsim/memory_accessor.h` + `src/ptxsim/core/memory_accessor.cpp`
- [ ] 3.1.2 迁移 `get_memory_addr()` / `acquire_operand()` / `mov()` / `mov_data()` / `initialize_shared_memory()` / `set_local_memory_space()`
- [ ] 3.1.3 迁移 `shared_mem_space`, `local_mem_space`, `name2Share`, `name2Sym`, `cta_context_`, `SHMEMADDR` 到新类

### 3.2 提取控制流编排

- [ ] 3.2.1 新建 `include/ptxsim/instruction_pipeline.h` + `src/ptxsim/core/instruction_pipeline.cpp`
- [ ] 3.2.2 迁移 `_execute_once()` / `execute_thread_instruction()` / `collect_operands()` / `commit_operand()` / `init()` / `reset()` / `clear_temporaries()` / `isIMMorVEC()` / `dump_state()` / `prepare_breakpoint_context()` / `trace_status()` / `print_instruction_status()`
- [ ] 3.2.3 迁移 `operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`, `dst_operand_reg_name_`, `call_stack` 到新类

### 3.3 清理与文档

- [ ] 3.3.1 `ThreadContext` 类缩减到 ~200 行：所有方法为内联委托
- [ ] 3.3.2 删除 `exec_state_`, `reg_pred_`, `memory_`, `program_ref_` POD 到各自目标类
- [ ] 3.3.3 新增 `docs/adr/0017-pc-management-extraction.md`（记录决策理由 + 3-Phase 历程）
- [ ] 3.3.4 更新 `src/ptxsim/core/AGENTS.md`：替换 `WHERE TO LOOK` 表中 `thread_context.cpp` 条目
- [ ] 3.3.5 全部 ctest 通过 + baseline 对比无回归

### 3.4 Phase 3 提交

- [ ] 3.4.1 独立 commit（可进一步拆为 3.1 + 3.2 + 3.3 三个子 commit）
  ```
  refactor(core): extract memory accessor + instruction pipeline (Phase 3)

  Completes god-class refactor of ThreadContext (884 -> ~200 lines).
  ThreadContext is now a pure delegation orchestrator:
  - SimtPcManager: PC + execution state (Phase 1)
  - RegisterAccessLayer: register access + condition codes (Phase 2)
  - MemoryAccessor: memory address resolution (Phase 3.1)
  - InstructionPipeline: execution orchestration (Phase 3.2)

  Refs: debt-audit-2026-07-02.md §2.2 C-1
  Refs: ADR-0017 (pc-management-extraction)
  ```

---

## 4. 收尾工作

- [ ] 4.1 移除 baseline worktree：`cd /workspace/project/PTX-EMU && git worktree remove .worktrees/baseline-pre-c1-phase1`
- [ ] 4.2 运行 `./scripts/sanity.sh` 完整回归验证
- [ ] 4.3 OpenSpec archive（Phase 3 全部完成且合并后）

---

## Revert 策略

每个 Phase 独立 revert：
```bash
# Phase 1 回归 → revert Phase 1 的单个 commit
git revert <phase-1-commit-hash>

# Phase 2 回归 → revert Phase 2 commit（Phase 1 保持不变）
git revert <phase-2-commit-hash>

# Phase 1+2 回归 → revert 两个 commit（先 Phase 2 后 Phase 1）
git revert <phase-2-commit-hash> && git revert <phase-1-commit-hash>
```

**纪律**: 任何已有测试回归 → 立即 revert 该 Phase，不得混入后续 commit（lessons-learned §3）。