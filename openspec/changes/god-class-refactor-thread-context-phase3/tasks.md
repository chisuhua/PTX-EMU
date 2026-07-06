# Tasks: god-class-refactor-thread-context-phase3

> **Ref**: `archive/2026-07-06-god-class-refactor-thread-context/` (predecessor)
> **Ref**: `docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 (P1)
> **Lessons-Learned**: Checklist A+D + §1 (cross-module state translation) + §3 (multi-Phase commit) + §4 (baseline worktree) + §6 (artifacts-first)

---

## 0. Artifacts 提交（Phase 0 — 先于实施）

- [ ] 0.1 `git add openspec/changes/god-class-refactor-thread-context-phase3/`
- [ ] 0.2 验证 artifacts 已 tracked：`git ls-files openspec/changes/god-class-refactor-thread-context-phase3/` 不应为空
- [ ] 0.3 `git commit -m "docs(openspec): add god-class-refactor-thread-context-phase3 artifacts"

---

## 1. Phase 3.0 — 前置条件（公开成员 → 私有 + setter）

**目标**: 解决 Phase 3.1 取消时识别的 MemoryAccessor 状态发散根因

### 1.1 审计外部直接赋值

- [ ] 1.1.1 在 `src/` 下 grep 所有 `shared_mem_space =` 直接赋值
- [ ] 1.1.2 grep `local_mem_space =` 同上
- [ ] 1.1.3 grep `name2Sym =` / `name2Share =` / `cta_context_ =` 同上
- [ ] 1.1.4 记录所有外部调用点（用于 Phase 3.0 setter 调用迁移）

### 1.2 公开→私有转换

- [ ] 1.2.1 `include/ptxsim/thread_context.h`：
  - `shared_mem_space` → private `void *shared_mem_space_`
  - `local_mem_space` → private `void *local_mem_space_`
  - 添加 `set_shared_memory_space(void *)` setter（转发到 `mem_access_`）
  - 添加 `set_local_memory_space(void *)` setter（已在 Phase 1+2 实现）
  - 添加 `get_shared_memory_space() const` / `get_local_memory_space() const` 访问器
- [ ] 1.2.2 `src/ptxsim/core/thread_context.cpp`：
  - `initialize_shared_memory()` 改为 setter 委托
  - 删除 `set_local_memory_space()` 现有实现，改为 setter
- [ ] 1.2.3 更新所有外部直接赋值点为 setter 调用

### 1.3 验证

- [ ] 1.3.1 编译通过
- [ ] 1.3.2 全部 174 测试通过
- [ ] 1.3.3 独立 commit：`refactor(core): make shared/local memory private on ThreadContext`

---

## 2. Phase 3.1 — `MemoryAccessor` 提取（~3h）

**目标**: `thread_context.cpp` 从 471 行降至 ~270 行（新增 `memory_accessor.cpp` ~200 行）

### 2.1 Baseline worktree

- [ ] 2.1.1 `git worktree add .worktrees/baseline-pre-c3-phase1 HEAD~1`
- [ ] 2.1.2 baseline 编译通过
- [ ] 2.1.3 baseline ctest 通过

### 2.2 创建 MemoryAccessor 类

- [ ] 2.2.1 新建 `include/ptxsim/memory_accessor.h`：声明 `MemoryAccessor` 类
  - 成员：`void *shared_mem_space_`、`void *local_mem_space_`、CTAContext 指针、symbol table 指针
  - 公开方法：`get_memory_addr()`, `set_shared_memory_space()`, `set_local_memory_space()`, `mov_data()`, `mov()`, `initialize_shared_memory()`, `set_name2sym()`, `set_name2share()`, `set_cta_context()`
- [ ] 2.2.2 新建 `src/ptxsim/core/memory_accessor.cpp`：从 `thread_context.cpp` 迁移所有相关方法
- [ ] 2.2.3 更新 `src/CMakeLists.txt`：添加 `memory_accessor.cpp`

### 2.3 迁移 ThreadContext 方法为委托

- [ ] 2.3.1 修改 `include/ptxsim/thread_context.h`：添加 `std::unique_ptr<MemoryAccessor> mem_access_`，删除 public `shared_mem_space`/`local_mem_space` 成员
- [ ] 2.3.2 所有 `get_memory_addr`/`mov_data`/`mov`/`initialize_shared_memory` 方法改为内联委托
- [ ] 2.3.3 `init()` 创建 `MemoryAccessor` 并注册依赖

### 2.4 验证（行级 Diff）

- [ ] 2.4.1 行级 diff `get_memory_addr` 原实现 vs 新实现（260 行 → ~50 行委托）
- [ ] 2.4.2 行级 diff `initialize_shared_memory` 同上
- [ ] 2.4.3 验证 `shared_mem_space_` 仅通过 setter 设置（不能再从外部直接赋值）

### 2.5 测试验证

- [ ] 2.5.1 ctest unit/integration/e2e 全部通过（174/174 起步）
- [ ] 2.5.2 对比 baseline 无回归
- [ ] 2.5.3 如有测试回归 → 立即 revert Phase 3.1 commit（lessons-learned §3）

### 2.6 Phase 3.1 提交

- [ ] 2.6.1 commit：`refactor(core): extract MemoryAccessor from ThreadContext (Phase 3.1)`

---

## 3. Phase 3.2 — `InstructionPipeline` 提取（~4h）

**目标**: `thread_context.cpp` 从 ~270 行降至 ~150 行（新增 `instruction_pipeline.cpp` ~300 行）

### 3.1 Sub-step 3.2.0 — Handler 签名变更（前置条件）

- [ ] 3.1.1 修改 `include/ptxsim/instruction_handlers.h` 基类 `ExecPipe` 签名
  - 原签名：`ExecPipe(ThreadContext*, ...)` + 各 handler 子签名
  - 新签名：`ExecPipe(ThreadContext*, InstructionPipeline*, ...)`
- [ ] 3.1.2 更新 40+ handler `.cpp` 实现：添加 `InstructionPipeline*` 参数（部分使用 `nullptr`，实施迁移前）
- [ ] 3.1.3 更新 `instruction_handlers.cpp` X-Macro 调度：传递 `nullptr` 作为 pipeline 参数
- [ ] 3.1.4 编译通过 + 全部 174 测试通过
- [ ] 3.1.5 独立 commit：`refactor(core): add InstructionPipeline parameter to handler ExecPipe signature`

### 3.2 Sub-step 3.2.1 — `InstructionPipeline` 类创建

- [ ] 3.2.1 新建 `include/ptxsim/instruction_pipeline.h`：声明 `InstructionPipeline` 类
  - 成员：operand buffers (`operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`)、`dst_operand_reg_name_`、`ThreadContext*` owner reference
- [ ] 3.2.2 新建 `src/ptxsim/core/instruction_pipeline.cpp`：从 `thread_context.cpp` 迁移 `collect_operands`/`commit_operand`
- [ ] 3.2.3 更新 `src/CMakeLists.txt`：添加 `instruction_pipeline.cpp`

### 3.3 Sub-step 3.2.2 — 控制流方法迁移

- [ ] 3.3.1 迁移 `_execute_once()` 方法
- [ ] 3.3.2 迁移 `execute_thread_instruction()` / `init()` / `reset()` / `clear_temporaries()`
- [ ] 3.3.3 迁移 `isIMMorVEC()` / `dump_state()` / `prepare_breakpoint_context()` / `trace_status()` / `print_instruction_status()`
- [ ] 3.3.4 ThreadContext 保留这些方法为内联委托

### 3.4 Sub-step 3.2.3 — 更新 Handler 调用

- [ ] 3.4.1 X-Macro 调度：传递 `thread_context->get_instruction_pipeline()`
- [ ] 3.4.2 验证所有 handler 通过 pipeline 访问 operand buffers

### 3.5 验证

- [ ] 3.5.1 行级 diff `_execute_once()` 原实现 vs 新实现 — 验证 `already_blocked` guard 通过 pipeline 调用保留
- [ ] 3.5.2 编译通过
- [ ] 3.5.3 全部 174 测试通过
- [ ] 3.5.4 对比 Phase 3.1 baseline 无回归
- [ ] 3.5.5 如回归 → 立即 revert Phase 3.2 commits（按 sub-step 反向顺序）

### 3.6 Phase 3.2 提交

- [ ] 3.6.1 commit (per sub-step)：3 个子 commit — `3.2.0 handler signature` + `3.2.1 pipeline class` + `3.2.2 method migration`

---

## 4. Phase 3.3 — 清理 + ADR（~1h）

**目标**: ThreadContext 降至 ~100 行（纯委托 hub）

### 4.1 删除遗留 POD

- [ ] 4.1.1 grep `exec_state_` / `reg_pred_` / `memory_` / `program_ref_` 在 `src/`/`include/`/`tests/` — 确认零读者（Phase 1+2 已验证）
- [ ] 4.1.2 删除 `exec_state_` 等 POD 成员从 `ThreadContext`
- [ ] 4.1.3 更新 `reset()` 移除 `exec_state_` 回填代码

### 4.2 ADR 追加

- [ ] 4.2.1 新建 `docs/adr/0017-pc-management-extraction.md`
  - 记录 Phase 1+2 决策
  - 记录 Phase 3.1 取消教训（状态发散根因分析）
  - 记录 Phase 3.2 路径（handler 签名重构）
- [ ] 4.2.2 更新 `docs/adr/README.md` 添加 ADR-0017 链接

### 4.3 AGENTS.md 同步

- [ ] 4.3.1 更新 `src/ptxsim/core/AGENTS.md`
  - WHERE TO LOOK: 添加 MemoryAccessor + InstructionPipeline 行
  - KEY FILES: 添加新文件条目

### 4.4 验证

- [ ] 4.4.1 全部 174 测试通过
- [ ] 4.4.2 对比 Phase 1+2 baseline 无回归
- [ ] 4.4.3 `wc -l src/ptxsim/core/thread_context.cpp` ≤ 250

### 4.5 Phase 3.3 提交

- [ ] 4.5.1 commit：`refactor(core): delete legacy PODs and add ADR-0017 (Phase 3.3)`

---

## 5. 收尾工作

- [ ] 5.1 移除 baseline worktree
- [ ] 5.2 运行 `./scripts/sanity.sh` 完整回归验证
- [ ] 5.3 OpenSpec archive：移动到 `archive/2026-XX-XX-god-class-refactor-thread-context-phase3/`

---

## Revert 策略

每个 Phase 独立 revert（按反序）：
```bash
git revert <phase-3.3-commit>
git revert <phase-3.2-commit>  # 最后子步骤
git revert <phase-3.2-commit>  # 倒数第二子步骤
git revert <phase-3.1-commit>
git revert <phase-3.0-commit>
```

**纪律**: 任何已有测试回归 → 立即 revert 该 Phase，不得混入后续 commit（lessons-learned §3）。
