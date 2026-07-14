# Tasks: god-class-refactor-thread-context-phase3

> **Ref**: `archive/2026-07-06-god-class-refactor-thread-context/` (predecessor)
> **Ref**: `docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 (P1)
> **Lessons-Learned**: Checklist A+D + §1 (cross-module state translation) + §3 (multi-Phase commit) + §4 (baseline worktree) + §6 (artifacts-first) + §7 (pre-impl Metis review)
> **Pre-impl Metis review**: `ses_0a11eea61ffe0HTZX5uQEUvP7L` (2026-07-14) — all 10 blocking findings addressed in this revision
> **Verified baseline (2026-07-14)**: `src/ptxsim/core/thread_context.cpp` 727 lines, `include/ptxsim/thread_context.h` 324 lines, **ctest baseline = 198 tests**

---

## 0. Artifacts 提交（Phase 0 — 先于实施）

- [ ] 0.1 `git add openspec/changes/god-class-refactor-thread-context-phase3/`
- [ ] 0.2 验证 artifacts 已 tracked：`git ls-files openspec/changes/god-class-refactor-thread-context-phase3/` 不应为空
- [ ] 0.3 清理工作区无关变更（lessons-learned §3 + §6）：
  - `git status` 显示有未提交删除 `openspec/changes/archive/2026-06-24-integrate-barrier-module-cta-warp/` 和未跟踪 `openspec/changes/cleanup-deprecated-barrier-apis/`、`openspec/changes/migrate-bar-warp-sync-to-barrier-module/`、`docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md`、以及 `AGENTS.md` 修改
  - 决策：将这些无关变更**单独 commit** 或 **stash**，**不**与 Phase 3 提交混入
- [ ] 0.4 `git commit -m "docs(openspec): revise god-class-refactor-thread-context-phase3 per Metis review 2026-07-14"
- [ ] 0.5 记录 Phase 3.0 commit SHA（后续 baseline worktree 绑定用）：`<phase-3.0-commit-sha>`

---

## 1. Phase 3.0 — 前置条件（公开成员 → 私有 + setter）

**目标**: 解决 Phase 3.1 取消时识别的 MemoryAccessor 状态发散根因。

**注意**: Phase 3.0 仅做 setter/getter 转换，**不**创建 `MemoryAccessor` 类（按 lessons-learned §3，prerequisite 变更必须独立 commit，单独可回退）。

### 1.1 审计外部直接赋值

- [ ] 1.1.1 `grep -rn 'shared_mem_space\s*=' src/ include/ tests/` — 记录所有外部直接赋值（已知：`src/ptxsim/core/cta_context.cpp:320`）
- [ ] 1.1.2 `grep -rn 'local_mem_space\s*=' src/ include/ tests/` — 记录所有外部直接赋值
- [ ] 1.1.3 `grep -rn 'name2Sym\s*=' src/ include/ tests/` — 区分 map 容器赋值（CTAContext 内部）与直接给 ThreadContext 字段赋值
- [ ] 1.1.4 `grep -rn 'name2Share\s*=' src/ include/ tests/` — 同上
- [ ] 1.1.5 `grep -rn 'cta_context_\s*=' src/ include/ tests/` — 记录所有外部直接赋值
- [ ] 1.1.6 把审计结果写到 `openspec/changes/god-class-refactor-thread-context-phase3/phase-3.0-audit.md`，**作为 commit 提交**

### 1.2 公开→私有转换

- [ ] 1.2.1 `include/ptxsim/thread_context.h`:
  - `shared_mem_space` (line 74) → `private void *shared_mem_space_ = nullptr;` 加 `set_shared_memory_space(void*)` + `get_shared_memory_space() const`
  - `local_mem_space` (line 77) → `private void *local_mem_space_ = nullptr;` 加 `set_local_memory_space(void*)` + `get_local_memory_space() const`（`set_local_memory_space` 现有实现保留为 setter）
  - `name2Sym` (line 40) → `private` 加 `set_name2sym(std::map<...>*)` + `get_name2sym() const`
  - `name2Share` (line 41) → `private` 加 `set_name2share(std::map<...>*)` + `get_name2share() const`
  - `cta_context_` (private, set in `init()`) → 加 `set_cta_context(CTAContext*)` + `get_cta_context() const`
- [ ] 1.2.2 `src/ptxsim/core/thread_context.cpp`:
  - `init()` (lines 41-92) 内部赋值改为 setter 调用（`this->set_name2sym(name2Sym)` 等）
  - 保留所有现有 public 方法签名（`get_memory_addr`, `set_local_memory_space`, `mov`, `mov_data`, `initialize_shared_memory`）

### 1.3 外部直接赋值迁移

- [ ] 1.3.1 `src/ptxsim/core/cta_context.cpp:320` — `thread->shared_mem_space = shared_mem_space` → `thread->set_shared_memory_space(shared_mem_space)`
- [ ] 1.3.2 把 1.1.6 审计中识别的其他外部直接赋值也全部迁移

### 1.4 验证

- [ ] 1.4.1 编译通过
- [ ] 1.4.2 全部 198 测试通过
- [ ] 1.4.3 独立 commit：`refactor(core): make shared/local memory private on ThreadContext (Phase 3.0)`
- [ ] 1.4.4 记录 commit SHA：`<phase-3.0-commit-sha>`

---

## 2. Phase 3.1 — `MemoryAccessor` 提取（~3h）

**目标**: 把内存相关状态从 `ThreadContext` 移到 `MemoryAccessor`。`thread_context.cpp` 727 → ~530 行；新增 `memory_accessor.cpp` ~250 行。

### 2.1 Baseline worktree（绑定到 Phase 3.0 commit，**不**是 `HEAD~1`）

- [ ] 2.1.1 `git worktree add .worktrees/baseline-pre-c3-phase1 <phase-3.0-commit-sha>` （**不**用 `HEAD~1`，否则基线不包含 Phase 3.0 变更，对比失去意义）
- [ ] 2.1.2 baseline 编译通过
- [ ] 2.1.3 baseline 198 个 ctest 全部通过
- [ ] 2.1.4 记录 baseline commit SHA

### 2.2 TDD Red — 单元测试先行

- [ ] 2.2.1 新建 `tests/unit/core/test_memory_accessor.cpp`：
  - 测试 1: `set_shared_memory_space(addr)` → `get_shared_memory_space()` 返回同一地址
  - 测试 2: `initialize_shared_memory` 首次成功；二次同地址成功；二次不同地址抛 `InvalidMemoryAccessException`
  - 测试 3: 构造带 mock `ThreadContext*` 的 `MemoryAccessor`，调用 `get_memory_addr("%tid.x")` 返回 `thread_->acquire_register` 的结果
- [ ] 2.2.2 `cmake --build build --target unit_memory_accessor` — **确认测试编译失败**（`MemoryAccessor` 类尚不存在）
- [ ] 2.2.3 记录 Red 阶段结果

### 2.3 创建 `MemoryAccessor` 类

- [ ] 2.3.1 新建 `include/ptxsim/core/memory_accessor.h`：声明 `MemoryAccessor` 类
  - 私有成员：`void *shared_mem_space_`、`void *local_mem_space_`、3 个 non-owning 指针（`name2Sym_`、`name2Share_`、`cta_context_`）、`ThreadContext *thread_`、`static uint64_t SHMEMADDR_`（**类静态，非文件静态**）
  - 公开方法：`get_memory_addr(const AddrOperand&, const std::vector<Qualifier>&)`、`mov_data`、`mov`、`initialize_shared_memory`、`set_shared_memory_space`、`get_shared_memory_space`、`set_local_memory_space`、`get_local_memory_space`、`set_name2sym`、`set_name2share`、`set_cta_context`、`set_thread`、`get_thread`
  - 构造器：`MemoryAccessor(ThreadContext *thread)` 必须设置 `thread_`
- [ ] 2.3.2 新建 `src/ptxsim/core/memory_accessor.cpp`（~250 行）
  - `uint64_t MemoryAccessor::SHMEMADDR_ = 0;` 类静态定义
  - 从 `thread_context.cpp` 迁移 `get_memory_addr`、`mov_data`、`mov`、`initialize_shared_memory`、`set_local_memory_space` 的实现
  - **`get_memory_addr` 不使用 `std::function` 回调** — 通过 `thread_->acquire_register()` 直接调用（design.md Decision 5）
- [ ] 2.3.3 更新 `src/CMakeLists.txt` 与 `src/ptxsim/core/CMakeLists.txt` 添加 `memory_accessor.cpp`
- [ ] 2.3.4 更新 `include/ptxsim/thread_context.h`：
  - 添加 `std::unique_ptr<MemoryAccessor> mem_access_` 私有成员
  - 添加 `set_mem_accessor(MemoryAccessor*)` 内部 setter（由 `init()` 调用）
- [ ] 2.3.5 `ThreadContext::init()` 在 `simt_pc_mgr_` 创建之后添加 `mem_access_ = std::make_unique<MemoryAccessor>(this)`

### 2.4 迁移 ThreadContext 方法为委托

- [ ] 2.4.1 `get_memory_addr` → 单行 inline 委托 `return mem_access_->get_memory_addr(op, qualifiers);`
- [ ] 2.4.2 `set_local_memory_space` → 单行 inline 委托（同时保留对 `local_mem_space_` 的回填，因为 Phase 3.0 加了这个字段）
- [ ] 2.4.3 `mov_data` / `mov` → 单行 inline 委托
- [ ] 2.4.4 `initialize_shared_memory` → 单行 inline 委托

### 2.5 TDD Green + 行级 Diff 验证

- [ ] 2.5.1 重新构建 `unit_memory_accessor` 目标 — 3 个测试通过
- [ ] 2.5.2 `wc -l src/ptxsim/core/thread_context.cpp` ≤ 600（粗略验证）
- [ ] 2.5.3 `wc -l src/ptxsim/core/memory_accessor.cpp` ~250
- [ ] 2.5.4 行级 diff `get_memory_addr` 原实现 vs 新实现（除 `thread_context.cpp` 内的 inline 委托外，行为必须字节对齐）
- [ ] 2.5.5 行级 diff `initialize_shared_memory` 同上
- [ ] 2.5.6 验证 `shared_mem_space_` 仅通过 setter 设置（外部 grep 验证 `shared_mem_space =` 0 处直接赋值）

### 2.6 完整回归

- [ ] 2.6.1 `ctest --output-on-failure` — 全部 198 个原有测试 + 3 个新单元测试 = 201 通过
- [ ] 2.6.2 对比 baseline `<phase-3.0-commit-sha>` 无回归
- [ ] 2.6.3 屏障回归子集必须通过（设计 §R1 屏障/分歧核心路径）:
  - `ctest -R 'barrier|post_barrier|divergence|sync' --output-on-failure`
  - 已知相关测试: `tests/integration/barrier/`, `tests/integration/divergence/test_post_barrier_two_halves.cpp`, `tests/unit/barrier/test_post_barrier_two_halves.cpp`
- [ ] 2.6.4 如有测试回归 → 立即 revert Phase 3.1 commit（lessons-learned §3），**不**混入后续 commit

### 2.7 Phase 3.1 提交

- [ ] 2.7.1 `git add src/ptxsim/core/memory_accessor.{h,cpp} src/ptxsim/core/CMakeLists.txt src/CMakeLists.txt include/ptxsim/thread_context.h src/ptxsim/core/thread_context.cpp tests/unit/core/test_memory_accessor.cpp`
- [ ] 2.7.2 `git commit -m "refactor(core): extract MemoryAccessor from ThreadContext (Phase 3.1)

- Move shared/local memory and symbol table references to MemoryAccessor
- 3 new type-1 unit tests (TDD red-green)
- Barrier regression verified
- 198 existing tests pass"`
- [ ] 2.7.3 记录 Phase 3.1 commit SHA

---

## 3. Phase 3.2 — `InstructionPipeline` 提取（~4h，REDESIGNED：accessor 方案）

**目标**: `thread_context.cpp` ~530 → ~300 行；新增 `instruction_pipeline.cpp` ~350 行。**handler 签名不变**（accessor 方案，design.md Decision 2）。

### 3.1 Sub-step 3.2.0 — 添加 operand-buffer accessor（前置条件）

- [ ] 3.1.1 `include/ptxsim/thread_context.h` 添加：
  ```cpp
  std::vector<void*> &get_operand_collected();
  const std::vector<void*> &get_operand_collected() const;
  std::vector<char> &get_operand_is_immediate();
  const std::vector<char> &get_operand_is_immediate() const;
  ```
- [ ] 3.1.2 初始实现返回 `ThreadContext` 自身字段（`operand_collected_`、`operand_is_immediate_`，仍是 public 字段）
- [ ] 3.1.3 编译通过 + 全部 198 测试通过
- [ ] 3.1.4 独立 commit：`refactor(core): add operand-buffer accessors to ThreadContext (Phase 3.2.0)`
- [ ] 3.1.5 记录 commit SHA：`<phase-3.2.0-commit-sha>`

### 3.2 Sub-step 3.2.1 — 迁移 4 个 PipelineHandler 基类

- [ ] 3.2.1 `src/ptxsim/instruction_base.cpp:172-173` (`GenericPipelineHandler::executeOperation`):
  - `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`
  - `&context->operand_is_immediate_` → `&context->get_operand_is_immediate()`
- [ ] 3.2.2 `src/ptxsim/instruction_base.cpp:200` (`AtomicPipelineHandler::executeOperation`):
  - `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`
- [ ] 3.2.3 `src/ptxsim/instruction_base.cpp:231` (`Tcgen05PipelineHandler::executeOperation`):
  - `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`
- [ ] 3.2.4 编译通过 + 全部 198 测试通过
- [ ] 3.2.5 屏障回归子集通过（`ctest -R 'barrier|post_barrier|tcgen05' --output-on-failure`）
- [ ] 3.2.6 独立 commit：`refactor(core): route PipelineHandler base classes through operand accessors (Phase 3.2.1)`
- [ ] 3.2.7 记录 commit SHA：`<phase-3.2.1-commit-sha>`

### 3.3 Sub-step 3.2.2 — 迁移 `BarWarpSyncHandler`

- [ ] 3.3.1 `src/ptxsim/instructions/barrier.cpp:92-93`:
  - `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`
  - `&context->operand_is_immediate_` → `&context->get_operand_is_immediate()`
- [ ] 3.3.2 编译通过 + 全部 198 测试通过
- [ ] 3.3.3 屏障回归子集通过（`ctest -R 'barrier|post_barrier|warp_sync' --output-on-failure`，重点 `tests/integration/divergence/test_post_barrier_two_halves.cpp`）
- [ ] 3.3.4 独立 commit：`refactor(core): route BarWarpSyncHandler through operand accessor (Phase 3.2.2)`
- [ ] 3.3.5 记录 commit SHA：`<phase-3.2.2-commit-sha>`

### 3.4 Sub-step 3.2.3 — 创建 `InstructionPipeline` 类（TDD）

#### 3.4.1 TDD Red — 单元测试 + 集成测试先行

- [ ] 3.4.1.1 新建 `tests/unit/core/test_instruction_pipeline.cpp`:
  - 测试 1: `collect_operands` 设置 `operand_collected_[i] = operands[i].operand_phy_addr` 且 `operand_is_immediate_[i] = IMM 标志`
  - 测试 2: 多 VEC push-must-pair-with-pop — V4 指令后 `vecOp_phy_addrs_` 恰好有 1 个新 entry（锁住 `thread_context.cpp:63-66` BUGFIX 注释语义）
  - 测试 3: `execute_thread_instruction` PC 生命周期 — `get_pc()` 在前，`set_next_pc(current_pc+1) + commit_pc()` 在后 → `get_pc()` 返回 `current_pc+1`（锁住 AGENTS.md §CONVENTIONS 不变式）
- [ ] 3.4.1.2 新建 `tests/integration/pc/test_pc_lifecycle_invariant.cpp`:
  - 集成测试：通过 `execute_warp_instruction` 驱动一个非分支非屏障指令，断言 `pc` 精确 +1
- [ ] 3.4.1.3 `cmake --build build --target unit_instruction_pipeline integration_pc_lifecycle` — **确认测试编译失败**（`InstructionPipeline` 类尚不存在）

#### 3.4.2 实现 `InstructionPipeline`

- [ ] 3.4.2.1 新建 `include/ptxsim/core/instruction_pipeline.h`（~80 行）
  - 私有成员：`std::vector<void*> operand_collected_`、`std::vector<char> operand_is_immediate_`、`std::vector<std::vector<void*>> vecOp_phy_addrs_`、`ThreadContext *thread_`
  - 公开方法：`InstructionPipeline(ThreadContext*)`、`collect_operands`、`commit_operand`、`clear_temporaries`、`isIMMorVEC`、`dump_state`、`prepare_breakpoint_context`、`trace_status` 模板、`print_instruction_status`、`_execute_once`、`execute_thread_instruction`、`get_operand_collected`、`get_operand_is_immediate`
- [ ] 3.4.2.2 新建 `src/ptxsim/core/instruction_pipeline.cpp`（~350 行）— 从 `thread_context.cpp` 迁移 `collect_operands`、`commit_operand`、`_execute_once`、`execute_thread_instruction`、`isIMMorVEC`、`dump_state`、`prepare_breakpoint_context`、`print_instruction_status` 等
- [ ] 3.4.2.3 更新 `src/CMakeLists.txt` 与 `src/ptxsim/core/CMakeLists.txt`
- [ ] 3.4.2.4 `include/ptxsim/thread_context.h`:
  - 添加 `std::unique_ptr<InstructionPipeline> instruction_pipeline_` 私有成员
  - 把 `operand_collected` (line 151-152) 改为 `private operand_collected_`
  - 把 `operand_is_immediate_` (line 167) 已经是 private（验证），如仍 public 改 private
  - accessors 改为转发：`return instruction_pipeline_->get_operand_collected();`
- [ ] 3.4.2.5 `ThreadContext::init()` 在 `warp_id_/lane_id_` 计算**之后**（lines 57-58）创建 `instruction_pipeline_ = std::make_unique<InstructionPipeline>(this)`

#### 3.4.3 TDD Green

- [ ] 3.4.3.1 `unit_instruction_pipeline` 3 个测试通过
- [ ] 3.4.3.2 `integration_pc_lifecycle` 通过
- [ ] 3.4.3.3 全部 198 个原有测试通过
- [ ] 3.4.3.4 独立 commit：`refactor(core): extract InstructionPipeline from ThreadContext (Phase 3.2.3)`
- [ ] 3.4.3.5 记录 commit SHA：`<phase-3.2.3-commit-sha>`

### 3.5 Sub-step 3.2.4 — 迁移控制流方法（最高风险区）

- [ ] 3.5.1 迁移 `ThreadContext::_execute_once` (lines 101-150) → `InstructionPipeline::_execute_once`:
  - **PC 生命周期保留**（AGENTS.md §CONVENTIONS, lessons-learned §1）:
    - `int current_pc = get_pc();` (line 121)
    - `StatementContext &statement = (*statements)[current_pc];` (line 122)
    - `set_next_pc(current_pc + 1);` (line 130)
    - `handler->ExecPipe(this, statement);` (line 141)
    - `commit_pc();` (line 149)
  - 顺序与原实现**字节对齐**
- [ ] 3.5.2 迁移 `execute_thread_instruction`、`collect_operands`、`commit_operand`、`clear_temporaries`、`isIMMorVEC`、`dump_state`、`prepare_breakpoint_context`、`print_instruction_status`
- [ ] 3.5.3 `ThreadContext` 保留所有这些方法为 inline 委托（一行：`return instruction_pipeline_->method(...);`）
- [ ] 3.5.4 行级 diff `_execute_once` 原实现 vs 新实现 — 验证 `set_next_pc → ExecPipe → commit_pc` 顺序与行号保持
- [ ] 3.5.5 编译通过 + 全部 198 测试通过
- [ ] 3.5.6 **Oracle 建议 review**（design.md 最后一段）: 在 commit 之前调 Oracle 审查 `_execute_once` 迁移的 byte-level diff，确认 PC 生命周期不变
- [ ] 3.5.7 独立 commit：`refactor(core): migrate control-flow methods to InstructionPipeline (Phase 3.2.4)`
- [ ] 3.5.8 记录 commit SHA：`<phase-3.2.4-commit-sha>`

### 3.6 Phase 3.2 最终验证

- [ ] 3.6.1 全部 198 + 4 新单元测试 + 1 集成测试 = 203 通过
- [ ] 3.6.2 对比 Phase 3.1 baseline `<phase-3.1-commit-sha>` 无回归
- [ ] 3.6.3 屏障回归全集合通过:
  - `ctest -L 'unit;barrier' --output-on-failure`
  - `ctest -L 'integration;barrier' --output-on-failure`
  - `ctest -L 'integration;divergence' --output-on-failure`
- [ ] 3.6.4 `wc -l src/ptxsim/core/thread_context.cpp` ≤ 350（中间值，Phase 3.3 再降到 ≤ 300）
- [ ] 3.6.5 `wc -l src/ptxsim/core/instruction_pipeline.cpp` ~350
- [ ] 3.6.6 如有测试回归 → 立即 revert Phase 3.2 commits（按 3.2.4 → 3.2.3 → 3.2.2 → 3.2.1 → 3.2.0 反向顺序）

---

## 4. Phase 3.3 — 清理 + ADR（~1h）

**目标**: `thread_context.cpp` 降至 ~300 行（纯委托 hub）；新增 ADR-0019；删除 4 个遗留 POD。

### 4.1 删除遗留 POD（前置条件：先停后删）

- [ ] 4.1.1 grep `exec_state_` / `reg_pred_` / `memory_` / `program_ref_` 在 `src/`/`include/`/`tests/` — 确认**当前树**（post-Phase-3.2）的读者
- [ ] 4.1.2 已知写入点（来自代码审计）:
  - `src/ptxsim/core/thread_context.cpp:79-91` (`init()`) — 仍回填 4 个 POD
  - `src/ptxsim/core/thread_context.cpp:225-226` (`reset()`) — 仍回填 `exec_state_`
- [ ] 4.1.3 **第一步**：删除 `init()` 和 `reset()` 中的回填代码 — 单独 commit `refactor(core): stop back-filling legacy PODs in init/reset (Phase 3.3.a)`
- [ ] 4.1.4 编译通过 + 全部 203 测试通过
- [ ] 4.1.5 **第二步**：删除 4 个 POD 字段从 `ThreadContext` (`thread_context.h` + `.cpp`) — 单独 commit `refactor(core): delete legacy PODs exec_state_/reg_pred_/memory_/program_ref_ (Phase 3.3.b)`
- [ ] 4.1.6 编译通过 + 全部 203 测试通过

### 4.2 新建 ADR-0019

- [ ] 4.2.1 新建 `docs/adr/0019-pc-management-extraction.md`:
  - 复制 `docs/adr/template.md` 为起点
  - 记录 Phase 1+2 决策（`SimtPcManager`、`RegisterAccessLayer` 提取）
  - 记录 Phase 3.1 取消教训（状态发散根因分析，外部直接赋值）
  - 记录 Phase 3.2 accessor 方案（避免 handler 签名变更）
  - 链接 Metis 会话 `ses_0a11eea61ffe0HTZX5uQEUvP7L` 与对应 proposal/design
- [ ] 4.2.2 更新 `docs/adr/README.md`:
  - 在 "Active" 表格追加 `[0019](./0019-pc-management-extraction.md) | ThreadContext 持续瘦身：MemoryAccessor + InstructionPipeline accessor 方案 | Active | 2026-07-14 | openspec/changes/god-class-refactor-thread-context-phase3/`
  - "ADR 总数" 改为 16
  - "最后更新" 改为 2026-07-14
  - "最近更新" 表追加 2026-07-14 行
- [ ] 4.2.3 独立 commit：`docs(adr): add 0019 pc-management-extraction (Phase 3.3)`

### 4.3 AGENTS.md 同步

- [ ] 4.3.1 更新 `src/ptxsim/core/AGENTS.md`:
  - `WHERE TO LOOK` 表格追加：
    - `Per-thread memory resolution` → `memory_accessor.cpp` (Phase 3.1 extract)
    - `Per-instruction execution orchestration` → `instruction_pipeline.cpp` (Phase 3.2 extract)
  - `KEY FILES` 表格追加：
    - `memory_accessor.cpp` — Memory address resolution + data movement
    - `instruction_pipeline.cpp` — Operand collection + control flow + PC lifecycle

### 4.4 最终验证

- [ ] 4.4.1 全部 203 测试通过
- [ ] 4.4.2 对比 Phase 1+2 baseline 无回归
- [ ] 4.4.3 `wc -l src/ptxsim/core/thread_context.cpp` ≤ 300
- [ ] 4.4.4 `wc -l src/ptxsim/core/memory_accessor.cpp` ~250
- [ ] 4.4.5 `wc -l src/ptxsim/core/instruction_pipeline.cpp` ~350
- [ ] 4.4.6 `./scripts/sanity.sh` 完整回归（按 AGENTS.md TDD §Sanity 脚本用法）

### 4.5 Phase 3.3 提交

- [ ] 4.5.1 全部 Phase 3.3 commits（4.1.3、4.1.5、4.2.3、4.3.1）必须通过 4.4 验证后才合并
- [ ] 4.5.2 不强制 squash，保留独立 commit 以便独立 revert

---

## 5. 收尾工作

- [ ] 5.1 移除 baseline worktree：`git worktree remove .worktrees/baseline-pre-c3-phase1`
- [ ] 5.2 运行 `./scripts/sanity.sh` 完整回归（AGENTS.md §TDD）
- [ ] 5.3 OpenSpec archive：移动到 `archive/2026-XX-XX-god-class-refactor-thread-context-phase3/`
- [ ] 5.4 更新 `docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 标记 RESOLVED
- [ ] 5.5 沉淀新发现到 `docs/dev-process/lessons-learned.md`（如果 Phase 3 期间发现新模式）

---

## Revert 策略

每个 Phase/Sub-step 独立 revert（按反序）：
```bash
git revert <phase-3.3.b-commit>  # POD 删除
git revert <phase-3.3.a-commit>  # 停回填
git revert <phase-3.2.4-commit>  # 控制流迁移
git revert <phase-3.2.3-commit>  # Pipeline 类创建
git revert <phase-3.2.2-commit>  # BarWarpSyncHandler
git revert <phase-3.2.1-commit>  # PipelineHandler 基类
git revert <phase-3.2.0-commit>  # Accessor
git revert <phase-3.1-commit>    # MemoryAccessor
git revert <phase-3.0-commit>    # private 化
```

**纪律**: 任何已有测试回归 → 立即 revert 该 Phase/Sub-step，**不**混入后续 commit（lessons-learned §3）。
