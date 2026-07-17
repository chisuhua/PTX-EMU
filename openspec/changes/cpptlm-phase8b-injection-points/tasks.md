# Tasks: CppTLM Phase 8.B D1-Full 注入点

> **Status**: Proposed → **Phase 0 部分完成** (PTX-0.1/0.2/0.4 锁定于 2026-07-16, 来源 CppTLM commit `2b28505`)
> **Parent**: `proposal.md` + `design.md` (cpptlm-phase8b-injection-points)
> **ADR**: [docs/adr/0020-cpptlm-injection-points.md](../../../docs/adr/0020-cpptlm-injection-points.md)
> **总工时**: ~2.5d

---

## 📚 Phase 0: 对齐 + 基线（强制最先完成，~0.5d）

> ⚠️ **MUST**: 不完成本 Phase 不允许进入 Phase 1。Lessons Learned #7 Pre-implementation Review 强制项。
>
> **2026-07-16 更新**: PTX-0.1 / PTX-0.2 / PTX-0.4 已通过 CppTLM commit `2b28505` (RFC-P1-001~004) 完成对齐（见各子任务内 ✅ 来源标注）。
> PTX-0.3 (PTX-EMU 内部序列化协调) + PTX-0.5 (基线 worktree) 待 PTX-EMU 团队在 Phase 1 实施前完成。

### PTX-0.1: 枚举值对齐确认（30 min）

**验证内容**：
- [x] `PipelineId` 0-5 值与 CppTLM `tlm::PipelineId` 一致（双方 static_assert 验证）
  - **来源锁定**: CppTLM commit `2b28505` (RFC-P1-003 §3.1) — 双端字字对应
  - **CPPTLM 路径**: `CppTLM/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md` §3.1
- [x] `TcPrecision` 0-5 值与 CppTLM `tlm::TcPrecision` 一致
  - **来源锁定**: CppTLM commit `2b28505` (RFC-P1-003 §3.2) — 双端字字对应
  - **CPPTLM 路径**: `CppTLM/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md` §3.2
- [x] `StatementType` 枚举值稳定（X-Macro `include/ptx_ir/ptx_op.def` 不再变动）
  - **来源锁定**: CppTLM 端已确认（cpptlm-d1-p1-pipeline-scoreboard/internal-plan.md §3 PTX-EMU Sync Points）

**协作方式**: ✅ 已通过 CppTLM commit `2b28505` (2026-07-16) 完成对齐（替代原 §13.2 邮件方式）

**Commit**: ✅ 本 commit (2026-07-16 Phase 0 alignment)

### PTX-0.2: 新增 API 清单确认（30 min）

**验证内容**：
- [x] `WarpContext::set_blocked_cycles_for_active()` 签名（参数 + 返回类型）双方确认
  - **本侧锁定**: 本 change `design.md §4` 已锁定 `void set_blocked_cycles_for_active(uint32_t cycles)`
  - **CppTLM 答复 Q1**: 该方法不在 `IScoreboard` 接口中（属 WarpContext 子类行为），见 CppTLM commit `2b28505` (RFC-P1-004 Q1)
- [x] `StatementContext` 目标寄存器提取 API 签名（返回 `vector<uint32_t>`）双方确认
  - **本侧锁定**: 本 change `design.md §4` 已锁定 `std::vector<uint32_t> get_dest_registers_as_ids(const StatementContext&)`
- [x] Scoreboard stall 是否消耗 cycle（影响 `exe_once()` 行为）双方确认
  - **本侧锁定**: 本 change `spec.md §Step A Scenario` 已明确："`cycle_counter_` 不推进（不消耗功能性 cycle）"

**协作方式**: ✅ 已通过 CppTLM commit `2b28505` + 本 change design.md/spec.md 锁定

**Commit**: ✅ 本 commit (2026-07-16 Phase 0 alignment)

### PTX-0.3: 序列化协调（30 min）

**验证内容**：
- [ ] `cleanup-deprecated-barrier-apis` 归档状态确认 (待 PTX-EMU 团队验证)
- [ ] 与 `god-class-refactor-thread-context-phase3` 字段迁移路径协调 (PTX-EMU 内部协调)
- [ ] 与 `migrate-bar-warp-sync-to-barrier-module` barrier 交互测试协调 (PTX-EMU 内部协调)

**协作方式**: 🔵 PTX-EMU 内部协调（无需 CppTLM 参与）

**Commit**: 待实施时

### PTX-0.4: CppTLM 书面同步确认（30 min）

**验证内容**：
- [x] CppTLM 协作同步文档 `2026-07-01-f12b-ld-ptxemu-collaboration-sync.md §13` 包含 PTX-EMU 接收信号
  - **来源**: CppTLM 端已在 `2026-07-15-cpptlm-hsk-response.md` 确认 HSK-1/2/3
- [x] CppTLM 实施计划 `2026-06-24-gpu-soc-phase8b.md` 引用本 change ID
  - **来源**: CppTLM `cpptlm-d1-p1-pipeline-scoreboard/internal-plan.md §3` PTX-EMU Sync Points 引用 `cpptlm-phase8b-injection-points`
- [x] CppTLM 端 RFC-P1-001~004 已发送（commit `2b28505`）
  - **CPPTLM 路径**: `CppTLM/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md`
- [x] CppTLM P2 AsyncCompletion 占位已交付（commit `e69cd1d`）
  - **CPPTLM 路径**: `CppTLM/include/tlm/gpu/async_completion_adapter.hh` + 5 个 `[gpu][async]` 单测

**协作方式**: ✅ 已通过 CppTLM commit `2b28505` (2026-07-16) 完成对齐

**Commit**: ✅ 本 commit (2026-07-16 Phase 0 alignment)

### PTX-0.5: 基线 worktree 建立（10 min）

> ⚠️ **MUST**: 遵循 Lessons Learned #4 基线 worktree 强制项

**操作**：
```bash
cd /workspace/project/PTX-EMU
git worktree add ../ptxemu-baseline-2026-07-XX main
cd ../ptxemu-baseline-2026-07-XX
. env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure  # 验证全量测试基线
```

**Commit**: 无

---

## Phase 1: 3 个纯虚接口头文件（独立可测，~0.5d）

> 📌 **NOTE**: 本 Phase 完全独立，不影响任何现有 PTX-EMU 代码。每个头文件独立 commit。

### PTX-1: IScoreboard 纯虚接口（30 min）

**操作**：
- [x] 创建 `include/ptxsim/scoreboard_interface.h`
- [x] 包含 `<cstdint>` + `IScoreboard` 类（4 方法）
- [x] 编译验证：`cmake --build build --target ptxsim`

**约束**：
- ⚠️ MUST 仅 include `<cstdint>`（零外部依赖）
- ⚠️ MUST 不使用 `namespace`（与现有 PTX-EMU 风格一致）

**Commit**:
```bash
git add include/ptxsim/scoreboard_interface.h
git commit -m "feat(ptxsim): IScoreboard interface for CppTLM injection (Phase 8.B PTX-1)

Pure virtual interface for Scoreboard injection into SMContext.
Zero external dependencies (only <cstdint>).

Refs:
- ADR-0020 (cpptlm-injection-points)
- CppTLM docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md §3.1
- openspec/changes/cpptlm-phase8b-injection-points"
```

**验证**: `cmake --build build --target ptxsim` PASS

### PTX-2: IPipelineLatencyProvider + PipelineId（30 min）

**操作**：
- [x] 创建 `include/ptxsim/pipeline_interface.h`
- [x] 包含 `<cstdint>` + `<string>` + `PipelineId` enum + `IPipelineLatencyProvider` 类
- [x] 编译验证

**约束**：
- ⚠️ MUST `PipelineId` 值 0-5 与 CppTLM `tlm::PipelineId` 一致
- ⚠️ MUST `get_fractional_cycles_by_type` 参数为 `int` 而非 `StatementType`（避免接口文件依赖 ptx_ir/ptx_types.h）

**Commit**:
```bash
git add include/ptxsim/pipeline_interface.h
git commit -m "feat(ptxsim): IPipelineLatencyProvider + PipelineId enum (Phase 8.B PTX-2)

Pure virtual interface for fractional cycle latency injection.
PipelineId enum (0-5) must match CppTLM tlm::PipelineId (Adapter static_assert).

Refs: ADR-0020, CppTLM docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md §3.2"
```

**验证**: `cmake --build build --target ptxsim` PASS

### PTX-3: ITensorCoreTiming + TcPrecision（30 min）

**操作**：
- [x] 创建 `include/ptxsim/tensor_core_interface.h`
- [x] 包含 `<cstdint>` + `TcPrecision` enum + `ITensorCoreTiming` 类（3 方法）
- [x] `get_latency_mnk` 提供默认实现（退化到 `get_latency`）

**约束**：
- ⚠️ MUST `TcPrecision` 值 0-5 与 CppTLM `tlm::TcPrecision` 一致

**Commit**:
```bash
git add include/ptxsim/tensor_core_interface.h
git commit -m "feat(ptxsim): ITensorCoreTiming + TcPrecision enum (Phase 8.B PTX-3)

Pure virtual interface for TensorCore timing injection.
TcPrecision enum (0-5) must match CppTLM tlm::TcPrecision (Adapter static_assert).

Refs: ADR-0020, CppTLM docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md §3.3"
```

**验证**: `cmake --build build --target ptxsim` PASS

---

## Phase 2: SMContext 接口扩展（~0.2d）

### PTX-4: SMContext 6 个 public 方法 + 3 私有成员（1 hour）

**操作**：
- [x] 修改 `include/ptxsim/sm_context.h`
- [x] +3 `#include`（3 个接口头文件）
- [x] +3 setter：`set_scoreboard` / `set_pipeline_latency_provider` / `set_tensor_core_timing`
- [x] +3 getter
- [x] +3 私有成员（裸指针，默认 nullptr）
- [x] **不修改构造函数**

**约束**：
- ⚠️ MUST 使用裸指针（非 `unique_ptr`）：所有权归外部 libcpptlm_cudart.so
- ⚠️ MUST 不修改 `set_warp_scheduler` 现有接口
- ⚠️ MUST 编译通过 + 现有测试 0 回归

**Commit**:
```bash
git add include/ptxsim/sm_context.h
git commit -m "feat(ptxsim): SMContext 3 setters + 3 getters for CppTLM injection (Phase 8.B PTX-4)

3 new public setters (raw pointer, ownership external) + 3 getters + 3 private members.
nullptr default = backward compatible (no behavior change when not injected).

Refs: ADR-0020, openspec/changes/cpptlm-phase8b-injection-points"
```

**验证**:
- `cmake --build build --target ptxsim` PASS
- `ctest -L "unit;smcontext"` PASS（无回归）
- `ctest -L "integration;smcontext"` PASS

---

## Phase 3: WarpContext 扩展 + RegisterAnalyzer 增强（~0.5d）

### PTX-5a: WarpContext::set_blocked_cycles_for_active（30 min）

**操作**：
- [x] 修改 `include/ptxsim/warp_context.h` + `src/ptxsim/core/warp_context.cpp`
- [x] 新增 public 方法 `set_blocked_cycles_for_active(uint32_t cycles)`
- [x] 内部遍历 `warp_state_.threads`，对 `is_active && !is_blocked` 的线程设置 `blocked_cycles_remaining = cycles; is_blocked = true`
- [x] 调用 `update_active_mask()` 同步 active_mask[] / active_count (T2-1 contract)

**约束**：
- ⚠️ MUST 不修改 `ThreadState` 结构体布局
- ⚠️ MUST 与现有 `decrement_blocked_cycles()` 协同工作
- ⚠️ MUST 现有 LD-only 路径 0 回归

**Commit**:
```bash
git add include/ptxsim/warp_context.h src/ptxsim/core/warp_context.cpp
git commit -m "feat(ptxsim): WarpContext::set_blocked_cycles_for_active (Phase 8.B PTX-5a)

Per-warp blocked_cycles extension. Replaces LD-only per-thread path.
Internal loop iterates warp_state_.threads and sets blocked_cycles_remaining
on active non-blocked threads.

Refs: ADR-0020, CppTLM docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md §A.4"
```

**验证**:
- `cmake --build build --target ptxsim` PASS
- `ctest -L "unit;memory"` PASS（LD 路径 0 回归）
- `ctest -L "unit;barrier"` PASS

### PTX-5b: RegisterAnalyzer::get_dest_registers_as_ids（1.5 hour）

**操作**：
- [x] 修改 `include/ptxsim/register_analyzer.h` + `src/ptxsim/register_analyzer.cpp`
- [x] 新增 public static 方法 `get_dest_registers_as_ids(const StatementContext&) -> vector<uint32_t>`
- [x] 通过 `stmt.visit()` 处理 `StatementContext.data` variant（与 `extract_registers_from_all_operands` 一致）
- [x] PoC 单元测试验证 7 种关键指令：`add.f32` / `ld.global.f32` / `st.global.f32` / `setp.eq.f32` / `atom.global.add.u32` / `bra` / `bar.sync`

**约束**：
- ⚠️ MUST **不修改**现有 `analyze_registers()`（避免破坏现有用户）
- ⚠️ MUST 使用 `OperandContext::kind()` 返回 `OperandKind`，从 `std::get<RegOperand>(dst.data).index` 取 reg ID（helper 不存在 — 严格使用 API 而非 design.md 旧版伪代码中的 `get_kind()/get_reg_id()`）
- ⚠️ MUST 沿用 `stmt.visit()` 而非 `std::visit(stmt.data)`（statement_context.h:329-335 已有封装，等价但与现有 register_analyzer.cpp:58 风格一致）
- ⚠️ MUST 不处理 `instr.dest` 字段（25 个 variant 无同时有 operands + 独立 dest 字段）

**Commit**:
```bash
git add include/ptxsim/register_analyzer.h src/ptxsim/register_analyzer.cpp
git commit -m "feat(ptxsim): RegisterAnalyzer::get_dest_registers_as_ids (Phase 8.B PTX-5b)

Distinguishes src/dst registers (existing analyze_registers returns all operands).
Implementation uses std::visit on StatementContext.data variant.
PoC test: 'add.f32 %f1, %f2, %f3' -> [%f1].

Refs: ADR-0020, CppTLM docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md §A.4"
```

**验证**:
- `cmake --build build --target ptxsim` PASS
- PoC 单元测试 PASS
- 现有 `analyze_registers()` 用户 0 回归

---

## Phase 4: exe_once() 注入（~1.0d）

### PTX-6: 三段式注入 + 4 辅助函数（4 hours）

**操作**：
- [x] 修改 `src/ptxsim/core/sm_context.cpp`
- [x] 新增 3 辅助函数 (匿名namespace, file-local)：`step_a_scoreboard_check` / `step_b_set_blocked_cycles` / `step_c_release_scoreboard` + 3 public static helpers (`map_instruction_to_pipeline` / `is_tensor_core_instruction` / `map_instruction_to_tc_precision`)
- [x] `exe_once()` 插入 3 处注入点（Step A / B / C），使用 `warp_executed` 守卫和 `goto warp_done` 控制流（fast + slow 两条路径）
- [x] nullptr 完全回退（4 个注入点全 nullptr 时行为与改造前**字节级相同**）

**约束**：
- ⚠️ MUST 严格遵循设计文档 `design.md §7` 的改造代码
- ⚠️ MUST 使用 `goto warp_done` 实现 Step A 失败跳过（目标在 `set_scheduled(false)` **之前**）
- ⚠️ MUST Step B 和 Step C **仅在 warp_executed 路径执行**（Step A 失败时跳过 — Oracle 2026-07-17 BUG-2/BUG-3 验证）
- ⚠️ MUST Step C 用 `warp_executed` 守卫（防止释放未分配 regs 导致 scoreboard 状态损坏）
- ⚠️ MUST Step B 用 `ptxsim::getLatency(stmt->type).cycles` 作为 fallback（向后兼容 free function）
- ⚠️ MUST `is_tensor_core_instruction()` 用 `stmt.type >= S_TCGEN05_ALLOC && stmt.type <= S_TCGEN05_FENCE` 范围比较（X-Macro 11 entries 连续，ptx_op.def:127-137）
- ⚠️ MUST 延迟取 `ceil(double) → uint32_t`
- ⚠️ MUST Pipeline 优先级高于 TensorCore 高于 InstructionLatencyTable

**Commit**:
```bash
git add src/ptxsim/core/sm_context.cpp
git commit -m "feat(ptxsim): exe_once 3-step injection for CppTLM (Phase 8.B PTX-6)

3 injection points in SMContext::exe_once():
  Step A: Scoreboard check (nullptr = skip)
  Step B: Pipeline latency query (priority: pipeline > tensor_core > InstructionLatencyTable)
  Step C: Scoreboard release (nullptr = skip)

nullptr fallback: 4 injection points all nullptr = byte-identical to pre-change behavior.

Refs: ADR-0020, openspec/changes/cpptlm-phase8b-injection-points, CppTLM task doc §3"
```

**验证**:
- `cmake --build build --target ptxsim` PASS
- `ctest -L "unit"` 全 PASS（无回归）
- `ctest -L "integration"` 全 PASS
- `ctest -L "e2e"` 全 PASS
- 任何测试回归 → **立即 revert**（Lessons Learned #3）

---

## Phase 5: 测试（~0.7d）

### PTX-7a: 7 个 Mock 单元测试（3 hours）

**操作**：
- [ ] 创建 `tests/unit/cpptlm/test_smcontext_injection.cpp`
- [ ] 7 个测试用例（任务书 §5.2 完整移植）：
  1. `SMContext: nullptr injection = no-op (backward compat)`
  2. `SMContext: scoreboard limits concurrent operations`
  3. `SMContext: scoreboard release after instruction completes`
  4. `SMContext: pipeline overrides InstructionLatencyTable`
  5. `SMContext: tensor_core overrides default TC latency`
  6. `SMContext: tensor_core falls back when pipeline returns 0`
  7. `SMContext: all three injection points active simultaneously`
- [ ] 测试标签：`[unit;cpptlm;injection]` / `[unit;cpptlm;scoreboard]` / `[unit;cpptlm;pipeline]` / `[unit;cpptlm;tensor_core]`
- [ ] ctest 名称：`unit_smcontext_injection`

**约束**：
- ⚠️ MUST 严格遵循 `template.md` 项目测试规范（`unit_/integration_/e2e_` 前缀）
- ⚠️ MUST 7 个测试覆盖 nullptr fallback + 4 注入点单独 + 组合

**Commit**:
```bash
git add tests/unit/cpptlm/test_smcontext_injection.cpp tests/CMakeLists.txt
git commit -m "test(ptxsim): SMContext injection mock tests (Phase 8.B PTX-7a)

7 Mock test cases covering nullptr fallback + 3 injection points + combinations.
Tests:
- MockScoreboardLimited (12 entries)
- MockScoreboardRAW (RAW hazard)
- MockPipelineFixed (4.22 cyc)
- MockTensorCoreFixed (29 cyc)

Refs: ADR-0020, CppTLM docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md §5.2"
```

**验证**:
- `ctest -R "unit_smcontext_injection" --output-on-failure` 7/7 PASS
- `ctest -L "unit"` 全 PASS

### PTX-7b: 集成测试（2 hours）

**操作**：
- [ ] 创建 `tests/integration/cpptlm/test_scoreboard_allocation.cpp`
- [ ] 4 个测试用例：
  1. `Scoreboard: detect RAW hazard across warp instructions` (RAW hazard)
  2. `Scoreboard: allocate/release cycle through execute_warp_instruction` (full cycle)
  3. `Pipeline injection: FFMA latency override` (4.22 cyc → ceil 5)
  4. `Blocked cycles: LD-no-longer-only` (扩展至非 LD 指令)
- [ ] 测试标签：`[integration;cpptlm;scoreboard]` / `[integration;cpptlm;pipeline]` / `[integration;cpptlm;blocked_cycles]`
- [ ] ctest 名称：`integration_scoreboard_allocation`

**Commit**:
```bash
git add tests/integration/cpptlm/test_scoreboard_allocation.cpp tests/CMakeLists.txt
git commit -m "test(ptxsim): scoreboard allocation integration test (Phase 8.B PTX-7b)

4 integration tests with real warp + Mock scoreboard:
- RAW hazard detection across warp instructions
- allocate/release cycle through execute_warp_instruction
- Pipeline injection: FFMA latency override
- Blocked cycles: extend LD-only path to all instruction types

Refs: ADR-0020, openspec/changes/cpptlm-phase8b-injection-points"
```

**验证**:
- `ctest -R "integration_scoreboard_allocation" --output-on-failure` 4/4 PASS

### PTX-7c: 回归测试（1 hour）

**操作**：
- [ ] 完整回归基线：
  - `ctest -L "unit"` 全 PASS
  - `ctest -L "integration"` 全 PASS
  - `ctest -L "e2e"` 全 PASS
- [ ] 与 baseline worktree 对比：
  ```bash
  cd ../ptxemu-baseline-2026-07-XX/build && ctest > baseline_ctest.txt
  cd /workspace/project/PTX-EMU/build && ctest > current_ctest.txt
  diff baseline_ctest.txt current_ctest.txt  # 仅有 7+4 个新测试通过差异
  ```

**约束**：
- ⚠️ MUST 现有 600+ 测试用例 0 回归（除新增 11 个测试通过差异外）
- ⚠️ MUST 任何测试回归 → **立即 revert** 对应 Phase（Lessons Learned #3）

**验证**:
- `ctest` 全 PASS
- `diff` 仅显示 11 个新测试通过差异

---

## 验收 Gates

- [ ] **G1** `ctest -L "unit;cpptlm"` 7 个 Mock 测试全 PASS
- [ ] **G2** `ctest -L "integration;cpptlm"` 4 个集成测试全 PASS
- [ ] **G3** 现有 `[unit;memory]` `[unit;barrier]` `[integration;simt]` 测试基线 0 回归
- [ ] **G4** nullptr 注入时行为与改造前**字节级相同**（通过 baseline worktree 对照）
- [ ] **G5** 4 个新接口头文件零依赖（`grep -r '#include' include/ptxsim/{scoreboard,pipeline,tensor_core}_interface.h` 仅 `<cstdint>`/`<string>`）
- [ ] **G6** `clang-format -i` 对所有修改文件运行
- [ ] **G7** `./scripts/sanity.sh` 全绿（含 11 个新测试 + 现有 600+ 测试）
- [ ] **G8** Oracle 审查通过（调用 oracle subagent 验证接口设计 + exe_once 改造）

## 实施后节点

- [ ] **ADR-0020 状态更新**: Proposed → Accepted（所有 Phase 完成 + Oracle 审查通过后）→ Active（归档 OpenSpec change 后）
- [ ] **OpenSpec 归档**: `git mv openspec/changes/cpptlm-phase8b-injection-points openspec/changes/archive/2026-07-14-cpptlm-phase8b-injection-points`
- [ ] **CppTLM 同步**: 通知 CppTLM 团队 PTX-1~4 接口已就绪，可启动 Task 15（Adapter 层）
- [ ] **AGENTS.md / lessons-learned.md 沉淀**: 实施过程中如发现新 bug 模式，按经验沉淀元规则记录

## 依赖

- **前置已满足**: `cleanup-deprecated-barrier-apis` 已归档（2026-06-20 per `openspec/changes/archive/2026-06-20-cleanup-deprecated-barrier-apis/`）✅
- **前置已满足**: `migrate-bar-warp-sync-to-barrier-module` 已归档（2026-07-03 per `openspec/changes/archive/2026-07-03-migrate-bar-warp-sync-to-barrier-module/`）✅
- **必须先完成**: Phase 0 对齐（PTX-0.1~0.5）
- **下游依赖**: CppTLM Task 15（Adapter 层）依赖本 change 归档
- **并行实施**: 仅与 `god-class-refactor-thread-context-phase3` 并行（需关注 `blocked_cycles` 字段迁移）

## 序列化考虑

| Active Change | 关系 | 建议 |
|--------------|------|------|
| `cleanup-deprecated-barrier-apis` | **已归档**（2026-06-20） | 前置条件已满足 ✅ |
| `migrate-bar-warp-sync-to-barrier-module` | **已归档**（2026-07-03） | 并行协调已解除 ✅ |
| `god-class-refactor-thread-context-phase3` | **并行** | 关注 `blocked_cycles` 字段迁移 |
