## Why

Oracle 2026-07-11 审计发现 `tcgen05.commit` 和 `tcgen05.wait` 指令存在 3 个 HIGH-confidence 架构缺陷，**阻碍 FlashAttention 多阶段同步**：

> **NOTE**: Oracle session ID 在原始 Oracle 咨询中被引用，但本 artifact 不依赖特定 session ID 验证；设计决策（Option (b) parse tree walk、instr.cta_group 路由、ADR-0018 throw semantics）独立可验证。

1. **`Tcgen05Instr::cta_group` 便利字段从未被填充**（`include/ptx_ir/statement_context.h:186` 默认 `1`）。`makeTcgen05Instr`（`include/ptx_ir/statement_factory.h:265-292`）和 `visitTcgen05Inst`（`src/ptx_parser/ptx_visitor.cpp:841-885`）均**不**从 `.cta_group::N` 限定符中提取 IMMEDIATE 值。
2. **`extractQualifiersFromContext`**（`src/ptx_parser/ptx_visitor.cpp:155-183`）走 `TCGEN_CTA_GROUP COLONCOLON IMMEDIATE` 语法树时，遇到 `IMMEDIATE` terminal 调用 `tokenToQualifier` 返回 `Q_UNKNOWN` 后**静默丢弃**。该函数被 **21 个 call sites** 调用（1 definition + 20 callers，empirically verified by `grep -rn "extractQualifiersFromContext(" src/ptx_parser/`）：`ptx_visitor.cpp:858` (visitTcgen05Inst 自身调用), `_atom.cpp:81`, `_branch.cpp:30`, `_barrier.cpp:86,97`, `_call.cpp:23,44`, `_generic.cpp:14`, `_memory.cpp:16,29,42,55,68`, `_special.cpp:16,24,32,45,58`, `_warp.cpp:24,46`。改其返回类型破坏面过大。
3. **Handler 硬编码 `group_id=1` + `lane_id=0`**：`src/ptxsim/instructions/tcgen05.cpp:512` `cta->tc_queue().commit(1)` + `:550` `cta->tc_queue().wait(warp, 0, 1)`。`TcQueue` 基础架构本身（`src/ptxsim/async/tc_queue.h:53-55`）已支持多 group（`commit(group_id_t)` + `wait(warp, lane_id_t, group_id_t)`），**只有 handler 仍在用 hardcoded `1`**。

**FlashAttention 影响**：FA3 producer-consumer pipeline 需要 QK^T group + softmax group + PV group 区分 commit/wait，硬编码 `group_id=1` 无法表达多阶段同步。Sister change `fix-tcgen05-mma-accumulator-and-f32-storage`（Oracle H1+H2 fix，已于 commit `fd0fbb2` merge + `ea60934` 归档清理后存在于 `openspec/changes/archive/2026-07-11-fix-tcgen05-mma-accumulator-and-f32-storage/`）修复了 helper 的累加能力，但 B2 测试暴露了"即使累加路径修复，commit/wait 仍无法验证同步边界"。

## What Changes

- **新增** capability `tcgen05-multi-group-commit-wait`：支持 `tcgen05.commit/wait` 携带多个 group_id（每个 group 独立同步屏障），为 FlashAttention producer-consumer pipeline 提供基础。
- **修改** `src/ptx_parser/ptx_visitor.cpp::visitTcgen05Inst`：在 `extractQualifiersFromContext(ctx)` 调用后**新增单独 parse tree walk**，从 `tcgen05Qual` 上下文（ANTLR 生成的 `Tcgen05InstContext::tcgen05Qual()` vector accessor，per `build/antlr4_generated_src/ptxParser.h:3967`）中提取 `TCGEN_CTA_GROUP` 后的 `IMMEDIATE` 值，填充 `instr.cta_group`（这是 Option (b)，per design.md D1 — 改 `extractQualifiersFromContext` 返回类型会破坏 20 个 caller，blast radius 过大）。
- **修改** `include/ptx_ir/statement_factory.h::makeTcgen05Instr`：新增可选参数 `uint32_t cta_group = 1`（默认 1 保留所有现有调用点向后兼容）。
- **修改** `src/ptxsim/instructions/tcgen05.cpp::processTcgen05Commit`（line 493）：删除 `(void)instr;` 忽略，改调 `cta->tc_queue().commit(instr.cta_group)` 替代 `commit(1)`。
- **修改** `src/ptxsim/instructions/tcgen05.cpp::processTcgen05Wait`（line 530）：同上，改调 `cta->tc_queue().wait(warp, /*lane_id=*/0, instr.cta_group)` 替代 `wait(warp, 0, 1)`（lane_id 操作数解析属于 future FU-3.5 子任务，本 change 仅做 group_id）。
- **新增** 测试 `tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp`：验证 `commit(2)` + `wait(2)` 序列；验证 `cta_group::2` 解析路径（per [ADR-0018](../../../docs/adr/0018-tcgen05-cta-group-restriction.md) 已 throw 路径不变）。
- **NEW**: 测试 `tests/integration/ptx/test_tcgen05_mma_parse.cpp`：追加 `cta_group::2` 解析验证（`instr.cta_group == 2`）。注：本测试仅 factory-level（直接调用 `makeTcgen05Instr` 构造 instr），**不驱动 ANTLR parser**；per `test_tcgen05_mma_parse.cpp:7-9` 头部注释明确说明。ANTLR parser 路径验证由 `./tests/ptx/test_all_ptx.sh` 覆盖（per ptx-lessons-learned §9 + Checklist L）。

**BREAKING**: 无（`cta_group` 默认 `1` + 所有现有测试 PTX 默认隐含 `cta_group::1`，handler 现在读 `instr.cta_group` 但值仍为 1，行为不变）。

## Capabilities

### New Capabilities

- `tcgen05-multi-group-commit-wait`: 支持 `tcgen05.commit/wait` 携带多个独立同步 group，每个 group 维护单独的 commit-counter 与 pending-wait 列表，使 FlashAttention 的 QK^T、softmax、PV 多阶段 pipeline 能在 simulator 中表达。

### Modified Capabilities

- `tcgen05-handlers-extended`: `processTcgen05Commit` + `processTcgen05Wait` handler 从"硬编码 `group_id=1`"演进为"读 `instr.cta_group`"。spec 层级行为不变（默认仍为 1），但 visitor + IR 层的 `cta_group` 字段从"声明未用"变为"实际填充"。

## Impact

### 影响代码

| 文件 | 变更类型 | LoC 估计 | 风险 |
|------|----------|---------|------|
| `src/ptx_parser/ptx_visitor.cpp` | 修改（`visitTcgen05Inst` 加 IMMEDIATE walk） | +15 / 0 | 低 — 仅追加 parse tree walk，不改 extractQualifiersFromContext |
| `include/ptx_ir/statement_factory.h` | 修改（`makeTcgen05Instr` 加默认参数） | +5 / 0 | 极低 — 默认值保留向后兼容 |
| `src/ptxsim/instructions/tcgen05.cpp` | 修改（commit/wait handler 读 `instr.cta_group`） | +4 / -4 | 低 — `(void)instr;` 删除但 `instr` 已是参数 |
| `tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp` | 新增 | ~120 | — |
| `tests/integration/ptx/test_tcgen05_mma_parse.cpp` | 修改（追加 `cta_group::2` TC） | +15 / 0 | 极低 — 已有测试不变 |
| **总计** | | **~160 / -4** | |

### 不影响的依赖

- **TcQueue**（`src/ptxsim/async/tc_queue.h/cpp`）：已支持多 group，本 change 不动
- **Tcgen05PipelineHandler / 11 S_TCGEN05_* dispatch**：不变（仅 commit/wait 两个 handler 内部修改）
- **TmemAllocator**：无关
- **Grammar / Lexer / parser**：不修改（避免 lessons-learned §9 ANTLR bare token 风险）
- **Tests PTX fixtures**：不修改

### ADR / Spec 同步

- **ADR-0016** (`docs/adr/0016-blackwell-only-tcgen05.md`)：追加 "2026-07-12 Postmortem: C3 fix" 段
- **ADR-0018** (`docs/adr/0018-tcgen05-cta-group-restriction.md`)：**本 change 新建**（formalize `cta_group::2` throw 语义，之前隐式分散在 11 个 handler 的注释中）
- 根 `AGENTS.md`：不变（handler dispatch 表无变化）

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性（非适用 — 本 change 不迁移函数，而是填充已有但未用的便利字段）

- [x] **Baseline 函数清单**: 仅修改 `visitTcgen05Inst` + `makeTcgen05Instr` + commit/wait handler 的内部引用 — 无新函数迁移
- [x] **逐行 diff 计划**: 见 `design.md` §Migration Plan
- [x] **跨模块状态翻译表**: 不适用（无 `set_state(BAR_SYNC)` 类副作用）

### 多 Phase 推进

- [x] **Phase 拆分**: 1 atomic commit（scope 小 — 3 文件 + 2 测试，不需分 Phase）
- [x] **基线 worktree 计划** (per `ptx-lessons-learned` §4):
  ```bash
  # Step 1: 建立 baseline（commit `d3be589` 包含 sister change 的 persistence test）
  git worktree add .worktrees/baseline-c3 $(git rev-parse HEAD)
  cd .worktrees/baseline-c3
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  cd build && ctest -L tcgen05 --output-on-failure
  ```
- [x] **失败处理策略**: 任何已有测试回归 → 立即 `git revert HEAD`，不混入后续 commit（lessons-learned §3）

### 文档同步

- [x] AGENTS.md: 不需修改（handler dispatch 表不变）
- [x] ADR 追加段: ADR-0016 §2026-07-12 Postmortem 段
- [x] OpenSpec tasks.md 状态变更: Phase 完成后 tasks.md 标记 `[x]`

### Pre-implementation Review（per lessons-learned §7/Checklist H）

- [x] **Oracle 决策建议**: ✅ 2026-07-11（Q1+Q2+Q3+Q4+Q5+Q6 全部验证，CONDITIONAL GO；Oracle session ID 在本 artifact 中省略以避免反幻觉 — 设计决策独立可验证）
- [x] **Metis pre-impl review**: 必跑（在 `tasks.md` §0.5）
- [x] **关键假设实证验证**:
  - `grep -rn "tc_queue().commit(\|tc_queue().wait(" src/ptxsim/instructions/tcgen05.cpp` 验证当前硬编码 ✓
  - `wc -l src/ptx_parser/ptx_visitor.cpp` 确认文件存在 ✓
  - `grep "cta_group" include/ptx_ir/statement_context.h` 验证字段存在 ✓
- [x] **区分 active debt vs stale debt**: 本 change 是 active debt（C3 是 Oracle H1+H2 之后的下一步前置）

### OpenSpec artifacts 内部一致性（Checklist J）

- [x] 范围数字对齐：本 change = 3 handler/visitor/factory 文件 + 2 测试文件（proposal/design/tasks 三个 artifact 保持一致）
- [x] Decision 路径与 Scenario 路径一致：`design.md` D1 写"在 visitTcgen05Inst 加 parse tree walk" → `specs/.../spec.md` Scenario 写"verifying `instr.cta_group == 2`"` — 一致
- [x] tasks 验证命令与 design 路径示例一致
- [x] 验证归档未变任务在 `tasks.md` §3.6

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| `archive/2026-07-10-implement-tcgen05-handlers-extended/` | **fix-tcgen05-commit-wait-group** | `fix-tcgen05-idesc-parsing` (FU-2) |
| `archive/2026-07-11-fix-tcgen05-mma-accumulator-and-f32-storage/` (sister, Oracle H1+H2, 已 merge + 归档) | | `fix-tcgen05-ld-st-slot-routing` (FU-3) |
| | | `fix-tcgen05-multi-warp-fragment` (FU-4) |
| | | `tcgen05-flashattention-coverage` (FU-5) |

- **上游**: 已归档的 `implement-tcgen05-handlers-extended` 提供 commit/wait handler 骨架；已归档的 sister change `fix-tcgen05-mma-accumulator-and-f32-storage` 提供 helper 累加能力（commits `df1f6de` + `f97863c` + `58cbff9` + `3d8c4e2` 已 merge + 归档）
- **下游**: C3 完成后，FU-2 (C1) / FU-3 (C2) / FU-4 (C4) 三个 follow-up 可并行开工，FU-5 (FlashAttention coverage + E2E) 在 FU-1..4 全部完成后
- **本 change 是 FlashAttention-readiness 的第二个前置**（继 sister change H1+H2 merge + 归档之后）

## Ref 链接

- Sister change (archived 2026-07-11): [`../../archive/2026-07-11-fix-tcgen05-mma-accumulator-and-f32-storage/`](../../archive/2026-07-11-fix-tcgen05-mma-accumulator-and-f32-storage/)
- Ref (archived 2026-07-10): [`../../archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/)
- ADR-0016: [docs/adr/0016-blackwell-only-tcgen05.md](../../../docs/adr/0016-blackwell-only-tcgen05.md)
- ADR-0018 (本 change 新建): [docs/adr/0018-tcgen05-cta-group-restriction.md](../../../docs/adr/0018-tcgen05-cta-group-restriction.md)
- ptx-lessons-learned: [.opencode/skills/ptx-lessons-learned/SKILL.md](../../../.opencode/skills/ptx-lessons-learned/SKILL.md)
- PTX ISA §9.7.16 (tcgen05.commit/wait semantics)
