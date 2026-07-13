# Fix tcgen05.ld/.st/.cp Hardcoded TMEM Slot Routing

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) — Blackwell-only tcgen05
> **前置 change**: [`../fix-tcgen05-mma-accumulator-and-f32-storage/`](../fix-tcgen05-mma-accumulator-and-f32-storage/) (helper 累加器 + f32 storage 已完成；当前 change 解决 BLOCKER C2)
> **Oracle 2026-07-11 BLOCKER 审计**: session `ses_0b3791d78ffewb52428kJJ2Irz` 第 C2 条（HIGH confidence）— ld/st/cp slot 硬编码 `0`，与 mma 写 C 到 `slot[64..95]` 矛盾
> **强制 lessons-learned**: [`ptx-lessons-learned`](../../../.opencode/skills/ptx-lessons-learned/SKILL.md) §3 (Phase commit) + §4 (baseline worktree) + §6 (artifacts-first) + §7 (Pre-impl Review) + §9 (ANTLR bare token 反模式)
> **范围**: 2 atomic commits（Phase 1: grammar+IR+visitor+factory；Phase 2: handlers + tests）
> **关联 follow-up**: `fix-tcgen05-commit-wait-group` (FU-1/C3) 必须**先于**本 change 完成（建立 IMMEDIATE 提取 pattern）

## Why

Oracle 2026-07-11 审计 `tcgen05.ld` / `tcgen05.st` / `tcgen05.cp` 三个 handler 均硬编码 TMEM slot `0`：

| 文件:行 | 现状 | 问题 |
|---------|------|------|
| `src/ptxsim/instructions/tcgen05.cpp:434` (ld) | `tmem.write(0, tmp, Tmem::kSlotSize)` | FlashAttention 中 `ld` 加载 K/V 到 mma 实际消费的 slot，但 mma 写 C 到 `slot[64..95]`（`tcgen05_helpers.cpp:23`） |
| `src/ptxsim/instructions/tcgen05.cpp:476` (st) | `tmem.read(0, tmp, Tmem::kSlotSize)` | 与 mma 输出不连通，softmax→PV 路径完全断裂 |
| `src/ptxsim/instructions/tcgen05_cp.cpp:138` | `kDestSlot = 0` (源码已有 TODO 注释 "resolve from operand and shape qualifier") | 同上 |

`Tmem::read/write` (`src/ptxsim/memory/tmem.h:35-36`) 本身已接受任意 `slot_id`，瓶颈是 handler 内部硬编码常量。`FlashAttention` 的 QK^T→softmax→PV 数据流需要 ld 把 K tile 写到 mma 实际消费的 slot，st 把 mma 输出移到 softmax 暂存区，**当前架构下不可能**。

## What Changes

### 修改

| 文件 | 范围 | Phase |
|------|------|-------|
| `src/grammar/ptxInstructions.g4:432-433` | ld/st/cp 子句添加 tmem_slot 操作数（位置：源地址之前） | 1 |
| `src/grammar/ptxParser.g4` (同文件其他相关规则) | 若需要补充 `tcgen05Operand` 规则覆盖 tmem_slot | 1 |
| `include/ptx_ir/ptx_op.def:130-132` | ld/st/cp `op_count` 从 2/2/3 → 3/3/4 | 1 |
| `include/ptx_ir/statement_context.h:180-190` | `Tcgen05Instr` 加 `uint32_t tmem_slot = 0` 字段 | 1 |
| `src/ptx_parser/ptx_visitor.cpp:841-885` | `visitTcgen05Inst` 提取 slot 操作数（per FU-1 模式，参考 C3 follow-up） | 1 |
| `include/ptx_ir/statement_factory.h:265-292` | `makeTcgen05Instr` 加可选 `tmem_slot` 参数 | 1 |
| `src/ptxsim/instructions/tcgen05.cpp:434` (ld) | `tmem.write(instr.tmem_slot, ...)` 替代硬编码 `0` | 2 |
| `src/ptxsim/instructions/ttcgen05.cpp:476` (st) | `tmem.read(instr.tmem_slot, ...)` 替代硬编码 `0` | 2 |
| `src/ptxsim/instructions/tcgen05_cp.cpp:138` | 删除 `kDestSlot = 0` 常量，使用 `instr.tmem_slot` | 2 |
| `tests/integration/ptx/test_tcgen05_ld_parse.cpp` | 新增 tmem_slot 操作数解析验证 | 1 |
| `tests/integration/ptx/test_tcgen05_st_parse.cpp` | 同上 | 1 |
| `tests/integration/ptx/test_tcgen05_cp_parse.cpp` | 同上（若存在） | 1 |
| `tests/integration/tcgen05/test_tcgen05_*.cpp`（现有） | 全量跑回归，确保 op_count 变化不破现有测试 | 1 + 2 |
| `tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp` (新文件) | 行为测试：ld 写 `tmem_slot=32` 后 mma 读 slot 32 等于 ld 输入 | 2 |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 "2026-07-12 Postmortem: C2 fix" 段 | 3 (archive) |

### 不修改（Non-Goals）

- ❌ 不修改 mma handler（FU-2 `fix-tcgen05-idesc-parsing` 独立 change）
- ❌ 不修复 commit/wait group_id 硬编码（FU-1 `fix-tcgen05-commit-wait-group` 独立 change 且**前置**）
- ❌ 不实现 multi-warp slot 偏移（FU-4 `fix-tcgen05-multi-warp-fragment` 独立 change）
- ❌ 不修改 mma C output slot (`tcgen05_helpers.cpp:23` `c_slot = 64 + lane_id`，由 FU-4 处理)
- ❌ 不引入新的 `TcQueue` / `TmemAllocator` API

### Anti-Grammar-Pattern 强制 (per ptx-lessons-learned §9)

**关键约束**：实现 Phase 1 的 grammar 修改前，必须验证：
- ❌ 禁止新增 bare string lexer tokens（如 `TMEM_SLOT : 'tmem_slot'`）— 与 `ID` 规则冲突
- ✅ 推荐用 `.tmem_slot::N` 作为 qualifier（`.` 前缀避开 ID 冲突）或数字立即数作为操作数

## Non-Goals

### 显式拒绝

- ❌ 不修改 grammar `tcgen05.qual::x` 之类（已 archive `tcgen05-grammar` capability，per lessons-learned §G）
- ❌ 不实现 cluster-wide（cta_group::2）tcgen05（永久抛异常 per ADR-0018）
- ❌ 不引入 TmemAllocator API 扩展（slot 由指令显式提供）
- ❌ 不修改 helper `tcgen05_fragment_mma_f16`（由 FU-2 + FU-4 处理）

### 范围限制

- 仅 ld/st/cp 三个 handler（5 个 S_TCGEN05_* 中的 3 个；alloc/dealloc/relinquish/fence/commit/wait 不变）
- 仅单 warp 假设（per `tcgen05_helpers.h:43-46`）— 多 warp 由 FU-4 处理
- 仅修改已有指令操作数，不引入新的 `S_TCGEN05_*` enum
- 不实现 cta_group::2（per ADR-0018）

## Goals

### Phase 1: Grammar + IR + Visitor + Factory（commit 1）

1. 在 `tcgen05Operands` 规则加 tmem_slot 操作数（数字立即数或新 qualifier — 待 Oracle/实测决定）
2. `Tcgen05Instr` 加 `tmem_slot` 字段（默认 `0` 保持向后兼容）
3. `visitTcgen05Inst` 提取 slot 到 `instr.tmem_slot`
4. `makeTcgen05Instr` 加可选 `tmem_slot` 参数
5. parser 测试验证 tmem_slot 操作数解析
6. 跑 `./tests/ptx/test_all_ptx.sh` 验证 grammar 修改（per lessons-learned §L + §9）
7. 跑 `cd build && ctest -R "tcgen05" --output-on-failure` 验证无回归
8. **commit**: `fix(tcgen05): parse tmem_slot operand for ld/st/cp (Oracle C2 Phase 1)`

### Phase 2: Handler 路由 + 行为测试（commit 2）

1. `tcgen05.cpp:434` 改 `tmem.write(instr.tmem_slot, ...)`
2. `tcgen05.cpp:476` 改 `tmem.read(instr.tmem_slot, ...)`
3. `tcgen05_cp.cpp:138` 删除 `kDestSlot = 0`，改用 `instr.tmem_slot`
4. 新增 `tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp`（行为验证）
5. 全量跑 ctest 验证无回归
6. **commit**: `fix(tcgen05): route ld/st/cp to instruction-specified tmem_slot (Oracle C2 Phase 2)`

### Phase 3: Archive + ADR Postmortem（commit 3）

1. artifacts git-tracked（per lessons-learned §6 E Checklist）
2. ADR-0016 追加 "2026-07-12 Postmortem: C2 fix" 段
3. `openspec archive fix-tcgen05-ld-st-slot-routing --yes`
4. **强制 prompt**：是否生成 postmortem（per openspec-archive-change skill）

## Capabilities

### New Capabilities
（slot routing 是 instruction semantics 修改，不是新 capability — 归入 Modified）

### Modified Capabilities

| Capability | 变化 |
|------------|------|
| `tcgen05-handlers-core` | 现有 LD/ST handler 行为修订：slot 必须来自 `instr.tmem_slot`（不再硬编码 `0`） |
| `tcgen05-handlers-extended` | 现有 CP handler 行为修订：删除 `kDestSlot = 0` 常量 |
| `tcgen05-ir-types` | `Tcgen05Instr` 结构体加 `tmem_slot` 字段（per `statement_context.h:180-190` 现有字段模式） |

每个 modified capability 需要 delta spec 文件，路径 `specs/<capability>/spec.md`。

## Impact

### 影响的代码（预计）

| 文件 | 变更类型 | LoC 估计 |
|------|----------|----------|
| `src/grammar/ptxInstructions.g4` | 修改（加 tmem_slot 操作数规则） | +10 |
| `include/ptx_ir/ptx_op.def` | 修改（op_count 2→3 / 2→3 / 3→4） | +0 / -0 (3 行) |
| `include/ptx_ir/statement_context.h` | 修改（加 `tmem_slot` 字段） | +2 |
| `src/ptx_parser/ptx_visitor.cpp` | 修改（提取 tmem_slot） | +5 |
| `include/ptx_ir/statement_factory.h` | 修改（加 tmem_slot 参数） | +2 |
| `src/ptxsim/instructions/tcgen05.cpp` | 修改（3 行 slot 0 → instr.tmem_slot） | +3 / -3 |
| `src/ptxsim/instructions/tcgen05_cp.cpp` | 修改（删除 kDestSlot 常量） | +1 / -3 |
| `tests/integration/ptx/test_tcgen05_ld_parse.cpp` | 修改（加 tmem_slot 断言） | +15 |
| `tests/integration/ptx/test_tcgen05_st_parse.cpp` | 修改（加 tmem_slot 断言） | +15 |
| `tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp` | 新增文件 | ~120 |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 Postmortem 段 | +30 |
| **总计** | | **+203 / -6** |

### 影响的依赖

- **前置**：FU-1 `fix-tcgen05-commit-wait-group` (C3) 必须**先完成** — 建立 IMMEDIATE 提取 pattern；如果本 change 选择 `.tmem_slot::N` qualifier 路径，则需要 C3 同样的 IMMEDIATE walk
- **无新外部依赖**

### 不影响的依赖

- 11 个 S_TCGEN05_* enum（`ptx_op.def:127-137`）— 仅 op_count 修改，不增 enum
- Tmem / TmemAllocator API — 不动
- TcQueue — 不动（由 FU-1 改）
- Tcgen05PipelineHandler — 不动
- Tcgen05Instr 现有 5 个字段（cta_group/dtype/num_regs/has_block_scale）— 不动（由 FU-1/C3 决定如何填充）

### 影响的文档

- `docs/adr/0016-blackwell-only-tcgen05.md` — 追加段
- `openspec/specs/tcgen05-handlers-core/spec.md` — 新增 delta spec（"LD/ST slot 来自指令"）
- `openspec/specs/tcgen05-handlers-extended/spec.md` — 新增 delta spec（"CP slot 来自指令"）
- `openspec/specs/tcgen05-ir-types/spec.md` — 新增 delta spec（"Tcgen05Instr.tmem_slot 字段"）
- 根 `AGENTS.md` 已知限制表 — 不需修改（C2 是内部修正，未改变对外能力）

## Cross-Change Dependencies

| 上游 | 本 change | 下游 |
|------|----------|------|
| FU-1: `fix-tcgen05-commit-wait-group` (C3) — **必须先完成** | **fix-tcgen05-ld-st-slot-routing** (C2) | FU-5: `tcgen05-flashattention-coverage` (E2E mini-kernel) |
| `archive/2026-07-10-implement-tcgen05-handlers-extended/` (提供 C2 root cause 范围) | | FU-4: `fix-tcgen05-multi-warp-fragment` (C4) 可在本 change 后并行 |
| `../fix-tcgen05-mma-accumulator-and-f32-storage/` (helper 已支持 accumulate + f32) | | |

## Design-Time Checklist (per ptx-lessons-learned 强制项)

### Checklist A: 函数迁移完整性（per lessons-learned §A）

- [x] **Baseline 函数清单**：
  - `tcgen05_fragment_mma_ld` (subset of `processTcgen05Ld` at `tcgen05.cpp:402-439`)
  - `tcgen05_fragment_mma_st` (subset of `processTcgen05St` at `tcgen05.cpp:448-484`)
  - `processTcgen05Cp` (`tcgen05_cp.cpp:127-156`)
- [x] **逐行 diff 计划**：见 `design.md` §Migration Plan
- [x] **跨模块状态翻译表**：
  - `Tcgen05Instr.tmem_slot` (IR) → `tmem.write(instr.tmem_slot, ...)` / `tmem.read(instr.tmem_slot, ...)` (handler)
  - 默认 `tmem_slot=0` 保持向后兼容（现有 caller 不感知新字段）
- [x] **回退策略**：每个 Phase 独立 commit、独立可 revert

### Checklist B: 重构前（per lessons-learned §B）

- [x] **基线 worktree 计划** (per lessons-learned §4)：
  ```bash
  git worktree add .worktrees/baseline-c2 <commit-after-FU-1-merged>
  ```
- [x] **Phase 拆分**：
  - Phase 1: grammar + IR + visitor + factory + parser tests
  - Phase 2: handlers + 行为测试
  - Phase 3: archive + ADR postmortem
- [x] **失败处理策略**：任何已有测试回归 → 立即 revert 该 Phase（per lessons-learned §3）

### Checklist C: 写注释（per lessons-learned §C）

- [x] **关键注释**：handler 处加 `// Post-C2: slot comes from instr.tmem_slot, not hardcoded 0`
- [x] **Tcgen05Instr 字段注释**：解释 `tmem_slot` 默认 `0` 与向后兼容性的关系

### Checklist D: Commit 前（per lessons-learned §D）

- [x] **跑过 baseline worktree 对比**
- [x] **AGENTS.md 同步项**：handler dispatch 表不变，不需更新
- [x] **ADR 追加段落**：Phase 3 追加到 ADR-0016
- [x] **OpenSpec tasks.md 状态变更**：Phase 3 archive 后 tasks.md 标记 `[x]`
- [x] **commit message 列出独立 fix 编号**：Phase 1 = "Oracle C2 Phase 1"，Phase 2 = "Oracle C2 Phase 2"

### Checklist E: OpenSpec artifacts 提交顺序（per lessons-learned §6 2026-07 新增）

- [x] **artifacts-first**：Phase 3 第一步 commit `docs(openspec): ...` 然后才 archive
- [x] **每个 commit 独立可 revert**

### Checklist G: OpenSpec lifecycle 约束（per lessons-learned §6/G）

- [x] **Ref 链接** 到 `archive/2026-07-10-implement-tcgen05-handlers-extended/` 与 `../fix-tcgen05-mma-accumulator-and-f32-storage/`
- [x] **禁止 amend 已归档 change**：本 change 是新建 `fix-*` change，不是 amend

### Checklist H: Pre-implementation Review 强制项（per lessons-learned §7 2026-07 新增）

- [x] **Metis pre-implementation review**：✅ 2026-07-11（本 change 提交前应再次调用 Metis 验证 Phase 1 grammar 修改方案 — 见 Phase 1 step 0）
- [x] **Oracle 决策建议**：✅ 2026-07-11 (`ses_0b3791d78ffewb52428kJJ2Irz`) C2 BLOCKER HIGH confidence
- [x] **真实 PTX 语法验证**（CUDA toolkit 可用时）：extract `cuobjdump -xptx` 验证 ld/st/cp 真实语法包含 slot 操作数 — **实施 Phase 1 前必跑**
- [x] **worktree 状态验证**：`ls .worktrees/baseline-c2/` 验证目录非空

### Checklist J: OpenSpec artifacts 内部一致性（per lessons-learned §10 2026-07 新增）

- [x] **范围数字对齐**：本 proposal Impact §"LoC 估计" 与 design/tasks 同 LoC 数字
- [x] **design Decision 路径 vs spec Scenario 路径**：design 写 "PTX `tmem_slot::N` qualifier" → spec 写"qualifier::N 路径" → 一致（待 Phase 1 0-step 决定方案后固化）

### Checklist L: ANTLR grammar modification（per lessons-learned §9 2026-07 新增）

- [x] **bare string token 反模式**：本 change 禁止新增 `'tmem_slot'` / `'slot'` 之类 bare token（FT-§9 ad808e3 → 55e216a 案例教训）
- [x] **必跑 TDD**：建立 baseline → 复制 `bench/cute/*.ptx` → 修改 lexer/parser → `./tests/ptx/test_all_ptx.sh`
- [x] **Commit 顺序**：fix(grammar) → test(ptx) regression guard → docs(dev-process) lesson

### Checklist M: Cross-cutting risk (FU-1 前置)

- [x] **必须先完成 FU-1**：`fix-tcgen05-commit-wait-group` 是本 change 的 IMMEDIATE 提取 pattern 来源
- [x] **如本 change 选择 qualifier 路径而 FU-1 未完成**：必须复用 FU-1 的 parse tree walk 代码（设计文档需明确这一点）

## Major Risks

| ID | 风险 | Severity | Mitigation |
|----|------|----------|------------|
| R1 | Phase 1 grammar 修改引发 ANTLR 预测冲突或 ID 抢占 | HIGH | 严格遵循 §9/§L：不用 bare token；改完跑 `./tests/ptx/test_all_ptx.sh` 47/47 |
| R2 | op_count 增加破坏现有测试（ld/st/cp 调用方对 op_count 的硬编码假设） | MEDIUM | Phase 1 跑全量 ctest 验证 parser 测试 + handler 测试；若回归立即 revert |
| R3 | FU-1 (C3) 未完成导致本 change Phase 1 visitor 实现受阻 | HIGH | 本 change 起点必须 FU-1 已 merge；或在 proposal phase 0 决定采用不依赖 FU-1 的方案（如把 IMMEDIATE walk 内联到 visitTcgen05Inst） |
| R4 | 默认 `tmem_slot=0` 引发调用方静默行为变化 | LOW | 默认值与现有硬编码一致（都是 0），零行为变化 |
| R5 | 真实 PTX 语法不包含 slot 操作数（CUDA 内部决定 slot 路由） | MEDIUM | Phase 1 step 0 必跑 `cuobjdump -xptx` 验证；若 PTX 不含 slot 操作数，需要 fallback 到 `.tmem_slot::N` qualifier 路径或文档化"simulator-only 扩展" |
| R6 | handler 改 `instr.tmem_slot` 但测试 fixture 未提供该字段 | MEDIUM | tests/integration/tcgen05/ 中的 handler-level fixtures 需 update 编译通过；现有 helper 直接调用的单元测试不受影响 |

## References

- Oracle 2026-07-11 BLOCKER 审计: session `ses_0b3791d78ffewb52428kJJ2Irz`（C2 BLOCKER, HIGH confidence）
- Oracle 2026-07-11 架构审查: session `ses_0b3791d78ffewb52428kJJ2Irz` 提议 FU-3 (C2) 作为 follow-up #3
- Oracle 2026-07-11 split 验证: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` Q1 + Q5（推荐 Option b IMMEDIATE walk）
- 前置 change: [`../fix-tcgen05-mma-accumulator-and-f32-storage/`](../fix-tcgen05-mma-accumulator-and-f32-storage/)
- Ref (archived): [`../../archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/)
- ADR-0016: [docs/adr/0016-blackwell-only-tcgen05.md](../../../docs/adr/0016-blackwell-only-tcgen05.md)
- ptx-lessons-learned: [`.opencode/skills/ptx-lessons-learned/SKILL.md`](../../../.opencode/skills/ptx-lessons-learned/SKILL.md) §3, §4, §6, §7, §9, §L
- ptx-grammar-modification skill: [`.opencode/skills/ptx-grammar-modification/SKILL.md`](../../../.opencode/skills/ptx-grammar-modification/SKILL.md)
- PTX ISA §9.7.16 (tcgen05.ld/.st/.cp semantics)
