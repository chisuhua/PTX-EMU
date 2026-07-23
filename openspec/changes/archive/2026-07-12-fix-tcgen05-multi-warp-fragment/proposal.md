# Fix tcgen05.mma Fragment — Multi-Warp Slot Offset (Oracle C4 BLOCKER)

> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) — Blackwell-only tcgen05
> **Sister change（已 propose）**: [`../fix-tcgen05-mma-accumulator-and-f32-storage/`](../fix-tcgen05-mma-accumulator-and-f32-storage/) — H1+H2 helper 内部修复（单 warp 路径）
> **Foundation change（dependency）**: [`../fix-tcgen05-commit-wait-group/`](../fix-tcgen05-commit-wait-group/) — visitor IMMEDIATE 提取 pattern（先合本 change 的前置）
> **Oracle 2026-07-11 审查**: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` (Oracle C4 BLOCKER + 4 follow-ups split 验证 Q1-Q6)
> **Metis pre-implementation review**: 必需（per ptx-lessons-learned §7/Checklist H — Oracle 已完成 Q1-Q6 验证，本 change 范围小，可视为 ⚠️ CONDITIONAL→GO）
> **强制 skills**: `ptx-lessons-learned` §3(分 Phase) + §4(基线 worktree) + §6(artifacts-first) + §7(Pre-impl Review) + Checklist H/J
> **范围**: 1 atomic commit（helper signature 加 warp_id 参数 + multi-warp slot math + multi-warp 集成测试）

## Why

Oracle 2026-07-11 审计（`ses_0aefd09c3ffeSqBIAGdxiRBFWC` Section C）发现 `tcgen05_fragment_mma_f16` helper 存在 1 个 HIGH-confidence BLOCKER 级多 warp 缺陷：

| ID | 缺陷 | FlashAttention 影响 | Oracle confidence |
|----|------|---------------------|-------------------|
| **C4** | Helper 单 warp 假设 — `tcgen05_helpers.cpp:23` `size_t c_slot = 64 + lane_id` 硬编码 | 多 warp 时 warp 0 和 warp 1 都写 `slot[64..95]`，C slot 冲突 → FlashAttention 多 warp tile 分配（warp 0:C[0:32], warp 1:C[32:64]）无法在 simulator 表达 | HIGH |

sister change `fix-tcgen05-mma-accumulator-and-f32-storage`（Oracle H1+H2 单 warp 路径修复）已 propose，但 README 已知限制（"single-warp 顺序执行"）表明 helper 不能跑多 warp mma。FlashAttention FA3 producer-consumer pipeline 需要至少 2 个 warp 同时 mma。

## What Changes

- **MODIFIED** capability `tcgen05-handlers-extended`: helper `tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false)` 新增 `int warp_id` 参数（位置在 `tmem` 之后），slot 计算改为 `c_slot = warp_id * 32 + 64 + lane_id`（per Oracle Q4 推荐 Option a）。A/B slot 保持不变（per Oracle 警告 #3：minimal fix 只 offset C 槽位，shared-input 模型适合 FlashAttention Q tile per-warp + K tile 共享 via cp 的实际语义；该决策需在 design.md D2 详细论证）。
- **MODIFIED** `src/ptxsim/instructions/tcgen05.cpp:383` — `processTcgen05Mma` 调用点改为 `tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), /*accumulate=*/false)`（与 H1 实施后状态协同）。
- **NEW** capability `tcgen05-multi-warp-fragment`: 多 warp mma fragment layout 测试覆盖（2-warp 配置 + 多 warp 隔离断言 + 多 warp 并行 mma 不冲突）。
- **NEW** 集成测试 `tests/integration/tcgen05/test_tcgen05_mma_multi_warp.cpp`:
  - TC1: 单 warp mma → `c_slot = 64 + lane_id`（向后兼容验证）
  - TC2: 2-warp 配置下 warp 0 mma → 验证 C 写入 `[64..95]`
  - TC3: 2-warp 配置下 warp 1 mma → 验证 C 写入 `[96..127]`（**无冲突**）
  - TC4: 2-warp 同时 mma → 验证两边 C slot 独立可读
- **MODIFIED** `include/ptxsim/instructions/tcgen05_helpers.h` doc: 更新 Layout 段记录多 warp slot 公式 + 移除 "Currently safe because SM scheduler runs one warp at a time" 注释（新增 "Each warp owns C slots [warp_id*32+64 : warp_id*32+95]; A/B slots [0..63] remain shared input"）。
- **MODIFIED** ADR-0016: 追加 "2026-07-11 Postmortem: Multi-warp fragment (Oracle C4 fix)" 段。

**BREAKING**: 1 个 caller API 变更（`tcgen05_fragment_mma_f16`），所有现有调用点编译失败直到更新（per Oracle 设计 §Phase 1 C1 mitigation — 编译期强制）。sister change `fix-tcgen05-mma-accumulator-and-f32-storage` Phase 1 实施后会引入新的 `accumulate` 默认参数，本 change 在 sister change 实施后再 apply 可避免双重签名变更（dependency 关系见 design.md §Dependencies）。

## Capabilities

### New Capabilities

- `tcgen05-multi-warp-fragment`: 多 warp mma fragment layout 能力。每个 warp 拥有独立的 C slot 范围 `[warp_id * 32 + 64 : warp_id * 32 + 95]`，A/B slot `[0..63]` 共享输入；使 FlashAttention FA3 的 multi-warp tile 分配能在 simulator 中表达。每个 warp 用 `warp->get_warp_id()` 显式传入 `tcgen05_fragment_mma_f16` helper。

### Modified Capabilities

- `tcgen05-handlers-extended`: helper `tcgen05_fragment_mma_f16` signature 新增 `int warp_id` 参数（位于 `Tmem&` 之后、`bool accumulate = false` 之前）。modifies `### Requirement: 6 extended tcgen05 handlers SHALL be implemented` 中 "scenarios that invoke the fragment kernel"。sister change `fix-tcgen05-mma-accumulator-and-f32-storage` 已修改同 capability（H1+H2，需 delta spec 协调）。

## Impact

### Affected Code

| 文件 | 变更类型 | LoC 估计 | 依赖 |
|------|----------|---------|------|
| `include/ptxsim/instructions/tcgen05_helpers.h` | 修改（helper signature + doc） | +5 / -3 | signature change 编译期强制 |
| `src/ptxsim/instructions/tcgen05_helpers.cpp` | 修改（slot 计算公式 + doc） | +3 / -2 | 影响所有 32 lane iterations |
| `src/ptxsim/instructions/tcgen05.cpp:383` | 修改（调用点传 warp_id） | +1 / -1 | 编译期强制 |
| `tests/integration/tcgen05/test_tcgen05_mma_multi_warp.cpp` | 新增 4 TC | +80 | 不依赖现有测试 |
| `tests/integration/tcgen05/CMakeLists.txt` | 新增 ctest target | +5 | 一致命名 `unit_*` / `integration_*` 前缀 |
| `docs/adr/ADR-0016-blackwell-only-tcgen05.md` | 追加 postmortem 段 | +25 | per checklist G |
| `AGENTS.md` | 更新已知限制表（"single-warp 顺序执行" → "multi-warp fragment layout 已 support"） | +1 / -1 | 取消旧限制 |
| **总计** | | **+120 / -8** | |

### Affected Dependencies

- 无新外部依赖
- 无 ANTLR grammar 修改（per Oracle Q4 验证 — slot 是 helper 内部数学，不需 PTX 语法）

### Non-Impacting

- `Tcgen05PipelineHandler` 3-stage 架构（per `tcgen05.cpp` dispatch table 不变）
- `TmemAllocator`（per `include/ptxsim/memory/tmem_allocator.h` — slot 是 helper 直读，不走 allocator）
- `tmem.read/write(slot_id, ...)` API（已接受 arbitrary slot_id）
- 11 S_TCGEN05_* handler dispatch（per `tcgen05.cpp:574-583` — 仅 helper 调用点签名变）
- 其他 4 个 follow-up changes（per Oracle Q1 验证 — C4 与 C1/C2/C3 完全独立，可并行；唯一约束是 C1/C2/C4 都修 `tcgen05.cpp:383` 调用点，避免 git conflict，建议 C4 最后 apply 或 rebase）

### Affected Documentation

- `include/ptxsim/instructions/tcgen05_helpers.h` — helper doc 注释（per Checklist C）
- `docs/adr/ADR-0016-blackwell-only-tcgen05.md` — 追加 postmortem 段（per Checklist G + lessons-learned §6）
- `根 AGENTS.md` — 已知限制表（"single-warp 顺序执行" 限制更新，per lessons-learned §8 — 重大功能交付须同步 README；multi-warp fragment 是 helper 内部修复，但限制表的"single-warp"已不再准确，需更新）
- `openspec/specs/tcgen05-handlers-extended/spec.md` — 通过 delta spec 更新（sister change 已 MODIFIED 一次；本 change 第二轮 MODIFIED 需协调）

## Open Questions

1. **A/B slot 是否也需要 per-warp 偏移**？当前决定：保持共享 input。理由：FlashAttention FA3 通常 Q tile per-warp + K tile 共享 via cp，所以 A（Q）per-warp、B（K）shared 是合理的；但这是 per-FlashAttention 假设。需 Oracle 在 design 阶段再次确认（Q1 follow-up）。
2. **多大 warp count 支持**？当前决定：限制为 2-4 warp（典型 SM 配置）。>4 warp 需要更复杂 layout（待真实 PTX 验证）。design.md D3 列出约束。
3. **tmem 总容量限制**：每个 warp 占 32 slot (4KB)，4 warp = 16KB。Tmem kTotalSize = 32KB (`tmem.h:30`)。design.md D4 论证此约束。

## Design-Time Checklist (Lessons-Learned, per `ptx-lessons-learned`)

### Checklist A: 函数迁移完整性

- [x] **Baseline 函数清单**: `tcgen05_fragment_mma_f16(Tmem&, bool)` 在 `include/ptxsim/instructions/tcgen05_helpers.h:51`，1 个 production caller (`tcgen05.cpp:383`) — 待变更 `tcgen05_fragment_mma_f16(Tmem&, int warp_id, bool)`
- [x] **逐行 diff 计划**: 见 design.md §Migration Plan Phase 1
- [x] **跨模块状态翻译表**: 不适用（per Oracle Q4 — 此 fix 是纯算术变更，不修改 `ThreadContext::state` / 互斥量 / PC）
- [x] **回退策略**: 1 atomic commit 独立 revert（改动小，影响隔离）

### Checklist B: 重构前

- [x] **基线 worktree 计划** (per lessons-learned §4):
  ```bash
  # 本 change 必须等 sister change (H1+H2) + foundation change (C3) 合并后再开始
  # 见 tasks.md §0
  git worktree add .worktrees/baseline-c4 $(git rev-parse HEAD)
  ```
- [x] **Phase 拆分**: 1 atomic commit（helper sig + caller + new test + AGENTS.md sync + ADR postmortem）
- [x] **失败处理策略**: 任何已有测试回归 → 立即 revert commit（per lessons-learned §3 + Oracle C1 mitigation）

### Checklist D: Commit 前

- [x] **跑过 baseline worktree 对比** (per Phase — tasks.md §0.5)
- [x] **AGENTS.md 同步项**: 已知限制表更新（per lessons-learned §8 — 重大功能交付同步根 README）
- [x] **ADR 追加段落**: ADR-0016 追加段（per D §3）
- [x] **OpenSpec tasks.md 状态变更**: archive 后 tasks.md 标记 `[x]` 全完成
- [x] **commit message 列出独立 fix 编号**: "Oracle C4"

### Checklist E: OpenSpec artifacts 提交顺序

- [x] **artifacts-first**: Phase 3 第一步 commit `docs(openspec): ...`
- [x] **每个 commit 独立可 revert**: 1 atomic commit (Phase 1)
- [x] **遵循 Checklist E (2026-07 新增)**: artifacts git-tracked

### Checklist G: OpenSpec lifecycle

- [x] **Ref 链接** 到 `sister-change/fix-tcgen05-mma-accumulator-and-f32-storage` + `foundation-change/fix-tcgen05-commit-wait-group`
- [x] **禁止 amend 已归档 change**: 本 change 是新建 `fix-*` change

### Checklist H: Pre-implementation Review 强制项

- [x] **Oracle 决策建议**: ✅ 2026-07-11 (`ses_0aefd09c3ffeSqBIAGdxiRBFWC`)，C4 HIGH confidence + Q1-Q6 split validation
- [x] **Metis pre-implementation review**: 待 execute（per tasks.md §0.1 — 鉴于本 change 范围小、Oracle 已给 split validation，可视为 ⚠️ CONDITIONAL → GO；但 4 个 artifacts 完成后必须跑 Metis 审计）

### Checklist J: artifacts 内部一致性

- [x] **proposal/design/spec/tasks 范围数字对齐**: 设计文档描述 3 个 LoC 数字 (helper sig + slot math + caller) 在 4 个 artifacts 一致出现
- [x] **design Decision 路径 = spec Scenario 路径**: design.md D2 描述 `warp_id * 32 + 64 + lane_id` 公式 ↔ spec.md Scenario 必须使用同一公式
- [x] **tasks 验证命令 = design 验证路径**: tasks.md Phase 1.5 验证命令引用 helper 文件路径必须与 design.md Migration Plan 同

## 跨 Change 依赖

| 上游依赖 | 本 change | 下游 |
|----------|----------|------|
| `sister: fix-tcgen05-mma-accumulator-and-f32-storage` (H1+H2) | **fix-tcgen05-multi-warp-fragment** | `tcgen05-flashattention-coverage` (FU-5) |
| `foundation: fix-tcgen05-commit-wait-group` (C3 IMMEDIATE pattern) | | |

- **上游依赖 1**: sister change H1+H2 必须**先合并**。原因：本 change 修改的 helper signature 是 sister change 已添加 `accumulate` 参数的扩展版。两 change 同时 apply 会产生 2 次连续 helper signature 改动（增大 rebase 风险）。Sequencing: sister (H1+H2) → 本 change (C4)。
- **上游依赖 2**: foundation change C3 推荐**先合并**（虽不强制 — 本 change 不依赖 C3 的 IMMEDIATE 提取 pattern）。原因：4 个 follow-up 中 C3 是 Oracle 验证的基础前置；如其他 follow-up 也都依赖 C3，先合 C3 减少后续 conflict 概率。Sequencing: foundation (C3) → 本 change (C4) 推荐但非强制。
- **下游**: 本 change 是 `tcgen05-flashattention-coverage` (FU-5) 的 5 个测试文件之一的依赖（`test_tcgen05_multi_warp_isolation.cpp`）。

### 推荐 Sequencing (per Oracle Q2)

```
foundation: fix-tcgen05-commit-wait-group (C3)
    │
    ├──► sister: fix-tcgen05-mma-accumulator-and-f32-storage (H1+H2)
    │        │
    │        └──► fix-tcgen05-multi-warp-fragment (本 change, C4) ◄── 当前
    │                  │
    │                  └──► tcgen05-flashattention-coverage (FU-5)
    │
    └──► (其他 follow-up changes C1, C2 与 C4 并行，各自独立 apply)
```

## 本 change 特有设计决策

详细论证见 `design.md`，摘要：

**决策 D1**: warp_id 作为 helper 参数显式传入（而非从 Tmem::owner_warp_id 推断）
- 理由：Tmem 无 `owner_warp_id` 字段；helper 调用点（`tcgen05.cpp:355`）已有 `WarpContext* warp`，`warp->get_warp_id()` 已存在 (`tcgen05_alloc.cpp:68` 使用过)

**决策 D2**: A/B slot 保持共享不变，仅 offset C slot
- 理由：minimal fix (Oracle 警告 #3)；FlashAttention FA3 的 Q per-warp + K shared via cp 实际语义符合

**决策 D3**: 限制为 2-4 warp 支持
- 理由：典型 SM 配置；>4 warp 需要 FA-3 aware 复杂 layout（待真实 PTX 验证）

**决策 D4**: 不修改 Tmem 容量（Tmem::kTotalSize = 32KB 不变）
- 理由：4 warp × 4KB = 16KB，剩余 16KB 可用，符合现有 tmem 模型
