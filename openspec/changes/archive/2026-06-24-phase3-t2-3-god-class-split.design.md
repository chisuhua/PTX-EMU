**Retroactive synthesis from git log — not an original design document**
> 合成日期: 2026-07-06
> 来源: `proposal.md` (god class 拆分 + 4-POD/3-POD 计划) + `tasks.md` (Phase A1-A5 + B1-B2 + C1-C2 实施记录) + git log (commits `7054593`, `7952120`, `67ad828`, `8b9b025`, `2a3b48a`, `5617665`, `421eec9`, `af6f52f`, `a0b8281`, `33e1f99`, `8248303`, `2d24403`, `7c5bede`) + 关联 change `phase3-t2-1-active-mask-unify` (prereq) + ADR-0012 (per-thread PC)

# Phase 3 T2-3: ThreadContext / WarpContext god class POD 拆分 — Retroactive Design

## Context

`ThreadContext` 有 26 public 数据字段 + 2 private (Errata §1.1 校正后"81"含方法/常量)，`WarpContext` 有 32 数据字段。god class 无法 mock，导致 `test_active_mask_consistency` 等核心测试稀薄（审计 §A1 关联 1）。`WarpState` 6 字段中 3 个是死代码（`thread_predicates` 0 引用、`wbars` + `current_wbar_id` 已 deprecated、`warp_pc` 0 引用）。

T2-3 是 T2-1 的下游：拆分后状态机必须先收敛到 single source of truth（POD 边界依赖统一数据源）。审计 P2-2 估计 2 周，本计划评估 7-10 天。**A5 真阻塞发现**: `BarWarpSyncHandler::processOperation` (`src/ptxsim/instructions/barrier.cpp:140-280`) 主体逻辑仍用 legacy `warp_state.wbars[]` + `current_wbar_id`，BarrierModule 状态未被更新，因此不能直接物理删除 `wbars` / `current_wbar_id` 字段。

## Goals / Non-Goals

**Goals:**
- ThreadContext 拆 4 个 POD: `ExecStatePod` / `RegisterPredicatePod` / `MemoryPod` / `ProgramRefPod`
- WarpContext 拆 3 个 POD: `LaneMaskPod` / `IdentityPod` / `BackendLinksPod`
- WarpState 精简 6 → 2 字段 (`threads[]` + `exec_mask`)
- 删除 deprecated 字段: `thread_predicates` (0 引用), `warp_pc` (0 引用), `wbars` + `current_wbar_id` (A5 部分完成 PoC)
- Facade 模式保持外部 API 兼容
- 重写 `include/ptxsim/testing/warp_test_utils.h` (207 行) + `scheduler_utils.h` (72 行) + `assertion_utils.h`
- 更新 69 个测试文件

**Non-Goals:**
- active_mask 三源统一（独立 change `phase3-t2-1-active-mask-unify`）
- CVT 策略模式重构（独立 change `phase3-t2-6-cvt-strategy-pattern`）
- WMMA / Tensor Core 相关字段
- A5 物理删除 EXE_STATE (统一到 `warp_state.threads[i].status`) — 实际由 `cleanup-deprecated-barrier-apis` 后续处理

## Decisions

### Decision 1: Facade 模式 + POD 持有 (而非 POD 替换)

**问题**: POD 拆分后 `ThreadContext` / `WarpContext` 是否完全替换为 POD?

**方案分析**:
- **方案 A**: Facade 模式 — `ThreadContext` / `WarpContext` 保留为薄 facade，**持有** POD 成员，外部 API 通过转发方法兼容。零 breaking change。
- **方案 B**: 完全替换为 POD 组合 (e.g. `using ThreadContext = std::variant<4 PODs>`)。激进 breaking change。
- **方案 C**: 拆分多个子类继承自基类。God class 拆分成 god 继承链，无实质改善。

**选择**: **方案 A**。保持外部 API 兼容 (69 个测试文件最小修改) + 内部状态可单元测试 (POD mock)。

**证据**: tasks.md Phase A3 "ThreadContext 改 facade" + proposal.md §What Changes "Facade 模式：ThreadContext + WarpContext 改为持有 POD 的 facade（保持外部 API 兼容）"。

### Decision 2: 4-POD (Thread) + 3-POD (Warp) 切分边界

**问题**: 字段如何分组到 POD?

**方案分析**:
- **方案 A** (proposal 推荐):
  - Thread 4 POD: `ExecStatePod` (state, bar_id, IDs) / `RegisterPredicatePod` (regbank, cc_reg, operand buffers) / `MemoryPod` (shared/local mem, cta/warp ctx) / `ProgramRefPod` (statements, name2Sym/Share, label2pc, call_stack, dst reg name)
  - Warp 3 POD: `LaneMaskPod` (active_mask, thread_ids, count, divergence) / `IdentityPod` (warp_id, physical ids, pc) / `BackendLinksPod` (regbank, sm/cta ctx, simt_stack, threads)
- **方案 B**: 更细粒度拆分 (8+ POD)。过度拆分增加跳转成本。
- **方案 C**: 更粗粒度 (2 POD per context)。失去拆分意义。

**选择**: **方案 A**。平衡"可 mock 性"与"跳转开销"。

**证据**: proposal.md §What Changes 详述 + tasks.md Phase A1 实施记录 (commit `7054593` "extract 7 POD structs")。

### Decision 3: 7 POD 而非 8 (commit `7054593` 实证)

**问题**: tasks.md §A1 计划 8 POD，但 commit `7054593` 实际为 7 POD。

**方案分析**:
- **方案 A**: 7 POD（实际 commit 数）— `ExecStatePod` + `RegisterPredicatePod` + `MemoryPod` + `ProgramRefPod` (Thread 4) + `LaneMaskPod` + `IdentityPod` + `BackendLinksPod` (Warp 3)。`BarrierControlPod` 合并到 `LaneMaskPod`（proposal 提的 4th Warp POD 实际未独立）。
- **方案 B**: 强行 8 POD (`BarrierControlPod` 独立)。Phase A1 实施时合并到 `LaneMaskPod`，因 WarpState 拆分后 `wbars` 不在 WarpContext 范围。

**选择**: **方案 A**。以 commit 实证为准。

**证据**: commit `7054593` 实际 message "extract 7 POD structs from ThreadContext/WarpContext" (注：proposal.md 写 7-8，tasks.md 写 8，但 commit 实证 7)。

### Decision 4: A5 物理删除字段分两阶段

**问题**: A5 (物理删除 `wbars` + `current_wbar_id` + `thread_predicates` + `warp_pc` + EXE_STATE) 在 `BarWarpSyncHandler` 仍用 legacy 状态下不能执行。

**方案分析**:
- **方案 A**: A2 (commit `5617665`) 已物理删除 `thread_predicates` + `warp_pc` (0 引用)。`wbars` + `current_wbar_id` 仅 deprecate，**留 A5 PoC 后续**。EXE_STATE 物理删除**未执行** (T2-3 archive 时)。
- **方案 B**: A5 一次性物理删除所有字段。触发 BUG-POSTBARRIER-TWOHALVES。
- **方案 C**: 等待 `integrate-barrier-module-cta-warp` 完整合并后再启动 T2-3。

**选择**: **方案 A**。最小化风险 + 留下游 `cleanup-deprecated-barrier-apis` (2026-06-20 archived) + `migrate-bar-warp-sync-to-barrier-module` (2026-07-03 archived) 处理 handler 迁移。

**证据**: tasks.md §A5 BLOCKER 段 + commit `af6f52f` "document A5 blocker discovery - barrier.cpp uses legacy state" + commit `a0b8281` "update A5 status with PoC + discovery findings"。

## Implementation Commits

> **注**: 以下 commits 在 change 归档时已合并到 main，本节为追溯原始实施链。

| Commit | Sub-task | 摘要 |
|--------|----------|------|
| `7054593` | A1 | `refactor(contexts): extract 7 POD structs from ThreadContext/WarpContext` (8 个 POD 头文件 + 占位实现 + 单元测试) |
| `7952120` | A3a | `refactor(thread): add 4-POD members at end of class` (additive, 无 breaking) |
| `67ad828` | A3b | `refactor(thread): init() populates 4 POD members as secondary view` (二次视图) |
| `8b9b025` | A4a | `refactor(warp): add 3-POD facade members at end of class` (additive) |
| `2a3b48a` | A4b | `refactor(warp): add_thread() mirrors LaneMaskPod fields` (字段同步) |
| `5617665` | A2 partial | `refactor(warp-state): remove thread_predicates + warp_pc dead fields` (0 引用字段) |
| `421eec9` | A5 PoC | `refactor(test): migrate BarrierModule test from get_wbar() shim to direct API` |
| `af6f52f` | A5 doc | `docs(t2-3): document A5 blocker discovery - barrier.cpp uses legacy state` |
| `a0b8281` | A5 doc | `docs(t2-3): update A5 status with PoC + discovery findings` |
| `33e1f99` | B1 | `refactor(test): use set_state() in reset_warp for forward compat` |
| `8248303` | B2 | `refactor(test): use set_state() in test_post_barrier_divergence` |
| `2d24403` | C1 | `fix(factory): register InstructionFactory::cleanup() via atexit` (ASan leak fix) |
| `7c5bede` | Audit | `docs(audit): mark god class (M2/M3) fixed by T2-3 (Phase C2)` |
| `02274a7` | Archive | `chore(openspec): archive completed T2-3 god class split` |
| `3006e11` | Orphan README (后续) | `docs(openspec): add READMEs to 6 orphan archive changes (Fix #4)` |

## Risks / Trade-offs

| 风险 | 缓解（per tasks.md §Phase 3 验证门禁 + proposal.md 风险）|
|------|------|
| A5 物理删除触发 BUG-POSTBARRIER-TWOHALVES | A2 partial (`thread_predicates` + `warp_pc` 0 引用) + `wbars` 仅 deprecate。handler 迁移留给 `cleanup-deprecated-barrier-apis` (2026-06-20) |
| 8 POD 拆分增加跳转开销 | POD facade 编译期 inline (callgrind 验证无 perf regression) |
| 69 个测试文件更新工作量大 | Phase B 单独 commit + B1 前向兼容 (`set_state()` 替代 `t->state = RUN`) |
| 4 字段同步 (`is_active` + `!is_exited` + `!is_blocked` + `status==Active`) 漏字段 | 跑 `test_post_barrier_two_halves` + `test_post_barrier_divergence` 回归 |
| ASan leak (每个 test binary 漏 ~1296 bytes) | commit `2d24403` `atexit` 注册 `InstructionFactory::cleanup()` |
| heap-buffer-overflow at `ThreadContext::collect_operands:393` (5 pre-existing 测试) | out of T2-3 scope，独立跟踪 |

## Cross-References

- 原 artifacts: `openspec/changes/archive/2026-06-24-phase3-t2-3-god-class-split/{proposal.md,tasks.md,README.md}`
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-t2-1-active-mask-unify/` (上游 prereq)
- 关联 change: `cleanup-deprecated-barrier-apis` (2026-06-20 archived, A5 物理删除 wbars 后续)
- 关联 change: `migrate-bar-warp-sync-to-barrier-module` (2026-07-03 archived, BarWarpSyncHandler 迁移)
- 关联 change: `integrate-barrier-module-cta-warp` (2026-06-18 archived, T2-3 A2/A5 协同 prereq)
- ADR-0005: Memory region registration (POD 边界依赖)
- ADR-0012: Per-thread PC (POD split 的核心驱动)
- Lessons-Learned: §1 (跨模块状态翻译 — handler 迁移发现是 A5 真阻塞，正是 §1 典型场景)
- Lessons-Learned: §4 (复杂迁移分 Phase commit — T2-3 三阶段 8 commits 完美示范)
- A5 阻塞发现文档: `docs/superpowers/findings/2026-06-24-t2-3-a5-blocker-discovery.md`
- 审计原文: `docs/audits/HEALTH-AUDIT-2026-06-21.md:32,62,103,104,106,519,588,629` (M2/M3 标 fixed by commit `7c5bede`)

## Notes

> 本文件为 retroactive synthesis，最佳努力重建。如发现与原 commit body 不一致，**以原 commit body 为准**。
> 任何修改归档目录内文件的尝试被禁止（per Checklist G + Decision 1）。
>
> **实施时长**: 7-10 天 (per proposal.md 顶部) | **实际**: 13 commits on main, Phase A1-A4 + B1-B2 + C1 完成, A5 仅 PoC
>
> **遗留**:
> - A5 EXE_STATE 物理删除 (统一到 `warp_state.threads[i].status`) — 留给后续 change
> - 5 pre-existing heap-buffer-overflow at `ThreadContext::collect_operands:393` — out of scope
> - `GeneralCvtStrategy::convert()` 919 行 god class — 由 `phase3-t2-6-cvt-strategy-pattern` 后续 C7 处理
