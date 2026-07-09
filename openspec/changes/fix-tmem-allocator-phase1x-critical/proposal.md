# Fix TmemAllocator Phase 1.x — 3 Critical Issues per Oracle 2026-07-09 Review

> **架构依据**: ADR-0016 (Blackwell-only tcgen05)
> **所属 change**: [implement-tcgen05-handlers-extended](../implement-tcgen05-handlers-extended/) (Phase 1 修订)
> **Oracle 审查日期**: 2026-07-09
> **Phase 1 commit**: `486246a` (2026-07-09)

## Why

Phase 1 of `implement-tcgen05-handlers-extended` 已 commit (`486246a`)，新增 `TmemAllocator` + 3 个 alloc-family handler。Oracle 2026-07-09 审查识别出 **3 个 Critical Issues** 必须在 Phase 2 前修复：

| # | Issue | 严重度 | 影响 |
|---|-------|--------|------|
| 1 | **数据竞争 (UB)**: read-only methods (`is_allocated_start` 等) 不持 `mu_` 读 `allocations_`/`allocation_map_` | 🔴 Critical | Phase 2 cp / Phase 3 mma_ws 并发调用时是 UB |
| 2 | **缺少 handler 集成测试**: 0 个测试通过 `execute_warp_instruction` 驱动 3 个 handler | 🔴 Critical | 违反 AGENTS.md TDD 要求 |
| 3 | **AGENTS.md 未同步**: Phase 1 commit 未含 `AGENTS.md` / `src/ptxsim/instructions/AGENTS.md` 更新 | 🔴 Critical | 违反 Oracle Q7-A "每 Phase 末尾 commit 文档" |

此外 4 个 Recommendations 应一并处理（修复成本极低）：

| # | Recommendation | 严重度 | 备注 |
|---|---------------|--------|------|
| 4 | 多线程死锁检测 bug (`test_tmem_allocator.cpp:253-259` `elapsed` 计算逻辑错误) | 🟡 Medium | deadlock 时 `join()` 永久阻塞, REQUIRE 永不执行 |
| 5 | `kSlotCount = 256` 在 `tmem.h:28` + `tmem_allocator.h:47` 重复定义 | 🟡 Medium | 加 `static_assert` 强制一致 |
| 6 | `tcgen05_alloc.cpp:142` 注释 "most-recent" vs 代码 "lowest slot_id" 矛盾 | 🟡 Medium | 修正注释 |

## What Changes

### 修改

| 文件 | 范围 |
|------|------|
| `src/ptxsim/memory/tmem_allocator.h` | read-only methods 加 `mutable` + 文档化持锁 |
| `src/ptxsim/memory/tmem_allocator.cpp` | `is_allocated_start`/`is_allocated`/`active_allocation_count`/`total_allocated_slots` 持 `mu_`; 添加 `static_assert` |
| `src/ptxsim/instructions/tcgen05_alloc.cpp` | 修正 `processTcgen05Dealloc` 注释 |
| `tests/unit/memory/test_tmem_allocator.cpp` | 修复多线程死锁检测 (用 `std::async` + `wait_for`) |
| `tests/integration/tcgen05/test_alloc_dispatch.cpp` (NEW) | 集成测试: alloc handler 通过 dispatch 路径 |
| `tests/integration/tcgen05/test_dealloc_dispatch.cpp` (NEW) | 集成测试: dealloc handler |
| `tests/integration/tcgen05/test_relinquish_dispatch.cpp` (NEW) | 集成测试: relinquish handler |
| `tests/integration/CMakeLists.txt` | 注册 3 个新集成测试 |
| `AGENTS.md` | "已知限制" 表: tcgen05 3/11 deferred (CP/MMA_WS/FENCE) |
| `src/ptxsim/instructions/AGENTS.md` | "TCGEN05 HANDLER DISPATCH" 节: 3 deferred → CP/MMA_WS/FENCE |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 Phase 1 + Phase 1.x 完成记录 |

### 不修改

- ❌ `Tmem` 类本体（`src/ptxsim/memory/tmem.{h,cpp}`）—— 数据存储层无问题
- ❌ 5 core handlers（mma/ld/st/commit/wait）—— 已确认不受影响
- ❌ `processTcgen05Alloc` / `processTcgen05Relinquish` —— 逻辑正确，仅 dealloc 注释需修正
- ❌ `implement-tcgen05-handlers-extended` 4 个 artifacts —— 提案仍准确

### Breaking Changes

无。这是修复性变更（bugfix），不修改 public API 行为，仅修复 UB 和测试覆盖。

## Capabilities

### New Capabilities

- `tmem-allocator-concurrency-safe`: TmemAllocator 的所有 public methods 在并发访问下定义良好（无 UB）。

### Modified Capabilities

无 spec-level 变更。`TmemAllocator` 的功能契约不变；本 change 仅修复实现缺陷。

## Impact

### 影响的代码

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `src/ptxsim/memory/tmem_allocator.h` | 加 `mutable` 标注 + 文档 | +15 |
| `src/ptxsim/memory/tmem_allocator.cpp` | read-only methods 加 `lock_guard` + `static_assert` | +20 |
| `src/ptxsim/instructions/tcgen05_alloc.cpp` | 注释修正 | +5 |
| `tests/unit/memory/test_tmem_allocator.cpp` | 多线程测试修复 | +20 |
| 3 个新集成测试 | NEW | +250 |
| `tests/integration/CMakeLists.txt` | 注册 | +25 |
| `AGENTS.md` + `src/ptxsim/instructions/AGENTS.md` + ADR | 文档 | +20 |
| **总计** | | **+355** |

### 影响的依赖

- `tests/integration/CMakeLists.txt` 结构（参考 `tests/integration/tcgen05/` 现有 5 个集成测试）

### 不影响的依赖

- Phase 2-4 的 `cp`/`mma_ws`/`fence` handler
- 5 core handlers
- ANTLR grammar
- Tmem 本体

### 影响的文档

- `AGENTS.md`（"已知限制" 表更新）
- `src/ptxsim/instructions/AGENTS.md`（TCGEN05 DISPATCH 节）
- `docs/adr/0016-blackwell-only-tcgen05.md`（Phase 1 + 1.x 完成记录）
- 增量沉淀至 `.opencode/skills/ptx-lessons-learned/SKILL.md`（Oracle "read-only methods don't hold mu_" 是新教训）

## Design-Time Checklist

### 函数审计完整性

- [x] 修复后所有 public methods (mutating + read-only) 都正确处理 mu_
- [x] `static_assert` 强制 Tmem::kSlotCount == TmemAllocator::kSlotCount

### 测试覆盖

- [x] 3 个新增 handler 集成测试 (alloc/dealloc/relinquish)
- [x] 多线程死锁检测修复 (std::async + wait_for)
- [x] 现有 12 个单元测试保持通过

### 文档同步（Oracle Q7-A）

- [x] AGENTS.md 更新条目
- [x] `src/ptxsim/instructions/AGENTS.md` 更新条目
- [x] ADR-0016 追加记录

### 经验沉淀

- [x] ptx-lessons-learned 增量更新 (§27 或类似编号) — read-only methods don't hold mu_ pattern

## 实施前必跑（per ptx-lessons-learned §7）

- [x] Metis pre-implementation review ✅ (Phase 1 提案)
- [x] Oracle 决策建议 ✅ (2026-07-08 + 2026-07-09)
- [x] 基线 worktree 存在 ✅ (`.worktrees/baseline-tcgen05-extended` at `bb30ea2`)
- [x] Phase 1 已 commit ✅ (`486246a`)
- [ ] **新增验证**: Phase 1.x 修订后跑 `ctest -L "unit;tcgen05"` + `./tests/ptx/test_all_ptx.sh` 确认无 regression

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| implement-tcgen05-handlers-extended (Phase 1) | **fix-tmem-allocator-phase1x-critical** | implement-tcgen05-handlers-extended (Phase 2 cp) |

- 本 change 必须**先 archive**，才能开始 Phase 2 (cp) 实施
- Phase 2 cp handler 会调用 `is_allocated_start` (无锁→已修复为持锁) 和 `allocate`/`dealloc` —— 数据竞争修复对 cp 至关重要