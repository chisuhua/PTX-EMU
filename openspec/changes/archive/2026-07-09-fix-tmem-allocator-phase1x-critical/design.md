## Context

Phase 1 of `implement-tcgen05-handlers-extended` (commit `486246a`) 引入 `TmemAllocator` + 3 个 alloc-family handler。Oracle 2026-07-09 审查识别出 3 个 Critical Issues + 4 个 Recommendations。本 change 修复所有发现的问题,作为 Phase 1.x 修订。

**修复策略**: 一次性原子提交（per ptx-lessons-learned §3 "复杂迁移分 Phase commit",但 Phase 1.x 是修订而非新功能, 单个 commit 即可）。

## Goals / Non-Goals

**Goals**:
- 修复 read-only methods 的 UB 数据竞争
- 添加 3 个 handler 集成测试
- 同步 AGENTS.md（Oracle Q7-A）
- 修复多线程死锁检测 bug
- 修正 `processTcgen05Dealloc` 注释/代码矛盾
- 添加 `kSlotCount` 一致性 static_assert

**Non-Goals**:
- 不重构 TmemAllocator public API（保持向后兼容）
- 不修改 5 core handlers
- 不开始 Phase 2 (cp) 实施

## Decisions

### D1: read-only methods 加锁 vs `_unsafe` 后缀变体

**采纳**: read-only methods 直接加 `lock_guard(mu_)`, 删除 `_unsafe` 后缀变体。

**理由**:
- 简单性: 调用者无需关心 thread-safety 语义
- 安全: 默认行为是 thread-safe,符合"安全默认值"原则
- 性能: read-only path 加锁开销可忽略（持锁时间极短, 无 IO）

**拒绝**: 提供 `_unsafe` 后缀变体（如 `is_allocated_start_unsafe`）—— 增加 API 表面积, 增加误用风险。

### D2: 多线程测试用 std::async + wait_for

**采纳**: 用 `std::async(std::launch::async, ...)` + `future.wait_for(30s)` 真正检测死锁。

**理由**:
- 当前实现: `th.join()` 在 deadlock 时永久阻塞, REQUIRE 永不执行
- 改进: `future.wait_for(30s)` 返回 `future_status::timeout` 表示可能死锁, 可主动 `REQUIRE(false)`

### D3: kSlotCount 用 static_assert 强制一致

**采纳**: `static_assert(TmemAllocator::kSlotCount == Tmem::kSlotCount)` 在 `tmem_allocator.cpp` 顶部。

**理由**:
- 编译期检查, 零运行时开销
- 防止未来 one-side 变更引入不一致

**拒绝**: 让 `TmemAllocator::kSlotCount` 直接引用 `Tmem::kSlotCount` (会引入硬 include 依赖, 与 tmem_allocator.h 头部注释避免 include 依赖的设计矛盾)。

### D4: 集成测试放 tests/integration/tcgen05/

**采纳**: 3 个新集成测试放在 `tests/integration/tcgen05/`, 与现有 5 个 tcgen05 集成测试（mma_parse/ld_parse/st_parse/commit_parse/wait_parse）并列。

**理由**:
- 现有目录结构已支持 (per AGENTS.md "类型二: 指令序列集成测试")
- 使用 `ptxsim::testing::step_warp` + `execute_warp_instruction` 驱动 dispatch 路径

**拒绝**: 放在 `tests/unit/memory/` (会绕过 dispatch, 违反 "handler 级集成测试" 语义)。

### D5: AGENTS.md 更新方式

**采纳**: 直接修改根 `AGENTS.md` 和 `src/ptxsim/instructions/AGENTS.md`, 同一 commit 内提交。

**理由**:
- Oracle Q7-A: "每 Phase 末尾 commit 文档"
- 文档与代码同步 commit 便于 review

## Risks / Trade-offs

| 风险 | 等级 | 缓解 |
|------|------|------|
| read-only methods 加锁引入死锁路径 | 中 | `static_assert` + 单元测试 (`is_allocated_start` 在持锁下被调用, 不应死锁) |
| 集成测试无法构造真实 warp + cta 上下文 | 低 | 复用 `tests/integration/tcgen05/` 现有 fixture 模式 |
| `static_assert` 触发编译失败 (Tmem kSlotCount 变更) | 极低 | PTX ISA §9.7.13 锁定 256 slot, 不会变 |

## Migration Plan

### 单 Phase 单 commit

```
1. 修改 tmem_allocator.h/.cpp (read-only 加锁 + static_assert)
2. 修改 tcgen05_alloc.cpp (注释修正)
3. 修改 test_tmem_allocator.cpp (多线程测试修复)
4. 新建 3 个集成测试 + 注册
5. 修改 AGENTS.md / src/ptxsim/instructions/AGENTS.md / ADR-0016
6. build + ctest + PTX 测试
7. commit "fix(tmem-allocator): Phase 1.x critical issues per Oracle 2026-07-09 review"
8. openspec archive fix-tmem-allocator-phase1x-critical
```

### 失败处理

若 build 失败或测试回归: `git revert` 整个 commit, 不分拆 (修订性变更, 整体可回退)。

## Open Questions

无 (Oracle 已给出明确修复建议, 直接执行)。