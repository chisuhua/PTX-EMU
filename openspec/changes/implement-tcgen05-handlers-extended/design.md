## Context

Change-3b 实施 5 core handler(mma/ld/st/commit/wait)。本 change 实施剩余 6 个 extended handler(alloc/dealloc/relinquish/cp/fence/mma_ws)。**可选**(5 core 已满足 4-change 路线图核心交付)。

## Goals / Non-Goals

**Goals**: 6 extended handler 实施 + 单元/集成/E2E 测试 + dispatcher 集成。

**Non-Goals**: 不修改 5 core handler(Change-3b scope)、不实现 cp.async.bulk.tensor(独立 follow-up)、不实现 cta_group::2 distributed_smem(独立 follow-up)、不实现 sm_120 sparse / FP4 / mxfp8。

## Decisions

### D1: 文件拆分 — 4 个独立源文件

**采纳**: `tcgen05_alloc.cpp` (3 handler: alloc/dealloc/relinquish) + `tcgen05_cp.cpp` + `tcgen05_fence.cpp` + `tcgen05_mma_ws.cpp`

**拒绝**: 单文件 `tcgen05_extended.cpp` 集中 6 handler(过度集中)

### D2: 优先级排序 — 简单优先

**采纳**: alloc/dealloc/relinquish(Phase 1) → fence(Phase 2) → cp(Phase 3) → mma.ws(Phase 4)

**理由**: 先建立 confidence,再处理复杂 case

### D3: mma.ws 共享 fragment 算术

**采纳**: mma.ws 复用 Change-3b 的 mma handler,只在 layout 上差异
**拒绝**: 完全独立实现(代码重复)

### D4: per-CTA 资源管理

**采纳**: alloc/dealloc 的 TMEM 槽位由 CTAContext 拥有(per `tmem.h` 现有架构)
**拒绝**: per-warp(违反 NVIDIA 硬件)

### D5: cp SMEM 源 = `.shared::cta` only

**采纳**: cp 只支持 per-CTA shared memory(per PTX ISA)
**拒绝**: `.shared::cluster` 需 distributed_smem(本 change scope 外)

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: alloc/dealloc 越界 | Tmem.h `validate_slot_id` 已存在 |
| **R2**: cp SMEM 越界 | SharedMemoryManager 已有 bounds check |
| **R3**: mma.ws fragment 错位 | 复用 mma + layout 转换 |
| **R4**: 6 commit 拆分过细 | 每 handler 1 commit,独立 revert |

## Migration Plan

### Phase 1: alloc/dealloc/relinquish(1 commit)
### Phase 2: fence(1 commit)
### Phase 3: cp(1 commit)
### Phase 4: mma.ws(1 commit)
### Phase 5: 单元/集成/E2E 测试(1 commit)
### Phase 6: 文档(1 commit)
### Phase 7: Archive(per Checklist G)

## Open Questions

无(6 handler scope 明确)。
