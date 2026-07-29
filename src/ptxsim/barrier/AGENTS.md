# PTX-EMU Barrier Module
**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md; this file only documents ptxsim/barrier-specific content.

Per-CTA 统一屏障状态机 — 管理 warp 级 (`bar.warp.sync`) 和 CTA 级 (`bar.sync`) 屏障同步，通过 `BarrierModule` API 路由。

## STRUCTURE

```
src/ptxsim/barrier/          include/ptxsim/barrier/
├── barrier_module.cpp       ├── barrier_module.h   # 统一入口，owns warp_barriers_[] + cta_barriers_[]
├── warp_barrier.cpp         ├── warp_barrier.h     # Per-warp barrier: 5-state enum + arrive/is_complete/reset
├── cta_barrier.cpp          ├── cta_barrier.h      # Per-CTA barrier: mutex + arrived thread set
                             └── barrier_types.h    # 常量: MAX_BARRIERS_PER_CTA, WARP_SIZE, 默认 ID
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| 屏障入口 | `barrier_module.h` — 16 个 warp barrier + 16 个 CTA barrier 的统一接口 |
| Warp barrier 状态机 | `warp_barrier.h` — State: Uninitialized→Initializing→Waiting→Complete→Released |
| CTA barrier 线程集 | `cta_barrier.h` — mutex 保护 + `std::set<ThreadContext*>` |
| 屏障指令 handler | `../instructions/barrier.cpp` — 经 BarrierModule API 路由 |
| 常量定义 | `barrier_types.h` — `DEFAULT_CTA_BARRIER_ID`, `MAX_WARP_BARRIERS` |

## KEY FILES

| File | Role |
|------|------|
| `barrier_module.h` | `BarrierModule` — 主 API: `init_warp_barrier()`, `arrive_at_warp_barrier()`, `is_warp_barrier_complete()`, `release_warp_barrier()`, `init_cta_barrier()`, `arrive_at_cta_barrier()`, `is_cta_barrier_complete()`, `release_cta_barrier()`, `reset_all()` |
| `warp_barrier.h` | `WarpBarrier` — per-lane arrive, `is_complete()`, `needs_to_wait()`, `reset()`, `mark_released()` |
| `cta_barrier.h` | `CTABarrier` — mutex 保护 arrived_threads_, `arrive()`, `is_complete()`, `reset()` (keep init state for reuse) |
| `barrier.cpp` | BarWarpSyncHandler + BarHandler — 薄派发层，不直接管理 barrier 状态 |

## CONVENTIONS

- **统一路由**: barrier handler 必须通过 `BarrierModule` API — 不直接操作 `WarpBarrier`/`CTABarrier` 内部状态
- **OR arrived_mask**: `release_warp_barrier()` 中做 `active_mask |= arrived_mask`，不在 `set_active_mask()` 中做 — ret handler 依赖 overwrite 语义 (0u 清空)
- **Wbar 已移除**: 无 `Wbar` struct 残留 — 全部使用 `BarrierModule` + `WarpBarrier`/`CTABarrier`
- **CTA 释放**: `release_cta_barrier()` 遍历 arrived threads, 设置 `state=RUN`, 解阻塞, 推进 PC, 调用 `update_active_mask()` 让调度器可见
- **Warp 释放**: `release_warp_barrier()` 遍历 arrived mask, 解阻塞 + 激活所有 lane, 推进 PC 到 reconvergence_pc, 然后 `wbar->reset()`

## ANTI-PATTERNS

- ❌ 新增 `Wbar` struct 使用 — 已全部迁移至 `BarrierModule` + `WarpBarrier`
- ❌ `set_active_mask()` 做 OR 合并 — OR 逻辑仅在 `BarrierModule::release_warp_barrier()` 中
- ❌ barrier handler 直接修改 warp barrier 状态 — 必须经过 `BarrierModule` API
- ❌ 在 `WarpBarrier::init()` 中重置 `arrived_mask_` 当 re-init — 第二 half 会丢失已到达 lane
- ❌ 在 `CTABarrier::reset()` 中清除 `is_initialized_` — barrier 需可复用 (bar.sync 0 多次出现)