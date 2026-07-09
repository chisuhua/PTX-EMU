## Context

Change-3b 实施 5 core handler(mma/ld/st/commit/wait)。本 change 实施剩余 6 个 extended handler(alloc/dealloc/relinquish/cp/fence/mma_ws)。**可选**(5 core 已满足 4-change 路线图核心交付)。

**Oracle 决策(2026-07-08)**: 7 关键问题已采纳 — Q1-A / Q2-A / Q3-A / Q4-B / Q5-C / Q6-B / Q7-A。详见 [proposal.md](proposal.md) "本 change 特有设计决策"。

## Goals / Non-Goals

**Goals**:
- 6 extended handler 实施
- 新增 TmemAllocator 抽象层(per Q1-A)
- 单元/集成/E2E 测试(per Q5-C 混合 oracle 策略)
- 递归锁审计(per `ptx-lessons-learned` §2)
- 文档同步(per Q7-A)

**Non-Goals**:
- 不修改 5 core handler(Change-3b scope)
- 不实现 cp.async.bulk.tensor(独立 follow-up)
- 不实现 cta_group::2 distributed_smem(per Q2-A,抛清晰异常)
- 不实现 sm_120 sparse / FP4 / mxfp8(per ADR-0016 锁定)
- 不实现 mma.ws 全部 collector 模式(per Q3-A,仅 `.warpspecialized::1`)

## Decisions

### D1: 文件拆分 — 4 个独立源文件

**采纳**: `tcgen05_alloc.cpp` (3 handler: alloc/dealloc/relinquish) + `tcgen05_cp.cpp` + `tcgen05_fence.cpp` + `tcgen05_mma_ws.cpp`

**拒绝**: 单文件 `tcgen05_extended.cpp` 集中 6 handler(过度集中)

### D2: Phase 优先级排序(Oracle 修订)

**采纳**: TmemAllocator + alloc/dealloc/relinquish(Phase 1) → cp(Phase 2) → mma.ws(Phase 3) → fence + 混合测试(Phase 4)

**理由**:
- Phase 1: 先建立 TMEM 生命周期管理 + 递归锁审计
- Phase 2: cp 依赖 Phase 1 的 TmemAllocator
- Phase 3: mma.ws 复杂,放到 cp 之后(可借助 cp 把数据搬进 TMEM)
- Phase 4: fence 是 no-op marker(Q6-B),作为"压测"测试集

### D3: mma.ws 共享 fragment 算术(Oracle Q3-A 范围限定)

**采纳**: mma.ws 复用 Change-3b 的 mma handler
- **范围**: 仅 `.kind::f16` + 单一 collector 模式 (`.warpspecialized::1`)
- **其他变体**: 抛清晰异常

**拒绝**: 完全独立实现(代码重复)

> **Phase 3 实施修订 (Oracle 2026-07-08 A-path)**: grammar 把 `.ws`
> 当作 `Q_TCGEN_WS` qualifier(不是独立的 `MMA_WS` sub-op),所以真实
> PTX 路径是 `op_kind=MMA + qualifiers={Q_TCGEN_WS, Q_F16, ...}`。
> ws qualifier 在 `processTcgen05Mma` 内部被识别,Q3-A 范围检查
> (Q_F16 必备),然后调 `tcgen05_fragment_mma_f16` helper(Phase 2.5
> 抽出,见 `include/ptxsim/instructions/tcgen05_helpers.h`)。
> ws-specific weight-stationary layout transform **deferred**
> (单 warp 简化下与 mma 算术相同)。
> `case Tcgen05OpKind::MMA_WS` 在 dispatch 表中保留(用于测试直接构造),
> 但与 `case MMA` 一样 route 到 `processTcgen05Mma`。

### D4: TmemAllocator 抽象层(Oracle Q1-A 新增)

**采纳**: 新增 `include/ptxsim/memory/tmem_allocator.h`,在 `cta->tmem()` 之上提供分配/释放/地址查询 API

**拒绝**:
- 直接 `cta->tmem()` 操作(缺乏分配语义)
- 把 `Tmem` 改为完全动态分配(过激,与 5 core 冲突)

**递归锁审计(必做)**:
- `tmem.h:47` 已有 `mutable std::mutex mu_`
- `cluster_context.h:50` 已有 `mutable std::mutex mu_`
- 审计命令: `grep -n "lock_guard\|unique_lock" src/ptxsim/memory/tmem_allocator.cpp`
- Falsification: 多线程并发 alloc/dealloc 单元测试,验证不死锁

### D5: per-CTA 资源管理

**采纳**: alloc/dealloc 的 TMEM 槽位由 CTAContext 通过 TmemAllocator 拥有

**拒绝**: per-warp(违反 NVIDIA 硬件)

### D6: cp SMEM 源 = `.shared::cta` only(Oracle Q4-B 简化)

**采纳**: cp 只支持 per-CTA shared memory;复用 `SharedMemoryManager` 已有的地址解析

**拒绝**:
- `.shared::cluster`(需 distributed_smem,scope 外)
- 新增 `SmemDescriptor` 抽象(冗余,`ptx_op.def:132` operand count=3,无 descriptor 字段)

### D7: cta_group::2 处理(Oracle Q2-A)

**采纳**: `cta_group::1` 完整实现;`cta_group::2` 抛清晰异常 `UnsupportedInstructionException`,message 包含 "cluster abstraction not yet implemented (ADR-0018)"

**拒绝**:
- 实现 SM-level barrier 越界(违反 ADR-0016)
- 整个 change 延后(过度保守)

### D8: fence 语义(Oracle Q6-B)

**采纳**: no-op marker — 仅记录 fence 位置,调 `warp->record_fence_position(before/after)`

**拒绝**:
- 调 `membar`/`FENCE` handler(模拟器无真实内存序)
- 集成到 barrier module(引入跨模块状态翻译 bug)

## Risks / Trade-offs

| 风险 | 等级 | 缓解 |
|------|------|------|
| **R1**: alloc/dealloc 越界 | 中 | TmemAllocator + `tmem.h:35` `validate_slot_id` |
| **R2**: cp SMEM 越界 | 中 | SharedMemoryManager bounds check |
| **R3**: mma.ws fragment 错位 | 中 | 复用 mma + layout 转换 + golden 标记 UNVERIFIED |
| **R4**: **递归锁死锁(Oracle 高风险)** | **高** | Phase 1 必做 `grep` 审计 + 多线程测试 |
| **R5**: cta_group::2 误用 | 低 | 清晰异常 + 文档说明 |
| **R6**: mma.ws 范围扩大 | 中 | 显式异常,其他 collector 模式 reject |
| **R7**: 6 commit 拆分过细 | 低 | per Phase 独立 revert |

## Migration Plan

### Phase 1: TmemAllocator + alloc/dealloc/relinquish(1 commit)
### Phase 2: cp(1 commit)
### Phase 3: mma.ws(1 commit)
### Phase 4: fence + 混合测试(1 commit)
### Phase 5: 文档(1 commit)
### Phase 6: Archive(per Checklist G)

### Baseline Worktree(per ptx-lessons-learned §4)

```bash
git worktree add .worktrees/baseline-tcgen05-extended <baseline-commit>
cd .worktrees/baseline-tcgen05-extended
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest -L tcgen05 --output-on-failure
```

每个 Phase 结束前对比 baseline:
```bash
cd /workspace/project/PTX-EMU/build && ctest -L tcgen05 --output-on-failure
```

## Open Questions

无(6 handler scope 明确,7 关键决策已通过 Oracle 审查)。
