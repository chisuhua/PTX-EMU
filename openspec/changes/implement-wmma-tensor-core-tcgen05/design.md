# Design: Blackwell tcgen05 Handler Implementation (Phase 1-3)

> **架构依据**: [ADR-0016](../../../../docs/adr/0016-blackwell-only-tcgen05.md)
> **前置 change**: `implement-wmma-tensor-core-phase-0-infra` (per `Ref:` after archive)
> **后置 change**: TBD (`fix-cute-sm100-headers` if spike fails, or follow-up features)

## Context

`replace-silent-stub-failures` (archived 2026-07-04) 将 `WmmaHandler` 改为
抛 `UnsupportedInstructionException`。基础设施前置 change
`implement-wmma-tensor-core-phase-0-infra` 交付 4 个 Blackwell 新范式子系统
(TMA + TMEM + cluster + TcQueue)。

本 change 在基础设施之上**真正实现** Blackwell `tcgen05.*` handler：

1. `tcgen05.mma.cta_group::1.kind::f16` fragment arithmetic (Phase 1)
2. `tcgen05.ld` / `tcgen05.st` TMA + TMEM 集成 (Phase 2.1)
3. `tcgen05.commit` / `tcgen05.wait` 异步流 (Phase 2.2)
4. 16×16 GEMM e2e kernel + AGENTS sync + spec publish (Phase 3)

pre-Blackwell WMMA 路径保持抛异常（per ADR-0016）。

## Goals / Non-Goals

**Goals** (Phase 1-3):
1. `tcgen05.mma.cta_group::1.kind::f16` 真实 fragment arithmetic（基于 PTX ISA §9.7.13）
2. `tcgen05.ld/st` 与 TMA descriptor + TMEM 集成
3. `tcgen05.commit/wait` 异步流（复用 Phase 0-archive TcQueue）
4. cutlass 3.x / cute tcgen05 风格 16×16 GEMM e2e kernel
5. AGENTS.md / spec publish（从 change 移到 main specs）
6. 现有 123 labeled ctest (73 unit + 42 integration + 8 e2e) 无 regression

**Non-Goals** (per ADR-0016):
- pre-Blackwell WMMA 任何实现
- sm_120 sparse / FP4 / mxfp8（separate changes per feature）
- 性能对标（仅 functional correctness）
- SASS-level semantics
- 基础设施实现（TMA / TMEM / cluster / TcQueue 都由 phase-0-archive 交付）

## Decisions

### Decision 5: divergent wmma 抛异常, divergent tcgen05 async wait

**Context**: pre-Blackwell `wmma.mma.sync` is synchronous — divergent warps
can be detected at instruction fetch and rejected. `tcgen05.mma` is async
— divergent warp at fetch 是 fine 的；divergence manifests at `tcgen05.wait`。

**Choice**:
- pre-Blackwell: throw `ExecutionStateException` at fetch if
  `active_mask != 0xFFFFFFFF` (matches original proposal)
- Blackwell tcgen05: no throw at fetch; `tcgen05.wait` blocks until all
  async ops in the commit group complete; lanes that didn't issue an mma
  still wait correctly

**Rationale**:
- Matches hardware semantics (Blackwell async model is fundamentally
  different from sync wmma)
- Preserves `replace-silent-stub-failures` contract for pre-Blackwell

**Alternatives considered**:
- ❌ Always throw on divergence: rejects valid Blackwell code
- ❌ Always async wait: wrong for pre-Blackwell

### Decision 6: Tests use synthetic fixtures, Phase 3 uses cutlass e2e

**Context**: Phase 0 用 synthetic fixtures（被 phase-0-archive 覆盖）。Phase 1-3
接续 — Phase 1-2 单元/集成测试用 synthetic fixtures（直接构造 TMEM 内容、
mock TcQueue 调用），Phase 3 用 cutlass 3.x GEMM kernel 做 e2e。

**Choice**: Phase 1-2 测试用 synthetic fixtures；Phase 3 e2e 用 cute tcgen05 风格
16×16 GEMM kernel (Compiled via nvcc -ptx → cuobjdump -xptx → PTX-EMU 执行)。

**Rationale**:
- Phase 1-2 unit/integration 测试 fast + low-cost 回归网
- Phase 3 e2e 验证端到端（含 cute header integration）
- Phase 3 是首个用 cute headers 的 e2e (per proposal 风险评估)

**Alternatives considered**:
- ❌ All synthetic: Phase 3 不真实 exercise cute path
- ❌ All e2e: Phase 1-2 太慢 + regression 信号弱

### Decision 7: tcgen05.wait blocking 复用 barrier 基础设施 via BAR_SYNC state translation

> **本 decision 是 Phase 0-archive Decision 7 的应用层**。本 change 仅在 `tcgen05.wait`
> handler 中调用 `TcQueue::wait(group_id)`，由 Phase 0-archive 的 TcQueue 实现
> 实际 BAR_SYNC state translation。

**Context**: `TcQueue::wait(group_id)` 框架在 Phase 0-archive 已建立。本 change
仅在 `wmma.cpp` 的 `tcgen05.wait` handler 中:
```cpp
void WmmaHandler::processWait(ThreadContext* ctx, uint32_t group_id) {
    auto* warp = ctx->get_warp_context();
    auto* tc_queue = warp->get_cta_context()->get_tc_queue();
    tc_queue->wait(group_id);  // 复用 Phase 0-archive BAR_SYNC 路径
}
```

**Rationale**:
- `tcgen05.wait` 在 PTX 层面与 `bar.warp.sync` 对 waiting warp 语义相同：
  block 直到条件满足，然后 advance PC
- Phase 0-archive TcQueue::wait 已通过 `state-modification-audit` skill 验证
  BAR_SYNC translation chain (lessons-learned §1 硬门)
- 本 change 添加 `tcgen05.wait` handler 不引入新 state，新 audit 范围 = handler 本身
  (调用 `tc_queue->wait()`)

**Cross-module audit** (本 change commit 前):
- `wmma.cpp::processWait` — 唯一调用点为 `TcQueue::wait(group_id)`
- `TcQueue::commit_group_counter` 写点集 ⊆ Phase 0-archive 设计 (unchanged)
- `WarpState.threads[i].is_blocked` 写点集 ⊆ Phase 0-archive 设计 (unchanged)
- 本 change 不引入新状态变量 → 无新 audit 工作量

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fragment layout 解读错误无硬件交叉验证 | **Critical** (Oracle review) | 每个输出 fragment 元素 `// UNVERIFIED-AGAINST-HARDWARE` 注释 + 手工对照 PTX ISA §9.7.13 latest；本项目为 Blackwell tcgen05 fragment layout 解释的 primary source |
| cute 模板 sm_100 fallback 到 sm_90 wgmma | Medium | cute header spike (前置 change) 已验证；e2e 测试自动验证 emit 的 PTX 是否走 Blackwell path |
| TcQueue BAR_SYNC reuse 集成 bug | High | 复用 Phase 0-archive 已审计的 BAR_SYNC path；wmma.cpp::processWait 只调用 `tc_queue->wait()` |
| cute_rmsnorm 升级触发后续依赖 | Low | 本 change 不升级 cute_rmsnorm；留作 follow-up |

## Migration Plan

**5 commits total** (per Oracle review 5→9 Phase 0 + 5 Phase 1-3 = 14 commits overall):

```
Phase 1: tcgen05.mma (2 commits, ~500-800 LoC)
  Commit 1.1: feat(wmma): implement tcgen05.mma fragment arithmetic (Fix #10)
              - rename src/ptxsim/instructions/tensor.cpp → wmma.cpp
              - replace throw with real fragment arithmetic
              - add // UNVERIFIED-AGAINST-HARDWARE annotations
  Commit 1.2: test(wmma): integration test (Fix #11)
              - tests/integration/tcgen05/test_tcgen05_mma_sync.cpp
              - 直接读 TMEM 验证 mma 结果

Phase 2: tcgen05.ld/st + commit/wait (2 commits, ~600-1000 LoC)
  Commit 2.1: feat(wmma): tcgen05.ld/st with TMA + TMEM integration (Fix #12)
  Commit 2.2: feat(wmma): tcgen05.commit/wait async flow (Fix #13)
              - 调用 Phase 0-archive TcQueue::commit/wait 框架

Phase 3: e2e GEMM + AGENTS + spec publish (1 commit, ~300-500 LoC)
  Commit 3.1: docs+test: e2e GEMM kernel + AGENTS sync + spec publish (Fix #14)
```

**Revert unit 澄清**:
- Commits 1.1, 1.2, 2.1, 2.2, 3.1 各自独立可 revert
- Revert 后 wmma.cpp 行为回到 `replace-silent-stub-failures` 抛异常合约
- 不破坏已有 cute_rmsnorm 等 e2e 测试

## Open Questions

1. **TMA host API 拦截策略** — separate ADR-0017 in future.
2. **sm_120 sparse / FP4 / mxfp8** — separate changes per feature.
3. **cute_rmsnorm 升级到 tcgen05** — 后续 follow-up issue，本 change 不实施。
4. **async queue priority vs scheduler** — 需真实 Blackwell hardware 数据 calibrate。

## 影响范围

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `src/ptxsim/instructions/wmma.cpp` | 重命名 (Phase 1.1) | 从 `tensor.cpp` rename + 抛异常 → tcgen05 实现 |
| `src/CMakeLists.txt` | 修改 (Phase 1.1) | line 103 `ptxsim/instructions/tensor.cpp` → `ptxsim/instructions/wmma.cpp` |
| `src/ptxsim/instructions/AGENTS.md` | 修改 (Phase 3.3) | KNOWN STUBS 移除 WMMA 条目 |
| 根 `AGENTS.md` | 修改 (Phase 3.4) | 已知限制表 WMMA 条目更新 |
| `tests/e2e/kernel/CMakeLists.txt` | 修改 (Phase 3.2) | 添加 `bench/cute/include/` include path |
| `tests/e2e/kernel/test_blackwell_gemm.cu` | 新建 (Phase 3.1) | Cute tcgen05 风格 16×16 GEMM kernel e2e |
| `openspec/specs/wmma-tensor-core/spec.md` | publish (Phase 3 archive) | 从 change specs/ 移到 main specs |

## 相关 ADR

- **ADR-0016**：本 design 的依据 (Blackwell-only vision)
- **ADR-0012** (per-thread-pc)：复用 Phase 0-archive TcQueue 独立抽象层
- **ADR-0008** (barrier-semantics)：TcQueue wait 复用 BAR_SYNC state translation
  (unchanged from phase-0-archive Decision 7)
- **未来 ADR-0017**：TMA host API 拦截策略
- **未来 ADR-0018**：cluster mode distributed_smem（deferred）
- **未来 ADR-0019**：async queue 与 WarpState 集成模式（if needed）
