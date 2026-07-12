# Postmortem: fix-tcgen05-mma-accumulator-and-f32-storage

> **Archived**: 2026-07-11 | **Commits**: 5 (d3be589 → df1f6de → f97863c → 3235227 → 58cbff9)
> **Verdict**: ✅ 0 regressions | 24/24 tcgen05 + 191/191 ctest + 45/45 PTX syntax

## TL;DR

`tcgen05_fragment_mma_f16` helper 零初始化 `c_frag` + f16 storage 双重 bug，导致 FlashAttention QK^T 沿 K 维循环累加**永远为一次结果**（`C = A*B` 而非 `C += A*B`），且 `f16×f16→f32` 输出违反 PTX ISA §9.7.16。Oracle 2026-07-10 审计揭示 5 个 HIGH/MEDIUM blockers。本 change 修复两处 root cause（H1 累加器缺失 + H2 f32 storage 格式），新增 4 个测试（T1_k_loop_4 + B2 commit/wait sequence + B7 epsilon 收紧），并追踪 4 个 BLOCKER 级 architectural debt（C1-C4）到 FU follow-up changes。**FlashAttention 基本要求（accumulate）已满足，但 4 个 FU 阻塞完整 kernel 执行。**

## Root Causes

### H1 — Helper 累加器缺失 (HIGH)

**Symptom**: `tcgen05_fragment_mma_f16` 零初始化 `c_frag` + 覆写写入，无法表达 FlashAttention QK^T/PV 矩阵乘的 `C += A*B` 沿 K 维循环累加。

**Detection**: Oracle 2026-07-10 审计 (5 HIGH/MEDIUM blockers)，helper body `src/ptxsim/instructions/tcgen05_helpers.cpp:42,45,57` 三处症状：
1. `T sum = 0;` 初始化（line 42）— 永远从零开始，不读 c_slot
2. `out[idx] = sum;`（line 57）— 覆写写入，不读 c_slot
3. 无 `load_c_slot` 调用 — 整个累加缺失

**Fix**: 新增 `bool accumulate = false` 参数 + `load_c_slot<T>` 模板 helper（per Oracle Q4：`alignas(T) std::array<uint8_t, kSlotSize>` + `memcpy` 避免 reinterpret_cast UB）+ `processTcgen05Mma` 显式传 `false`。

### H2 — Helper f16 storage 违反 PTX §9.7.16 (HIGH)

**Symptom**: `c_frag` 类型 `uint16_t`（f16）+ `f32_to_f16(sum)` 转换；与 PTX ISA §9.7.16 规定 `f16×f16→f32` 输出矛盾。Slot 利用率 50%（64 bytes / 128 bytes）。Golden header `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h:6` 声称 "32 f32 elements" 但实际是 f16 storage + f16→f32 readback 掩盖不一致。

**Detection**: Oracle 2026-07-10 审计。

**Fix**: `c_frag` 类型 `float` + 删除 `f32_to_f16` + memcpy size 64B → 128B + 4 处 readback site 迁移到 `alignas(16) float c_arr[32] + memcpy` 模式。

## Architectural Debt (Oracle 2026-07-11 — 4 BLOCKERs)

| ID | Issue | File:Line | Follow-up Change |
|----|-------|-----------|------------------|
| **C1** | `processTcgen05Mma` 显式传 `accumulate=false`，handler 永不累加 | `tcgen05.cpp:383` | `fix-tcgen05-idesc-parsing` (已 propose) |
| **C2** | ld/st 硬编码 `tmem.write(0, ...)` / `tmem.read(0, ...)` | `tcgen05.cpp:434,476` + `tcgen05_cp.cpp:138` | `fix-tcgen05-ld-st-slot-routing` (已 propose) |
| **C3** | `commit(1)` / `wait(warp, 0, 1)` 硬编码；`extractQualifiersFromContext` 丢弃 `IMMEDIATE` 值（19 个 call sites） | `tcgen05.cpp:512,550` + `ptx_visitor.cpp:155-183` | `fix-tcgen05-commit-wait-group` (已 propose) |
| **C4** | `c_slot = 64 + lane_id` 多 warp 冲突（warp 0/1 都写 slot 64） | `tcgen05_helpers.cpp:23` | `fix-tcgen05-multi-warp-fragment` (已 propose) |

## Test Coverage Improvement (per Oracle 审计)

| ID | Gap | Fix (in this change) |
|----|-----|---------------------|
| **B1** | K=128 累加无测试（T1 仅 2 次） | `T1_k_loop_4` 新增 TC（4 次 partial） |
| **B2** | `mma → commit → wait → mma` 序列**零行为测试**（grep 仅 parse 测试） | `tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp` 新增 |
| **B7** | `Catch::Approx` 默认 epsilon 1.19e-5 太松，K=128 累加误差无法早期发现 | 收紧到 `.epsilon(1e-6)` |

## Lessons Learned (5 entries)

### L1: Helper 累加器 "single-warp execution" 是脆弱假设

**问题**: `tcgen05_fragment_mma_f16` helper 假设 single-warp 执行（per SM scheduler 的顺序调度保证）。FlashAttention 多 warp 协作时 `c_slot = 64 + lane_id` 让 warp 0 和 warp 1 都写 slot 64 → 数据竞争（C4）。

**经验**:
- "Currently safe because SM scheduler runs one warp at a time" 这类注释是**已知 debt**的标记，必须在 helper header 显式标注 `[SINGLE-WARP ASSUMPTION]`
- 新增累加路径时必须考虑多 warp → 扩展 helper 接受 warp_id 或显式拒绝
- 单元测试用 `SMContext(1 warp, 32, 1 cta)` 是 single-warp 配置，多 warp 必须独立测试

**诊断**: `grep -rn "single-warp\|one warp at a time\|sequential execution" src/ include/`

**修复模板**: FU-4 `fix-tcgen05-multi-warp-fragment` — `c_slot = warp_id * 32 + 64 + lane_id`

### L2: TcQueue wait() 必须先检查 commit_group_counter

**问题**: `TcQueue::wait(warp, lane_id, group_id)` 先 push 到 `pending_waiters_` 再检查 counter，导致 commit→wait 序列后 `pending_count()` 返回 1（waiter 仍在 list 中）。

**经验**:
- TcQueue 状态机: commit bumps counter, wait checks counter — 但 wait 必须**先** check counter 再 push
- B2 integration test 第一次跑时 `pending_count() == 0` 断言暴露此问题

**诊断**: `grep -n "wait\|pending_waiters_" src/ptxsim/async/tc_queue.cpp`

**修复模板** (FU-1):
```cpp
void wait(...) {
    { std::lock_guard lock(mutex_);
      if (commit_group_counter_ >= group_id) return; }
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this, group_id]{ return commit_group_counter_ >= group_id; });
}
```

### L3: PTX §9.7.16 f16×f16→f32 不变量 — storage format 必须硬件对齐

**问题**: Helper 改 `c_frag` 为 `float` 后，readback 站点未同步迁移，f32 bits 被当 f16 bits 读 → 垃圾值。

**经验**:
- Helper 输出 dtype 是 hardware contract，golden header 必须声明 storage format
- `grep "c_buf[idx * 2]" tests/` 是 readback 残留快速检测
- `Catch::Approx` 默认 epsilon 对 storage format 错误不敏感

**诊断**: `grep -rn "f16_to_f32\|c_buf\[idx \* 2\]" tests/integration/tcgen05/`

### L4: ANTLR extractQualifiersFromContext 丢失 IMMEDIATE 值

**问题**: `extractQualifiersFromContext` 只映射 terminal token 到 `Qualifier` enum，`IMMEDIATE` 节点被 `tokenToQualifier` 返回 `Q_UNKNOWN` 后静默丢弃。`instr.cta_group` 永远 defaults to 1。

**经验**:
- 被 19 个 call sites 调用，改返回类型会破坏所有 caller
- 需要 IMMEDIATE 的 caller 必须**单独 walk parse tree**

**诊断**: `grep -n "Q_UNKNOWN" src/ptx_parser/ptx_visitor.cpp` + `grep -rn "extractQualifiersFromContext" src/`

### L5: Type 判断依赖 qualifiers.back() 风险（强化 ptx-lessons-learned §5）

本 change 未触发现象，但在 Oracle 审计中发现同样的脆弱模式（参见 lessons-learned §16 完整案例）。此风险已在 ptx-lessons-learned §5 中记录。

## Process Lessons

### P1: Phase-based commit 救了一次
- 严格 2-Phase 拆分（H1 → H2 → ADR → artifacts）让每个修复独立可 revert
- H2 commit 时 `commit_wait_sequence` 测试暴露一处漏 readback（`require_c_slot_matches`）— Phase-based 强制 build+test 捕获
- 无 Phase commit 时这类 bug 会混入后续 commit 污染历史

### P2: Baseline worktree + artifacts-first commit
- baseline worktree 节省了 "失败是基线的还是我的" 争论
- artifacts-first commit（per lessons-learned §6）避免后续 FU changes 误判 4 条 P0-A 为 active debt

### P3: TDD RED 阶段捕获了 helper signature 不存在的事实
- 写 4 个测试 → 5 个 compile error（`too many arguments to function call`）— 证明 RED 是真 RED 不是 typo
- 没有 RED 阶段的话 AI 容易写出"凑巧通过"的实现

## Commits (chronological)

| Hash | Phase | Description |
|------|-------|-------------|
| d3be589 | step 1 (precursor) | test(tcgen05): add multi-op TMEM persistence integration test |
| df1f6de | Phase 1 (H1) | fix(tcgen05): add accumulate parameter to fragment_mma_f16 helper |
| f97863c | Phase 2 (H2) | fix(tcgen05): store mma C output as f32 per PTX §9.7.16 |
| 3235227 | Phase 3 (ADR) | docs(adr): ADR-0016 postmortem H1+H2 |
| 58cbff9 | Phase 3 (artifacts) | docs(openspec): fix-tcgen05-mma-accumulator-and-f32-storage artifacts |

## References
- Oracle 2026-07-10 审计 session: `ses_0b3791d78ffewb52428kJJ2Irz` (5 blockers)
- Oracle 2026-07-10 API 审查: `ses_0b026333bffePgrqVq7PDJNeR1` (idesc=RegOperand)
- Oracle 2026-07-11 审计 (C1-C4 BLOCKER): session context (per previous conversation)
- Metis pre-impl review: `ses_0b1a0cdb1ffenbhbciQ1n0x236` (CONDITIONAL GO, 3 MUST-RESOLVE)
- ADR-0016 §2026-07-11 Postmortem H1+H2
- ptx-lessons-learned SKILL.md