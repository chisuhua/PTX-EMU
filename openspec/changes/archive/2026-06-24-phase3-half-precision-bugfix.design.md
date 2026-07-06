**Retroactive synthesis from git log — not an original design document**
> 合成日期: 2026-07-06
> 来源: `proposal.md` (Bug 1 + Bug 2 同源) + `tasks.md` (Task 1-2 TDD) + git log (commit `fbb7a29` 协同修复) + 关联 change `phase3-cvt-precision-bugfix` (正本算法参考) + 关联 change `phase3-t2-6-cvt-strategy-pattern` (commit `2f3c150` 委托)

# Phase 3 Half Precision Bugfix — Retroactive Design

## Context

T2-6 Step 1 (commit `d3c77b5`) 抽离 CVT helpers 时，agent 通过对比 `include/ptxsim/utils/half_utils.h::f16_to_f32` 与新抽离的 `cvt_helpers.cpp::half_to_float`，发现两者**同源 bug**：denormal 路径都错（历史 bug 在 `arithmetic_conversion.cpp` 内部被复制粘贴到 `half_utils.h`，再被 Step 1 抽离时原样保留）。`phase3-cvt-precision-bugfix` (commit `32ce8a0`) 修复了**本地** `cvt_helpers.cpp::half_to_float`，但**未触碰** `half_utils.h` 源文件。本 change 修复 `half_utils.h` 源文件中的同源 denormal bug，恢复所有 `f16 ↔ f32` 调用方的正确性。`half_utils.h` 是项目内任何 f16 转换的共用工具，调用方包括 CVT 路径之外的模块（如 half-precision ops、quantization 等）。

## Goals / Non-Goals

**Goals:**
- 修复 `half_utils.h::f16_to_f32` denormal 路径（与 `cvt_helpers.cpp::half_to_float` 同源）
- 修复 `half_utils.h::f32_to_f16` denormal 路径（Task 1 Step 2 调研确认后）
- 新增 `tests/unit/utils/test_half_utils.cpp` 覆盖 denormal + Inf + NaN + 边界
- 新增 `tests/unit/utils/test_half_utils_consistency.cpp` 验证 `half_utils.h` ↔ `cvt_helpers.cpp` bit-perfect 一致

**Non-Goals:**
- `cvt_helpers.cpp::half_to_float`（已在 `phase3-cvt-precision-bugfix` 修）
- 改 API 签名（保持 `inline uint16_t f32_to_f16(float)` 等）
- 性能优化（不引入 SIMD / 查找表）

## Decisions

### Decision 1: 修复算法来源

**问题**: `half_utils.h::f16_to_f32` 的 denormal 路径与 `cvt_helpers.cpp::half_to_float` 的历史 bug 同源。**任何重新发明算法的尝试都是反模式** — 容易引入新 bug。

**方案分析**:
- **方案 A**: 参考已验证正确的 `cvt_helpers.cpp::half_to_float` (commit `32ce8a0`) 算法，**一字不差**移植到 `half_utils.h`。
- **方案 B**: 独立设计新算法。
- **方案 C**: 委托 `half_utils.h` ↔ `cvt_helpers.cpp` 互相调用，消除两份实现。

**选择**: **方案 A**（提交实施时实际为方案 A + 后续 commit `2f3c150` 走方案 C 的过渡）。原因: T2-6 Step 2 计划 "复用 `half_utils.h`" 是独立 sub-task。本 change 仅修复 bug，**严格采用方案 A**，把 `cvt_helpers.cpp::half_to_float` 验证过的算法移植到 `half_utils.h::f16_to_f32`。后续 T2-6 Step 2 (commit `2f3c150`) 才实施"委托"消除重复。

**证据**: 修复算法来自 `cvt_helpers.cpp::half_to_float` (commit `32ce8a0`)，Task 1 Step 5 显式声明 "必须参考 ... 正确算法，不要重新发明"。

### Decision 2: `f32_to_f16` denormal 调研策略

**问题**: proposal 假设 `f32_to_f16` 也可能携带同源 bug，但需 Task 1 Step 2 调研确认。

**方案分析**:
- **方案 A**: 在 Task 1 Step 2 调研时**直接看实现**，如发现 bug 一并修复。
- **方案 B**: 拆为独立 change。过度拆分。

**选择**: **方案 A**。一次调研 + 一次修复，最大化 commit 价值。

**证据**: proposal.md §Bug 2 + tasks.md Task 1 Step 2 "对比 `f32_to_f16` 实现与 `cvt_helpers.cpp::float_to_half`（如有）。如确认有 bug，一并记录到 proposal"。

### Decision 3: 双向一致性测试

**问题**: 修复后 `half_utils.h::f16_to_f32` 与 `cvt_helpers.cpp::half_to_float` 必须行为一致 (bit-perfect)，为后续 T2-6 Step 2 委托创造前提。

**方案分析**:
- **方案 A**: 写 65536 case 全 half 值循环对比（`for (uint16_t h = 0; h <= 0xFFFF; h++)`），用 union 强制位级比较，NaN 也 bit-perfect。
- **方案 B**: 抽样几个 case。覆盖不足。

**选择**: **方案 A**。65536 case 跑过即穷尽验证。

**证据**: `tests/unit/utils/test_half_utils_consistency.cpp` by commit `f9238cd` "test(half-utils): verify half_utils.h consistent with cvt_helpers.cpp"。

## Implementation Commits

> **注**: 以下 commits 在 change 归档时已合并到 main，本节为追溯原始实施链。

| Commit | Sub-task | 摘要 |
|--------|----------|------|
| `d3c77b5` | T2-6 Step 1 (关联 prereq) | `refactor(cvt): extract 4 helpers to cvt_helpers` (暴露同源 bug) |
| `32ce8a0` | `phase3-cvt-precision-bugfix` (关联 prereq) | `fix(cvt): correct half_to_float denormal path` (提供正本算法) |
| `fbb7a29` | Task 1 + Task 2 | `fix(half-utils): correct f16_to_f32 denormal path` — 移植 `cvt_helpers.cpp` 算法到 `half_utils.h::f16_to_f32` + 修复 `f32_to_f16` 同源 bug (调研后) |
| `f9238cd` | Task 2 | `test(half-utils): verify half_utils.h consistent with cvt_helpers.cpp` — 65536 case bit-perfect 一致性测试 |
| `2f3c150` | T2-6 Step 2 (后续) | `refactor(cvt): delegate half_to_float/float_to_half to half_utils.h` (本 change 解封 Step 2) |
| `ccbbe2a` | Archive | `chore(openspec): archive completed Phase 3 changes` |
| `3006e11` | Orphan README (后续) | `docs(openspec): add READMEs to 6 orphan archive changes (Fix #4)` |

## Risks / Trade-offs

| 风险 | 缓解（per proposal.md 风险表）|
|------|------|
| 修复破坏 `f32_to_f16` 的正常范围（如最大 half normal）| Task 1 调研后跑全量 `ctest -L "ptx;cvt"` + `sanity.sh --quick` |
| 修复后 `half_utils.h` 与 `cvt_helpers.cpp::half_to_half` 行为不一致 | Task 1 末对比测试（65536 case bit-perfect）|
| `half_utils.h` 有调用方已依赖 buggy 行为 | 行为违反 IEEE 754，依赖视为错误；dev 角度 0 用户 |
| T2-6 Step 2 后续复用变可执行 | 本 change 完成即解封 Step 2 (commit `2f3c150`) |

## Cross-References

- 原 artifacts: `openspec/changes/archive/2026-06-24-phase3-half-precision-bugfix/{proposal.md,tasks.md,README.md}`
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-cvt-precision-bugfix/` (同源 bug 修复，提供正本算法)
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` (T2-6 Step 2 委托 commit `2f3c150`)
- Lessons-Learned: §6 (类型一 vs 类型二测试发现能力差异 — 一致性测试 (类型一) 直接锁住两实现 bit-perfect 等价)

## Notes

> 本文件为 retroactive synthesis，最佳努力重建。如发现与原 commit body 不一致，**以原 commit body 为准**。
> 任何修改归档目录内文件的尝试被禁止（per Checklist G + Decision 1）。
>
> **实施时长**: 0.3 天 (per tasks.md 顶部) | **实际**: 2 commits (`fbb7a29` + `f9238cd`) 完成
>
> **TDD 三阶段**: Task 1 Step 3-6 (Red → Green → refactor) + Task 2 Step 1-3 (一致性双向锁)
