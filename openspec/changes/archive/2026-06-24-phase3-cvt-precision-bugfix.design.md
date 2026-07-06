**Retroactive synthesis from git log — not an original design document**
> 合成日期: 2026-07-06
> 来源: `proposal.md` (Bug 1 + Bug 2 详述) + `tasks.md` (Task 1-3 TDD) + git log (commit `32ce8a0` 等) + 关联 change `phase3-t2-6-cvt-strategy-pattern` (commit `d3c77b5` 抽离 helpers) + 关联 change `phase3-half-precision-bugfix` (同源 bug)

# Phase 3 CVT Precision Bugfix — Retroactive Design

## Context

T2-6 Step 1（commit `d3c77b5`，"extract 4 helpers to cvt_helpers"）在 zero behavior change 约束下，把 `arithmetic_conversion.cpp` 顶部 4 个 inline helper（`round_half_to_even` / `half_to_float` / `float_to_half` / `should_saturate_uint32`）原样抽离到独立 `cvt_helpers.{h,cpp}`。这次抽离**保留**了 2 个历史 bug。Step 1 agent 通过对比 `include/ptxsim/utils/half_utils.h::f16_to_f32` 确认：`cvt_helpers.cpp::half_to_float` 的 denormal 路径错（返回 `+2^102` 而非 `2^-24`），`should_saturate_uint32` 用严格 `<` 而非 `<=`（边界值静默不饱和）。这些 bug 在原 1288 行单文件里"隐藏"在嵌套 switch 内层，单独 helper 抽离后才显形。本 change 修复这 2 个 bug，恢复 PTX `cvt.f32.f16` denormal 与 `cvt.u32.f32.sat` 边界正确性。

## Goals / Non-Goals

**Goals:**
- 修复 `cvt_helpers.cpp::half_to_float` denormal 路径，对齐 `half_utils.h::f16_to_f32` 正确算法
- 修复 `cvt_helpers.cpp::should_saturate_uint32` 边界判断 (`<` → `<=`)
- 恢复 `tests/unit/ptx/test_cvt_helpers.cpp` 中被 Step 1 故意宽松的 denormal 断言
- 新增 `tests/integration/ptx/test_cvt_edge_cases.cpp` 端到端覆盖

**Non-Goals:**
- T2-6 Step 2-6 推迟到本 change 完成后继续
- 其他 CVT 指令族的精度/边界修复（独立审计）
- `half_utils.h::f16_to_f32` 自身精度问题（与本 change 同源，独立 change `phase3-half-precision-bugfix`）

## Decisions

### Decision 1: `half_to_float` denormal 路径修复算法

**问题**: denormal (exp=0, mantissa≠0) 路径在 `arithmetic_conversion.cpp::half_to_float` 与 `cvt_helpers.cpp::half_to_float` 都有 bug。原代码把 denormal 当 normal 处理：exp 用 `127 - 15 = 112`，循环左移 mantissa 10 次到 bit 10（`mantissa=1` → `0x400`），最后 `exp + 127 = 229`，输出 `+2^102`。对 half 最小 denormal `0x0001` 返回 `+2^102`（错），正确应是 `2^-24`。

**方案分析**:
- **方案 A**: 维持原循环结构，改正指数偏移。原代码 `exp = 127 - 15` 是 normal 路径的偏移，denormal 应该用 `127 - 24 = 103`（half bias 15, mantissa 是 10-bit，所以 `mantissa × 2^-24`）。
- **方案 B**: 完全重写 denormal 路径，用 union 位操作直接构造 float32（`bits.u = (sign << 31) | (103 << 23) | (mantissa << 13)`）。
- **方案 C**: 删除 `cvt_helpers.cpp::half_to_float` 本地实现，改用 `half_utils.h::f16_to_f32`（零重复代码）。

**选择**: **方案 A**（直接修正 denormal 路径，最小侵入，保持 helpers 独立存在）。原因: Step 2 的 "复用 `half_utils.h`" 是独立 sub-task，**不在本 change 范围**。本 change 仅修复 bug，不重构。

**证据**: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float` 修复 by commit `32ce8a0` "fix(cvt): correct half_to_float denormal path (PTX cvt.f32.f16 edge case)"。Task 1 Step 4 伪代码参考: `f = (sign << 31) | (103 << 23) | (mantissa << 13)`，与 `half_utils.h::f16_to_f32` 的正常实现对齐。

### Decision 2: `should_saturate_uint32` 边界修复

**问题**: 原实现 `return temp >= 4294967295.0f && temp < sat_high;`。当 `temp == sat_high == 4294967295.0f` 时返回 false，违反 PTX `.sat` 边界语义。注意 IEEE 754 精度问题：`4294967295.0f` 在 float32 中被舍入为 `4294967296.0f`，所以 `>= 4294967295.0f` 已经包含 `4294967296.0f` 的值。

**方案分析**:
- **方案 A**: 仅改 `<` 为 `<=`。最小修改。
- **方案 B**: 用 `std::nextafter` 避免 float 精度问题。过度工程化。

**选择**: **方案 A**。修复后 `temp <= sat_high` 正确捕获边界相等。

**证据**: commit `32ce8a0` 同 commit 内一并修复。

### Decision 3: 测试恢复策略

**问题**: Step 1 (commit `d3c77b5`) 抽离 helpers 时为"零行为变更"曾故意宽松 denormal 断言（`REQUIRE(denorm < 1e-7f)` 而非严格值）。本 change 恢复严格 IEEE 754 行为断言。

**方案分析**:
- **方案 A**: 修改 `tests/unit/ptx/test_cvt_helpers.cpp`，删除宽松断言，加严格值断言 (`5.9604644775390625e-08f`)
- **方案 B**: 新增严格断言文件，保留宽松断言。重复覆盖。

**选择**: **方案 A**。直接恢复正确断言（与 TDD "测试即规范" 一致）。

**证据**: `tests/unit/ptx/test_cvt_helpers.cpp` 修改 by commit `32ce8a0`。

## Implementation Commits

> **注**: 以下 commits 在 change 归档时已合并到 main，本节为追溯原始实施链。

| Commit | Sub-task | 摘要 |
|--------|----------|------|
| `d3c77b5` | T2-6 Step 1 (prereq) | `refactor(cvt): extract 4 helpers to cvt_helpers` (zero behavior change, 保留 bug) |
| `32ce8a0` | Task 1 + Task 2 + Task 3 | `fix(cvt): correct half_to_float denormal path (PTX cvt.f32.f16 edge case)` — 修 denormal + 修 `should_saturate_uint32` 边界 + 恢复单元测试 + 新增 integration test |
| `ccbbe2a` | Archive | `chore(openspec): archive completed Phase 3 changes (CVT, half-precision, T2-1, T2-6)` |
| `3006e11` | Orphan README (后续) | `docs(openspec): add READMEs to 6 orphan archive changes (Fix #4)` |

## Risks / Trade-offs

| 风险 | 缓解（per proposal.md 风险表）|
|------|------|
| 修复 `half_to_float` 改变 denormal 之外的 behavior | 完整 5 TEST_CASEs 回归 + 16/16 CVT/PTX 集成测试 |
| `should_saturate_uint32` 改 `<=` 后破坏其他路径 | 改后跑全量 `ctest -L "ptx;cvt"` + `sanity.sh --quick` |
| `half_utils.h::f16_to_f32` 本身也有 bug（与本 change 同源）| 修复前对比测试 `half_utils.h` vs `half_to_float` denormal 输出；同源 bug 由独立 change `phase3-half-precision-bugfix` 处理 |

## Cross-References

- 原 artifacts: `openspec/changes/archive/2026-06-24-phase3-cvt-precision-bugfix/{proposal.md,tasks.md,README.md}`
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` (T2-6 重构暴露本 bug)
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-half-precision-bugfix/` (同源 bug 修复)
- Lessons-Learned: §16 (类型判断只看 `qualifiers.back()` 失败模式 — `is_float_type` 修复虽不同根因，但 IEEE 754 边界修复同 TDD 流程)
- 正本算法参考: `include/ptxsim/utils/half_utils.h::f16_to_f32` (正确实现)
- IEEE 754 half precision: denormal = `mantissa × 2^-14` (half bias 15)

## Notes

> 本文件为 retroactive synthesis，最佳努力重建。如发现与原 commit body 不一致，**以原 commit body 为准**。
> 任何修改归档目录内文件的尝试被禁止（per Checklist G + Decision 1）。
> 
> **实施时长**: 0.3-0.5 天 (per tasks.md 顶部) | **实际**: 单 commit `32ce8a0` 完成 (bug 修复与测试恢复合并提交)
> 
> **遗留**: `half_utils.h::f16_to_f32` 自身仍携带同源 denormal bug，已由 `phase3-half-precision-bugfix` (commit `fbb7a29`) 协同修复。
