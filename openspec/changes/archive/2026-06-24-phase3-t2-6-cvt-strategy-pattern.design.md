**Retroactive synthesis from git log — not an original design document**
> 合成日期: 2026-07-06
> 来源: `proposal.md` (1288 行 god class + 5 策略) + `tasks.md` (Sub-task 1-7 TDD) + git log (commits `86e0786`, `d3c77b5`, `2f3c150`, `620d066`, `edbce54`, `fc3c352`, `9837d44`, `d6123e0`, `204b5cd`, `40b331b`, `8440f49`) + ADR-0015 (CVT 策略模式)

# Phase 3 T2-6: CVT 策略模式重构 — Retroactive Design

## Context

`src/ptxsim/instructions/arithmetic_conversion.cpp` 是 PTX-EMU 最长单 handler 实现（1288 行），`CvtHandler::processOperation` 单方法占 1145 行（line 141-1286），核心逻辑在嵌套 4 层深的 `switch (dst_bytes)` (line 224-1284, 1063 行) 中。`half_to_float` / `float_to_half` 与 `half_utils.h::f16_to_f32` / `f32_to_f16` 重复 ~70 行。int → int 4×4×4 嵌套逻辑在 4 个 case 中几乎相同模板。

维护成本：每次 CVT bug 修复需在 4 个 case 中同步修改，新指令族扩展需复制整个 case 块。审计 P2-1 估计 3 天，本计划评估 3.5 天（含测试补全 + P1-4.1 协同）。

## Goals / Non-Goals

**Goals:**
- 提取 4 个 inline helper 到 `include/ptxsim/instructions/cvt/cvt_helpers.{h,cpp}`
- 复用 `half_utils.h` (commit `2f3c150`)，删除 ~70 行重复代码
- 策略模式重构（**Composition 而非 Inheritance**）: `IntToIntStrategy` / `IntToFloatStrategy` / `FloatToFloatStrategy` / `FloatToIntStrategy` / `RoundingMode`
- 新增 94 个 integration tests
- P1-4.1 bug 修复（f32→s32 / f64→s64 路径补 r2 写入）
- 写 ADR-0015 (`docs/adr/ADR-0015-cvt-strategy-pattern.md`)

**Non-Goals:**
- CvtHandler 类拆分（X-Macro 约束禁止）
- CVT 指令的精度修复（属于独立 change `phase3-cvt-precision-bugfix` + `phase3-half-precision-bugfix`）
- WMMA / Tensor Core CVT 变体
- T2-4 (PTX 8.7+ 占位清理) — 留待 Phase 4
- `GeneralCvtStrategy::convert()` god class (919 行) — 后续 C7 处理

## Decisions

### Decision 1: Composition 而非 Inheritance (策略模式)

**问题**: 1288 行单文件如何拆分?

**方案分析**:
- **方案 A**: Composition — 保留 `CvtHandler` 单类（X-Macro 约束），内部用 `ConversionStrategy` 接口 + 5 个具体策略实例。`processOperation` 改为 `select_strategy(ctx).convert(...)`。
- **方案 B**: Inheritance — 拆 `CvtFloatToIntHandler` / `CvtIntToFloatHandler` 等多 handler 类，每个独立 X-Macro 注册。**违反 X-Macro 约束** (`instruction_factory.cpp:14-17` 实例化 `new CvtHandler()`)。
- **方案 C**: 简单函数化拆分 — 拆 helper 函数 + 用表驱动 `std::map<QualKey, ConvertFunc>`。不解决 `CvtContext` 状态传递问题。

**选择**: **方案 A**。X-Macro 约束关上了"拆 Handler"大门，必须用 Composition。

**证据**: ADR-0015 §关键约束 "DO NOT 拆 CvtHandler 类：X-Macro 在 `instruction_factory.cpp:14-17` 实例化 `new CvtHandler()`，多 Handler 会改变注册机制"。

### Decision 2: helpers 抽离 + 复用 half_utils.h 的两步走

**问题**: 4 个 inline helper 抽离后是否立即复用 `half_utils.h`?

**方案分析**:
- **方案 A** (实施路径): Step 1 (commit `d3c77b5`) 抽离 helpers 到 `cvt_helpers.cpp`，**保留本地 `half_to_float` / `float_to_half`**（零行为变更）。Step 2 (commit `2f3c150`) 才复用 `half_utils.h`。分两步走。
- **方案 B**: Step 1 直接复用 `half_utils.h`。一步完成，但绑死两个 change（`phase3-half-precision-bugfix` 必须先合并）。
- **方案 C**: 永远不复用，保留两份实现。重复代码持续存在。

**选择**: **方案 A**。Step 1 仅做 helper 抽离 + 零行为变更，Step 2 等 `phase3-half-precision-bugfix` 修复 `half_utils.h` 同源 bug 后再复用。

**证据**: tasks.md Sub-task 1 Step 6 "第一版用本地实现，验证零行为变更" + Sub-task 2 显式作为独立 sub-task。

### Decision 3: 5 个策略分类边界

**问题**: 策略模式如何切分 4×4×4 嵌套 switch?

**方案分析**:
- **方案 A** (实施路径): 5 策略:
  - `IntToIntStrategy` (最大复用价值，4×4×4 模板化)
  - `IntToFloatStrategy`
  - `FloatToFloatStrategy` (最简单，~30 行)
  - `FloatToIntStrategy` (含 .sat / 5 种舍入 / 默认 truncate)
  - `RoundingMode` 工具类（独立头 `cvt_rounding.h`）
- **方案 B**: 按 5 舍入模式 + signed/unsigned 切分。粒度太细。
- **方案 C**: 简单 2 策略 (Float↔Int / Int↔Float)。失去精度处理能力。

**选择**: **方案 A**。

**证据**: proposal.md §What Changes 详述 + ADR-0015 §Maintenance 表格 "CvtStrategy 数量 = 5"。

### Decision 4: P1-4.1 修复与策略模式同 PR

**问题**: P1-4.1 bug (f32→s32 / f64→s64 路径缺 r2 写入) 何时修?

**方案分析**:
- **方案 A**: 与 Sub-task 5 协同提交 (commit `204b5cd`)。一次 PR 包含 94 个新测试 + bug 修复。
- **方案 B**: 独立 change 修复 P1-4.1。过度拆分。
- **方案 C**: 推迟到 C7 (后续 `GeneralCvtStrategy::convert()` 拆分时)。延迟 bug 修复。

**选择**: **方案 A**。TDD 协同效应 — 94 个新测试自然覆盖 P1-4.1 路径。

**证据**: tasks.md Sub-task 5 "P1-4.1 修复与本 change 同 PR 提交" + commit `204b5cd` "feat(cvt): P1-4.1 fix + 94 new integration tests"。

## Implementation Commits

> **注**: 以下 commits 在 change 归档时已合并到 main，本节为追溯原始实施链。

| Commit | Sub-task | 摘要 |
|--------|----------|------|
| `86e0786` | Sub-task 1 Red | `test(cvt): add 5 helper unit tests for T2-6 (TDD Red phase)` |
| `d3c77b5` | Sub-task 1 Green | `refactor(cvt): extract 4 helpers to cvt_helpers (T2-6 Step 1)` — 暴露 denormal bug |
| `fbb7a29` | `phase3-half-precision-bugfix` 协同 | `fix(half-utils): correct f16_to_f32 denormal path` — 解封 Step 2 prereq |
| `2f3c150` | Sub-task 2 | `refactor(cvt): delegate half_to_float/float_to_half to half_utils.h` |
| `620d066` | Sub-task 3 Red | `test(cvt): add 10 CvtContext unit tests for T2-6 Sub-task 3 (TDD Red phase)` |
| `edbce54` | Sub-task 3 Green | `refactor(cvt): add CvtContext + select_strategy() skeleton` |
| `fc3c352` | Sub-task 4a | `refactor(cvt): add FloatToFloatStrategy` |
| `9837d44` | Sub-task 4d+4e | `refactor(cvt): add IntToIntStrategy + wire up 5 strategies` |
| `d6123e0` | Sub-task 4 follow-up | `test(cvt): add FloatToIntStrategy + IntToFloatStrategy unit tests` |
| `204b5cd` | Sub-task 5 | `feat(cvt): P1-4.1 fix + 94 new integration tests (T2-6 Sub-task 5)` |
| `40b331b` | Sub-task 6 | `refactor(cvt): delete arithmetic_conversion.cpp` (1288 行 → 0) |
| `8440f49` | Sub-task 7 | `docs(adr): record CVT strategy pattern decision + mark T2-6 complete` (写 ADR-0015) |
| `ccbbe2a` | Archive | `chore(openspec): archive completed Phase 3 changes` |
| `3006e11` | Orphan README (后续) | `docs(openspec): add READMEs to 6 orphan archive changes (Fix #4)` |

## Risks / Trade-offs

| 风险 | 缓解（per proposal.md 风险 + tasks.md 验证门禁）|
|------|------|
| Composition 而非 Inheritance 受 X-Macro 约束 | ADR-0015 记录决策理由 + `select_strategy()` 工厂模式解耦 |
| Step 1 抽离 helpers 暴露 denormal bug | 独立 change `phase3-cvt-precision-bugfix` (commit `32ce8a0`) 修复 |
| Step 2 复用 `half_utils.h` 受 half precision bug 阻塞 | 独立 change `phase3-half-precision-bugfix` (commit `fbb7a29`) 修复 |
| `IntToIntStrategy` 4×4×4 模板化代码复杂 | 4a → 4d → 4e 渐进实施，每步独立 commit 验证 |
| P1-4.1 bug 漏修 | 与 Sub-task 5 同 PR (commit `204b5cd`)，94 个新测试自然覆盖 |
| `GeneralCvtStrategy::convert()` 仍 919 行 god class | 后续 C7 处理（详见 `docs/audits/debt-audit-2026-07-02.md` §3.4）|

## Cross-References

- 原 artifacts: `openspec/changes/archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/{proposal.md,tasks.md,README.md}`
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-cvt-precision-bugfix/` (Step 1 暴露的 denormal bug)
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-half-precision-bugfix/` (Step 2 复用的 half_utils 修复)
- ADR-0015: CVT 策略模式 (本 change 实施产生) — `docs/adr/ADR-0015-cvt-strategy-pattern.md`
- ADR-0009: X-Macro + Weak Symbol 指令分发模式 (Composition 约束根因)
- Lessons-Learned: §6 (类型一 vs 类型二测试发现能力差异 — 5 helpers 单元测试 + 94 integration tests 三层覆盖)
- 审计原文: `docs/audits/HEALTH-AUDIT-2026-06-21.md` (P2-1 触发本 change)
- 后续债务: `docs/audits/debt-audit-2026-07-02.md` §3.4 (`GeneralCvtStrategy::convert()` 919 行 C7)

## Notes

> 本文件为 retroactive synthesis，最佳努力重建。如发现与原 commit body 不一致，**以原 commit body 为准**。
> 任何修改归档目录内文件的尝试被禁止（per Checklist G + Decision 1）。
>
> **实施时长**: 3.5 天 (per proposal.md 顶部) | **实际**: 11 commits on main (Sub-task 1-7 全完成)
>
> **完成度评估**:
> - ✅ 7 个 Sub-task 全部完成（archive 通过）
> - ✅ 4 个 helper 抽离 + 复用 `half_utils.h`
> - ✅ 5 个策略实施 + Composition 模式
> - ✅ 94 个新 integration tests + 1 bug fix (P1-4.1)
> - ✅ ADR-0015 落地
> - ⚠️ **遗留**: `GeneralCvtStrategy::convert()` 919 行 god class — C7 candidate (后续 change)
>
> **核心架构成果** (per ADR-0015 量化):
> - `CvtHandler::processOperation` 行数: 1145 → 31 (Sub-task 3) → 0 (Sub-task 6)
> - `arithmetic_conversion.cpp` 总行数: 1288 → 0 (Sub-task 6 删除)
> - CVT 单元测试数: 1 → 6 unit + 8 integration + 1 cvta = 15
> - CVT 集成测试断言数: ~30 → 94 + P1-4.1 fix 启用
