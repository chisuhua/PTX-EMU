## Why

[Stale artifact 修复] `openspec/changes/archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` 归档时（commit `ccbbe2a`，2026-06-24）遗漏了 `GeneralCvtStrategy::convert()` god class 的实际拆分，archive 中 4 个 Strategy 类（`FloatToFloatStrategy` / `FloatToIntStrategy` / `IntToFloatStrategy` / `IntToIntStrategy`）已在 commit `fc3c352`/`9837d44`/`d6123e0` 实际部署并被 `select_strategy()` dispatch（`src/ptxsim/instructions/cvt/cvt_strategy.cpp:1034-1046`），但 `GeneralCvtStrategy` 类本身（line 104-1031，~920 行）保留在文件中成为死代码。

**根因**（Metis pre-implementation review 揭示）：原 proposal 假设"919 行 switch 块未拆分"，与代码现实矛盾。实际上 4 个 Strategy 类已实施并生效，差异仅在于死代码未清理 + 文件头注释未更新。

**2026-07 debt audit 误标记**（`.opencode/notes/debt-audit-2026-07-02.md` §P0-C1 — 实际位于 `docs/audits/debt-audit-2026-07-02.md`）将 919 行 switch 标为 active debt，本 change 修复该误标记。

这是 lessons-learned §6 的"stale artifact"真实案例：`archive/<date>-<name>/` 终态后，任何修补需求应**新建 fix-* change + Ref 链接**，禁止 amend 已归档 change（OpenSpec Checklist G lifecycle 约束）。

## What Changes

**核心变更**：

- **删除 `GeneralCvtStrategy` god class**（`src/ptxsim/instructions/cvt/cvt_strategy.cpp:104-1031`）：~920 行死代码（class 定义 + convert() switch 块 + name() override）
- **修复 stale 文件头注释**（`cvt_strategy.cpp:1-16`）：将"Sub-task 4 将 GeneralCvtStrategy::convert() 拆为 5 个具体策略"改为实际状态描述（4 Strategy 类已就位）
- **同步 4 个文档**：
  1. `docs/audits/debt-audit-2026-07-02.md §P0-C1` — 状态 active → ✅ RESOLVED（引用新 commit）
  2. `docs/adr/ADR-0015-cvt-strategy-pattern.md` — 追加"2026-07 Fix: 死代码清理"段
  3. `src/ptxsim/instructions/cvt/README.md` 或等效 — 说明 4 个活 Strategy 文件的角色
  4. `src/ptxsim/instructions/AGENTS.md` STRUCTURE — 更新 cvt/ 目录清单

**显式标记**：本 change 是 `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` 的修补（**非 amend**），不修改原 archive 内容，仅通过 `Ref:` 链接建立 lineage。

**预估代码改动量**：~30 行删除（`GeneralCvtStrategy` 类） + ~30 行文档同步 = **约 60 行净改动**。

## Capabilities

### New Capabilities

无（不引入新功能）。

### Modified Capabilities

无（spec-level 行为零变化 — `select_strategy()` 在修改前后返回相同 4 个 Strategy 实例之一，`CvtContext` / `ConversionStrategy` 接口不变）。

## Impact

**受影响的代码/文件**：

- `src/ptxsim/instructions/cvt/cvt_strategy.cpp`（1061 行 → ~140 行 dispatcher）— 删除 line 104-1031
- `docs/audits/debt-audit-2026-07-02.md §P0-C1` — 状态更新
- `docs/adr/ADR-0015-cvt-strategy-pattern.md` — 追加 Fix 段
- `src/ptxsim/instructions/AGENTS.md` 或 `cvt/README.md` — STRUCTURE 段

**未受影响（明确列出以避免误改）**：

- `src/ptxsim/instructions/cvt/cvt_strategy.h` — 接口不变（`ConversionStrategy` / `CvtContext` / `select_strategy` 返回类型 `const ConversionStrategy&` 保持）
- 4 个活 Strategy 文件（`cvt_int_to_float.cpp` / `cvt_float_to_float.cpp` / `cvt_int_to_int.cpp` / `cvt_float_to_int.cpp` / `cvt_helpers.cpp`）— 不修改
- `arithmetic_conversion.cpp` — 已由 commit `40b331b` 删除，不在范围

**回归风险**：极低。`GeneralCvtStrategy::convert()` 是死代码（`grep` 0 external callers），删除不会导致链接错误或运行时行为变化。

**回归 oracle 验证**：

1. 现有 14 个 CVT 测试（6 unit + 8 integration）全部 PASS — 通过 ctest `-L "unit;cvt"` + `-L "integration;cvt"` 验证
2. `./tests/ptx/test_all_ptx.sh` 全套 PTX 语法测试 PASS（不变性 oracle）
3. `ctest -R e2e_blackwell_gemm` PASS — 关键 kernel oracle

**注**：原 proposal 引用"94 个 integration 测试"为虚构 oracle（实际为 14 个）。本 change 用更准确的实际数字。

**集成点**：

- ADR-0015 (CVT 策略模式) — 追加"2026-07 Fix"段确认完成状态
- ptx-lessons-learned §6 — 本 change 自身也是该反模式的修复案例
- 开源 OpenSpec Checklist G — lifecycle 约束（禁止 amend archived change）

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性（Checklist A）

- [x] **无 set_state/lock_guard/set_pc 调用**：`GeneralCvtStrategy::convert()` 是纯函数（输入输出 buffer，无副作用）—— 通过 grep 验证（0 matches）
- [x] **无外部 caller**：`grep "GeneralCvtStrategy" src/ include/ tests/` 在 `cvt_strategy.cpp` 之外 0 matches
- [x] **删除风险为零**：纯删除操作，无迁移
- [x] **select_strategy() 不返回 GeneralCvtStrategy**：line 1034-1046 用 `static` 局部实例 dispatch 4 个活 Strategy

### 多 Phase 推进（Checklist B/C）

- [x] **Phase 拆分**：3 个 Phase（Phase 0 artifacts / Phase 1 删死代码 / Phase 2 文档同步 + 验证），每 Phase 独立 commit + 可 revert
- [x] **基线 worktree**：`.worktrees/fix-pre-p0-baseline` 不存在（已确认），改用当前 `HEAD (66e3e2e)` 作为 baseline 直接验证
- [x] **失败处理**：任何已有测试回归 → 立即 revert 该 Phase 不混入后续 commit

### 文档同步（Checklist D）

- [x] **同步 debt-audit P0-C1** → RESOLVED（引用本 change commit hash）
- [x] **修复 cvt_strategy.cpp:1-16 文件头注释**（不再声称"Sub-task 4 将..."）
- [x] **同步 ADR-0015** 追加 "2026-07 Fix" 段
- [x] **同步 AGENTS.md / cvt README** STRUCTURE 段，反映"4 Strategy + dispatcher"实际结构

### Stale Artifact 修复（Checklist G - lifecycle）

- [x] 本 change 是**新建 fix-*** change（`fix-cvt-strategy-actual-split`），非 amend
- [x] 通过 `Ref: archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` 链接建立 lineage
- [x] 修改原 archive **禁止**（OpenSpec Checklist G 约束）

### artifacts 强制 git-tracked（Checklist E）

- [x] **Phase 0** 先 `git add openspec/changes/fix-cvt-strategy-actual-split/` + commit（避免 lessons-learned §6 教训重现）
- [x] 实施 commits 合并后立即验证 artifacts 仍 tracked（`git ls-files` 不应为空）

## Ref

- `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` — Stale artifact 本 change 修复
- `docs/adr/ADR-0015-cvt-strategy-pattern.md` — CVT 策略模式 ADR（本 change 追加完成确认）
- `docs/audits/debt-audit-2026-07-02.md §P0-C1` — Active debt（修复后 RESOLVED）
- `.opencode/skills/ptx-lessons-learned/SKILL.md §6` — Stale artifact 反模式来源
- `.opencode/skills/openspec-propose/SKILL.md §Design-Time Checklist` — Checklist A/B/D/E/G 集成要求
- `AGENTS.md §TDD 开发流程` — 测试规范
- 实证基准：`HEAD = 66e3e2e19f64f74b92ab0c1a25d53f937eb2f03f` (main, 2026-07-05)
