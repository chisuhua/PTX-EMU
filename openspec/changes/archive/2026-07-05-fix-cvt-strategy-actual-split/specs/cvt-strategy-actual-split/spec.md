# cvt-strategy-actual-split

## Purpose

修复 `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` 的 stale artifact：

- **过去状态**：archive 标记 ✅ COMPLETED，但 `src/ptxsim/instructions/cvt/cvt_strategy.cpp` 仍有 1061 行，`GeneralCvtStrategy::convert()` 包含 ~920 行未拆分 switch 块
- **现实**：4 个活 Strategy 类（`FloatToFloatStrategy` / `FloatToIntStrategy` / `IntToFloatStrategy` / `IntToIntStrategy`）已在 archive 实施 commits（`fc3c352`/`9837d44`/`d6123e0`）部署并被 `select_strategy()` dispatch
- **本 change**：删除死代码 `GeneralCvtStrategy` 类（~920 行）+ 修复文件头注释 + 同步 4 个文档，建立 lessons-learned §6 反模式修复的可复用模板

## ADDED Requirements

### Requirement: 删除 `GeneralCvtStrategy` 死代码类

The system SHALL 删除 `src/ptxsim/instructions/cvt/cvt_strategy.cpp:104-1031` 的 `GeneralCvtStrategy` 类整体定义（包括 class 声明、`convert()` 方法的 ~920 行 switch 块、`name()` override）。删除后 `cvt_strategy.cpp` 仅保留：
- `build_context()` factory
- `select_strategy()` dispatch（保持 4 个活 Strategy 引用）
- `CvtHandler::processOperation()` 顶层入口

The `ConversionStrategy` 接口、`CvtContext` 强类型上下文、`cvt_strategy.h` 公共 API SHALL 不变。

#### Scenario: 删除后 `cvt_strategy.cpp` 总行数 < 200
- **WHEN** `wc -l src/ptxsim/instructions/cvt/cvt_strategy.cpp`
- **THEN** 输出 < 200（预期 ~140 行）

#### Scenario: 删除后 `GeneralCvtStrategy` 类无残留引用
- **WHEN** `grep -rn "GeneralCvtStrategy" src/ include/ tests/ --include="*.cpp" --include="*.h"`
- **THEN** 仅 0 输出（已删除类的引用也应全部消失）

#### Scenario: 删除后编译无错误
- **WHEN** `cmake --build build --target ptxsim -j$(nproc)`
- **THEN** 退出码 0（无 undefined reference 或其他链接错误）

### Requirement: 修复 stale 文件头注释

The system SHALL 将 `src/ptxsim/instructions/cvt/cvt_strategy.cpp:1-16` 的文件头注释从"Sub-task 4 将 GeneralCvtStrategy::convert() 拆为 5 个具体策略"改为反映实际"4 Strategy + dispatcher"状态。

#### Scenario: 文件头注释不再含 "Sub-task 4 将"
- **WHEN** `grep "Sub-task 4 将" src/ptxsim/instructions/cvt/cvt_strategy.cpp`
- **THEN** 无匹配

#### Scenario: 文件头注释反映 4 Strategy 实例状态
- **WHEN** `head -16 src/ptxsim/instructions/cvt/cvt_strategy.cpp`
- **THEN** 包含 4 个 Strategy 类名（`FloatToFloatStrategy` / `FloatToIntStrategy` / `IntToFloatStrategy` / `IntToIntStrategy`）

### Requirement: 同步 `docs/audits/debt-audit-2026-07-02.md` P0-C1 状态

The system SHALL 更新 `docs/audits/debt-audit-2026-07-02.md §P0-C1` 的状态从 active debt 改为 RESOLVED，并引用本 change 的 commits。

#### Scenario: P0-C1 状态标记为 RESOLVED
- **WHEN** `grep "P0-C1" docs/audits/debt-audit-2026-07-02.md`
- **THEN** 该行包含 "RESOLVED" + 引用 `change fix-cvt-strategy-actual-split`

#### Scenario: 误判说明包含
- **WHEN** 阅读 P0-C1 段
- **THEN** 段内说明本 debt 系"误判"（archive 已实施 4 个 Strategy 类）

### Requirement: 同步 `docs/adr/ADR-0015-cvt-strategy-pattern.md`

The system SHALL 在 `docs/adr/ADR-0015-cvt-strategy-pattern.md` 追加 "2026-07 Fix: 死代码清理" 段，说明本 change 的背景与结果。

#### Scenario: ADR 含 2026-07 Fix 段
- **WHEN** `grep "2026-07 Fix" docs/adr/ADR-0015-cvt-strategy-pattern.md`
- **THEN** 至少 1 行匹配

### Requirement: 同步 STRUCTURE 文档（如存在）

The system SHALL 更新 `src/ptxsim/instructions/AGENTS.md`（若存在 STRUCTURE 段）或 cvt 子目录 README，反映"4 Strategy + dispatcher"实际结构。

#### Scenario: STRUCTURE 段列出 4 个 Strategy 文件
- **WHEN** 阅读 STRUCTURE 段
- **THEN** 包含 `cvt_int_to_int` / `cvt_float_to_float` / `cvt_int_to_float` / `cvt_float_to_int` 4 个文件

### Requirement: 零行为变更回归验证

The system SHALL 保持 `select_strategy()` dispatch 行为完全一致：
- 返回类型 `const ConversionStrategy&` 不变
- 4 个 Strategy 类引用不变（`FloatToFloatStrategy` / `FloatToIntStrategy` / `IntToFloatStrategy` / `IntToIntStrategy`）
- dispatch 逻辑（`if (ctx.dst_is_float)` 二分 + `ctx.src_is_float` 二分）不变

#### Scenario: 14 个 CVT 测试零回归
- **WHEN** `cd build && ctest --output-on-failure -R "cvt"`
- **THEN** 全部 PASS（6 unit + 8 integration）

#### Scenario: 关键 e2e GEMM 测试 PASS
- **WHEN** `ctest -R e2e_blackwell_gemm --output-on-failure`
- **THEN** PASS

#### Scenario: 完整 PTX 语法测试 PASS
- **WHEN** `./tests/ptx/test_all_ptx.sh`
- **THEN** 退出码 0（不变性 oracle）

#### Scenario: 完整 sanity 检查 PASS
- **WHEN** `./scripts/sanity.sh --quick`
- **THEN** 无回归

## RENAMED Requirements

无（命名未变，仅实现清理）。

## REMOVED Requirements

无（不删除任何现有能力 — 仅删除未被调用的死代码）。

## 设计约束

### 必须遵守的约束

1. **Pure deletion 性质** — `GeneralCvtStrategy` 是死代码（grep 验证 0 external callers），删除操作零行为变更
2. **不修改 4 个活 Strategy 类**（`cvt_int_to_int.cpp` / `cvt_float_to_float.cpp` / `cvt_int_to_float.cpp` / `cvt_float_to_int.cpp`）—— 已生效实现，删除范围仅限 `cvt_strategy.cpp:104-1031`
3. **不修改 `cvt_strategy.h` 公共接口** —— `ConversionStrategy` / `CvtContext` / `select_strategy()` 签名保持
4. **不实现 `CvtSatStrategy` composition wrapper** —— 现有 4 个 Strategy 内部已处理 `.sat`，新增 wrapper 会导致双重饱和

### 禁止的反模式

1. **禁止保留 `GeneralCvtStrategy` 作为兜底** —— 宁可让 dispatcher 失败抛 `UnsupportedOperationException`，也不要保留死代码
2. **禁止修改 `select_strategy()` 返回类型**（如改 `unique_ptr<>`）—— 改变接口超出 scope
3. **禁止 amend 已归档 change**（OpenSpec Checklist G 约束）—— 修改 `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` 是 lifecycle 违规
4. **禁止删除 4 个活 Strategy 文件** —— 它们是 dispatcher 实际 dispatch 的目标
5. **禁止修改 `cvt_helpers.cpp`** —— 独立 helper 文件，不在清理范围

## OpenSpec artifacts 完整性（lessons-learned §6 Checklist E）

本 change 的 OpenSpec artifacts MUST 在 Phase 0 一次性 `git add` 并 commit，后续实施 commits 才不会与 artifacts 修改冲突。

#### Scenario: artifacts git-tracked 验证
- **WHEN** `git ls-files openspec/changes/fix-cvt-strategy-actual-split/`
- **THEN** 输出 5+ 个文件路径（不应为空）

## Reference

- `proposal.md` §What Changes — 详细范围声明
- `design.md` §Decisions — 5 个关键决策（保留 dispatcher 签名、不引入 SatStrategy、3 Phase 等）
- `tasks.md` §Phase 0-2 — 实施步骤
- `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` — 本 change 修复对象
- `docs/adr/ADR-0015-cvt-strategy-pattern.md` — Phase 2 待追加 Fix 段
- `docs/audits/debt-audit-2026-07-02.md §P0-C1` — Phase 2 待标记 RESOLVED
- `.opencode/skills/ptx-lessons-learned/SKILL.md §6` — 关键决策依据
- Metis pre-implementation review（2026-07-05）— 揭示 change scope 修订原因
- 实证基准：`HEAD = 66e3e2e19f64f74b92ab0c1a25d53f937eb2f03f`
