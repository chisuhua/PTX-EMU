## Context

### 当前状态（基于 Metis pre-implementation review 实证）

**活 Strategy 类已实施并生效**（`git log` + grep 验证）：

| 文件 | 行数 | 内容 | Commit |
|------|------|------|--------|
| `src/ptxsim/instructions/cvt/cvt_float_to_float.cpp` | 2891 B | `FloatToFloatStrategy` (f32↔f64↔f16) | `fc3c352` |
| `src/ptxsim/instructions/cvt/cvt_int_to_int.cpp` | 6914 B | `IntToIntStrategy` + wire up 5 strategies | `9837d44` |
| `src/ptxsim/instructions/cvt/cvt_float_to_int.cpp` | 10039 B | `FloatToIntStrategy` (含 .sat 处理) | `d6123e0` |
| `src/ptxsim/instructions/cvt/cvt_int_to_float.cpp` | 2253 B | `IntToFloatStrategy` | `d6123e0` |
| `src/ptxsim/instructions/cvt/cvt_helpers.cpp` | 1858 B | 4 个 helper 函数（round_half_to_even 等） | `d3c77b5` |

**`select_strategy()` dispatch**（`cvt_strategy.cpp:1034-1046`）：

```cpp
const ConversionStrategy &select_strategy(const CvtContext &ctx) {
    static const FloatToFloatStrategy f2f;
    static const FloatToIntStrategy f2i;
    static const IntToFloatStrategy i2f;
    static const IntToIntStrategy i2i;

    if (ctx.dst_is_float) {
        return ctx.src_is_float ? static_cast<const ConversionStrategy &>(f2f)
                                : static_cast<const ConversionStrategy &>(i2f);
    }
    return ctx.src_is_float ? static_cast<const ConversionStrategy &>(f2i)
                            : static_cast<const ConversionStrategy &>(i2i);
}
```

✅ 4 个 Strategy 类已生效。`GeneralCvtStrategy` 是死代码（编译在但从未被调用）。

**`CvtHandler::processOperation()`**（`cvt_strategy.cpp:1058-1061`）：

```cpp
void CvtHandler::processOperation(...) {
    void *dst = operands[0];
    void *src = operands[1];
    auto ctx = ptxsim::cvt_strategy::build_context(qualifiers);
    const auto &strategy = ptxsim::cvt_strategy::select_strategy(ctx);
    strategy.convert(dst, src, ctx);
}
```

✅ 11 行 dispatcher，已是合适状态。

### 待清理的死代码

- `cvt_strategy.cpp:104-1031`：class 定义 + `~920 行 switch 块` + `name() override`
- `cvt_strategy.cpp:1-16`（文件头注释）：仍声称 "Sub-task 4 将 GeneralCvtStrategy::convert() 拆为 5 个具体策略" — **与代码现实矛盾**

### 待更新的文档

- `docs/audits/debt-audit-2026-07-02.md §P0-C1` — 标为 active debt，实际已实现
- `docs/adr/ADR-0015-cvt-strategy-pattern.md` — ADR 中是否已追加"完成"段待确认
- `src/ptxsim/instructions/AGENTS.md` 或 `cvt/README.md` — STRUCTURE 段是否反映"4 Strategy + dispatcher"实际结构

### 约束（来自 lessons-learned §6 + AGENTS.md）

- **Checklist G（lifecycle）**：禁止 amend 已归档 change `2026-06-24-phase3-t2-6-cvt-strategy-pattern`（OpenSpec 终态约束）
- **Checklist E（artifacts tracking）**：本 change 的 4 个 artifacts（proposal/design/specs/tasks）必须 git-tracked，**禁止 working tree 遗漏**
- **Checklist A（函数迁移）**：删除操作无 set_state/lock_guard/set_pc 风险（pure deletion of dead code）

### 利益相关者

- **实施者**：本 Sisyphus agent
- **审计者**：任何后续 debt audit（应读取 AGENTS.md §Ref 链接 + 本 change artifacts）
- **未来扩展者**：添加 CVT 变体（`.relu` 等）的开发者

## Goals / Non-Goals

**Goals:**

1. **删除死代码 god class**：移除 `cvt_strategy.cpp:104-1031` 的 `GeneralCvtStrategy` 类（~920 行）
2. **修复 stale 文件头注释**：`cvt_strategy.cpp:1-16` 改为反映实际"4 Strategy + dispatcher"结构
3. **同步 4 个文档**：debt-audit P0-C1 → RESOLVED, ADR-0015 追加 Fix 段, AGENTS.md / cvt README 更新
4. **零行为变更**：所有现有 CVT 测试零回归（oracle 实际为 14 个测试，非虚构 94 个）
5. **建立 reusable 模板**：通过本 change 展示"纯删除 dead code + 文档同步"的 OpenSpec 修补模式

**Non-Goals:**

- ❌ **不修改 4 个活 Strategy 类**（`cvt_int_to_int` / `cvt_float_to_float` / `cvt_int_to_float` / `cvt_float_to_int`）
- ❌ **不修改 `cvt_strategy.h` 接口**（`select_strategy()` 返回 `const ConversionStrategy&` 不变）
- ❌ **不实现 `CvtSatStrategy` composition wrapper**：现有 4 个 Strategy 内部已处理 `.sat`（双重饱和风险，且违反"零行为变更"）
- ❌ **不实现新 CVT 变体**（`.sat.s8` / `.relu` 等）
- ❌ **不创建新单元测试文件**（`tests/unit/cvt/` 目录不存在，避免引入新测试扩大 surface）
- ❌ **不修改 `cvt_helpers.cpp`**（独立 helper 文件，不在范围）

## Decisions

### Decision 1：删除 `GeneralCvtStrategy` 类整体（非局部保留）

**选择**：删除 `cvt_strategy.cpp:104-1031` 全部内容（包括 class 定义 + convert() switch + name() override），不留任何形式的兜底。

**理由**：
- `select_strategy()` 不返回 `GeneralCvtStrategy`，删除不会导致链接错误
- 死代码无任何外部 caller（grep 验证 0 matches）
- 保留兜底会引入再次形成 god class 的风险
- 删除是最简单、最安全、可逆性最高的操作

**替代方案对比**：

| 替代方案 | 缺点 |
|---------|------|
| 保留 `GeneralCvtStrategy` 作为 fallback | 死代码风险，未来易被误用 |
| 将 `convert()` switch 迁移到 `cvt_legacy_strategy.cpp` 单独文件 | 无 caller，无意义迁移 |
| 局部注释掉 `GeneralCvtStrategy` 类 | 仍编译，浪费编译时间 |

### Decision 2：保留 `select_strategy()` 返回 `const ConversionStrategy&`

**选择**：保留当前签名。`CvtSatStrategy` composition wrapper **不引入**。

**理由**：
- 当前 4 个 Strategy 类是 static 局部对象（line 1035-1038），返回引用最自然
- 如改 `unique_ptr<>` 需要修改 `cvt_strategy.h` 接口（违反 Non-Goal "不修改 cvt_strategy.h"）
- `CvtSatStrategy` 与现有 `.sat` 处理会产生双重饱和（详见 Decision 3）
- `.sat` 已在 4 个 Strategy 内部正确处理（`cvt_float_to_int.cpp:60-61` 等位置）

### Decision 3：**不**实现 `CvtSatStrategy` composition wrapper

**选择**：保留 4 个 Strategy 类内部各自处理 `.sat`。

**理由**：
- 现有 `.sat` 处理在 4 个 Strategy 内部已正确
- 若 wrapper 也 apply saturate，会双重饱和（违反"零行为变更"）
- 若 wrapper 仅当 inner 不处理 .sat 时调用，需先移除内部处理（行为变更）
- 现状已是最简洁的方案（composition wrapper 是 over-engineering）

### Decision 4：Minimal-touch Phase 划分（3 Phase 而非 6）

**选择**：3 个 Phase，每 Phase ≤ 1 个 commit。

| Phase | 内容 | 风险 |
|-------|------|------|
| Phase 0 | 4 artifacts git-tracking（Checklist E 强制） | 零 |
| Phase 1 | 删除 `GeneralCvtStrategy` 类 + 修复文件头注释 | 极低（pure deletion） |
| Phase 2 | 4 个文档同步 + 最终验证 | 零（仅文档） |

**理由**：原 6 Phase 计划基于错误前提（"919 行 switch 未拆分"）。实际工作只需删除 + 文档同步，规模匹配 3 Phase。

### Decision 5：基线验证用 HEAD 直接验证（非 worktree）

**选择**：用当前 `HEAD (66e3e2e)` 作为 baseline。

**理由**：
- `.worktrees/fix-pre-p0-baseline` 不存在（`git worktree list` 验证）
- 删除死代码风险极低，无需 15-20 分钟 baseline build
- `cmake --build build && ctest` 即可验证（增量 build < 30s）

**降级方案**：若 Phase 1 出现 regression，创建 worktree 做 bisect。

## Risks / Trade-offs

| 风险 | 等级 | 缓解 |
|------|------|------|
| 删除 `GeneralCvtStrategy` 导致链接错误 | 🟢 低 | grep 验证 0 external callers |
| `select_strategy()` 静态实例生命周期问题 | 🟢 低 | static 局部对象 = 进程生命周期（CUDA runtime 标准用法） |
| 文档路径错误（proposal 引用 `.opencode/notes/` 而真实在 `docs/audits/`） | 🟢 低 | 已修正（本 design.md + 重写后 proposal.md 用正确路径） |
| Phase 1 改动后 `wc -l cvt_strategy.cpp` 不满足 < 200 行目标 | 🟢 低 | 删除 ~920 行后应 ~140 行，远低于 200 |
| 任何文档同步遗漏 | 🟡 中 | Phase 2 详尽清单 + git grep 验证 |

## Migration Plan

### Phase 0：artifact git-tracking（**强制第一 Phase**）

避免 lessons-learned §6 反模式：

1. 验证 OpenSpec change 目录结构完整（已有 4 artifacts + 1 spec.md）
2. 在 main 上创建工作分支 `git checkout -b refactor/fix-cvt-strategy-actual-split`
3. `git add openspec/changes/fix-cvt-strategy-actual-split/`
4. commit artifacts（独立 commit）
5. `git ls-files openspec/changes/fix-cvt-strategy-actual-split/` 验证非空

### Phase 1：删除 `GeneralCvtStrategy` + 修复文件头注释

1. **删除 `cvt_strategy.cpp:104-1031`**：使用 `edit` 工具一次替换块状内容为空
2. **重写 `cvt_strategy.cpp:1-16` 文件头注释**：
   - 删除 "Sub-task 4 将 GeneralCvtStrategy::convert() 拆为 5 个具体策略" 段
   - 改为实际状态："策略模式完整部署（4 个 Strategy 类 + dispatcher，archive Sub-task 3-4 完成，详见 ADR-0015）"
3. **验证 `wc -l cvt_strategy.cpp` < 200**（预期 ~140 行）
4. **编译 + 跑 ctest**：增量 build < 30s
5. **commit**：`refactor(cvt): remove dead code GeneralCvtStrategy (Fix #1)`

### Phase 2：4 个文档同步 + 最终验证

1. **更新 `docs/audits/debt-audit-2026-07-02.md §P0-C1`**：
   - status: active → ✅ RESOLVED by `change fix-cvt-strategy-actual-split` (commit <hash>)
2. **更新 `docs/adr/ADR-0015-cvt-strategy-pattern.md`**：
   - 追加 "2026-07 Fix: 死代码清理" 段
3. **更新 `src/ptxsim/instructions/AGENTS.md` STRUCTURE**：
   - cvt/ 目录清单：cvt_strategy.{h,cpp} + cvt_{int_to_int,float_to_float,int_to_float,float_to_int,helpers}.{h,cpp}
4. **最终验证**：
   - `./scripts/sanity.sh --quick` 无回归
   - `./tests/ptx/test_all_ptx.sh` PASS
   - `ctest -R e2e_blackwell_gemm` PASS
   - `grep "Sub-task 4 将" cvt_strategy.cpp` 无匹配
5. **commit**：`docs(cvt): sync stale artifact fix + debt-audit RESOLVED (Fix #2)`

### 回退策略（每个 Phase 独立可 revert）

```bash
# 任何 Phase 后如出现 test 回归
git revert <phase-commit-sha> --no-edit
cmake --build build && ctest -L "unit;cvt" && ctest -L "integration;cvt"
```

## Open Questions

1. **`src/ptxsim/instructions/AGENTS.md` 与 `src/ptxsim/instructions/cvt/README.md` 哪个更适合更新？**
   - 选项 A：AGENTS.md（OpenCode 自动加载，agent 可见）
   - 选项 B：cvt/README.md（cvt 子目录专属）
   - **倾向**：选项 A（agent 友好）

2. **是否需要创建 `tests/unit/cvt/` 目录？**
   - 选项 A：暂不创建（保持 scope 最小，避免新测试 surface 扩大）
   - 选项 B：创建空目录占位（防止后续目录冲突）
   - **倾向**：选项 A（最小触碰，遵循 Non-Goal）

3. **`docs/adr/ADR-0015-cvt-strategy-pattern.md` 是否已有 "完成状态" 段？**
   - 待 Phase 2 实施时验证（先 `grep "2026-07" docs/adr/ADR-0015-...md`）
   - 已有则补充"死代码清理"小节；无则新建段

## Ref

- `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` — Stale artifact 本 change 修复
- `docs/adr/ADR-0015-cvt-strategy-pattern.md` — CVT 策略模式 ADR（本 change 追加确认状态）
- `docs/audits/debt-audit-2026-07-02.md §P0-C1` — Active debt（修复后 RESOLVED）
- `.opencode/skills/ptx-lessons-learned/SKILL.md §6` — Stale artifact 反模式来源
- `.opencode/skills/openspec-propose/SKILL.md §Design-Time Checklist` — Checklist A/B/D/E/G 集成要求
- `AGENTS.md §TDD 开发流程` — 测试规范
- Metis pre-implementation review（2026-07-05）— 揭示 change scope 错误，发现 4 个 Strategy 类已实施
- 实证基准：`HEAD = 66e3e2e19f64f74b92ab0c1a25d53f937eb2f03f` (main, 2026-07-05)
