# Fix CVT Strategy Actual Split — Tasks

> **Type**: Stale Artifact Fix (lessons-learned §6)
> **Ref**: archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/
> **HEAD baseline**: `66e3e2e19f64f74b92ab0c1a25d53f937eb2f03f`
> **Scope**: 3 Phase（修订自原 6 Phase 计划 — Metis pre-implementation review 揭示原计划基于错误前提）
> **Strategy**: 死代码删除 + 文档同步。每 Phase 独立可 revert。

---

## 🔍 Scope 修订说明（重要）

原 proposal 假设"919 行 switch 块未拆分"，与代码现实矛盾。

**Metis pre-implementation review（2026-07-05）实证确认**：
- ✅ 4 个 Strategy 类（FloatToFloat / FloatToInt / IntToFloat / IntToInt）已在 archive commit `fc3c352`/`9837d44`/`d6123e0` 中实施
- ✅ `select_strategy()`（`cvt_strategy.cpp:1034-1046`）已 dispatch 4 个活 Strategy 实例
- ✅ `GeneralCvtStrategy::convert()`（lines 109-1031，~920 行）是**死代码**（grep 0 external callers）
- ❌ 实际工作仅 = **删除 ~920 行死代码 + 修复文件头注释 + 同步 4 个文档**

**修订策略**：从原 6 Phase（假设拆分）→ **修订为 3 Phase（清理 + 文档同步）**

---

## Phase 0: Artifacts Git-Tracking（**强制第一 Phase** —— 避免 lessons-learned §6 反模式）

> **来源**: lessons-learned §6 — 实施 OpenSpec change 必须 2-Phase commit：artifacts FIRST, 代码 SECOND

- [x] 0.1 验证 OpenSpec change 目录结构完整
  ```bash
  ls openspec/changes/fix-cvt-strategy-actual-split/
  # 期望: .openspec.yaml, proposal.md, design.md, tasks.md, specs/cvt-strategy-actual-split/spec.md
  ```
- [x] 0.2 在 main 上创建工作分支
  ```bash
  git checkout -b refactor/fix-cvt-strategy-actual-split
  ```
- [x] 0.3 git-tracked artifacts
  ```bash
  git add openspec/changes/fix-cvt-strategy-actual-split/
  git status  # 应显示 4 个新文件 + 1 个 spec.md + 1 个 .openspec.yaml
  ```
- [x] 0.4 commit artifacts（**独立 commit**）
  ```bash
  git commit -m "docs(openspec): add fix-cvt-strategy-actual-split artifacts (scope revised)

  Ref: archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/
  Phase 0 of stale artifact fix — track 4 OpenSpec artifacts before code.
  Scope revised after Metis pre-implementation review:
  - Original 6-Phase plan assumed 919-line switch unsplit (wrong premise)
  - Actual: 4 Strategy classes already deployed (commits fc3c352/9837d44/d6123e0)
  - Revised scope: dead code removal (cvt_strategy.cpp:104-1031) + doc sync
  - See design.md Decisions for rationale.

  Lessons-learned §6 Checklist E: artifacts FIRST, code SECOND."
  ```
- [x] 0.5 验证 artifacts 已 tracked
  ```bash
  git ls-files openspec/changes/fix-cvt-strategy-actual-split/
  # 应输出 5+ 个文件路径（不应为空）
  ```

---

## Phase 1: 删除 `GeneralCvtStrategy` 死代码 + 修复文件头注释

> **范围**: `src/ptxsim/instructions/cvt/cvt_strategy.cpp` 删除 lines 104-1031（~920 行死代码）+ 重写 lines 1-16 文件头注释

### 1.1 实证基线验证

- [x] 1.1.1 验证 `GeneralCvtStrategy` 在 cvt_strategy.cpp 之外无引用
  ```bash
  grep -rn "GeneralCvtStrategy" src/ include/ tests/ --include="*.cpp" --include="*.h" \
    | grep -v "src/ptxsim/instructions/cvt/cvt_strategy.cpp"
  # 期望: 无输出（grep 0 external callers）
  ```
- [x] 1.1.2 验证 file header 仍然声称"Sub-task 4 将..."
  ```bash
  sed -n '1,16p' src/ptxsim/instructions/cvt/cvt_strategy.cpp
  # 期望: 应包含 "Sub-task 4 将 GeneralCvtStrategy::convert() 拆为 5 个具体策略"
  ```
- [x] 1.1.3 跑现有 CVT 测试，记录 baseline
  ```bash
  cd build && ctest --output-on-failure -R "cvt" 2>&1 | tee /tmp/cvt-baseline.log
  # 期望: 全部 PASS（14 个测试）
  ```

### 1.2 删除死代码（pure deletion）

- [x] 1.2.1 验证 `cvt_strategy.cpp` 当前行数与定位死代码
  ```bash
  wc -l src/ptxsim/instructions/cvt/cvt_strategy.cpp
  # 期望: 1061
  ```
- [x] 1.2.2 阅读 line 104-1031 内容（确认是 dead `GeneralCvtStrategy` 类）
- [x] 1.2.3 使用 `Edit` 工具删除 `cvt_strategy.cpp:104-1031`（class 定义 + convert() + name()）
  ```cpp
  // 旧内容（line 104-1031）:
  // GeneralCvtStrategy: 暂存原 arithmetic_conversion.cpp 整个 switch
  // ... (~920 行 class 定义) ...
  // const char *name() const override { return "GeneralCvtStrategy"; }
  // };

  // 新内容: 完全移除（保留 line 1031 之后的 select_strategy 函数及以上内容）
  ```
- [x] 1.2.4 验证 `wc -l cvt_strategy.cpp` < 200（期望 ~140 行）

### 1.3 修复文件头注释

- [x] 1.3.1 使用 `Edit` 工具重写 `cvt_strategy.cpp:1-16`
  ```cpp
  // cvt_strategy.cpp
  // =============================================================================
  // CVT 策略模式 — dispatcher 实现
  //
  // 状态（fix-cvt-strategy-actual-split 修订后）:
  //   - build_context():  从 Qualifier 列表构造强类型 CvtContext
  //   - select_strategy(): 返回 4 个具体 Strategy 实例之一（按 dst/src 类型）
  //   - CvtHandler::processOperation(): 顶层入口，调用 select_strategy + convert
  //
  // 4 个活 Strategy 类（已由 archive Sub-task 3-4 实施，commit fc3c352/9837d44/d6123e0）:
  //   - FloatToFloatStrategy  → cvt_float_to_float.cpp
  //   - FloatToIntStrategy    → cvt_float_to_int.cpp    (含 .sat/5 rounding/.ftz)
  //   - IntToFloatStrategy    → cvt_int_to_float.cpp
  //   - IntToIntStrategy      → cvt_int_to_int.cpp
  //
  // 详见:
  //   - ADR-0015 (CVT 策略模式)
  //   - docs/adr/0015-cvt-strategy-pattern.md
  // =============================================================================
  ```
- [x] 1.3.2 验证 `grep "Sub-task 4 将" cvt_strategy.cpp` 无匹配

### 1.4 编译 + 测试验证

- [x] 1.4.1 增量编译
  ```bash
  cmake --build build --target ptxsim -j$(nproc) 2>&1 | tee /tmp/build-phase1.log
  # 期望: 0 errors（删除死代码不应引入链接错误）
  ```
- [x] 1.4.2 跑 CVT 测试，与 baseline 对比零回归
  ```bash
  cd build && ctest --output-on-failure -R "cvt" 2>&1 | tee /tmp/cvt-after.log
  diff /tmp/cvt-baseline.log /tmp/cvt-after.log  # 期望无 diff
  ```
- [x] 1.4.3 跑完整 ctest 验证零侧效
  ```bash
  ctest --output-on-failure 2>&1 | tee /tmp/ctest-phase1.log
  ```
- [x] 1.4.4 跑 PTX 语法全量测试（不变性 oracle）
  ```bash
  ./tests/ptx/test_all_ptx.sh 2>&1 | tee /tmp/ptx-after.log
  ```

### 1.5 Commit Phase 1

- [x] 1.5.1 提交
  ```bash
  git add src/ptxsim/instructions/cvt/cvt_strategy.cpp
  git commit -m "refactor(cvt): remove dead GeneralCvtStrategy class (Fix #1)

  Phase 1 of fix-cvt-strategy-actual-split.
  Removed 920 lines of dead code (cvt_strategy.cpp:104-1031).

  Background:
  - Archive 'phase3-t2-6-cvt-strategy-pattern' (commit ccbbe2a, 2026-06-24)
    marked Sub-task 4 complete but GeneralCvtStrategy::convert() god class
    was never split. Four Strategy classes were actually deployed
    (commits fc3c352/9837d44/d6123e0).
  - select_strategy() (lines 1034-1046) dispatches 4 active Strategy
    instances; GeneralCvtStrategy has 0 external callers (verified via
    grep). Pure deletion, zero behavior change.

  Changes:
  - Deleted class GeneralCvtStrategy (lines 109-1031)
  - Updated file header (lines 1-16) to reflect actual 4-Strategy state
  - cvt_strategy.cpp reduced from 1061 to ~140 lines

  Tests:
  - 14 CVT tests (6 unit + 8 integration) zero regression
  - test_all_ptx.sh pass
  - Full ctest pass

  Ref: archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/
  Ref: design.md Decision 1-2
  Lessons-learned §6 Checklist A: pure deletion, no set_state risk."
  ```

---

## Phase 2: 4 个文档同步 + 最终验证

> **范围**: debt-audit + ADR-0015 + AGENTS.md（或 cvt README） + e2e 验证

### 2.1 更新 `docs/audits/debt-audit-2026-07-02.md §P0-C1`

- [x] 2.1.1 定位 P0-C1 当前状态
  ```bash
  grep -n "P0-C1\|GeneralCvtStrategy\|cvt_strategy.cpp" docs/audits/debt-audit-2026-07-02.md
  ```
- [x] 2.1.2 将 active debt 状态改为 RESOLVED
  ```markdown
  ### P0-C1 — cvt_strategy.cpp god class（曾标 active）

  **Status**: ✅ RESOLVED by `change fix-cvt-strategy-actual-split` (commits <hash1>, <hash2>)

  **背景**: 误判。原 archive `phase3-t2-6-cvt-strategy-pattern` 已通过 commits `fc3c352/9837d44/d6123e0` 部署 4 个 Strategy 类。本 change 仅清理死代码 `GeneralCvtStrategy`（1061 → ~140 行）+ 修复文件头注释。

  **Resolution evidence**:
  - `cvt_strategy.cpp` 从 1061 行降至 ~140 行 dispatcher
  - 4 个 Strategy 类（`cvt_int_to_int` / `cvt_float_to_float` / `cvt_int_to_float` / `cvt_float_to_int`）已生效
  - 14 个 CVT 测试零回归
  - ADR-0015 追加 "2026-07 Fix" 段
  - lessons-learned §6 反模式案例（stale artifact 误标 debt）
  ```

### 2.2 更新 `docs/adr/0015-cvt-strategy-pattern.md`

- [x] 2.2.1 验证当前 ADR 是否有 "2026-07 Fix" 段
  ```bash
  grep -n "2026-07" docs/adr/0015-cvt-strategy-pattern.md || echo "no existing fix section"
  ```
- [x] 2.2.2 追加 "2026-07 Fix: Dead Code Cleanup" 段（若无）
  ```markdown
  ## 2026-07 Fix: 死代码清理

  `change fix-cvt-strategy-actual-split`（Metis pre-implementation review 后修订 scope）：

  - **背景**：原 archive 标记 Sub-task 4 完成但 `GeneralCvtStrategy::convert()` 919 行 god class 未拆分。Metis 实证发现 4 个 Strategy 类已部署。
  - **本次修订**：删除死代码 `GeneralCvtStrategy` 类（约 920 行），更新文件头注释。
  - **结果**：`cvt_strategy.cpp` 从 1061 行降至 ~140 行 dispatcher，零行为变更。
  ```

### 2.3 更新 `src/ptxsim/instructions/AGENTS.md` STRUCTURE 段

- [x] 2.3.1 定位 STRUCTURE 段
  ```bash
  grep -n "STRUCTURE" src/ptxsim/instructions/AGENTS.md 2>/dev/null || \
  grep -n "STRUCTURE" src/ptxsim/instructions/README.md 2>/dev/null
  ```
- [x] 2.3.2 更新 cvt/ 子目录文件清单（如有 STRUCTURE 段）
  ```markdown
  src/ptxsim/instructions/cvt/
  ├── cvt_strategy.{h,cpp}            # dispatcher (~140 行) + 接口定义
  ├── cvt_int_to_int_strategy.{h,cpp} # IntToIntStrategy    (commits 9837d44)
  ├── cvt_float_to_float_strategy.{h,cpp} # FloatToFloatStrategy (fc3c352)
  ├── cvt_int_to_float_strategy.{h,cpp}   # IntToFloatStrategy   (d6123e0)
  ├── cvt_float_to_int_strategy.{h,cpp}   # FloatToIntStrategy   (d6123e0, 含 .sat/5 rounding/.ftz)
  └── cvt_helpers.{h,cpp}             # 4 helper 函数 (round_half_to_even 等)
  ```
- [x] 2.3.3 如无 AGENTS.md 文件，跳过此步骤（OpenSpec 工具层不需要）

### 2.4 最终验证

- [x] 2.4.1 完整 build pass
  ```bash
  cmake --build build -j$(nproc) 2>&1 | tee /tmp/full-build.log
  ```
- [x] 2.4.2 CVT 测试 PASS
  ```bash
  cd build && ctest --output-on-failure -R "cvt"
  # 期望: 14 个测试 PASS
  ```
- [x] 2.4.3 关键 e2e oracle 测试 PASS
  ```bash
  ctest -R e2e_blackwell_gemm --output-on-failure
  ```
- [x] 2.4.4 完整 PTX 语法测试 PASS
  ```bash
  ./tests/ptx/test_all_ptx.sh
  ```
- [x] 2.4.5 完整 sanity check
  ```bash
  ./scripts/sanity.sh --quick
  ```
- [x] 2.4.6 验证 deprecation 残留
  ```bash
  grep -rn "TODO.*extract\|FIXME.*god class\|TODO.*split" src/ptxsim/instructions/cvt/
  # 期望: 无输出
  grep "Sub-task 4 将" src/ptxsim/instructions/cvt/cvt_strategy.cpp
  # 期望: 无匹配
  grep "GeneralCvtStrategy" src/ptxsim/instructions/cvt/cvt_strategy.cpp
  # 期望: 无匹配
  ```

### 2.5 Commit Phase 2

- [x] 2.5.1 提交
  ```bash
  git add docs/audits/debt-audit-2026-07-02.md \
          docs/adr/0015-cvt-strategy-pattern.md \
          src/ptxsim/instructions/AGENTS.md 2>/dev/null || true
  git commit -m "docs(cvt): sync stale artifact fix + debt-audit RESOLVED (Fix #2)

  Phase 2 of fix-cvt-strategy-actual-split.
  Synced 3 documentation artifacts to reflect actual completion:
  - docs/audits/debt-audit-2026-07-02.md §P0-C1: active → RESOLVED
  - docs/adr/0015-cvt-strategy-pattern.md: appended '2026-07 Fix' section
  - src/ptxsim/instructions/AGENTS.md STRUCTURE: cvt/ layout

  Final verification:
  - 14 CVT tests pass
  - e2e_blackwell_gemm pass
  - test_all_ptx.sh pass
  - sanity --quick pass
  - No 'Sub-task 4 将' / 'GeneralCvtStrategy' residue

  Ref: archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/
  Ref: design.md Decisions 4-5
  Lessons-learned §6 Checklists D + E: doc sync + artifact tracking."
  ```

---

## 3. 归档（Phase 2 完成后）

- [x] 3.1 跑完整 sanity check
  ```bash
  ./scripts/sanity.sh --quick 2>&1 | tee /tmp/sanity-final.log
  ```
- [x] 3.2 postmortem 与 lessons-learned 沉淀
  ```bash
  # 在 docs/dev-process/lessons-learned.md 追加
  # §6 stale artifact 反模式案例（fix-cvt-strategy-actual-split 是该案例的修复）
  # 新教训：实施 OpenSpec change 前必跑 pre-implementation review（防止 scope 漂移）
  ```
- [x] 3.3 调用 openspec-archive-change skill
  ```bash
  # 参考 .opencode/skills/openspec-archive-change/SKILL.md
  # 用户会被 prompt 询问生成 postmortem
  ```
- [ ] 3.4 验证归档包含 .openspec.yaml
  ```bash
  # 归档后 git ls-files openspec/changes/archive/fix-cvt-strategy-actual-split/ 不应为空
  ```

---

## 关键验证检查点（Phase 1→2 全程）

每个 Phase commit 前必须验证：

- [x] **零回归**：Phase 1 跑 `ctest -R cvt`（14 测试） + Phase 2 跑 `sanity --quick` 全部 PASS
- [x] **基线对比**：与上一 Phase baseline 对比零 diff（Phase 1 与 HEAD baseline，Phase 2 与 Phase 1）
- [x] **e2e GEMM**：Phase 1 + 2 末跑 `ctest -R e2e_blackwell_gemm` PASS（不变性 oracle）
- [x] **PTX 语法**：`./tests/ptx/test_all_ptx.sh` PASS

任何 Phase 测试回归 → **立即 revert 该 Phase**，不混入后续 commit（lessons-learned §3）。

## 失败处理纪律（lessons-learned §3）

> **核心原则**：已有测试回归 = 立即 revert，禁止"再改一点看看"

具体执行：
```bash
# 1. 检测到回归
git log --oneline -5  # 找出当前 Phase commit

# 2. 立即 revert
git revert <phase-commit-sha> --no-edit

# 3. 验证 revert 后编译 + 测试
cmake --build build && ctest -R "cvt"

# 4. 调查 root cause（参考 debugging-strategy.md）
# 5. 重新实施该 Phase
```

---

## 与原 6 Phase 计划的关键差异

| 项目 | 原计划 | 修订后 |
|------|--------|--------|
| 总 Phase 数 | 6 | **3** |
| 工作量 | ~500 行迁移 + 5 个新 Strategy 类 | **~60 行删除 + 文档同步** |
| 风险 | 高（5 个新类 + CMake 注册） | **极低（pure deletion）** |
| 测试影响 | 5 个新单元测试文件 | **零**（仅验证现有测试不回归） |
| CvtSatStrategy | 计划引入 wrapper | **不引入**（避免双重饱和） |
| Worktree | 引用不存在的 `.worktrees/fix-pre-p0-baseline` | **用 HEAD 直接验证** |

---

## Ref

- `proposal.md` — Stale artifact fix 范围声明
- `design.md` — 5 个 Decision + 3 Phase 规划
- `spec.md` — 修订后的 Requirements
- `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/` — 本 change 修复对象
- `docs/adr/0015-cvt-strategy-pattern.md` — Phase 2 待追加完成段
- `docs/audits/debt-audit-2026-07-02.md §P0-C1` — Phase 2 待标记 RESOLVED
- `.opencode/skills/ptx-lessons-learned/SKILL.md §6` — 关键决策依据
- Metis pre-implementation review（2026-07-05）— 揭示原计划 scope 错误
- 实证基准：`HEAD = 66e3e2e19f64f74b92ab0c1a25d53f937eb2f03f` (main, 2026-07-05)
