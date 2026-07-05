# Tasks: Add Extern Function Declaration Oracle Coverage

> **Type**: Stale Artifact Fix + Oracle Coverage（lessons-learned §22 续集）
> **HEAD baseline**: `72b2bde7ae165a0c0af149ac4445fa6746e99ca9` (parser-completeness merge)
> **Scope**: 2 Phase（stale TODO 删除 + oracle test → AGENTS.md 同步）
> **Reference**: 关联 `archive/2026-07-05-parser-completeness/`（MR-7 排除项）
> **Lessons-learned**: §6（artifacts 必 tracked）+ §20（Metis pre-impl review）+ §22（multi-PTX postmortem 模式）

---

## Phase 0: Artifacts Git-Tracking + Baseline（强制，per lessons-learned §6）

- [ ] 0.1 验证 OpenSpec change 目录结构完整
  ```bash
  ls openspec/changes/add-extern-function-declaration/
  # 期望: .openspec.yaml, proposal.md, design.md, tasks.md, specs/extern-function-parse-coverage/spec.md
  ```
- [ ] 0.2 在 main 上创建工作分支
  ```bash
  git checkout -b refactor/add-extern-function-declaration
  ```
- [ ] 0.3 git-tracked artifacts（**强制第一 Phase**，避免 §6 反模式）
  ```bash
  git add openspec/changes/add-extern-function-declaration/
  git status  # 应显示 5 个新文件
  ```
- [ ] 0.4 commit artifacts（独立 commit，**不要混入代码改动**）
  ```bash
  git commit -m "docs(openspec): add add-extern-function-declaration artifacts (2-phase scope)

  Metis pre-impl review applied (parser-completeness MR-7 排除项).
  Scope revised: extern function declaration parsing 已通过双路径实现
  (PtxListener.exitExternFuncStatement + PtxVisitor.visitFunctionDecl),
  本 change 仅覆盖：stale TODO 删除 + oracle test + AGENTS.md 同步.

  Refs:
  - archive/2026-07-05-parser-completeness/ (直接续集)
  - lessons-learned.md §20 (Metis pre-impl review pattern)
  - lessons-learned.md §22 (multi-PTX postmortem scope 修订模式)
  "
  ```
- [ ] 0.5 验证 git ls-files（**防 §6 反模式**）
  ```bash
  git ls-files openspec/changes/add-extern-function-declaration/
  # 期望: 至少 5 个文件，不应为空
  ```
- [ ] 0.6 验证当前 ctest baseline（main HEAD 已知 PASS）
  ```bash
  cd build && ctest -L "unit;ptx" --output-on-failure 2>&1 | tail -5
  # 期望: 100% PASS
  ```

---

## Phase 1: 删除 stale TODO + Oracle Test（Fix #1）

> **Reference**: `openspec/changes/add-extern-function-declaration/specs/extern-function-parse-coverage/spec.md`
> **Risk**: 🟢 低（删除 1 行 stale 注释 + 添加新 test）

- [ ] 1.1 创建实施 worktree
  ```bash
  cd /workspace/project/PTX-EMU
  git worktree add .worktrees/add-extern-impl refactor/add-extern-function-declaration
  cd .worktrees/add-extern-impl
  ```
- [ ] 1.2 **Fix #1.1 删除 stale TODO**（`src/ptx_parser/ptx_visitor.cpp:350`）
  - [ ] 1.2.1 替换为说明性注释：
    ```
    // function decl 由 visitFunctionDecl 直接处理（不在 declaration 上下文中）
    ```
- [ ] 1.3 **Fix #1.2 创建 oracle test**（`tests/unit/parser/test_extern_function.cpp`）
  - [ ] 1.3.1 创建测试文件，3 个 TEST_CASE：
    - (1) `extern-func-simple-form-added-to-externFuncs`
    - (2) `extern-func-with-single-param` + `extern-func-with-multiple-params`
    - (3) `visit-function-decl-distinguishes-entry-vs-extern`
- [ ] 1.4 **Fix #1.3 注册 test**（`tests/unit/CMakeLists.txt`）
  ```cmake
  add_catch_test(unit_extern_function
      parser/test_extern_function.cpp
  )
  set_tests_properties(unit_extern_function PROPERTIES LABELS "unit;parser;extern")
  ```
- [ ] 1.5 **Fix #1.4 编译验证**
  ```bash
  cmake --build build --target unit_extern_function
  ```
- [ ] 1.6 **Fix #1.5 oracle test PASS**
  ```bash
  cd build && ctest -R unit_extern_function --output-on-failure
  # 期望: 1/1 Passed
  ```
- [ ] 1.7 **Phase 1 回归验证**
  - [ ] 1.7.1 `ctest -L "unit;ptx" --output-on-failure` 100% PASS
  - [ ] 1.7.2 `./tests/ptx/test_all_ptx.sh` 100% PASS
- [ ] 1.8 **Commit Fix #1**（独立 commit）
  ```bash
  git commit -am "refactor(parser): delete stale extern TODO + add oracle test (Fix #1)

  src/ptx_parser/ptx_visitor.cpp:350:
  - Remove TODO: 'Add extern function declaration handling'
  - Replace with: 'function decl 由 visitFunctionDecl 直接处理（不在 declaration 上下文中）'
  - Reason: visitDeclaration 处理 directives + variableDecl，function decl 在 PtxFile 级别
    由 visitFunctionDecl 直接处理。stale TODO 误导未来 reader。

  tests/unit/parser/test_extern_function.cpp (new):
  - 3 TEST_CASE 覆盖 extern function 双路径
  - Test #1: simple .extern .func name → externFuncs.size()==1
  - Test #2: with-params (.param .b32 x) name → params.size()==1
  - Test #3: visitFunctionDecl 区分 entry vs extern

  Per lessons-learned §22 scope 修订模式 (parser-completeness fix #1):
  - Metis pre-impl review 揭示外部假设 vs 代码现实差异
  - Scope 从 \"实施 extern function\" 缩减为 \"覆盖既有实现的 oracle test\"

  Refs:
  - archive/2026-07-05-parser-completeness/ (直接续集 MR-7)
  - lessons-learned.md §20 (Metis pre-impl review)
  - lessons-learned.md §22 (multi-PTX postmortem)

  Co-Authored-By: Metis <metis@openspec>"
  ```

---

## Phase 2: AGENTS.md 同步 + Debt Audit RESOLVED（Fix #2）

> **Reference**: spec requirement `AGENTSMD-Extern-Function-Doc-Sync` + `No-Regression-Extern-Function`
> **Risk**: 🟢 低（仅文档同步 + 1 行 audit 更新）

- [ ] 2.1 **Fix #2.1 根 AGENTS.md 同步**
  - [ ] 2.1.1 定位 "已知限制" 章节的 extern function 条目
  - [ ] 2.1.2 更新描述：从"未处理" → "已支持（双路径：PtxListener.exitExternFuncStatement + PtxVisitor.visitFunctionDecl）"
- [ ] 2.2 **Fix #2.2 子 AGENTS.md 同步**（`src/ptx_parser/AGENTS.md`）
  - [ ] 2.2.1 在 STRUCTURE 章节增加 extern function 处理路径说明
  - [ ] 2.2.2 引用具体行号（line 486 visitFunctionDecl, line 996 exitExternFuncStatement）
- [ ] 2.3 **Fix #2.3 Debt audit RESOLVED 标记**（`docs/audits/debt-audit-2026-07-02.md`）
  - [ ] 2.3.1 定位 A-8 PARTIAL 条目
  - [ ] 2.3.2 A-8 → ✅ RESOLVED（extern 函数声明已支持 + oracle test 已添加）
- [ ] 2.4 **Fix #2.4 二次 ctest 验证**
  ```bash
  cd build && ctest --output-on-failure
  # 期望: 100% PASS
  ./tests/ptx/test_all_ptx.sh
  # 期望: 100% PASS
  ```
- [ ] 2.5 **Commit Fix #2**（独立 commit）
  ```bash
  git commit -am "docs(parser): sync AGENTS.md extern function status + audit RESOLVED (Fix #2)

  AGENTS.md (root):
  - 已知限制: extern 函数声明条目更新
  - 从 '未处理' → '已支持（双路径：PtxListener.exitExternFuncStatement + PtxVisitor.visitFunctionDecl）'

  src/ptx_parser/AGENTS.md:
  - STRUCTURE 章节增加 extern function 处理路径说明
  - 引用具体行号（line 486 visitFunctionDecl, line 996 exitExternFuncStatement）

  docs/audits/debt-audit-2026-07-02.md:
  - A-8 (PARTIAL) → ✅ RESOLVED
  - 引用 commit hash（add-extern-function-declaration Fix #1）

  Per lessons-learned §22 + §20:
  - AGENTS.md 双层同步（根 + 子）
  - Debt audit RESOLVED 引用 commit hash 而非文件路径

  Co-Authored-By: Metis <metis@openspec>"
  ```

---

## Phase 3: Archive + Merge（per Checklist G + I）

- [ ] 3.1 二次 ctest 完整验证
  ```bash
  cd build && ctest --output-on-failure
  ./tests/ptx/test_all_ptx.sh
  # 期望: 全部 100% PASS
  ```
- [ ] 3.2 Archive change（per Checklist G lifecycle）
  ```bash
  openspec archive add-extern-function-declaration --yes
  # 应自动 rename 到 openspec/changes/archive/<date>-add-extern-function-declaration/
  # 并 promote specs 到 openspec/specs/
  ```
- [ ] 3.3 Post-archive hygiene
  - [ ] 3.3.1 `git status` 无遗漏
  - [ ] 3.3.2 `git log --oneline -10` 包含 4 个 commit（artifacts + Fix #1 + Fix #2 + archive）
  - [ ] 3.3.3 清理 worktree（合并后）：
    ```bash
    git worktree remove .worktrees/add-extern-impl
    ```
- [ ] 3.4 Commit archive move（**重要，避免 §6 反模式**）
  ```bash
  git add openspec/changes/add-extern-function-declaration/ openspec/changes/archive/<date>-add-extern-function-declaration/ openspec/specs/extern-function-parse-coverage/
  git commit -m "chore(openspec): archive add-extern-function-declaration (Checklist G)

  Archive 'add-extern-function-declaration' as '<date>-add-extern-function-declaration'.
  Specs promoted to openspec/specs/.

  Co-Authored-By: Metis <metis@openspec>"
  ```
- [ ] 3.5 Merge to main
  ```bash
  git checkout main
  git merge --no-ff refactor/add-extern-function-declaration -m "Merge: add-extern-function-declaration

  2-Phase scope per parser-completeness MR-7 continuation:
  - Phase 1: stale TODO 删除 + oracle test (3 scenarios)
  - Phase 2: AGENTS.md 双层同步 + debt audit RESOLVED

  Refs:
  - archive/2026-07-05-parser-completeness/ (直接续集)
  - lessons-learned.md §20 (Metis pre-impl review)
  - lessons-learned.md §22 (multi-PTX postmortem 模式)

  Co-Authored-By: Metis <metis@openspec>"
  ```

---

## 风险缓解矩阵（per design.md Risks）

| 风险 | 缓解任务 | 验证 |
|------|---------|------|
| R1: TODO 删除后未来 reader 不知为何没有 extern 分支 | 1.2.1 说明性注释 | grep 验证 |
| R2: oracle test 失败揭示 parser bug | 1.7.1 baseline 对比 | ctest 100% PASS |
| R3: AGENTS.md 描述与代码不一致 | 1.6 oracle test PASS | 隐式验证 |
| R4: oracle test 触发 pre-existing parser bug | 1.3.1 minimal PTX 输入 | ctest 无 FAIL |

---

## Lessons-Learned 集成清单

- ✅ **Checklist D**（Commit 前）：每 Phase 独立 commit + AGENTS.md 同步
- ✅ **Checklist E**（OpenSpec 实施后）：artifacts git-tracked（Phase 0.3-0.5）
- ✅ **Checklist G**（lifecycle）：新 change + Phase 3.2 archive
- ✅ **Checklist H**（Pre-impl review）：Metis 已审计 → scope 修订
- ✅ **Checklist I**（重大功能交付）：本 change 范围小（oracle + docs）→ 仅 AGENTS.md 同步，不需根 README

---

## Open Questions（per design.md）

- **OQ-1**: oracle test 是 mock 还是端到端 → 倾向 (b) 端到端（与 parser-completeness 一致）
- **OQ-2**: debt audit A-8 RESOLVED 标记 → Phase 2 完成
- **OQ-3**: lessons-learned §22 追加 "parser 系列连续 change" → 暂不在本 change scope

---

## Quick Wins（独立 worktree，与 add-extern-function-declaration 并行）

> 用户确认方向包括 Quick Wins（5 项 P2 清理，工时 ~4h）。
> 这些可独立 commit，不混入 add-extern-function-declaration change。

### Quick Win C-11: 删除 arithmetic.cpp 注释的 assert(0)

```bash
# 位置: src/ptxsim/instructions/arithmetic.cpp:48-394
# 12 行注释掉的 assert(0)，应删除
# 验证: 删后无 lint 警告 + ctest 100% PASS
```

### Quick Win C-12: 提取 UNSUPPORTED_TYPESIZE() 宏

```bash
# 位置: bitwise.cpp / comparison.cpp / math.cpp
# 3 处重复的 `assert(0 && "Unsupported data size for...")`
# 抽到 include/ptxsim/utils/macros.h
# 验证: ctest 100% PASS
```

### Quick Win C-13: 删除 cvt_int_to_float.cpp forwarding stub

```bash
# 位置: src/ptxsim/instructions/cvt/cvt_int_to_float.cpp (56 行 forwarding)
# 仅转发到 cvt_strategy.cpp，应删除
# 验证: ctest 100% PASS
```

### Quick Win C-14: 删除 data_transfer.cpp stub

```bash
# 位置: src/ptxsim/instructions/data_transfer.cpp (32 行)
# 2 个空函数 stub，应删除
# 验证: ctest 100% PASS
```

### Quick Win C-23: build/ 清理 + .gitignore 增强

```bash
# 位置: build/ 584MB
# 清理: rm -rf build/
# 验证: .gitignore 已包含 build/
```