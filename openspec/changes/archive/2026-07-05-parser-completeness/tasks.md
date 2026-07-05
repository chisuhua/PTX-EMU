# Tasks: Parser Completeness

> **Type**: 3-Phase 清理 + 1 P0 fix（Metis pre-impl review 修订自原 10-debt + 6-Phase 计划）
> **HEAD baseline**: `5bda4eff7ba1d46eb18cee4ea15df0ef4fb44121`
> **Reference**: 关联 `archive/2026-07-05-fix-cvt-strategy-actual-split/`（Stale Artifact Fix 先例）
> **Lessons-learned**: §6（artifacts 必 tracked）+ §20（pre-impl review）+ Checklist E/G/H/I

---

## Phase 0: Artifacts Git-Tracking + Baseline（强制，per lessons-learned §6 + §4）

- [ ] 0.1 验证 OpenSpec change 目录结构完整
  ```bash
  ls openspec/changes/parser-completeness/
  # 期望: .openspec.yaml, proposal.md, design.md, tasks.md, specs/{parser-deadcode-cleanup,parser-multi-ptx-warning}/spec.md
  ```
- [ ] 0.2 在 main 上创建工作分支
  ```bash
  git checkout -b refactor/parser-completeness
  ```
- [ ] 0.3 **git-tracked artifacts**（强制第一 Phase！）
  ```bash
  git add openspec/changes/parser-completeness/
  git status  # 应显示 5 个新文件（proposal.md, design.md, tasks.md, .openspec.yaml, specs/*）
  ```
- [ ] 0.4 commit artifacts（**独立 commit**，**不要混入代码改动**）
  ```bash
  git commit -m "docs(openspec): add parser-completeness artifacts (3-phase scope)

  Metis pre-impl review applied (lessons-learned §20 + Checklist H).
  Scope revised from initial 10-debt + 6-phase to 12-debt + 3-phase:
  - Phase 1: dead code + stale comments cleanup (~15 lines, 0 behavior change)
  - Phase 2: P0 multi-PTX warning + 累加语义 (~10 lines)
  - Phase 3: archive change per Checklist G

  Refs:
  - archive/2026-07-05-fix-cvt-strategy-actual-split/ (stale artifact pattern)
  - lessons-learned.md §20 4-category classification
  - ptx-parser/multi-PTX: align parser with cubin_utils.cpp append-all behavior
  "
  ```
- [ ] 0.5 验证 git ls-files（**防 §6 反模式**）
  ```bash
  git ls-files openspec/changes/parser-completeness/
  # 期望: 至少 5 个文件，不应为空
  ```
- [ ] 0.6 建立 baseline worktree（per lessons-learned §4，预算 ~15-20 分钟全量 build）
  ```bash
  git worktree add .worktrees/parser-completeness-baseline HEAD
  cd .worktrees/parser-completeness-baseline
  . env.sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # MUST 全量！
  cd build && ctest --output-on-failure 2>&1 | tee /tmp/parser-baseline.log
  ```
- [ ] 0.7 验证 baseline 100% PASS（无 FAIL/Pending）
  ```bash
  grep -c "FAILED\|XPASS\|FAIL!" /tmp/parser-baseline.log  # 期望: 0
  ```

---

## Phase 1: 死代码 + 过期注释清理（Fix #1）

> **Reference**: `openspec/changes/parser-completeness/specs/parser-deadcode-cleanup/spec.md`
> **Risk**: 🟢 低（grep 验证 0 调用者，0 行为变更）

- [ ] 1.1 创建实施 worktree（基于 Phase 0.4 工作分支）
  ```bash
  cd /workspace/project/PTX-EMU
  git worktree add .worktrees/parser-completeness-impl refactor/parser-completeness
  cd .worktrees/parser-completeness-impl
  ```
- [ ] 1.2 **Fix #1.1 删除 calculateTypeSize 死代码**（ptx_visitor.cpp:323-326）
  - [ ] 1.2.1 二次验证 0 调用者：`grep -rn "calculateTypeSize" src/ include/ tests/`
  - [ ] 1.2.2 删除 `src/ptx_parser/ptx_visitor.cpp` 函数体 + 注释
  - [ ] 1.2.3 删除 `include/ptx_parser/ptx_visitor.h` 函数声明
  - [ ] 1.2.4 验证：`cmake --build build --target ptx_parser` 编译通过
- [ ] 1.3 **Fix #1.2 删除 processFunctionAttributes 死代码**（ptx_visitor.cpp:303-310 + line 607 注释）
  - [ ] 1.3.1 二次验证 0 调用者：`grep -rn "processFunctionAttributes" src/ include/ tests/`
  - [ ] 1.3.2 删除函数体 + 注释
  - [ ] 1.3.3 删除 `include/ptx_parser/ptx_visitor.h` 函数声明
  - [ ] 1.3.4 删除 line 607 `// TODO: Process function attributes` 注释
  - [ ] 1.3.5 验证：编译通过
- [ ] 1.4 **Fix #1.3 更新 statement_context.h:24 FIXME 注释**（描述 std::optional 语义）
  - [ ] 1.4.1 替换 `// FIXME total size in bytes` 为说明性注释
  - [ ] 1.4.2 引用 `ptx_interpreter.cpp:124-145` 作为已知 optional 处理点
- [ ] 1.5 **Fix #1.4 删除 barrier.cpp:11 Stage 3 TODO 列表**（所有子项已 ✓）
  - [ ] 1.5.1 删除 `// Stage 3 TODO: 1. ✅ ... 2. ✅ ...` 列表
  - [ ] 1.5.2 保留 Stage 3 描述（历史记录）
- [ ] 1.6 **Fix #1.5 精简 ptx_interpreter.cpp:124 BUGFIX 注释**（避免误导新读者）
  - [ ] 1.6.1 精简 `// BUGFIX: 之前使用 localDecl.size (FIXME 字段，几乎未设置)` 为 1-2 行
- [ ] 1.7 **Fix #1.6 精简 ptx_visitor.cpp:435-436 P0 cleanup 注释**（已修复的注释）
  - [ ] 1.7.1 精简为说明性注释（解释 anonymous declaration naming）
- [ ] 1.8 **Fix #1.7 标记 thread_context.cpp:171,181 dumpContext**（DEBUG STUB）
  - [ ] 1.8.1 更新 TODO 注释为 "DEBUG STUB: register dump not implemented"
- [ ] 1.9 **Fix #1.8 thread_context.cpp:410 FIXME → Design Decision 注释**
  - [ ] 1.9.1 替换 FIXME 为 Design Decision 风格说明
  - [ ] 1.9.2 解释 instr.qualifiers 是 canonical source
- [ ] 1.10 **Phase 1 验证**（per spec No-Behavior-Change-Cleanup）
  - [ ] 1.10.1 `cmake --build build` 编译通过（无 warning/error）
  - [ ] 1.10.2 `ctest --output-on-failure` 100% PASS（与 baseline 对比 0 regression）
  - [ ] 1.10.3 `./tests/ptx/test_all_ptx.sh` 100% PASS（验证无 ANTLR regression）
  - [ ] 1.10.4 `cute_rmsnorm_debug` + `barrier_warp_sync` + `cute_rmsnorm_bar_sync_pattern` 关键 e2e PASS
- [ ] 1.11 **Commit Fix #1**（独立 commit，**不要混入 Phase 2 代码**）
  ```bash
  git commit -am "refactor(parser): delete dead code + update 5 stale comments (Fix #1)

  Dead code removed:
  - calculateTypeSize (0 callers, getBytes() replacement exists in qualifier_utils.cpp:56)
  - processFunctionAttributes (0 callers, ptxParser.g4 grammar doesn't support)

  Stale comments cleaned:
  - statement_context.h:24 size FIXME (std::optional semantics correct, BUGFIX at ptx_interpreter.cpp:124-145)
  - barrier.cpp:11 Stage 3 TODO (all sub-items completed per commit f564004)
  - ptx_interpreter.cpp:124 BUGFIX (no longer needed)
  - ptx_visitor.cpp:435-436 P0 cleanup (was: TODO as identifier bug)
  - thread_context.cpp:171,181 dumpContext TODO (DEBUG STUB)
  - thread_context.cpp:410 FIXME (Design Decision per lessons-learned §6)

  Per lessons-learned §20 4-category classification (Metis pre-impl review 2026-07-05).
  0 behavior change — pure cleanup. ctest 100% PASS.

  Refs:
  - lessons-learned.md §6 (artifact tracking discipline)
  - lessons-learned.md §20 (Metis pre-impl review)
  - archive/2026-07-05-fix-cvt-strategy-actual-split/ (stale artifact pattern)
  "
  ```

---

## Phase 2: P0 fix - multi-PTX warning（Fix #2）

> **Reference**: `openspec/changes/parser-completeness/specs/parser-multi-ptx-warning/spec.md`
> **Risk**: 🟡 中（改变 PTX 加载行为，仅 multi-section cubin 受影响）

- [ ] 2.1 建立 oracle 测试基础设施（per spec MultiPTX-Parser-Oracle-Test）
  - [ ] 2.1.1 创建 `tests/unit/parser/` 目录（如不存在）
  - [ ] 2.1.2 创建 `tests/unit/parser/CMakeLists.txt`（参考现有 unit test 结构）
  - [ ] 2.1.3 注册到 `tests/unit/CMakeLists.txt`
- [ ] 2.2 **Fix #2.1 实现 oracle 测试**（先失败再实现 per TDD）
  - [ ] 2.2.1 创建 `tests/unit/parser/test_multi_ptx.cpp`
  - [ ] 2.2.2 编写 3 个测试场景：
    - (1) multi-section cubin → `PTX_WARN_EMU` 触发 + 累加
    - (2) single-section cubin → 无 warning
    - (3) 0-section cubin → 不 crash
  - [ ] 2.2.3 **验证测试失败**（Red Phase）：`ctest -R test_multi_ptx` 失败
- [ ] 2.3 **Fix #2.2 修改 ptx_parser.cpp:56-62**（P0 fix）
  - [ ] 2.3.1 添加 section 计数器（int 或 size_t）
  - [ ] 2.3.2 修改 `ptx_code = of_ptx.str();` 为 `ptx_code += of_ptx.str();`
  - [ ] 2.3.3 循环结束后：当 counter > 1 时调用 `PTX_WARN_EMU`
  - [ ] 2.3.4 warning message 包含 `"Multiple PTX sections found in cubin (count=N)"`
- [ ] 2.4 **Fix #2.3 同步 AGENTS.md**（per spec MultiPTX-Symbol-Conflict-Documented）
  - [ ] 2.4.1 根 `AGENTS.md` "已知限制" 章节：multi-PTX cubins 描述更新
  - [ ] 2.4.2 根 `AGENTS.md` "已知限制" 章节：WMMA/Tensor Core 条目二次确认（不是 stub）
- [ ] 2.5 **Phase 2 验证**
  - [ ] 2.5.1 `cmake --build build` 编译通过
  - [ ] 2.5.2 **oracle 测试通过**（Green Phase）：`ctest -R test_multi_ptx` PASS
  - [ ] 2.5.3 `ctest --output-on-failure` 100% PASS（与 baseline 对比）
  - [ ] 2.5.4 `./tests/ptx/test_all_ptx.sh` 100% PASS
  - [ ] 2.5.5 `./scripts/sanity.sh --quick` PASS
  - [ ] 2.5.6 diff baseline：0 regression
- [ ] 2.6 **Commit Fix #2**（独立 commit）
  ```bash
  git commit -am "fix(parser): multi-PTX cubin warning + 累加语义 (Fix #2)

  src/ptx_parser/ptx_parser.cpp:60 FIXED:
  - Restore += 累加 (was: = 覆盖丢失前 N-1 sections)
  - Add PTX_WARN_EMU when section count > 1
  - Align parser with cubin_utils.cpp append-all behavior (per c5 Fix #3)

  Tests:
  - tests/unit/parser/test_multi_ptx.cpp (new): 3 scenarios
    - multi-section triggers warning + 累加
    - single-section no warning
    - 0-section no crash

  Docs:
  - AGENTS.md known-limitations sync

  Per lessons-learned §1 (cross-module state translation):
  - cubin_utils.cpp already appends all sections
  - parser layer was inconsistent (覆盖 last only)
  - This fix aligns parser with cubin_utils behavior

  Refs: stub-explicit-failure spec (MultiPTX-Extraction-Warns)
  "
  ```

---

## Phase 3: 文档同步 + 二次 ctest + Archive（Fix #3，per Checklist G + I）

> **Risk**: 🟢 低（仅文档同步 + archive 操作）

- [ ] 3.1 **Fix #3.1 同步 lessons-learned.md**（新增 §P0-multi-ptx postmortem）
  - [ ] 3.1.1 在 `.opencode/skills/ptx-lessons-learned/SKILL.md` 添加 §"multi-PTX cubin 静默截断" 案例
  - [ ] 3.1.2 在 `docs/dev-process/lessons-learned.md` 添加对应段落
- [ ] 3.2 **Fix #3.2 二次 ctest 验证**（per spec No-Regression-MultiPTX）
  - [ ] 3.2.1 `ctest --output-on-failure` 100% PASS
  - [ ] 3.2.2 `./scripts/sanity.sh` 完整 sanity PASS
  - [ ] 3.2.3 diff baseline：0 regression
- [ ] 3.3 **Fix #3.3 更新 docs/audits/debt-audit-2026-07-02.md**（如有相关 P0 条目）
  - [ ] 3.3.1 检查 §P0 中是否有 multi-PTX / ptx_visitor 4 TODO 条目
  - [ ] 3.3.2 如有：标记 RESOLVED（引用 Phase 1 / Phase 2 commit hash）
- [ ] 3.4 **Fix #3.4 提交文档同步**
  ```bash
  git commit -am "docs(parser): sync lessons-learned + audit RESOLVED post-Fix #1+2 (Fix #3)

  Per lessons-learned Checklist I:
  - lessons-learned.md §multi-PTX: new postmortem case
  - debt-audit-2026-07-02.md: 标记 P0 multi-PTX RESOLVED
  - AGENTS.md: 已知限制二次确认

  Fix #3 — 文档同步
  "
  ```
- [ ] 3.5 **二次 Metis review**（optional but recommended per Checklist H）
  - [ ] 3.5.1 调用 Metis 验证最终 scope 与实现一致
  - [ ] 3.5.2 任何 ⚠️ CONDITIONAL 必须解决
- [ ] 3.6 **Archive change**（per Checklist G lifecycle）
  ```bash
  openspec archive parser-completeness --yes
  # 或手动：git mv openspec/changes/parser-completeness openspec/changes/archive/
  ```
- [ ] 3.7 **Post-archive hygiene**
  - [ ] 3.7.1 `git status` 无遗漏
  - [ ] 3.7.2 `git log --oneline -10` 包含 3 个 Fix commit + artifacts commit
  - [ ] 3.7.3 清理 worktree（可选）：
    ```bash
    git worktree remove .worktrees/parser-completeness-impl
    git worktree remove .worktrees/parser-completeness-baseline  # 仅在合并后
    ```
- [ ] 3.8 **Merge to main**
  ```bash
  git checkout main
  git merge --no-ff refactor/parser-completeness -m "Merge: parser-completeness (Fix #1-#3)

  3-Phase scope per Metis pre-impl review (2026-07-05):
  - Phase 1: 死代码 + 过期注释清理 (0 behavior change)
  - Phase 2: P0 multi-PTX warning + 累加语义
  - Phase 3: 文档同步 + archive

  Refs:
  - archive/2026-07-05-fix-cvt-strategy-actual-split/ (stale artifact pattern)
  - lessons-learned.md §20 (Metis pre-impl review)

  Co-Authored-By: Metis <metis@openspec>
  "
  ```

---

## 风险缓解矩阵（per design.md Risks / Trade-offs）

| 风险 | 缓解任务 | 验证 |
|------|---------|------|
| R1: multi-PTX 累加破坏现有 cubin | 1.10.4 + 2.5.6 | baseline diff 0 regression |
| R2: 死代码删除遗漏真实调用者 | 1.2.1 + 1.3.1 + 1.10 | grep 0 调用者 + ctest 100% |
| R3: 注释更新引入歧义 | 1.10 | Oracle review（optional） |
| R4: Oracle 测试不存在 | 2.1 + 2.2 + 2.5.2 | tests/unit/parser/ 创建 + 3 scenarios |
| R5: Phase 2 改变 PTX 加载行为 | 2.5.6 | baseline diff + e2e PASS |
| R6: artifact 提交遗漏 | 0.5 | git ls-files 验证 |

---

## Lessons-Learned 集成清单

- ✅ **Checklist D**（Commit 前）：每 Phase 独立 commit + AGENTS.md 同步 + ADR 追加（如适用）+ OpenSpec tasks.md 更新
- ✅ **Checklist E**（OpenSpec 实施后）：artifacts 已 git-tracked（Phase 0.3-0.5）
- ✅ **Checklist F**（Debt audit 撰写）：引用 commit hash 而非文件路径
- ✅ **Checklist G**（lifecycle）：本 change 是**新** change + Phase 3.6 archive
- ✅ **Checklist H**（Pre-impl review）：Metis 已审计，scope 修订为 3 Phase
- ✅ **Checklist I**（重大功能交付）：Phase 3 文档同步 + post-archive hygiene

---

## Open Questions（per design.md）

- **OQ-1**: multi-PTX 累加的符号冲突风险 → Phase 2 实施时验证
- **OQ-2**: thread_context.cpp:410 Design Decision 注释 → Phase 1 Fix #1.8 已采用
- **OQ-3**: barrier.cpp:11 Stage 3 TODO 删除后追溯 → commit message + git blame 足够