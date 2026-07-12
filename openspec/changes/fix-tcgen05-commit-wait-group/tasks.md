## 0. Pre-Implementation

- [ ] 0.1 Metis pre-implementation review (per ptx-lessons-learned §7/Checklist H):
  - 输入：当前目录下 4 个 artifacts + `src/ptxsim/instructions/tcgen05.cpp:493,512,530,550` + `src/ptx_parser/ptx_visitor.cpp:155-183,841-885` + `include/ptx_ir/statement_factory.h:265-292` + `include/ptx_ir/statement_context.h:180-190`
  - 要求：5 项 MUST-RESOLVE 全部解决才能 apply（hidden intentions / ambiguities / AI failure points / missing context）
  - 输出决策：GO / ⚠️ CONDITIONAL / ❌ NO-GO
- [ ] 0.2 验证 Oracle 引用真实存在：
  ```bash
  grep -n "tc_queue().commit(" src/ptxsim/instructions/tcgen05.cpp  # 验证 line 512 硬编码
  grep -n "tc_queue().wait(" src/ptxsim/instructions/tcgen05.cpp   # 验证 line 550 硬编码
  grep -n "(void)instr" src/ptxsim/instructions/tcgen05.cpp       # 验证 line 493,530 cast
  ```
- [ ] 0.3 验证 baseline tcgen05-tagged 测试通过:
  ```bash
  cd build && ctest -L "tcgen05" --output-on-failure
  # 预期：所有现有 tcgen05 测试 PASS（包括 sister change H1+H2 实施后状态）
  ```
- [ ] 0.4 跑 `./tests/ptx/test_all_ptx.sh` 确认 12 fixtures PASS（47/47 baseline）
- [ ] 0.5 **建立基线 worktree** (per ptx-lessons-learned §4):
  ```bash
  git worktree add .worktrees/baseline-c3 $(git rev-parse HEAD)
  cd .worktrees/baseline-c3
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # 必须全量 build
  cd build && ctest -L tcgen05 --output-on-failure
  # 预期：所有 tcgen05-tagged 测试 PASS（与 main 一致）
  ```
- [ ] 0.6 创建工作分支:
  ```bash
  git checkout -b fix/tcgen05-commit-wait-group
  ```

## 1. Phase 0 — Artifacts FIRST（per ptx-lessons-learned §6）

- [ ] 1.1 验证 4 个 OpenSpec artifacts 在 working tree:
  - `openspec/changes/fix-tcgen05-commit-wait-group/proposal.md` ✓
  - `openspec/changes/fix-tcgen05-commit-wait-group/design.md` ✓
  - `openspec/changes/fix-tcgen05-commit-wait-group/specs/tcgen05-multi-group-commit-wait/spec.md` ✓
  - `openspec/changes/fix-tcgen05-commit-wait-group/specs/tcgen05-handlers-extended/spec.md` ✓
- [ ] 1.2 `git add openspec/changes/fix-tcgen05-commit-wait-group/`
- [ ] 1.3 `git commit -m "docs(openspec): fix-tcgen05-commit-wait-group artifacts (Oracle C3 fix, Metis pre-impl review)"` (per §6 — artifacts-first 必须)
- [ ] 1.4 验证 artifacts git-tracked:
  ```bash
  git ls-files openspec/changes/fix-tcgen05-commit-wait-group/
  # 不应为空；应包含 4 个 md 文件
  ```

## 2. Phase 1 — Implementation

### 2.1 makeTcgen05Instr 加可选 cta_group 参数（per design D2）

- [ ] 2.1.1 读 `include/ptx_ir/statement_factory.h:265-292` 当前 `makeTcgen05Instr` 签名
- [ ] 2.1.2 修改签名为:
  ```cpp
  inline StatementContext makeTcgen05Instr(
      Tcgen05OpKind op_kind,
      const std::vector<Qualifier>& qualifiers,
      const std::vector<OperandContext>& operands,
      const std::string& text = "",
      uint32_t cta_group = 1);  // NEW: 默认 1 = 当前行为
  ```
- [ ] 2.1.3 在函数体内将 `cta_group` 写到 `out.tcgen05.cta_group`（per `statement_factory.h:265-292` 的赋值模式）
- [ ] 2.1.4 验证编译通过：`cmake --build build --target ptx_parser`

### 2.2 visitTcgen05Inst 加 IMMEDIATE walk（per design D1）

- [ ] 2.2.1 读 `src/ptx_parser/ptx_visitor.cpp:841-885` 当前 `visitTcgen05Inst` 实现
- [ ] 2.2.2 在 `extractQualifiersFromContext(ctx)` 调用后追加 parse tree walk:
  ```cpp
  // NEW: C3 fix — extract cta_group IMMEDIATE value
  // Grammar: TCGEN_CTA_GROUP COLONCOLON IMMEDIATE (ptxInstructions.g4:451)
  // extractQualifiersFromContext drops the IMMEDIATE child silently.
  uint32_t cta_group = 1;  // default per statement_context.h:186
  if (ctx->tcgen05QualList()) {
      for (auto* qualCtx : ctx->tcgen05QualList()->tcgen05Qual()) {
          if (qualCtx->TCGEN_CTA_GROUP() && qualCtx->IMMEDIATE()) {
              cta_group = static_cast<uint32_t>(
                  std::stoul(qualCtx->IMMEDIATE()->getText()));
          }
      }
  }
  ```
- [ ] 2.2.3 传给 `makeTcgen05Instr`:`makeTcgen05Instr(op_kind, qualifiers, operands, text, cta_group)`
- [ ] 2.2.4 **MUST 验证 ANTLR 生成代码**:`ctx->tcgen05QualList()` 与 `qualCtx->TCGEN_CTA_GROUP()` API 名拼写正确（参考生成的 `ptxParser.cpp`）
- [ ] 2.2.5 验证编译通过：`cmake --build build --target ptx_parser`

### 2.3 processTcgen05Commit 读 instr.cta_group（per design D3/D4）

- [ ] 2.3.1 读 `src/ptxsim/instructions/tcgen05.cpp:493-512` 当前 `processTcgen05Commit`
- [ ] 2.3.2 删除 `(void)instr;` 忽略
- [ ] 2.3.3 改 line 512:
  ```cpp
  // BEFORE: cta->tc_queue().commit(1);
  // AFTER:  cta->tc_queue().commit(instr.cta_group);
  cta->tc_queue().commit(instr.cta_group);  // Oracle C3 fix — read from IR
  ```
- [ ] 2.3.4 验证编译通过：`cmake --build build --target ptxsim`

### 2.4 processTcgen05Wait 读 instr.cta_group（per design D3/D4）

- [ ] 2.4.1 读 `src/ptxsim/instructions/tcgen05.cpp:530-550` 当前 `processTcgen05Wait`
- [ ] 2.4.2 删除 `(void)instr;` 忽略
- [ ] 2.4.3 改 line 550:
  ```cpp
  // BEFORE: cta->tc_queue().wait(warp, 0, 1);
  // AFTER:  cta->tc_queue().wait(warp, /*lane_id=*/0, instr.cta_group);
  cta->tc_queue().wait(warp, /*lane_id=*/0, instr.cta_group);  // Oracle C3 fix
  ```
- [ ] 2.4.4 **NOTE**: `lane_id=0` 硬编码保留（per design D3 — multi-lane wait 属 FU-3.5 子任务）
- [ ] 2.4.5 验证编译通过：`cmake --build build --target ptxsim`

## 3. Phase 2 — Tests（per specs/tcgen05-multi-group-commit-wait）

### 3.1 New integration test: commit/wait group 序列

- [ ] 3.1.1 新文件 `tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp`
- [ ] 3.1.2 添加 CMakeLists.txt 到 `tests/integration/tcgen05/CMakeLists.txt`:
  ```cmake
  add_catch_test(integration_commit_wait_group
      test_tcgen05_commit_wait_group.cpp
  )
  set_tests_properties(integration_commit_wait_group PROPERTIES
      LABELS "integration;tcgen05")
  ```
- [ ] 3.1.3 复用 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` 中的 `TestRig` + `fill_tmem_with_golden_inputs` helper（如不可复用，新建专属 helper）
- [ ] 3.1.4 TC1: `mma → commit(cta_group=2) → wait(cta_group=2) → mma` 序列立即返回（counter 满足）
  ```cpp
  TEST_CASE("processTcgen05Commit + processTcgen05Wait with cta_group=2 succeeds") {
      TestRig rig;
      auto instr_mma = make_mma_instr();
      instr_mma.cta_group = 2;  // NEW: cta_group is settable now
      auto instr_commit = make_commit_instr();
      instr_commit.cta_group = 2;
      auto instr_wait = make_wait_instr();
      instr_wait.cta_group = 2;
      
      REQUIRE_NOTHROW(rig.execute(instr_mma));
      REQUIRE_NOTHROW(rig.execute(instr_commit));
      REQUIRE_NOTHROW(rig.execute(instr_wait));  // counter=2 ≥ waited=2 → return
      REQUIRE_NOTHROW(rig.execute(instr_mma));
  }
  ```
- [ ] 3.1.5 TC2: `wait(cta_group=2) → 阻塞` 在没有 `commit(2)` 时（per Oracle 验证）
  - **NOTE**: 当前 `TcQueue::wait` 是阻塞实现，测试需要 `std::async(std::launch::async, ...)` + `future.wait_for(30s)` 检测 deadlock（per lessons-learned 失败模式速查表 "th.join() deadlock" 行）
- [ ] 3.1.6 验证编译 + 运行: `ctest -R integration_commit_wait_group --output-on-failure`

### 3.2 New parse test: cta_group::2 解析

- [ ] 3.2.1 读 `tests/integration/ptx/test_tcgen05_mma_parse.cpp` 现有 TC 结构
- [ ] 3.2.2 在同文件末尾追加新 TC:
  ```cpp
  TEST_CASE("tcgen05.mma.cta_group::2 populates instr.cta_group == 2", 
             "[integration][tcgen05][parse][cta_group]") {
      auto instr = parse_single_mma_instr(".cta_group::2");
      REQUIRE(instr.cta_group == 2u);
      REQUIRE(instr.qualifiers.size() >= 1);
      // 验证 Q_TCGEN_CTA_GROUP 在 qualifier 列表
      REQUIRE(std::find(instr.qualifiers.begin(), instr.qualifiers.end(),
                        Qualifier::Q_TCGEN_CTA_GROUP) != instr.qualifiers.end());
  }
  ```
- [ ] 3.2.3 **NOTE**: `parse_single_mma_instr` 是 helper — 若不存在需要从 `test_tcgen05_mma_persistence.cpp` 或新建 Tcgen05Instr builder 复制
- [ ] 3.2.4 验证编译 + 运行: `ctest -R test_tcgen05_mma_parse --output-on-failure`

## 4. Phase 3 — Validation

### 4.1 完整测试套件

- [ ] 4.1.1 `cmake --build build` 全量编译通过
- [ ] 4.1.2 `cd build && ctest -R "tcgen05" --output-on-failure` 全 PASS（含新 2 测试）
- [ ] 4.1.3 `cd build && ctest --output-on-failure` 全量 PASS（无回归）
- [ ] 4.1.4 `./tests/ptx/test_all_ptx.sh` 47/47 PASS（grammar 未改，预期不变）
- [ ] 4.1.5 `./scripts/sanity.sh --quick` PASS

### 4.2 Baseline 对比（per ptx-lessons-learned §4）

- [ ] 4.2.1 对比 baseline tcgen05-tagged 测试:
  ```bash
  cd .worktrees/baseline-c3/build && ctest -L tcgen05 --output-on-failure
  # 预期：所有现有测试 PASS（无 baseline 回归）
  ```
- [ ] 4.2.2 对比 main tcgen05-tagged 测试:
  ```bash
  cd build && ctest -L tcgen05 --output-on-failure
  # 预期：现有测试 PASS + 新增 2 测试 PASS（22+2 = 至少 24/24）
  ```
- [ ] 4.2.3 **失败处理**: 任何已有测试回归 → 立即 `git revert HEAD`（per lessons-learned §3）

### 4.3 ADR-0016 Postmortem 追加

- [ ] 4.3.1 读 `docs/adr/0016-blackwell-only-tcgen05.md` 找到 H1+H2 postmortem 段（per sister change）
- [ ] 4.3.2 在其后追加:
  ```markdown
  ## 2026-07-12 Postmortem: C3 fix (commit/wait group routing)
  
  ### Root Cause
  `processTcgen05Commit` (`tcgen05.cpp:493,512`) + `processTcgen05Wait`
  (`:530,550`) hardcoded `group_id=1` + `lane_id=0`. `Tcgen05Instr::cta_group`
  field (`statement_context.h:186`) was declared but never populated by
  visitor (`visitTcgen05Inst` at `ptx_visitor.cpp:858` only stored
  qualifiers as enum values, silently discarding the IMMEDIATE child of
  `TCGEN_CTA_GROUP COLONCOLON IMMEDIATE` per
  `extractQualifiersFromContext` at `ptx_visitor.cpp:155-183`).
  
  ### Fix
  1. Visit-time extraction: `visitTcgen05Inst` walks parse tree to find
     `TCGEN_CTA_GROUP` contexts and reads the IMMEDIATE child (Option (b)
     from Oracle 2026-07-11 Q5 analysis — avoids breaking 19 other
     `extractQualifiersFromContext` callers)
  2. Handler reads IR field: `processTcgen05Commit` now calls
     `commit(instr.cta_group)` instead of `commit(1)`; same for `wait`
  3. Default `cta_group=1` preserves backward compatibility for all
     existing PTX without explicit `.cta_group::N`
  
  ### Known Semantic Gaps (debt for future)
  - `tcgen05.wait N` lane_id operand not parsed (per
    `ptx_op.def:136` `op_count=0`). Future change:
    `fix-tcgen05-wait-lane-id` (or merge into FU-3 slot routing).
  - Multi-group synchronization not yet exercised by E2E test. Future:
    `tcgen05-flashattention-coverage`.
  
  Follow-up changes enabled by this fix:
  - `fix-tcgen05-idesc-parsing` (FU-2, C1)
  - `fix-tcgen05-ld-st-slot-routing` (FU-3, C2)
  - `fix-tcgen05-multi-warp-fragment` (FU-4, C4)
  - `tcgen05-flashattention-coverage` (FU-5, B1-B6 + E2E)
  ```
- [ ] 4.3.3 `git add docs/adr/0016-blackwell-only-tcgen05.md`

## 5. Phase 4 — Commit（per ptx-lessons-learned §6 artifacts-first + §3 2-Phase discipline）

- [ ] 5.1 **Commit 1 (Phase 1 — 实施)**:
  ```bash
  git add include/ptx_ir/statement_factory.h \
          src/ptx_parser/ptx_visitor.cpp \
          src/ptxsim/instructions/tcgen05.cpp \
          tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp \
          tests/integration/tcgen05/CMakeLists.txt \
          tests/integration/ptx/test_tcgen05_mma_parse.cpp
  git commit -m "fix(tcgen05): route commit/wait group_id from instr.cta_group (Oracle C3)"
  ```
- [ ] 5.2 **Commit 2 (Phase 2 — ADR postmortem)**:
  ```bash
  git add docs/adr/0016-blackwell-only-tcgen05.md
  git commit -m "docs(adr): ADR-0016 postmortem C3 fix (commit/wait group routing)"
  ```
- [ ] 5.3 验证 commits: `git show HEAD --stat` + `git log --oneline -5`
- [ ] 5.4 验证 `git revert HEAD` 后可恢复 baseline 状态（per design 回退策略）

## 6. Phase 5 — Archive（per ptx-lessons-learned §G/OpenSpec lifecycle）

- [ ] 6.1 跑 `openspec archive fix-tcgen05-commit-wait-group --yes`
- [ ] 6.2 验证: `git log --all --oneline -- "openspec/changes/fix-tcgen05-commit-wait-group/"` 应包含 archive commit
- [ ] 6.3 跑 `cd build && ctest --output-on-failure` 全量验证
- [ ] 6.4 跑 `./tests/ptx/test_all_ptx.sh` 47/47 验证
- [ ] 6.5 `git add openspec/changes/archive/` + commit "chore(openspec): archive fix-tcgen05-commit-wait-group"

## 7. Phase 6 — Postmortem Prompt（per openspec-archive-change skill）

- [ ] 7.1 **必须询问用户**: "是否生成 postmortem？(Yes/No/Defer)"
- [ ] 7.2 若 Yes: 追加 `.opencode/notes/postmortem-fix-tcgen05-commit-wait-group.md` + commit
- [ ] 7.3 若 Defer: 在 `.opencode/notes/` 留 TODO 项

## 8. Phase 7 — Final Verification

- [ ] 8.1 `cd build && ctest --output-on-failure` 全量 PASS
- [ ] 8.2 `./tests/ptx/test_all_ptx.sh` 47/47 PASS
- [ ] 8.3 `./scripts/sanity.sh` 全 PASS
- [ ] 8.4 `git log --oneline -10` 验证 4 commits 落地:
  - Phase 0: docs(openspec) artifacts
  - Phase 1: fix(tcgen05) commit/wait group routing
  - Phase 2: docs(adr) postmortem
  - Phase 5: chore(openspec) archive
- [ ] 8.5 `git worktree remove .worktrees/baseline-c3` 清理 baseline worktree

## 关键禁止（per ptx-lessons-learned §3 + §9）

- ❌ 不许跳过 baseline worktree（lessons-learned §4）
- ❌ 不许忘记 artifacts-first commit（lessons-learned §6 — `git ls-files openspec/changes/<name>/` 不应为空）
- ❌ 不许 amend 已归档 change（lessons-learned §G — 即使 sister change H1+H2 也归档后会独立）
- ❌ 不许修改 grammar/lexer（lessons-learned §9 — ANTLR bare token 风险，Oracle Q5 推荐 Option (b) 正是不改 grammar）
- ❌ 不许改 `extractQualifiersFromContext` 返回类型（19 个 call sites 回归 — Option (a) 拒绝）
- ❌ 不许 commit 中忘记删除 `(void)instr;` cast（per design D4 — 编译期错误兜底）

## Effort 估算

| Phase | Tasks | 估计时间 |
|-------|-------|----------|
| Phase 0 (Pre-impl) | Metis review + baseline worktree | 1-2h |
| Phase 1 (Implementation) | 3 文件改动 | 1-2h |
| Phase 2 (Tests) | 2 个新测试 | 1-2h |
| Phase 3 (ADR + Archive) | postmortem + commits | 30-60min |
| **总计** | | **3.5-7h** |
