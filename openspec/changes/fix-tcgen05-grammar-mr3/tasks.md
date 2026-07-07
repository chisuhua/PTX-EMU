# Tasks: Fix tcgen05 Grammar LL(*) Conflict + Migrate Old Tests

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 3 specs in [specs/](specs/)
> **范围**: 6 atomic commits,每步独立可 revert(per `ptx-lessons-learned` §3)
> **Lessons-learned**: Checklist E(artifacts tracked) + Checklist G(OpenSpec lifecycle)
> **审查**: Sisyphus review 2026-07-07 → ⚠️ CONDITIONAL(P0-P2 resolved, 本文件反映修复后状态)

## 0. Pre-Implementation Review(强制 FIRST)

> **来源**: `ptx-lessons-learned` §7 + Checklist H — 实施 OpenSpec change 前必跑

- [ ] 0.1 跑 Metis pre-implementation review 子代理,验证:
  - [ ] 0.1.1 `wc -l src/grammar/ptxInstructions.g4 src/grammar/ptxLexer.g4` 数字
  - [ ] 0.1.2 验证 `tcgen05Qual` 规则 16+ alternations(per change-1 design.md)
  - [ ] 0.1.3 验证 `tests/ptx/tcgen05_alloc.ptx tcgen05_mma.ptx` 当前 fail(`mismatched input '.all'`)
  - [ ] 0.1.4 验证 2 个旧测试引用 `S_WMMA`/`makeWmmaInstr`/`WmmaType`
  - [ ] 0.1.5 验证 `Q_TCGEN05_*` 4 stub 位置(`include/ptx_ir/ptx_qualifier.def:196-199`)
  - [ ] 0.1.6 跑 `./tests/ptx/test_all_ptx.sh` 记录 baseline(2 fail)
  - [ ] 0.1.7 验证 spec `tcgen05-old-test-migration/spec.md` 使用 additive 语义(per Sisyphus review P0 fix)
  - [ ] 0.1.8 Metis 输出 `GO` 或 `⚠️ CONDITIONAL` 后继续

- [ ] 0.2 基线 worktree(per `ptx-lessons-learned` §4):
  - [ ] 0.2.1 `git worktree add .worktrees/baseline-grammar-fix -b feat/fix-tcgen05-grammar-mr3 main`
  - [ ] 0.2.2 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
  - [ ] 0.2.3 `cd build && ctest --output-on-failure` 验证 baseline

## 1. Artifacts Tracking(commit 1,per `ptx-lessons-learned` §6 + Checklist E)

- [ ] 1.1 检查已存在的 feat/fix-tcgen05-grammar-mr3 分支:
        `git checkout feat/fix-tcgen05-grammar-mr3`
        (NOTE: 分支由 Task 0.2.1 worktree 创建,无需 `-b`)
- [ ] 1.2 `git add openspec/changes/fix-tcgen05-grammar-mr3/{proposal.md,design.md,tasks.md,specs/}`
- [ ] 1.3 `git status` 验证 5 个文件全部 staged(proposal + design + tasks + 3 specs)
- [ ] 1.4 `git ls-files openspec/changes/fix-tcgen05-grammar-mr3/` 验证非空
- [ ] 1.5 `git commit -m "docs(openspec): add fix-tcgen05-grammar-mr3 artifacts (ADR-0016)"`
- [ ] 1.6 NOTE:此 commit 独立可 revert(删除 openspec/changes/fix-tcgen05-grammar-mr3/)

## 2. Phase 1: Grammar 修复(commit 2,atomic)

> **MUST**: 修复后立即验证 grammar 编译 + 2 fixture PASS

### 2.1 诊断 LL(*) 冲突根因

- [ ] 2.1.1 读 `src/grammar/ptxInstructions.g4` 当前 `tcgen05Qual` 规则(16+ alternatives)
- [ ] 2.1.2 跑 `antlr4 src/grammar/ptxParser.g4` 单独编译,查看 ANTLR 警告/错误
- [ ] 2.1.3 分析根因(per `ptx-grammar-modification` skill):

  > **已知根因(基于 2026-07-07 代码审查):**
  >
  > `tcgen05Qual` 中 token 的 dot-prefix 不一致:
  > - **自带 dot 的 token**: ALIGNED(`'.aligned'`), SHARED(`'.shared'`), TCGEN_CTA_GROUP(`'.cta_group'`) 等 — token 已包含前导点
  > - **不带 dot 的 token**: TCGEN05_SYNC(`'sync'`), TCGEN_SP(`'sp'`), TCGEN_WS(`'ws'`) 等 — token 不含前导点
  >
  > 对于 `.sync.aligned...` 输入: `.sync` 被 lexer 拆为 `DOT + TCGEN05_SYNC`(因为 TCGEN05_SYNC 不含 dot),而 `.aligned` 是单 token `ALIGNED`。`tcgen05Qual*` 循环遇到 `DOT` 时无法将其匹配到任何 qualifier alternative → `mismatched input '.all' expecting ':'`(ANTLR 报告时取 raw text)。
  >
  > **修复方向**:
  > - **方案 A(推荐)**: 将所有不带 dot 的 qualifier token 统一加上前导点(`'sync' → '.sync'`, `'sp' → '.sp'`, `'ws' → '.ws'`) — 最小侵入,对齐其他 qualifier 的 token 风格
  > - **方案 B**: 在 `tcgen05Qual` 中为不带 dot 的 token 添加 `DOT?` 可选前缀 — 增加 grammar 复杂度,不推荐
  >
  > ⚠️ 若选方案 A,需同步更新:
  > - `src/grammar/ptxLexer.g4`: TCGEN05_SYNC(`.sync`), TCGEN_SP(`.sp`), TCGEN_WS(`.ws`)
  > - `include/ptx_ir/ptx_qualifier.def`: X-Macro 第 4 列的注释字符串同步

- [ ] 2.1.4 选择修复方案(推荐方案 A),记录决策理由到 design.md §D1

### 2.2 修复 grammar

- [ ] 2.2.1 统一 dot-prefix token(per §2.1.3 方案 A):
        - `src/grammar/ptxLexer.g4`: `TCGEN05_SYNC: '.sync'` | `TCGEN_SP: '.sp'` | `TCGEN_WS: '.ws'`
        - `include/ptx_ir/ptx_qualifier.def`: 第 4 列注释字符串同步(如 `".tcgen05_sync"` → `".sync"`)
- [ ] 2.2.2 调整 `tcgen05Qual` alternatives 顺序(高歧义子集前置):将 `KIND COLONCOLON`、`TCGEN_CTA_GROUP COLONCOLON`、`SHARED COLONCOLON` 移到最前(有 `::` 的 pattern 先行决策)
- [ ] 2.2.3 调整 `typeSpecifier` 位置:已从 `tcgen05Qual` 移除,在主规则 `tcgen05Inst` 显式位置(`tcgen05Qual* typeSpecifier?`)

### 2.3 Build + 验证

- [ ] 2.3.1 `cmake --build build --target GenerateParser` 验证 ANTLR 重新生成
- [ ] 2.3.2 `cmake --build build` 验证编译
- [ ] 2.3.3 `./tests/ptx/test_all_ptx.sh` 验证 2 个现有 fixture(`tcgen05_alloc.ptx` + `tcgen05_mma.ptx`)PASS
- [ ] 2.3.4 `ctest -L "unit|integration" --output-on-failure` 验证零回归
- [ ] 2.3.5 `git diff --stat` 验证 diff 在 `src/grammar/` + `include/ptx_ir/ptx_qualifier.def` 范围内(不触及 IR 或 handler)

### 2.4 Commit

- [ ] 2.4.1 `git add src/grammar/ include/ptx_ir/ptx_qualifier.def`
- [ ] 2.4.2 `git commit -m "fix(grammar): resolve tcgen05 LL(*) prediction conflict (ADR-0016, Change-1 MR-3)"`
- [ ] 2.4.3 验证:commit 独立可 revert(`git revert HEAD` 后 grammar baseline 仍工作)

## 3. Phase 2: 补全 .ptx fixtures(commit 3,atomic)

> **MUST**: 10 个新 fixture 必须全部通过 `test_all_ptx.sh`

### 3.1 创建 10 个新 fixtures(基于 PTX ISA 8.6 §9.7.16)

- [ ] 3.1.1 创建 `tests/ptx/tcgen05_dealloc.ptx`
- [ ] 3.1.2 创建 `tests/ptx/tcgen05_relinquish.ptx`
- [ ] 3.1.3 创建 `tests/ptx/tcgen05_ld.ptx`
- [ ] 3.1.4 创建 `tests/ptx/tcgen05_st.ptx`
- [ ] 3.1.5 创建 `tests/ptx/tcgen05_cp.ptx`
- [ ] 3.1.6 创建 `tests/ptx/tcgen05_cp_multicast.ptx`
- [ ] 3.1.7 创建 `tests/ptx/tcgen05_mma_block_scale.ptx`
- [ ] 3.1.8 创建 `tests/ptx/tcgen05_commit.ptx`
- [ ] 3.1.9 创建 `tests/ptx/tcgen05_wait.ptx`
- [ ] 3.1.10 创建 `tests/ptx/tcgen05_fence.ptx`

### 3.2 Build + 验证

- [ ] 3.2.1 `cmake --build build` 验证编译
- [ ] 3.2.2 `./tests/ptx/test_all_ptx.sh` 验证 12/12 PASS(2 现有 + 10 新)
- [ ] 3.2.3 `ls tests/ptx/tcgen05_*.ptx | wc -l` 应输出 12
- [ ] 3.2.4 验证 `test_all_ptx.sh` glob 自动发现新 fixtures:
        `grep -E "tcgen05|\.ptx" tests/ptx/test_all_ptx.sh` (确认 glob 覆盖 `tests/ptx/tcgen05_*.ptx`)

### 3.3 Commit

- [ ] 3.3.1 `git add tests/ptx/`
- [ ] 3.3.2 `git commit -m "test(ptx): add 10 tcgen05.* PTX fixtures (ADR-0016)"`
- [ ] 3.3.3 验证:commit 独立可 revert(`git revert HEAD` 后 fixtures 消失但 grammar 仍工作)

## 4. Phase 3a: B2 factory fix — `makeTcgen05Instr` op_kind dispatch (commit 4a)

> **MUST run BEFORE Phase 3b(test migration)** — test migration 使用的 `makeTcgen05Instr` 依赖正确的 factory 派发

> **Context**: `include/ptx_ir/statement_factory.h:278-289` 当前硬编码
  `static_cast<StatementType>(S_TCGEN05_MMA)`,导致所有 11 个 Tcgen05OpKind
  变体(ALLOC/DEALLOC/RELINQUISH/LD/ST/CP/MMA/MMA_WS/COMMIT/WAIT/FENCE)
  都解析为同一个 `S_TCGEN05_MMA` statement type。
  枚举定义见 `include/ptx_ir/statement_context.h:169`
  以及 `include/ptx_ir/ptx_types.h:28-38` (S_TCGEN05_* enum 值)。

- [ ] 4.1 读 `include/ptx_ir/statement_factory.h:278-289`
- [ ] 4.2 编辑: 将硬编码的 `static_cast<StatementType>(S_TCGEN05_MMA)` 替换为
        `switch (op_kind)` 映射到全部 11 个 `S_TCGEN05_*` 值:
        * `Tcgen05OpKind::ALLOC → S_TCGEN05_ALLOC`
        * `Tcgen05OpKind::DEALLOC → S_TCGEN05_DEALLOC`
        * `Tcgen05OpKind::RELINQUISH → S_TCGEN05_RELINQUISH`
        * `Tcgen05OpKind::LD → S_TCGEN05_LD`
        * `Tcgen05OpKind::ST → S_TCGEN05_ST`
        * `Tcgen05OpKind::CP → S_TCGEN05_CP`
        * `Tcgen05OpKind::MMA → S_TCGEN05_MMA`
        * `Tcgen05OpKind::MMA_WS → S_TCGEN05_MMA_WS`
        * `Tcgen05OpKind::COMMIT → S_TCGEN05_COMMIT`
        * `Tcgen05OpKind::WAIT → S_TCGEN05_WAIT`
        * `Tcgen05OpKind::FENCE → S_TCGEN05_FENCE`
- [ ] 4.3 `cmake --build build` 验证编译零错误
- [ ] 4.4 `ctest -L "unit|integration" --output-on-failure` 必须全 PASS(factory 改动影响所有用 makeTcgen05Instr 的代码路径)
- [ ] 4.5 `./tests/ptx/test_all_ptx.sh` 必须全 PASS(grammar 修复 + factory 修复组合验证)
- [ ] 4.6 验证:switch cases 与 `Tcgen05OpKind` 枚举完全对应
        (见 `include/ptx_ir/statement_context.h:169`)
- [ ] 4.7 `git add include/ptx_ir/statement_factory.h`
- [ ] 4.8 `git commit -m "fix(factory): switch makeTcgen05Instr on op_kind for all 11 S_TCGEN05_* types"`
- [ ] 4.9 验证:commit 独立可 revert(`git revert HEAD` 后 factory 回到硬编码 S_TCGEN05_MMA,旧 WMMA 路径仍工作)

## 5. Phase 3b: Compile-time alias verification (commit 4b)

> **依赖**: commit 4a(B2 factory fix)已完成 — 测试使用的 `makeTcgen05Instr` 现在正确派发 11 种 op_kind

> **MUST**: 添加 `makeTcgen05Instr` 编译期别名验证(不加入执行向量),旧路径不变;所有测试仍 PASS(behavior 不变)
>
> **关键约束**: 本 change 无 `S_TCGEN05_*` handler (`get_handler()` 返回 nullptr)。新别名**仅做编译期类型验证**,不插入 `step_warp` 执行向量。实现模式见 `design.md` D3。

### 5.1 添加 tcgen05 编译期别名到 test_tcgen05_mma_sync.cpp

- [ ] 5.1.1 读 `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp`
- [ ] 5.1.2 在现有 `makeWmmaInstr(WmmaType::WMMA_MMA, ...)` 调用**旁**(不修改执行向量)添加编译期别名:
        ```cpp
        // Compile-time alias: verify makeTcgen05Instr factory + B2 op_kind switch
        auto tcgen05_alias = makeTcgen05Instr(Tcgen05OpKind::MMA, quals, {});
        static_assert(std::is_same_v<decltype(tcgen05_alias), StatementContext>);
        // NOTE: implement-tcgen05-handlers-core will replace stmts[MMA_PC] with this
        ```
- [ ] 5.1.3 不删除旧 `makeWmmaInstr`/`WmmaType` 调用,不修改执行向量 `stmts[...]` 赋值
- [ ] 5.1.4 验证:编译通过 = factory fix 正确;ctest PASS = 旧路径未受影响

### 5.2 添加 tcgen05 编译期别名到 test_tcgen05_ld_st_commit.cpp

- [ ] 5.2.1 读 `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`
- [ ] 5.2.2 在现有 5 处 `makeWmmaInstr(WmmaType::WMMA_*, ...)` 调用旁添加等效编译期别名(同上模式,不插入执行向量)
- [ ] 5.2.3 验证:编译通过 + ctest -R tcgen05 PASS(旧路径执行不变)

### 5.3 Build + 验证

- [ ] 5.3.1 `cmake --build build` 验证编译
- [ ] 5.3.2 `ctest -R tcgen05 -V` 验证 2 个旧测试 PASS(旧路径 + 新别名共存)
- [ ] 5.3.3 `ctest -L "unit|integration" --output-on-failure` 验证零回归
- [ ] 5.3.4 `./tests/ptx/test_all_ptx.sh` 验证 12/12 fixtures 仍 PASS

### 5.4 Commit

- [ ] 5.4.1 `git add tests/integration/tcgen05/`
- [ ] 5.4.2 `git commit -m "test(refactor): add tcgen05 aliases alongside existing WMMA paths (ADR-0016)"`
- [ ] 5.4.3 验证:commit 独立可 revert(`git revert HEAD` 后旧 WMMA 路径仍工作,新别名消失)

## 6. Phase 4: Archive(commit 5,per Checklist G)

- [ ] 6.1 跑 `openspec archive fix-tcgen05-grammar-mr3 --yes`
- [ ] 6.2 跑 `ctest --output-on-failure` + `test_all_ptx.sh` 最终验证
- [ ] 6.3 跑 `openspec status` 确认 change 已 archive
- [ ] 6.4 `git add openspec/changes/archive/`
- [ ] 6.5 `git commit -m "chore(openspec): archive fix-tcgen05-grammar-mr3 (ADR-0016)"`
- [ ] 6.6 验证:`git log --oneline | head -6` 显示 6 个 atomic commits(commit 1 artifacts + commit 2 grammar + commit 3 fixtures + commit 4a factory + commit 4b migration + commit 5 archive)

## 7. Final Validation

- [ ] 7.1 `./scripts/sanity.sh` 全量验证
- [ ] 7.2 `./scripts/sanity.sh --ptx` PTX 语法验证
- [ ] 7.3 `cd build && ctest --output-on-failure` 全量测试
- [ ] 7.4 验证:`git log --oneline feat/fix-tcgen05-grammar-mr3` 显示 6 个 atomic commits

## Risks & Mitigations Recap

| Risk | Mitigation in Tasks |
|------|---------------------|
| **R1**: Grammar 修复后 ANTLR 错误 | Task 2.3.1-2.3.2 立即验证 |
| **R2**: 12 fixtures 中某些无法匹配 PTX 规范 | Task 3.2.2 跑 test_all_ptx.sh,失败单独调试 |
| **R3**: 旧测试迁移 behavior 变化 | Task 5.3.2 跑 ctest -R tcgen05 -V 验证 |
| **R4**: statement_factory.h B2 修复后 switch 不匹配 Tcgen05OpKind | Task 4.3-4.4 编译 + ctest 验证 |

## Out-of-Scope Reminder(per [proposal.md](proposal.md))

- ❌ 不实施 tcgen05 handler(change-3b scope)
- ❌ 不修改 wmma.cpp 中 5 个 execute_tcgen05_(change-3b scope)
- ❌ 不实现 cp.async.bulk.tensor(独立 follow-up)
- ❌ 不实现 cta_group::2 distributed_smem(独立 follow-up)
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8

## 8. Handoff Verification

- [ ] 8.1 Confirm handoff.md created with all 6 deferred items
- [ ] 8.2 Verify design.md D4 uses Oracle Strategy D text(含 wmma.cpp:29-59 + 2 测试文件引用)
- [ ] 8.3 Run `ptx-lessons-learned` skill checklist(16 items); confirm no regressions
- [ ] 8.4 Verify §4 factory fix runs BEFORE §5 test migration (commit 4a before 4b) per Sisyphus review P1 fix