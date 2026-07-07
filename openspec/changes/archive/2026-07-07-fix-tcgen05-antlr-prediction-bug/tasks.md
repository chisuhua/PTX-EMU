# Tasks: Fix tcgen05 ANTLR LL(*) Prediction Bug

> **⚠️ CANCELLED (2026-07-07)** — 此 change 未实施，因根因诊断错误。
> **实际根因**：[commit `ad808e3`](../../../) 引入的 bare lexer tokens `TCGEN_F16 : 'f16'` 与 ID 规则冲突。
> **实际修复**：[commit `55e216a`](../../../)（5 行 lexer/parser diff），同时解决了：
> - 5 个 ctest 失败（simpleGEMM-float, 2Dentropy, e2e_blackwell_gemm, cute_rmsnorm, cute_rmsnorm_debug）
> - 7 个 tcgen05 fixture LL(*) 预测失败（test_all_ptx.sh 47/47 PASS）
> **Pre-impl review (Checklist H) 未跑**：未通过 Metis 子代理验证根因假设，导致定位错误。
> **教训**：[`docs/dev-process/lessons-learned.md`](../../../docs/dev-process/lessons-learned.md) §25 + [`.opencode/skills/ptx-lessons-learned/SKILL.md`](../../../.opencode/skills/ptx-lessons-learned/SKILL.md) §9 + Checklist L
>
> 本 change 仅保留 artifacts 作为历史参考。如需重新激活，请新建 `fix-*` change（per lessons-learned §6）并先跑 Pre-impl review (Checklist H)。

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec in [specs/](specs/)
> **范围**: 3-4 atomic commits,每步独立可 revert(per `ptx-lessons-learned` §3)
> **Lessons-learned**: Checklist A(函数迁移)+ D(commit 前)+ E(artifacts tracked) + H(pre-impl review)
> **审查**: ⚠️ Sisyphus review pending — 实施前必跑

## 0. Pre-Implementation Review(强制 FIRST)

> **来源**: `ptx-lessons-learned` §7 + Checklist H — 实施 OpenSpec change 前必跑

- [ ] 0.1 跑 Metis pre-implementation review 子代理,验证:
  - [ ] 0.1.1 验证 `wc -l src/grammar/ptxInstructions.g4 src/grammar/ptxLexer.g4` 数字
  - [ ] 0.1.2 验证 `which grun` 或 ANTLR TestRig 可用性
  - [ ] 0.1.3 验证 change-3a archive 的 design.md D1 描述的根因(LL(*) 冲突)
  - [ ] 0.1.4 验证 12 个 tcgen05 fixture 中 4 个含 workaround reordering
  - [ ] 0.1.5 验证 4 个失败 case 真实存在(`tcgen05.ld.sync`、`tcgen05.mma.cta_group::1.kind::f16` 等)
  - [ ] 0.1.6 跑 `./tests/ptx/test_all_ptx.sh` 记录 baseline(46/46 PASS, 4 个 fixture 含 workaround)
  - [ ] 0.1.7 评估 design.md D1 的 3 个候选方案,确认选项 A 最优
  - [ ] 0.1.8 Metis 输出 `GO` 或 `⚠️ CONDITIONAL` 后继续

- [ ] 0.2 基线 worktree(per `ptx-lessons-learned` §4):
  - [ ] 0.2.1 `git worktree add .worktrees/baseline-antlr-fix -b feat/fix-tcgen05-antlr-prediction-bug main`
  - [ ] 0.2.2 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
  - [ ] 0.2.3 `cd build && ctest --output-on-failure` 验证 baseline (123/123)

## 1. Phase 1: 根因精确化 + 最小复现(commit 1,per `ptx-lessons-learned` §6 + Checklist E)

> **MUST**: 提交最小复现 fixture 先于修复,确认 bug 真实存在(Red phase)

- [ ] 1.1 读 `openspec/changes/archive/2026-07-07-fix-tcgen05-grammar-mr3/design.md` 了解 change-3a 上下文
- [ ] 1.2 跑 `./tests/ptx/test_all_ptx.sh` 确认 baseline (46/46)
- [ ] 1.3 跑 ANTLR grun (或 `java -cp $CLASSPATH org.antlr.v4.gui.TestRig ptxparser ptxFile -tokens`) 抓取失败 case 的 token 流
  - [ ] 1.3.1 抓 `tcgen05.ld.sync.aligned` → 记录 token 序列
  - [ ] 1.3.2 抓 `tcgen05.mma.cta_group::1.kind::f16` → 记录 token 序列
  - [ ] 1.3.3 (可选) 抓 ATN 决策图(深度诊断)
- [ ] 1.4 创建最小复现 fixture `tests/ptx/tcgen05_antlr_bug_repro.ptx`:
  ```
  // 4 个失败 case,每个 1 行
  .visible .entry test1() { tcgen05.ld.sync.aligned.32x32b.shared::cta.b32 [s], %r0; ret; }
  .visible .entry test2() { tcgen05.st.sync.aligned.32x32b.shared::cta.b32 [s], %r0; ret; }
  .visible .entry test3() { tcgen05.cp.sync.aligned.128x128b.shared::cta.b32 [s], %r0; ret; }
  .visible .entry test4() { tcgen05.mma.cta_group::1.kind::f16 %r1, %r2, %r3, %r0; ret; }
  ```
- [ ] 1.5 跑 `./tests/ptx/test_all_ptx.sh` 确认 4 个新 fixture FAIL(确认 bug 真实存在)
- [ ] 1.6 跑 `git add tests/ptx/tcgen05_antlr_bug_repro.ptx` (注意:**先不 commit**,只 add 到 working tree)
- [ ] 1.7 跑 `git status` 验证 file staged 但未 commit

## 2. Phase 2: Grammar 修复(commit 2,atomic)

> **MUST**: 修复后立即验证最小复现 PASS + 12 个现有 fixture 仍 PASS

### 2.1 编辑 grammar(per design.md D1 选项 A)

- [ ] 2.1.1 读 `src/grammar/ptxInstructions.g4` 第 443 行当前 `tcgen05Inst` 规则
- [ ] 2.1.2 改写为:
  ```antlr
  tcgen05Inst
      : TCGEN05 DOT? tcgen05SubOp tcgen05QualList typeSpecifier? tcgen05Operands? SEMI
      ;

  tcgen05QualList
      : tcgen05Qual
      | tcgen05QualList tcgen05Qual
      ;
  ```
- [ ] 2.1.3 删除原 `(DOT? tcgen05Qual)*` 形式
- [ ] 2.1.4 验证:ANTLR4 自动将左递归改写为 iterative loop,**没有 Kleene star 决策点**

### 2.2 Build + 验证

- [ ] 2.2.1 `cmake --build build --target GenerateParser` 验证 ANTLR 重新生成
- [ ] 2.2.2 `cmake --build build -j$(nproc)` 验证编译
- [ ] 2.2.3 跑 `./tests/ptx/test_all_ptx.sh` 验证:
  - [ ] 4 个最小复现 fixture PASS(Red → Green)
  - [ ] 12 个 tcgen05 fixture 仍 PASS
  - [ ] 总数 50/50 PASS (46 + 4)
- [ ] 2.2.4 跑 `ctest -L "unit|integration" --output-on-failure` 验证零回归
- [ ] 2.2.5 跑 `./scripts/sanity.sh` 全量验证
- [ ] 2.2.6 (可选) 测量 ANTLR 生成代码大小:`wc -l build/antlr4_generated_src/ptxParser.cpp` 对比 baseline

### 2.3 Commit

- [ ] 2.3.1 `git add src/grammar/ptxInstructions.g4 tests/ptx/tcgen05_antlr_bug_repro.ptx`
- [ ] 2.3.2 `git commit -m "fix(grammar): resolve tcgen05 qualifier Kleene star prediction (ADR-0016, Change-3a follow-up)"`
- [ ] 2.3.3 验证:commit 独立可 revert(`git revert HEAD` 后 Phase 1 fixtures 重新 FAIL)

## 3. Phase 3: 完整覆盖矩阵(commit 3,atomic)

> **MUST**: 新增 permutation 验证矩阵,确保未来 regression 不会再次引入

### 3.1 创建 permutation fixture

- [ ] 3.1.1 创建 `tests/ptx/tcgen05_permutations.ptx` 包含 design.md D3 验证矩阵:
  ```
  // 8 个 case 覆盖各种 qualifier 顺序
  .visible .entry p1() { tcgen05.mma.cta_group::1.kind::f16 %r1, %r2, %r3, %r0; ret; }
  .visible .entry p2() { tcgen05.mma.kind::f16.cta_group::1 %r1, %r2, %r3, %r0; ret; }
  .visible .entry p3() { tcgen05.ld.sync.aligned.32x32b.shared::cta.b32 [s], %r0; ret; }
  .visible .entry p4() { tcgen05.ld.aligned.sync.32x32b.shared::cta.b32 [s], %r0; ret; }
  .visible .entry p5() { tcgen05.mma.kind::f16.cta_group::1.block_scale %r1, %r2, %r3, %r0; ret; }
  .visible .entry p6() { tcgen05.mma.block_scale.kind::f16.cta_group::1 %r1, %r2, %r3, %r0; ret; }
  .visible .entry p7() { tcgen05.st.aligned.sync.32x32b.shared::cta.b32 [s], %r0; ret; }
  .visible .entry p8() { tcgen05.cp.multicast::cluster.aligned.128x128b.shared::cta.b32 [s], %r0; ret; }
  ```
- [ ] 3.1.2 跑 `./tests/ptx/test_all_ptx.sh` 验证 8/8 PASS

### 3.2 (可选) 恢复 4 个 workaround fixture 的 `.sync`

- [ ] 3.2.1 跑以下测试,确认 Phase 2 修复已使原 workaround 不再需要:
  ```
  tcgen05.ld.sync.aligned.32x32b.shared::cta.b32  (已在 Phase 2 通过)
  tcgen05.st.sync.aligned.32x32b.shared::cta.b32  (已在 Phase 2 通过)
  ```
- [ ] 3.2.2 若 PASS:更新 `tcgen05_ld.ptx` / `tcgen05_st.ptx` / `tcgen05_cp.ptx` / `tcgen05_cp_multicast.ptx` 恢复 `.sync`(从 Phase 2 重新 design)
- [ ] 3.2.3 若 FAIL:保留 workaround,记录到 design.md Open Questions

### 3.3 Commit

- [ ] 3.3.1 `git add tests/ptx/tcgen05_permutations.ptx` (及恢复的 fixtures,若有)
- [ ] 3.3.2 `git commit -m "test(ptx): add tcgen05 qualifier permutation coverage (ADR-0016)"`
- [ ] 3.3.3 验证:commit 独立可 revert

## 4. Phase 4: 文档同步(commit 4,per `ptx-lessons-learned` §6 + Checklist E + G)

> **MUST**: 文档同步是 lessons-learned §6 强制要求,防止"经验随归档而消失"

### 4.1 更新 lessons-learned

- [ ] 4.1.1 读 `docs/dev-process/lessons-learned.md` 当前章节列表
- [ ] 4.1.2 追加 §23 ANTLR Kleene Star 预测陷阱:
  ```markdown
  ### §23 ANTLR4 Kleene Star 预测陷阱(2026-07-07 fix-tcgen05-antlr-prediction-bug 实战)

  **问题模式**: `(X Y)*` 形式的规则在某些 token 序列后,ANTLR4 LL(*) 预测会失败,
  尽管每个独立 Y 都能解析。例如 `(DOT? tcgen05Qual)*` 在 IMMEDIATE 后不能正确
  预测 KIND/TCGEN05_SYNC。

  **关键经验**:
  - Kleene star 决策点可能在某些 token 后失效
  - 递归规则 (`list : Y | list Y`) 绕过 Kleene star 决策,ANTLR 自动改写为 loop
  - 必须用真实 fixture 验证任意 token 顺序,不能只测 happy path

  **诊断命令**:
  ```bash
  # 最小复现: 写 .ptx 含 1-2 个失败 case,跑 test_all_ptx.sh
  grun ptxparser ptxFile -tokens  # 抓 token 流
  # 深度: 用 ANTLR ATN 工具生成决策图
  ```

  **修复模式**:
  ```antlr
  // BEFORE (Kleene star,可能 LL(*) 失效)
  rule : X (Y)* Z ;
  // AFTER (递归,无 Kleene star 决策点)
  rule : X YList Z ;
  YList : Y | YList Y ;
  ```
  ```

### 4.2 更新 change-3a archive

- [ ] 4.2.1 读 `openspec/changes/archive/2026-07-07-fix-tcgen05-grammar-mr3/design.md`
- [ ] 4.2.2 D1 追加一行:`**RESOLVED** by commit <hash> in fix-tcgen05-antlr-prediction-bug (2026-07-XX)`
- [ ] 4.2.3 移除 handoff.md 中的"Kleene star 预测已知限制"标记

### 4.3 Archive 当前 change

- [ ] 4.3.1 跑 `openspec archive fix-tcgen05-antlr-prediction-bug --yes`
- [ ] 4.3.2 跑 `openspec status` 确认 change 已 archive
- [ ] 4.3.3 跑 `git log --oneline feat/fix-tcgen05-antlr-prediction-bug` 确认 4 个 atomic commits

### 4.4 Commit

- [ ] 4.4.1 `git add docs/dev-process/lessons-learned.md openspec/changes/archive/2026-07-07-fix-tcgen05-grammar-mr3/ openspec/specs/ openspec/changes/archive/2026-07-07-fix-tcgen05-antlr-prediction-bug/`
- [ ] 4.4.2 `git commit -m "docs(openspec): archive fix-tcgen05-antlr-prediction-bug + lessons-learned §23 (ADR-0016)"`

## 5. Final Validation

- [ ] 5.1 `./scripts/sanity.sh` 全量验证
- [ ] 5.2 `./scripts/sanity.sh --ptx` PTX 语法验证(应 50/50 PASS)
- [ ] 5.3 `cd build && ctest --output-on-failure` 全量测试(应 123/123)
- [ ] 5.4 验证:`git log --oneline feat/fix-tcgen05-antlr-prediction-bug` 显示 4 个 atomic commits

## Risks & Mitigations Recap

| Risk | Mitigation in Tasks |
|------|---------------------|
| **R1**: 选项 A 递归规则导致 ANTLR 代码大小爆炸 | Task 2.2.6 测量;若爆炸回退到选项 B |
| **R2**: 修复后某些 case 仍失败 | Phase 2 立即验证所有 D3 matrix;回退选项 B |
| **R3**: 现有 12 个 fixture regression | Task 2.2.3 严格 TDD;任何 fail 立即 revert |
| **R4**: lessons-learned 经验未沉淀 | Phase 4 强制 commit lessons-learned §23 |
| **R5**: Metis 建议被忽略 | Phase 0.1 强制 Metis pre-impl review |

## Out-of-Scope Reminder(per [proposal.md](proposal.md))

- ❌ 不重写整个 ANTLR grammar
- ❌ 不修改 wmma/handler/visitor 代码
- ❌ 不删除 `S_TCGEN05_MMA_WS` dead code(与本 bug 无关)
- ❌ 不实现 tcgen05 handler(change-3b scope)
- ❌ 不引入 lexer mode 到整个 grammar(仅在 tcgen05 scope 考虑)

## Handoff Verification

- [ ] 6.1 确认 handoff.md 创建,列 3 项 deferred items:
  - (1) `tcgen05_permutations.ptx` 作为未来 regression 测试覆盖
  - (2) 任何后续 grammar 修改需跑 `test_all_ptx.sh` + `tcgen05_permutations.ptx` 双验证
  - (3) lessons-learned §23 作为未来 ANTLR 重构的参考
- [ ] 6.2 验证 design.md D1 选项 A 正确实施
- [ ] 6.3 跑 `ptx-lessons-learned` skill checklist(16 items);确认无遗漏
