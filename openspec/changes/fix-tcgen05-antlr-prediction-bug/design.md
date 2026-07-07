# Fix tcgen05 ANTLR LL(*) Prediction Bug — Design

## Context

Change-3a (`fix-tcgen05-grammar-mr3`, archived 2026-07-07) 修复了基础 grammar LL(*) 冲突（`tcgen05.alloc` vs `ALL` token 冲突等），使 `test_all_ptx.sh` 达到 46/46 PASS。

但**通过 fixture reordering 绕过了**一个**更深的 ANTLR4 预测 bug**：

- 失败的 case 都是 `TCGEN05_SYNC` (`.sync`) 或 `KIND COLONCOLON tcgen05Dtype` (`.kind::f16`) 出现在 `IMMEDIATE` (数字) **之后**
- 例如 `tcgen05.cta_group::1.kind::f16` 失败；`tcgen05.mma.kind::f16.cta_group::1` 通过
- 例如 `tcgen05.ld.sync.aligned` 失败；`tcgen05.mma.sync.aligned` 通过
- 这违反了 PTX ISA §9.7.16 的"qualifiers 任意顺序"原则

**当前 workaround**: 4 个 fixture 中 reordering qualifiers（例如 `tcgen05.mma_block_scale.ptx` 中将 `.block_scale` 提前）。但这只是临时绕过，不是真正的 grammar 修复。

**目标**: 修复 `tcgen05Inst` 规则的 Kleene star 预测问题，恢复 qualifier 任意顺序支持。

## Goals / Non-Goals

**Goals**: 修复 ANTLR4 LL(*) 预测在 `tcgen05Inst` 规则 Kleene star 决策点失效的 bug；恢复 PTX qualifier 任意顺序通过；不引入 lexer mode 切换的全局影响。

**Non-Goals**: 不重写整个 grammar；不修改 wmma/handler/visitor 代码；不删除 `S_TCGEN05_MMA_WS` dead code（与本 bug 无关）；不实现 tcgen05 handler（change-3b scope）。

## Decisions

### D1: 修复方案选择 — 3 候选对比

**选项 A(采纳候选)**: 递归重写 `tcgen05Inst` 规则消除 Kleene star

将 `TCGEN05 DOT? tcgen05SubOp (DOT? tcgen05Qual)* typeSpecifier? tcgen05Operands? SEMI` 重写为：
```antlr
tcgen05Inst
    : TCGEN05 DOT? tcgen05SubOp tcgen05QualList typeSpecifier? tcgen05Operands? SEMI
    ;

tcgen05QualList
    : tcgen05Qual
    | tcgen05QualList tcgen05Qual
    ;
```

**理由**:
- 递归左递归形式在 ANTLR4 中被自动改写为 iterative loop，**没有 Kleene star 决策点**
- 不需要修改 token 定义
- 最小侵入（仅重写规则结构）
- ANTLR4 优化保证生成的代码等价

**风险**:
- 递归深度可能受 `setMaxRecursiveRuleAlt` 限制（默认 100，足够）
- 需要验证 12 个现有 fixture 仍 PASS

**选项 B**: 引入 lexer mode `TCGEN05_QUALIFIER_MODE`

```antlr
lexer grammar ptxLexer {
    ...
    TCGEN05_QUALIFIER_MODE : 'mode' -> mode(DEFAULT_MODE);
    
    mode TCGEN05_QUALIFIER_MODE;
    TCGEN05_QUAL_SYNC : '.sync' -> ...;
    TCGEN05_QUAL_KIND : '.kind' -> ...;
    ...
}
```

**理由**: 显式控制 qualifier 词法分析，消除歧义。

**风险**:
- 需要修改 `ptxLexer.g4` 大幅（~50-100 LoC）
- 引入 lexer mode 机制对整个 grammar 维护性有影响
- ANTLR4 lexer mode 在嵌入式使用中可能与 parser 协调问题

**拒绝理由**: 风险/收益不匹配，选项 A 侵入更小。

**选项 C**: 拆分 `tcgen05Qual` 为多个子规则

将 `tcgen05Qual` 拆为 `tcgen05SimpleQual`（无 `COLONCOLON`）+ `tcgen05CtypeQual`（带 `COLONCOLON`），并显式列出预测路径：

```antlr
tcgen05Qual : tcgen05SimpleQual | tcgen05CtypeQual ;
```

**理由**: 通过细分决策点帮助 ANTLR 预测。

**风险**:
- 仍依赖 ANTLR 优化能力，可能不能根本解决
- 拆分后 maintenance cost 略增
- 不保证解决所有失败 case

**拒绝理由**: 没有根本改变 Kleene star 决策点结构，预测问题可能仍存在。

### D2: 修复验证策略 — TDD + ANTLR 工具链

**采纳**:
1. **Phase 1**: 用 ANTLR `grun` (或 TestRig) 抓取失败 case 的 token 流和 ATN 决策图
2. **Phase 2**: 先写最小复现 fixture（1 条 instruction），确认修复后 PASS
3. **Phase 3**: 跑全 12 个 tcgen05 fixture + 46 个 .ptx 总数，验证零回归
4. **Phase 4**: 新增 `tcgen05_permutations.ptx`（覆盖之前失败的 qualifier 顺序）

**理由**: 严格遵循 TDD，先建立 failing test，再修复，最后回归。符合 `ptx-grammar-modification` skill 的强制 TDD 流程。

### D3: 失败 case 量化 — 测试覆盖矩阵

**采纳**: 新增 `tests/ptx/tcgen05_permutations.ptx` 包含以下 instruction 变体（每行一个）：

```
// .kind::f16 在 .cta_group::1 之后
tcgen05.mma.cta_group::1.kind::f16 [%rd0], %rd1, %rd2, %r0;

// .kind::f16 在 .cta_group::1 之前
tcgen05.mma.kind::f16.cta_group::1 [%rd0], %rd1, %rd2, %r0;

// .sync 在 .aligned 之后
tcgen05.ld.sync.aligned.32x32b.shared::cta.b32 [smem_alloc], [%rd0];

// .block_scale 在 .kind::f16 之后
tcgen05.mma.kind::f16.cta_group::1.block_scale [%rd0], %rd1, %rd2, %r0;
```

**理由**: 显式测试失败 case，确保未来 regression 不会再次引入。

### D4: 不破坏现有行为

**采纳**:
- 选项 A 递归重写保持相同的语义（ANTLR4 自动优化）
- 修改前后 12 个现有 fixture 全部应保持 PASS
- 任何意外 break 立即 revert Phase 2

**理由**: 符合 lessons-learned §3 "已有测试回归 → 立即 revert 该 Phase"。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: 选项 A 递归规则导致 ANTLR 生成代码大小爆炸 | 用 `setMaxRecursiveRuleAlt` 限制；benchmark 验证 |
| **R2**: 修复后某些 case 仍失败（ANTLR4 优化限制） | 退到选项 B（lexer mode） |
| **R3**: 现有 12 个 fixture 出现 regression | 严格 TDD + 立即 revert |
| **R4**: 实施者误删重要注释（per change-3a D2）| tasks.md 明确列出保留/删除项 |

## Migration Plan

### Phase 1: 根因精确化(1 commit)

1. 读 change-3a archive 的 design.md D1 了解历史
2. 跑 `./tests/ptx/test_all_ptx.sh` 确认 baseline (46/46 PASS)
3. 跑 ANTLR grun 抓取 `tcgen05.ld.sync.aligned` 的 token 流（如果有 antlr4 命令）
4. 在 `tests/ptx/` 写最小复现 fixtures (3-5 个) 包含失败 case
5. 确认这些 fixtures 在当前 grammar 下 FAIL（验证 bug 仍存在）
6. commit: `test(ptx): add minimal repro for tcgen05 qualifier ordering bug`

### Phase 2: 修复(1 commit, 方案 A)

1. 编辑 `src/grammar/ptxInstructions.g4`:
   - 删除 `(DOT? tcgen05Qual)*` 形式
   - 添加 `tcgen05QualList` 递归规则
2. 跑 `cmake --build build --target GenerateParser` 验证 ANTLR 重新生成
3. 跑 Phase 1 的最小复现 fixtures — 应全部 PASS
4. 跑 `./tests/ptx/test_all_ptx.sh` 验证 46/46 仍 PASS
5. 跑 `ctest -L "unit|integration"` 验证零回归
6. 跑 `./scripts/sanity.sh` 全量验证
7. commit: `fix(grammar): resolve tcgen05 qualifier Kleene star prediction (ADR-0016)`

### Phase 3: 复测与回归(1 commit)

1. 创建 `tests/ptx/tcgen05_permutations.ptx` 包含 D3 验证矩阵
2. 跑 `./tests/ptx/test_all_ptx.sh` 验证 47/47 PASS
3. 更新现有 4 个 workaround fixture（`tcgen05_ld/st/cp/cp_multicast`）恢复 `.sync`（若实现 D3 矩阵覆盖后已 PASS）
4. 跑 `./scripts/sanity.sh` 全量验证
5. commit: `test(ptx): add tcgen05 qualifier permutation coverage`

### Phase 4: 文档同步(1 commit, per lessons-learned §6)

1. 更新 `docs/dev-process/lessons-learned.md` 追加 §23 ANTLR Kleene Star 预测陷阱
2. 更新 `openspec/changes/archive/2026-07-07-fix-tcgen05-grammar-mr3/design.md` D1 标记为 RESOLVED
3. 跑 `openspec archive fix-tcgen05-antlr-prediction-bug --yes`
4. commit: `docs(openspec): archive fix-tcgen05-antlr-prediction-bug + lessons-learned §23`

### 回退策略

- Phase 1 失败: `git revert HEAD` 回到 baseline worktree
- Phase 2 失败: `git revert HEAD` → 保留 fixtures → 改用选项 B（lexer mode）
- Phase 3 失败: 保留 fix 但标注 D3 验证矩阵作为 deferred
- 整体失败: `git reset --hard <pre-change-sha>` (前提 baseline worktree 已验证)

## Open Questions

- **Q1**: 递归重写 vs 现有 `(DOT? tcgen05Qual)*` 在 ANTLR4 生成的代码大小/性能差异？
  - 待 Phase 1 用 `wc -l build/antlr4_generated_src/ptxParser.cpp` 测量
- **Q2**: ANTLR grun 工具是否可用？
  - 待 Phase 1 跑 `which grun` 验证；若不可用，用 `java -cp $CLASSPATH org.antlr.v4.gui.TestRig`
- **Q3**: 方案 A 是否完全解决所有失败 case？
  - 待 Phase 2 验证全部 D3 矩阵通过；若仍有失败，回退到选项 B
- **Q4**: 现有 4 个 workaround fixture 的 `.sync` 是否能恢复？
  - 待 Phase 3 验证 — 若 D3 矩阵通过，workaround fixture 可去除 reordering 注释
