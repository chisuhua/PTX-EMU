# Fix tcgen05 ANTLR LL(*) Prediction Bug in Qualifier Kleene Star

> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) Accepted
> **前置 change**: `archive/2026-07-07-fix-tcgen05-grammar-mr3` (Change-3a, archived 2026-07-07)
> **后续 change 软依赖**: `implement-tcgen05-handlers-core` (Change-3b, pending) — 不强制，但 grammar 修正提升其测试覆盖率
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §6(artifacts-first)+ §7(Pre-impl review)

## Why

Change-3a (`fix-tcgen05-grammar-mr3`) 修复了 `test_all_ptx.sh` 36/36 PASS 的基线，但**通过 fixture reordering 绕过**了一个**未解决的 ANTLR LL(*) 预测 bug**：

- **症状**: `tcgen05.<subop>.sync.aligned.*` 在 ANTLR 解析时报 `no viable alternative`，尽管每个独立 qualifier 单独存在时都能解析
- **根因**: `tcgen05Inst` 规则的 `(DOT? tcgen05Qual)*` Kleene star 在某些 qualifier 序列上的 ANTLR4 LL(*) 预测失效
- **当前 workaround**: 在 4 个 fixture 中 reordering qualifiers（把 `.sync` 放到 `.cta_group::1` 之前；把 `.block_scale` 放到 `.kind::f16` 之前）— 但这**不是 PTX ISA 标准的任意顺序**
- **影响范围**:
  - PTX 解析器接受**qualifier 子集**的顺序，无法保证任意顺序
  - 5 个 fixture 需 reordering（`tcgen05_ld/st/cp/cp_multicast/...` 中如果加 `.sync` 会失败）
  - `change-3b` 实施者若添加新的 `.ptx` 测试可能踩到同一个 bug
  - `cute_rmsnorm` 等真实 kernel 的 PTX 输出若包含 `tcgen05.ld.sync` 序列会失败

**本 change 实施真正的 ANTLR 预测 bug 修复**，恢复 PTX qualifier 的任意顺序支持。

### 失败示例(empirical)

| Input | 结果 |
|-------|------|
| `tcgen05.ld.aligned.32x32b.shared::cta.b32` | PASS |
| `tcgen05.ld.sync.aligned.32x32b.shared::cta.b32` | **FAIL** (LL(*) 预测失败) |
| `tcgen05.st.aligned.*` | PASS |
| `tcgen05.st.sync.aligned.*` | **FAIL** |
| `tcgen05.cp.aligned.*` | PASS |
| `tcgen05.cp.sync.aligned.*` | **FAIL** |
| `tcgen05.mma.kind::f16.cta_group::1.*` | PASS |
| `tcgen05.mma.cta_group::1.kind::f16.*` | **FAIL** |
| `tcgen05.mma.block_scale.kind::f16.cta_group::1.*` | PASS |
| `tcgen05.mma.kind::f16.cta_group::1.block_scale.*` | **FAIL** |

**规律**: 失败的 case 都是 `TCGEN05_SYNC`(`.sync`) 或 `KIND COLONCOLON tcgen05Dtype` 出现在 `IMMEDIATE`(数字) 之后。ANTLR 的 LL(*) 预测在 `IMMEDIATE` 之后的 Kleene star 决策点不能正确识别 `KIND`/`TCGEN05_SYNC` 作为合法 qualifier 开始。

## What Changes

### 修改

| 文件 | 范围 |
|------|------|
| `src/grammar/ptxInstructions.g4` | 重构 `tcgen05Inst` 规则，修复 Kleene star 预测（3 个候选方案，见 design.md D1）|
| `src/grammar/ptxLexer.g4` | 可能补充 lexer mode（若 D1 方案 B 采纳）|

### 复测/回归

| 文件 | 范围 |
|------|------|
| `tests/ptx/tcgen05_*.ptx` | 12 个 fixture 验证 qualifier 任意顺序通过 |
| 新增 `tests/ptx/tcgen05_permutations.ptx` | 单一 instruction 多种 qualifier 顺序，验证任意顺序通过 |
| `tests/ptx/test_all_ptx.sh` | 46/46 → 47/47 PASS |

### 不修改(范围外)

- ❌ `src/ptxsim/instructions/wmma.cpp` 中 5 个 `execute_tcgen05_*` 函数（change-3b scope）
- ❌ `src/ptx_parser/ptx_visitor_wmma.cpp` 中 `visitTcgen05Inst` 的 operand 提取完善（change-3b scope）
- ❌ 任何 handler 实现（change-3b scope）
- ❌ `S_TCGEN05_MMA_WS` 仍是 dead code（`.ws` 是 qualifier，非 sub-op；与本 bug 无关）
- ❌ 不引入新的 PTX 指令族

## Non-Goals

### 显式拒绝

- ❌ 不重写整个 ANTLR grammar（仅修复 Kleene star 预测）
- ❌ 不修改非 tcgen05 指令的 grammar（`wmma.*`、其他指令族的 qualifier 顺序问题不在范围）
- ❌ 不引入 lexer mode 切换机制到整个 grammar（仅在 tcgen05 scope 内）
- ❌ 不修改 `Tcgen05OpKind` 枚举或 `S_TCGEN05_*` StatementType 映射
- ❌ 不修改 `wmma.cpp` 中 `is_tcgen05_*` helper 函数（与本 bug 无关）

### 范围限制

- 仅修复 `tcgen05Inst` 规则的 Kleene star 预测问题
- 不变更 12 个现有 fixture 的内容
- 不修改测试或 handler 代码

## Goals

### Phase 1: 根因精确化(1 commit)

1. 跑 ANTLR4 `grun ptxparser ptxFile -tokens`（或 TestRig 等价工具）抓取 `tcgen05.ld.sync.aligned` 的 token 流
2. 用 ANTLR4 生成的 `atn` (Augmented Transition Network) 分析 Kleene star 的预测路径
3. 确认根因为 ANTLR4 LL(*) 预测在 `IMMEDIATE → qualifier 序列` 的特定转移上不能合并多个等价路径
4. 写 1-2 个最小复现 .ptx 单元（仅 1 条 instruction），用于 TDD

### Phase 2: 修复(1-2 commits,per design.md D1 方案)

**3 个候选方案**（见 `design.md` D1）：

- **A**: 重写 `tcgen05Inst` 规则为 recursive `qualifierList` 形式（消除 Kleene star 歧义）
- **B**: 引入 lexer mode `TCGEN05_QUALIFIER_MODE`，显式控制 qualifier 词法分析
- **C**: 拆分 `tcgen05Qual` 为多个子规则（`tcgen05SimpleQual` + `tcgen05CtypeQual`）并显式列出预测路径

每个方案 commit 后跑 `./tests/ptx/test_all_ptx.sh` + `ctest -L "unit|integration"` 验证零回归。

### Phase 3: 复测与回归(1 commit)

1. 12 个现有 tcgen05 fixture 仍 PASS
2. 新增 `tests/ptx/tcgen05_permutations.ptx`（覆盖 qualifier 任意顺序的失败 case）
3. 跑 `./scripts/sanity.sh` 全量验证
4. 更新 `docs/dev-process/lessons-learned.md` 追加 ANTLR 预测 bug 模式（per §23 模板）

### Phase 4: 文档同步(1 commit,per lessons-learned §6)

1. 移除 fixture 文件中的 workaround 注释（`// Fixed: preprocessor-compatible format` 等）
2. 更新 `design.md` D1 标记为 RESOLVED
3. 更新 handoff 引用（指明后续 change 不需再处理此 bug）

## Capabilities

### New Capabilities

- `tcgen05-antlr-prediction-fix`: Kleene star 预测修复（spec 范围不变，补到 `tcgen05-grammar` spec）

### Modified Capabilities

- `tcgen05-grammar`: spec 修订（qualifier 任意顺序通过）
- `tcgen05-fixtures`: spec 修订（新增 permutations fixture）

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `src/grammar/ptxInstructions.g4` | 重构 `tcgen05Inst` 规则 | ±50 |
| `src/grammar/ptxLexer.g4` | 可能新增 lexer mode | ±30 |
| `tests/ptx/tcgen05_permutations.ptx` | 新增 | +30 |
| **总计** | | **+110 / ±80** |

### 影响的依赖

- `ptx-grammar-modification` skill(强制 TDD 流程,先跑 baseline)
- `grun` 或 `java -cp antlr4.jar org.antlr.v4.gui.TestRig` 用于 token 流分析(可选)
- ANTLR4 生成的 `atn` API（`build/antlr4_generated_src/ptxParserATN.cpp`）用于深度诊断(可选)

### 不影响的依赖

- `src/ptxsim/*`、handler 实现（change-3b scope，独立）
- `src/ptx_parser/ptx_visitor_wmma.cpp`（与 grammar 修复正交）
- `ptx_qualifier.def` 中 `Q_TCGEN05_*` stubs（保留，change-3b 删除）

### 影响的文档

- `docs/dev-process/lessons-learned.md`(追加 §23 ANTLR Kleene Star 预测陷阱)
- `openspec/changes/archive/2026-07-07-fix-tcgen05-grammar-mr3/design.md`(移除 workaround 标记)
- ADR-0016 不变更(架构决策不变)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性(本 change 主要改 grammar)

- [x] Baseline 函数清单:`tcgen05Inst` 规则 Kleene star 决策点(per change-3a design.md D1)
- [x] 现有测试数量已修正:46/46 PASS(workaround 后)
- [x] 跨模块状态翻译:无(本 change 不动 handler/visitor)
- [x] invariant 清单:ANTLR grammar 必须 deterministic 且支持任意 qualifier 顺序

### 多 Phase 推进(3-4 atomic commits)

- [x] Phase 1: 根因精确化(独立 commit,先跑 baseline 验证)
- [x] Phase 2: 修复(独立 commit,基于 Phase 1 选定的方案)
- [x] Phase 3: 复测与回归(独立 commit)
- [x] Phase 4: 文档同步(独立 commit,per Checklist E)
- [x] 基线 worktree 计划:`.worktrees/baseline-antlr-fix`(per `ptx-lessons-learned` §4)
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 Phase

### 文档同步

- [x] ADR 追加段落已规划(无变更)
- [x] tasks.md 任务规划(占位 TODO,实施时细化)
- [x] archive 路径已列出:`archive/2026-07-XX-fix-tcgen05-antlr-prediction-bug/`

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证 Phase 1 根因分析、Phase 2 方案选择
- [ ] 跑 `./tests/ptx/test_all_ptx.sh` 记录 baseline(46/46 PASS,但有 4 个 fixture 含 workaround)
- [ ] 用 ANTLR4 grun/TestRig 抓取 `tcgen05.ld.sync.aligned` 的 token 流
- [ ] 用 ANTLR4 ATN 工具生成 Kleene star 决策图,定位失败路径

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| `fix-tcgen05-grammar-mr3` (Change-3a, archived) | **fix-tcgen05-antlr-prediction-bug** | `implement-tcgen05-handlers-core` (Change-3b, pending) |
| `extend-blackwell-tcgen05-infra` (Change-2, pending) | | |

- **Change-3a → 本 change**: 继承修复的 sub-op dot-prefix + 12 个 fixture
- **本 change → Change-3b**: 解决 Change-3b 实施者可能踩到的 PTX 解析坑
- **本 change → Change-3b**: 软依赖,不强制(若 Change-3b 已有自己的 fixture,可独立进行)
