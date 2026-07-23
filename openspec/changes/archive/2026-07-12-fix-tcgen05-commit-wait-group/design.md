## Context

`tcgen05.commit` 和 `tcgen05.wait` 是 Blackwell Tensor Core 的屏障同步机制。`commit(N)` 推进 group_id=N 的 commit-counter，`wait(N)` 阻塞直到 counter ≥ N。FlashAttention FA3 producer-consumer pipeline 需要多个 group（QK^T、softmax、PV）区分同步边界。

### 现状问题（已逐行验证，per Oracle 2026-07-11 审计 + 本 design 的独立实证验证）

**问题 1：`Tcgen05Instr::cta_group` 便利字段从未填充**
- 位置：`include/ptx_ir/statement_context.h:186` 默认 `cta_group = 1`
- 验证：
  - `visitTcgen05Inst` (`src/ptx_parser/ptx_visitor.cpp:841-885`) 不填该字段（只 populates `op_kind`/`qualifiers`/`operands`/`instructionText`）
  - `makeTcgen05Instr` (`include/ptx_ir/statement_factory.h:265-292`) 同上
- **风险**：所有 Tcgen05Instr 的 `cta_group` 永远 = 1，即使 PTX 写 `.cta_group::2`

**问题 2：`extractQualifiersFromContext` 静默丢弃 IMMEDIATE 值**
- 位置：`src/ptx_parser/ptx_visitor.cpp:155-183`
- 验证：
  - Grammar 规则 `tcgen05Qual` (`src/grammar/ptxInstructions.g4:451`):
    ```
    TCGEN_CTA_GROUP COLONCOLON IMMEDIATE
    ```
  - 解析树进入 `extractQualifiersFromContext` 时：
    - `TCGEN_CTA_GROUP` → `tokenToQualifier` 返回 `Q_TCGEN_CTA_GROUP` → 推入 vector ✓
    - `COLONCOLON` → `tokenToQualifier` 返回 `Q_UNKNOWN` → 丢弃
    - `IMMEDIATE` → `tokenToQualifier` 返回 `Q_UNKNOWN` → **静默丢弃** ✗
- **风险**：21 个 call sites（1 definition + 20 callers）调用此函数（已 grep 实证验证 `ptx_visitor*.cpp`），改返回类型会破坏所有 20 个 caller

**问题 3：Handler 硬编码 `group_id=1`**
- `src/ptxsim/instructions/tcgen05.cpp:512`:
  ```cpp
  cta->tc_queue().commit(1);  // 硬编码 group_id=1
  ```
- `src/ptxsim/instructions/tcgen05.cpp:550`:
  ```cpp
  cta->tc_queue().wait(warp, 0, 1);  // 硬编码 lane_id=0, group_id=1
  ```
- `(void)instr;` 显式忽略输入（`tcgen05.cpp:493,530`）

### 目标状态

| 项 | Before | After |
|---|---|---|
| `instr.cta_group` | 永远 = 1 | 从 `.cta_group::N` 提取 |
| Handler `processTcgen05Commit` | `commit(1)` | `commit(instr.cta_group)` |
| Handler `processTcgen05Wait` | `wait(warp, 0, 1)` | `wait(warp, 0, instr.cta_group)` |
| `cta_group::1` 行为 | OK | OK（不变） |
| `cta_group::2` 行为 | throw（ADR-0018）| throw（不变） |
| 多 group 同步 | 不支持 | 支持（pending FU-3 lane_id + FU-5 E2E） |

### 约束

- **不修改 grammar**（per Oracle Q5 + lessons-learned §9 ANTLR bare token 风险）
- **不修改 TcQueue**（已支持多 group，不动）
- **不实现 `tcgen05.wait N` 的 lane_id 操作数**（属于 FU-3.5 未来子任务）
- **保持 `cta_group::2` throw 行为**（per ADR-0018）

## Goals / Non-Goals

**Goals:**
- 解析 PTX `.cta_group::N` 限定符的 IMMEDIATE 值填充 `Tcgen05Instr::cta_group`
- `processTcgen05Commit` + `processTcgen05Wait` handler 读 `instr.cta_group` 替代硬编码 `1`
- 多 group_id 同步路径可用（通过现有 `TcQueue`）
- 所有现有测试零修改通过（`cta_group` 默认 1，向后兼容）
- 新增 2 测试：`commit/wait group=2` 集成测试 + `cta_group::2` 解析测试

**Non-Goals:**
- ❌ 不修改 grammar/lexer（避免 lessons-learned §9 ANTLR 风险）
- ❌ 不修改 `TcQueue` 内部实现（已支持多 group）
- ❌ 不实现 `tcgen05.wait N` 的 lane_id 操作数（FU-3.5 子任务）
- ❌ 不修复 H1/H2 (mma helper accumulator + f32 storage) — sister change 范围
- ❌ 不修复 ld/st slot 路由（FU-3 范围）
- ❌ 不修复 mma multi-warp slot 冲突（FU-4 范围）
- ❌ 不更新 E2E tests（Priority 3 fallback 与本次范围无关）

## Decisions

### D1: IMMEDIATE 提取策略 — Option (b) 在 visitTcgen05Inst 加单独 parse tree walk

**采纳**: Option (b) — 在 `visitTcgen05Inst` 中加独立 parse tree walk

**ANTLR 生成代码验证**（per Checklist H 实证 + Checklist L 强制）：

| API 名称 | 真实存在 | 来源 |
|---------|---------|------|
| `Tcgen05InstContext::tcgen05Qual()` 返回 `std::vector<Tcgen05QualContext*>` | ✅ | `build/antlr4_generated_src/ptxParser.h:3967` |
| `Tcgen05InstContext::tcgen05Qual(size_t i)` 返回单个 `Tcgen05QualContext*` | ✅ | `build/antlr4_generated_src/ptxParser.h:3968` |
| `Tcgen05InstContext::tcgen05QualList()` | ❌ **不存在** | （ANTLR 规则 `(DOT? tcgen05Qual)*` 不生成单独的 list context）|
| `Tcgen05QualContext::TCGEN_CTA_GROUP()` 返回 `antlr4::tree::TerminalNode*` | ✅ | `build/antlr4_generated_src/ptxParser.h:4009` |

**采纳代码**:

```cpp
// src/ptx_parser/ptx_visitor.cpp:858 后插入（line 858 是 makeTcgen05Instr 调用）
std::vector<Qualifier> qualifiers = extractQualifiersFromContext(ctx);

// NEW: C3 fix — extract cta_group IMMEDIATE value
// Grammar: TCGEN_CTA_GROUP COLONCOLON IMMEDIATE (ptxInstructions.g4:451)
// extractQualifiersFromContext drops the IMMEDIATE child silently.
// IMPORTANT: use tcgen05Qual() (NOT tcgen05QualList()) — grammar
// (DOT? tcgen05Qual)* generates direct vector accessor on
// Tcgen05InstContext, NOT a separate list context.
uint32_t cta_group = 1;  // default per statement_context.h:186
for (auto* qualCtx : ctx->tcgen05Qual()) {        // ✅ verified API
    if (qualCtx->TCGEN_CTA_GROUP() && qualCtx->IMMEDIATE()) {
        cta_group = static_cast<uint32_t>(
            std::stoul(qualCtx->IMMEDIATE()->getText()));
    }
}

// 传给 makeTcgen05Instr (line 883 — D2 adds 5th param)
makeTcgen05Instr(op_kind, qualifiers, operands, ctx->getText(), cta_group);
```

**拒绝的备选**:

| 选项 | 拒绝理由 |
|------|---------|
| (a) 改 `extractQualifiersFromContext` 返回 `std::vector<std::pair<Qualifier, std::optional<int>>>` | 破坏 20 个 caller（21 个 call sites - 1 definition，已 grep 实证）— blast radius 过大 |
| (b') `ctx->tcgen05QualList()->tcgen05Qual()`（**错误 API**，artifact 初稿误用）| ANTLR 生成的 accessor 是 `tcgen05Qual()` 直接在 `Tcgen05InstContext` 上，**无 `tcgen05QualList()` 方法**（per ptxParser.h:3958-3975）。如果按初稿实施，编译失败 |
| (c) 新 grammar rule 捕获 IMMEDIATE 到 IR 直接 | 触发 ANTLR LL(*) prediction conflicts（per lessons-learned §9 ANTLR bare token 风险）|

**Tradeoff**: 单次 parse tree walk 有 O(n_qualifiers) 开销（每个 Tcgen05Instr 多 ~1μs），可接受

### D2: `makeTcgen05Instr` 加可选参数 `uint32_t cta_group = 1`

**采纳**: 在 `include/ptx_ir/statement_factory.h:265` 的 `makeTcgen05Instr` 加可选参数

```cpp
inline StatementContext makeTcgen05Instr(
    Tcgen05OpKind op_kind,
    const std::vector<Qualifier>& qualifiers,
    const std::vector<OperandContext>& operands,
    const std::string& text = "",
    uint32_t cta_group = 1);  // NEW: 默认 1 = 当前行为
```

**理由**:
- 默认值 1 保留所有现有调用点（`tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` 等）
- function arg 是编译期强制 — 漏更新 `processTcgen05Commit`/`processTcgen05Wait` 会编译失败

**Tradeoff**: 1 个参数增长可忽略

### D3: Handler 改 `instr.cta_group` 但保留 lane_id=0 硬编码

**采纳**: 
```cpp
// tcgen05.cpp:512 (commit):
cta->tc_queue().commit(instr.cta_group);

// tcgen05.cpp:550 (wait):
cta->tc_queue().wait(warp, /*lane_id=*/0, instr.cta_group);
```

**理由**: `lane_id` 操作数解析属于 FU-3.5 未来子任务（per Oracle Q5 + OpenSpec lifecycle discipline），单独 propose 避免 scope creep

**Tradeoff**: multi-lane 等待仍硬编码 lane 0（pending FU-3.5）

### D4: 保留 `(void)instr;` → 删除，改用 `instr.cta_group`

**采纳**: `processTcgen05Commit` 函数签名已接收 `instr`（`tcgen05.cpp:493`），删 `(void)instr;` 不需要额外参数

**理由**: 已有 `instr` 参数，直接读 `instr.cta_group`

**Tradeoff**: 零

## Risks / Trade-offs

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| ANTLR 生成代码修改后 LL(*) 预测冲突 | — | **不修改 grammar**（per lessons-learned §9） |
| `extractQualifiersFromContext` 21 个 call sites（20 caller + 1 definition）回归 | High | Option (b) 不改该函数签名；新逻辑独立加在 `visitTcgen05Inst` |
| IMMEDIATE 值溢出（PTX 字面量超过 uint32） | Low | `std::stoul` 自然处理 + `static_cast<uint32_t>` 截断 |
| `cta_group=0` 边界值（PTX 不允许但解析仍生效） | Low | handler 不校验，行为交由 `TcQueue` 处理 |
| `tcgen05_commit_parse.cpp` 现有测试期望 `cta_group=1` 默认 | Medium | 新增 TC 验证 `cta_group::2` 解析（factory-level）；现有测试不动（默认 1 不变）。**ANTLR parser 路径**由 `./tests/ptx/test_all_ptx.sh` 覆盖（per lessons-learned §9 + Checklist L） |
| `processTcgen05Commit/Wait` 未真正读 `instr`（`(void)instr;`） | Low | D4 删除该 cast；编译期强制 `instr` 必须有 `cta_group` 字段 |
| ANTLR API 名写错（`tcgen05QualList()` 不存在）| **Critical**（artifact 初稿误用，已纠正）| D1 采纳代码块注释强制使用 `ctx->tcgen05Qual()`；已 `grep -n "tcgen05Qual" build/antlr4_generated_src/ptxParser.h` 实证 |
| `tests/integration/tcgen05/` 整个子目录不存在 | **Critical**（artifact 初稿假设）| 实证：子目录**存在**（含 7 个测试文件），但**无 CMakeLists.txt**；测试注册追加到 `tests/integration/CMakeLists.txt:432-...`（per ls 实证 + AGENTS.md "ctest 命名约束"）|

## Migration Plan

### 实施步骤

1. **Phase 1 (本 change 唯一 commit)**:
   - `include/ptx_ir/statement_factory.h`: `makeTcgen05Instr` 加 `cta_group = 1` 默认参数
   - `src/ptx_parser/ptx_visitor.cpp:858` 后追加 IMMEDIATE walk (D1)
   - `src/ptxsim/instructions/tcgen05.cpp:512,550`: 改读 `instr.cta_group`（保留 `lane_id=0`）
   - `src/ptxsim/instructions/tcgen05.cpp:493,530`: 删除 `(void)instr;`
   - `tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp`: 新增（commit group=2 + wait group=2 序列）
   - `tests/integration/ptx/test_tcgen05_mma_parse.cpp`: 追加 `cta_group::2` 解析验证 TC
   - `docs/adr/ADR-0016-blackwell-only-tcgen05.md`: 追加 "2026-07-12 Postmortem: C3 fix" 段

2. **Pre-Phase 0 (实施前必做)**:
   - Metis pre-implementation review (per lessons-learned §7/Checklist H)
   - Baseline worktree 建立（per lessons-learned §4，worktree path `.worktrees/baseline-c3`）
   - 4 个 OpenSpec artifacts git-tracked (per lessons-learned §6 — artifacts FIRST)

3. **Commit 顺序** (per lessons-learned §6 — artifacts-first 2-Phase):
   - Commit 1 (Phase 0 — artifacts): `git add openspec/changes/fix-tcgen05-commit-wait-group/` → "docs(openspec): fix-tcgen05-commit-wait-group artifacts"
   - Commit 2 (Phase 1 — 实施): 修改 3 文件 + 2 测试 + ADR → "fix(tcgen05): route commit/wait group_id from instr.cta_group (Oracle C3)"

### 回退策略

- 单 commit 包含所有变更（如有 Phase 拆分失败则 revert 整个 commit）
- `git revert HEAD` 后：
  - `makeTcgen05Instr` 恢复默认 1 参数（向后兼容）
  - visitTcgen05Inst 恢复不提取 cta_group
  - handler 恢复硬编码 `1`
  - 所有现有测试应过（基线 worktree 可对比验证）

## Open Questions

| 问题 | 影响 | 解决路径 |
|------|------|---------|
| `tcgen05.wait N` 的 lane_id 操作数何时处理？ | multi-lane 同步不完整 | 留作 FU-3.5 子任务或合并到 FU-3 |
| `cta_group::2` 测试是否会因 ADR-0018 throw 而无法验证？ | 测试 TC 必须 catch exception | 现有 4 个 extended parse tests 已验证 throw 模式（per `tests/integration/tcgen05/test_tcgen05_extended_parse.cpp`），复用模式 |
| IMMEDIATE 值小于 0 或大于 2？ | PTX 字面量边界 | handler 不校验；OpenSpec 接受 PTX 字面量语法限制 |

## Acceptance Criteria

### Phase 1 (实施) Acceptance

1. `makeTcgen05Instr` 接受可选 `uint32_t cta_group = 1`
2. `visitTcgen05Inst` 从 parse tree 提取 IMMEDIATE 填充 `instr.cta_group`
3. `processTcgen05Commit` 调 `commit(instr.cta_group)`
4. `processTcgen05Wait` 调 `wait(warp, 0, instr.cta_group)`
5. `tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp` 新增并 PASS
6. `tests/integration/ptx/test_tcgen05_mma_parse.cpp` 新增 `cta_group::2` TC 并 PASS
7. `cd build && ctest -R "tcgen05" --output-on-failure` 全部 PASS
8. baseline worktree 对比：所有 tcgen05-tagged 测试 PASS（除新测试外，行为不变）
9. ADR-0016 追加 postmortem 段

### Phase 2 (Archive) Acceptance

1. 4 个 md artifacts + 2 spec files git-tracked
2. ADR-0016 postmortem 段 git-tracked
3. `cd build && ctest --output-on-failure` 全量 PASS
4. `./tests/ptx/test_all_ptx.sh` 全量 PASS（47/47）
5. archive commit 含 postmortem 引用

## References

- Proposal: [proposal.md](proposal.md)
- Specs:
  - New: [specs/tcgen05-multi-group-commit-wait/spec.md](specs/tcgen05-multi-group-commit-wait/spec.md)
  - Modified delta: [specs/tcgen05-handlers-extended/spec.md](specs/tcgen05-handlers-extended/spec.md)
- ADR-0016: [docs/adr/ADR-0016-blackwell-only-tcgen05.md](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md)
- ADR-0018 (本 change 新建): [docs/adr/ADR-0018-tcgen05-cta-group-restriction.md](../../../docs/adr/ADR-0018-tcgen05-cta-group-restriction.md)
- ptx-lessons-learned: [.opencode/skills/ptx-lessons-learned/SKILL.md](../../../.opencode/skills/ptx-lessons-learned/SKILL.md) §3 + §4 + §6 + §7 + §9
- Sister change (archived 2026-07-11): [../../archive/2026-07-11-fix-tcgen05-mma-accumulator-and-f32-storage/](../../archive/2026-07-11-fix-tcgen05-mma-accumulator-and-f32-storage/)
