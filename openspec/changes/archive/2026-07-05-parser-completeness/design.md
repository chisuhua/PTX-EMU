## Context

PTX-EMU parser 代码累积 12 条 TODO/FIXME 债务（基于 Metis pre-implementation review 实证分类），其中仅 1 条是实质 P0 功能缺失。本 design 反映 Metis 修订后的 **3 Phase 清理 + 1 P0 fix** scope。

### 12 条债务分类（Metis pre-impl review 实证）

| 类别 | 数量 | 代表条目 |
|------|------|---------|
| **第1类 已实施未清理** | 2 | `barrier.cpp:11` Stage 3 ✅、`ptx_interpreter.cpp:124` BUGFIX |
| **第2类 未实施有影响** | 3 | `ptx_parser.cpp:60` multi-PTX ⚠️P0、`ptx_visitor.cpp:363` extern、`thread_context.cpp:410` 设计 |
| **第3类 死代码** | 3 | `calculateTypeSize`、`processFunctionAttributes` ×2 |
| **第4类 过期注释** | 4 | `statement_context.h:24`、`ptx_visitor.cpp:435-436`、`thread_context.cpp:171-181`、`ptx_visitor.cpp:607` |

### Metis Review History

**Review date**: 2026-07-05 | **Decision**: ⚠️ CONDITIONAL → 解决 7 项 MUST-RESOLVE 后 GO

**关键修订**（与 fix-cvt-strategy-actual-split 同模式）：
- 原提案"10 条债务 + 6 Phase" → 修订为"12 条债务 + 3 Phase"
- 原提案 Fix #1（修改 statement_context.h:24 注释）→ 改为"仅清理过期注释"（`ptx_interpreter.cpp:124-145` 已 BUGFIX）
- 原提案 Fix #3-#4（实施 processFunctionAttributes）→ 改为"删除死代码"（0 调用者 + grammar 不支持）
- 原提案未覆盖 `ptx_visitor.cpp:363` extern → 拆出独立 change `add-extern-function-declaration`

### 当前状态（grep + git log 实证）

| 文件:行 | 内容 | 状态 |
|---------|------|------|
| `ptx_visitor.cpp:303` | `// TODO: Implement based on new grammar` + 函数体空 | 死代码（0 调用者）|
| `ptx_visitor.cpp:323` | `// TODO: Implement proper type size calculation` + `return 4;` | 死代码（`getBytes()` 已替代）|
| `ptx_visitor.cpp:363` | `// TODO: Add extern function declaration handling` | 实质 gap — 独立 change |
| `ptx_visitor.cpp:435-436` | `// P0 cleanup: silently using "TODO" as identifier...` | 已修复的注释 |
| `ptx_visitor.cpp:607` | `// TODO: Process function attributes` | 死代码注释 |
| `statement_context.h:24` | `std::optional<int> size; // FIXME total size in bytes` | 过期注释（optional 语义正确）|
| `ptx_parser.cpp:60` | `// FIXME cubin have multiple ptx` + `ptx_code = of_ptx.str();` | **真 P0**（静默截断）|
| `ptx_interpreter.cpp:124` | `// BUGFIX: 之前使用 localDecl.size (FIXME 字段，几乎未设置)` | 过期注释（BUGFIX 已生效）|
| `barrier.cpp:11` | `// Stage 3 TODO: 1. ✅ ... 2. ✅ ...` | 过期 TODO 列表（全部完成）|
| `thread_context.cpp:171,181` | `// TODO` + `size_t reg_size = 8; // TODO: 获取实际寄存器大小` | dumpContext 注释代码 |
| `thread_context.cpp:410` | `// FIXME should use stmt qualifier?` | 设计选择（无代码改动需要）|

## Goals / Non-Goals

### Goals

1. **删除 3 处死代码**：`calculateTypeSize`、`processFunctionAttributes`、`visitFunctionDecl:607` 注释（~15 行）
2. **更新 5 处过期注释**：statement_context.h:24、barrier.cpp:11、ptx_interpreter.cpp:124、ptx_visitor.cpp:435-436、thread_context.cpp:171-181
3. **P0 fix**：multi-PTX cubin warning + 累加语义（~10 行）
4. **同步 4 个文档**：AGENTS.md × 2、lessons-learned.md、ADR（如适用）

### Non-Goals（明确排除）

1. ❌ **实施 `processFunctionAttributes`**：需 ANTLR 语法扩充（触及 `ptx-grammar-modification` 高危流程），拆为独立 change
2. ❌ **实施 `extern` 函数声明**（`ptx_visitor.cpp:363`）：涉及完整函数调用栈（未实现），拆为独立 change `add-extern-function-declaration`
3. ❌ **修改 `statement_context.h:24` size 字段为必填**：当前 `std::optional` 语义正确，`ptx_interpreter.cpp:124-145` 已 BUGFIX 处理
4. ❌ **修改 `thread_context.cpp:410` 设计选择**：当前实现正确（`instr.qualifiers` 优先），仅清理 FIXME 注释
5. ❌ **实施 dumpContext 寄存器 dump**（`thread_context.cpp:171,181`）：debug 功能，不在清理 scope

## Decisions

### Decision 1: 死代码删除 vs 实施

**Choice**: 删除（Fix #1-#2）

**Rationale**：
- `calculateTypeSize()` 0 调用者，`getBytes(qualifier)` 在 `qualifier_utils.cpp:56` 已替代其功能
- `processFunctionAttributes()` 0 调用者 + `ptxParser.g4` 无 `functionAttribute` 规则
- 实施需 ANTLR 语法扩充 → 触及 `ptx-grammar-modification` skill（强制 TDD + `test_all_ptx.sh` 全量验证）

**Alternatives Considered**：
- (A) 删除死代码 — **采纳**（最低风险，零行为变更）
- (B) 实施功能 — 拒绝（grammar 不支持，工作量 >8h，触及多个 module）

### Decision 2: multi-PTX 累加语义 vs 警告

**Choice**: 警告 + 累加

**Rationale**：
- 当前 `ptx_code = of_ptx.str()` 覆盖语义导致多 section cubin 仅保留最后 section
- `+=` 累加语义更接近 `cubin_utils.cpp` 的 append-all 行为
- `PTX_WARN_EMU` 基础设施已存在（`src/ptxsim/barrier/barrier_module.cpp` 多次使用）
- 累加可能引入符号冲突（不同 PTX section 定义同名符号）→ 警告告知用户

**Alternatives Considered**：
- (A) 仅 PTX_WARN_EMU + 保留 `=` 覆盖 — 拒绝（用户仍会失去前 N-1 section）
- (B) PTX_WARN_EMU + `+=` 累加 — **采纳**（保留信息 + 告知风险）
- (C) 完整 multi-section 管理 — 拒绝（触及 cubin_utils 深层，工作量 >16h）

### Decision 3: Phase 拆分粒度

**Choice**: 3 Phase（清理 → fix → archive）

**Rationale**：
- Phase 1（清理）独立可 revert，0 行为变更
- Phase 2（multi-PTX fix）独立可 revert，仅 multi-section cubin 受影响
- 每 Phase 独立 commit（Fix #1, #2, #3）便于 bisect 定位回归

**Alternatives Considered**：
- (A) 1 Phase 单 commit — 拒绝（违反 Checklist D "每个 commit 独立 fix 编号"）
- (B) 6 Phase（细粒度）— 拒绝（与 fix-cvt-strategy-actual-split 6→3 Phase 教训一致）
- (C) 3 Phase — **采纳**

### Decision 4: 基线 worktree 复用 vs 新建

**Choice**: 新建 `.worktrees/parser-completeness-baseline`

**Rationale**：
- `.worktrees/` 目录为空（Metis 实证）
- `fix-pre-p0-baseline` 不存在（`ls .worktrees/` 验证）
- 全量 build 预算 ~15-20 分钟（lessons-learned §4）

**Alternatives Considered**：
- (A) 复用现有 worktree — 拒绝（不存在）
- (B) 新建 worktree — **采纳**

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **R1: multi-PTX 累加语义破坏现有 cubin** | 🟡 中（仅 multi-section cubin 受影响）| (1) PTX_WARN_EMU 告知用户 (2) 测试构造 multi-section cubin mock (3) 检查所有现有测试 cubin 是否为 single-section |
| **R2: 死代码删除遗漏真实调用者** | 🟢 低（grep 已验证 0 调用者）| (1) grep -rn "calculateTypeSize\|processFunctionAttributes" src/ include/ tests/ (2) ctest 全 PASS |
| **R3: 注释更新引入歧义** | 🟢 低（仅注释）| (1) 引用 lessons-learned §6 / §20 风格 (2) Oracle 二次 review |
| **R4: Oracle 测试不存在**（tests/unit/parser/ 不存在） | 🟡 中（multi-PTX fix 无 oracle）| (1) 实施 Phase 2 时先建立 tests/unit/parser/test_multi_ptx.cpp (2) 构造 mock cubin |
| **R5: Phase 2 改变 PTX 加载行为** | 🟡 中（可能影响 cute_rmsnorm 等 e2e）| (1) 现有 e2e 验证 multi-section cubin 不存在 (2) baseline 对比（lessons-learned §4）|
| **R6: artifact 提交遗漏**（lessons-learned §6） | 🟢 低（已 git-tracked Phase 0.1）| (1) Phase 0.1 完成后立即 git add + commit (2) git ls-files 验证 |

## Migration Plan

### Phase 0: Artifacts Git-Tracking（强制，per lessons-learned §6）

```bash
# Step 0.1: 创建工作分支
git checkout -b refactor/parser-completeness

# Step 0.2: git add artifacts FIRST（关键！）
git add openspec/changes/parser-completeness/
git status  # 应显示 4 个新文件
git ls-files openspec/changes/parser-completeness/  # 不应为空

# Step 0.3: commit artifacts
git commit -m "docs(openspec): add parser-completeness artifacts (3-phase scope)

Metis pre-impl review applied (lessons-learned §20 + Checklist H).
Scope revised from initial 10-debt + 6-phase to 12-debt + 3-phase:
- Phase 1: dead code + stale comments (~15 lines)
- Phase 2: P0 multi-PTX warning + 累加语义 (~10 lines)
- Phase 3: archive change (per Checklist G)

Refs: archive/2026-07-05-fix-cvt-strategy-actual-split/ (stale artifact pattern)
"
```

### Phase 0.4: Baseline Worktree（per lessons-learned §4）

```bash
# 全量 build 预算 ~15-20 分钟
git worktree add .worktrees/parser-completeness-baseline HEAD
cd .worktrees/parser-completeness-baseline
. env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)  # 必须全量！
cd build && ctest --output-on-failure 2>&1 | tee /tmp/parser-baseline.log
```

### Phase 1: 死代码 + 过期注释清理（Fix #1）

```bash
# 创建实施 worktree
cd /workspace/project/PTX-EMU
git worktree add .worktrees/parser-completeness-impl refactor/parser-completeness

# 实施 Fix #1.1: 删除 calculateTypeSize
# Edit src/ptx_parser/ptx_visitor.cpp
# - 删除 lines 323-326 (函数体)
# - 删除 declaration in ptx_visitor.h

# 实施 Fix #1.2: 删除 processFunctionAttributes
# Edit src/ptx_parser/ptx_visitor.cpp
# - 删除 lines 303-310 (空函数体)
# - 删除 declaration in ptx_visitor.h
# - 删除 line 607 注释

# 实施 Fix #1.3: 更新 5 处过期注释
# - statement_context.h:24 → "size: optional total bytes (may be unset for some decls)"
# - barrier.cpp:11 → 删除 Stage 3 TODO 列表
# - ptx_interpreter.cpp:124 → 精简 BUGFIX 注释
# - ptx_visitor.cpp:435-436 → 精简为说明性注释
# - thread_context.cpp:171,181 → "dumpContext: debug stub, register dump not implemented"

# 验证
cd build && ctest --output-on-failure  # 必须 100% PASS

# Commit Fix #1
git commit -am "refactor(parser): delete dead code + update 5 stale comments (Fix #1)

Dead code removed:
- calculateTypeSize (0 callers, getBytes() replacement exists)
- processFunctionAttributes (0 callers, grammar doesn't support)

Stale comments cleaned:
- statement_context.h:24 size FIXME (optional semantics correct)
- barrier.cpp:11 Stage 3 TODO (all sub-items completed)
- ptx_interpreter.cpp:124 BUGFIX (no longer needed)
- ptx_visitor.cpp:435-436 P0 cleanup
- thread_context.cpp:171,181 dumpContext TODO

Per lessons-learned §20 4-category classification (Metis review 2026-07-05).
0 behavior change — pure cleanup.

Co-Authored-By: Metis <metis@openspec>
"
```

### Phase 2: P0 multi-PTX warning（Fix #2）

```bash
# 实施 Fix #2: multi-PTX warning + 累加语义
# Edit src/ptx_parser/ptx_parser.cpp:56-62
# - 添加 section 计数器
# - 当 count > 1 时调用 PTX_WARN_EMU
# - 恢复 += 累加

# 建立 oracle 测试
# Edit tests/unit/parser/test_multi_ptx.cpp (new file)
# - 构造 multi-section cubin mock
# - 验证 warning 触发 + 累加语义

# 验证
cd build && ctest --output-on-failure  # 必须 100% PASS
./scripts/sanity.sh --quick  # 快速 sanity check

# Commit Fix #2
git commit -am "fix(parser): multi-PTX cubin warning + 累加语义 (Fix #2)

src/ptx_parser/ptx_parser.cpp:60 FIXED:
- Restore += 累加 (was: = 覆盖丢失前 N-1 sections)
- Add PTX_WARN_EMU when section count > 1
- Update AGENTS.md KNOWN LIMITATIONS

Per lessons-learned §1 cross-module state translation:
- cubin_utils.cpp already appends all sections
- parser layer was inconsistent (覆盖 last only)
- This fix aligns parser with cubin_utils behavior

Test: tests/unit/parser/test_multi_ptx.cpp (new, 3 scenarios)

Co-Authored-By: Metis <metis@openspec>
"
```

### Phase 3: 同步文档 + Archive（per Checklist G）

```bash
# 同步 4 个文档
# Edit src/ptxsim/instructions/AGENTS.md KNOWN LIMITATIONS
# Edit src/ptxsim/core/AGENTS.md (如 dumpContext 相关)
# Edit docs/dev-process/lessons-learned.md (新增 §P0 multi-PTX postmortem)
# Edit openspec/changes/archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/ (cross-ref 如有)

# 验证
./scripts/sanity.sh  # 完整 sanity check

# 二次 ctest 验证
cd build && ctest --output-on-failure

# Commit 文档同步
git commit -am "docs(parser): sync AGENTS.md + lessons-learned post-Phase 2

Per Checklist I (lessons-learned §21):
- AGENTS.md KNOWN LIMITATIONS: multi-PTX 描述更新
- lessons-learned.md §P0-multi-ptx: 新增 postmortem
- docs/audits/debt-audit-2026-07-02.md: 标记 RESOLVED（如有 P0 条目）

Fix #3 — 文档同步"
```

### Rollback Strategy

```bash
# 任何 Phase 失败立即 revert 该 Phase（不混入后续 commit）
git revert HEAD  # revert 最近的 Phase
cmake --build build  # 重新编译
ctest --output-on-failure  # 验证回退后通过

# 或直接回退整个分支
git checkout main
git branch -D refactor/parser-completeness
git worktree remove .worktrees/parser-completeness-impl
git worktree remove .worktrees/parser-completeness-baseline  # 仅在 merge 后清理
```

## Open Questions

### OQ-1: multi-PTX 累加的符号冲突风险

**Question**: 不同 PTX section 定义同名符号时，累加是否会引起符号表冲突？

**Status**: 待 Phase 2 实施时验证。如触发冲突，需切换到 Decision 2(A) 仅警告不累加。

### OQ-2: `thread_context.cpp:410` 是否需要添加 `// Decision:` 注释

**Question**: 当前 FIXME 注释可能误导读者认为是 bug。是否改为 "Design decision: instr.qualifiers is canonical source"？

**Status**: 待 Phase 1 Fix #1.3 决定。如选择清理，建议采用 lessons-learned §6 风格的"Decision:"注释。

### OQ-3: `barrier.cpp:11` Stage 3 TODO 删除后是否需要保留 git 历史注释

**Question**: 删除过期 TODO 后，未来读 git log 的人能否追溯 Stage 3 的实施历史？

**Status**: commit message + git blame 足以追溯。无需额外保留注释。