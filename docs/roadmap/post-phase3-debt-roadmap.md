# PTX-EMU Post-Phase3 Debt Roadmap (2026-07)

> **维护者**: PTX-EMU Architecture Team
> **最后更新**: 2026-07-05（HEAD `b5d3092`，parser-completeness + add-extern-function-declaration + Quick Wins 后）
> **目的**: 记录当前剩余技术债务 + 提供未来 OpenSpec change 创建指南
> **配套文档**:
> - [`docs/audits/debt-audit-2026-07-02.md`](../audits/debt-audit-2026-07-02.md) — 原始债务审计基线
> - [`docs/dev-process/lessons-learned.md`](../dev-process/lessons-learned.md) — 经验沉淀（16+ 章节）
> - [`.opencode/skills/ptx-lessons-learned/SKILL.md`](../../.opencode/skills/ptx-lessons-learned/SKILL.md) — 快速决策树
> - [`docs/dev-process/post-tcgen05-roadmap.md`](../dev-process/post-tcgen05-roadmap.md) — H5+ 战略方向

---

## 0. 执行摘要（30 秒读完）

### 当前状态（截至 `b5d3092`）

| 维度 | 数据 |
|------|------|
| **OpenSpec 已 Archived changes** | 18 个（最近：`2026-07-05-add-extern-function-declaration`, `2026-07-05-parser-completeness`, `2026-07-05-fix-cvt-strategy-actual-split`） |
| **Active changes** | 0 |
| **已 RESOLVED 债务**（本次会话） | A-5, A-7, A-8, C-11, C-12 |
| **剩余 A 系列债务** | 0（A-5 ✅ **RESOLVED 2026-07-06**, A-9 ✅ ARCHIVED, A-10 ✅ ARCHIVED） |
| **剩余 C 系列债务** | 18（god class + tests + includes；C-19 已移除：测试已存在） |
| **剩余 D 系列债务** | 6（docs README + OpenSpec 孤儿） |
| **Oracle tests 新增** | `unit_multi_ptx` + `unit_extern_function`（parser series） |
| **specs 已 promoted** | `parser-deadcode-cleanup`, `parser-multi-ptx-warning`, `extern-function-parse-coverage`, `stub-explicit-failure`, `wmma-tensor-core` 等 |

### 关键教训（来自 parser-completeness + add-extern-function-declaration 实战）

1. **审计假设必须 git verify**（lessons-learned §20）— C-13/C-14 审计错误（声称 stub 但实际已实现）
2. **Metis pre-impl review 必要**（Checklist H）— scope 修订为 2-Phase（add-extern-function-declaration 从"实施"缩到"覆盖既有 oracle"）
3. **artifacts-first 强制**（§6）— 任何 OpenSpec change 必须先 git add + commit artifacts 再实施代码
4. **CHECKLIST G lifecycle**— 归档后**禁止 amend**，必须新建 `fix-*`/`sync-*` change + `Ref:` 链接
5. **parser 系列 continuation 模式**（§22）— parser-completeness → add-extern-function-declaration（MR-7 续集）模式可复用

---

## 1. 剩余债务优先级矩阵

### 1.1 剩余 A 系列（架构）

| # | 债务 | 风险 | 优先级 | 推荐 change 名 | 工时 |
|---|------|------|--------|---------------|------|
| A-9 | ~~`atomic.cpp` 80% 完整但 CAS 未实现，无真正原子性~~ ✅ **ARCHIVED 2026-07-06** | ~~🟡 中~~ | ~~**🟡 P1**~~ | `implement-atomic-cas-and-true-atomicity` ✅ | 8h |
| A-10 | ~~嵌套分歧测试缺失（`test_nested_divergence.cpp:106`）~~ ✅ **ARCHIVED 2026-07-06** | ~~🟢 P2~~ | ~~`add-nested-divergence-tests`~~ | 5h |

### 1.2 剩余 C 系列（代码）

| # | 债务 | 优先级 | 推荐 change 名 | 工时 |
|---|------|--------|---------------|------|
| C-1 | `thread_context.cpp` **885 行**（parser-completeness 后 -19）22 个 include god class | 🟡 P1 | `god-class-refactor-thread-context` | 10h |
| C-2 | `sm_context.cpp` 703 行 god class | 🟡 P2 | `god-class-refactor-sm-context` | 6h |
| C-3 | `arithmetic.cpp` + `arithmetic_ext.cpp` 应合并 | 🟢 P3 | `merge-arithmetic-handlers` | 3h |
| C-4 | `src/ptx_ir/ptxir_writer.cpp::write_instruction()` **函数 246 行**（文件 374 行）degree 184 | 🟡 P2 | `refactor-ptxir-writer` | 3h |
| C-5 | 7 个子 AGENTS.md 与根 70%+ 重复 | 🟢 P3 | `consolidate-sub-agents-md` | 2h |
| C-6 | `tests/unit/contexts/` 7 个 <50 行 POD 测试太浅 | 🟢 P3 | `strengthen-pod-tests` | 2h |
| C-7 | `include/ptxsim/thread_context.h` 23 个 include | 🟢 P3 | `reduce-thread-context-includes` | 3h |
| C-8 | `include/ptxsim/testing/memory_test_utils.h` 18 个 include | 🟢 P3 | `reduce-memory-test-utils-includes` | 1h |
| C-9 | `src/CMakeLists.txt` 手动 `set(SOURCES)` 非 GLOB | 🟢 P3 | `cmake-use-glob-for-sources` | 1h + CI 检查 |
| C-10 | 仅 1 个 cmake option（无 ASAN/UBSAN） | 🟢 P3 | `add-cmake-options` | 1h |
| C-15 | `instruction_handlers.cpp` X-Macro 仅调用 1 次 | 🟢 P3 | `complete-x-macro-dispatch` | 3h |
| C-16 | ~~`atomic.cpp` 115 行 stub（CAS 缺失）~~ ✅ **合并到 A-9 已 archive** | ~~🟢 P3~~ | ~~(合并到 A-9)~~ | 8h |
| C-17 | `ptx_visitor.cpp` **998 行**（parser-completeness 后 -16）+ 12 TODO | 🟡 P2 | `split-ptx-visitor-god-class` | 5h |
| C-18 | `warp_context.cpp` **537 行**（清理后 -19）+ 6 次/30 commits | 🟡 P2 | `refactor-warp-context` | 4h |
| C-20 | `ptx_visitor_atom.cpp:28` 硬编码 ptx_op.def 格式（DRY） | 🟢 P3 | `dedupe-ptx-op-def-format` | 0.5h |
| C-21 | `assert(false && "...")` 3 处应改 throw | 🟢 P3 | `replace-assert-false-with-throw` | 1h |
| C-22 | 6 个 "docs(t2-4)" commit 占最近 50 commit 12% | 🟢 P3 | (流程性，非 change) | — |
| C-24 | `tests/e2e/divergence/test_divergence.cu` 仅 1 个非 barrier E2E | 🟢 P3 | `expand-e2e-divergence-coverage` | 8h |

### 1.3 剩余 D 系列（文档）

| # | 债务 | 优先级 | 推荐 change 名 | 工时 |
|---|------|--------|---------------|------|
| D-1 | `docs/README.md` 索引遗漏 9/17 子目录（注：parser-completeness 修复后，docs/README.md 索引 17 个但实际可能仍漏，需 re-verify） | 🟡 P2 | `docs-readme-fixes-remaining` | 1h |
| D-2 | `docs/README.md` 统计数据过时（已修复） | ✅ RESOLVED | — | — |
| D-3 | `docs/skills/README.md` 列 9 vs 实际 18 技能（需 re-verify） | 🟢 P3 | (合并到 D-1) | 0.5h |
| D-4 | 6 个 OpenSpec 孤儿 change 缺 design.md | 🟡 P2 | `cleanup-openspec-orphans` | 2h |
| D-5 | `docs/skills/` vs `.opencode/skills/` 内容分叉 | 🟢 P3 | (合并到 D-1) | 1h |
| D-6 | `HEALTH-AUDIT-2026-06-21.md` 8 个事实错误未合并 | 🟡 P2 | `merge-health-audit-errata` | 1h |

### 1.4 优先级排序总结与 Tier 映射规则

**优先级定义**（按**风险 + 架构影响**排序）：

```
🟡 P1（架构影响大，2 条）:
   A-9  atomic CAS                       → 8h
   C-1  thread_context god class         → 10h

🟡 P2（本月清理，8 条）:
   A-10, C-2, C-4, C-17, C-18, D-1, D-4, D-6

🟢 P3（季度清理，13 条）:
   C-3, C-5, C-6, C-7, C-8, C-9, C-10, C-15, C-20, C-21, C-24, D-3, D-5
```

**Tier vs Priority 映射**（工时时间箱）：

| Tier | 时间窗口 | 选 P1 | 选 P2 | 选 P3 |
|------|---------|-------|-------|-------|
| **Tier 1** | 本周（< 4h 累计）| ✅ 满足 < 4h 的项 | — | — |
| **Tier 2** | 本月（**月度预算** 15h 累计）| 单 Phase ≤ 15h，与其他 P1 项**累计 ≤ 15h** | ✅ 全部 | — |
| **Tier 3** | 季度（按需） | 跨 Phase 大重构（如 C-1 拆 3 Phase） | ✓ 大块 | ✅ 全部 |

**关键原则**：
- P 是**风险等级**（high/medium/low），Tier 是**时间窗口**（week/month/quarter），二者正交
- P1 项如工时超 Tier 1 预算 → 拆 Phase 进 Tier 2/3（保留 P1 风险标签）
-     P3 项如工时 ≤ 4h → 可前移至 Tier 1（quick win）
- 当前（2026-07-06）：A-9 已 archive 后 **Tier 1 仍空**（剩余 P1 项 C-1 = 10h too large for Tier 1; A-10 = 5h 接近上限）。Tier 2 月度预算 15h 重置，C-1 Phase 1（SIMT stack extraction ~3h）现在是 Tier 2 候选

---

## 2. 创建 OpenSpec Change 标准流程（Lessons-Learned 集成）

### 2.1 7 阶段标准流程

```
1. 债务识别              ← docs/audits/debt-audit-*.md 或 lessons-learned 触发
2. ⚠️ Metis pre-impl review  ← Checklist H 强制（lessons-learned §20）
3. scope 修订（如果 MUST-RESOLVE > 0）
4. 创建 4 个 artifacts     ← openspec-propose skill
   - proposal.md
   - design.md
   - specs/<name>/spec.md
   - tasks.md
5. git add + commit artifacts FIRST  ← §6 强制
6. 实施（每 Phase 独立 commit + 独立 fix 编号）
7. ctest PASS + archive   ← Checklist G lifecycle
```

### 2.2 Step 2 — Metis pre-impl review 强制模板

每次创建 OpenSpec change **前** 调用 Metis 子代理（subagent_type="metis"）：

```bash
# 在 openspec/propose 之前调用
task(subagent_type="metis", prompt="[PROMPT 见下文]")
```

#### Metis Prompt 模板

```markdown
# Pre-implementation Review for OpenSpec change: <change-name>

[CONTEXT] 简要描述 change 目的（来自 debt audit / lessons-learned）

[GOAL] 对此 change 进行 pre-implementation audit，输出 GO / ⚠️ CONDITIONAL / ❌ NO-GO

[REQUEST]

1. 验证提案的实证基础：
   - 用 grep / wc -l / git log 验证所有 API 假设真实存在
   - 验证 oracle 测试数量（ctest -N -L <label>）
   - 验证提到的 worktree / 路径 / 工具真存在
   - 区分"已实施但未清理"vs"未实施"

2. 评估每个债务的精确影响范围：
   - 是否真的是功能缺失 vs 过期注释 vs 设计选择
   - 是否有真实测试覆盖

3. 检查潜在 scope 风险：
   - interdependency（哪些 Fix 必须按特定顺序）
   - lessons-learned 风险（§5 qualifier.back(), §6 artifacts, §20 Metis）
   - 测试覆盖缺口

4. 检查 OpenSpec artifacts 完整性：
   - artifacts 目录结构（.openspec.yaml / proposal / design / specs / tasks）
   - artifacts 是否已 git-tracked

5. 检查工作目录与工具准备：
   - worktree 可用性
   - build baseline 是否已建立

6. 输出格式（严格按 lessons-learned §20 + Checklist H）：
   ### 6.1 Hidden Intentions（隐藏意图）
   ### 6.2 Ambiguities（歧义点）
   ### 6.3 AI Failure Points（AI 失败模式）
   ### 6.4 Missing Context（缺失上下文）
   ### 6.5 Decision + MUST-RESOLVE
      - GO | ⚠️ CONDITIONAL（列出 MUST-RESOLVE） | ❌ NO-GO
```

### 2.3 Step 4 — proposal.md 模板增强

按 `openspec-propose` skill Design-Time Checklist：

```markdown
## Why

<!-- 1-2 sentences，问题 + 为什么现在 -->

## What Changes

<!-- 具体改动，breaking 标记 **BREAKING** -->

## Capabilities

### New Capabilities
- `<name>`: <brief description>

### Modified Capabilities
- `<existing-name>`: <what requirement is changing>

## Impact

<!-- 受影响的代码/APIs/dependencies -->

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性（如适用）
- [ ] Baseline 函数所有 set_*/commit_*/force_*/lock_* 调用已列出
- [ ] 逐行 diff 计划已写入 design.md
- [ ] 跨模块状态翻译路径已文档化

### 多 Phase 推进（如适用）
- [ ] Phase 拆分方案 + 独立 commit 粒度已说明
- [ ] 基线 worktree 命令已记录
- [ ] 失败处理策略（revert 该 Phase）已说明

### 文档同步
- [ ] AGENTS.md 同步项已列出
- [ ] ADR 追加段落已规划
- [ ] tasks.md Phase 状态变更已说明
```

### 2.4 Step 5 — artifacts-first 强制（§6）

```bash
# 强制第一 Phase：artifacts 必须在实施 commits 之前 git add + commit
git checkout -b refactor/<change-name>
git add openspec/changes/<change-name>/
git ls-files openspec/changes/<change-name>/  # 不应为空
git commit -m "docs(openspec): add <change-name> artifacts (<scope>)

Metis pre-impl review applied.
Refs: lessons-learned §6, §20
"
```

### 2.5 Step 6 — 实施阶段纪律

```bash
# 每个 Fix 独立 commit（避免 git revert 牵连）
git commit -m "refactor: <action> (Fix #N)

<evidence-based description>
Refs: <commit hash or file path>
"
```

**Lessons-learned §20 实战教训**:
- 不要在 proposal 中声称"5+ oracle tests"如果 `ctest -N -L <label>` 显示只有 1 个
- 不要"复用现有 worktree"如果 `.worktrees/` 为空
- 不要假设"`X 行未拆分`"如果 git log 显示已部署到其他文件

### 2.6 Step 7 — archive + merge（Checklist G）

```bash
# ⚠️ Check G: 归档后禁止 amend，必须新建 fix-* change
openspec archive <change-name> --yes
git add openspec/changes/<change-name>/ openspec/changes/archive/<date>-<change-name>/ openspec/specs/<spec-name>/
git commit -m "chore(openspec): archive <change-name> (Checklist G)"
git checkout main
git merge --no-ff refactor/<change-name>
```

---

## 3. 推荐的下一步 Change（按 ROI 排序）

### 3.1 Tier 1 — 本周（< 4h 总工时）

> **当前 Tier 1 为空**：最近一次审计（2026-07-05, MR-1 修复后）移除 C-19（虚假债务）后，剩余 P1 项（A-9 = 8h, C-1 = 10h）的工时均超过 4h 单时间箱。
>
> **后续策略**：
> - 等待 quick-win 类型的 P1 项（如 C-8 减少 test util includes 1h、C-21 替换 assert(false) 1h）出现
> - 或将 P1 项 scope 拆分（如 A-9 拆为 Phase 1: CAS handler only，~3h）

### 3.2 Tier 2 — 本月（**15h 月度预算**，硬上限）

#### `implement-atomic-cas-and-true-atomicity`（A-9 + C-16，8h）

**Why**: 当前 `atomic.cpp` 用 load→compute→store 无锁序列，**多 warp 竞争条件下有数据竞争**。CAS 是 PTX 原子性的基础原语。

**Scope**（需 Metis pre-impl 验证）：
- Phase 1: CAS handler 实施（atomic.exch, atomic.compare_and_swap）
- Phase 2: 真正原子性 mutex（per warp serializes + cross-warp mutex）
- Phase 3: oracle test (multi-warp 并发正确性)

**风险**: 🔴 高（涉及并发正确性 + mutex 引入死锁风险 §2）
**Lessons-learned 集成**:
- ✅ §1 跨模块状态翻译（state 翻译表）
- ✅ §2 递归锁死锁（per "持锁方法不能再锁"）
- ✅ Checklist A: 函数迁移完整性

### 3.3 Tier 3 — 季度（按需）

#### `god-class-refactor-thread-context`（C-1，10h）🆕 **已从 §3 孤儿恢复**

**Why**: `thread_context.cpp` 当前 885 行，单文件 22 个 include，跨 SIMT stack / 寄存器 / 内存 / 控制流 4 个子系统（§1.2 P1）。是 P1 中工时最大项（10h）。**虽 10h ≤ 15h 单 Phase 上限**，但 Tier 2 本月预算 15h 已部分被 A-9 (8h) 占用（A-9 + C-1 = 18h > 15h 月度累计），且 C-1 拆 3 Phase 后 Phase 3 (~3h) 需跨季度执行 — 故归入 Tier 3 季度窗口。

**Scope**（待 Metis pre-impl 拆 Phase）：
- Phase 1（~3h，Tier 2 可承载）: 提取 SIMT stack 状态到独立类
- Phase 2（~4h）: 提取寄存器访问层
- Phase 3（~3h，跨季度）: 提取内存访问 + 控制流

**决策说明**：保留 P1 优先级（架构影响大），但实际执行分 Phase 跨季度，避免单 Phase 过大导致 lessons-learned §3 风险（Phases 必须独立可回退）。

#### 其他 Tier 3 项

参见 §1.2 完整列表，按工时 + 影响排序选择。优先序列：
- 🟡 P2 高 ROI: C-2 (sm_context 6h), C-4 (ptxir_writer 函数 246 行 3h), C-17 (ptx_visitor 5h), C-18 (warp_context 4h)
- 🟢 P3 流程性: 全部 C-5~C-10, C-15, C-20~C-22, C-24
- 🟡 P2 文档: D-1, D-4, D-6
- 🟢 P3 文档: D-3, D-5

---

## 4. 避免的反模式（来自 lessons-learned）

### 4.1 ❌ 修正 C-13/C-14 审计错误（避免重蹈）

```cpp
// ❌ WRONG: 假设"X 文件是 stub"而无验证
// debt audit: "cvt_int_to_float.cpp is a forwarding stub"
// → 实际：完整 IntToFloatStrategy 实现（fix-cvt-strategy-actual-split Sub-task 4b）
// → 修复行动：跳过 C-13，因审计错误

// ✅ CORRECT: 修复前用 git log + grep 验证
git log --oneline -- src/ptxsim/instructions/cvt/cvt_int_to_float.cpp
# 应显示: fc3c352/9837d44/d6123e0 已实施
grep -l "Strategy::convert" src/ptxsim/instructions/cvt/cvt_int_to_float.cpp
# 应显示: 文件本身
```

### 4.2 ❌ 修正 parser-completeness MR-1~A4（stale artifacts）

```cpp
// ❌ WRONG: 基于 untracked reconstructed artifacts 误判 debt 为 active
// → 实际: cleanup-deprecated-barrier-apis 已于 2026-06-20 归档
// → 影响: 浪费 12 天误判 + 重复工作

// ✅ CORRECT: 任何审计前 git verify
git log --all --oneline -- "openspec/changes/<change-name>/"
# 应包含 archive commit
git ls-files openspec/changes/<change-name>/
# 不应为空
```

### 4.3 ❌ 修正 parser-completeness `__VA_ARGS__` 宏嵌套坑

```cpp
// ❌ WRONG: Catch2 REQUIRE_NOTHROW 与 variadic macro 嵌套展开失败
REQUIRE_NOTHROW(PTX_WARN_EMU("fmt %d", 2));
// 编译错误: expected primary-expression before ')'

// ✅ CORRECT: lambda 包装（避免 __VA_ARGS__ 嵌套）
auto warn = []() { PTX_WARN_EMU("fmt %d", 2); };
REQUIRE_NOTHROW(warn());
// 必须在测试文件中保留注释解释原因
```

---

## 5. 相关 change 链接（参考实现）

### 5.1 parser-completeness（参考的"scope 修订"模式）

```
openspec/changes/archive/2026-07-05-parser-completeness/
├── proposal.md         (10 条 → 3 Phase)
├── design.md
├── tasks.md
├── specs/
│   ├── parser-deadcode-cleanup/   (7 Requirements)
│   └── parser-multi-ptx-warning/  (5 Requirements)
```

**关键**: Metis pre-impl review 修订 scope 6→3 Phase，避免 §6 stale artifact 陷阱。

### 5.2 add-extern-function-declaration（"覆盖既有 oracle" 模式）

```
openspec/changes/archive/2026-07-05-add-extern-function-declaration/
├── proposal.md         (2 Phase scope)
├── design.md
├── tasks.md
├── specs/
│   └── extern-function-parse-coverage/   (7 Requirements)
```

**关键**: Metis pre-impl 揭示"完全未支持"假设错误，修订为"覆盖既有 oracle"。

### 5.3 fix-cvt-strategy-actual-split（"stale artifact 修复"模式）

```
openspec/changes/archive/2026-07-05-fix-cvt-strategy-actual-split/
├── proposal.md         (Ref: archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/)
├── design.md
└── tasks.md
```

**关键**: 通过 `Ref:` 链接建立 lineage（不 amend 已归档 change）。

---

## 6. 工具快捷方式

### 6.1 债务 grep 快捷方式

```bash
# 列出所有 TODO/FIXME/XXX/stub 标记
grep -rn "TODO\|FIXME\|XXX\|stub" src/ include/ --include="*.cpp" --include="*.h" 2>/dev/null

# 列出所有 deprecated API
grep -rn "\[\[deprecated\]\]\|__attribute__.*deprecated" include/ src/ 2>/dev/null

# 列出所有 assert(false) 散落
grep -rn "assert\s*(\s*false" src/ include/ 2>/dev/null

# 列出所有空 catch(...) 块（应禁用）
grep -rn "catch\s*(\s*\.\.\.\s*)\s*\{\s*\}" src/ include/ 2>/dev/null

# 列出大函数（> 200 行）
# code-review-graph 工具: find_large_functions min_lines=200

# 列出未测试热点
# code-review-graph 工具: get_knowledge_gaps
```

### 6.2 OpenSpec lifecycle 命令

```bash
# 创建 change
openspec new change "<kebab-case-name>"

# 获取 artifact build order
openspec status --change "<name>" --json

# 获取 artifact instructions
openspec instructions <artifact-id> --change "<name>" --json

# Archive
openspec archive <name> --yes

# 列出 archive
ls openspec/changes/archive/

# 检查 specs 已 promoted
ls openspec/specs/
```

### 6.3 验证 checklist（每个 change 必跑）

```bash
# 1. ctest PASS（无 regression）
cd build && ctest --output-on-failure

# 2. PTX 语法测试
./tests/ptx/test_all_ptx.sh

# 3. Artifacts git-tracked（防 §6 反模式）
git ls-files openspec/changes/<name>/

# 4. 0 残留 dead code
grep -rn "<deleted_symbol>" src/ include/ tests/

# 5. Debt audit 更新（如适用）
# 手动更新 docs/audits/debt-audit-*.md 中相应条目
```

---

## 7. 维护说明

### 7.1 何时更新本 roadmap

- ✅ 任何**新 OpenSpec change 归档后** → 更新 §0 状态 + §1 债务优先级
- ✅ 任何**debt audit 新版本发布** → 更新 §1 债务清单
- ✅ 任何**新 lessons-learned §N 新增** → 更新 §4 反模式
- ✅ 任何**新归档 change 值得参考** → 添加到 §5

### 7.2 不要更新本 roadmap

- ❌ 任何 `docs/audits/*.md` 之外的债务基线变更
- ❌ 任何"未实施 OpenSpec change"的临时债务
- ❌ 任何单个 Quick Win（应在 commit message 中记录即可）

### 7.3 版本控制

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-05 | 初版：parser-completeness + add-extern-function-declaration + Quick Wins 后状态 |
| 1.1 | 2026-07-05 | MR-1~MR-5 修复：移除 C-19 虚假债务，澄清 C-4 函数/文件行数，恢复 C-1 至 §3 Tier 3，添加 Tier↔Priority 映射规则，更新 §1.2 + 附录过期行数（C-1, C-17, C-18, C-4） |
| 1.2 | 2026-07-05 | MR-N1~N3 (Oracle review) |
| 1.3 | 2026-07-06 | A-9 `implement-atomic-cas-and-true-atomicity` archive（commits `3a38ca0` + `5a328ac` + `6cb5baa` + archive commit）。3 Phase 全部完成（CAS handler + cross-warp mutex + multi-warp oracle test）。§0 更新 A-9 RESOLVED、§1.1 A-9 ✅、§1.2 C-16 ✅ RESOLVED（合并到 A-9）、附录 A.1 atomic.cpp RESOLVED |
| 1.4 | 2026-07-06 | A-10 `add-nested-divergence-tests` archive（commit `ef425d2` + archive commit）。32-lane 两级嵌套 setp+selp 覆盖率 added；TODO 2026-05-08 at line 106 removed。注：direct @%p bra 变体发现预存在的 SIMT stack 32-lane 跟踪 bug（lanes 16..31 inherit taken-branch state 无论 predicate）— 已 documented in file header, follow-up change required。§0 更新 A-10 RESOLVED、§1.1 A-10 ✅ |

---

## 附录 A：完整剩余债务清单（按文件分组）

### A.1 架构债务（A 系列）

| 文件 | 债务 | 行号 | 优先级 |
|------|------|------|--------|
| `src/ptxsim/instructions/atomic.cpp` | CAS + 真正原子性 | 全文 | 🟡 P1 |
| `tests/integration/divergence/test_nested_divergence.cpp` | 嵌套分歧测试 | line 106 | 🟢 P2 |

### A.2 代码债务（C 系列）

| 文件 | 债务 | 行号/范围 | 优先级 |
|------|------|----------|--------|
| `src/ptxsim/core/thread_context.cpp` | god class | 全文 **885 行**（parser-completeness 后 -19） | 🟡 P1 |
| `src/ptxsim/core/sm_context.cpp` | god class | 全文 703 行 | 🟡 P2 |
| `src/ptx_ir/ptxir_writer.cpp` | `write_instruction()` 长函数 | 函数 246 行（始于 line 129） | 🟡 P2 |
| `src/ptx_parser/ptx_visitor.cpp` | god class | 全文 **998 行**（parser-completeness 后 -16） | 🟡 P2 |
| `src/ptxsim/core/warp_context.cpp` | 多次修改的 god class | 全文 **537 行**（清理后 -19） | 🟡 P2 |
| `src/ptxsim/instructions/arithmetic.cpp` + `arithmetic_ext.cpp` | 应合并 | 全文 | 🟢 P3 |
| `tests/unit/contexts/` 7 文件 | POD 测试太浅 | <50 行/文件 | 🟢 P3 |
| 7 个 `src/**/AGENTS.md` | 与根 AGENTS.md 70%+ 重复 | 全文 | 🟢 P3 |
| `include/ptxsim/thread_context.h` | 23 个 include | line 1-30 | 🟢 P3 |
| `include/ptxsim/testing/memory_test_utils.h` | 18 个 include | line 1-25 | 🟢 P3 |
| `src/CMakeLists.txt` | 手动 `set(SOURCES)` 68 个 .cpp | line 41-48 | 🟢 P3 |
| `CMakeLists.txt` | 仅 1 个 cmake option | line 34 | 🟢 P3 |
| `src/ptxsim/instructions/instruction_handlers.cpp` | X-Macro 仅 1 次 | line 190 | 🟢 P3 |
| `src/ptxsim/instructions/atomic.cpp` | ~~stub（C-16 = A-9 合并）~~ ✅ **A-9 archive 2026-07-06** | 全文 | 🟢 |
| `src/ptx_parser/ptx_visitor_atom.cpp` | 硬编码 ptx_op.def 格式（DRY） | line 28 | 🟢 P3 |
| `ptx_types.cpp` + `statement_context.cpp` | 3 处 `assert(false && "...")` | — | 🟢 P3 |
| `tests/e2e/divergence/test_divergence.cu` | 仅 1 个非 barrier E2E | 全文 | 🟢 P3 |

### A.3 文档债务（D 系列）

| 文件 | 债务 | 行号 | 优先级 |
|------|------|------|--------|
| `docs/README.md` | 索引可能仍漏（parser-completeness 修复后需 re-verify） | — | 🟡 P2 |
| `docs/skills/README.md` | 列 9 vs 实际 18 技能 | — | 🟢 P3 |
| `openspec/changes/archive/2026-06-24-phase3-cvt-precision-bugfix` 等 6 个 | 缺 design.md | — | 🟡 P2 |
| `docs/skills/` vs `.opencode/skills/` | 内容分叉 | — | 🟢 P3 |
| `docs/audits/HEALTH-AUDIT-2026-06-21.md` | 8 个事实错误未合并 | — | 🟡 P2 |

---

## 附录 B：OpenSpec Change 命名规范

| 类型 | 前缀 | 示例 |
|------|------|------|
| 新功能 | `add-` / `implement-` | `add-extern-function-declaration`, `implement-atomic-cas` |
| 重构 | `refactor-` / `split-` / `merge-` | `refactor-arithmetic-handlers`, `split-ptx-visitor-god-class` |
| 修复 | `fix-` | `fix-cvt-strategy-actual-split`, `fix-multi-ptx-warning` |
| 同步 | `sync-` | `sync-readme-after-tcgen05` |
| 文档 | `docs-` | `docs-readme-fixes-remaining` |
| 清理 | `cleanup-` | `cleanup-deprecated-barrier-apis`, `cleanup-openspec-orphans` |
| 测试 | `add-*-tests` / `coverage-` | `add-nested-divergence-tests`, `expand-e2e-divergence-coverage` |

**`Ref:` 约定**：任何修补已归档 change 的新 change，proposal.md 必须有：

```markdown
## What Changes

**显式标记**：本 change 是 `archive/<date>-<original-name>/` 的修补（**非 amend**），不修改原 archive 内容，仅通过 `Ref:` 链接建立 lineage。
```

---

## 附录 C：相关 ADR / Specs

| ADR | 标题 | 影响范围 |
|-----|------|---------|
| ADR-0008 | barrier semantics | barrier module + cleanup-deprecated-barrier-apis |
| ADR-0015 | cvt strategy pattern | fix-cvt-strategy-actual-split |
| ADR-0016 | Blackwell-only tcgen05 | implement-wmma-tensor-core-tcgen05 |
| ADR-0018 | distributed shared memory (cta_group::2) | H5+ 候选 |

| Spec | 标题 | 影响 change |
|------|------|------------|
| `parser-deadcode-cleanup` | parser 死代码清理 | parser-completeness |
| `parser-multi-ptx-warning` | multi-PTX cubin 警告 | parser-completeness |
| `extern-function-parse-coverage` | extern function 双路径 oracle | add-extern-function-declaration |
| `stub-explicit-failure` | stub 必须显式抛错 | replace-silent-stub-failures |
| `wmma-tensor-core` | Blackwell tcgen05 实现 | implement-wmma-tensor-core-* |
| `docs-discoverability` | docs 索引统一 | docs-readme-rebuild |
| `pc-api` | pc 统一 API | pc unification |