## Context

PTX-EMU parser 已通过 **两条并行路径**完整支持 extern 函数声明（Metis pre-impl review 2026-07-05 实证）：

### 当前实现状态

**路径 1：ANTLR tree walker（自动调用）**

| 组件 | 位置 | 状态 |
|------|------|------|
| `EXTERN` token | `src/grammar/ptxLexer.g4:50` | ✅ 存在（`.extern`） |
| `externFuncStatement` rule | `src/grammar/ptxParser.g4` | ✅ 存在 |
| `exitExternFuncStatement` handler | `src/ptx_parser/ptx_parser.cpp:996-1009` | ✅ 实施 |
| `ExternFuncDecl` struct | `include/ptx_ir/ptx_context.h:14-18` | ✅ 存在 |
| `ptxContext.externFuncs` | `include/ptx_ir/ptx_context.h:22` | ✅ 容器 |
| `push_back(decl)` | `src/ptx_parser/ptx_parser.cpp:1004` | ✅ 写入 |

**路径 2：PtxVisitor 手动遍历**

| 组件 | 位置 | 状态 |
|------|------|------|
| `visitFunctionDecl` extern form | `src/ptx_parser/ptx_visitor.cpp:486-503` | ✅ 实施 |
| `ctx->ID()` extern name extract | `src/ptx_parser/ptx_visitor.cpp:494` | ✅ 实施 |
| `ifVisibleKernel=false`（extern） | `src/ptx_parser/ptx_visitor.cpp:503` | ✅ 实施 |

### 待清理问题

| 问题 | 位置 | 严重性 |
|------|------|--------|
| Stale TODO | `src/ptx_parser/ptx_visitor.cpp:350` | 🟡 中（误导未来 reader）|
| Oracle test 缺失 | `tests/unit/parser/` 无 extern_function test | 🟡 中 |
| AGENTS.md 不同步 | 根 + `src/ptx_parser/AGENTS.md` 未描述 extern 处理 | 🟡 中 |

### 双路径关系

```
PTX source → ANTLR parse → ParseTree
                       ↓
       ┌──────────────┴──────────────┐
       ↓ (auto via listener)        ↓ (manual via visitor)
PtxListener::exitExternFuncStatement  PtxVisitor::visitFunctionDecl
       ↓                                ↓
ptxContext.externFuncs.push_back     currentKernel->kernelName
                                     currentKernel->ifVisibleKernel=false
```

两条路径**互补不冲突**：path 1 收集到符号表供运行时调用查找，path 2 填充 kernel context 供指令执行。

## Goals / Non-Goals

### Goals

1. **删除 stale TODO** at `ptx_visitor.cpp:350` — 此函数不处理 function decl，TODO 误导
2. **添加 oracle test** — 3 个测试场景覆盖 extern 函数声明的 3 种形式
3. **同步 AGENTS.md** — 根 + `src/ptx_parser/AGENTS.md` 描述实际处理状态

### Non-Goals（明确排除）

1. ❌ **ANTLR grammar 修改** — grammar 已支持 `EXTERN` token + `externFuncStatement` 规则
2. ❌ **重构 `visitFunctionDecl`** — 现状正确（line 494-503 处理 extern form）
3. ❌ **重构 `exitExternFuncStatement`** — 现状正确（line 996-1009 处理 params + push_back）
4. ❌ **实现 extern 函数调用** — 独立 change `add-user-function-call`（debt A-6）
5. ❌ **新增 ADR** — 经查无对应 ADR，无需创建

## Decisions

### Decision 1: 删除 TODO 注释 vs 修复 TODO

**Choice**: 删除 TODO 注释（不改函数行为）

**Rationale**：
- `visitDeclaration` 只处理 directives + variableDecl，**不**处理 function decl
- Function decl 是 PtxFile 级别的兄弟节点，由 `visitFunctionDecl` 直接处理
- TODO 注释误导未来 reader 试图在此函数加 extern 分支（错误方向）
- 删除 TODO + 添加注释解释"function decl 不在此处处理"是最小且正确的修复

**Alternatives Considered**：
- (A) 删除 TODO + 加说明注释 — **采纳**
- (B) 在 visitDeclaration 加 extern 分支 — 拒绝（错误位置 + 重复功能）
- (C) 保留 TODO + 标注 stale — 拒绝（仍误导）

### Decision 2: oracle test 范围

**Choice**: 3 个测试场景（简单 / 带参数 / vs entry kernel）

**Rationale**：
- 简单形式：覆盖 `.extern .func funcName`（最常用）
- 带参数：覆盖参数解析（验证 `tempExternFuncParams` 复制到 `decl.params`）
- vs entry kernel：验证 `ifEntryKernel` 区分（extern = false，entry = true）

**Alternatives Considered**：
- (A) 3 个场景 — **采纳**（平衡覆盖率与可维护性）
- (B) 1 个综合场景 — 拒绝（粒度太粗，失败定位困难）
- (C) 5+ 个场景 — 拒绝（scope 膨胀，超出清理目标）

### Decision 3: AGENTS.md 同步粒度

**Choice**: 根 AGENTS.md + `src/ptx_parser/AGENTS.md` 双层同步

**Rationale**：
- 根 AGENTS.md "已知限制" 章节：从"未处理"改为"已支持"是**事实修正**（不是新增内容）
- 子 AGENTS.md STRUCTURE：添加"双路径处理"说明供开发者快速定位
- 不新增 ADR：无架构决策变更

**Alternatives Considered**：
- (A) 仅根 AGENTS.md — 拒绝（开发者查 source 时看不到详细路径）
- (B) 仅子 AGENTS.md — 拒绝（新人从根 AGENTS.md 仍看不到状态）
- (C) 双层同步 — **采纳**

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **R1: TODO 删除后未来 reader 不知为何没有 extern 分支** | 🟡 中 | 🟢 低 | 添加说明注释 "function decl processed in visitFunctionDecl, not here" |
| **R2: oracle test 失败揭示 parser bug** | 🟡 中 | 🟡 中 | (1) Phase 2 前先跑 baseline 确认现状 (2) 失败时立即 revert Phase 2 (3) 视为新 bug，单独修复 |
| **R3: AGENTS.md 描述与代码不一致** | 🟢 低 | 🟢 低 | Oracle test PASS 作为"描述正确性"的隐式验证 |
| **R4: Phase 2 oracle test 触发 pre-existing parser bug** | 🟡 中 | 🟡 中 | 已知 `ptx_parser.cpp` 有 LSP 错误（pre-existing 债务），oracle test 用最小 PTX 避免触发 |

## Migration Plan

### Phase 0: Artifacts Git-Tracking（强制，per §6）

```bash
# Step 0.1: 工作分支
git checkout -b refactor/add-extern-function-declaration

# Step 0.2: git add artifacts FIRST
git add openspec/changes/add-extern-function-declaration/
git status  # 验证 5 个新文件
git ls-files openspec/changes/add-extern-function-declaration/  # 不应为空

# Step 0.3: commit artifacts（独立 commit）
git commit -m "docs(openspec): add add-extern-function-declaration artifacts"
```

### Phase 1: 删除 stale TODO + oracle test（Fix #1）

```bash
# Step 1.1: 删除 TODO + 加说明注释（ptx_visitor.cpp:350）
# BEFORE:
// TODO: Add extern function declaration handling

# AFTER:
// function decl 由 visitFunctionDecl 直接处理（不在 declaration 上下文中）

# Step 1.2: 创建 oracle test
# Edit tests/unit/parser/test_extern_function.cpp（新建）
# - 3 个 TEST_CASE：simple / with_params / vs_entry

# Step 1.3: 注册 test
# Edit tests/unit/CMakeLists.txt
add_catch_test(unit_extern_function
    parser/test_extern_function.cpp
)
set_tests_properties(unit_extern_function PROPERTIES LABELS "unit;parser;extern")

# Step 1.4: 编译验证
cmake --build build --target unit_extern_function

# Step 1.5: oracle test PASS
ctest -R unit_extern_function --output-on-failure

# Step 1.6: 回归验证
ctest -L "unit;ptx" --output-on-failure  # 必须 100% PASS
./tests/ptx/test_all_ptx.sh  # 必须 100% PASS

# Step 1.7: Commit Fix #1
git commit -am "refactor(parser): delete stale extern TODO + oracle test (Fix #1)"
```

### Phase 2: AGENTS.md 同步 + 二次验证（Fix #2）

```bash
# Step 2.1: 根 AGENTS.md 更新
# Edit AGENTS.md "已知限制" 章节
# Remove: extern 函数声明未处理（事实错误）
# Add: extern 函数声明已支持（双路径：PtxListener + PtxVisitor）

# Step 2.2: 子 AGENTS.md 更新
# Edit src/ptx_parser/AGENTS.md STRUCTURE
# Add: extern 函数处理 — PtxListener.exitExternFuncStatement + PtxVisitor.visitFunctionDecl

# Step 2.3: 二次 ctest
ctest --output-on-failure  # 100% PASS

# Step 2.4: Commit Fix #2
git commit -am "docs(parser): sync AGENTS.md extern function status (Fix #2)"
```

### Phase 3: Archive + Merge（per Checklist G + I）

```bash
# Step 3.1: 二次 ctest 完整 sanity
./scripts/sanity.sh

# Step 3.2: Archive
openspec archive add-extern-function-declaration --yes

# Step 3.3: Post-archive commit
git add openspec/changes/add-extern-function-declaration/ openspec/changes/archive/<date>-add-extern-function-declaration/ openspec/specs/extern-function-parse-coverage/
git commit -m "chore(openspec): archive add-extern-function-declaration (Checklist G)"

# Step 3.4: Merge to main
git checkout main
git merge --no-ff refactor/add-extern-function-declaration
```

### Rollback Strategy

```bash
# 任何 Phase 失败立即 revert 该 Phase
git revert HEAD
cmake --build build
ctest --output-on-failure
```

## Open Questions

### OQ-1: oracle test 是 mock 还是端到端？

**Question**: `externFuncs` 填充依赖 ANTLR parse tree。oracle test 应该：
- (a) 构造 mock PtxContext + externFuncs（无需 PTX 输入）
- (b) 加载真实 PTX 字符串走 ANTLR parse（端到端）

**Status**: Phase 1 实施时决定。倾向 (b) 端到端（更真实，且 parser-completeness 的 oracle test 也用类似模式）。

### OQ-2: 是否同步更新 debt-audit-2026-07-02.md？

**Question**: A-8 PARTIAL 标记 extern 函数声明未处理，本 change 完成后是否更新为 RESOLVED？

**Status**: Phase 2 同步更新。

### OQ-3: parser-completeness 的 lessons-learned §22 是否更新？

**Question**: 本 change 与 parser-completeness 紧密关联（MR-7 排除项）。是否在 §22 追加一段"parser 系列连续 change"？

**Status**: 暂不在本 change scope（避免 scope 膨胀）。下次 lessons-learned 更新时统一追加。