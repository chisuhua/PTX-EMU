## Why

PTX-EMU parser 代码中存在 12 条 TODO/FIXME 债务条目（基于 git verify，非审计假设），其中：

- **1 条真 P0**：`src/ptx_parser/ptx_parser.cpp:60` multi-PTX cubin 静默截断（`ptx_code = of_ptx.str()` 覆盖语义），遇多 section cubin 时只保留最后 section，无任何 warning
- **3 条死代码**（lessons-learned §20 第 3 类）：
  - `ptx_visitor.cpp:323` `calculateTypeSize()` 硬编码 `return 4;`，无调用者（`getBytes(qualifier)` 在 `qualifier_utils.cpp:56` 已替代其功能）
  - `ptx_visitor.cpp:303` + `ptx_visitor.cpp:607` `processFunctionAttributes()` 死代码 + 关联 TODO 注释
- **5 条过期注释/已处理设计**（lessons-learned §20 第 4 类）：
  - `statement_context.h:24` FIXME size（`std::optional` 语义正确，ptx_interpreter.cpp:124-145 已 BUGFIX 处理）
  - `barrier.cpp:11` Stage 3 TODO（所有子项已 ✓）
  - `ptx_interpreter.cpp:124` 过期 FIXME 注释（已修但注释未清）
  - `ptx_visitor.cpp:435-436` P0 cleanup TODO（已修复的注释）
  - `thread_context.cpp:171,181` dumpContext 注释代码（debug 功能不完整但非关键）
- **2 条设计选择**（lessons-learned §20 第 2 类）：
  - `thread_context.cpp:410` FIXME（`commit_operand` 已正确使用 `instr.qualifiers`，不是 bug）
  - `ptx_visitor.cpp:363` extern 函数声明（提案未覆盖，建议独立 change）

**Scope 修订说明**（lessons-learned §20 + Checklist H）：

按 Metis pre-implementation review（2026-07-05）实证验证，本 change 的 scope 从初始提案的"10 条债务 + 6 Phase"修订为 **3 Phase 清理 + 1 P0 fix**，主要差异：
1. **6 条不需改代码**（过期注释清理，仅注释更新）
2. **3 条死代码删除**（替代原"实施"计划，因函数无调用者）
3. **仅 1 条实质修复**（multi-PTX warning）

完整 Metis 审计结果：`openspec/changes/parser-completeness/design.md` §Metis Review History。

## What Changes

**核心变更**：

- **删除 3 处死代码**（`ptx_visitor.cpp`）：
  - `calculateTypeSize()` 函数体（line 323-326）— 替代品 `getBytes(qualifier)` 已存在
  - `processFunctionAttributes()` 函数体（line 303-310）— 0 调用者
  - `visitFunctionDecl()` 中 `// TODO: Process function attributes` 注释（line 607）
- **更新 5 处过期注释**：
  - `statement_context.h:24` FIXME → 说明 `std::optional` 语义
  - `barrier.cpp:11` Stage 3 TODO → 删除（所有子项已 ✓）
  - `ptx_interpreter.cpp:124` 过期 FIXME → 删除或精简
  - `ptx_visitor.cpp:435-436` P0 cleanup → 精简为说明性注释
  - `thread_context.cpp:171,181` dumpContext TODO → 标注"debug stub, not implemented"
- **P0 fix**：multi-PTX cubin warning（`ptx_parser.cpp:60`）
  - 检测 PTX section 数量 > 1 时输出 `PTX_WARN_EMU`
  - 恢复 `+=` 累加语义（保留所有 section 而非仅最后）
- **同步 4 个文档**：
  - `src/ptxsim/instructions/AGENTS.md` STRUCTURE 章节
  - `src/ptxsim/core/AGENTS.md` DUMP_CONTEXT 章节（如适用）
  - `docs/dev-process/lessons-learned.md` §P0 multi-PTX postmortem
  - `openspec/changes/archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern` 注释（如果有 cross-ref）

**显式排除**（不放入本 change scope）：
- `ptx_visitor.cpp:363` extern 函数声明 → 新建独立 change `add-extern-function-declaration`
- `thread_context.cpp:410` FIXME → 设计选择，无代码改动需要
- `processFunctionAttributes` 实现（非删除）→ 需 ANTLR 语法扩充，触及 `ptx-grammar-modification` 高危流程

**预估代码改动量**：~20 行删除 + ~15 行注释更新 + ~10 行 multi-PTX warning = **约 45 行净改动**。

## Capabilities

### New Capabilities

- `parser-deadcode-cleanup`: 删除 `calculateTypeSize` + `processFunctionAttributes` 死代码（~15 行）+ 更新关联 TODO 注释
- `parser-multi-ptx-warning`: 当 cubin 含多个 PTX section 时输出 `PTX_WARN_EMU`，恢复 `+=` 累加语义

### Modified Capabilities

- `stub-explicit-failure`: AGENTS.md "KNOWN LIMITATIONS" 章节 — multi-PTX cubin 描述从"仅提取第一个 PTX"改为"输出 warning + 保留所有 section"

## Impact

**受影响的代码/文件**：

| 文件 | 改动 | 影响 |
|------|------|------|
| `src/ptx_parser/ptx_visitor.cpp` | 死代码删除 + 注释更新 | line 303, 323, 363-366, 435-436, 607（5 处） |
| `include/ptx_ir/statement_context.h` | 注释更新 | line 24（1 处） |
| `src/ptxsim/instructions/barrier.cpp` | 注释删除 | line 11（1 处） |
| `src/cudart/ptx_interpreter.cpp` | 注释精简 | line 124（1 处） |
| `src/ptxsim/core/thread_context.cpp` | 注释更新 | line 410, 171, 181（3 处） |
| `src/ptx_parser/ptx_parser.cpp` | P0 fix（multi-PTX warning + `+=`）| line 56-62（约 10 行） |
| `src/ptxsim/instructions/AGENTS.md` | 同步 multi-PTX 限制 | KNOWN LIMITATIONS 章节 |

**受影响的 ADR**：
- 无直接 ADR 影响（无架构决策变更）

**测试覆盖**：
- 现有 `tests/unit/parser/` `tests/unit/ptx_ir/` 不存在 — 需建立测试基础设施
- 现有 1 个 parser 标签测试（`unit_parse_immediate` #90）— 不受影响
- multi-PTX warning 需要新增测试（构造 multi-section cubin mock）

**回归风险**：
- 🟢 低：死代码删除无行为变更（grep 验证 0 调用者）
- 🟡 中：multi-PTX 累加语义可能改变现有 PTX 加载行为（仅在 cubin 含多 section 时）
- 🟢 低：注释更新无代码逻辑变更

**Lessons-learned 集成**：
- ✅ Checklist E（artifacts 必 tracked）：2-Phase commit（artifacts FIRST）
- ✅ Checklist F（git verify）：所有引用 commit hash 而非文件路径
- ✅ Checklist H（pre-impl review）：Metis 已审计，scope 修订为 3 Phase
- ✅ Checklist G（lifecycle）：本 change 是**新** change，不是 amend

**关联 change**：
- `archive/2026-07-05-fix-cvt-strategy-actual-split/` — Stale artifact fix 先例（6→3 Phase 修订模式）
- `archive/2026-07-04-replace-silent-stub-failures/` — multi-PTX warning 模式（PTX_WARN_EMU 用法）
- `archive/2026-07-04-implement-wmma-tensor-core-tcgen05/` — 14 fixes 实施（Checklist I "重大功能交付" 模式）