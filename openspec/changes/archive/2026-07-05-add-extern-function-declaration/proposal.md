## Why

PTX-EMU parser 已通过 **两条并行路径**支持 extern 函数声明（Metis pre-impl review 实证）：

1. **ANTLR tree walker 路径**：`src/ptx_parser/ptx_parser.cpp:996` `exitExternFuncStatement` — 提取 name + params，推入 `ptxContext.externFuncs`（`include/ptx_ir/ptx_context.h:22`）
2. **PtxVisitor 手动遍历路径**：`src/ptx_parser/ptx_visitor.cpp:486` `visitFunctionDecl` — 处理 `functionHeader` 和 extern form（`ctx->ID()`），设置 `ifVisibleKernel = false`

但仍残留 **stale TODO + oracle 缺失 + 文档不同步**：

- **stale TODO**：`src/ptx_parser/ptx_visitor.cpp:350` `// TODO: Add extern function declaration handling` — 此函数（`visitDeclaration`）**不处理 function decl**（function decl 在 PtxFile 级别由 `visitFunctionDecl` 处理），TODO 误导未来 reader
- **oracle test 缺失**：`tests/unit/parser/` 仅有 `test_multi_ptx.cpp`（parser-completeness Fix #2 创建），无 extern function 专项测试。`tests/ptx/parser/test-ptx.cpp:89` 仅打印 `externFuncs.size()`，无断言
- **AGENTS.md 不同步**：根 `AGENTS.md` "已知限制" 章节未描述 extern 函数支持状态

这是 `parser-completeness` change 的直接续集（Metis MR-7 显式排除项）。

**Scope 修订**（lessons-learned §20）：原 Metis 假设"extern 函数完全未支持"，实际仅 3 类清理工作（stale TODO + oracle + docs）。

## What Changes

**核心变更**：

- **删除 stale TODO**（`src/ptx_parser/ptx_visitor.cpp:350`）— 此函数不处理 function decl，TODO 误导未来 reader 试图在 `visitDeclaration` 中加 extern 分支
- **添加 oracle test**（`tests/unit/parser/test_extern_function.cpp`）— 3 个测试场景：
  - (1) `.extern .func` 简单形式 → 验证 `externFuncs` 包含 name
  - (2) `.extern .func (.param .b32 x, .b64 y) name` 带参数 → 验证 params
  - (3) extern 函数 vs entry kernel 区分 → 验证 `ifEntryKernel=false`
- **同步 AGENTS.md**（根 + `src/ptx_parser/AGENTS.md`）：
  - 根 `AGENTS.md` "已知限制" 章节：移除 "extern 函数声明未处理" 描述（实际已支持）
  - `src/ptx_parser/AGENTS.md` "STRUCTURE"：说明 extern 函数的双路径处理
- **新增 docs/adr/ 章节引用**（如有相关 ADR）— 经查无对应 ADR，不创建

**显式排除**（不放入本 change scope）：
- ❌ ANTLR grammar 修改 — 不需要（grammar 已支持 `EXTERN` token + `externFuncStatement` 规则）
- ❌ 重构 `PtxVisitor::visitFunctionDecl` — 现状正确，无需重构
- ❌ 修改 `exitExternFuncStatement` 实现 — 现状正确
- ❌ 新增 extern 函数调用支持 — 独立 change `add-user-function-call`（即 debt A-6）

**预估代码改动量**：~10 行删除（stale TODO + 注释）+ ~80 行 oracle test + ~5 行 AGENTS.md = **约 95 行净改动**。

## Capabilities

### New Capabilities

- `extern-function-parse-coverage`: oracle test 覆盖 3 种 extern function 形式（简单 / 带参数 / vs entry kernel），所有测试 100% PASS

### Modified Capabilities

- `parser-multi-ptx-warning`: AGENTS.md "已知限制" 章节 — extern 函数状态从"未处理"改为"已支持（双路径：PtxListener + PtxVisitor）"

## Impact

**受影响的代码/文件**：

| 文件 | 改动 | 影响 |
|------|------|------|
| `src/ptx_parser/ptx_visitor.cpp` | 删除 line 350 stale TODO 注释 | 1 行（避免误导）|
| `tests/unit/parser/test_extern_function.cpp` | 新建 oracle test（3 TEST_CASE）| ~80 行 |
| `tests/unit/CMakeLists.txt` | 注册 `unit_extern_function` | 5 行 |
| `AGENTS.md` | "已知限制" extern 函数描述更新 | 1 行 |
| `src/ptx_parser/AGENTS.md` | STRUCTURE 章节增加 extern 函数说明 | ~5 行 |

**受影响的 ADR**：
- 无直接 ADR 影响（无架构决策变更）

**测试覆盖**：
- 新增 `unit_extern_function`（3 scenarios）
- 现有 79 unit + 15 ptx tests 不受影响

**回归风险**：
- 🟢 低：仅删除 1 行 stale TODO（无行为变更）
- 🟢 低：oracle test 是 additive（仅添加）
- 🟢 低：AGENTS.md 同步（仅文档）

**Lessons-learned 集成**：
- ✅ Checklist E（artifacts 必 tracked）：2-Phase commit（artifacts FIRST）
- ✅ Checklist F（git verify）：引用 commit hash 而非文件路径
- ✅ Checklist G（lifecycle）：本 change 是**新** change，非 amend
- ✅ Checklist H（pre-impl review）：Metis 已审计 → scope 修订
- ✅ Checklist I（重大功能交付）：本 change 范围小，无需根 README 同步

**关联 change**：
- `archive/2026-07-05-parser-completeness/`（直接续集，MR-7 排除项）
- `archive/2026-07-05-fix-cvt-strategy-actual-split/`（stale artifact 修复模式先例）
- `archive/2026-06-24-phase3-t2-6-cvt-strategy-pattern/`（死代码删除模式参考）