# Tasks: Cleanup wmma Namespace

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec
> **前置 changes**(必须 archive): Change-1, 2, 3a, 3b(3d 可选)
> **范围**: 3 atomic commits
>
> ## Pre-Existing State (2026-07-07)
>
> `implement-tcgen05-handlers-core` (commit `df6dde7`) 已提前完成部分工作:
> - ✅ `src/ptxsim/instructions/wmma.cpp`: 564→30 行(5 个 `execute_tcgen05_*` 已移除,仅剩 `WmmaHandler::processWmmaOperation` throw stub)
> - ✅ 4 个旧 WmmaHandler API 测试已删除
> - ✅ `tcgen05.cpp` 已创建(5 `processTcgen05Xxx` handler)
>
> **本 change 剩余工作**: 删除 IR 层 dead code(S_WMMA/WmmaInstr/WmmaType) + grammar wmma rules + visitor/parser macros + 最终删除 30 行 wmma.cpp stub

## 0. Pre-Implementation Review

- [ ] 0.1 跑 Metis:
  - [ ] 0.1.1 `grep -rn "S_WMMA\|WmmaInstr\|WmmaType\|makeWmmaInstr"` 列出所有引用点
  - [ ] 0.1.2 验证 Change-3b 已 archive(5 core handler 替代 wmma)
  - [ ] 0.1.3 验证 Change-3d 已 archive 或文档化为"optional"(extended handler 替代 wmma 剩余)
  - [ ] 0.1.4 验证 `tests/integration/tcgen05/test_tcgen05_*.cpp` 已迁移(无 S_WMMA 引用)

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `git checkout -b feat/cleanup-wmma-namespace`
- [ ] 1.2 `git add openspec/changes/cleanup-wmma-namespace/`
- [ ] 1.3 `git commit -m "docs(openspec): add cleanup-wmma-namespace artifacts (ADR-0016)"`

## 2. Phase 1: 删除 dead code(commit 2)

### 2.1 IR 层删除

- [ ] 2.1.1 编辑 `include/ptx_ir/ptx_op.def`:删除 `X(S_WMMA, wmma, Wmma, 4, WMMA_INSTR, matrix)` (line 127)
- [ ] 2.1.2 编辑 `include/ptx_ir/statement_context.h`:删除 `struct WmmaInstr { WmmaType wmmaType; ... }`(line 148-152) + `InstrVariant` 中的 `WmmaInstr,`
- [ ] 2.1.3 编辑 `include/ptx_ir/ptx_types.h`:删除 `enum WmmaType { ... };`(line 30)
- [ ] 2.1.4 编辑 `include/ptx_ir/statement_factory.h`:删除 `makeWmmaInstr` 函数(line 265-275)

### 2.2 Grammar 删除

- [ ] 2.2.1 编辑 `src/grammar/ptxInstructions.g4`:删除 5 个 wmma rules(`wmmaInst` / `wmmaOp` / `wmmaLayout` / `wmmaShape` / `wmmaKind`,line 424-433)
- [ ] 2.2.2 编辑 `src/grammar/ptxLexer.g4`:删除 `WMMA: 'wmma';` (line 373)
- [ ] 2.2.3 跑 `cmake --build build --target GenerateParser` 验证 ANTLR 重新生成

### 2.3 Visitor/Handler 宏删除

- [ ] 2.3.1 编辑 `include/ptx_parser/ptx_visitor_categories.h`:删除 `#define VISITOR_WMMA_INSTR ...`(line 14-15)
- [ ] 2.3.2 编辑 `include/ptx_parser/ptx_visiter.h`:删除 `#define VISITOR_DECL_WMMA_INSTR ...`(line 51-52)
- [ ] 2.3.3 编辑 `include/ptx_parser/ptx_parser.h`:删除 `STATEMENT_DECL_WMMA_INSTR`(line 153)
- [ ] 2.3.4 编辑 `src/ptxsim/instruction_handlers.cpp`:删除 `IMPLEMENT_WMMA_INSTR_HANDLER`(line 110-120)
- [ ] 2.3.5 删除 `src/ptxsim/instructions/wmma.cpp` 剩余 30 行 stub 文件(df6dde7 已缩减 564→30 行)
- [ ] 2.3.6 编辑 `src/CMakeLists.txt`:删除 wmma.cpp 引用(同步移除 `src/ptxsim/instructions/wmma.cpp` 行)
- [ ] 2.3.7 编辑 `src/ptxsim/instructions/AGENTS.md`:删除 wmma.cpp 描述

### 2.4 pre-Blackwell 测试删除

- [ ] 2.4.1 删除 `tests/ptx/dummy-float.1.sm_80.ptx`(pre-Blackwell fixture)
- [ ] 2.4.2 删除 `tests/ptx/dummy.1.sm_80.ptx`
- [ ] 2.4.3 编辑 `tests/ptx/test_all_ptx.sh` 排除(若需要)

### 2.5 验证

- [ ] 2.5.1 `cmake --build build` 验证编译
- [ ] 2.5.2 `ctest --output-on-failure` 验证
- [ ] 2.5.3 `./tests/ptx/test_all_ptx.sh` 验证
- [ ] 2.5.4 `grep -rn "S_WMMA\|WmmaInstr\|WmmaType\|makeWmmaInstr\|VISITOR_WMMA_INSTR\|IMPLEMENT_WMMA_INSTR" src/ include/ tests/` 验证零输出

### 2.6 Commit

- [ ] 2.6.1 `git add -A` (大改)
- [ ] 2.6.2 `git commit -m "refactor: complete removal of wmma namespace (ADR-0016, §20 '已实施但未清理' fix)"`

## 3. Phase 2: 验证(commit 3)

- [ ] 3.1 跑 `ctest --output-on-failure` 最终验证
- [ ] 3.2 跑 `./tests/ptx/test_all_ptx.sh` 验证
- [ ] 3.3 写 `docs/audits/2026-07-XX-wmma-cleanup-audit.md`(列出删除的所有 dead code + grep 验证)
- [ ] 3.4 `git add docs/audits/`
- [ ] 3.5 `git commit -m "docs(audit): add wmma cleanup audit report (ADR-0016)"`

## 4. Phase 3: Archive(commit 4,per Checklist G)

- [ ] 4.1 `openspec archive cleanup-wmma-namespace --yes`
- [ ] 4.2 `ctest --output-on-failure` 最终验证
- [ ] 4.3 `git add openspec/changes/archive/`
- [ ] 4.4 `git commit -m "chore(openspec): archive cleanup-wmma-namespace (ADR-0016)"`

## Final Validation

- [ ] 5.1 `git log --oneline | head -4` 显示 4 atomic commits
- [ ] 5.2 `grep -rn "wmma\|WMMA" src/ include/ | grep -v "ADRS\|docs/"` 验证仅保留 `wmma*` 作为 historical reference
- [ ] 5.3 根 `AGENTS.md` 标注 "pre-Blackwell 永久拒绝" 为历史记录

## Risks Recap

| Risk | Mitigation |
|------|------------|
| R1: 误删仍需用的代码 | pre-Blackwell 仅 throw,可全删 |
| R2: 删除后旧测试 fail | pre-Blackwell 测试已无意义,应删 |
