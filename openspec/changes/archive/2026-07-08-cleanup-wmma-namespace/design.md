## Context

Change-1/2/3a/3b(3d) archive 后,pre-Blackwell WMMA 路径已完全 dead code。`df6dde7` (implement-tcgen05-handlers-core) 已将 `wmma.cpp` 从 564 行缩减至 30 行(仅剩 `WmmaHandler::processWmmaOperation` throw stub)。本 change 彻底清理剩余的 IR/grammar/visitor wmma dead code,符合 `ptx-lessons-learned` §20 "已实施但未清理模式" 的修正。

## Goals / Non-Goals

**Goals**: 完全删除 wmma 命名空间(S_WMMA / WmmaInstr / WmmaType / wmma.cpp / 5 grammar rules / pre-Blackwell fixtures)。

**Non-Goals**: 不修改任何 handler(已 archive)、不修改 tcgen05(已 archive)、不实现新功能。

## Decisions

### D1: 3 atomic commits(delete → validate → archive)

- **Phase 1**: 删除 wmma dead code(1 commit,大改)
- **Phase 2**: 验证零回归(1 commit,仅 docs/audit)
- **Phase 3**: Archive(1 commit)

### D2: pre-Blackwell 测试处理

**采纳**: 删除 `tests/ptx/dummy*sm_80.ptx`(pre-Blackwell 测试已无意义)

**拒绝**: 保留 + relabel(per 提议)— 增加无意义维护成本

### D3: docs 中 "pre-Blackwell 永久 throw" 标注

**保留**: 根 `AGENTS.md` 仍标注 "pre-Blackwell WMMA 永久抛异常"— 现在是历史记录(无代码路径),不删除

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: 删除后旧测试 fail | pre-Blackwell 测试已无意义(应删) |
| **R2**: 误删 wmma.cpp 仍需用的代码 | pre-Blackwell 仅 `throw UnsupportedInstructionException`,可全删 |

## Migration Plan

### Phase 1: 删除(1 commit)

1. `S_WMMA` 删 `include/ptx_ir/ptx_op.def:127`
2. `WmmaInstr` struct 删 `include/ptx_ir/statement_context.h`
3. `WmmaType` enum 删 `include/ptx_ir/ptx_types.h`
4. `makeWmmaInstr` 删 `include/ptx_ir/statement_factory.h`
5. `wmma.cpp` 文件删
6. 5 grammar rules 删 `src/grammar/ptxInstructions.g4`
7. `WMMA: 'wmma'` token 删 `src/grammar/ptxLexer.g4`
8. `VISITOR_WMMA_INSTR` / `IMPLEMENT_WMMA_INSTR_HANDLER` 删
9. `tests/ptx/dummy*sm_80.ptx` 删
10. `tests/integration/tcgen05/test_tcgen05_*.cpp` 已在 Change-3a 迁移
11. `src/CMakeLists.txt` 删 wmma.cpp 引用
12. `git add -A` + commit

### Phase 2: 验证(1 commit)

1. `ctest --output-on-failure` 验证
2. `./tests/ptx/test_all_ptx.sh` 验证
3. `grep -rn "S_WMMA\|WmmaInstr\|WmmaType\|makeWmmaInstr\|VISITOR_WMMA_INSTR\|IMPLEMENT_WMMA_INSTR"` 验证零输出
4. 写 `docs/audits/2026-07-XX-wmma-cleanup-audit.md`
5. commit

### Phase 3: Archive(per Checklist G)

`openspec archive` + commit

## Open Questions

无(纯 dead code 删除)。
