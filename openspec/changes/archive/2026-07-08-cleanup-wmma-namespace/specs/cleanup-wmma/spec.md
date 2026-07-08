## ADDED Requirements

### Requirement: WMMA namespace fully purged from codebase
No source file SHALL reference any wmma-related type, function, or
macro (S_WMMA, WmmaInstr, WmmaType, makeWmmaInstr, WmmaHandler,
WmmaPipelineHandler, processWmmaOperation, VISITOR_WMMA_INSTR,
IMPLEMENT_WMMA_INSTR_HANDLER, DECLARE_WMMA_INSTR_HANDLER,
ptx_visitor_wmma.cpp, wmma.cpp).

#### Scenario: grep finds zero wmma source references
- **WHEN** `grep -rn "S_WMMA\|WmmaInstr\|WmmaType\|makeWmmaInstr\|WmmaPipeline\|processWmma\|VISITOR_WMMA\|IMPLEMENT_WMMA\|DECLARE_WMMA" src/ include/ tests/ | grep -v "\.git/\|build/\|openspec/\|docs/\|AGENTS\|tcgen05\|archive/"` is run
- **THEN** zero output (all wmma code deleted)

## REMOVED Requirements

### Requirement: S_WMMA StatementType SHALL be deleted
The `S_WMMA` StatementType enum value (in `include/ptx_ir/ptx_op.def`)
SHALL be deleted, since pre-Blackwell WMMA is permanently unsupported
(per ADR-0016).

#### Scenario: S_WMMA no longer present
- **WHEN** `grep "S_WMMA" include/ptx_ir/ptx_op.def` is run
- **THEN** zero output (deleted)

### Requirement: WmmaInstr struct + WmmaType enum SHALL be deleted
The `WmmaInstr` struct (in `include/ptx_ir/statement_context.h`) and
`WmmaType` enum SHALL be deleted, as no handler uses them after
Change-3b/3d.

#### Scenario: WmmaInstr removed
- **WHEN** `grep "struct WmmaInstr" include/ptx_ir/statement_context.h` is run
- **THEN** zero output (deleted)

#### Scenario: WmmaType removed
- **WHEN** `grep "enum WmmaType" include/ptx_ir/ptx_types.h` is run
- **THEN** zero output (deleted)

### Requirement: src/ptxsim/instructions/wmma.cpp SHALL be deleted
The entire `wmma.cpp` file SHALL be deleted (pre-Blackwell path is
no longer needed after Change-3b removed the 5 tcgen05 handlers).

#### Scenario: wmma.cpp file removed
- **WHEN** `ls src/ptxsim/instructions/wmma.cpp` is run
- **THEN** "No such file" error (deleted)

### Requirement: pre-Blackwell WMMA grammar rules SHALL be deleted
The `wmmaInst` / `wmmaOp` / `wmmaLayout` / `wmmaShape` / `wmmaKind`
grammar rules in `src/grammar/ptxInstructions.g4` SHALL be deleted.

#### Scenario: wmma grammar rules removed
- **WHEN** `grep -E "wmmaInst|wmmaOp|wmmaLayout|wmmaShape|wmmaKind" src/grammar/ptxInstructions.g4` is run
- **THEN** zero output (deleted)

### Requirement: pre-Blackwell test fixtures SHALL be removed or relabeled
Any `tests/ptx/dummy*sm_80.ptx` or pre-Blackwell fixtures SHALL be
removed (per ptx-lessons-learned §20 "已实施但未清理").

#### Scenario: pre-Blackwell fixtures removed
- **WHEN** `ls tests/ptx/dummy*sm_80.ptx` is run
- **THEN** "No such file" (or relabeled to `sm_100+`)
