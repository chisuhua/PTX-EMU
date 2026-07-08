# cleanup-wmma Specification

## Purpose
TBD - created by archiving change cleanup-wmma-namespace. Update Purpose after archive.
## Requirements
### Requirement: WMMA namespace fully purged from codebase
No source file SHALL reference any wmma-related type, function, or
macro (S_WMMA, WmmaInstr, WmmaType, makeWmmaInstr, WmmaHandler,
WmmaPipelineHandler, processWmmaOperation, VISITOR_WMMA_INSTR,
IMPLEMENT_WMMA_INSTR_HANDLER, DECLARE_WMMA_INSTR_HANDLER,
ptx_visitor_wmma.cpp, wmma.cpp).

#### Scenario: grep finds zero wmma source references
- **WHEN** `grep -rn "S_WMMA\|WmmaInstr\|WmmaType\|makeWmmaInstr\|WmmaPipeline\|processWmma\|VISITOR_WMMA\|IMPLEMENT_WMMA\|DECLARE_WMMA" src/ include/ tests/ | grep -v "\.git/\|build/\|openspec/\|docs/\|AGENTS\|tcgen05\|archive/"` is run
- **THEN** zero output (all wmma code deleted)

