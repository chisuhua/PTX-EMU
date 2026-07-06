## ADDED Requirements

### Requirement: IR MUST Provide 11 Independent StatementType Enums for tcgen05

The system SHALL provide 11 independent `S_TCGEN05_*` StatementType enum values in `include/ptx_ir/ptx_op.def`, one per Blackwell tcgen05 instruction family. Each enum MUST be registered via X-Macro pattern with `struct_kind = TCGEN05_INSTR`. The system MUST also DELETE the existing `S_WMMA` enum (per ADR-0016: pre-Blackwell wmma is permanently unsupported).

The 11 enums are:
- `S_TCGEN05_ALLOC` (tcgen05.alloc)
- `S_TCGEN05_DEALLOC` (tcgen05.dealloc)
- `S_TCGEN05_RELINQUISH` (tcgen05.relinquish_alloc_permit)
- `S_TCGEN05_LD` (tcgen05.ld)
- `S_TCGEN05_ST` (tcgen05.st)
- `S_TCGEN05_CP` (tcgen05.cp)
- `S_TCGEN05_MMA` (tcgen05.mma, includes .sp and .block_scale variants)
- `S_TCGEN05_MMA_WS` (tcgen05.mma.ws)
- `S_TCGEN05_COMMIT` (tcgen05.commit, includes .arrive variant)
- `S_TCGEN05_WAIT` (tcgen05.wait, includes .load and .store)
- `S_TCGEN05_FENCE` (tcgen05.fence)

#### Scenario: ptx-op-def-has-11-entries
- **WHEN** the file `include/ptx_ir/ptx_op.def` is read
- **THEN** 11 new `X(S_TCGEN05_*, ..., TCGEN05_INSTR, tcgen05)` entries exist
- **AND** no `S_WMMA` entry exists
- **AND** the X-Macro list is well-formed (no syntax errors)

#### Scenario: tcgen05-instr-struct-kind-defined
- **WHEN** the X-Macro is expanded via `IMPLEMENT_TCGEN05_INSTR_HANDLER`
- **THEN** a `Tcgen05Handler` weak symbol is generated per enum
- **AND** the weak symbol is registered for dispatch

#### Scenario: wmma-enum-removed-no-orphan-references
- **WHEN** `grep -rn "S_WMMA" src/ include/ tests/` is run
- **THEN** zero references to `S_WMMA` exist (per Change-3 cleanup, this change MUST remove all references)
- **AND** any leftover references are fixed in this change

### Requirement: IR MUST Provide Tcgen05Instr Struct Independent of WmmaInstr

The system SHALL provide a new `Tcgen05Instr` struct in `include/ptx_ir/statement_context.h` that is structurally independent of any pre-Blackwell `WmmaInstr`. The struct MUST contain:

- `Tcgen05OpKind op_kind`: enum identifying the instruction family (one of 11 values)
- `std::vector<Qualifier> qualifiers`: full qualifier list from PTX source
- `std::vector<OperandContext> operands`: parsed operands
- `std::string instructionText`: original PTX text (for logging)
- `uint32_t cta_group = 1`: parsed `.cta_group` value
- `Tcgen05Dtype dtype = Tcgen05Dtype::F16`: parsed `.kind` value
- `uint32_t num_regs = 0`: parsed `x1/x2/x4` for ld/st
- `bool has_block_scale = false`: parsed `.block_scale` flag

The system MUST also provide:
- `enum class Tcgen05OpKind { ALLOC, DEALLOC, RELINQUISH, LD, ST, CP, MMA, MMA_WS, COMMIT, WAIT, FENCE }`
- `enum class Tcgen05Dtype { F16, BF16, TF32, F8, F4, MXF4, MXF8, I8, MXF4NVF4, INVALID }`

#### Scenario: tcgen05-instr-struct-has-all-fields
- **WHEN** `Tcgen05Instr` is instantiated with all fields set
- **THEN** all 8 fields are accessible and read/write correctly

#### Scenario: tcgen05-opkind-enum-has-11-values
- **WHEN** `Tcgen05OpKind` is enumerated
- **THEN** exactly 11 distinct enum values exist (ALLOC through FENCE)

#### Scenario: tcgen05-dtype-enum-has-10-values
- **WHEN** `Tcgen05Dtype` is enumerated
- **THEN** exactly 10 distinct enum values exist (F16 through INVALID)

#### Scenario: tcgen05-instr-no-wmma-coupling
- **WHEN** the Tcgen05Instr definition is read
- **THEN** zero references to `WmmaInstr` or `wmmaType` exist
- **AND** Tcgen05Instr can be compiled without `wmma.h` being included

### Requirement: IR MUST Provide ~25 Independent Qualifier Enums for tcgen05

The system SHALL provide approximately 25 new `Q_*` Qualifier enum values in `include/ptx_ir/ptx_qualifier.def` for tcgen05-specific qualifiers. The system MUST also DELETE the existing stub qualifiers `Q_TCGEN05_LD / Q_TCGEN05_ST / Q_TCGEN05_COMMIT / Q_TCGEN05_WAIT` (replaced by independent IR enums).

The new qualifiers include:
- Qualifier markers: `Q_CTA_GROUP / Q_KIND / Q_MULTICAST / Q_SEM / Q_PACK / Q_BLOCK_SCALE / Q_SCALE_VEC_SIZE_2X / Q_SCALE_VEC_SIZE_4X / Q_SP / Q_WS / Q_LOAD / Q_STORE / Q_BEFORE_THREAD_SYNC / Q_AFTER_THREAD_SYNC / Q_MBARRIER_ARRIVE_ONE / Q_SHARED_CTA / Q_SHARED_CLUSTER`
- dtype qualifiers: `Q_F16 / Q_BF16 / Q_TF32 / Q_F8 / Q_F4 / Q_MXF4 / Q_MXF8 / Q_I8 / Q_MXF4NVF4 / Q_F8F6F4`
- shape qualifiers: `Q_M64N8K16 / Q_M64N16K16 / Q_M64N32K16 / Q_M64N64K16 / Q_M64N128K16 / Q_M64N256K16`

#### Scenario: qualifier-def-has-25-new-entries
- **WHEN** the file `include/ptx_ir/ptx_qualifier.def` is read
- **THEN** approximately 25 new `X(Q_*, ..., ".token")` entries exist
- **AND** no `Q_TCGEN05_*` stub entries exist

#### Scenario: qualifier-token-strings-correct
- **WHEN** a Qualifier is converted to string via `Q2s(Q_CTA_GROUP)`
- **THEN** the result is `".cta_group"` (matching the literal text in ptx_qualifier.def)
- **AND** all 25+ new qualifiers have correct token strings

#### Scenario: qualifier-no-conflict-with-existing
- **WHEN** all existing tests are re-run after qualifier additions
- **THEN** no enum value conflicts occur
- **AND** all existing tests pass

### Requirement: IR MUST Provide StatementFactory Function for Tcgen05Instr

The system SHALL provide a `makeTcgen05Instr(...)` factory function in `include/ptx_ir/statement_factory.h` that creates a `StatementContext` with type `S_TCGEN05_*` and data `Tcgen05Instr`. The system MUST DELETE the existing `makeWmmaInstr(...)` factory function (per wmma cleanup).

#### Scenario: make-tcgen05-instr-creates-statement
- **WHEN** `makeTcgen05Instr(Tcgen05OpKind::MMA, quals, operands, "tcgen05.mma ...")` is called
- **THEN** the returned `StatementContext` has `type = S_TCGEN05_MMA`
- **AND** the data variant contains a `Tcgen05Instr` with the correct fields
- **AND** `instructionText` matches the input string

#### Scenario: make-wmma-instr-removed
- **WHEN** `makeWmmaInstr(...)` is called
- **THEN** the code does not compile (function does not exist)
- **AND** all references are removed from src/ include/ tests/

### Requirement: IR X-Macro Dispatch MUST Generate Tcgen05Handler Symbols

The system SHALL extend `src/ptxsim/instruction_handlers.cpp` with `IMPLEMENT_TCGEN05_INSTR_HANDLER(Name)` macro that generates `Tcgen05Handler::processTcgen05Operation(...)` weak symbols for all 11 `S_TCGEN05_*` enums. The system MUST DELETE the existing `IMPLEMENT_WMMA_INSTR_HANDLER` macro.

#### Scenario: weak-symbols-generated-per-instruction
- **WHEN** `src/ptxsim/instruction_handlers.cpp` is compiled
- **THEN** 11 `Tcgen05Handler::processTcgen05Operation` weak symbols are generated
- **AND** no `WmmaHandler::processWmmaOperation` weak symbols exist

#### Scenario: wmma-macro-removed-no-orphan-references
- **WHEN** `grep -rn "IMPLEMENT_WMMA_INSTR_HANDLER" src/ include/ tests/` is run
- **THEN** zero references exist
- **AND** any leftover references are fixed in this change

### Requirement: PipelineHandler MUST Route S_TCGEN05_* to Tcgen05PipelineHandler

The system SHALL provide a `Tcgen05PipelineHandler` class in `src/ptxsim/instruction_base.cpp` that mirrors `WmmaPipelineHandler` but routes `S_TCGEN05_*` statement types to `Tcgen05Handler::processTcgen05Operation`. The 3-phase pipeline (prepareOperands / executeOperation / commitResults) MUST reuse the same pattern.

#### Scenario: pipeline-routes-tcgen05-mma
- **WHEN** `ThreadContext::_execute_once()` processes a `S_TCGEN05_MMA` statement
- **THEN** `Tcgen05PipelineHandler::executeOperation()` is called
- **AND** `processTcgen05Operation(context, operands, qualifiers)` is invoked

#### Scenario: pipeline-routes-tcgen05-ld
- **WHEN** `ThreadContext::_execute_once()` processes a `S_TCGEN05_LD` statement
- **THEN** `Tcgen05PipelineHandler::executeOperation()` is called
- **AND** `processTcgen05Operation(context, operands, qualifiers)` is invoked

#### Scenario: pipeline-handles-all-11-families
- **WHEN** all 11 `S_TCGEN05_*` statement types are processed
- **THEN** all 11 are routed to `Tcgen05PipelineHandler`
- **AND** no `S_TCGEN05_*` statement falls through to generic handler
