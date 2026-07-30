## ADDED Requirements

### Requirement: Reader MUST handle all 24 V1 instruction types
The `PtxirReader::read_instruction()` method SHALL contain a `case` for every V1-supported `StatementType` enum value (24 types), matching the set of types handled by `PtxirWriter::write_instruction()`.

#### Scenario: All 24 types have explicit case branches
- **WHEN** the reader's `switch (type)` statement is inspected
- **THEN** the switch MUST include a `case` for each of: S_BRA, S_LABEL, S_EXIT, S_RET, S_BAR, S_MOV, S_ADD, S_SUB, S_MUL, S_LD, S_ST, S_SETP, S_CVT, S_PRAGMA, S_DOLLOR, S_REG, S_CONST, S_SHARED, S_LOCAL, S_GLOBAL, S_PARAM, S_BAR_WARP_SYNC, plus the additional 15 types (S_MEMBAR, S_FENCE, S_REDUX_SYNC, S_MBARRIER, S_CALL, S_VOTE, S_SHFL, S_ATOM, S_TEXTURE, S_SURFACE, S_REDUCTION, S_PREFETCH, S_CP_ASYNC, S_ABI_DIRECTIVE, plus any PredicatePrefix variant)

#### Scenario: All 24 types roundtrip successfully
- **WHEN** a `vector<StatementContext>` containing one example of each of the 24 supported types is serialized to a `.ptxir` file and deserialized
- **THEN** the deserialized vector MUST have identical length, identical types, and identical data fields per statement (per `ptxir-roundtrip-testing` spec)

### Requirement: Reader MUST throw on unknown instruction type
The `PtxirReader::read_instruction()` `default` case in the `switch (type)` statement MUST throw a `std::runtime_error` with a message identifying the unknown `StatementType` value, rather than silently constructing a default `GenericInstr` and discarding data.

#### Scenario: Unknown opcode throws
- **WHEN** the reader encounters a `StatementType` value that is not in the V1 supported set
- **THEN** `PtxirReader::read()` MUST throw `std::runtime_error` with message containing the numeric opcode value and "unknown"

#### Scenario: No silent data loss
- **WHEN** the reader's `default` case was previously invoked (silent skip behavior)
- **AND** a `.ptxir` file written by a future V2 writer is read by a V1 reader
- **THEN** the V1 reader MUST throw an exception identifying the unknown opcode, NOT return partially-constructed statements

### Requirement: Per-instruction case MUST reconstruct correct variant type
For each of the 24 V1 instruction types, the reader's `case` branch MUST reconstruct the corresponding `InstrVariant` type (BranchInstr, LabelInstr, VoidInstr, etc.) with all fields populated from the binary stream.

#### Scenario: MembarInstr reconstructed from binary
- **WHEN** a `MembarInstr` with qualifiers=[.cta, .sys] is serialized and the binary is deserialized
- **THEN** the resulting `StatementContext::data` MUST be `std::variant` alternative of `MembarInstr` type (NOT `GenericInstr`), with `qualifiers` field containing the original [.cta, .sys] list

#### Scenario: AtomInstr reconstructed from binary
- **WHEN** an `AtomInstr` of type S_ATOM with qualifiers=[.global, .add] and 3 operands is serialized and deserialized
- **THEN** the resulting `StatementContext::data` MUST be `std::variant` alternative of `AtomInstr` type, with `qualifiers` and `operands` fields matching the original

#### Scenario: 15 new case branches parity with writer
- **WHEN** the writer's `write_instruction()` switch-case is compared to the reader's `read_instruction()` switch-case
- **THEN** every `case` in the writer MUST have a corresponding `case` in the reader (no asymmetric coverage)
