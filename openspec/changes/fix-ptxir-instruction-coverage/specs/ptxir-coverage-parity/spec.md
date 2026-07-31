# ptxir-coverage-parity Specification

## Purpose
Defines reader/writer coverage parity across the full StatementType enum set: every enum the writer can serialize MUST be readable, and vice versa.

## MODIFIED Requirements

### Requirement: Reader MUST handle all 24 V1 instruction types
The `PtxirReader::read_instruction()` method SHALL contain a `case` for every V1-supported `StatementType` enum value (24 types), matching the set of types handled by `PtxirWriter::write_instruction()`.
target: ptxir-coverage-parity

#### Scenario: All 24 types have explicit case branches
- **WHEN** the reader's `switch (type)` statement is inspected
- **THEN** the switch MUST include a `case` for each of: S_BRA, S_LABEL, S_EXIT, S_RET, S_BAR, S_MOV, S_ADD, S_SUB, S_MUL, S_LD, S_ST, S_SETP, S_CVT, S_PRAGMA, S_DOLLOR, S_REG, S_CONST, S_SHARED, S_LOCAL, S_GLOBAL, S_PARAM, S_BAR_WARP_SYNC, plus the additional 15 types (S_MEMBAR, S_FENCE, S_REDUX_SYNC, S_MBARRIER, S_CALL, S_VOTE, S_SHFL, S_ATOM, S_TEXTURE, S_SURFACE, S_REDUCTION, S_PREFETCH, S_CP_ASYNC, S_ABI_DIRECTIVE, plus any PredicatePrefix variant)

#### Scenario: All 24 types roundtrip successfully
- **WHEN** a `vector<StatementContext>` containing one example of each of the 24 supported types is serialized to a `.ptxir` file and deserialized
- **THEN** the deserialized vector MUST have identical length, identical types, and identical data fields per statement (per `ptxir-roundtrip-testing` spec)

### Requirement: Reader MUST throw on unknown instruction type
The `PtxirReader::read_instruction()` `default` case in the `switch (type)` statement MUST throw a `std::runtime_error` with a message identifying the unknown `StatementType` value, rather than silently constructing a default `GenericInstr` and discarding data.
target: ptxir-coverage-parity

#### Scenario: Unknown opcode throws
- **WHEN** the reader encounters a `StatementType` value that is not in the supported set
- **THEN** `PtxirReader::read()` MUST throw `std::runtime_error` with message containing the numeric opcode value

#### Scenario: No silent data loss
- **WHEN** the reader's `default` case is reached with an unsupported opcode
- **THEN** the reader MUST throw an exception identifying the unknown opcode, NOT return partially-constructed statements

### Requirement: Per-instruction case MUST reconstruct correct variant type
For each supported instruction type, the reader's `case` branch MUST reconstruct the corresponding `InstrVariant` type (BranchInstr, LabelInstr, VoidInstr, etc.) with all fields populated from the binary stream.
target: ptxir-coverage-parity

#### Scenario: MembarInstr reconstructed from binary
- **WHEN** a `MembarInstr` with qualifiers=[.cta, .sys] is serialized and the binary is deserialized
- **THEN** the resulting `StatementContext::data` MUST be `std::variant` alternative of `MembarInstr` type (NOT `GenericInstr`), with `qualifiers` field containing the original [.cta, .sys] list

### Requirement: Reader and writer MUST have symmetric coverage across ALL 106 StatementType enums
The `PtxirReader::read_instruction()` switch SHALL cover every `StatementType` enum value that can be produced by the parser (all 106 enums in `ptx_op.def`), such that the set of enums the writer serializes is exactly the set the reader deserializes. No enum that appears in a kernel's `kernelStatements` MAY hit the reader's `default:` throw path.
target: ptxir-coverage-parity

#### Scenario: Every ptx_op.def enum has a reader case group
- **WHEN** the set of `case S_*` labels in `read_instruction()` is compared to the enum values defined in `include/ptx_ir/ptx_op.def`
- **THEN** every enum value in `ptx_op.def` MUST be covered by at least one `case` label in the reader

#### Scenario: Full-enum roundtrip test passes
- **WHEN** a test iterates all 106 enum values from `ptx_op.def`, constructs a representative `StatementContext` for each, and roundtrips through `serialize_to_string()` / `deserialize_from_string()`
- **THEN** the roundtrip MUST succeed for every enum (no exception)
- **AND** the deserialized statement type MUST match the original for every enum

#### Scenario: 15 new case branches parity with writer
- **WHEN** the writer's `write_instruction()` if-constexpr chain is compared to the reader's `read_instruction()` switch
- **THEN** every `InstrVariant` alternative the writer dispatches MUST have a corresponding reader case group (including `Tcgen05Instr`)
