# ptxir-full-enum-coverage Specification

## Purpose
Defines full StatementType enum coverage in the PTXIR reader/writer: all 106 supported instruction enums (including 11 tcgen05 instructions) must roundtrip through the binary format without exceptions or data loss.

## ADDED Requirements

### Requirement: Reader MUST handle all 53 GENERIC_INSTR enum values

The `PtxirReader::read_instruction()` GENERIC_INSTR case group SHALL cover all 53 `GENERIC_INSTR`-kind `StatementType` enum values (S_LD, S_ST, S_MOV, S_CVT, S_CVTA, S_PRMT, S_ISSPACEP, S_MAPA, S_ALLOCA, S_ADD, S_SUB, S_MUL, S_MUL24, S_DIV, S_REM, S_MIN, S_MAX, S_NEG, S_ABS, S_MAD, S_MAD24, S_FMA, S_ADDC, S_SUBC, S_SAD, S_COPYSIGN, S_TESTP, S_TANH, S_AND, S_OR, S_XOR, S_NOT, S_SHL, S_SHR, S_SHF, S_BFE, S_LOP3, S_SETP, S_SET, S_SELP, S_SLCT, S_CNOT, S_SIN, S_COS, S_LG2, S_EX2, S_RCP, S_RSQRT, S_SQRT, S_POPC, S_CLZ, S_ACTIVEMASK, S_ST_BULK), reconstructing a `GenericInstr` for each — matching the writer's `write_generic()` encoding.

#### Scenario: cvta roundtrips without exception
- **WHEN** a `GenericInstr` of type `S_CVTA` is serialized to a `.ptxir` file and deserialized
- **THEN** the reader MUST NOT throw `Unknown StatementType`
- **AND** the deserialized statement MUST have type `S_CVTA` and a `GenericInstr` payload

#### Scenario: fma roundtrips without exception
- **WHEN** a `GenericInstr` of type `S_FMA` with qualifiers and operands is serialized and deserialized
- **THEN** the deserialized statement MUST have type `S_FMA` with matching qualifiers

#### Scenario: Real kernel with cvta/fma/div loads successfully
- **WHEN** `generate_ptxir("kernel.ptx", "out.ptxir")` is called on a real kernel containing `cvta`, `fma`, and `div` instructions, followed by `load_ptxir("out.ptxir", false)`
- **THEN** the function MUST NOT throw `Unknown StatementType`
- **AND** the returned statements MUST be non-empty and preserve instruction types

### Requirement: Reader MUST handle S_BRX as BranchInstr

The `PtxirReader::read_instruction()` SHALL handle `S_BRX` in the S_BRA case group, reconstructing a `BranchInstr`.

#### Scenario: brx roundtrips
- **WHEN** a `BranchInstr` of type `S_BRX` is serialized and deserialized
- **THEN** the reader MUST NOT throw
- **AND** the deserialized statement MUST have type `S_BRX` with a `BranchInstr` payload

### Requirement: Reader MUST handle S_TRAP, S_BRK, S_BRKPT as VoidInstr

The `PtxirReader::read_instruction()` SHALL handle `S_TRAP`, `S_BRK`, and `S_BRKPT` in the S_EXIT/S_RET case group, reconstructing a `VoidInstr`.

#### Scenario: trap/brk/brkpt roundtrip
- **WHEN** `VoidInstr` statements of types `S_TRAP`, `S_BRK`, and `S_BRKPT` are serialized and deserialized
- **THEN** the reader MUST NOT throw
- **AND** each deserialized statement MUST preserve its original type with a `VoidInstr` payload

### Requirement: Tcgen05Instr MUST be serialized by the writer

The `PtxirWriter::write_instruction()` SHALL dispatch `Tcgen05Instr` (all 11 `S_TCGEN05_*` statement types) to a `write_tcgen05()` method that serializes `qualifiers` and `operands`, matching the encoding pattern of similar instruction types (qualifiers + operand count + operand IDs).

#### Scenario: tcgen05.mma serializes without data loss
- **WHEN** a `Tcgen05Instr` of type `S_TCGEN05_MMA` with qualifiers and operands is serialized
- **THEN** the writer MUST emit qualifier count + qualifiers + operand count + operand IDs
- **AND** the output MUST NOT silently drop the instruction (no-op dispatch)

### Requirement: Reader MUST reconstruct Tcgen05Instr for all 11 S_TCGEN05_* types

The `PtxirReader::read_instruction()` SHALL handle all 11 `S_TCGEN05_*` enum values (ALLOC, DEALLOC, RELINQUISH, LD, ST, CP, MMA, MMA_WS, COMMIT, WAIT, FENCE), reconstructing a `Tcgen05Instr` with `op_kind` derived from the statement type (1:1 mapping) and qualifiers/operands read from the stream.

#### Scenario: tcgen05.mma roundtrips
- **WHEN** a `Tcgen05Instr` of type `S_TCGEN05_MMA` with qualifiers is serialized and deserialized
- **THEN** the deserialized statement MUST have type `S_TCGEN05_MMA`
- **AND** `op_kind` MUST equal `Tcgen05OpKind::MMA`
- **AND** qualifiers MUST match the original

#### Scenario: tcgen05.mma.ws op_kind derivation
- **WHEN** a `Tcgen05Instr` of type `S_TCGEN05_MMA_WS` is serialized and deserialized
- **THEN** `op_kind` MUST equal `Tcgen05OpKind::MMA_WS`

#### Scenario: tcgen05.alloc roundtrips
- **WHEN** a `Tcgen05Instr` of type `S_TCGEN05_ALLOC` is serialized and deserialized
- **THEN** the deserialized statement MUST have type `S_TCGEN05_ALLOC` with `op_kind == Tcgen05OpKind::ALLOC`
