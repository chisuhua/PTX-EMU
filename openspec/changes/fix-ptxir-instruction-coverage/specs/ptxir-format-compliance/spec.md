# ptxir-format-compliance Specification

## Purpose
PTXIR binary format specification for section layout, TOC, and instruction encoding.

## ADDED Requirements

### Requirement: Tcgen05Instr instruction encoding SHALL include qualifiers and operands

The `S_TCGEN05_*` instruction family (11 instruction types: ALLOC, DEALLOC, RELINQUISH, LD, ST, CP, MMA, MMA_WS, COMMIT, WAIT, FENCE) SHALL be encoded in the PTXIR binary format as: `opcode(u16) | qualifier_count(u8) | qualifiers[](u16 each) | operand_count(u8) | operand_ids[](u32 each)`, matching the encoding pattern of `MbarrierInstr`/`CallInstr`-style instructions.

#### Scenario: v3 tcgen05.mma encoding
- **WHEN** `PtxirWriter::write_tcgen05()` serializes a `Tcgen05Instr` of type `S_TCGEN05_MMA` with 2 qualifiers and 3 operands
- **THEN** the output bytes MUST be: `opcode(u16) | 2(u8) | qualifier1(u16) | qualifier2(u16) | 3(u8) | op1_id(u32) | op2_id(u32) | op3_id(u32)`

#### Scenario: tcgen05 instruction count in format constant
- **WHEN** the number of `S_TCGEN05_*` statement types is counted in `ptx_op.def`
- **THEN** there MUST be exactly 11 tcgen05 statement types, all serializable

### Requirement: Tcgen05Instr op_kind SHALL be derivable from statement type

For `S_TCGEN05_*` instructions, the reader SHALL derive `Tcgen05Instr::op_kind` from the statement type via a deterministic 1:1 mapping (S_TCGEN05_ALLOC→ALLOC, S_TCGEN05_DEALLOC→DEALLOC, S_TCGEN05_RELINQUISH→RELINQUISH, S_TCGEN05_LD→LD, S_TCGEN05_ST→ST, S_TCGEN05_CP→CP, S_TCGEN05_MMA→MMA, S_TCGEN05_MMA_WS→MMA_WS, S_TCGEN05_COMMIT→COMMIT, S_TCGEN05_WAIT→WAIT, S_TCGEN05_FENCE→FENCE).

#### Scenario: op_kind derived from type
- **WHEN** a `Tcgen05Instr` of type `S_TCGEN05_MMA_WS` is deserialized
- **THEN** `op_kind` MUST equal `Tcgen05OpKind::MMA_WS`
