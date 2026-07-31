# ptxir-format-compliance Specification

## Purpose
TBD - created by archiving change ptxir-format-compliance. Update Purpose after archive.
## Requirements
### Requirement: Writer MUST produce Section TOC per ADR-0023 Decision 1
The `PtxirWriter::write()` method SHALL write a complete Section TOC (Table of Contents) immediately after the 24-byte `PtxirHeader`, with one `PtxirSectionTOC` entry (6 bytes each: type:u8, reserved:u8, offset:u32) per section, and `PtxirHeader::section_count` MUST equal the actual number of TOC entries.
target: ptxir-format-compliance

#### Scenario: TOC entries present in output
- **WHEN** `PtxirWriter::write()` serializes a non-empty statement vector
- **THEN** the bytes at offset `sizeof(PtxirHeader) = 24` MUST contain `section_count` TOC entries, each 6 bytes
- **AND** the writer MUST advance the stream position past the TOC area before writing section data

#### Scenario: section_count matches TOC entries
- **WHEN** a `.ptxir` file is read
- **THEN** the value of `PtxirHeader::section_count` MUST equal the number of TOC entries present in the file

#### Scenario: REGDECL TOC entry references REGDECL section
- **WHEN** the REGDECL section is written
- **THEN** a TOC entry with `type=REGDECL=1` MUST exist with `offset` pointing to the start of the REGDECL section data

#### Scenario: KERNEL TOC entry references KERNEL section
- **WHEN** the KERNEL section is written
- **THEN** a TOC entry with `type=KERNEL=3` MUST exist with `offset` pointing to the start of the KERNEL section data

#### Scenario: STRING_TABLE TOC entry references STRING_TABLE section
- **WHEN** the STRING_TABLE section is written
- **THEN** a TOC entry with `type=STRING_TABLE=5` MUST exist with `offset` pointing to the start of the STRING_TABLE section data

### Requirement: Writer MUST backfill header offset/size fields
After writing all sections, the `PtxirWriter` SHALL backfill `PtxirHeader::string_table_offset` and `PtxirHeader::string_table_size` with the actual values.
target: ptxir-format-compliance

#### Scenario: string_table_offset backfilled
- **WHEN** the writer completes the STRING_TABLE section
- **THEN** the writer MUST seek back to header offset 12 and write the absolute file offset of the STRING_TABLE section start

#### Scenario: string_table_size backfilled
- **WHEN** the writer completes the STRING_TABLE section
- **THEN** the writer MUST seek back to header offset 16 and write the byte size of the STRING_TABLE section

#### Scenario: section_count backfilled
- **WHEN** the writer begins writing TOC entries
- **THEN** the writer MUST update `PtxirHeader::section_count` to match the actual number of sections written (typically 3: REGDECL, KERNEL, STRING_TABLE)

### Requirement: Writer MUST follow prescribed section order
The `PtxirWriter::write()` method SHALL write sections in the following order: (1) 24-byte header, (2) TOC entries, (3) REGDECL section, (4) KERNEL section, (5) STRING_TABLE section.
target: ptxir-format-compliance

#### Scenario: Section order verification
- **WHEN** a `.ptxir` file is parsed by an independent reader
- **THEN** the REGDECL section MUST appear before the KERNEL section, and the KERNEL section MUST appear before the STRING_TABLE section
- **AND** the file size MUST be exactly `header_size + (section_count * 6) + sum(section sizes)`

### Requirement: Reader MUST parse via TOC, not hardcoded offsets
The `PtxirReader` SHALL locate each section by reading its offset from the corresponding TOC entry, rather than using hardcoded offsets.
target: ptxir-format-compliance

#### Scenario: Reader uses TOC offsets
- **WHEN** `PtxirReader::read()` parses a `.ptxir` file
- **THEN** the reader MUST iterate the TOC entries and `seek_to(entry.offset)` before reading each section
- **AND** the reader MUST NOT assume any fixed offset for section data

### Requirement: TYPE and CONSTANT sections reserved for future use
The `PtxirSectionType` enum includes TYPE=2 and CONSTANT=4. The V1 writer MUST NOT write these sections, but V1 reader MUST handle them gracefully (skip or throw, per the design's intent).

#### Scenario: V1 reader encounters unknown section type
- **WHEN** a V2 (future) writer includes a TYPE or CONSTANT section and a V1 reader parses it
- **THEN** the V1 reader MUST throw `std::runtime_error` with message identifying the unsupported section type
- **AND** V2 readers will implement block-skip mechanism (out of scope for V1)

### Requirement: S_BAR instruction encoding MUST include reconvergence_pc in v3 format
Starting from PTXIR version 3, the `S_BAR` (BarrierInstr) instruction encoding SHALL include a `reconvergence_pc` field (int32_t) after the `barId` field, matching the existing encoding of `S_BRA` (BranchInstr).

#### Scenario: v3 S_BAR encoding includes reconvergence_pc
- **WHEN** `PtxirWriter::write_barrier()` serializes a `BarrierInstr` with `reconvergence_pc = 42`
- **THEN** the output bytes for the barrier instruction MUST be: `opcode(u16) | barId(i32) | reconvergence_pc(i32)`
- **AND** the total size of the barrier instruction data MUST be `sizeof(int32_t) * 2 = 8 bytes`

#### Scenario: v3 S_BAR reader extracts reconvergence_pc
- **WHEN** `PtxirReader::read_instruction()` reads a v3 PTXIR file with `S_BAR` instructions
- **THEN** the deserialized `BarrierInstr` MUST have `reconvergence_pc` set to the encoded value
- **AND** `reconvergence_pc` MUST equal the PC index of the post-dominator block

#### Scenario: v2 S_BAR reader skips reconvergence_pc (backward compatibility)
- **WHEN** `PtxirReader::read_instruction()` reads a v2 PTXIR file with `S_BAR` instructions
- **THEN** the deserialized `BarrierInstr` MUST have `reconvergence_pc = -1` (default sentinel)
- **AND** the reader MUST NOT attempt to read the extra `reconvergence_pc` field (file format compatibility)

### Requirement: PTXIR_VERSION MUST be bumped to 3
The `PTXIR_VERSION` constant in `ptxir_format.h` SHALL be updated from `2` to `3` to reflect the S_BAR encoding change.

#### Scenario: PTXIR_VERSION updated
- **WHEN** `ptxir_format.h` is inspected
- **THEN** `PTXIR_VERSION` MUST equal `3`
- **AND** `PtxirWriter::write_header()` MUST write `version = 3` to the output file header
- **AND** `PtxirReader::read_header()` MUST accept both `version = 2` and `version = 3`

### Requirement: BARRIER_ENCODED_SIZE MUST include reconvergence_pc
The `ptxir_encoding::BARRIER_ENCODED_SIZE` constant in `ptxir_format.h` SHALL be updated to include the `reconvergence_pc` field.

#### Scenario: BARRIER_ENCODED_SIZE updated
- **WHEN** `ptxir_format.h` is inspected
- **THEN** `BARRIER_ENCODED_SIZE` MUST equal `sizeof(uint16_t) + sizeof(int32_t) + sizeof(int32_t)`

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

