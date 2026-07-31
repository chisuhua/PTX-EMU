# ptxir-format-compliance Specification

## Purpose
PTXIR binary format specification for section layout, TOC, and instruction encoding.

## MODIFIED Requirements

### Requirement: Writer MUST produce Section TOC per ADR-0023 Decision 1
The `PtxirWriter::write()` method SHALL write a complete Section TOC (Table of Contents) immediately after the 24-byte `PtxirHeader`, with one `PtxirSectionTOC` entry (6 bytes each: type:u8, reserved:u8, offset:u32) per section, and `PtxirHeader::section_count` MUST equal the actual number of TOC entries.

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

#### Scenario: Section order verification
- **WHEN** a `.ptxir` file is parsed by an independent reader
- **THEN** the REGDECL section MUST appear before the KERNEL section, and the KERNEL section MUST appear before the STRING_TABLE section
- **AND** the file size MUST be exactly `header_size + (section_count * 6) + sum(section sizes)`

### Requirement: Reader MUST parse via TOC, not hardcoded offsets
The `PtxirReader` SHALL locate each section by reading its offset from the corresponding TOC entry, rather than using hardcoded offsets.

#### Scenario: Reader uses TOC offsets
- **WHEN** `PtxirReader::read()` parses a `.ptxir` file
- **THEN** the reader MUST iterate the TOC entries and `seek_to(entry.offset)` before reading each section
- **AND** the reader MUST NOT assume any fixed offset for section data

### ADDED Requirements

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