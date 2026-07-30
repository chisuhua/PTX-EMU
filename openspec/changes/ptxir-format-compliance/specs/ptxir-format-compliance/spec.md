## ADDED Requirements

### Requirement: Writer MUST produce Section TOC per ADR-0023 Decision 1
The `PtxirWriter::write()` method SHALL write a complete Section TOC (Table of Contents) immediately after the 24-byte `PtxirHeader`, with one `PtxirSectionTOC` entry (6 bytes each: type:u8, reserved:u8, offset:u32) per section, and `PtxirHeader::section_count` MUST equal the actual number of TOC entries.

#### Scenario: TOC entries present in output
- **WHEN** `PtxirWriter::write()` serializes a non-empty statement vector
- **THEN** the bytes at offset `sizeof(PtxirHeader) = 24` MUST contain `section_count` TOC entries, each 6 bytes
- **AND** the writer MUST advance the stream position past the TOC area before writing section data

#### Scenario: section_count matches TOC entries
- **WHEN** a `.ptxir` file is read
- **THEN** the value of `PtxirHeader::section_count` MUST equal the number of TOC entries present in the file (readable from `(file_size - 24) / 6` only if all sections are 6 bytes, but generally MUST match by iteration)

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
After writing all sections, the `PtxirWriter` SHALL backfill `PtxirHeader::string_table_offset` and `PtxirHeader::string_table_size` with the actual values (currently both are always 0, breaking the format contract).

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
The `PtxirReader` SHALL locate each section by reading its offset from the corresponding TOC entry, rather than using hardcoded offsets (e.g., `sizeof(PtxirHeader)`).

#### Scenario: Reader uses TOC offsets
- **WHEN** `PtxirReader::read()` parses a `.ptxir` file
- **THEN** the reader MUST iterate the TOC entries and `seek_to(entry.offset)` before reading each section
- **AND** the reader MUST NOT assume any fixed offset for section data

#### Scenario: Reader rejects out-of-range TOC offsets
- **WHEN** a TOC entry's `offset` value exceeds the file size
- **THEN** `PtxirReader::read()` MUST throw `std::runtime_error` with message indicating the invalid offset

#### Scenario: Reader rejects duplicate TOC entries
- **WHEN** two TOC entries have the same `type` value
- **THEN** `PtxirReader::read()` MUST throw `std::runtime_error` indicating the duplicate section

### Requirement: TYPE and CONSTANT sections reserved for future use
The `PtxirSectionType` enum includes TYPE=2 and CONSTANT=4. The V1 writer MUST NOT write these sections, but V1 reader MUST handle them gracefully (skip or throw, per the design's intent).

#### Scenario: V1 reader encounters unknown section type
- **WHEN** a V2 (future) writer includes a TYPE or CONSTANT section and a V1 reader parses it
- **THEN** the V1 reader MUST throw `std::runtime_error` with message identifying the unsupported section type
- **AND** V2 readers will implement block-skip mechanism (out of scope for V1)
