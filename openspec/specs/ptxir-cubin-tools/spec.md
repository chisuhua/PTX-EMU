# ptxir-cubin-tools Specification

## Purpose
TBD - created by archiving change implement-ptxir-cubin-embed-extension. Update Purpose after archive.
## Requirements
### Requirement: ptxir_embed CLI tool contract (target = exe OR cubin)

The system SHALL provide `ptxir_embed` as a CLI tool that takes either an executable (`--in-exe`) or a cubin (`--in-cubin`) plus a PTXIR section and `--kernel-name` (required), producing an embedded payload (prefix + PTXIR section + `ptxir_section_size_le` + `PTXIR_EMBED_MAGIC`). The tool SHALL verify byte-level prefix preservation via SHA-256 hash stored in the embedded PTXIR section's `cubin_hash` field.

#### Scenario: Embed into executable (PTX-EMU loading path)

- **WHEN** the user invokes `ptxir_embed --in-exe myapp --in-ptxir kernel.ptxir --kernel-name vector_add --out myapp.embedded`
- **THEN** the system SHALL write `myapp.embedded` to disk
- **AND** the SHA-256 hash of `myapp` (read up to size-12) SHALL equal the `cubin_hash` field stored in the embedded PTXIR section
- **AND** the file SHALL end with `uint32_le ptxir_section_size` followed by `PTXIR_EMBED_MAGIC = {'P','T','X','E','M','B','\x01','\x00'}` (8 bytes)
- **AND** the tool SHALL exit with code 0

#### Scenario: Embed into cubin (NVIDIA-compat path)

- **WHEN** the user invokes `ptxir_embed --in-cubin kernel.cubin --in-ptxir kernel.ptxir --kernel-name vector_add --out kernel.embedded.cubin`
- **THEN** the system SHALL write `kernel.embedded.cubin` to disk
- **AND** the SHA-256 hash of `kernel.cubin` SHALL equal the `cubin_hash` field
- **AND** the file SHALL end with `ptxir_section_size_le` + `PTXIR_EMBED_MAGIC`
- **AND** the tool SHALL exit with code 0

#### Scenario: Missing --kernel-name (required flag)

- **WHEN** the user invokes `ptxir_embed` without `--kernel-name`
- **THEN** the tool SHALL print an error message to stderr indicating `--kernel-name is required`
- **AND** exit with non-zero code (4 for usage error)

#### Scenario: Missing input file

- **WHEN** `--in-exe` / `--in-cubin` / `--in-ptxir` references a non-existent file
- **THEN** the tool SHALL print an error message to stderr
- **AND** exit with non-zero code (2 for I/O error)

#### Scenario: --help output

- **WHEN** the user invokes `ptxir_embed --help`
- **THEN** the tool SHALL print usage instructions including all CLI flags (especially `--in-exe` vs `--in-cubin` mutual exclusion + `--kernel-name` required)
- **AND** exit with code 0

#### Scenario: --version output

- **WHEN** the user invokes `ptxir_embed --version`
- **THEN** the tool SHALL print the version string
- **AND** exit with code 0

### Requirement: ptxir_extract CLI tool contract

The system SHALL provide `ptxir_extract` as a CLI tool that takes an embedded binary (exe or cubin) and produces a pure prefix and/or a PTXIR section. The tool SHALL verify byte-level equality of the extracted prefix with the original via SHA-256 hash comparison against the embedded `cubin_hash` field.

#### Scenario: Successful dual extraction

- **WHEN** the user invokes `ptxir_extract --in kernel.embedded.cubin --out-cubin kernel.pure.cubin --out-ptxir kernel.pure.ptxir`
- **THEN** the system SHALL write `kernel.pure.cubin` and `kernel.pure.ptxir` to disk
- **AND** the SHA-256 hash of `kernel.pure.cubin` SHALL equal the `cubin_hash` field stored in the embedded PTXIR section
- **AND** the tool SHALL exit with code 0

#### Scenario: Extraction of plain binary (passthrough)

- **WHEN** the input binary does not contain `PTXIR_EMBED_MAGIC` (e.g., plain cubin or unembedded executable)
- **THEN** `--out-cubin` SHALL receive a byte-identical copy of the input
- **AND** `--out-ptxir` SHALL NOT be written (or written empty, per CLI choice)

#### Scenario: Hash mismatch rejection

- **WHEN** the extracted prefix SHA-256 does NOT match the embedded `cubin_hash`
- **THEN** the tool SHALL print an error message to stderr
- **AND** exit with non-zero code (3 for integrity error)
- **AND** the output file SHALL NOT be written

#### Scenario: --help and --version

- **WHEN** the user invokes `ptxir_extract --help` or `ptxir_extract --version`
- **THEN** the tool SHALL print the appropriate output
- **AND** exit with code 0

### Requirement: Footer layout (zip-EOCD style) integrity

The system SHALL produce and consume binaries using the footer layout: `prefix[N] || ptxir_section[M] || uint32_le ptxir_section_size || PTXIR_EMBED_MAGIC[8]`. The `ptxir_section_size` field SHALL appear AFTER the section (not before), enabling O(1) locator algorithm without ELF/cubin parsing.

#### Scenario: Footer parsing from embedded binary

- **WHEN** `PTXIRLoader::hasEmbeddedPTXIR(data, size)` is invoked on a binary with the footer layout
- **THEN** the system SHALL locate the section by reading `data[size-12..size-8]` as `uint32_le ptxir_section_size`
- **AND** verify `size >= 12 + ptxir_section_size` (security: prevent OOB read)

#### Scenario: Footer construction by ptxir_embed

- **WHEN** `ptxir_embed` constructs the output binary
- **THEN** the output SHALL be `prefix || section || htole32(section.size()) || PTXIR_EMBED_MAGIC`
- **AND** `htole32` ensures little-endian encoding on all platforms

### Requirement: Byte-level prefix preservation guarantee

The system SHALL guarantee that the prefix of an embedded binary, when extracted by `ptxir_extract` (or directly read by NVIDIA tools via `cuobjdump` for cubin target), is byte-identical to the original input passed to `ptxir_embed`.

#### Scenario: NVIDIA tool compatibility (cubin target)

- **WHEN** `cuobjdump --dump-sass kernel.embedded.cubin` is invoked directly on an embedded cubin
- **THEN** the system SHALL output SASS disassembled from the cubin prefix
- **AND** the disassembled SASS SHALL match the output of `cuobjdump --dump-sass kernel.cubin` (the original)
- **AND** the tool SHALL exit with code 0

#### Scenario: cuModuleLoadData compatibility (when driver available)

- **WHEN** `cuModuleLoadData(kernel.embedded.cubin)` is invoked in an environment with a real NVIDIA driver
- **THEN** the system SHALL return `CUDA_SUCCESS`
- **AND** SHALL NOT report errors related to the trailing magic bytes

#### Scenario: Test SKIP for unavailable driver

- **WHEN** e2e tests verify `cuModuleLoadData(kernel.embedded.cubin)` compatibility
- **AND** the test environment has no real NVIDIA driver
- **THEN** the test SHALL print `[SKIP] cuModuleLoadData test — no driver` to stdout/stderr
- **AND** the test SHALL NOT silently pass (Oracle review blocking fix)

#### Scenario: ELF trailing overlay tolerance (exe target)

- **WHEN** `myapp.embedded` is loaded by the standard Linux ELF loader (via PTX-EMU's `LD_PRELOAD=./libcudart.so`)
- **THEN** the loader SHALL successfully map the executable segments
- **AND** SHALL ignore the trailing PTXIR section + magic (out-of-segment data)
- **NOTE**: ELF format tolerates trailing data after the last LOAD segment; verified by ELF specification

