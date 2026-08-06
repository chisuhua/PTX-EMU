# Spec: ptxir-cubin-tools

## ADDED Requirements

### Requirement: ptxir_embed CLI tool contract

The system SHALL provide `ptxir_embed` as a CLI tool that takes a cubin and a PTXIR section and produces an embedded cubin (cubin prefix + PTXIR section + magic suffix). The tool SHALL verify byte-level cubin prefix preservation.

#### Scenario: Successful embedding

- **WHEN** the user invokes `ptxir_embed --in-cubin kernel.cubin --in-ptxir kernel.ptxir --out kernel.embedded.cubin`
- **THEN** the system SHALL write `kernel.embedded.cubin` to disk
- **AND** the SHA-256 hash of `kernel.cubin` SHALL equal the `cubin_hash` field stored in the embedded PTXIR section
- **AND** the file SHALL end with the literal bytes of `PTXIR_EMBED_MAGIC`
- **AND** the tool SHALL exit with code 0

#### Scenario: Missing input file

- **WHEN** `--in-cubin` or `--in-ptxir` references a non-existent file
- **THEN** the tool SHALL print an error message to stderr
- **AND** exit with non-zero code (e.g., 2 for I/O error)

#### Scenario: --help output

- **WHEN** the user invokes `ptxir_embed --help`
- **THEN** the tool SHALL print usage instructions including all CLI flags
- **AND** exit with code 0

#### Scenario: --version output

- **WHEN** the user invokes `ptxir_embed --version`
- **THEN** the tool SHALL print the version string
- **AND** exit with code 0

### Requirement: ptxir_extract CLI tool contract

The system SHALL provide `ptxir_extract` as a CLI tool that takes an embedded cubin and produces a pure cubin and/or a PTXIR section. The tool SHALL verify byte-level equality of the extracted cubin with the original.

#### Scenario: Successful dual extraction

- **WHEN** the user invokes `ptxir_extract --in kernel.embedded.cubin --out-cubin kernel.pure.cubin --out-ptxir kernel.pure.ptxir`
- **THEN** the system SHALL write `kernel.pure.cubin` and `kernel.pure.ptxir` to disk
- **AND** the SHA-256 hash of `kernel.pure.cubin` SHALL equal the `cubin_hash` field stored in the embedded PTXIR section
- **AND** the tool SHALL exit with code 0

#### Scenario: Extraction of plain cubin (passthrough)

- **WHEN** the input cubin does not contain `PTXIR_EMBED_MAGIC`
- **THEN** `--out-cubin` SHALL receive a byte-identical copy of the input
- **AND** `--out-ptxir` SHALL NOT be written (or written empty, per CLI choice)

#### Scenario: Hash mismatch rejection

- **WHEN** the extracted cubin prefix SHA-256 does NOT match the embedded `cubin_hash`
- **THEN** the tool SHALL print an error message to stderr
- **AND** exit with non-zero code (e.g., 3 for integrity error)
- **AND** the output file SHALL NOT be written

#### Scenario: --help and --version

- **WHEN** the user invokes `ptxir_extract --help` or `ptxir_extract --version`
- **THEN** the tool SHALL print the appropriate output
- **AND** exit with code 0

### Requirement: Byte-level cubin preservation guarantee

The system SHALL guarantee that the cubin prefix of an embedded cubin, when extracted by `ptxir_extract` (or directly read by NVIDIA tools via `cuobjdump`), is byte-identical to the original input cubin passed to `ptxir_embed`.

#### Scenario: NVIDIA tool compatibility

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