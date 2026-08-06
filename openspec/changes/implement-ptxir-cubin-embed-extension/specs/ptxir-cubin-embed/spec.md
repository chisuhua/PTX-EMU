# Spec: ptxir-cubin-embed

## ADDED Requirements

### Requirement: PTXIR-Embedded CUBIN binary format

The system SHALL define an append-only PTXIR-Embedded CUBIN format that preserves the original cubin prefix verbatim and appends a PTXIR section + 8-byte `PTXIR_EMBED_MAGIC` suffix. The format SHALL reuse the ADR-0023 Section TOC and PTXIRHeader structures with an additional `cubin_hash` field.

#### Scenario: Embedded cubin byte layout

- **WHEN** a cubin is processed by `ptxir_embed`
- **THEN** the output is `cubin_bytes || ptxir_section_bytes || PTXIR_EMBED_MAGIC_8bytes`
- **AND** the original cubin prefix is byte-identical to the input cubin (verified by `cubin_hash` SHA-256 match)

#### Scenario: Magic suffix detection

- **WHEN** `PTXIRLoader::hasEmbeddedPTXIR(cubin_bytes, cubin_size)` is invoked
- **THEN** the system SHALL check the last 8 bytes of `cubin_bytes` against `PTXIR_EMBED_MAGIC`
- **AND** return `true` if the bytes match exactly
- **AND** return `false` if `cubin_size < 8` or any byte differs

#### Scenario: Magic governance check

- **WHEN** a developer proposes changing the literal value of `PTXIR_EMBED_MAGIC`
- **THEN** the change MUST trigger a review of ADR-0024 (governance check)
- **AND** MUST NOT be merged via a single OpenSpec change without ADR-0024 update

### Requirement: PTXIRLoader API contract

The system SHALL expose `PTXIRLoader` as a stateless class with all methods `public static`. The loader SHALL provide four methods for embedded-cubin detection, extraction, and deserialization.

#### Scenario: hasEmbeddedPTXIR with valid embedded cubin

- **WHEN** `hasEmbeddedPTXIR(embedded_bytes, embedded_size)` is called with a valid embedded cubin
- **THEN** the system SHALL return `true`

#### Scenario: hasEmbeddedPTXIR with plain cubin

- **WHEN** `hasEmbeddedPTXIR(plain_cubin_bytes, plain_cubin_size)` is called with a cubin that does not contain `PTXIR_EMBED_MAGIC`
- **THEN** the system SHALL return `false`

#### Scenario: hasEmbeddedPTXIR with truncated input

- **WHEN** `hasEmbeddedPTXIR(some_bytes, size_less_than_8)` is called
- **THEN** the system SHALL return `false` (not throw, not crash)

#### Scenario: extractPTXIR returns null for non-embedded cubin

- **WHEN** `extractPTXIR(plain_cubin_bytes, plain_cubin_size)` is called
- **THEN** the system SHALL return `std::optional<PTXIRSection>` containing `nullopt`

#### Scenario: extractPureCubin returns hash-mismatch failure

- **WHEN** `extractPureCubin(embedded_bytes, embedded_size)` extracts the cubin prefix
- **AND** the SHA-256 hash of the extracted prefix does NOT match the `cubin_hash` field in the Section TOC
- **THEN** the system SHALL return `std::optional<std::vector<uint8_t>>` containing `nullopt`

#### Scenario: deserializeForCubin graceful degradation

- **WHEN** `deserializeForCubin(ptxir_section_bytes)` is called with a corrupted PTXIRHeader
- **THEN** the system SHALL return `std::vector<StatementContext>` that is empty (length 0)
- **AND** SHALL NOT throw an exception

### Requirement: config::isPTXIRModeEnabled dispatch control

The system SHALL provide `config::isPTXIRModeEnabled()` as the single source of truth for `PTXIR_MODE` dispatch decisions. The function MUST read from both the `PTXIR_MODE` environment variable and `configs/*.ini` files (two-source precedence documented per project convention).

#### Scenario: PTXIR_MODE=off completely bypasses detection

- **WHEN** `config::isPTXIRModeEnabled()` returns `false`
- **THEN** `__cudaRegisterFatBinary` SHALL skip the entire PTXIR detection branch
- **AND** SHALL behave byte-identically to the pre-change implementation
- **AND** the function call overhead SHALL be O(1) (single env var read cached in static)

#### Scenario: PTXIR_MODE=auto enables detection

- **WHEN** `config::isPTXIRModeEnabled()` returns `true`
- **THEN** `__cudaRegisterFatBinary` SHALL invoke `PTXIRLoader::hasEmbeddedPTXIR()`
- **AND** if true, SHALL invoke the extraction + deserialization pipeline
- **AND** on any pipeline failure, SHALL fall back to the standard cubin path

#### Scenario: Default behavior preserves compatibility

- **WHEN** `PTXIR_MODE` is not set in env var or INI
- **THEN** `config::isPTXIRModeEnabled()` SHALL return `false`
- **AND** the system SHALL behave byte-identically to the pre-change implementation

### Requirement: __cudaRegisterFatBinary ABI stability

The system SHALL NOT modify the public signature of `__cudaRegisterFatBinary(void* fatbin)`. The function SHALL only add an internal dispatch branch gated by `config::isPTXIRModeEnabled()`.

#### Scenario: Existing ABI consumers unaffected

- **WHEN** an existing CUDA program calls `__cudaRegisterFatBinary` without setting `PTXIR_MODE`
- **THEN** the function SHALL exhibit identical runtime behavior to the pre-change implementation
- **AND** SHALL NOT allocate new memory beyond the pre-change baseline
- **AND** SHALL NOT call any new function pointers that could break dlsym consumers

#### Scenario: PTXIR detection branch does not modify existing call paths

- **WHEN** `__cudaRegisterFatBinary` invokes the PTXIR detection branch
- **AND** the detection returns `false` or any extraction fails
- **THEN** the control flow SHALL transfer to the existing standard-cubin path
- **AND** NO side effects SHALL leak from the PTXIR branch (no global state mutation, no log spam)