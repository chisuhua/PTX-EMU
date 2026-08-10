# Spec: ptxir-cubin-cleanup

## ADDED Requirements

### Requirement: `__cudaRegisterFatBinary` PTXIR dispatch 分支

The system SHALL, when `config::isPTXIRModeEnabled()` returns `true` AND `PTXIRLoader::hasEmbeddedPTXIR(data, size)` returns `true`, dispatch the embedded PTXIR image via the PTXIRLoader + PtxContextAdapter pipeline, instead of falling back to the cuobjdump path.

#### Scenario: Embedded binary + PTXIR_MODE=auto → PTXIR dispatch

- **WHEN** `__cudaRegisterFatBinary` is called with a valid embedded binary (prefix + PTXIR section + size + magic footer) AND `PTXIR_MODE=auto`
- **THEN** the system SHALL extract PTXIR via `PTXIRLoader::extractPTXIR()`, deserialize via `PTXIRLoader::deserializeForCubin()`, and adapt via `PtxContextAdapter::fromEmbedded()`
- **AND** the resulting `PtxContext` SHALL be registered with the GPU
- **AND** `cudaLaunchKernel` SHALL execute the kernel via PTXIR path

#### Scenario: Embedded binary + PTXIR_MODE=off → cuobjdump path

- **WHEN** `__cudaRegisterFatBinary` is called AND `PTXIR_MODE=off`
- **THEN** the system SHALL skip the PTXIR detection branch entirely
- **AND** behavior SHALL be byte-identical to pre-change baseline

#### Scenario: Plain binary + PTXIR_MODE=auto → no error

- **WHEN** `__cudaRegisterFatBinary` is called with a plain binary (no magic footer) AND `PTXIR_MODE=auto`
- **THEN** the system SHALL detect `hasEmbeddedPTXIR() == false`
- **AND** SHALL fall back to cuobjdump path without error

#### Scenario: Malformed embedded PTXIR → error (NOT silent fallback)

- **WHEN** `__cudaRegisterFatBinary` is called with a binary that has the magic footer but malformed PTXIR section AND `PTXIR_MODE=auto`
- **THEN** the system SHALL report an error
- **AND** SHALL NOT silently fall back to cuobjdump path

#### Scenario: Manifest mismatch → error

- **WHEN** the `cubin_hash` in the PTXIR manifest does not match the SHA-256 of the cubin prefix
- **THEN** the system SHALL report an error

### Requirement: INI `[ptxir] mode` 段集成

The system SHALL read `[ptxir] mode` from the INI config file during `initialize_environment()`, and pass it to `config::setPTXIRModeFromIni(bool)`. The env var `PTXIR_MODE` overrides the INI value.

#### Scenario: env auto overrides INI off

- **WHEN** `PTXIR_MODE=auto` is set AND INI `[ptxir] mode = off`
- **THEN** `config::isPTXIRModeEnabled()` returns `true`

#### Scenario: env unset, INI on

- **WHEN** `PTXIR_MODE` is unset AND INI `[ptxir] mode = on`
- **THEN** `config::isPTXIRModeEnabled()` returns `true`

#### Scenario: env unset, INI unset (default off)

- **WHEN** `PTXIR_MODE` is unset AND INI has no `[ptxir]` section
- **THEN** `config::isPTXIRModeEnabled()` returns `false` (byte-identical to pre-change default)
