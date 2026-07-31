# embed-reconvergence-pc Specification

## Purpose
TBD - created by archiving change embed-reconvergence-pc-in-ptxir. Update Purpose after archive.
## Requirements
### Requirement: generate_ptxir() MUST embed reconvergence_pc via CFGBuilder
`generate_ptxir()` SHALL compute and populate `reconvergence_pc` for all `S_BRA` and `S_BAR` instructions in the kernel statements before serialization.

#### Scenario: S_BRA reconvergence_pc populated in generated PTXIR
- **WHEN** `generate_ptxir("input.ptx", "output.ptxir")` is called with a valid PTX file containing branching instructions
- **THEN** the serialized PTXIR file MUST contain non-default `reconvergence_pc` values for all `S_BRA` instructions
- **AND** `load_ptxir("output.ptxir", false)` MUST return statements with `reconvergence_pc` values matching those from `load_ptxir("output.ptxir", true)`

#### Scenario: S_BAR reconvergence_pc populated in generated PTXIR
- **WHEN** `generate_ptxir("input.ptx", "output.ptxir")` is called with a valid PTX file containing barrier instructions
- **THEN** the serialized PTXIR file MUST contain non-default `reconvergence_pc` values for all `S_BAR` instructions
- **AND** `load_ptxir("output.ptxir", false)` MUST return statements with `reconvergence_pc` values matching those from `load_ptxir("output.ptxir", true)`

### Requirement: load_ptxir(apply_cfg=false) MUST NOT recompute CFG
When loading a PTXIR v3 file (or later) with `apply_cfg=false`, the reader MUST use the `reconvergence_pc` values embedded in the binary, without calling `CFGBuilder::build()`.

#### Scenario: v3 load with apply_cfg=false skips CFG
- **WHEN** `load_ptxir("output.ptxir", false)` is called on a PTXIR v3 file
- **THEN** the function MUST NOT call `CFGBuilder::build()` or `CFGBuilder::computePostDominators()`
- **AND** the returned statements MUST have `reconvergence_pc` correctly populated from the binary

### Requirement: Old PTXIR v2 files MUST be loadable with apply_cfg=true fallback
PTXIR v2 files (without embedded `reconvergence_pc` for `S_BAR`) MUST be loadable via `load_ptxir(apply_cfg=true)`, which recomputes CFG to fill `reconvergence_pc`.

#### Scenario: v2 file load with apply_cfg=true
- **WHEN** `load_ptxir("v2_file.ptxir", true)` is called on a PTXIR v2 file
- **THEN** the function MUST call `CFGBuilder::build()` to populate `reconvergence_pc` for both `S_BRA` and `S_BAR` instructions
- **AND** the results MUST be equivalent to loading the same PTX source through `load_ptx_statements(ptx_path, "", true)`

#### Scenario: v2 file load with apply_cfg=false returns default reconvergence_pc
- **WHEN** `load_ptxir("v2_file.ptxir", false)` is called on a PTXIR v2 file
- **THEN** `S_BAR` instructions MUST have `reconvergence_pc = -1` (default/unitialized, same as current behavior)
- **AND** `S_BRA` instructions MUST have `reconvergence_pc` correctly populated from the binary (v2 format already includes it for S_BRA)

