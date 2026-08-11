# Spec: multi-kernel-manifest

## ADDED Requirements

### Requirement: `ManifestSection` 支持 `vector<kernel_entry>` 多 entry 存储

The system SHALL extend `include/ptx_ir/ptxir_format.h::ManifestSection` from single `kernel_name` to `vector<kernel_entry>`, where each `kernel_entry` contains at minimum the kernel symbol name and parameters. The schema extension MUST follow the ADR-0023 §决策 6 Extend-Only rule: bump `PTXIR_VERSION` before changing the schema; old readers MUST remain able to load v1 single-kernel binaries.

#### Scenario: v1 single-kernel binary loads on new runtime

- **WHEN** a v1 single-kernel binary (containing `ManifestSection.kernel_name` single value, no `vector`) is loaded on the new runtime
- **THEN** the reader treats the single entry as `vector` of length 1; behavior is byte-identical to v1

#### Scenario: multi-entry binary fixture loads and exposes 3 kernels

- **WHEN** a PTXIR image contains 3 `.entry` symbols (kernel A, B, C) is loaded
- **THEN** the reader parses all 3 entries; `ModuleRecord` exposes 3 distinct `kernel_entry` records; `cuModuleGetFunction` can resolve each by name

#### Scenario: PTXIR_VERSION bumped per Extend-Only

- **WHEN** this change ships
- **THEN** `PTXIR_VERSION` is bumped by 1; old v1 reader returns "unknown version, skipping" but still loads the v1 binary's manifest section

### Requirement: 旧 reader 跳过未知 section

Per ADR-0023 §决策 6 Extend-Only: any old reader that encounters an unknown section (e.g., new `kernels[]` vector section) MUST skip it gracefully without error.

#### Scenario: v1 reader encounters v2 binary with new `kernels[]` section

- **WHEN** a v1 reader processes a v2 binary containing a new section not defined in v1 schema
- **THEN** the reader skips the unknown section without raising an error

### Requirement: tools (ptxir_build / ptxir_embed / ptxir_extract) 支持多 kernel

The system SHALL update `tools/ptxir_build.cpp`, `tools/ptxir_embed`, `tools/ptxir_extract` to handle multi-entry PTXIR binaries.

#### Scenario: ptxir_build emits multi-entry PTXIR

- **WHEN** `ptxir_build --out foo.ptxir foo.ptx` where foo.ptx has 3 `.entry` symbols
- **THEN** the output PTXIR contains all 3 entries in `kernels[]`; roundtrip via `ptxir_extract` recovers all 3 names

#### Scenario: ptxir_extract recovers all kernel names

- **WHEN** `ptxir_extract foo.ptxir` runs on a multi-entry binary
- **THEN** output lists all kernel names with their parameters