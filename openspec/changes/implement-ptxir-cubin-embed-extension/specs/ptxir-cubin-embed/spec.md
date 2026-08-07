# Spec: ptxir-cubin-embed

## ADDED Requirements

### Requirement: PTXIR-Embedded CUBIN/EXE binary format (footer layout v1.1)

The system SHALL define an append-only PTXIR-Embedded binary format with footer-layout (ZIP-EOCD style). The format SHALL preserve the original cubin/exe prefix verbatim and append a PTXIR section, a 4-byte `ptxir_section_size_le` (little-endian), and an 8-byte `PTXIR_EMBED_MAGIC` suffix. The format SHALL reuse the ADR-0023 Section TOC and PTXIRHeader structures with an additional `PtxirSectionType::MANIFEST = 6` section (Extend-Only compatible per ADR-0023 Decision 6).

#### Scenario: Embedded binary byte layout

- **WHEN** a binary (cubin or final executable) is processed by `ptxir_embed`
- **THEN** the output is `prefix_bytes || ptxir_section_bytes || uint32_le ptxir_section_size || PTXIR_EMBED_MAGIC_8bytes`
- **AND** `ptxir_section_bytes` contains PTXIRHeader + Section TOC + standard sections (REGDECL/TYPE/KERNEL/CONSTANT/STRING_TABLE) + a new MANIFEST section (type=6) holding `cubin_hash[32]`, `kernel_name[]`, `ptx_address_size`, `params[]`
- **AND** the original prefix is byte-identical to the input (verified by `cubin_hash` SHA-256 match against MANIFEST section)
- **AND** `PTXIR_EMBED_MAGIC` equals `{'P','T','X','E','M','B','\x01','\x00'}` (8 bytes)

#### Scenario: O(1) magic suffix detection

- **WHEN** `PTXIRLoader::hasEmbeddedPTXIR(data, size)` is invoked
- **THEN** the system SHALL check the last 8 bytes of `data` against `PTXIR_EMBED_MAGIC`
- **AND** return `true` if the bytes match exactly AND `size >= 12` AND the `uint32_le ptxir_section_size` at `data[size-12..size-8]` does not extend beyond `data[0..size-12)`
- **AND** return `false` if any check fails

#### Scenario: Section locator algorithm

- **WHEN** `PTXIRLoader::extractPTXIR(data, size, &out_size)` is invoked on a valid embedded binary
- **THEN** the system SHALL read `ptxir_section_size` from `data[size-12..size-8]` (little-endian)
- **AND** return a pointer to `data[size-12-ptxir_section_size..size-12)` with `out_size = ptxir_section_size`

#### Scenario: Magic governance check (post-amendment)

- **WHEN** a developer proposes changing the literal value of `PTXIR_EMBED_MAGIC`
- **THEN** the change MUST trigger an amendment to ADR-0024 §更新记录 (governance check)
- **AND** MUST NOT be merged via a single OpenSpec change without ADR-0024 update
- **NOTE**: The 2026-08-07 amendment changed magic from `{'P','T','X','I','R','\x00','\x01','\x00'}` to `{'P','T','X','E','M','B','\x01','\x00'}` (governance check passed)

#### Scenario: MANIFEST section (type=6) Extend-Only backward compatibility

- **WHEN** an old PTXIR reader (pre-MANIFEST) consumes an embedded binary with section type 6
- **THEN** the reader SHALL skip the MANIFEST section (Extend-Only protocol per ADR-0023 Decision 6)
- **AND** the reader SHALL still correctly process KERNEL/REGDECL/TYPE/CONSTANT/STRING_TABLE sections
- **NOTE**: New readers SHALL additionally consume MANIFEST for kernelName/params/ptxAddressSize lookup; old readers fall back to requiring these via separate channels (out of scope for v1)

### Requirement: PTXIRLoader API contract

The system SHALL expose `PTXIRLoader` as a stateless class with all methods `public static`. The loader SHALL provide four methods for embedded-binary detection, extraction, and deserialization.

#### Scenario: hasEmbeddedPTXIR with valid embedded binary

- **WHEN** `hasEmbeddedPTXIR(embedded_bytes, embedded_size)` is called with a valid embedded binary
- **THEN** the system SHALL return `true`

#### Scenario: hasEmbeddedPTXIR with plain binary (no embed)

- **WHEN** `hasEmbeddedPTXIR(plain_bytes, plain_size)` is called with a binary that does not contain `PTXIR_EMBED_MAGIC`
- **THEN** the system SHALL return `false`

#### Scenario: hasEmbeddedPTXIR with truncated input

- **WHEN** `hasEmbeddedPTXIR(some_bytes, size_less_than_12)` is called
- **THEN** the system SHALL return `false` (not throw, not crash)

#### Scenario: hasEmbeddedPTXIR with size_le pointing outside prefix

- **WHEN** `hasEmbeddedPTXIR(data, size)` where `data` ends with `PTXIR_EMBED_MAGIC` BUT the embedded `ptxir_section_size` would place the section start before `data[0]`
- **THEN** the system SHALL return `false` (security: prevent OOB read)

#### Scenario: extractPTXIR returns nullopt for non-embedded binary

- **WHEN** `extractPTXIR(plain_bytes, plain_size, &out)` is called
- **THEN** the system SHALL return `nullptr`

#### Scenario: extractPureCubin returns hash-mismatch failure

- **WHEN** `extractPureCubin(embedded_bytes, embedded_size)` extracts the prefix
- **AND** the SHA-256 hash of the extracted prefix does NOT match the `cubin_hash` field in the Section TOC
- **THEN** the system SHALL return `std::nullopt`

#### Scenario: deserializeForCubin graceful degradation

- **WHEN** `deserializeForCubin(ptxir_section_bytes, ptxir_size)` is called with a corrupted PTXIRHeader
- **THEN** the system SHALL return `std::vector<StatementContext>` that is empty (length 0)
- **AND** SHALL NOT throw an exception (PTXIRLoader MUST wrap `deserialize_from_string` in try/catch)

### Requirement: config::isPTXIRModeEnabled dispatch control (env-var-overrides-INI)

The system SHALL provide `config::isPTXIRModeEnabled()` as the single source of truth for `PTXIR_MODE` dispatch decisions. The function MUST read from both the `PTXIR_MODE` environment variable (highest priority) and `configs/*.ini` `[ptxir] mode` field (fallback). Default value SHALL be `off` (compatible with pre-change behavior).

#### Scenario: env var wins over INI

- **WHEN** `PTXIR_MODE=auto` is set in environment
- **AND** `[ptxir] mode = off` is set in INI
- **THEN** `config::isPTXIRModeEnabled()` SHALL return `true`
- **NOTE**: follows precedent `PTX_EMU_GPU_CONFIG` env-overrides-INI (`src/cudart/cudart_sim.cpp:277-281`)

#### Scenario: PTXIR_MODE=off completely bypasses detection

- **WHEN** `config::isPTXIRModeEnabled()` returns `false`
- **THEN** `__cudaRegisterFatBinary` SHALL skip the entire PTXIR detection branch
- **AND** SHALL behave byte-identically to the pre-change implementation
- **AND** the function call overhead SHALL be O(1) after first call (Meyers singleton cache)

#### Scenario: PTXIR_MODE=auto enables detection

- **WHEN** `config::isPTXIRModeEnabled()` returns `true`
- **THEN** `__cudaRegisterFatBinary` SHALL read `/proc/self/exe` tail 12 bytes and invoke `PTXIRLoader::hasEmbeddedPTXIR()`
- **AND** if true, SHALL invoke the extraction + deserialization pipeline
- **AND** on any pipeline failure, SHALL fall back to the standard cubin path

#### Scenario: Default behavior preserves compatibility

- **WHEN** `PTXIR_MODE` is not set in env var or INI
- **THEN** `config::isPTXIRModeEnabled()` SHALL return `false`
- **AND** the system SHALL behave byte-identically to the pre-change implementation

### Requirement: __cudaRegisterFatBinary ABI stability

The system SHALL NOT modify the 4-parameter signature of `__cudaRegisterFatBinary(void **fatCubinHandle, void *fat_bin, unsigned long long fat_bin_size, unsigned int version)` (verified at `src/cudart/cudart_sim.cpp:354`). The function SHALL only add an internal dispatch branch gated by `config::isPTXIRModeEnabled()`.

#### Scenario: ABI signature unchanged

- **WHEN** `nm -D lib/libcudart.so | grep cudaRegisterFatBinary` is invoked before and after the change
- **THEN** the symbol size SHALL be identical (delta == 0)
- **AND** the mangled name SHALL be unchanged

#### Scenario: fat_bin parameter not dereferenced

- **WHEN** `__cudaRegisterFatBinary` is called with `fat_bin = nullptr`
- **THEN** the dispatch branch SHALL NOT crash
- **AND** the system SHALL NOT attempt to read memory pointed to by `fat_bin`
- **NOTE**: byte source = `/proc/self/exe` tail, NOT `fat_bin` (verified Oracle 2026-08-07: `fat_bin` is a dead parameter at `cudart_sim.cpp:372`, only debug-printed)

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

### Requirement: PtxContextAdapter contract

The system SHALL provide `PtxContextAdapter::fromEmbedded(stmts, manifest) → PtxContext` to construct a valid `PtxContext` from PTXIR-deserialized `StatementContext[]` plus an `EmbeddedKernelManifest`. This is REQUIRED because `deserialize_statements` does not write `kernelName`/`kernelParams`/`ptxAddressSize` fields (`ptxir_writer.cpp:154-160` + `ptxir_serialization.cpp:111` verified).

#### Scenario: fromEmbedded populates kernelName from manifest

- **WHEN** `fromEmbedded(stmts, manifest)` is called with `manifest.kernelName = "myKernel"`
- **THEN** `PtxContext.ptxKernels[0].kernelName` SHALL equal `"myKernel"`

#### Scenario: fromEmbedded populates kernelParams from manifest

- **WHEN** `fromEmbedded(stmts, manifest)` is called with `manifest.params` containing 2 `ParamContext` entries
- **THEN** `PtxContext.ptxKernels[0].kernelParams.size()` SHALL equal 2

#### Scenario: fromEmbedded populates ptxAddressSize from manifest

- **WHEN** `fromEmbedded(stmts, manifest)` is called with `manifest.ptxAddressSize = 32`
- **THEN** `PtxContext.ptxAddressSize` SHALL equal 32

#### Scenario: fromEmbedded propagates statements

- **WHEN** `fromEmbedded(stmts, manifest)` is called with `stmts.size() = N`
- **THEN** `PtxContext.ptxKernels[0].kernelStatements.size()` SHALL equal N

#### Scenario: PtxContextAdapter guards against silent total_param_size=0 failure

- **WHEN** a kernel with `.param .u64 x, .param .u64 y` is deserialized from PTXIR
- **AND** the manifest is correctly populated with both `ParamContext` entries
- **THEN** `cudaLaunchKernel(func, grid, block, args, ...)` SHALL correctly pass both `x` and `y` to the kernel
- **NOTE**: without `PtxContextAdapter`, `setupKernelArguments` (`ptx_interpreter.cpp:421-434`) computes `total_param_size = 0` and silently drops args