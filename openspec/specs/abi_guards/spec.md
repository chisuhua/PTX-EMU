# abi_guards Specification

## Purpose
TBD - created by archiving change cleanup-cudart-cpptlm-bridge-coupling. Update Purpose after archive.
## Requirements
### Requirement: `abi_guards.h` preserves ABI-level static_asserts after cpptlm_bridge.h removal

The file `PTX-EMU/include/cudart/abi_guards.h` SHALL contain exactly 17 `static_assert` declarations that were originally in `PTX-EMU/include/cudart/cpptlm_bridge.h` (deleted in commit `09786635`):

- 1 `static_assert` verifying `sizeof(cudaStream_t) <= sizeof(uint64_t)` (ABI width guard)
- 6 `PipelineId` endpoint static_asserts (per ADR-0016 G-D4 12-endpoint contract subset for pipeline)
- 6 `TcPrecision` endpoint static_asserts (per ADR-0016 G-D4 12-endpoint contract subset for tensor-core precision)
- 4 `std::is_same` signature checks for `IScoreboard`/`IPipelineLatencyProvider`/`ITensorCoreTiming`/`CppTLMBridge` ABI compatibility

The file MUST include required headers (`cudart/cudart_intrinsics.h` for `cudaStream_t`, `ptxsim/{scoreboard,pipeline,tensor_core}_interface.h` for vendored interfaces).

#### Scenario: abi_guards.h compiles standalone

- **WHEN** `PTX-EMU/include/cudart/abi_guards.h` is included in a translation unit
- **THEN** compilation succeeds (no missing symbols)
- **AND** all 17 static_asserts evaluate to true against the current `ptxsim/` vendored interface headers

#### Scenario: cpptlm_bridge.h `sizeof(PtxEmuDriverApi) == 64` lock NOT preserved

- **WHEN** `abi_guards.h` is read
- **THEN** the file does NOT contain `static_assert(sizeof(PtxEmuDriverApi) == 64, ...)`
- **AND** the `PtxEmuDriverApi` type is no longer referenced (the type itself was deleted)

> **Rationale**: The `sizeof(PtxEmuDriverApi) == 64` lock was a CppTLM-side constraint that required `PtxEmuDriverApi` struct to be exactly 64 bytes. With `PtxEmuDriverApi` deleted (no consumer remains), this lock is obsolete.