# error-mapping Specification

## Purpose
TBD - created by archiving change ptxir-driver-api-front-door. Update Purpose after archive.
## Requirements
### Requirement: 7-class error mapping covers all expected Driver API failure modes

The system SHALL map Driver API failures to one of 7 `CUresult` codes per architecture §7:

| Driver API failure | Returned code |
|---|---|
| Image class not supported (cubin/fatbin/Tile IR) | `CUDA_ERROR_INVALID_IMAGE` |
| Malformed PTX text | `CUDA_ERROR_INVALID_PTX` |
| Malformed PTXIR binary | `CUDA_ERROR_INVALID_PTXIR` |
| Unknown `CUmodule` handle | `CUDA_ERROR_INVALID_HANDLE` |
| Unknown `CUfunction` handle | `CUDA_ERROR_INVALID_HANDLE` |
| Missing kernel symbol in module | `CUDA_ERROR_NOT_FOUND` |
| Stale function handle (parent module unloaded) | `CUDA_ERROR_INVALID_HANDLE` |

All failure paths MUST return `CUresult` (no C++ exceptions thrown across the Driver API boundary).

#### Scenario: malformed PTX returns INVALID_PTX

- **WHEN** application calls `cuModuleLoadData` with bytes classified as PTX_TEXT that fail to parse
- **THEN** system returns `CUDA_ERROR_INVALID_PTX`

#### Scenario: malformed PTXIR returns INVALID_PTXIR

- **WHEN** application calls `cuModuleLoadData` with bytes classified as STANDALONE_PTXIR that fail `PTXIRLoader::deserializeForCubin()`
- **THEN** system returns `CUDA_ERROR_INVALID_PTXIR`

#### Scenario: missing kernel symbol returns NOT_FOUND

- **WHEN** application calls `cuModuleGetFunction(&func, module, "missing_kernel")` on a valid module
- **THEN** system returns `CUDA_ERROR_NOT_FOUND`

