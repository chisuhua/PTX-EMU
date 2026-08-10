# Spec: cuda-driver-api

## ADDED Requirements

### Requirement: `cuModuleLoadData` 接受 standalone PTXIR image bytes 并返回 opaque handle

The system SHALL expose `cuModuleLoadData(CUmodule* module, const void* image)` from `libcudart.so`, which:
- Performs eager parse of the image bytes (NOT lazy)
- Deep-copies image bytes into `ModuleRecord` private storage (caller-owned pointer does NOT survive as the handle)
- Returns `CUDA_SUCCESS` on success; `CUDA_ERROR_INVALID_IMAGE` for unsupported image classes per `image-classifier` spec
- Reuses `PTXIRLoader::deserializeForCubin()` as the ONLY deserialization entry point

#### Scenario: standalone PTXIR image loaded successfully

- **WHEN** application calls `cuModuleLoadData(&module, ptxir_bytes)` with valid standalone PTXIR image
- **THEN** system deep-copies the bytes, calls `PTXIRLoader::deserializeForCubin()`, stores `ModuleRecord` in `ModuleRegistry`, and sets `*module` to the opaque handle

#### Scenario: caller-owned pointer freed after call returns safely

- **WHEN** application calls `cuModuleLoadData` and then frees the original image pointer
- **THEN** the `CUmodule` handle remains valid because the system owns an internal deep-copy

#### Scenario: concurrent `cuModuleLoadData` calls are thread-safe

- **WHEN** N host threads call `cuModuleLoadData` simultaneously
- **THEN** all N calls succeed (or fail with deterministic errors) and `ModuleRegistry::insert()` is mutex-protected

### Requirement: `cuModuleGetFunction` returns opaque `CUfunction` handle keyed by kernel name

The system SHALL expose `cuModuleGetFunction(CUfunction* func, CUmodule module, const char* name)`:
- Resolves `name` against the `ModuleRecord`'s `kernel_entries` (post Phase 12.4: `vector<kernel_entry>`)
- Returns `CUDA_SUCCESS` if the name matches; `CUDA_ERROR_NOT_FOUND` if missing
- `CUfunction` handle remains valid until the parent `CUmodule` is unloaded
- All operations are protected by `ModuleRegistry` mutex; lock order vs per-`PtxContext` lock MUST be defined (per `ptx-lessons-learned.md` recursive-lock lessons)

#### Scenario: existing kernel name resolves to handle

- **WHEN** application calls `cuModuleGetFunction(&func, module, "kernel_A")` on a module containing kernel A
- **THEN** `*func` is set to the opaque handle and `cuLaunchKernel(func, ...)` later executes kernel A

#### Scenario: unknown kernel name returns NOT_FOUND

- **WHEN** application calls `cuModuleGetFunction(&func, module, "nonexistent")`
- **THEN** system returns `CUDA_ERROR_NOT_FOUND` and `*func` is unchanged

#### Scenario: stale module handle returns INVALID_HANDLE

- **WHEN** application calls `cuModuleGetFunction(&func, stale_module, ...)` after the module was unloaded
- **THEN** system returns `CUDA_ERROR_INVALID_HANDLE`

### Requirement: `cuLaunchKernel(CUfunction, ...)` executes kernel via per-launch fresh `PtxContext`

The system SHALL expose `cuLaunchKernel(CUfunction func, ...)` Driver API version:
- Creates a fresh `PtxContext` per launch (per ADR-0029 §D3 — fixes `ptx_interpreter.cpp:100-140` mutation bug)
- Reuses the existing `cudaLaunchKernel` main execution path
- Returns `CUDA_SUCCESS` on success; appropriate error otherwise

#### Scenario: per-launch fresh PtxContext prevents mutation

- **WHEN** application launches the same `CUfunction` 1000 times with different `blockDim`
- **THEN** image bytes SHA-256 hash is unchanged after N launches (per ADR-0029 §D3 acceptance)

#### Scenario: concurrent cuLaunchKernel on same CUfunction is serialized

- **WHEN** N host threads call `cuLaunchKernel(same_func, ...)` simultaneously
- **THEN** all launches execute serially under `ModuleRegistry` mutex; no data race; deterministic output

### Requirement: `cuModuleUnload(CUmodule)` releases ModuleRecord and invalidates child function handles

The system SHALL expose `cuModuleUnload(CUmodule module)`:
- If a kernel launched from this module is in-flight, returns `CUDA_ERROR_INVALID_HANDLE` (busy)
- Otherwise, removes `ModuleRecord` from `ModuleRegistry`, releases image bytes deep-copy, and marks all child `CUfunction` handles as stale
- Subsequent `cuLaunchKernel` on any child `CUfunction` returns `CUDA_ERROR_INVALID_HANDLE`

#### Scenario: in-flight unload returns busy

- **WHEN** application calls `cuModuleUnload(module)` while a kernel from this module is still executing
- **THEN** system returns `CUDA_ERROR_INVALID_HANDLE` and the module remains loaded

#### Scenario: idle unload succeeds and invalidates children

- **WHEN** application calls `cuModuleUnload(module)` after all kernels from this module have completed
- **THEN** system removes the ModuleRecord and any subsequent `cuLaunchKernel(child_func, ...)` returns `CUDA_ERROR_INVALID_HANDLE`

### Requirement: legacy front door (`__cudaRegisterFatBinary`) is NOT modified

The system SHALL NOT modify the existing `__cudaRegisterFatBinary` legacy front door. The in-memory and legacy paths MUST coexist and not pollute each other (per architecture §4.2).

#### Scenario: PTXIR_MODE=off does not disable in-memory module loading

- **WHEN** `PTXIR_MODE=off` is set AND application calls `cuModuleLoadData`
- **THEN** in-memory PTXIR dispatch remains ON (per architecture §4.2 explicit precedence)

#### Scenario: legacy and in-memory paths coexist in same process

- **WHEN** application uses both `__cudaRegisterFatBinary` (legacy) and `cuModuleLoadData` (in-memory) in the same process
- **THEN** both paths function independently without cross-contamination