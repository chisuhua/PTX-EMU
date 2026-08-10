# Spec: image-classifier

## ADDED Requirements

### Requirement: 6-class image classifier categorizes input bytes

The system SHALL provide `ImageClassifier::classify(const void* bytes, size_t size) -> ImageClass` returning one of 6 categories:
- `PTX_TEXT` — source PTX text (SUPPORTED)
- `STANDALONE_PTXIR` — standalone PTXIR binary (SUPPORTED)
- `EXECUTABLE_TAIL_PTXIR` — PTXIR suffix appended to executable (REJECTED — defer to legacy front door)
- `NVIDIA_CUBIN` — raw NVIDIA cubin (NOT SUPPORTED → `CUDA_ERROR_INVALID_IMAGE`)
- `NVIDIA_FATBIN` — NVIDIA fatbin container (NOT SUPPORTED → `CUDA_ERROR_INVALID_IMAGE`)
- `TILE_IR` — Tile IR (NOT SUPPORTED → `CUDA_ERROR_INVALID_IMAGE`)

Classification MUST be a pure function (no side effects, no allocations on hot path) and MUST be unit-testable in isolation.

#### Scenario: PTX text recognized

- **WHEN** input bytes start with PTX keywords (`.version`, `.entry`, `.target`) after stripping optional shebang
- **THEN** classifier returns `PTX_TEXT`

#### Scenario: standalone PTXIR recognized via magic footer

- **WHEN** input bytes end with the PTXIR magic footer pattern (per architecture §4.1)
- **THEN** classifier returns `STANDALONE_PTXIR`

#### Scenario: executable-tail PTXIR suffix detected and REJECTED

- **WHEN** input bytes start with ELF magic (`\x7fELF`) AND end with PTXIR magic footer
- **THEN** classifier returns `EXECUTABLE_TAIL_PTXIR` (deferred to legacy front door — in-memory path returns `CUDA_ERROR_NOT_SUPPORTED`)

#### Scenario: NVIDIA cubin returns INVALID_IMAGE

- **WHEN** input bytes start with NVIDIA cubin magic (per architecture §5.1)
- **THEN** classifier returns `NVIDIA_CUBIN` and `cuModuleLoadData` returns `CUDA_ERROR_INVALID_IMAGE`

#### Scenario: NVIDIA fatbin returns INVALID_IMAGE

- **WHEN** input bytes start with NVIDIA fatbin magic
- **THEN** classifier returns `NVIDIA_FATBIN` and `cuModuleLoadData` returns `CUDA_ERROR_INVALID_IMAGE`

#### Scenario: Tile IR returns INVALID_IMAGE

- **WHEN** input bytes start with Tile IR magic
- **THEN** classifier returns `TILE_IR` and `cuModuleLoadData` returns `CUDA_ERROR_INVALID_IMAGE`

### Requirement: classifier MUST NOT mutate input bytes or read external state

The classifier MUST NOT:
- Read `/proc/self/exe`
- Call `cuobjdump`
- Read `PTXIR_MODE` env var
- Touch `libcudart.so` global state

#### Scenario: classifier works on read-only memory-mapped bytes

- **WHEN** input bytes point to memory mapped with `PROT_READ` only
- **THEN** classifier returns the correct category without crashing