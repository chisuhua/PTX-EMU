# Spec: kernel-selection

## ADDED Requirements

### Requirement: 运行时按 kernel 名字选择 entry

The system SHALL provide `cuModuleGetFunction(CUfunction* func, CUmodule module, const char* name)` (in-memory path) and `__cudaRegisterFatBinary` legacy path, both of which resolve `name` against the module's `kernel_entries` vector and return the matching entry as a `CUfunction` handle.

#### Scenario: legacy path 按名选择 multi-entry binary 的指定 kernel

- **WHEN** application loads a multi-entry binary via legacy `__cudaRegisterFatBinary` and requests kernel "kernel_B"
- **THEN** system returns the CUfunction handle for kernel B; launch executes B's `kernelStatements`

#### Scenario: in-memory path 按名选择 multi-entry binary 的指定 kernel

- **WHEN** application calls `cuModuleGetFunction(&func, module, "kernel_B")` on a multi-entry module loaded via `cuModuleLoadData`
- **THEN** system returns the CUfunction handle for kernel B

### Requirement: `PtxEmuImageExecutor` 多 entry handle 解析

The system SHALL update `PtxEmuImageExecutor::load_image` to return handles supporting multi-entry lookup. Either `ptxemu_image_kernel_name` is upgraded to support multi-entry queries, OR a new `ptxemu_image_get_function_by_name` API is added.

#### Scenario: libptxemu_device.so 暴露 multi-entry lookup API

- **WHEN** UsrLinuxEmu HAL calls `ptxemu_image_get_function_by_name(handle, "kernel_B", &func)`
- **THEN** system returns the CUfunction handle for kernel B within the multi-entry image

### Requirement: 3 个已 ship ADR 的 §v1 限制段落更新

The system SHALL update `docs/adr/ADR-0025-ptxir-build-cli.md`, `docs/adr/ADR-0027-ptx-nvcc-wrapper.md`, `docs/adr/ADR-0029-ptxemu-image-executor.md` to reflect that v1 single-kernel limitation is **removed**.

#### Scenario: ADR-0025 §v1 段落改为"已支持 multi-kernel"

- **WHEN** ADR-0028 ships
- **THEN** ADR-0025 §v1 limitation paragraph is updated to remove the "single kernel" caveat

#### Scenario: ptxir-toolchain-stack.md 升级到 v1.4

- **WHEN** ADR-0028 ships
- **THEN** `docs/architecture/ptxir-toolchain-stack.md` is bumped to v1.4 with changelog entry; §11 BLOCKING DEPENDENCY mark is removed