# kernel-selection Specification

## Purpose
TBD - created by archiving change multi-kernel-manifest-adr-0028. Update Purpose after archive.
## Requirements
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

### Requirement: `cuModuleGetFunction` 多 kernel distinct-handle 解析

The system SHALL replace the existing `cuModuleGetFunction` stub at `src/cudart/cudart_sim.cpp:556-570` with a real implementation that maintains a per-module name→`CUfunction` registry (in `ModuleRegistry`), supporting multi-kernel lookup where each unique kernel name maps to a distinct `CUfunction` handle.

#### Scenario: 多 kernel module 中每个 kernel name 返回独立 handle

- **WHEN** a multi-kernel module (loaded with kernels `vec_add`, `mat_mul`, `reduce_sum`) receives 3 sequential calls to `cuModuleGetFunction(&fn, module, "<name>")` for each kernel name
- **THEN** the system returns 3 distinct `CUfunction` handles; each handle is unique (verified by address inequality)

#### Scenario: within-module duplicate kernel name first-match wins (SC-8)

- **WHEN** a multi-kernel module contains 2 entries with the same kernel name (e.g., `_Z7vec_add` appearing twice with different arg lists)
- **THEN** `cuModuleGetFunction(&fn, module, "vec_add")` returns the first-match entry's `CUfunction`; subsequent calls with the same name return the same handle (no duplicate allocation)

#### Scenario: 不存在的 kernel name 返回 invalid handle error

- **WHEN** `cuModuleGetFunction(&fn, module, "nonexistent_kernel")` is called on a loaded module
- **THEN** the system returns `CUDA_ERROR_NOT_FOUND` (or equivalent error code per `cudaError.h` enum)

#### Scenario: stale module handle 返回 CUDA_ERROR_INVALID_HANDLE

- **WHEN** a module has been unloaded (`cuModuleUnload(hmod)`) and a stale `cuModuleGetFunction(&fn, hmod, "kernel_a")` is called with the old handle
- **THEN** the system returns `CUDA_ERROR_INVALID_HANDLE` (per `cudart_sim.cpp:563` existing behavior)

### Requirement: v2 PTXIR writer 完整 multi-entry 写入

The system SHALL update `src/ptx_ir/ptxir_writer.cpp::writeManifestSection` to write the full `ManifestSection.kernels` vector (when non-empty), preserving the `kernel_name` field as `kernels[0].name` for v1 backward-compat.

#### Scenario: v2 writer 输出 multi-entry binary

- **WHEN** a writer call provides a `ManifestSection` with `kernels.size() == 3` (kernel_a, kernel_b, kernel_c) and matching `kernel_name == "kernel_a"`
- **THEN** the output PTXIR binary contains all 3 `KernelEntry` records in the manifest section; `kernel_name` field is preserved as `"kernel_a"`

#### Scenario: round-trip v2 writer → reader 保留所有 entry

- **WHEN** a v2 PTXIR binary (written with multi-entry) is loaded by the reader
- **THEN** `manifest.kernels` has size 3 with names matching the original (`kernel_a`, `kernel_b`, `kernel_c`); `manifest.kernel_name == "kernel_a"`

#### Scenario: v1 reader 加载 v2 binary 触发 backward-compat synthesis

- **WHEN** a v1 reader (without `kernels` field awareness) loads a v2 binary (with empty `kernels` after synthesis fallback)
- **THEN** the reader synthesizes a single-entry `kernels` vector from `kernel_name`; behavior is byte-identical to v1

#### Scenario: empty kernels + empty kernel_name 抛错

- **WHEN** a writer call provides `ManifestSection` with both `kernels.empty()` and `kernel_name.empty()`
- **THEN** the writer throws `std::invalid_argument("ManifestSection must have at least one kernel entry")`

### Requirement: `libptxemu_device.so` 多 kernel 枚举 API (3 新函数)

The system SHALL add 3 new `extern "C"` functions to `include/cudart/cpptlm_module.h` (with `CPPTLM_MODULE_VERSION 1→2` bump):
1. `int ptxemu_image_kernel_count(uint64_t handle)` — returns N (negative on error)
2. `int ptxemu_image_kernel_name_at(uint64_t handle, uint32_t idx, char* buf, size_t buf_size)` — writes kernel name at index `idx`
3. `int ptxemu_image_execute_named(uint64_t handle, const char* kernel_name, ...)` — executes the named kernel (replacing `kernels[0]` hardcoded fallback)

#### Scenario: ptxemu_image_kernel_count 返回 N

- **WHEN** a multi-kernel image with 3 kernels is loaded and `ptxemu_image_kernel_count(handle)` is called
- **THEN** returns 3; for a single-kernel image, returns 1

#### Scenario: ptxemu_image_kernel_name_at 遍历所有 kernel

- **WHEN** a multi-kernel image (kernels: vec_add, mat_mul, reduce_sum) is loaded and `ptxemu_image_kernel_name_at(handle, idx, buf, buf_size)` is called with idx=0,1,2 and sufficient buf_size
- **THEN** each call writes the corresponding kernel name; sequential calls enumerate all kernels

#### Scenario: ptxemu_image_kernel_name_at 截断契约

- **WHEN** `buf_size == 0` is passed to `ptxemu_image_kernel_name_at`
- **THEN** returns -1 (caller should retry with larger buf_size to query required length)

- **WHEN** `buf_size < strlen(kernel_name) + 1` is passed
- **THEN** returns the truncated length; caller must verify NUL termination; no buffer overflow

#### Scenario: ptxemu_image_execute_named 替代 kernels[0] 硬编码

- **WHEN** a multi-kernel image is loaded and `ptxemu_image_execute_named(handle, "mat_mul", ...)` is called
- **THEN** the mat_mul kernel executes (not vec_add); kernel selection respects the `kernel_name` parameter

#### Scenario: stale handle 返回 error code

- **WHEN** `ptxemu_image_kernel_count(handle)`, `ptxemu_image_kernel_name_at(handle, ...)`, or `ptxemu_image_execute_named(handle, ...)` is called with a handle that has been unloaded
- **THEN** each function returns -EINVAL (-22) (per existing `cpptlm_module.cpp` convention)

### Requirement: ABI baseline 回归 (v1 binary 加载)

The system SHALL maintain a regression test suite that loads v1 single-kernel PTXIR binaries and verifies byte-identical behavior to v2 reader with backward-compat synthesis.

#### Scenario: v1 binary 加载触发 backward-compat synthesis

- **WHEN** a v1 PTXIR binary (PTXIR_VERSION=3, `ManifestSection.kernels` empty, `kernel_name` non-empty) is loaded on v2 reader
- **THEN** `manifest.kernels` synthesizes a single-entry vector with the kernel name from `kernel_name`; behavior is byte-identical to v1

#### Scenario: cpptlm_module v1 binary 执行无变化

- **WHEN** a v1 PTXIR binary (with single kernel) is loaded via `ptxemu_image_load` and `ptxemu_image_execute` is called
- **THEN** execution proceeds identically to v2 with synthesized single-entry; no ABI breakage

#### Scenario: cuModuleLoadData v1 binary 路径无变化

- **WHEN** a v1 binary is loaded via `cuModuleLoadData` and `cuLaunchKernel` is called with the registered kernel
- **THEN** execution proceeds via legacy path with no multi-entry overhead

### Requirement: 锁顺序契约 (`exec_mu_` → `mu_`)

The system SHALL maintain the lock ordering `exec_mu_` (acquired first) → `mu_` (acquired second) for all public methods that hold both locks. The `unload()` method uses `try_lock(exec_mu_)` to detect in-flight execution.

#### Scenario: execute_named 与 unload 并发无 race window

- **WHEN** thread A calls `ptxemu_image_execute_named(handle, ...)` while thread B calls `ptxemu_image_unload(handle)` concurrently
- **THEN** thread B's `try_lock(exec_mu_)` either succeeds (no in-flight execute) or fails immediately (in-flight execute detected); no race window between mu_ release and exec_mu_ acquire

#### Scenario: 同一 handle 并发 execute_named 串行化

- **WHEN** 2 threads call `ptxemu_image_execute_named(handle, ...)` concurrently for the same handle
- **THEN** `exec_mu_` serializes the executions; no data race on `images_[handle]` map access

### Requirement: 文档同步 (`KernelEntry` 数据冗余段落)

The system SHALL document the data redundancy between `KernelEntry.arg_count` and `ManifestParam` vector size in `docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md` §8, declaring `ManifestParam` vector as the source of truth and `KernelEntry.arg_count` as a derived convenience field.

#### Scenario: arg_count 与 ManifestParam size 数字一致

- **WHEN** a multi-kernel module is loaded and `manifest.kernels[i].arg_count` is compared with `manifest.params.size()`
- **THEN** both values are equal (verifiable via unit test in `tests/unit/cudart/test_multi_kernel_selection.cpp`)

#### Scenario: documentation paragraph present in gap analysis

- **WHEN** `docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md` is reviewed
- **THEN** §8 contains a "Data Redundancy" section explaining the source-of-truth relationship

