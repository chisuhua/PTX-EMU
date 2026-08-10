# ptxemu-image-executor Specification

## Purpose
TBD - created by archiving change feat-ptxemu-image-executor. Update Purpose after archive.
## Requirements
### Requirement: cpptlm_module.h 公共 C-API(5 extern "C" 入口)

PTX-EMU SHALL provide a public C-API header `include/cudart/cpptlm_module.h` for in-memory PTXIR image loading and execution. The header SHALL declare **5 `extern "C"` functions** and **one version macro**, with no PTX-EMU internal type exposure(governance per `include/cudart/AGENTS.md` anti-pattern).

#### Scenario: cpptlm_module.h header content

- **WHEN** the header is included
- **THEN** the following declarations are present:
  ```c
  #define CPPTLM_MODULE_VERSION 1
  extern "C" uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size);
  extern "C" int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size);
  extern "C" int ptxemu_image_execute(uint64_t handle,
                                       uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                       uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                       size_t shared_mem_bytes,
                                       void** kernel_args, size_t args_count);
  extern "C" int ptxemu_image_unload(uint64_t handle);
  extern "C" int ptxemu_module_version(void);
  ```
- **AND** the header does NOT include any PTX-EMU internal type(per governance)
- **AND** `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` is present(if `cudaStream_t` is in scope)

#### Scenario: ABI version governance

- **WHEN** any of the 5 function signatures changes
- **THEN** `CPPTLM_MODULE_VERSION` MUST be bumped (e.g., `1` → `2`)
- **AND** `ptxemu_module_version()` MUST return the new value
- **AND** callers MAY verify via `ptxemu_module_version() == CPPTLM_MODULE_VERSION` at startup
- **NOTE**: Per `include/cudart/cpptlm_bridge.h:18-21` governance pattern; same model as `CPPTLMBRIDGE_VERSION` and `PTXIR_VERSION`

### Requirement: Image bytes input classification(4 类接受 + 3 类拒绝)

`ptxemu_image_load` SHALL classify the input `image_bytes` by leading bytes / trailing magic, then route to the appropriate deserializer. 4 image formats SHALL be accepted; 3 image formats SHALL be rejected with distinct error codes.

#### Scenario: Accepted image formats

- **WHEN** `ptxemu_image_load` is invoked with `image_bytes` matching one of the following:
  - **standalone PTXIR**: leading 4 bytes == `"PTXI"` (per `ptxir_format.h:43` `PtxirHeader::magic`)
  - **PTXIR-Embedded CUBIN**: trailing 8 bytes == `'P','T','X','E','M','B','\x01','\x00'` (per `ADR-0024 v1.1` `PTXIR_EMBED_MAGIC`)
  - **PTXIR-Embedded EXE**: same as PTXIR-Embedded CUBIN but with ELF prefix instead of cubin prefix
- **THEN** the function SHALL return a non-zero opaque handle (uint64_t)
- **AND** the image bytes SHALL be deep-copied into the executor's private storage(per D3 mutation bug fix)

#### Scenario: Rejected image formats

- **WHEN** `ptxemu_image_load` is invoked with image bytes that DO NOT match accepted formats:
  - **NVIDIA cubin**(`.cubin` ELF magic + cubin-specific section)
  - **NVIDIA fatbin**(fatbinary container format)
  - **Tile IR**(Tile-specific binary format)
- **THEN** the function SHALL return `0`(zero handle)
- **AND** the caller MAY check `errno` or subsequent `ptxemu_image_kernel_name` for specific error code

#### Scenario: Reject zero-size image

- **WHEN** `ptxemu_image_load` is invoked with `image_size == 0`
- **THEN** the function SHALL return `0` and SHALL NOT allocate a handle

#### Scenario: Reject corrupt PTXIR (deserialization failure)

- **WHEN** `ptxemu_image_load` is invoked with PTXIR-like bytes that fail `PTXIRLoader::deserializeFromString` (corrupted header / truncated section / unknown section type)
- **THEN** the function SHALL return `0`
- **AND** the underlying exception SHALL be caught and converted to error code(per `ptxir_loader.cpp:102-112` exception-safe decode)

### Requirement: v1 single-kernel constraint(per ADR-0029 §D4 + `ptxir_format.h:36-41`)

`ptxemu_image_kernel_name` SHALL return at most one kernel name per image handle in v1. This reflects the `ManifestSection` single-kernel-name v1 limitation in the PTXIR binary format.

#### Scenario: v1 single kernel per image

- **WHEN** `ptxemu_image_kernel_name(valid_handle, buf, buf_size)` is invoked
- **THEN** the function SHALL copy up to `buf_size - 1` bytes of the kernel name into `buf`, NUL-terminate
- **AND** return `0` on success
- **AND** return negative error code if the image contains zero kernels (manifest anomaly)

#### Scenario: Multi-kernel images rejected at load time

- **WHEN** an image containing N>1 `.entry` kernel symbols is loaded
- **THEN** `ptxemu_image_load` SHALL return `0` (rejected at load, per ADR-0029 §D4 + B1 decision)
- **NOTE**: Multi-kernel support is **deferred to ADR-0028 (multi-kernel manifest + runtime selection)**, which will require `PTXIR_VERSION` bump per ADR-0023 Extend-Only principle

### Requirement: Synchronous launch semantics(per ADR-0029 §D6)

`ptxemu_image_execute` SHALL block the calling thread until the kernel completes (or fails). Async launch with fence/callback is **explicitly OUT OF SCOPE for v1**.

#### Scenario: Synchronous execute returns when kernel completes

- **WHEN** `ptxemu_image_execute` is invoked with valid handle + grid/block/args
- **THEN** the calling thread SHALL block until `PtxInterpreter::launchPtxInterpreter` returns
- **AND** the function returns `0` on success
- **AND** the function returns negative error code on failure(per `cudaError_t → errno` mapping per `cpptlm_bridge.h` error convention)

#### Scenario: Async execute deferred to v2

- **WHEN** v2 introduces async launch (per ADR-0029 §D6.2 future work)
- **THEN** the v2 SHALL introduce a new `CPPTLM_MODULE_VERSION` and add fence/callback entry points
- **AND** v1 callers using the synchronous API SHALL continue to work unchanged

### Requirement: D3 mutation bug fix(per `src/cudart/ptx_interpreter.cpp:100-140`)

The executor SHALL NOT cache the deserialized `PtxContext` across launches. Each `ptxemu_image_execute` call SHALL re-deserialize from the stored image bytes, ensuring no shared mutable state between launches.

#### Scenario: Fresh PtxContext per launch (no shared state)

- **WHEN** `ptxemu_image_execute(valid_handle, ...)` is invoked
- **THEN** the executor SHALL:
  1. Acquire `exec_mu_` (mutex)
  2. Call `PTXIRLoader::deserializeForCubin(image_bytes)` + `PtxContextAdapter::fromEmbedded()` to construct a **fresh** `PtxContext`
  3. Construct a **fresh** `PtxInterpreter` instance (per [SINGLE-GPU-INSTANCE] #6 — `PtxInterpreter` is stateful non-reentrant)
  4. Call `PtxInterpreter::launchPtxInterpreter(...)` synchronously
  5. **Destruct** the fresh `PtxContext` and `PtxInterpreter`
  6. Release `exec_mu_`
- **AND** the previously-destructed `PtxContext` SHALL NOT be reused for subsequent launches

#### Scenario: Concurrent launch serialization (executor mutex)

- **WHEN** multiple threads concurrently invoke `ptxemu_image_execute(same_handle, ...)`
- **THEN** the executor mutex (`exec_mu_`) SHALL serialize the calls
- **AND** all launches SHALL complete successfully(NO deadlock, NO corruption)
- **AND** wall-clock time SHALL be ≈ N × (deserialize + execute) per launch

#### Scenario: Sequential launch determinism (no state accumulation)

- **WHEN** `ptxemu_image_execute(same_handle, ...)` is invoked N=1000 times with varying `block_dim` parameters
- **THEN** the stored `image_bytes_` SHALL NOT be mutated
- **AND** the SHA-256 hash of `image_bytes_` SHALL be byte-identical before and after N launches

#### Scenario: Double-deserialize byte-identity

- **WHEN** the same `image_bytes` is deserialized twice via `PTXIRLoader::deserializeForCubin`
- **THEN** the resulting `kernelStatements` SHALL be byte-identical(memory equality, not just logical equivalence)

### Requirement: [SINGLE-GPU-INSTANCE] 7 assumptions(per ADR-0029 §D6 + Lesson §10)

`PtxEmuImageExecutor` SHALL document 7 single-instance assumptions in its class header comment. The executor MUST be process-global singleton; concurrent multi-instance construction MUST fail loudly.

#### Scenario: g_gpu_context process-global singleton

- **WHEN** the executor is initialized
- **THEN** `extern std::unique_ptr<GPUContext> g_gpu_context;` in `ptx_interpreter.h` SHALL be defined in `src/cudart/ptx_interpreter.cpp` (single TU)
- **NOTE**: [SINGLE-GPU-INSTANCE] #1 — all images share one simulated GPU

#### Scenario: CudaDriver singleton (global memory pool)

- **WHEN** the executor allocates global/local/param memory
- **THEN** it uses `CudaDriver::instance().malloc(...)` from `cuda_driver.h`
- **NOTE**: [SINGLE-GPU-INSTANCE] #2 — all images share one global memory pool

#### Scenario: g_cpptlm_bridge nullptr (standalone mode)

- **WHEN** the executor runs in standalone mode (no CppTLM)
- **THEN** `g_cpptlm_bridge == nullptr` (per `cpptlm_bridge.h:61`)
- **NOTE**: [SINGLE-GPU-INSTANCE] #3 — standalone mode is the default; CppTLM attachment is an orthogonal concern

#### Scenario: g_image_executor process-global singleton

- **WHEN** the executor is accessed
- **THEN** `g_image_executor` is a process-global pointer to the singleton instance
- **NOTE**: [SINGLE-GPU-INSTANCE] #4 — multiple-instance construction MUST fail loudly(per Lesson §10: never silently corrupt)

#### Scenario: exec_mu_ mutex serializes same-handle launches

- **WHEN** `ptxemu_image_execute(same_handle, ...)` is called from multiple threads
- **THEN** `exec_mu_` SHALL hold the call until the previous launch completes
- **NOTE**: [SINGLE-GPU-INSTANCE] #5 — single in-flight launch per process

#### Scenario: PtxInterpreter stateful non-reentrant

- **WHEN** `ptxemu_image_execute` is called
- **THEN** a **fresh** `PtxInterpreter` instance SHALL be constructed per launch (per `src/cudart/ptx_interpreter.cpp:19-36` — caches `ptxContext/kernelContext/kernelArgs/param_space` as members)
- **NOTE**: [SINGLE-GPU-INSTANCE] #6 — PtxInterpreter MUST NOT be shared across launches

#### Scenario: No SingletonGuard coupling

- **WHEN** the executor is initialized
- **THEN** `__cudaRegisterFatBinary`'s `SingletonGuard` SHALL NOT be triggered by the executor
- **NOTE**: [SINGLE-GPU-INSTANCE] #7 — image executor path is orthogonal to legacy LD_PRELOAD `__cudaRegisterFatBinary` registration

### Requirement: Unload semantics(in-flight busy return)

`ptxemu_image_unload` SHALL reject unload if a kernel is currently in-flight on the target handle, allowing the caller to retry after kernel completion.

#### Scenario: Unload while handle has no in-flight kernel

- **WHEN** `ptxemu_image_unload(valid_handle)` is invoked and no `ptxemu_image_execute(handle, ...)` is currently in-flight
- **THEN** the handle's image bytes SHALL be erased from the executor's internal map
- **AND** the handle SHALL become invalid(returns error code on subsequent `ptxemu_image_execute(invalid_handle, ...)`)
- **AND** the function returns `0`

#### Scenario: Unload while kernel in-flight

- **WHEN** `ptxemu_image_unload(valid_handle)` is invoked while `ptxemu_image_execute(valid_handle, ...)` is currently executing in another thread
- **THEN** the function SHALL return `-EBUSY` immediately (non-blocking)
- **AND** the in-flight kernel SHALL complete normally
- **AND** the handle SHALL remain valid after the in-flight launch completes (caller may retry `ptxemu_image_unload`)

#### Scenario: Unload invalid handle

- **WHEN** `ptxemu_image_unload(0)` or `ptxemu_image_unload(already_unloaded_handle)` is invoked
- **THEN** the function SHALL return `-EINVAL` (invalid handle)

### Requirement: Invalid handle rejection

All kernel-execution functions SHALL reject invalid handles (zero handle, unknown handle, or already-unloaded handle) with a distinct error code.

#### Scenario: Execute with zero handle

- **WHEN** `ptxemu_image_execute(0, ...)` is invoked
- **THEN** the function SHALL return `-EINVAL` without invoking `PtxInterpreter`

#### Scenario: Execute with unknown handle

- **WHEN** `ptxemu_image_execute(0xDEADBEEF, ...)` is invoked(handle never loaded or already unloaded)
- **THEN** the function SHALL return `-EINVAL` without invoking `PtxInterpreter`

#### Scenario: Kernel name query with invalid handle

- **WHEN** `ptxemu_image_kernel_name(invalid_handle, buf, buf_size)` is invoked
- **THEN** the function SHALL return `-EINVAL` without writing to `buf`

### Requirement: Phase 0 5 byte-identical fallback gates(per ADR-0029 §D7)

After Phase 0 Step 1 implementation (5 global symbol relocation), 5 gates MUST all pass to prove the default LD_PRELOAD path is byte-level unchanged.

#### Scenario: Gate 1 — exported symbol surface unchanged

- **WHEN** Phase 0 Step 1 is complete
- **THEN** `diff <(nm -D --defined-only build/lib/libcudart.so | sort) <(nm -D --defined-only build-baseline/lib/libcudart.so | sort)` SHALL produce no output (empty diff)

#### Scenario: Gate 2 — SONAME preserved

- **WHEN** Phase 0 Step 1 is complete
- **THEN** `objdump -p build/lib/libcudart.so | grep SONAME` SHALL output `libcudart.so.12`

#### Scenario: Gate 3 — symlinks preserved

- **WHEN** Phase 0 Step 1 is complete
- **THEN** `ls -la build/lib/libcudart.so*` SHALL show `.12` versioned symlink + main unversioned symlink + `libcudart.so.12` real file

#### Scenario: Gate 4 — g_cpptlm_bridge nullptr path test

- **WHEN** a unit test invokes the standalone LD_PRELOAD path with `g_cpptlm_bridge == nullptr`
- **THEN** the test SHALL pass (per `cpptlm_bridge.h:61` "nullptr = 独立模式，字节级兼容" contract)

#### Scenario: Gate 5 — logger→g_gpu_context clock path test

- **WHEN** a unit test invokes `get_gpu_clock_from_context()` (per `src/utils/logger.cpp:8` extern)
- **THEN** the function SHALL return a monotonically-increasing clock value (proving `logger.cpp → ptx_interpreter.cpp::g_gpu_context` linkage survived the relocation)

### Requirement: D3 performance gate(cute_rmsnorm < 1.10 wall-time ratio)

The per-launch re-deserialize cost SHALL be measured against an eager-parse baseline; the ratio MUST be < 1.10 (10% overhead tolerance) or trigger A1 fallback.

#### Scenario: D3 perf acceptance (PASS)

- **WHEN** `bench/cute/cute_rmsnorm.ptx` PTXIR is executed via:
  - **Group A** (baseline): `ptxemu_image_load + ptxemu_image_execute × 1` (single launch, cached PtxContext)
  - **Group B** (D3 model): `ptxemu_image_load + ptxemu_image_execute × 100` (100 launches, per-launch re-deserialize)
- **THEN** the wall time ratio `B/A` SHALL be < 1.10
- **AND** the perf benchmark test SHALL PASS

#### Scenario: D3 perf failure triggers A1 fallback

- **WHEN** the wall time ratio `B/A ≥ 1.10`
- **THEN** the perf benchmark test SHALL FAIL with structured output: `deserialize_cost=1.15x  FAIL (触发 A1 fallback)`
- **AND** the implementation SHALL be flagged for A1 fallback (launch-time `kernelStatements` deep-copy)
- **NOTE**: A1 fallback is OUT OF SCOPE for this change; it SHALL be implemented in a follow-up `fix-ptxemu-image-executor-a1-fallback` change

### Requirement: Cross-repo unblock — UsrLinuxEmu adr-076 Step 1

When `feat-ptxemu-image-executor` reaches `Accepted` state and `v0.1.0` tag is published, UsrLinuxEmu's [adr-076 §Migration Step 2](https://example.com/adr-076) SHALL become unblocked.

#### Scenario: Trigger condition for UsrLinuxEmu

- **WHEN** `git tag v0.1.0` is published
- **THEN** UsrLinuxEmu may begin implementing the HAL extension(3 new ioctl 0x27/0x28/0x29 + 3 HAL fn-ptrs #66/#67/#68)
- **NOTE**: This change does NOT implement Phase 2 (cross-repo); see [UsrLinuxEmu adr-076 §Migration Step 2-3](https://example.com/adr-076)

#### Scenario: Trigger condition for TaskRunner

- **WHEN** UsrLinuxEmu's HAL extension is shipped(separate change)
- **THEN** TaskRunner may begin implementing [tadr-307](../../../../../UsrLinuxEmu/external/TaskRunner/docs/shared/adr/tadr-307-igpu-driver-kernel-module-extension.md) consumer-side integration
- **NOTE**: This change does NOT implement Phase 2 TaskRunner side; see tadr-307

