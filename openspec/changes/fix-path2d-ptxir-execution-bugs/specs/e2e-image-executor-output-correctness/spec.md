# e2e-image-executor-output-correctness Specification

## MODIFIED Requirements

### Requirement: cute_rmsnorm output baseline 验证

PTX-EMU SHALL provide an E2E test under `tests/e2e/path_2D_image_executor/` that validates Image Executor output against the byte-level baseline format defined by `tests/ptxir/baselines/baseline_format.md`. The test SHALL load the PTXIR fixture through the path_2D image-executor API, execute the kernel synchronously, and compare the output buffer with `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` using `memcmp == 0`. A missing baseline SHALL be an explicit test failure, not an automatic generation or skip.

#### Scenario: cute_rmsnorm output matches baseline

- **WHEN** the test loads the PTXIR fixture, executes it through `ptxemu_image_load` and `ptxemu_image_execute`, and reads the output buffer
- **THEN** the output buffer is byte-identical to the baseline

### Requirement: D3 mutation 回归测试

PTX-EMU SHALL test that loading and executing the same image twice produces independent handles and byte-identical outputs. The test SHALL verify that synchronous execution does not mutate the stored image bytes and that both handles can be unloaded successfully. The test SHALL preserve the existing ABI and `[SINGLE-GPU-INSTANCE]` assumptions documented by ADR-0029.

#### Scenario: Repeated load and execute

- **WHEN** the same fixture is loaded twice and each handle is executed
- **THEN** the handles differ, both executions complete, outputs are byte-identical, and both unloads succeed

### Requirement: ABI baseline 回归

PTX-EMU SHALL verify that the exported `libptxemu_device.so` ABI symbols remain byte-identical to the checked-in baseline after the image-executor fix.

#### Scenario: ABI symbols match baseline

- **WHEN** `libptxemu_device.so` is built and the ABI baseline test runs
- **THEN** the diff between the exported symbol list and `libptxemu_abi_baseline.txt` is empty

### Requirement: Error path coverage

PTX-EMU SHALL include at least four error-path tests for the image executor. The tests SHALL cover loading a garbage or non-PTXIR payload, executing an invalid handle, unloading an invalid handle, and querying a kernel name that does not exist in the embedded or standalone image manifest.

#### Scenario: load garbage bytes

- **WHEN** `ptxemu_image_load(garbage_bytes, garbage_size)` receives a payload that is neither standalone PTXIR nor a valid PTXIR-Embedded binary
- **THEN** the call returns an error and does not expose a valid kernel

#### Scenario: execute invalid handle

- **WHEN** `ptxemu_image_execute(invalid_handle, ...)` is called with an uninitialized or already unloaded handle
- **THEN** the call returns a nonzero error code

#### Scenario: unload invalid handle

- **WHEN** `ptxemu_image_unload(invalid_handle)` is called with an uninitialized or already unloaded handle
- **THEN** the call returns a nonzero error code without crashing

#### Scenario: kernel name not present

- **WHEN** a v2 multi-kernel image is queried for a kernel name that is not in its `kernels[]` list
- **THEN** the lookup returns a not-found or invalid-handle error