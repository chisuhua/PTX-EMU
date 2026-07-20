## ADDED Requirements

### Requirement: CUDA vectorAdd kernel end-to-end co-simulation

The system SHALL include an E2E test that launches a CUDA vectorAdd kernel via the CppTLM bridge path and verifies correct execution results.

#### Scenario: kernel compiles and launches via bridge path

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=ON` and `g_cpptlm_bridge != nullptr`
- **THEN** `cudaLaunchKernel` SHALL enqueue the kernel to `GPUContext::task_queue`
- **THEN** `prepareKernelLaunchRequest()` SHALL return a valid `KernelLaunchRequest` with IR
- **THEN** `cpptlm_set_driver` SHALL be callable (CppTLM strong definition active)

#### Scenario: PTX instructions execute correctly

- **WHEN** `GPUContext::exe_once()` processes the kernel
- **THEN** LD/ST instructions SHALL read/write correct device memory addresses
- **THEN** ADD instruction SHALL compute correct sum

#### Scenario: output matches golden value

- **WHEN** `cudaDeviceSynchronize` returns after all kernels complete
- **THEN** device memory output SHALL equal CPU-computed golden value
- **THEN** `is_kernel_complete(kernel_id)` SHALL return true

#### Scenario: test skips when bridge not enabled

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=OFF`
- **THEN** the test SHALL be filtered from ctest (not compiled)