# e2e-driver-api-exec Specification

## Purpose
TBD - created by archiving change fix-path-coverage-gaps. Update Purpose after archive.
## Requirements
### Requirement: Driver API 真实成功 kernel 执行

PTX-EMU SHALL 提供 `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp` e2e 测试，验证 Driver API 真实成功执行 kernel，而非仅验证错误路径（当前 `test_error_mapping.cpp` 已覆盖 NULL/stale handle）。该测试覆盖 `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cudaLaunchKernel` 全链路在生产场景下的可用性。

测试 SHALL 使用 v2 manifest（`kernels[]` 非空）的 PTXIR image fixture，满足 NOT_FOUND 测试要求。测试 SHALL 真实调用 `cuLaunchKernel`（不是仅测 error path），验证 kernel output buffer 内容正确（不能只验证 rc == 0）。

#### Scenario: 完整 cuModule* 流程成功执行

- **WHEN** 给定 PTXIR image bytes（含 `ManifestSection` + statements），调用 cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel → cudaLaunchKernel
- **THEN** 三个调用均 CUDA_SUCCESS, mod != nullptr, func != nullptr，输出 buffer 与 Path 1B 字节级一致

#### Scenario: Duplicate handle

- **WHEN** 同一 PTXIR image 两次 cuModuleLoadData，第二次调用
- **THEN** 第二次 CUDA_SUCCESS 但生成新 mod handle，两次 mod 不相等

#### Scenario: Not-found error (CUDA_ERROR_NOT_FOUND)

- **WHEN** cuModuleLoadData 后的 module fixture manifest 为 v2 格式（`kernels[]` 非空），调用 `cuModuleGetFunction(&func, mod, "nonexistent_kernel")`
- **THEN** 返回 `CUDA_ERROR_NOT_FOUND`，`func` 保持未修改

#### Scenario: cuLaunchKernel 错误路径回归

- **WHEN** func == nullptr 或 params == nullptr，调用 cuLaunchKernel
- **THEN** 返回 CUDA_ERROR_INVALID_VALUE（per `cudart_sim.cpp:607`）

### Requirement: cuModuleUnload func2name 失效验证

PTX-EMU SHALL 在 Phase 2 e2e 测试中显式覆盖 `cuModuleUnload` 后 func2name 失效（per `cudart_sim.cpp:573-592`）。测试 SHALL 验证 func2name 失效行为，不要修改 `ModuleRegistry::insert` 语义（依赖现有重复 handle 行为）。

#### Scenario: cuModuleUnload 后 cuModuleGetFunction 失败

- **WHEN** cuModuleUnload 后再调 `cuModuleGetFunction(&func, mod, "valid_kernel")`
- **THEN** 返回 CUDA_ERROR_INVALID_CONTEXT 或类似错误（func2name 已失效）

### Requirement: cuModuleLoadData negative paths 覆盖

PTX-EMU SHALL 在 Phase 2 e2e 测试中覆盖 `cuModuleLoadData` 的 negative paths：null args, non-PTXIR magic, cubin/fatbin image。测试 SHALL 验证错误返回路径，不依赖 production code 修改。

#### Scenario: cuModuleLoadData null args

- **WHEN** 调用 cuModuleLoadData 时 image == nullptr 或 mod == nullptr
- **THEN** 返回 CUDA_ERROR_INVALID_VALUE

#### Scenario: cuModuleLoadData non-PTXIR magic

- **WHEN** 调用 cuModuleLoadData 时 image bytes 不是 PTXIR 格式
- **THEN** 返回 CUDA_ERROR_INVALID_IMAGE 或类似错误

