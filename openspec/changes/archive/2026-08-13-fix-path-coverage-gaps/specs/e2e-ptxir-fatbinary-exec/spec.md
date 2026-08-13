## ADDED Requirements

### Requirement: PTXIR fat-binary 真实端到端执行

PTX-EMU SHALL 提供 `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp` e2e 测试，验证 PTXIR fat-binary 端到端真实执行，而非仅验证格式兼容性。该测试覆盖 `try_ptxir_dispatch_from_memory` → `g_ptx_interpreter` → `cudaLaunchKernel` 全链路在生产场景下的可用性。

测试 SHALL fork+exec 启动 standalone CUDA binary（避免 SingletonGuard 二次调用 FATAL abort，per `cudart_sim.cpp:329-335`）。standalone binary 必须 link PTX-EMU `lib/libcudart.so`（不能用 nvcc 自带的 cuda runtime），且 binary 末尾必须通过 `build/bin/ptxir_embed --in-cubin/--in-ptx/--kernel-name/--out` 嵌入 PTXIR section。

#### Scenario: 标准 PTXIR dispatch 流程 (kSuccess)

- **WHEN** 用户执行 standalone binary `kernel_exec_ptxir`，触发 `__cudaRegisterFatBinary` → `try_ptxir_dispatch_from_memory` → `g_ptx_interpreter->set_ptx_context` → `cudaLaunchKernel<<<grid,block>>>(vec_add_kernel, args)`
- **THEN** binary stdout 输出 `RESULT: vec_add=<expected> matmul=<expected> reduce=<expected>`，binary exit code = 0，PTXIR dispatch 命中（PATH="" 保证 fallback cuobjdump 失败；若 PTXIR 也失败 binary 会输出错误而非 RESULT 行）

#### Scenario: PTXIR footer 缺失 (kNoFooter)

- **WHEN** standalone CUDA binary 无 PTXIR footer（但 `PTXIR_MODE=auto` 已设置）触发 `__cudaRegisterFatBinary` 调用
- **THEN** `try_ptxir_dispatch_from_memory` 返回 `kNoFooter`，控制流转入 `extract_ptx_with_cuobjdump` 但 PATH="" 使其失败，`__cudaRegisterFatBinary` 返回 nullptr，stderr 输出 `Error: Could not extract PTX code`

#### Scenario: PTXIR footer 损坏 (kMalformedPtxir)

- **WHEN** standalone CUDA binary 含 magic 但 footer body 损坏（e.g., u32 size 超过 binary 长度）触发 `__cudaRegisterFatBinary` 调用
- **THEN** `try_ptxir_dispatch_from_memory` 返回 `kMalformedPtxir`，`__cudaRegisterFatBinary` 返回 nullptr，stderr 输出 `malformed embedded PTXIR: footer present but deserialize failed`

#### Scenario: manifest kernel_name 为空 (kMalformedManifest)

- **WHEN** standalone CUDA binary 含 valid PTXIR footer + valid statements，但 `manifest.kernel_name` 为空（v1 manifest 缺失必填字段）触发 `__cudaRegisterFatBinary` 调用
- **THEN** `try_ptxir_dispatch_from_memory` 返回 `kMalformedManifest`，`__cudaRegisterFatBinary` 返回 nullptr，stderr 输出 `manifest mismatch: kernel_name is empty`

#### Scenario: Path 1B vs Path 1A 字节级一致

- **WHEN** 同一 `path_1B_kernels.cu` 编译两份 binary: `kernel_exec_ptxir`（含 PTXIR footer, 走 Path 1B）+ `kernel_exec_legacy`（无 PTXIR footer, 走 Path 1A）各执行一次
- **THEN** 两 binary 输出 stdout 字节级一致（验证 PTXIR fast-path 与 ANTLR parse 路径语义等价）

### Requirement: Anti-fallback guard 防止 cuobjdump 误命中

PTX-EMU SHALL 在 Phase 1 e2e 测试中设置 `PATH=""`（或 unset cuobjdump location），使 `extract_ptx_with_cuobjdump` 子进程调用失败。若 PTXIR dispatch 也失败，测试 SHALL 收到 FATAL 或空输出（证明 fallback 没生效）。

#### Scenario: Anti-fallback PATH 操作生效

- **WHEN** 在 fork+exec 前设置 `PATH=""`（or unset cuobjdump location），使 `extract_ptx_with_cuobjdump` 子进程调用失败
- **THEN** 若 PTXIR dispatch 也失败，test_ptxir_fatbinary_exec.cpp 收到 FATAL 或空输出（证明 fallback 没生效）

### Requirement: ≥3 kernels 真实执行

PTX-EMU SHALL 在 Phase 1 e2e 测试中覆盖 ≥3 个不同复杂度的 kernels（vector_add, matmul, reduction），每个 kernel 的 PTXIR SHALL 含 valid `manifest.kernel_name`（v1 manifest，触发 kMalformedManifest 会失败）。测试 SHALL 验证 output 与 Path 1A 字节级一致（per `tests/ptxir/...` 已有 precedent: 5 byte-identical gates verified）。

#### Scenario: vector_add / matmul / reduction 三个 kernels 均执行成功

- **WHEN** standalone binary 包含 ≥3 个 kernels 的 PTXIR section，每个 kernel 都有 valid manifest.kernel_name
- **THEN** binary 输出每个 kernel 的 expected result，且与 Path 1A 编译的 baseline binary 输出字节级一致