## ADDED Requirements

### Requirement: cute_rmsnorm output baseline 验证

PTX-EMU SHALL 提供 `tests/e2e/path_2D_image_executor/` 下的 e2e 测试，验证 Image Executor（Path 2D）输出与 baseline 字节级一致。Baseline 文件 SHALL 定义为 8-byte magic `PTXR_OUT\0\0` + 4-byte LE u32 size + bytes 格式（per `tests/ptxir/baselines/baseline_format.md`），文件名为 `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin`。

测试 SHALL 读 `tests/ptxir/fixtures/cute_rmsnorm.ptxir`（5294 B）作为 fixture，调 `ptxemu_image_load + ptxemu_image_execute`，验证输出 buffer 与 baseline 字节级一致（`memcmp == 0`）。测试 SHALL 在确认 simulator 正确后 commit baseline 文件（避免 baseline 错误被固化）。测试 SHALL 在运行时若 baseline 不存在则明确报错（而非自动生成或 skip）。

#### Scenario: cute_rmsnorm output 与 baseline 字节级一致

- **WHEN** 读 `tests/ptxir/fixtures/cute_rmsnorm.ptxir`（5294 B）作为 fixture，调 `ptxemu_image_load(bytes, size)` → `ptxemu_image_execute(handle, grid, block, args)`，读取输出 buffer
- **THEN** 输出 buffer 与 baseline 字节级一致（`memcmp == 0`）

### Requirement: D3 mutation 回归测试

PTX-EMU SHALL 在 Phase 3 e2e 测试中加 D3 mutation 回归测试（RED PHASE header comment，per `tests/AGENTS.md:91`）。同一 fixture 加载 2 次，两 handle 不同、两 output 字节级一致、两 unload 成功。

测试 SHALL 不修改 `ptxemu_image_*` ABI（7 符号）+ `ptxemu_module_version`，共 **8 extern "C" 符号**（`cpptlm_module.cpp:227-262`）。任何 ABI 修改必须经过 ADR 流程。测试 SHALL 不修改 `cpptlm_module.cpp` 的 SINGLE-GPU-INSTANCE 假设（per ADR-0029 D6）。

#### Scenario: 重复 load 同一 fixture

- **WHEN** 同一 fixture 加载 2 次，load + execute 两次
- **THEN** 两 handle 不同，两 output 字节级一致，两 unload 成功

### Requirement: ABI baseline 回归

PTX-EMU SHALL 在 Phase 3 测试中验证 ABI baseline 回归。`tests/integration/cpptlm/test_libptxemu_abi_baseline.cpp` SHALL 验证 ABI symbols 与 `libptxemu_abi_baseline.txt` 字节级一致（`diff <(nm -D libptxemu_device.so | grep ptxemu_ | sort) libptxemu_abi_baseline.txt` 返回 0）。

#### Scenario: ABI symbols 与 baseline 字节级一致

- **WHEN** libptxemu_device.so 已构建，执行 `tests/integration/cpptlm/test_libptxemu_abi_baseline.cpp`
- **THEN** ABI symbols 与 `libptxemu_abi_baseline.txt` 字节级一致

### Requirement: ≥4 个 error path tests

PTX-EMU SHALL 在 Phase 3 测试中新增 ≥4 个 error path tests：load garbage, execute invalid handle, unload invalid handle, kernel_name 不存在。

#### Scenario: load garbage bytes

- **WHEN** 调用 `ptxemu_image_load(garbage_bytes, garbage_size)`（非 PTXIR 格式）
- **THEN** 返回 nullptr 或 error handle

#### Scenario: execute invalid handle

- **WHEN** 调用 `ptxemu_image_execute(invalid_handle, ...)`（handle 未初始化或已 unload）
- **THEN** 返回非 0 error code

#### Scenario: unload invalid handle

- **WHEN** 调用 `ptxemu_image_unload(invalid_handle)`（handle 未初始化或已 unload）
- **THEN** 返回非 0 error code，不 crash

#### Scenario: kernel_name 不存在

- **WHEN** 调用 `ptxemu_image_get_function(handle, "nonexistent")`（PTXIR image 含 v2 manifest 但 queried name 不在 `kernels[]` 内）
- **THEN** 返回 nullptr