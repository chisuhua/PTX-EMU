## Why

CppTLM P1 PTX-EMU shim 已实现但缺少**用真实 CUDA kernel 验证完整协同仿真链路**的测试。现有 cpptlm 测试仅验证：
- 单元层面：bridge ABI 签名、inject 接口、scoreboard 分配
- Smoke：PtxEmuDriverShim 生命周期（无 kernel 执行）

缺少的验证：
- bridge 路径下 `cudaLaunchKernel` → `prepareKernelLaunchRequest()` → IR 正确性
- `GPUContext::exe_once()` 真实执行 PTX 指令（由测试内 `g_ptx_emu_driver_shim->advance()` 驱动）
- `on_complete` → `mark_complete` 回调链（通过 golden value 匹配 + `cudaDeviceSynchronize` 返回间接验证）
- kernel 执行后内存/寄存器输出正确性（golden value 对比）

## What Changes

### 新增

- **新增** `tests/e2e/cosim/kernel_vector_add.cu` — CUDA vectorAdd kernel（~30 LOC CUDA）
- **新增** `tests/e2e/cosim/test_cosim_vector_add.cpp` — Catch2 E2E 测试
  - 编译 CUDA → 提取 PTX → `__cudaRegisterFatBinary` → `cudaLaunchKernel`
  - bridge 模式下验证完整链路：submit → 测试驱动 advance() 执行 → on_complete → mark_complete
  - 回读 GPU 内存输出，与 CPU golden value 对比
- **修改** `tests/e2e/CMakeLists.txt` — 添加 `e2e_cosim_vector_add` 测试目标
- **修改** `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` — 新增 `extern PtxEmuDriverShim* g_ptx_emu_driver_shim;`（+1 行）
- **修改** `src/cudart/cudart_sim.cpp:137` — 移除 `static` 关键字（±0 行）

### 影响

| 文件 | 类型 | LOC |
|------|------|:---:|
| `tests/e2e/cosim/kernel_vector_add.cu` | 新增 | ~30 |
| `tests/e2e/cosim/test_cosim_vector_add.cpp` | 新增 | ~100 |
| `tests/e2e/CMakeLists.txt` | 修改 | +10 |
| `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` | 修改 | +1 |
| `src/cudart/cudart_sim.cpp` | 修改 | ±0 |
| **合计** | | **~141** |

## Capabilities

### New Capabilities

- `cosim-e2e-vector-add`: CUDA vectorAdd kernel 端到端测试 — 验证 CppTLM bridge 路径下完整 PTX 执行 + 结果回读

## Impact

以测试新增为主（~140 LOC），附带 1 行生产代码符号可见性变更（+1 / ±0，零行为变更）。
详见 [design.md §D6 - 执行驱动模型](design.md#d6-执行驱动模型)。`BUILD_LIB_CPPTLM_CUDART=OFF` 时测试目标不存在（ctest 无匹配）。

## Design-Time Checklist

- [x] 无函数迁移（纯新增 + 1 行可见性提升）
- [x] 无状态修改
- [x] 单 Phase 推进（1 commit）
- [x] 引用 ADR-0021 §2026-07-19 Postmortem