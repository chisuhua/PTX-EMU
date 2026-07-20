## Why

CppTLM P1 PTX-EMU shim 已实现但缺少**用真实 CUDA kernel 验证完整协同仿真链路**的测试。现有 cpptlm 测试仅验证：
- 单元层面：bridge ABI 签名、inject 接口、scoreboard 分配
- Smoke：PtxEmuDriverShim 生命周期（无 kernel 执行）

缺少的验证：
- bridge 路径下 `cudaLaunchKernel` → `prepareKernelLaunchRequest()` → IR 正确性
- `GPUContext::exe_once()` 真实执行 PTX 指令
- `on_complete` → `mark_complete` → `is_kernel_complete` 回调链
- kernel 执行后内存/寄存器输出正确性（golden value 对比）

## What Changes

### 新增

- **新增** `tests/e2e/cosim/kernel_vector_add.cu` — CUDA vectorAdd kernel（~30 LOC CUDA）
- **新增** `tests/e2e/cosim/test_cosim_vector_add.cpp` — Catch2 E2E 测试
  - 编译 CUDA → 提取 PTX → `__cudaRegisterFatBinary` → `cudaLaunchKernel`
  - bridge 模式下验证完整链路：submit → execute → on_complete → mark_complete
  - 回读 GPU 内存输出，与 CPU golden value 对比
- **修改** `tests/e2e/CMakeLists.txt` — 添加 `e2e_cosim_vector_add` 测试目标

### 影响

| 文件 | 类型 | LOC |
|------|------|:---:|
| `tests/e2e/cosim/kernel_vector_add.cu` | 新增 | ~30 |
| `tests/e2e/cosim/test_cosim_vector_add.cpp` | 新增 | ~100 |
| `tests/e2e/CMakeLists.txt` | 修改 | +10 |
| **合计** | | **~140** |

## Capabilities

### New Capabilities

- `cosim-e2e-vector-add`: CUDA vectorAdd kernel 端到端测试 — 验证 CppTLM bridge 路径下完整 PTX 执行 + 结果回读

## Impact

纯测试新增，不影响任何现有代码路径。`BUILD_LIB_CPPTLM_CUDART=OFF` 时测试标记为 SKIP。

## Design-Time Checklist

- [x] 无函数迁移（纯新增测试）
- [x] 无状态修改
- [x] 单 Phase 推进（1 commit）
- [x] 引用 ADR-0021 §2026-07-19 Postmortem