## Why

PTX-EMU 的 CppTLM co-simulation 基础设施已就绪（bridge ABI + PtxEmuDriverShim + dual-enqueue），但存在三个缺口使**标准 CUDA 程序无法零修改跑协同仿真**：

1. **arg-count segfault**：`cudaLaunchKernel` bridge 路径的 `count_kernel_args` nullptr 哨兵遍历假设所有参数都是指针（`nullptr` 终止），当 kernel 含 `int N` 等非指针参数时越界 segfault
2. **2-cycle completion bug**：`GPUContext::exe_once()` 在 bridge 路径下 admit kernel 后立即判 `EXIT`（SM `all_warps_finished()` 误判 true），kernel 实际未执行，输出全零
3. **bridge 需手动 attach**：`g_cpptlm_bridge` 默认 `nullptr`，测试必须手动 `cpptlm_attach_bridge(&mock)` 激活 bridge 路径
4. **advance 需显式调用**：`cudaDeviceSynchronize` bridge 路径仅轮询（`poll_kernel`），不驱动 `exe_once()`，测试必须显式 `advance()`

**当前 `test_cosim_vector_add.cu` 因为缺口 3+4，混入了大量 PTX-EMU 专用代码（`MockBridge` 类定义、`cpptlm_attach_bridge`、`g_ptx_emu_driver_shim->advance()` 等），不是标准 CUDA 程序。修复缺口 1-4 后，该测试退化为纯标准 CUDA `.cu` 文件。

## What Changes

### 新增（生产代码）

- **修改** `src/cudart/cudart_sim.cpp` — 三处改动：
  1. **count_kernel_args 安全化**（bridge 路径 deep-copy args）：从 PTX context `kernelParams.size()` 获取参数计数，fallback 到原 sentinel walk（**16 行**）
  2. **自动 attach StubBridge**（`initialize_environment()` 内）：`BUILD_LIB_CPPTLM_CUDART=ON` 时内部创建 `StubBridge` 实例并 `g_cpptlm_bridge = &stub`（**~20 行**），包括 StubBridge 类体（5 虚方法，无外部依赖）
  3. **auto-advance in cudaDeviceSynchronize**（bridge path polling loop 前）：若 `g_ptx_emu_driver_shim != nullptr`，先 `advance(UINT32_MAX, actual)` 驱动 PTX 执行，再 poll_kernel 等待确认（**+4 行**）

- **修改** `src/ptxsim/core/gpu_context.cpp` — `exe_once()` 修复 2-cycle completion bug（**待 Phase 0 调试定位，expected <30 行**），候选方案见 design.md

### 修改（测试）

- **修改** `tests/e2e/cosim/test_cosim_vector_add.cu` — 删除所有 PTX-EMU 专用代码：
  - 删除 `MockBridge` 类定义（~40 行）
  - 删除 `cpptlm_attach_bridge(&mock)` 及相关断言（~8 行）
  - 删除 `g_ptx_emu_driver_shim->advance()` 及相关断言（~6 行）
  - **保留**：Kernel 定义 + `cudaDeviceSynchronize` + golden compare（**标准 CUDA 程序**）
  - 最终测试 ~100 LOC（从 ~170 LOC 精简到 ~100 LOC）

### 修改（构建）

- **修改** `tests/e2e/CMakeLists.txt`：移除 `BUILD_LIB_CPPTLM_CUDART` 条件编译——测试变为通用 E2E test，不分 bridge 模式
- **修改** `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h`：移除 `extern PtxEmuDriverShim* g_ptx_emu_driver_shim;`（不再需要测试直接访问）
- **修改** `src/cudart/cudart_sim.cpp:137`：恢复 `static PtxEmuDriverShim* g_ptx_emu_driver_shim`（恢复原声明）

### 影响

| 文件 | 类型 | LOC | 说明 |
|------|------|:---:|------|
| `src/cudart/stub_bridge.h` | 新增 | +30 | StubBridge 类体（5 虚方法 + submitted_ids_ 追踪） |
| `src/cudart/cudart_sim.cpp` | 修改 | +60/-5 | count_kernel_args fix + StubBridge + auto-advance + override flag + ceiling |
| `src/ptxsim/core/gpu_context.cpp` | 修改 | <+30 | 2-cycle completion bug fix（待调试） |
| `tests/e2e/cosim/test_cosim_vector_add.cu` | 修改 | -70/+0 | 删除 MockBridge + advance 代码 |
| `tests/e2e/CMakeLists.txt` | 修改 | -10/+1 | 移除条件编译 |
| `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` | 修改 | -5 | 恢复无 extern |
| `src/cudart/cudart_sim.cpp:137` | 修改 | ±1 | 恢复 static |
| **合计** | | **~+85 / -70** | 净增 ~15 LOC |

## Capabilities

### New Capabilities

- `auto-co-simulation`: 标准 CUDA 程序在 `BUILD_LIB_CPPTLM_CUDART=ON` 下自动走 CppTLM bridge 路径协同仿真，无需任何 PTX-EMU 专用代码（`cpptlm_attach_bridge`、`advance()`、`MockBridge`）

## Impact

| 维度 | 详情 |
|------|------|
| ABI | 不修改 `cpptlm_bridge.h` |
| Bridge attach | `g_cpptlm_bridge` 改为由 `initialize_environment()` 自动设置（`BUILD_LIB_CPPTLM_CUDART=ON` 时内部分配 StubBridge） |
| BACKWARD | 测试改为标准 CUDA 程序，删除 `BUILD_LIB_CPPTLM_CUDART` 条件编译（无桥接模式下 `g_cpptlm_bridge == nullptr` 行为字节级兼容） |
| 回归风险 | `gpu_context.cpp` 修改涉及 `exe_once` 核心流程，需充分多 warp 回归验证 |
| 已知限制 | follow-up `fix-bridge-path-2-cycle-exit` 和 `fix-bridge-arg-count-segfault` 合并到本 change 实施 |

## Design-Time Checklist

- [ ] 1. 无函数迁移（纯 bug fix + 新增 StubBridge 类）
- [ ] 2. 跨模块状态严格审查（`cudaDeviceSynchronize` auto-advance 改变了 bridge 路径的执行流程——原仅 poll，现 also drive PTX execution）
- [ ] 3. 多 Phase 推进（Phase 0: 调试定位 2-cycle bug；Phase 1: segfault fix + StubBridge；Phase 2: 2-cycle fix；Phase 3: 测试清理）
- [ ] 4. 引用 ADR-0021 §2026-07-20 Postmortem L1+L2
- [ ] 5. Oracle 调试协助 Phase 0（根因定位）

## Parent

- `e2e-cosim-kernel-verify`（archived 2026-07-20）
- `cpptlm-p1-ptxemu-shim`（archived 2026-07-19）

## Refs

- [ADR-0021 §2026-07-20 Postmortem L1+L2](docs/adr/0021-cpptlm-d1-full-integration.md)
- [`openspec/changes/archive/2026-07-20-e2e-cosim-kernel-verify/`](openspec/changes/archive/2026-07-20-e2e-cosim-kernel-verify/)
- Oracle 审查 sessions `ses_0812c1edfffekcd3iB4cDf1xKN` + `ses_08120fb5fffexGNRjWWupXHw58`
- 相关 skills：`ptx-debug`、`ptx-instruction-pipeline`、`ptx-barrier-mechanism`、`ptx-lessons-learned`、`regression-bisect`、`state-modification-audit`