## Why

src/cudart/cudart_sim.cpp 是 CUDA runtime 拦截层的核心入口（cudaLaunchKernel / __cudaRegisterFatBinary / memory / stream / event API），当前 1255 行、零直接单元测试。自 debt-audit-2026-07-02 以来已膨胀 +322 行（+34.5%）。所有 E2E 测试通过此层执行，任何回归都是静默的——只有 kernel 级测试才能发现，定位成本极高。对应 debt-audit-2026-07-02.md P0-C2。

## What Changes

- **新建 tests/unit/cudart/ 目录**: 为 cudart_sim.cpp 提供直接单元测试
- **Phase 1 — Memory API 测试**: cudaMalloc/cudaFree/cudaMemcpy/cudaMemset 的最小测试覆盖
- **Phase 2 — Stream API 测试**: cudaStreamCreate/cudaStreamSynchronize 的基础路径
- **测试夹具**: 最小 GPUContext mock 辅助函数（不引入新依赖）
- 不修改 cudart_sim.cpp 行为 — 仅补充测试

## Capabilities

### New Capabilities
- `cudart-unit-test`: 为 CUDA runtime 拦截层提供直接单元测试，覆盖 Memory 和 Stream API 关键路径

### Modified Capabilities
<!-- none -->

## Impact

- 新增: tests/unit/cudart/CMakeLists.txt + test_cudart_memory.cpp + test_cudart_stream.cpp + cudart_test_helpers.h
- 修改: tests/unit/CMakeLists.txt（+1 add_subdirectory）
- 代码: 无生产代码变更（可能需将 cudart_sim.cpp 的部分 static 函数提取为可测试的 internal header，不改变行为）
- 测试: ctests: unit_cudart_memory, unit_cudart_stream
- 风险: 低 — 仅添加测试，不改生产逻辑
- 相关: debt-audit-2026-07-02.md §P0-C2, ADR-0005 (MemoryRegion)