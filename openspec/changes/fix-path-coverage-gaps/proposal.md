## Why

PTX-EMU 当前 4 条 cudart 加载路径 (Path 1A Legacy / 1B PTXIR fat-binary / 1C Driver API / 2D Image Executor) 中，3 条路径的端到端覆盖存在结构性缺口：Path 1B 仅验证 NVIDIA cuobjdump 格式兼容性，Path 1C 仅有错误路径覆盖，Path 2D 仅验证 rc==0 而非 output correctness。这些缺口在 `multi-entry-handle-api`、`ptxir-driver-api-front-door` 等重构持续推进的背景下，已构成 `tests/e2e/` 隐含技术债。下一步 Blackwell tcgen05 重构、跨仓 HAL extension 等都依赖 PTXIR 路径稳定，必须补齐 4-path e2e。

## What Changes

- **新建 5 Phase 工作**：Path 1B PTXIR fat-binary 真实 e2e（fork+exec standalone binary）、Path 1C Driver API 真实 kernel 执行、Path 2D Image Executor 输出正确性 baseline、`tests/e2e/` 重组织为路径化目录、归档 change `implement-ptxir-cubin-embed-extension` proposal 文档一致性修正
- **新增 4 个 `tests/e2e/path_X/` 子目录**，每个含独立 `CMakeLists.txt`（新模式），按 ctest label `e2e;path_1X;...` 隔离
- **新增 golden output baseline**：`tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin`（8-byte `PTXR_OUT\0\0` magic + 4-byte LE size + bytes），用于 Phase 3 output-correctness 验证
- **修改归档 change 文档**：`openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` §Capabilities 补 disclaimer，明确 `test_ptxir_cubin_embed` 仅验证格式兼容性，不验证 PTX-EMU 真实加载执行
- **不修改生产代码**：仅在 `tests/e2e/` 子树工作，不动 `src/cudart/`、`include/cudart/`、`cpptlm_module.cpp`、`ptxir_loader.cpp` 等

## Capabilities

### New Capabilities

- `e2e-ptxir-fatbinary-exec`: PTXIR fat-binary 真实端到端执行测试（Path 1B）—— 覆盖 4 个 dispatch 状态 + 字节级一致性 + anti-fallback guard
- `e2e-driver-api-exec`: Driver API 真实 kernel 执行测试（Path 1C）—— 覆盖 cuModule* 完整流程 + duplicate handle + not-found error + cuLaunchKernel 错误路径
- `e2e-image-executor-output-correctness`: Image Executor output baseline 验证（Path 2D）—— cute_rmsnorm output 与 baseline 字节级一致 + D3 mutation 回归 + ABI baseline
- `e2e-path-organized`: `tests/e2e/` 重组织 —— 4 个 path_X/ 子目录独立构建、ctest label 路径过滤

### Modified Capabilities

（无现有 capability 改动 —— 本改进仅新建测试能力，不改任何 spec 级行为）

## Impact

| 组件 | 影响类型 | 说明 |
|------|----------|------|
| `tests/e2e/` | 重组织 | 4 个 path_X/ 子目录新建，部分 e2e 测试 `git mv` |
| `tests/ptxir/baselines/` | 新建 | `cute_rmsnorm_output_baseline.bin` baseline 文件 |
| `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` | 文档修正 | 仅 §Capabilities 文案补全 disclaimer |
| `tests/CMakeLists.txt` | 扩展 | 新增 4 个 `add_subdirectory(path_X/)` 调用 + ctest label 段加 `e2e` |
| `.gitignore` | 微调 | 新增 `!tests/e2e/path_X/**` 白名单避免 `.ptx` 全局 ignore 误命中 |
| 生产代码 | 不动 | `cudart_sim.cpp`、`cpptlm_module.cpp`、`ptxir_loader.cpp` 等不修改 |
| `tests/unit/`、`tests/integration/` | 不动 | 仅修改 `tests/e2e/` 子树 |