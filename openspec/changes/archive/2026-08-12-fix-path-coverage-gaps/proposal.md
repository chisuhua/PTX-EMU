## Why

PTX-EMU 当前 4 条 cudart 加载路径 (Path 1A Legacy / 1B PTXIR fat-binary / 1C Driver API / 2D Image Executor) 中，3 条路径的端到端覆盖存在结构性缺口：Path 1B 仅验证 NVIDIA cuobjdump 格式兼容性，Path 1C 仅有错误路径覆盖，Path 2D 仅验证 rc==0 而非 output correctness。这些缺口在 `multi-entry-handle-api`、`ptxir-driver-api-front-door` 等重构持续推进的背景下，已构成 `tests/e2e/` 隐含技术债。下一步 Blackwell tcgen05 重构、跨仓 HAL extension 等都依赖 PTXIR 路径稳定，必须补齐 4-path e2e。

**架构依据**：
- **ADR-0024**（PTXIR-Embedded CUBIN, 2026-08-06 Accepted）— Risk 1: NVIDIA cuobjdump 必须容忍尾部 PTXIR。test_ptxir_cubin_embed.cpp 验证 Risk 1 成立，但**未验证 PTX-EMU 真的能加载并执行**该 embedded binary
- **ADR-0029**（PTX-EMU Image Executor, 2026-08-10 ship）— D6: SINGLE-GPU-INSTANCE 假设。test_cpptlm_module.cpp 仅验证 API 调用成功，未验证 RMSNorm 输出正确
- **ADR-0021**（CppTLM D1-Full MemoryBridge, 2026-07-17 归档）— D-PTX-1 至 D-PTX-6 债务编号体系，本改进新增 D-PTX-7（PTXIR fat-binary 端到端未验证）、D-PTX-8（Driver API 真实成功 kernel 执行未验证）

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

## In Scope

- Phase 1: 新建 `tests/e2e/path_1B_ptxir_fatbinary/` 子目录 + `test_ptxir_fatbinary_exec.cpp` e2e 测试（含 5 个 scenario：kSuccess / kNoFooter / kMalformedPtxir / kMalformedManifest / Path 1B vs 1A 字节级一致）+ `path_1B_kernels.cu` (≥3 kernels)
- Phase 2: 新建 `tests/e2e/path_1C_driver_api/` 子目录 + `test_cuda_driver_exec.cpp` e2e 测试（含 4 个 scenario：cuModule* 完整流程 / duplicate handle / not-found error / cuLaunchKernel 错误路径）+ v2 manifest PTXIR fixture
- Phase 3: 新建 `tests/e2e/path_2D_image_executor/` 子目录 + 增强 `tests/integration/cudart/test_libptxemu_device.cpp` + 生成 `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` golden baseline（含 8-byte `PTXR_OUT\0\0` magic + 4-byte LE size header）+ `baseline_format.md` 文档
- Phase 4: `git mv` 现存 e2e 测试到 4 个 path_X/ 子目录 + 4 个独立 `path_X/CMakeLists.txt` + 修改 `tests/e2e/CMakeLists.txt`（add_subdirectory + ctest labels）
- Phase 5: 修改 `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` §Capabilities 文案，添加 `[修正: 2026-08-12, see fix-path-coverage-gaps]` disclaimer

## Out of Scope

- **不修改 Path 1A/1B/1C/2D 的实现代码** — 仅补测试，不动 `cudart_sim.cpp` / `cpptlm_module.cpp` / `ptxir_loader.cpp` 等生产代码
- **不修复 `multi-entry-handle-api` 任务未勾选状态** — 这是 archive gate 的 process gap，需另立 `archive-gate-incomplete-tasks` improvement
- **不引入新测试框架** — 沿用 Catch2 + `add_catch_test`（commit ab55e06 约定）
- **不创建新的 PTXIR fixture 生成工具** — Phase 1 用现有 nvcc 编译简单 kernels；Phase 3 用现有 `cute_rmsnorm.ptxir` fixture（5294 B）
- **不修改 openspec CLI / openspec validate 规则** — 测试 failure 不应被 openspec 误判
- **不修复 Path 1A 现有 e2e 的 SingletonGuard 问题** — 现有 `test_divergence.cu` 已 inline kernel 避免该问题，足够覆盖
- **不做 Blackwell tcgen05 路径 1B/1C 集成** — 现有 `test_tcgen05_*.cu` 已走 Path 1A 间接覆盖；Phase 4 仅把它们移到 `path_1A/` 子目录但保留 Path 1A 守护
- **不修改 ctest 标签体系** — 仅添加新 label（`e2e;path_1X`），现有 LABELS 不变
- **不动 PTX-EMU 整体测试目录结构** — 仅修改 `tests/e2e/` 子树，不动 `tests/unit/` 或 `tests/integration/`
- **不引入新 dispatch marker ABI** — Phase 1 anti-fallback guard 仅用 `PATH=""`，不暴露 `ptxemu_ptxir_dispatch_hits()` extern "C" 符号（避免动 cpptlm_module.cpp 现有 8 个 extern "C" 符号）