# Design: multi-kernel-manifest-adr-0028

## 现状问题

`include/ptx_ir/ptxir_format.h::ManifestSection`（line 36-41）只有单 `kernel_name` 字段，导致 3 个已 ship ADR 同时受 v1 单 kernel 限制拖累：
- ADR-0025 (`ptxir_build` CLI) — wrapper 拒绝 multi-entry PTX
- ADR-0027 (`ptx-nvcc` wrapper) — 同样限制
- ADR-0029 (image executor) D4 — `libptxemu_device.so` 的 `ptxemu_image_kernel_name` 只返回首个

架构 §11 明示 **ADR-0028 是 BLOCKING DEPENDENCY**，状态从"预留占位"于 2026-08-09 升级。

## 目标状态

新建 ADR-0028 + bump `PTXIR_VERSION`（per ADR-0023 Extend-Only），扩展 `ManifestSection` 为 `vector<kernel_entry>`，解除 ADR-0025/0027/0029 §v1 单 kernel 限制。向后兼容：旧 v1 单 kernel binary 仍可在新 runtime 下加载（reader 把单 entry 视为 `vector` 长度 1 的特例）。

## 影响范围

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `docs/adr/ADR-0028-multi-kernel-manifest.md` | 新建（Oracle C1 先建） | 设计决策 + Extend-Only 合规说明 |
| `include/ptx_ir/ptxir_format.h:36-41` | 修改 | `ManifestSection` 扩展为 `vector<kernel_entry>` |
| `include/ptx_ir/ptxir_format.h` | 修改 | `PTXIR_VERSION` bump (per ADR-0023) |
| `src/cudart/ptxir_loader.cpp` | 修改 | `deserializeForCubin()` 返回 `vector<kernel_entry>` |
| `src/cudart/cudart_sim.cpp` | 修改 | `__cudaRegisterFatBinary` + `cuModuleGetFunction` 多 kernel 名查询 |
| `src/cudart/cpptlm_module.cpp` | 修改 | `PtxEmuImageExecutor::load_image` 多 entry handle |
| `tools/ptxir_build.cpp` 等 3 工具 | 修改 | 多 kernel 支持 |
| `docs/adr/ADR-0025/0027/0029` | 修改 | §v1 限制段落更新 |
| `docs/architecture/ptxir-toolchain-stack.md` | 修改 | v1.3 → v1.4，§11 BLOCKING 标记移除（Oracle C4 changelog） |
| `docs/adr/README.md` | 修改 | 新增 ADR-0028 条目 |
| `cpptlm_bridge.h` | **不变** | 与 Phase 12.3.A 共享约束 |
| `libptxemu_device.so` ABI | **不变**（除可选新增 API） | ADR-0029 D7 |

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 旧 v1 binary 不可读 | 中 | reader 容错把单 `kernel_name` 视为 `vector` 长度 1（per archive change `2026-08-07` 约束） |
| silent failure | 中 | 所有失败路径返回明确错误码 |
| `PTXIR_VERSION` 不 bump 即改 schema | 低 | Oracle C1 + ADR-0023 §决策 6 强制：必须 bump |
| 阻塞 Phase 12.3.A（`deserializeForCubin` 签名冲突） | 中 | **Oracle C2 硬串行**：本 change 必须在 Phase 12.3.A 完成后启动；建立 `task_id` 依赖 |
| v1 fixture 不可用 | 中 | 用 `tests/ptxir/fixtures/cute_rmsnorm.ptxir` 做 runtime 回归测试 |

## 关键约束 (MUST)

- 复用 `PTXIRLoader::deserializeForCubin()` 扩展（不改 ANTLR 解析路径）
- bump `PTXIR_VERSION` 才扩展 schema
- 保持 v1 binary 字节级可读
- 继承现有 section TOC 布局
- 不为 silent failure
- 实施前必须先 amend `ptxir-toolchain-stack.md` v1.4

## 测试策略

按 `ptx-lessons-learned` §3，分 Phase commit：
- Commit 1: ADR-0028 + adr/README 同步
- Commit 2: `ptxir_format.h` 多 entry 扩展 + bump version
- Commit 3: `PTXIRLoader` + `PtxEmuImageExecutor` 多 entry 支持
- Commit 4: `__cudaRegisterFatBinary` / `cuModuleGetFunction` 多 kernel 名查询
- Commit 5: tools/ 多 kernel + tests
- Commit 6: ADR-0025/0027/0029 §v1 段落更新 + 架构文档 v1.4