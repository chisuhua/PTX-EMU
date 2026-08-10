# ptxir-cubin-cleanup

> **Phase**: 12.2 收尾
> **优先级**: � P0（PTXIR 工具链从"工具做出"到"工具跑通"的关键一步）
> **状态**: Active
> **创建日期**: 2026-08-10
> **关联**: [roadmap.md](../../../../roadmap.md) §Phase 12.2 + [archive/2026-08-07-implement-ptxir-cubin-embed-extension/](../../archive/2026-08-07-implement-ptxir-cubin-embed-extension/) + ADR-0024 v1.1 + ADR-0026

## Why

ADR-0024 v1.1（PTXIR-Embedded CUBIN 格式 + footer-layout detection + `PTXIR_EMBED_MAGIC`）已于 2026-08-07 governance check 通过；`ptxir_embed` / `ptxir_extract` CLI 工具已实现并构建（`build/bin/`）；`PTXIRLoader` + `PtxContextAdapter` + `config::isPTXIRModeEnabled` 已 ship。

**但是工具链是"做出来了但跑不通"**：legacy front door 的 `__cudaRegisterFatBinary`（`libcudart.so` T 符号）**仍未实现 PTXIR dispatch 分支**。结果是 `ptxir_embed` 能生成嵌入 PTXIR 的 binary，但运行时不会走 PTXIR 路径——直接走 cuobjdump（或 PTXIR 路径完全失效），用户感知不到 PTXIR-Embedded CUBIN 工具链的端到端价值。

具体缺口（per 2026-08-10 implementation status audit）：
- archive change `2026-08-07-implement-ptxir-cubin-embed-extension/tasks.md` commit 1-4 大量 `[ ]` 未完成
- `__cudaRegisterFatBinary` 入口未调用 `PTXIRLoader::hasEmbeddedPTXIR` + `extractPTXIR` + `deserializeForCubin`
- INI `[ptxir] mode = off` 段未集成到 `initialize_environment()`
- integration / e2e tests 缺失覆盖 legacy front door PTXIR 分支

## What Changes

**In Scope**（按 TDD 5 步结构）：
- **R1**: 补齐 `PTXIRLoader::extractPureCubin` 测试覆盖（实现已在 `src/cudart/ptxir_loader.cpp:79`）
- **R2**: INI `[ptxir] mode = off` 段集成到 `initialize_environment()`（per archive tasks 1.6.7 + 1.7.x）
- **R3**（核心）：`__cudaRegisterFatBinary` PTXIR dispatch 分支真实实现（legacy front door 的 PTXIR 路径）
- **R4**: integration tests ≥5 场景（含 PTXIR_MODE auto/off 分支）
- **R5**: e2e tests 扩展至 ≥5 真实 CUDA kernel 场景（extend 现有 `tests/e2e/kernel/test_ptxir_cubin_embed.cpp`）
- **R6**: 完整 ctest + sanity.sh 全绿

**Out of Scope**:
- Driver API front door（Phase 12.3.A 范围）
- 新 CLI 工具（Phase 12.3.B/C 范围）
- multi-kernel manifest（Phase 12.4 范围）
- HAL extension（Phase 13 范围）

## Capabilities

- **`__cudaRegisterFatBinary` 入口 PTXIR detection**：`PTXIRLoader::hasEmbeddedPTXIR(data, size)` 检查 `/proc/self/exe` 末尾 footer
- **PTXIR dispatch 分支**：提取 → 反序列化 → `PtxContextAdapter::fromEmbedded()` → 复用现有 `set_ptx_context()` 主路径
- **`PTXIR_MODE` precedence**：env > INI > default（per 架构 §6 precedence matrix）
- **`PTXIR_MODE=off` 完全 bypass**：与现状字节级兼容（CI 守门）
- **malformed fallback**：malformed embedded PTXIR 或 manifest mismatch → 报告错误，**不**静默 fallback
- **缺少 footer fallback**：未发现 footer 时正常 fallback 到 cuobjdump（per 架构 §4.1）

## Impact

- `__cudaRegisterFatBinary` 新增 ~50-80 行 PTXIR dispatch 分支
- `initialize_environment()` 新增 INI `[ptxir]` 段加载 + 优先级处理
- `src/cudart/cudart_sim.cpp` line 12 入口加 `if (config::isPTXIRModeEnabled() && PTXIRLoader::hasEmbeddedPTXIR(...))` 分支
- 完整 PTXIR 工具链从"工具做出"到"工具跑通"

## Acceptance

- [ ] **R1**: `tests/unit/cudart/test_ptxir_loader.cpp` 补齐 extractPureCubin 3 场景 + ctest PASS
- [ ] **R2**: `tests/unit/cudart/test_ptxir_config.cpp` 4 场景 + INI 集成 + ctest PASS
- [ ] **R3**: `__cudaRegisterFatBinary` PTXIR 分支实现 + integration test 验证 dispatch 路径 + ctest PASS
- [ ] **R4**: `tests/integration/test_ptxir_cubin_loader.cpp`（新建）≥5 场景 PASS
- [ ] **R5**: `tests/e2e/kernel/test_ptxir_cubin_embed.cpp`（已存在）扩展 ≥5 场景 PASS
- [ ] **R6**: `cmake --build && ctest --output-on-failure` 全绿 + `./scripts/sanity.sh` 全绿
- [ ] `PTXIR_MODE=off` 行为字节级不变（CI 守门）
- [ ] `nm -D build/lib/libcudart.so` 不减少导出符号（保持 ABI 兼容）

## 关键约束（per `improvements/implement-ptxir-cubin-embed-extension.md` MUST）

- 复用 ADR-0023 的 Section TOC + PTXIRHeader 格式（不重新发明二进制格式）
- `__cudaRegisterFatBinary` ABI 不变，仅添加 dispatch 分支
- `PTXIR_MODE` 环境变量可完全 bypass 检测分支（默认 OFF，行为等价于当前）
- `Section TOC` 中显式嵌入 `cubin_hash` 字段，loader 校验一致性
- 所有失败路径返回 `nullopt` / 空 vector，不抛异常

## Commit 拆分（per `ptx-lessons-learned` §3）

- Commit 1 (R1): extractPureCubin 测试覆盖补齐
- Commit 2 (R2): INI 集成
- Commit 3 (R3): PTXIR dispatch 分支（核心）
- Commit 4 (R4): integration tests
- Commit 5 (R5): e2e tests
- Commit 6 (R6): 验证 + 文档同步

每个 commit 独立可回退，失败立即 revert。
