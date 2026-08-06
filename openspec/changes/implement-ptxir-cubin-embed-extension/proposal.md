# implement-ptxir-cubin-embed-extension

## Why

PTX-EMU 当前无法直接加载标准 NVIDIA cubin（cuModuleLoadData 链路），而 PTXIR 仅在内部 pipeline 中可用。本提案依据 ADR-0024（PTXIR-Embedded CUBIN 格式），将 PTXIR section 追加到 cubin 末尾，使 PTX-EMU 既能执行标准 cubin 又保留 PTXIR 快速加载优势，同时不破坏 NVIDIA 工具链兼容性。

触发事件：`ptxir-format-compliance` 提案 2026-08-01 被拒绝（与 ADR-0023 7 决策不完全一致），但 cubin + PTXIR 兼容路径缺失需填补 — 本提案填补该缺口，遵循 ADR-0024 已审批的设计。

## What Changes

- **新增** `src/cudart/ptxir_loader.{h,cpp}` — PTXIRLoader 类，提供 magic 检测 / PTXIR 提取 / 纯 cubin 提取 / 反序列化四个 static 方法
- **新增** `include/cudart/ptxir_loader.h` — PTXIRLoader 公开 API
- **新增** `tools/ptxir_extract.cpp` — CLI 工具，从 embedded cubin 提取纯 cubin + PTXIR section
- **新增** `tools/ptxir_embed.cpp` — CLI 工具，将 PTXIR 追加到 cubin 末尾生成 embedded cubin
- **修改** `src/cudart/cudart_sim.cpp` — `__cudaRegisterFatBinary` 增加 PTXIR 检测分支（约 +30-40 行）
- **新增** `config::isPTXIRModeEnabled()` — 读取 `PTXIR_MODE` env var + `configs/*.ini`（MUST 级，loader dispatch 依赖）
- **新增** `PTXIR_EMBED_MAGIC` 8 字节 magic 后缀（loader O(1) 末尾检测，magic 字面值变更触发 ADR-0024 重新审视 — governance check）
- **修改** `src/cudart/CMakeLists.txt` — 注册 `ptxir_loader.cpp` 子目标
- **修改** `tools/CMakeLists.txt` — 注册两个工具目标
- **新增** `tests/unit/test_ptxir_loader.cpp` — 4 个 public static 方法全覆盖（覆盖率 ≥ 90%）
- **新增** `tests/integration/test_ptxir_cubin_loader.cpp` — `__cudaRegisterFatBinary` dispatch 全场景（≥5 场景）
- **新增** `tests/e2e/test_ptxir_cubin_embed.cu` — nvcc + ptxir_embed + PTX-EMU + ptxir_extract → cuobjdump 双向验证（≥5 真实 kernel）

## Capabilities

### New Capabilities

- `ptxir-cubin-embed`: PTXIR-Embedded CUBIN 二进制格式与 PTXIRLoader 类。定义 magic 字节、Section TOC 中 `cubin_hash` 字段约束、`hasEmbeddedPTXIR()`/`extractPTXIR()`/`extractPureCubin()`/`deserializeForCubin()` API 契约、`config::isPTXIRModeEnabled()` 行为契约、loader dispatch 行为（`PTXIR_MODE=auto`/`off` 两路）。
- `ptxir-cubin-tools`: ptxir_extract / ptxir_embed CLI 工具契约。定义 CLI 参数（`--in-cubin`/`--in-ptxir`/`--in`/`--out`/`--out-cubin`/`--out-ptxir`）、退出码、字节级等价性保证（提取后纯 cubin 与原始 cubin hash 相等）、`--help` 与 `--version` 输出格式。

### Modified Capabilities

_None._ ANTLR 解析路径、NVIDIA cubin 格式前缀、GPU registry / WarpContext / ThreadContext 核心执行路径均不变（`PTXIR_MODE=off` 时行为完全等价于现状）。

## Impact

- **受影响代码**：
  - `src/cudart/cudart_sim.cpp` — `__cudaRegisterFatBinary` 增加 dispatch 分支（**ABI 不变**，仅新增分支）
  - `src/cudart/CMakeLists.txt` — 注册新子目标
  - `tools/CMakeLists.txt` — 注册两个工具目标
  - `configs/` — 新增 `PTXIR_MODE` INI 字段
- **新增依赖**：无（PTXIRLoader 复用 ADR-0023 Section TOC 与 PTXIRHeader 格式）
- **ABI 影响**：**无破坏性变更**。`__cudaRegisterFatBinary` 签名不变，新 dispatch 分支由 `config::isPTXIRModeEnabled()` 控制（默认 OFF 完全等价现状）
- **治理约束**：`PTXIR_EMBED_MAGIC` 字面值变更必须触发 ADR-0024 重新审视（不可在 proposal 层面单方面修改 — governance check）
- **文档影响**：
  - `docs/adr/ADR-0024-ptxir-embedded-cubin-format.md` §合规检查 6 项全部通过
  - `tools/README.md` 新增（说明用法与限制）