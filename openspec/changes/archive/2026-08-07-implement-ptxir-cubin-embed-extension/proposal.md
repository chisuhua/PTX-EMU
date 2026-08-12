# implement-ptxir-cubin-embed-extension

## Why

PTX-EMU 当前架构 (`__cudaRegisterFatBinary` @ `src/cudart/cudart_sim.cpp:354-386`) 不接收 cubin 字节 — `fat_bin` 参数仅出现在 debug print (line 372)，未解引用。实际执行 payload 来自 `/proc/self/exe` 经外部 `cuobjdump` 子进程提取的 PTX 文本。本提案依据 ADR-0024 v1.1（2026-08-07 amendment），将 PTXIR section 追加到最终可执行文件末尾（ELF 容忍尾部 overlay data），使 PTX-EMU 能从 embed 段反序列化 PTXIR 并复用 `set_ptx_context()` 主路径，同时保留 NVIDIA 工具链兼容性（cub level 工具独立支持）。

触发事件：`ptxir-format-compliance` 提案 2026-08-01 被拒绝（与 ADR-0023 7 决策不完全一致），但 cubin + PTXIR 兼容路径缺失需填补 — 本提案填补该缺口，遵循 ADR-0024 v1.1 已审批的设计（含 footer layout + magic literal 变更 + PtxContextAdapter 引入）。

## What Changes

- **新增** `src/cudart/ptxir_loader.{h,cpp}` — PTXIRLoader 类，提供 footer-layout magic 检测 / PTXIR 提取 / 纯 cubin 提取 / 反序列化四个 static 方法
- **新增** `src/cudart/ptx_context_adapter.{h,cpp}` — `PtxContextAdapter::fromEmbedded(StatementContext[], EmbeddedKernelManifest)` + `EmbeddedKernelManifest` 结构（kernelName 来源 CLI flag，params 从 manifest，addressSize 默认 64）
- **新增** `src/cudart/ptxir_config.{h,cpp}` — `config::isPTXIRModeEnabled()` 函数（env var `PTXIR_MODE` 静态缓存 + `[ptxir]` INI section 加载，env 覆盖 INI 遵循 `PTX_EMU_GPU_CONFIG` 模式 cudart_sim.cpp:277-281）
- **新增** `include/cudart/ptxir_loader.h` / `ptx_context_adapter.h` / `ptxir_config.h` — 公开 API
- **新增** `tools/` 目录（当前不存在）+ `tools/CMakeLists.txt` — 注册 CLI 工具目标，并在根 `CMakeLists.txt` 添加 `add_subdirectory(tools)`
- **新增** `tools/ptxir_extract.cpp` — CLI 工具，从 embedded exe/cubin 提取纯 cubin + PTXIR section
- **新增** `tools/ptxir_embed.cpp` — CLI 工具，将 PTXIR 追加到 exe/cubin 末尾生成 embedded payload（支持 `--in-exe` 与 `--in-cubin` 两种 target，必填 `--kernel-name`）
- **修改** `src/cudart/cudart_sim.cpp` — `__cudaRegisterFatBinary` 在 `readlink("/proc/self/exe")` (line 377) 之后立即增加 PTXIR dispatch（约 +30-40 行；**byte source = `/proc/self/exe` 末尾 12 字节**，非 `fat_bin`）
- **新增** `PTXIR_EMBED_MAGIC` 8 字节 magic 后缀 = `{'P','T','X','E','M','B','\x01','\x00'}`（loader O(1) 末尾检测；2026-08-07 ADR-0024 amendment 变更；触发 §合规检查 #6 governance check 已解决）
- **修改** `configs/*.ini` — 新增 `[ptxir] mode=off` 段（默认值 = off 保证行为字节级兼容现状）
- **新增** `tests/integration/cudart/` 目录 + `CMakeLists.txt`（新集成测试子目录）
- **新增** `tests/unit/cudart/test_ptxir_config.cpp` — `config::isPTXIRModeEnabled()` 双源配置测试
- **新增** `tests/unit/cudart/test_ptxir_loader.cpp` — PTXIRLoader 4 方法单元测试（覆盖率 ≥ 90%）
- **新增** `tests/unit/cudart/test_ptx_context_adapter.cpp` — PtxContextAdapter 含 kernelName/params/addressSize 字段填充验证
- **新增** `tests/integration/test_ptxir_cubin_loader.cpp` — `__cudaRegisterFatBinary` dispatch 集成测试（≥ 5 场景）
- **新增** `tests/e2e/test_ptxir_cubin_embed.cu` — nvcc + ptxir_embed + PTX-EMU 加载 + ptxir_extract → cuobjdump 双向验证（≥ 5 真实 kernel，含 Oracle review 新增 2 个直接对 embedded 解析场景）
  <br>**[勘误: 2026-08-12, see fix-path-coverage-gaps]** — 原 archived proposal line 26 声称此 e2e 文件后缀为 `.cu`，**实际交付为 `.cpp`**（silent descoping 证据，见 `.rddf/improvements/fix-path-coverage-gaps.md` §3 Oracle review 真实阻断历史）。本 e2e 验证 PTXIR-Embedded CUBIN 格式兼容性（Phase 12.2 R5 / ADR-0024 Risk 1），**不验证 PTX-EMU 真实加载执行**。真实加载执行验证见 `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`（D-PTX-7，关闭于 `docs/audits/D-PTX-debt-registry-ERRATA.md`）。
- **修改** `roadmap.md` — 新增 Phase 12.2 (PTXIR Cubin 集成) 条目

## Capabilities

### New Capabilities

- `ptxir-cubin-embed`: PTXIR-Embedded CUBIN/EXE 二进制格式与 PTXIRLoader 类。定义 footer-layout magic 字节（`PTXEMB\x01\x00`）、Section TOC 中 `cubin_hash` 字段约束、`hasEmbeddedPTXIR()`/`extractPTXIR()`/`extractPureCubin()`/`deserializeForCubin()` API 契约、`config::isPTXIRModeEnabled()` 行为契约（env var 优先于 INI）、loader dispatch 行为（`PTXIR_MODE=auto`/`off` 两路）、**PtxContextAdapter** 契约（`fromEmbedded(stmts, manifest) → PtxContext` 必须填充 `kernelName`/`kernelParams`/`ptxAddressSize`）。
- `ptxir-cubin-tools`: ptxir_extract / ptxir_embed CLI 工具契约。定义 CLI 参数（`--in-exe`/`--in-cubin` 二选一 + `--in-ptxir` + `--out` + `--kernel-name` 必填 / `--out-cubin` / `--out-ptxir`）、退出码、字节级等价性保证（提取后纯 cubin 与原始 cubin hash 相等）、`--help` 与 `--version` 输出格式。

### Modified Capabilities

_None._ ANTLR 解析路径、NVIDIA cubin 格式前缀、GPU registry / WarpContext / ThreadContext 核心执行路径均不变（`PTXIR_MODE=off` 时行为完全等价于现状）。

## Impact

- **受影响代码**：
  - `src/cudart/cudart_sim.cpp` — `__cudaRegisterFatBinary` 在 `readlink` + `cuobjdump` 之间增加 dispatch（**ABI 不变**：4 参签名 `void** __cudaRegisterFatBinary(void**, void*, unsigned long long, unsigned int)` 保持，仅新增分支）
  - `configs/config.ini` 等 — 新增 `[ptxir]` 段
  - `roadmap.md` — 新增 Phase 12.2
- **新增依赖**：无（PTXIRLoader 复用 ADR-0023 Section TOC + PTXIRHeader 格式）
- **ABI 影响**：**无破坏性变更**。`__cudaRegisterFatBinary` 4 参签名不变，新 dispatch 分支由 `config::isPTXIRModeEnabled()` 控制（默认 OFF 完全等价现状）
- **治理约束**：`PTXIR_EMBED_MAGIC` 字面值变更触发 ADR-0024 §合规检查 #6（已 2026-08-07 通过 amendment 解决）
- **byte source 架构决策**：dispatch 读取 `/proc/self/exe` 末尾 12 字节（非 `fat_bin` 参数 — 该参数在当前架构中为 dead parameter，仅 debug print @ cudart_sim.cpp:372）
- **文档影响**：
  - `docs/adr/ADR-0024-ptxir-cubin-embed-extension.md` v1.1 amendment（已 2026-08-07 提交）
  - `tools/README.md` 新增（说明用法与限制）
  - `roadmap.md` 新增 Phase 12.2
  - 根 `README.md` §已实现功能 / §已知限制 同步更新（参考 ptx-lessons-learned §8）