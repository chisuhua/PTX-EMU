# implement-ptxir-cubin-embed-extension — Design

## Context

### 背景

PTX-EMU 当前 (`2026-08`) 接收 CUDA 程序通过以下路径（**经 Oracle 2026-08-07 实地验证**）：
1. **PTX 路径**：`__cudaRegisterFatBinary` (cudart_sim.cpp:354-386) → `readlink("/proc/self/exe")` (line 377) → `extract_ptx_with_cuobjdump(self_exe_path)` (line 386) → ANTLR4 解析 → `set_ptx_context()` → 解释执行

`fat_bin` 参数在当前架构中**未被解引用**（仅出现在 debug print @ line 372），即 PTX-EMU 不接收 cubin 字节。标准 NVIDIA cubin（nvcc 默认产物）通过 ELF 嵌入到可执行文件，PTX-EMU 通过 `cuobjdump --dump-ptx` 提取 PTX 文本。

**Oracle 调研结论**：`ptxir-format-compliance` 提案（2026-08-01 被拒）的 sibling 模型要求 PTX-EMU 调用方显式传 PTXIR 路径，破坏了单文件部署兼容性。本提案通过**将 PTXIR 追加到最终可执行文件末尾**（ELF 容忍尾部 overlay data）解决 — dispatch 在 `readlink` 之后立即读取 `/proc/self/exe` 末尾 12 字节，O(1) 检测 magic。

### 现状（基于代码实测）

- `__cudaRegisterFatBinary` 签名（cudart_sim.cpp:354）：`void **__cudaRegisterFatBinary(void **fatCubinHandle, void *fat_bin, unsigned long long fat_bin_size, unsigned int version)` — **4 参**，返回 `void**`
- `fat_bin` 参数**dead**（仅 debug print @ line 372）
- 当前实现走 PTX 文本提取（`extract_ptx_with_cuobjdump(self_exe_path)` @ line 386），无 PTXIR 检测
- PTXIRLoader 类**不存在**
- `config::isPTXIRModeEnabled()` **不存在**（`config::` 命名空间也不存在，需新建 `src/cudart/ptxir_config.cpp` 适配层）
- `PtxContextAdapter` 类**不存在**（PTXIR deserializer 仅产生 `StatementContext[]`，缺少 `kernelName`/`kernelParams`/`ptxAddressSize` 字段填充 — ptxi_writer.cpp:154-160 + ptxi_serialization.cpp:111 证实）
- `PTXIR_EMBED_MAGIC` **不存在**（定义于 ADR-0024 v1.1 amendment）
- `tools/` 目录**不存在**

### 约束

- **ADR-0024 v1.1 Accepted**（2026-08-07 amendment）：footer layout + magic literal `PTXEMB\x01\x00` + PtxContextAdapter + tools/ 目录新建
- **ADR-0023 已 Accepted**（sibling, 2026-07-30）：PTXIR 二进制格式 + Section TOC 复用约束；本提案在 embed section 中追加 `EmbeddedKernelManifest`（Extend-Only 兼容）
- **`PTXIR_EMBED_MAGIC` 字面值 governance check**：magic 任何变更触发 ADR-0024 §合规检查 #6（已 2026-08-07 通过 amendment 解决）
- **`PTXIR_MODE` 默认 OFF**：保证未启用用户的行为字节级兼容现状
- **byte source 架构决策**：dispatch 读取 `/proc/self/exe` 末尾 12 字节（非 `fat_bin`）

### 利益相关方

- 模拟器核心（`ptxsim`）— 无直接影响（loader dispatch 在 cudart 层）
- CUDA runtime 拦截（`cudart`）— 主要修改点
- 工具链用户（`tools/`）— 新增 CLI（目录需新建）
- ADR 治理委员会 — magic number governance check 守门（2026-08-07 通过）

## Goals / Non-Goals

### Goals

1. PTX-EMU 能加载带 PTXIR 嵌入段的最终可执行文件（通过 `/proc/self/exe` 末尾 overlay）而不破坏 ELF 加载
2. `PTXIR_MODE=auto` 时自动检测嵌入段并走 PTXIR 快速加载路径
3. `PTXIR_MODE=off` 时行为字节级等价现状（默认）
4. 提供 `ptxir_embed` / `ptxir_extract` CLI 工具 — 支持 `--in-exe`（PTX-EMU 加载）与 `--in-cubin`（NVIDIA 工具链兼容）两种 target
5. ADR-0024 §合规检查 6 项全部通过（含 Oracle review 新增的 2 个直接对 embedded cubin 解析场景）
6. PtxContextAdapter 正确填充 `PtxContext` 全部必填字段（`kernelName`/`kernelParams`/`ptxAddressSize`/`ptxStatements`/`externFuncs`），避免 `setupKernelArguments` 静默 `total_param_size=0` 失败

### Non-Goals

- 不修改 NVIDIA cubin 格式前缀（仅追加嵌入段，独立 `--in-cubin` 工具场景支持）
- 不修改 ANTLR 解析路径
- 不在 GPU registry / WarpContext / ThreadContext 添加新依赖
- 不修改 `__cudaRegisterFatBinary` 4 参 ABI 签名
- 不解引用 `fat_bin` 参数（Oracle 已验证其为 dead parameter）
- 不实现 PTXIR Section TOC v1 → v2 升级（独立 change）
- 不支持 N-kernel manifest（v1 显式为 single-kernel 范围，PTXIR v3 单 kernel 序列化器限制 — ptxir_serialization.cpp:97-108）
- 不集成 CppTLM bridge（独立 ADR-0021 范围）

## Decisions

### 决策 1：footer layout (ZIP-EOCD style) + magic literal 变更

**选择**（2026-08-07 ADR-0024 amendment）：
```
[0 .. N)        cubin/exe prefix (verbatim)
[N .. N+M)      ptxir_section (PTXIRHeader + TOC + sections + manifest)
[N+M .. N+M+4)  uint32_le ptxir_section_size (= M)
[N+M+4 .. +8)   PTXIR_EMBED_MAGIC = {'P','T','X','E','M','B','\x01','\x00'}
```
Loader O(1) 检测算法：
1. 读取末尾 8 字节比对 `PTXIR_EMBED_MAGIC`
2. 读取 `end-12` 处 uint32_le `ptxir_section_size`
3. section 起始 = `end - 12 - ptxir_section_size`
4. prefix 起始 = 0, 长度 = section 起始

**理由**：
- **真 O(1)** 检测（仅读末尾 12 字节，无需 ELF/cubin 解析）
- magic `PTXEMB` 与 NVIDIA 已有 magic + PTXIR 文件 magic（4 字节 `PTXI`）不冲突
- footer 长度字段允许 locator 跳回 section 起始，无需 PTXIRHeader 内部字段反推

**备选**（均被拒绝）：
- ❌ **size-before-section**（ADR v1.0 原 layout）：chicken-and-egg — 不知道 M 无法定位 section，需 ELF 解析
- ❌ **size from PTXIRHeader.string_table_offset**：需先找到 section 才能读 header，正是待求解
- ❌ **`{'P','T','X','I','R',...}'` magic**：与 PTXIR 文件 magic 前 4 字节碰撞风险，读者混淆
- ❌ **ELF section 注入**：破坏 NVIDIA 兼容性

**governance check 已通过**：2026-08-07 ADR-0024 §更新记录已记录 magic + layout 变更。

### 决策 2：loader dispatch 顺序 — magic → hash → deserialize

**选择**：`hasEmbeddedPTXIR()` → `extractPTXIR()` → `extractPureCubin()` 校验 `cubin_hash` → `deserializeForCubin()`

**理由**：
- 三道防线：magic 检测避免误判；hash 校验避免静默篡改；deserialize 容错返回 null
- 每个方法独立可测（unit 测试覆盖率 ≥ 90%）
- 失败时统一返回 null，调用方走标准 cubin 路径（优雅降级）

**备选**：
- ❌ 单个 `tryLoad()` 方法：错误处理不清晰，难以 unit 覆盖
- ❌ 异常抛出：与 `config::isPTXIRModeEnabled()=off` 默认行为冲突

### 决策 3：PTXIRLoader 全部用 `public static` 方法（无状态类）

**选择**：PTXIRLoader 不持有任何实例状态，所有方法 `static`。

**理由**：
- 工具（`ptxir_extract` / `ptxir_embed`）直接复用 loader，无需初始化
- unit 测试无需 setup / teardown
- 与项目其他静态工具类风格一致（`PtxUtil`、`PTXIRReader` 等）

**备选**：
- ❌ 单例模式：增加生命周期管理成本，cudart 入口无状态需求
- ❌ 实例方法：测试 setup 成本增加，无状态需求场景冗余

### 决策 4：复用 ADR-0023 Section TOC + PTXIRHeader（不重新发明格式）

**选择**：嵌入段使用 ADR-0023 已定义的 Section TOC 格式，PTXIRHeader 在 Section TOC 中追加 `cubin_hash` 字段 + `EmbeddedKernelManifest` 扩展块（Extend-Only 兼容）。

**理由**：
- ADR-0023 7 决策已通过 sibling 决策固化（ADR-0024 引用）
- PTXIR 反序列化逻辑无需修改（仅添加 `cubin_hash` 字段解析）
- 工具链（`ptx-serializer` 等）无需适配

**备选**：
- ❌ 自定义嵌入格式：违反 ADR-0023 复用约束
- ❌ 修改 Section TOC v1 → v2：超出本 change 范围

### 决策 5：`config::isPTXIRModeEnabled()` 必须实现 + env-var-overrides-INI 优先级

**选择**：新建 `src/cudart/ptxir_config.{h,cpp}`（`config::` 命名空间当前不存在，需新建），复用现有 `inipp::Ini<char>` 解析器（cudart_sim.cpp:247）+ `PTX_EMU_GPU_CONFIG` env-var-overrides-INI 模式（cudart_sim.cpp:277-281）。

**理由**：
- 复用 `inipp` 基础设施，避免重新发明 INI 解析
- env-var-overrides-INI 已是项目既定 idiom（`PTX_EMU_MAX_ADVANCE_CYCLES` @ line 222-232 是更近的 precedent）
- 使用 Meyers singleton `static int cached = []{...}();` 实现 O(1) 后续访问 + 线程安全 lazy init
- `[ptxir]` INI section 显式加载（`g_ini_mode` 静态变量由 `initialize_environment()` 设置）

**备选**：
- ❌ 新建独立 INI 解析器：违反 DRY
- ❌ 仅 env var：与项目双源 idiom 不一致
- ❌ INI 优先于 env var：违反 cudart_sim.cpp:277-281 precedent（env wins）

### 决策 6：PtxContextAdapter 适配 StatementContext[] → PtxContext

**选择**：新增 `src/cudart/ptx_context_adapter.{h,cpp}` 提供 `PtxContextAdapter::fromEmbedded(StatementContext[], EmbeddedKernelManifest) → PtxContext`，其中 `EmbeddedKernelManifest` 结构包含 `kernelName`（来自 CLI `--kernel-name` 必填 flag）、`params`（manifest 或 sidecar）、`ptxAddressSize = 64`（默认）。

**理由**（Oracle 2026-08-07 调研证实）：
- `PtxContext`（ptx_context.h:13-27）含 `ptxKernels[]`、`ptxStatements[]`、`ptxAddressSize` 等字段
- `set_ptx_context` 后续使用要求 `kernelName`（ptx_interpreter.cpp:39-44 kernel 查找）、`kernelParams`（ptx_interpreter.cpp:421-434 `setupKernelArguments`，若 `total_param_size=0` 则 args 静默丢失）+ `ptxAddressSize`（line 384）
- `deserialize_statements` 仅产出 `StatementContext[]`（ptxir_writer.cpp:154-160 + ptxir_serialization.cpp:111 证实不写 kernelName/params/addressSize）
- 直接传入 `StatementContext[]` 会导致 `total_param_size=0` 静默失败（Oracle A3 关键发现）

**备选**：
- ❌ 扩展 `PtxirReader` API 接受 kernelName 参数：污染 ADR-0023 序列化抽象
- ❌ 重构 `set_ptx_context` 接受 `StatementContext[]`：破坏现有调用者（ANTLR path）
- ❌ 在 deserialize 阶段硬编码 kernelName=`<default>`：无法支持多 kernel（虽然 v1 不需要）

### 决策 7：工具链拆分 4 个 commits（参考 `worktree-archive-workflow` v2.0.5+）

**选择**（2026-08-07 修订）：
- Commit 0 (pre-code): ADR-0024 v1.1 amendment（已 2026-08-07 提交）
- Commit 1: PTXIRLoader 类 + PtxContextAdapter + config + unit 测试
- Commit 2: cudart_sim.cpp dispatch 集成 + integration 测试（依赖 C1）
- Commit 3: tools/ 目录 + ptxir_embed + ptxir_extract CLI + e2e 测试（依赖 C1，独立于 C2）
- Commit 4: roadmap.md + 根 README.md 文档同步

**理由**：
- 每个 commit 独立可回退
- Commit 0 单独保证 governance check 通过
- Commit 3 独立于 Commit 2（e2e 测试中 `cuobjdump --dump-sass` 不需要 dispatch；`PTX-EMU executes embedded cubin` 场景需要 Commit 2 — 在 tasks.md §3.4 中明确拆分）
- 与项目分 Phase commit 实践一致（参见 `ptx-lessons-learned` 经验）

**备选**：
- ❌ 3 commits（旧版本）：混淆 ABI 修改与新文件创建，难以独立回退
- ❌ 单 commit：单元/集成测试同时失败时难以定位根因

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| [R1] NVIDIA driver 拒绝尾部 magic（破坏 cuModuleLoadData 兼容性） | e2e 测试必须包含 `cuobjdump --dump-sass kernel.embedded.cubin` + `cuModuleLoadData` 直接验证（Oracle review blocking fix，ADR-0024 §风险 risk 1 / factor 2）；**仅 `--in-cubin` target 涉及此风险** |
| [R2] `PTXIR_EMBED_MAGIC` 字面值冲突 NVIDIA 已有 magic | 已 2026-08-07 改为 `PTXEMB\x01\x00`（与 NVIDIA magic + PTXIR 文件 magic `PTXI` 均不碰撞）；ADR-0024 v1.1 §合规检查 #6 governance check 已通过 |
| [R3] PTXIR section 损坏导致 cudart 崩溃 | deserialize 失败统一返回 null，dispatch 优雅降级到标准 cubin 路径；`deserializeFromString` 抛异常时 try/catch 包裹 |
| [R4] cubin 末尾空间不足，无法追加 magic + section | magic 检测必须在 cubin/exe size 上限内（loader 不假设末尾永远有充足空间，MUST NOT） |
| [R5] `cubin_hash` 不匹配（PTXIR 与 cubin 不对应） | loader 校验失败返回 null，dispatch 走标准 cubin 路径 |
| [R6] `PTXIR_MODE` env var 在 CI / 测试环境未正确传播 | config 双源（env var 优先于 INI，遵循 cudart_sim.cpp:277-281 precedent），CI 显式 `PTXIR_MODE=off` 验证默认行为 |
| [R7] 单元/集成/e2e 三层测试覆盖率不足 | 测试验收标准明确覆盖率 ≥ 90% + ≥ 5 场景 |
| [R8] ANTLR 解析路径意外被修改 | MUST NOT（proposal 级强制），code review 守门 |
| [R9] PtxContextAdapter 字段填充错误导致 kernel launch 静默失败（Oracle A3） | unit 测试覆盖 kernelName/params/addressSize 三字段；integration 测试用 ≥ 2 个 `.param` 的 kernel 验证端到端 |
| [R10] `fat_bin` 参数被错误解引用导致 crash | `__cudaRegisterFatBinary` dispatch **禁止**读取 `fat_bin` 指向的内存（cudart_sim.cpp:372 debug print 也不解引用）；tests 验证 `fat_bin=nullptr` 不 crash |
| [R11] `tools/` 目录缺失导致 Commit 3 无法实施 | tasks.md §3.3 显式列出"新增 `tools/` 目录"作为 Commit 3 第一个 sub-task |

## Migration Plan

### 部署步骤

1. **Commit 0**（已 2026-08-07 提交）：ADR-0024 v1.1 amendment（footer layout + magic literal change）— governance check 通过
2. **Commit 1**：合并 PTXIRLoader + PtxContextAdapter + config + unit 测试 → 默认 `PTXIR_MODE=off`，无运行时影响
3. **Commit 2**：合并 cudart dispatch + integration 测试 → `PTXIR_MODE=auto` 默认 OFF，运行时行为不变
4. **Commit 3**：合并 CLI 工具 + e2e 测试 → 用户主动使用 `PTXIR_MODE=auto` 启用
5. **Commit 4**：roadmap.md + 根 README.md 文档同步

### 回滚策略

- 任意 Commit 失败 → revert 该 commit，前序 commits 仍生效（PTXIRLoader 类保留，下次升级重试）
- 不需要数据迁移（无 schema 变更）
- 不需要配置迁移（新增 INI 字段，默认 OFF）

### 兼容性保证

- `PTXIR_MODE=off`（默认）：行为字节级等价现状
- `PTXIR_MODE=auto`：仅在检测到合法嵌入段时切换路径
- 未设置 `PTXIR_MODE`：等同 `off`

## Open Questions

1. ~~**`PTXIR_EMBED_MAGIC` 字面值**~~：**已 2026-08-07 解决**（变为 `{'P','T','X','E','M','B','\x01','\x00'}`，ADR-0024 v1.1 amendment governance check 通过）
2. **Section TOC 中 `cubin_hash` 字段的 hash 算法**：SHA-256（self-evident；与项目现有 hash 实践一致，无 library 依赖）
3. **PTXIRLoader 错误日志格式**：`[PTXIRLoader]` 前缀 + 严重性等级（`ERROR`/`WARN`/`INFO`），使用 `PTX_DEBUG_EMU`/`PTX_ERROR_EMU` 宏（self-evident；与 cudart AGENTS.md §日志风格一致）
4. ~~**`PTXIR_MODE` 接受的值**~~：仅 `auto`/`off`，`force` 推迟到后续 change（v1 最小集；Oracle 推荐）
5. **e2e 测试中真实 NVIDIA driver 可用性**：若环境无真实 driver，测试输出 `[SKIP] cuModuleLoadData test — no driver`（Oracle review blocking fix 要求显式 SKIP）
6. **Multi-kernel scope**：v1 显式为 single-kernel（PTXIR v3 限制 — ptxir_serialization.cpp:97-108）。后续 change 升级 PTXIR 支持 N kernels 后，v2 manifest 可扩展