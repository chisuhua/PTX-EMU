# implement-ptxir-cubin-embed-extension — Design

## Context

### 背景

PTX-EMU 当前 (`2026-08`) 接收 CUDA 程序通过两种路径：
1. **PTX 路径**：nvcc → `--ptx` → ANTLR4 解析 → `StatementContext[]` → 解释执行
2. **PTXIR 路径**（仅内部 pipeline）：`ptx-serializer` → PTXIR 字节流 → 反序列化 → `StatementContext[]`

但**标准 NVIDIA cubin**（nvcc 默认产物）无法被 PTX-EMU 直接加载 — 因为 ANTLR 解析路径只接受 PTX 文本，PTXIR 反序列化路径不在 `__cudaRegisterFatBinary` 入口暴露。

`ptxir-format-compliance` 提案（2026-08-01）试图将 PTXIR 序列化为 cubin 的 sibling 文件，但被拒绝（与 ADR-0023 的 7 项架构决策不完全一致 — sibling 模型要求 PTX-EMU 调用方显式传 PTXIR 路径，破坏了 cubin 单文件部署的兼容性）。

### 现状

- `__cudaRegisterFatBinary(void* fatbin)` 接收 NVIDIA 标准 fat binary 句柄
- 当前实现只走 PTX 文本提取（`extractPtxFromFatBinary`），无 PTXIR 检测
- PTXIRLoader 类**不存在**
- `config::isPTXIRModeEnabled()` **不存在**
- `PTXIR_EMBED_MAGIC` **不存在**

### 约束

- **ADR-0024 已 Accepted**（commit `18ad58cb`，2026-08-06）：PTXIR-Embedded CUBIN 格式设计决策
- **ADR-0023 已 Accepted**（sibling, 2026-07-30）：PTXIR 二进制格式 + Section TOC 复用约束
- **`PTXIR_EMBED_MAGIC` 字面值 governance check**：magic 任何变更必须触发 ADR-0024 重新审视（不可在 proposal 层面单方面修改）
- **`PTXIR_MODE` 默认 OFF**：保证未启用用户的行为字节级兼容现状

### 利益相关方

- 模拟器核心（`ptxsim`）— 无直接影响（loader dispatch 在 cudart 层）
- CUDA runtime 拦截（`cudart`）— 主要修改点
- 工具链用户（`tools/`）— 新增 CLI
- ADR 治理委员会 — magic number governance check 守门

## Goals / Non-Goals

### Goals

1. PTX-EMU 能加载标准 cubin（含/不含 PTXIR 嵌入段）而不破坏 NVIDIA 工具链兼容性
2. `PTXIR_MODE=auto` 时自动检测嵌入段并走 PTXIR 快速加载路径
3. `PTXIR_MODE=off` 时行为字节级等价现状（默认）
4. 提供 `ptxir_embed` / `ptxir_extract` CLI 工具供 nvcc 工具链集成
5. ADR-0024 §合规检查 6 项全部通过（包含 Oracle review 新增的 2 个直接对 embedded cubin 解析场景）

### Non-Goals

- 不修改 NVIDIA cubin 格式前缀（仅追加嵌入段）
- 不修改 ANTLR 解析路径
- 不在 GPU registry / WarpContext / ThreadContext 添加新依赖
- 不修改 `__cudaRegisterFatBinary` ABI
- 不实现 PTXIR Section TOC v1 → v2 升级（独立 change）
- 不集成 CppTLM bridge（独立 ADR-0021 范围）

## Decisions

### 决策 1：magic 检测采用尾部 O(1) 后缀扫描

**选择**：`PTXIR_EMBED_MAGIC` 8 字节后缀，loader 检测末尾 8 字节是否匹配。

**理由**：
- O(1) 检测 vs 完整解析 cubin 内部 ELF/EFT 结构
- 与 NVIDIA 已有 magic 不冲突（需在实施前用 `cuobjdump --dump-elf` 验证 — ADR-0024 决策 4）
- `config::isPTXIRModeEnabled()` OFF 时跳过整个 8 字节读取（零开销）

**备选**：
- ❌ ELF section 注入：需修改 ELF 工具链，破坏 NVIDIA 兼容性
- ❌ ELF header magic 复用：与 NVIDIA driver magic 冲突，无法区分

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

**选择**：嵌入段使用 ADR-0023 已定义的 Section TOC 格式，PTXIRHeader 在 Section TOC 中追加 `cubin_hash` 字段。

**理由**：
- ADR-0023 7 决策已通过 sibling 决策固化（ADR-0024 引用）
- PTXIR 反序列化逻辑无需修改（仅添加 `cubin_hash` 字段解析）
- 工具链（`ptx-serializer` 等）无需适配

**备选**：
- ❌ 自定义嵌入格式：违反 ADR-0023 复用约束
- ❌ 修改 Section TOC v1 → v2：超出本 change 范围

### 决策 5：`config::isPTXIRModeEnabled()` 必须实现，不可降级为 SHOULD

**理由**：
- loader dispatch 行为**完全依赖**此函数
- `PTXIR_MODE` env var + `configs/*.ini` 双源配置（与项目现有全局 config 机制一致）
- CI 守门：必须确保 `PTXIR_MODE=off` 完全 bypass 检测分支

**备选**：
- ❌ 仅 env var：与项目现有 INI 配置风格不一致
- ❌ 仅 INI 配置：env var 调试不便

### 决策 6：工具链拆分 3 个独立 commits（参考 `worktree-archive-workflow` v2.0.5+）

**选择**：
- Commit 1: PTXIRLoader 类 + unit 测试
- Commit 2: cudart_sim.cpp dispatch 集成 + integration 测试
- Commit 3: tools/ptxir_extract.cpp + tools/ptxir_embed.cpp + e2e 测试

**理由**：
- 每个 commit 独立可回退（Commit 1 失败不影响 Commit 0/2）
- unit → integration → e2e 渐进式风险释放
- 与项目分 Phase commit 实践一致（参见 `ptx-lessons-learned` 经验）

**备选**：
- ❌ 单 commit：单元/集成测试同时失败时难以定位根因
- ❌ 5+ commits：颗粒度过细，集成成本增加

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| [R1] NVIDIA driver 拒绝尾部 magic（破坏 cuModuleLoadData 兼容性） | e2e 测试必须包含 `cuobjdump --dump-sass kernel.embedded.cubin` + `cuModuleLoadData` 直接验证（Oracle review blocking fix，ADR-0024 §风险 risk 1 / factor 2） |
| [R2] `PTXIR_EMBED_MAGIC` 字面值冲突 NVIDIA 已有 magic | 实施前用 `cuobjdump --dump-elf` + `strings` 验证，ADR-0024 §合规检查 6 项第 6 条强制 governance check |
| [R3] PTXIR section 损坏导致 cudart 崩溃 | deserialize 失败统一返回 null，dispatch 优雅降级到标准 cubin 路径 |
| [R4] cubin 末尾空间不足，无法追加 magic + section | magic 检测必须在 cubin size 上限内（loader 不假设 cubin 末尾永远有充足空间，MUST NOT） |
| [R5] `cubin_hash` 不匹配（PTXIR 与 cubin 不对应） | loader 校验失败返回 null，dispatch 走标准 cubin 路径 |
| [R6] `PTXIR_MODE` env var 在 CI / 测试环境未正确传播 | config 双源（env var + INI），CI 显式 `PTXIR_MODE=off` 验证默认行为 |
| [R7] 单元/集成/e2e 三层测试覆盖率不足 | 测试验收标准明确覆盖率 ≥ 90% + ≥ 5 场景 |
| [R8] ANTLR 解析路径意外被修改 | MUST NOT（proposal 级强制），code review 守门 |

## Migration Plan

### 部署步骤

1. **Commit 1 部署**：合并 PTXIRLoader + unit 测试 → 默认 `PTXIR_MODE=off`，无运行时影响
2. **Commit 2 部署**：合并 cudart dispatch + integration 测试 → `PTXIR_MODE=auto` 默认 OFF，运行时行为不变
3. **Commit 3 部署**：合并 CLI 工具 + e2e 测试 → 用户主动使用 `PTXIR_MODE=auto` 启用

### 回滚策略

- 任意 Commit 失败 → revert 该 commit，前序 commits 仍生效（PTXIRLoader 类保留，下次升级重试）
- 不需要数据迁移（无 schema 变更）
- 不需要配置迁移（新增 INI 字段，默认 OFF）

### 兼容性保证

- `PTXIR_MODE=off`（默认）：行为字节级等价现状
- `PTXIR_MODE=auto`：仅在检测到合法嵌入段时切换路径
- 未设置 `PTXIR_MODE`：等同 `off`

## Open Questions

1. **`PTXIR_EMBED_MAGIC` 字面值**：由实施者提议，需在 ADR-0024 §合规检查中验证不与 NVIDIA 已有 magic 冲突（governance check 触发条件）
2. **Section TOC 中 `cubin_hash` 字段的 hash 算法**：建议 SHA-256（与项目现有 `StatementContextHash` 一致），需在 design.md 实施前确认
3. **PTXIRLoader 错误日志格式**：建议 `[PTXIRLoader]` 前缀 + 严重性等级（`ERROR`/`WARN`/`INFO`），与项目现有日志风格一致
4. **`PTXIR_MODE` 接受的值**：`auto` / `off`（最小集），是否需要 `force`（跳过 hash 校验）？需在 spec 中确认
5. **e2e 测试中真实 NVIDIA driver 可用性**：若环境无真实 driver，测试输出 `[SKIP] cuModuleLoadData test — no driver`（Oracle review blocking fix 要求显式 SKIP）