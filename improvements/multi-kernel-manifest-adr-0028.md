# multi-kernel-manifest-adr-0028

**优先级**: P1 BLOCKING | **来源**: [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) v1.3 §11 + 3 个已 ship ADR 的 v1 单 kernel 限制拖累
**阶段**: Phase 12.4 | **分类**: arch-design
**类型**: functional

## 架构依据

架构文档 §11 明示 **ADR-0028 是 BLOCKING DEPENDENCY**，状态从"预留占位"于 2026-08-09 升级。当前根因：

`include/ptx_ir/ptxir_format.h` 的 `ManifestSection`（line 36-41）只有单 `kernel_name` 字段。这导致 3 个已 ship ADR 同时受 v1 单 kernel 限制拖累：

| ADR | §v1 限制段落 | 影响 |
|-----|-------------|------|
| ADR-0025 (`ptxir_build` CLI) | 单 kernel per binary | wrapper 拒绝 multi-entry PTX |
| ADR-0027 (`ptx-nvcc` wrapper) | 单 kernel per binary | 同样限制 |
| ADR-0029 (image executor) D4 | 单 kernel per image | `libptxemu_device.so` 的 `ptxemu_image_kernel_name` 只返回首个 |

**下游 ADR 必须遵守的契约**（架构 §11）：
1. ADR-0025/0027/0029 §v1 限制段落须明示"等待 ADR-0028 解除"
2. ADR-0028 ship 时必须 bump `PTXIR_VERSION`（继承 ADR-0023 Extend-Only 原则）
3. backward-compat 策略：旧 v1 单 kernel binary 在 ADR-0028 后运行时仍可加载（manifest 格式向后可读）

**额外背景**：
- ADR-0023 §决策 6（Extend-Only）规定 PTXIR 格式演进必须保持旧 reader 可读新 section
- 架构 §10 item 10（multi-kernel selection）已被标记"推迟到 v2 / ADR-0028 范围"——这是 acceptance gate 的源点

## 范围

**In Scope**:
- **新建 ADR-0028** 文件 `docs/adr/ADR-0028-multi-kernel-manifest.md`（设计阶段产物）
- 扩展 `include/ptx_ir/ptxir_format.h::ManifestSection` 为 `vector<kernel_entry>`（或等效结构）
- bump `PTXIR_VERSION`（per ADR-0023 Extend-Only）
- 维护 backward-compat：v1 单 kernel binary 仍可在新 runtime 下加载
- 更新下游 ADR §v1 段落（ADR-0025/0027/0029）
- 更新架构文档 `ptxir-toolchain-stack.md` v1.4（解除 v1 限制描述）

**Out Scope**:
- NVIDIA cubin 格式本身的修改（仍只追加 PTXIR section）
- 新增 kernel metadata 字段（仅扩展 entry 数量，不改每个 entry 的 schema——除非 schema 演进需要）
- ANTLR 解析路径修改
- `cuInit` / `cuCtx*` 等 context management（架构 §12 Future-4 远期）

## 关键场景

### 场景 1：multi-entry binary 加载

- **GIVEN** 一个 PTXIR image 含 3 个 `.entry` symbol（kernel A、B、C）
- **WHEN** `cuModuleLoadData(module, image)` 然后 3 次 `cuModuleGetFunction(func, module, "kernel_A"/"kernel_B"/"kernel_C")`
- **THEN** 全部 3 次查找成功；每个 `CUfunction` handle 独立；3 个 kernel 可独立 launch

### 场景 2：v1 单 kernel binary backward-compat

- **GIVEN** 旧 v1 单 kernel binary（`ManifestSection.kernel_name` 单值，无 `vector<kernel_entry>`）
- **WHEN** 在新 runtime 下加载
- **THEN** 行为与 v1 完全一致（reader 把单 entry 视为 `vector` 长度 1 的特例）；不报错

### 场景 3：runtime selection by name

- **GIVEN** multi-entry binary（kernel A、B、C）
- **WHEN** `cuModuleGetFunction(func, module, "kernel_B")`（按名选择）
- **THEN** 返回的 `CUfunction` handle 对应 kernel B；launch 时执行 B 的 `kernelStatements`，不执行 A 或 C

## 技术约束

### MUST

- **bump `PTXIR_VERSION`**（per ADR-0023 §决策 6 Extend-Only）
- **backward-compat**：v1 binary 必须仍可加载（reader 容错把单 `kernel_name` 视为 `vector` 长度 1）
- **新 section / field 添加遵循 Extend-Only**：旧 reader 可跳过未知 section（ADR-0023 §决策 6）
- **`PtxEmuImageExecutor` 多 entry handle 解析**（ADR-0029 §D4）：`ptxemu_image_kernel_name` 需支持 multi-entry 查询（或新增 API）
- **下游 ADR §v1 段落更新**：ADR-0025/0027/0029 在 ADR-0028 ship 后必须明示"已支持 multi-kernel"（per 架构 §11 下游契约 #1）
- **`__cudaRegisterFatBinary` + `cuModuleGetFunction` 多 kernel 名查询**：legacy 与 in-memory front door 都要支持按名选择
- **`ptxir-toolchain-stack.md` 同步升级到 v1.4**：解除 v1 单 kernel 限制描述
- **`docs/adr/README.md` 索引同步**：新增 ADR-0028 条目

### MUST NOT

- **必须 bump version 才扩展 schema**（per ADR-0023 Extend-Only）：不可在不变更 version 的情况下改 `ManifestSection` 结构
- **必须保持 v1 binary 字节级可读**：不允许"v1 binary 不再支持"的破坏性变更
- **必须继承现有 section TOC 布局**：不重新发明二进制格式
- **不在 v1 reader 中引入 silent failure**（per archive change `2026-08-07` 约束）：所有失败路径返回明确错误

### SHOULD

- 优先采用 `vector<kernel_entry>` 扩展；如需新字段则 bump 2 次（先 bump 加 vector，再 bump 加字段）
- 与 ADR-0029 image executor 协同：`ptxemu_image_kernel_name` 升级为支持 multi-entry 查询或新增 `ptxemu_image_get_function_by_name` API
- `ptxir_build` / `ptxir_embed` / `ptxir_extract` 三个工具同步支持 multi-kernel（架构 §3 §10 item 10）
- ADR-0028 文本包含 v1 → v2 migration 示例（reader 侧代码片段）

## 验收标准（架构层）

提案被批准后，guide-design → openspec proposal.md → tasks.md 时应明确：

1. **ADR-0028 文件创建**：`docs/adr/ADR-0028-multi-kernel-manifest.md` 含设计依据 + 决策 + 影响范围 + Extend-Only 合规说明
2. **`PTXIR_VERSION` bump**：旧 v1 reader 不识别新 version 但能跳过未知 section 仍可读 v1 binary
3. **backward-compat acceptance**：v1 fixture binary（如 `tests/ptxir/fixtures/cute_rmsnorm.ptxir`）在新 runtime 下加载成功且执行结果不变（架构 §10 item 10）
4. **multi-kernel acceptance**：multi-entry binary fixture 可被 `cuModuleLoadData` 加载 + `cuModuleGetFunction` 按名选择 3 个 entry 成功（架构 §10 item 10）
5. **下游 ADR §v1 段落更新**：ADR-0025/0027/0029 三份 ADR 在 ADR-0028 ship 后均更新 v1 限制段落为"已支持"或"等待 ADR-0028 解除"
6. **架构文档升级**：`ptxir-toolchain-stack.md` 从 v1.3 升级到 v1.4，§11 移除 ADR-0028 BLOCKING DEPENDENCY 标记

---

**依赖关系**：本提案是 Phase 12.3.A 实施**完成后的串行依赖**——`PTXIRLoader::deserializeForCubin` 签名会在多 entry 升级时变化，与 Phase 12.3.A 的 image classifier / 单元测试冲突。roadmap v1 的"并行"措辞已修正（per 2026-08-10 Oracle review）。

**注**：本提案只回答"为什么"和"什么"，不维护详细 tasks/tasks.md。详细实施 tasks 由 guide-design 评审通过后创建的 openspec `proposal.md` → tasks.md 维护。
