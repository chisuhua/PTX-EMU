# multi-kernel-manifest-adr-0028

> **Oracle 评审结果（2026-08-10）**: ✅ APPROVE-WITH-CONDITIONS — 风险 MEDIUM  
> **关键约束**: ADR-0028 是新建 ADR（不存在），提案本质是 "创 ADR + ship change"，且必须**硬串行**排在 Phase 12.3.A 之后

## Why

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
3. backward-compat 策略：旧 v1 单 kernel binary 在 ADR-0028 后运行时仍可加载

**额外背景**：
- ADR-0023 §决策 6（Extend-Only）规定 PTXIR 格式演进必须保持旧 reader 可读新 section
- 架构 §10 item 10（multi-kernel selection）已被标记"推迟到 v2 / ADR-0028 范围"

## What Changes

**In Scope**:
- **新建 ADR-0028** 文件 `docs/adr/ADR-0028-multi-kernel-manifest.md`（Oracle 条件 #1：必须先于代码修改）
- 扩展 `include/ptx_ir/ptxir_format.h::ManifestSection` 为 `vector<kernel_entry>`
- bump `PTXIR_VERSION`（per ADR-0023 Extend-Only）
- 维护 backward-compat：v1 单 kernel binary 仍可在新 runtime 下加载（Oracle 条件 #3：runtime-tested 回归测试）
- 更新下游 ADR §v1 段落（ADR-0025/0027/0029）
- 更新架构文档 `ptxir-toolchain-stack.md` v1.4（Oracle 条件 #4：明确 changelog）

**Out of Scope**:
- NVIDIA cubin 格式本身的修改
- 新增 kernel metadata 字段
- ANTLR 解析路径修改
- `cuInit` / `cuCtx*` 等 context management（架构 §12 Future-4 远期）

### 关键场景

#### 场景 1：multi-entry binary 加载
- **GIVEN** PTXIR image 含 3 个 `.entry` symbol（kernel A、B、C）
- **WHEN** `cuModuleLoadData` + 3 次 `cuModuleGetFunction` 按名
- **THEN** 全部成功；每个 handle 独立

#### 场景 2：v1 单 kernel binary backward-compat
- **GIVEN** 旧 v1 单 kernel binary（无 vector）
- **WHEN** 在新 runtime 下加载
- **THEN** 行为与 v1 完全一致；不报错

#### 场景 3：runtime selection by name
- **GIVEN** multi-entry binary
- **WHEN** `cuModuleGetFunction(func, module, "kernel_B")`
- **THEN** 返回的 CUfunction 对应 kernel B

## Capabilities

- **bump `PTXIR_VERSION`**（per ADR-0023 §决策 6）
- **backward-compat**：v1 binary 仍可加载
- **新 section 添加遵循 Extend-Only**：旧 reader 可跳过未知 section
- **`PtxEmuImageExecutor` 多 entry handle 解析**
- **`__cudaRegisterFatBinary` + `cuModuleGetFunction` 多 kernel 名查询**
- **`ptxir-toolchain-stack.md` v1.4 升级**（ORACLE CONDITION）
- **`docs/adr/README.md` 索引同步**：新增 ADR-0028 条目
- **不为 silent failure**：所有失败路径返回明确错误

### MUST NOT（per improvements 头部 §MUST NOT）

- **必须 bump version 才扩展 schema**（per ADR-0023 Extend-Only）：不可在不变更 `PTXIR_VERSION` 的情况下改 `ManifestSection` 结构
- **必须保持 v1 binary 字节级可读**：不允许"v1 binary 不再支持"的破坏性变更
- **必须继承现有 section TOC 布局**：不重新发明二进制格式
- **不为 silent failure**（per archive change `2026-08-07-implement-ptxir-cubin-embed-extension` 约束）：所有失败路径返回明确错误
- **不修改 ANTLR 解析路径**：新格式演进只动 `ptxir_format.h` + reader/writer，不动 ANTLR .g4 文件

## Impact

- **PTXIR_VERSION bump**（per ADR-0023 Extend-Only）：v1 reader 不识别新 version 但能跳过未知 section
- **下游 ADR §v1 段落须更新**：ADR-0025/0027/0029 描述变更为"已支持 multi-kernel"
- **`ptxir-toolchain-stack.md` v1.3 → v1.4**：§11 移除 BLOCKING DEPENDENCY 标记

## Acceptance

### Oracle 评审通过条件（HARD）
- [ ] **C1**: 必须**先**创建 `docs/adr/ADR-0028-multi-kernel-manifest.md`（引用 ADR-0023 Extend-Only），再做代码修改
- [ ] **C2**: **硬串行**：此 change 必须在 Phase 12.3.A 完成后启动（`deserializeForCubin` 签名须在 12.3.A 期间保持稳定）——改为 `task_id` 依赖，非 prose
- [ ] **C3**: v1 backward-compat 必须用真实 v1 single-kernel binary（如 `tests/ptxir/fixtures/cute_rmsnorm.ptxir`）做 runtime 测试，lock 回归测试在 `tests/integration/ptxir/`
- [ ] **C4**: `ptxir-toolchain-stack.md` v1.3 → v1.4 必须含显式 changelog entry

### 标准交付物
- [ ] **ADR-0028 文件创建**：`docs/adr/ADR-0028-multi-kernel-manifest.md` 含设计依据 + 决策 + 影响范围 + Extend-Only 合规说明
- [ ] **`PTXIR_VERSION` bump**
- [ ] **multi-kernel acceptance**：multi-entry binary fixture 可被 `cuModuleLoadData` 加载 + `cuModuleGetFunction` 按名选择 3 个 entry 成功
- [ ] **下游 ADR §v1 段落更新**：ADR-0025/0027/0029 三份 ADR 在 ADR-0028 ship 后均更新

### 测试枚举（覆盖 A8-style 拆分）

**M1 - v1 backward-compat 回归**（Oracle C3，对应 `tests/integration/ptxir/`）：
- [ ] v1 single-kernel binary fixture（如 `tests/ptxir/fixtures/cute_rmsnorm.ptxir`）加载成功且执行结果不变
- [ ] v1 reader 跳过未知 section 不报错（per ADR-0023 Extend-Only）
- [ ] v1 binary 字节级校验：`xxd tests/ptxir/fixtures/cute_rmsnorm.ptxir` 与 archive baseline 一致

**M2 - multi-kernel fixture 测试**（roadmap.md:234）：
- [ ] 3-entry PTXIR fixture（kernel A/B/C）被 `cuModuleLoadData` 加载成功
- [ ] `cuModuleGetFunction(func, module, "kernel_A"/"_B"/"_C")` 3 次按名查找全部成功
- [ ] 每个 CUfunction handle launch 时执行对应 entry 的 `kernelStatements`（独立 dispatch，无串扰）

**M3 - unit 测试**（roadmap.md:234）：
- [ ] `PTXIRLoader::deserializeForCubin()` 返回 `vector<kernel_entry>` 类型测试
- [ ] `PtxEmuImageExecutor::load_image` 多 entry handle 解析测试
- [ ] `ptxir_build` / `ptxir_embed` / `ptxir_extract` 三工具多 kernel 支持测试

**M4 - e2e 测试**（roadmap.md:234）：
- [ ] `tests/e2e/test_multi_kernel.cu`：nvcc 编译多 entry PTX → embed → 加载 → 按名 launch 全部 entry
