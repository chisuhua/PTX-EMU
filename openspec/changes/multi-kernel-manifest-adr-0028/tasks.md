# Tasks: multi-kernel-manifest-adr-0028

> **硬串行依赖（Oracle C2）**: 本 change 必须在 `ptxir-driver-api-front-door` 完成后启动——保持 `PTXIRLoader::deserializeForCubin()` 签名在 Phase 12.3.A 期间稳定。
> **TDD 5-step** discipline applies。

## Phase 1: ADR-0028 创建

### Task 1.1: 编写 ADR-0028 (Oracle C1 — 必须先建)
- **MUST**: 新建 `docs/adr/ADR-0028-multi-kernel-manifest.md`
- **MUST**: 引用 ADR-0023 §决策 6 Extend-Only
- **MUST**: 包含 v1 → v2 migration 示例（reader 侧代码片段）
- **MUST**: 包含下游契约（ADR-0025/0027/0029 §v1 段落更新要求）
- **验证**: 文件存在 + 内容含以上要素

### Task 1.2: 更新 docs/adr/README.md 索引
- **MUST**: 新增 ADR-0028 条目
- **验证**: 索引中出现 ADR-0028

### Task 1.3: 提交 Commit 1
- **MUST**: `docs(adr): add ADR-0028 multi-kernel manifest`

## Phase 2: ptxir_format.h 多 entry 扩展

### Task 2.1: 单元测试 - PTXIRLoader v1 backward-compat (TDD Red, Oracle C3)
- **MUST**: 写 `tests/unit/ptxir/test_ptxir_loader.cpp` 验证：
  - v1 binary (fixture: `tests/ptxir/fixtures/cute_rmsnorm.ptxir`) 加载成功且行为不变
  - v1 reader 跳过未知 section 不报错
- **验证**: FAIL（PTXIR_VERSION 未 bump）

### Task 2.2: 扩展 ManifestSection 为 vector<kernel_entry> + bump PTXIR_VERSION
- **MUST**: `include/ptx_ir/ptxir_format.h::ManifestSection` 增加 `vector<kernel_entry> kernels`
- **MUST**: `PTXIR_VERSION` bump（per ADR-0023 Extend-Only）
- **MUST**: 保留 `kernel_name` 单值字段作为 v1 backward-compat fallback
- **MUST NOT**: 修改 ANTLR 解析路径
- **验证**: Task 2.1 测试 PASS

### Task 2.3: 提交 Commit 2
- **MUST**: `feat(ptxir): extend ManifestSection to vector<kernel_entry> + bump PTXIR_VERSION`

## Phase 3: PTXIRLoader + PtxEmuImageExecutor 多 entry 支持

### Task 3.1: 单元测试 - deserializeForCubin 返回 vector
- **MUST**: 验证 `PTXIRLoader::deserializeForCubin()` 返回 `vector<kernel_entry>`
- **验证**: FAIL（接口未升级）

### Task 3.2: 更新 PTXIRLoader
- **MUST**: `src/cudart/ptxir_loader.cpp::deserializeForCubin()` 返回 `vector<kernel_entry>`
- **MUST**: v1 binary 处理：单 entry 视为 `vector` 长度 1
- **MUST**: v2 binary 处理：解析 `kernels[]` section
- **验证**: Task 3.1 测试 PASS

### Task 3.3: 更新 PtxEmuImageExecutor
- **MUST**: `src/cudart/cpptlm_module.cpp::load_image` 支持 multi-entry handle
- **MUST**: 可选：新增 `ptxemu_image_get_function_by_name` API（不修改 5 已 ship ABI）
- **验证**: 集成测试 PASS

### Task 3.4: 提交 Commit 3
- **MUST**: `feat(cudart): multi-entry support in PTXIRLoader + PtxEmuImageExecutor`

## Phase 4: runtime 多 kernel 名查询

### Task 4.1: 单元测试 - cuModuleGetFunction 按名选择
- **MUST**: 写 `tests/unit/cudart/test_multi_kernel_selection.cpp`：
  - multi-entry binary 加载
  - `cuModuleGetFunction(&func, module, "kernel_A"/"_B"/"_C")` 3 次成功
  - 每个 handle launch 执行对应 entry
- **验证**: FAIL

### Task 4.2: 实现 __cudaRegisterFatBinary 多 kernel 名查询
- **MUST**: `src/cudart/cudart_sim.cpp` 修改 `__cudaRegisterFatBinary` 支持 multi-entry
- **MUST**: 保留现有 legacy path 行为不变（架构 §4.1）
- **验证**: Task 4.1 部分测试 PASS（legacy path）

### Task 4.3: 实现 cuModuleGetFunction multi-entry 名查询
- **MUST**: 在 Phase 12.3.A 的 `cuModuleGetFunction` 实现上扩展 multi-entry 解析
- **验证**: Task 4.1 全部 PASS

### Task 4.4: 提交 Commit 4
- **MUST**: `feat(cudart): multi-kernel name selection in legacy + in-memory paths`

## Phase 5: tools + tests

### Task 5.1: 单元测试 - ptxir_build/embed/extract 多 kernel
- **MUST**: `tests/unit/tools/test_ptxir_multi_kernel.cpp`
- **MUST**: 3 工具的 multi-entry roundtrip
- **验证**: FAIL

### Task 5.2: 更新 tools
- **MUST**: `tools/ptxir_build.cpp` + `ptxir_embed` + `ptxir_extract` 多 kernel 支持
- **验证**: Task 5.1 测试 PASS

### Task 5.3: e2e 测试
- **MUST**: `tests/e2e/test_multi_kernel.cu` — nvcc 编译多 entry PTX → embed → 加载 → 按名 launch 全部 entry
- **验证**: e2e PASS

### Task 5.4: 提交 Commit 5
- **MUST**: `feat(tools): multi-kernel support in ptxir_build/embed/extract + e2e`

## Phase 6: ADR 更新 + 文档升级

### Task 6.1: 更新 ADR-0025 §v1 段落
- **MUST**: 改为 "已支持 multi-kernel"
- **验证**: diff 显示更新

### Task 6.2: 更新 ADR-0027 §v1 段落
- **MUST**: 同上
- **验证**: diff 显示更新

### Task 6.3: 更新 ADR-0029 D4 §v1 段落
- **MUST**: 同上
- **验证**: diff 显示更新

### Task 6.4: ptxir-toolchain-stack.md v1.3 → v1.4 (Oracle C4)
- **MUST**: 添加显式 changelog entry
- **MUST**: §11 移除 BLOCKING DEPENDENCY 标记
- **验证**: diff 显示 v1.4 + changelog

### Task 6.5: 提交 Commit 6
- **MUST**: `docs(adr): update v1 limitation paragraphs + architecture v1.4`

## Phase 7: archive readiness

### Task 7.1: openspec validate 通过
- **MUST**: `openspec validate multi-kernel-manifest-adr-0028` 无错误

### Task 7.2: 提交最终 commit
- **MUST**: `docs(changelog): phase 12.4 multi-kernel manifest ADR-0028 ship`

---

## 阻塞关系

- **本 change 阻塞**: 无（Phase 12.4 的产物）
- **本 change 被阻塞**: `ptxir-driver-api-front-door`（Phase 12.3.A）— Oracle C2 硬串行
- **本 change 阻塞后续**: 无（HAL extension Phase 13 独立启动）

## 风险与回退

- 任一 Phase commit 失败可独立 revert
- v1 backward-compat 测试失败必须停止推进（破坏性变更违反 ADR-0023）