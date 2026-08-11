# 架构差距分析: multi-kernel-manifest-gaps

> **生成日期**: 2026-08-11
> **状态**: 草案（待人工审查）
> **关联 ADR**: [ADR-0028 (multi-kernel-manifest)](../adr/ADR-0028-multi-kernel-manifest.md)
> **关联 change (已归档)**: `multi-kernel-manifest-adr-0028` (Phase 12.4, 2026-08-11 ship)
> **分析者**: Sisyphus (guide-arch Phase 3)
> **关联工具链文档**: [ptxir-toolchain-stack.md](./ptxir-toolchain-stack.md) v1.4

## 1. 目标架构

[ADR-0028](../adr/ADR-0028-multi-kernel-manifest.md) §决策内容定义的多 kernel manifest 目标架构：

### 1.1 Schema 扩展

- **`KernelEntry` struct**（`include/ptx_ir/ptxir_format.h`）
  ```cpp
  struct KernelEntry {
      std::string name;          // kernel symbol name
      uint32_t arg_count = 0;    // number of parameters
      uint32_t arg_byte_size = 0; // total argument bytes
      // (extend-only: future fields like ptx_version, sm_target)
  };
  ```
- **`ManifestSection` 扩展**：新增 `std::vector<KernelEntry> kernels`，**保留** `kernel_name`（v1 backward-compat）。
- **`PTXIR_VERSION` bump**：3 → 4（per [ADR-0023 §决策 6 Extend-Only](../adr/ADR-0023-ptxir-binary-format.md)）。

### 1.2 兼容性契约

- **Reader 端**：若 `kernels` vector 为空但 `kernel_name` 非空，synthesize 单-entry vector（v1 binary 直接可用）。
- **Writer 端**：v2 binary 同时写入 `kernel_name` 和 `kernels`，保证两者一致。
- **cpptlm_bridge.h ABI**：与 Phase 12.3.A 共享约束 — **不修改**。
- **`libptxemu_device.so` 5 函数 ABI**：仅扩展，不破坏现有签名。

### 1.3 解除的 v1 限制

| ADR | §v1 限制段落 | 解除后状态 |
|-----|------------|----------|
| [ADR-0025](../adr/ADR-0025-ptxir-build-cli.md) | §v1 单 kernel 限制 | `ptxir_build` 可处理 multi-entry PTX |
| [ADR-0027](../adr/ADR-0027-ptx-nvcc-wrapper.md) | §v1 单 kernel 限制 | `ptx-nvcc` wrapper 接受 multi-entry |
| [ADR-0029 D4](../adr/ADR-0029-ptxemu-image-executor.md) | v1 单 kernel per image | `ptxemu_image_kernel_name` 须遍历 `kernels` vector |

### 1.4 跨仓契约

- **`ptxir-toolchain-stack.md` §11**：BLOCKING DEPENDENCY 标记移除（v1.3 → v1.4 changelog）
- **下游 ADR §v1 限制段落**：ADR-0025/0027/0029 同步更新

## 2. 当前架构（Phase 12.4 ship 后状态）

### 2.1 已实施 (✅)

- **`include/ptx_ir/ptxir_format.h`**：已添加 `KernelEntry` struct + `ManifestSection.kernels` 字段 + `PTXIR_VERSION=4` 常量
- **`src/cudart/ptxir_loader.cpp`**：实现 backward-compat synthesis（`kernels` 空但 `kernel_name` 非空时 synthesize 单 entry）
- **`src/cudart/cpptlm_module.cpp`**：第 111-118 行实现 `kernels[0]` fallback：
  ```cpp
  // TODO Phase 12.5: full multi-entry handle API — for now, use kernels[0].
  if (manifest.kernels.empty()) {
      // synthesizes from kernel_name if kernels is empty.
  }
  em.kernelName = manifest.kernels[0].name;
  ```
- **测试覆盖**：
  - `tests/unit/cudart/test_multi_kernel_selection.cpp` — **结构性占位符**（明确标记"deferred to Phase 12.5"）
  - `tests/integration/cudart/test_cuda_driver_api.cpp`
  - `tests/integration/cudart/test_in_memory_mutation.cpp`
  - `tests/integration/cudart/test_ptxir_cubin_loader.cpp`
  - `tests/integration/divergence/test_post_barrier_two_halves.cpp`
  - `tests/e2e/`: co-sim advance ceiling + multi-kernel drain tests (commit `79617fde`)

### 2.2 文档同步 (✅)

- **`docs/architecture/ptxir-toolchain-stack.md`**：v1.4 修订已发布，§11 BLOCKING DEPENDENCY 标记已移除
- **ADR-0025/0027/0029 §v1 限制段落**：已更新（per `ptxir-toolchain-stack.md` v1.4 changelog）

### 2.3 显式延后到 Phase 12.5 (⏳)

- v2 PTXIR writer（multi-entry 写入能力）
- Multi-entry PTXIR fixture（实际多 kernel PTX → PTXIR 转换测试）
- `cuModuleGetFunction` distinct-handle 映射（多 kernel name → handle table）
- `cpptlm_module.cpp` full multi-entry handle API（当前仅用 `kernels[0]`）

## 3. 差距清单

| # | 差距项 | 严重程度 | 优先级 | 关联 change | 状态 |
|---|--------|---------|--------|------------|------|
| 1 | `cpptlm_module.cpp:111` 使用 `kernels[0]` 单 entry 路径（`load_image`/`get_kernel_name`/`execute` 多 entry handle 未实现） | 高 | P0 | Phase 12.5 | ⏳ deferred |
| 2 | `cuModuleGetFunction` 多 kernel name→handle 映射（reader 端 `kernels` vector 解析后无 handle table 关联） | 高 | P0 | Phase 12.5 | ⏳ deferred |
| 3 | v2 PTXIR writer（multi-entry 写入能力）— 当前 writer 仍只写 `kernel_name` 字段 | 高 | P0 | Phase 12.5 | ⏳ deferred |
| 4 | Multi-entry PTXIR fixture（真实多 kernel PTX 测试源）— 缺失则无法 e2e 验证 | 高 | P0 | Phase 12.5 | ⏳ deferred |
| 5 | `test_multi_kernel_selection.cpp` 是结构性占位符（3 个 TEST_CASE 中 1 个是 `SUCCEED("placeholder")`） | 中 | P1 | Phase 12.5 | ⏳ deferred |
| 6 | ADR-0029 D4 `ptxemu_image_kernel_name` 仅返回首个 — 多 kernel 暴露给 caller 的 API 不完整 | 中 | P1 | Phase 12.5 | ⏳ deferred |
| 7 | ABI baseline 重新生成（commit `c46bdfcc`）— 验证 PTXIR_VERSION=4 未引入 v1 binary 加载回归 | 低 | P2 | 验证测试 | ⚠️ 待回归验证 |
| 8 | `KernelEntry.arg_count` / `arg_byte_size` 与 `ManifestParam` vector 存在数据冗余（未说明哪个是 source of truth） | 低 | P2 | 文档澄清 | 📝 待 ADR-0028 §下游契约扩展 |

## 4. 补齐路径

### Phase 12.5（必需, 阻塞多 kernel 工具链闭环）

**目标**: 完整 multi-kernel end-to-end 支持

**步骤**:

1. **v2 PTXIR writer**（依赖 #3）
   - 在 `src/ptx_ir/ptxir_writer.cpp` 添加 multi-entry 写入路径
   - 同时写入 `kernel_name`（首个）+ `kernels` vector（全部）
   - 添加 writer 单元测试：单/多 entry 双向 round-trip

2. **Multi-entry PTXIR fixture**（依赖 #4）
   - 在 `tests/fixtures/ptx/` 添加 multi-kernel `.ptx`（例：`vec_add + mat_mul + reduce_sum`）
   - 配套生成 `.ptxir` 文件
   - 添加 fixture generator 脚本（`tests/scripts/gen_multi_kernel_ptxir.py`）

3. **cuModuleGetFunction name→handle 映射**（依赖 #2）
   - `src/cudart/cudart_sim.cpp` 添加 `cuFunction` 注册表（key = kernel_name）
   - 修改 `cuModuleGetFunction` 实现：在 `manifest.kernels` 中查找 name，返回对应 handle
   - 添加 unit + integration 测试

4. **cpptlm_module multi-entry handle API**（依赖 #1）
   - 替换 `kernels[0]` fallback 为完整遍历
   - 新增 `cpptlm_module::get_kernel_handle(name)` 公开 API
   - 集成 `ptxemu_image_kernel_name` 暴露多 kernel 名（依赖 #6）

5. **test_multi_kernel_selection.cpp 升级**（依赖 #5）
   - 替换 `SUCCEED("placeholder")` 为实际 fixture 加载 + handle 查找测试
   - 添加 multi-kernel drain 场景（co-sim advance ceiling 集成）

### Phase 12.6（可选, 体验优化）

- 跨模块 kernel name 冲突解决（命名空间 + aliasing）
- `KernelEntry` 扩展字段（`ptx_version`, `sm_target`）— per ADR-0028 Decision 1 extend-only 注释
- 工具链 doc v1.5：§11 完整 multi-kernel 章节

### 回归验证（即刻, 必做）

- 跑 `tests/unit/cudart/test_ptxir_loader.cpp` 验证 v1 binary 仍可加载（backward-compat）
- 跑 `tests/integration/cudart/test_in_memory_mutation.cpp` 验证 PTXIR_VERSION=4 无 mutation regression
- ctest 全量回归（参考 `tests/integration/divergence/test_post_barrier_two_halves.cpp` 的 multi-kernel drain 行为）

## 5. 参考资料

### 5.1 关联 ADR

- [ADR-0023 (ptxir-binary-format)](../adr/ADR-0023-ptxir-binary-format.md) — Extend-Only 版本管理（PTXIR_VERSION bump 依据）
- [ADR-0024 (ptxir-cubin-embed-extension)](../adr/ADR-0024-ptxir-cubin-embed-extension.md) — PTXIR-Embedded CUBIN 格式
- [ADR-0025 (ptxir-build-cli)](../adr/ADR-0025-ptxir-build-cli.md) — `ptxir_build` CLI（§v1 限制已更新）
- [ADR-0027 (ptx-nvcc-wrapper)](../adr/ADR-0027-ptx-nvcc-wrapper.md) — `ptx-nvcc` wrapper（§v1 限制已更新）
- [ADR-0028 (multi-kernel-manifest)](../adr/ADR-0028-multi-kernel-manifest.md) — **本差距分析主题**
- [ADR-0029 (ptxemu-image-executor)](../adr/ADR-0029-ptxemu-image-executor.md) — Image Executor（D4 v1 限制已更新）

### 5.2 关联 change artifacts (已归档)

- `openspec/changes/archive/2026-08-11-multi-kernel-manifest-adr-0028/` (Phase 12.4 ship)
  - `proposal.md` — 6 轮 Oracle 评审通过
  - `tasks.md` — 11 个 task 全部完成
  - `specs/ptxir-manifest/spec.md` — v2 schema 规范

### 5.3 关键 commit (git log)

| Commit | 说明 |
|--------|------|
| `05504d0c` | feat(ptxir): extend ManifestSection to vector<kernel_entry> + bump PTXIR_VERSION |
| `c6ac1176` | test(cudart): add multi-kernel selection placeholder test |
| `79617fde` | test(e2e): add co-sim advance ceiling contract + multi-kernel drain tests |
| `c46bdfcc` | fix(ptxemu): regenerate ABI baseline (stale after multi-kernel schema bump) |
| `b801837b` | docs(changelog): phase 12.4 multi-kernel manifest ADR-0028 ship |
| `f4a95f2c` | chore(workflow): mark multi-kernel-manifest-adr-0028 as archived |

### 5.4 实施状态引用

- **`include/ptx_ir/ptxir_format.h`**: `PTXIR_VERSION = 4` (line 36-41), `KernelEntry` struct, `ManifestSection.kernels`
- **`src/cudart/cpptlm_module.cpp:111`**: `// TODO Phase 12.5: full multi-entry handle API — for now, use kernels[0]`
- **`tests/unit/cudart/test_multi_kernel_selection.cpp`**: 3 个 TEST_CASE, 1 个为 `SUCCEED("placeholder")`
- **`docs/architecture/ptxir-toolchain-stack.md`**: v1.4 修订已发布（§11 BLOCKING DEPENDENCY 解除）

### 5.5 Phase 12.5 推荐起点

- 阅读 [ADR-0028 §下游契约](../adr/ADR-0028-multi-kernel-manifest.md) — 完整下游契约列表
- 阅读 [ADR-0029 D8 HAL 方案](../adr/ADR-0029-ptxemu-image-executor.md) — Phase 12.5 与 HAL extension 的耦合点
- 阅读 [ptxir-toolchain-stack.md §12](./ptxir-toolchain-stack.md) — HAL extension future work 协调

---

**审查建议**:
1. 验证 §2.1 的代码引用（line numbers）是否仍准确（git blame on `cpptlm_module.cpp:111`）
2. 验证 §3 差距清单的优先级（与 ADR-0028 决策者 + Phase 12.5 owner 确认）
3. 验证 §4 补齐路径的步骤顺序（依赖图 + 并行机会）
4. 确认 §5 参考资料的 commit hashes 仍在 git history 中

**下次更新**: Phase 12.5 启动时（更新 §2.3 显式延后项 → 已实施）

## §8 KernelEntry 数据冗余 - Source of Truth (Post-C6)

After multi-entry handle API ship (commit `multi-entry-handle-api` C6), the
canonical sources of truth for kernel metadata are:

| Field | Source of Truth | Rationale |
|-------|-----------------|-----------|
| kernel name | `ManifestSection.kernels[i].name` | Multi-kernel primary |
| arg count | `ManifestSection.params.size()` (= `ManifestParam` count) | Single source for arg count, mirrored into `KernelEntry.arg_count` for reader convenience |
| arg byte size | `sum(ManifestParam.size for p in params)` | Sum of param sizes, mirrored into `KernelEntry.arg_byte_size` |

**Contract**: `KernelEntry.arg_count == ManifestParam.size()` and
`KernelEntry.arg_byte_size == sum(ManifestParam.size)`. The mirror fields are
**derived** and must not be set independently. Reader (Phase 12.4
backward-compat synthesis) must always recompute from `ManifestParam` to
ensure consistency.

This section closes gap #8 from §3.
