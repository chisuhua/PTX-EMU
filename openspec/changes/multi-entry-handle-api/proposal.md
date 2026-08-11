# multi-entry-handle-api

## Why

[ADR-0028 (Multi-Kernel Manifest)](docs/adr/ADR-0028-multi-kernel-manifest.md) §决策内容 (Decision 1/2/3) 已 ship 2026-08-11:
- `ManifestSection.kernels` vector 扩展完成
- `PTXIR_VERSION` 3→4 bump 完成
- backward-compat synthesis (`kernels` 空 + `kernel_name` 非空 → synthesize 单 entry) 完成

但 [multi-kernel-manifest-gaps-gap-analysis §3](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md) 揭示 8 个 runtime 缺口, 阻塞多 kernel 工具链闭环:

| # | 缺口 | 严重度 | 当前状态 |
|---|------|--------|----------|
| 1 | `cpptlm_module.cpp:111` 用 `kernels[0]` 单 entry 路径 | 高 (P0) | TODO Phase 12.5 注释 |
| 2 | `cuModuleGetFunction` 多 kernel name→handle 映射 | 高 (P0) | 缺注册表 |
| 3 | v2 PTXIR writer (multi-entry 写入) | 高 (P0) | 仅写 `kernel_name` 字段 |
| 4 | Multi-entry PTXIR fixture (真实多 kernel 测试源) | 高 (P0) | 缺 |
| 5 | `test_multi_kernel_selection.cpp` 结构性占位符 | 中 (P1) | `SUCCEED("placeholder")` |
| 6 | `ptxemu_image_kernel_name` 仅返回首个 | 中 (P1) | D4 v1 限制延续 |
| 7 | ABI baseline 回归验证 (v1 binary 加载) | 低 (P2) | 待 commit `c46bdfcc` 验证 |
| 8 | `KernelEntry` 数据冗余 (`arg_count` vs `ManifestParam`) | 低 (P2) | 缺 source of truth 文档 |

**核心阻塞**: Phase 12.4 schema ship 后, **runtime 侧无 multi-entry 端到端支持**。这导致:
- `libptxemu_device.so` 用户无法访问非首个 kernel
- `cuModuleGetFunction` 多 kernel 名称查询返回空 handle
- `test_multi_kernel_selection.cpp` 1/3 是 placeholder, 真实 e2e 验证缺失

## What Changes

**In Scope (8 gaps from [multi-kernel-manifest-gaps-gap-analysis §3](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md))**:

| # | Gap | Severity | 关联 commit |
|---|-----|----------|------------|
| 1 | `cpptlm_module.cpp:111` 用 `kernels[0]` 单 entry 路径 | P0 | C4 |
| 2 | `cuModuleGetFunction` 多 kernel name→handle 映射 | P0 | C3 |
| 3 | v2 PTXIR writer (multi-entry 写入) | P0 | C1 |
| 4 | Multi-entry PTXIR fixture (真实多 kernel 测试源) | P0 | C2 |
| 5 | `test_multi_kernel_selection.cpp` 升级 (placeholder) | P1 | C5 |
| 6 | `ptxemu_image_kernel_name` 多 kernel 暴露 | P1 | C6 |
| 7 | ABI baseline 回归验证 (v1 binary 加载) | P2 | C6 |
| 8 | `KernelEntry` 数据冗余文档化 | P2 | C6 |

**Out of Scope**:

- ❌ **HAL extension** (Phase 13 已 ship, 跨仓 UsrLinuxEmu/TaskRunner) — 不重复实施
- ❌ **`ptxir_build` CLI / `ptx-nvcc` wrapper** (Phase 12.3.B/C 仍待启动) — 独立 improvement
- ❌ **v1 → v2 PTXIR upgrade tool** — backward-compat 已保证, 无需升级工具
- ❌ **多 module 跨模块 kernel name 冲突解决** (Phase 12.6 远期) — 命名空间 + aliasing 推迟
- ❌ **KernelEntry extend-only 字段实施** (`ptx_version`/`sm_target`) — ADR-0028 注释预留, 未来扩展
- ❌ **D3 mutation bug 复检** — Phase 12.3.A 已 ship (commit `c46bdfcc` regenerate baseline)
- ❌ **cpptlm_bridge.h ABI 修改** — ADR-0029 D7 5 byte-identical gates 继续 hold
- ❌ **Within-module duplicate kernel name 详细语义** (per Oracle Q3) — 单 module 内重名 first-match 行为, 计划中 SC-8 明确 first-match wins (后续独立 change)

### 关键场景

### SC-1: Multi-entry PTXIR 完整 round-trip
- **GIVEN** v2 PTXIR writer 输出 multi-entry binary (kernel_a, kernel_b, kernel_c)
- **WHEN** reader 反序列化并查询每个 kernel name
- **THEN** 3 个 KernelEntry 全部可访问, name 字段精确匹配

### SC-2: Backward-compat (v1 binary 加载)
- **GIVEN** 旧 v1 single-kernel PTXIR binary (无 `kernels` vector)
- **WHEN** 在 v2 reader + v2 runtime 加载
- **THEN** 触发 backward-compat synthesis: `kernels` 为单 entry, `kernel_name` 保留, ABI 行为不变

### SC-3: cuModuleGetFunction 多 kernel 名称解析
- **GIVEN** 已加载多 kernel module (cuModuleLoadData 完成)
- **WHEN** 调用 `cuModuleGetFunction(&fn, module, "kernel_b")` 3 次 (a, b, c)
- **THEN** 返回 3 个不同 `CUfunction` handle, 每个对应正确 kernel

### SC-4: ptxemu_image_kernel_name 遍历 (libptxemu_device.so 新 API)
- **GIVEN** 加载多 kernel image (含 N 个 kernel, N > 1)
- **WHEN** 通过新 API 遍历:
  - `ptxemu_image_kernel_count(handle)` 返回 N
  - `ptxemu_image_kernel_name_at(handle, idx, buf, buf_size)` 写每个 kernel name
- **THEN** 返回 N 个 kernel 名, 索引 0..N-1 可访问
- **契约**: `buf_size` 截断行为: 返回值为 `min(strlen(name)+1, buf_size)`, 提供 `buf_size=0` 检测长度 (返回 -1)

### SC-5: 错误路径 (stale handle)
- **GIVEN** module 已 unload, function handle 仍被持有
- **WHEN** 调用 `cuLaunchKernel` / `ptxemu_image_execute_named` 携带 stale handle
- **THEN** 返回 `CUDA_ERROR_INVALID_HANDLE` (cudart 路径) / `-1` (cpptlm 路径) per 架构 §7 7 类 error mapping
- **扩展**: 枚举 vs unload race — 调用 `ptxemu_image_kernel_name_at` 在并发 `ptxemu_image_unload` 中, 返回 -1 (race detected)

### SC-6: Concurrent thread 隔离
- **GIVEN** 2 host thread 同时调用 `cuModuleGetFunction` / `ptxemu_image_kernel_name_at` (同名 / 不同名)
- **WHEN** 高并发场景 (per ModuleRegistry `std::mutex` 线程安全要求)
- **THEN** 无 data race, handle 独立有效

### SC-7: Multi-kernel drain e2e
- **GIVEN** 加载多 kernel module 后顺序 launch 3 个 kernel (kernel_a → kernel_b → kernel_c)
- **WHEN** 完成所有 launch + 验证输出
- **THEN** co-sim advance ceiling 满足 + drain 行为确定性

### SC-8: Within-module duplicate kernel name (Oracle Q3)
- **GIVEN** multi-kernel module 中含 2 个同名 kernel (e.g., `_Z7vec_add` 出现 2 次, 不同 arg 列表)
- **WHEN** 调用 `cuModuleGetFunction(fn, module, "vec_add")`
- **THEN** 返回 first-match kernel (per 设计选择), 后续同名调用返回同一 handle
- **注意**: 这不是 error, 是 first-match wins 策略, 与 cudart stub 行为一致

## Capabilities

### 新增能力 (3 项)

1. **Multi-entry PTXIR 序列化**
   - v2 PTXIR writer 写入多个 `KernelEntry` 到 `ManifestSection.kernels` vector
   - `kernel_name` 字段同步保留 (backward-compat)
   - 完整 round-trip 测试: 单/多 entry / 空 vector / 大端 / 异常 / fixture

2. **cuFunction name→handle 注册表 (libcudart.so)**
   - `cuModuleGetFunction` 真实实现 (替换 cudart_sim.cpp:514-521 stub)
   - name→CUfunction 映射表 (key = kernel_name, value = CUfunction)
   - 线程安全: `std::lock_guard` per ModuleRegistry

3. **libptxemu_device.so multi-kernel 枚举 API (3 个新函数)**
   - `ptxemu_image_kernel_count(handle)` — 返回 N
   - `ptxemu_image_kernel_name_at(handle, idx, buf, buf_size)` — 写 name + 截断契约
   - `ptxemu_image_execute_named(handle, name, ...)` — 按 name 选择 kernel 执行 (替代 `kernels[0]` 硬编码)
   - **ABI 影响**: `CPPTLM_MODULE_VERSION 1 → 2` (in `include/cudart/cpptlm_module.h:7`)

## Impact

### 受影响文件 (10 个)

| 文件 | 修改类型 | 关联 commit |
|------|---------|------------|
| `include/ptx_ir/ptxir_writer.h` | 修改 | C1 |
| `src/ptx_ir/ptxir_writer.cpp` | 修改 | C1 |
| `src/cudart/cudart_sim.cpp` | 修改 (cuModuleGetFunction stub → 真实现) | C3 |
| `include/cudart/cpptlm_module.h` | 修改 (3 新函数 + VERSION 1→2) | C6 |
| `src/cudart/cpptlm_module.cpp` | 修改 (替换 kernels[0] fallback) | C4 |
| `tests/unit/cudart/test_multi_kernel_selection.cpp` | 修改 (placeholder 升级) | C5 |
| `tests/fixtures/ptx/multi_kernel_*.ptx` | 新增 (≥3 kernel) | C2 |
| `tests/scripts/gen_multi_kernel_ptxir.py` | 新增 (generator 脚本) | C2 |
| `tests/integration/cudart/test_cuda_driver_api.cpp` | 修改 (cuModuleGetFunction 集成测试) | C3 |
| `tests/integration/cudart/test_in_memory_mutation.cpp` | 修改 (multi-kernel drain e2e) | C6 |

### 受影响 ADR

- [ADR-0028](docs/adr/ADR-0028-multi-kernel-manifest.md) — 引用 Decision 1/2/3 (已 ship)
- [ADR-0029 §D4](docs/adr/ADR-0029-ptxemu-image-executor.md) — 替换 v1 单 kernel 限制段落

### 受影响 spec

- PTXIR v4 manifest schema (extend-only 字段预留)
- cpptlm_module ABI v2 (per `CPPTLM_MODULE_VERSION` bump)

### ABI 风险与缓解

| 风险 | 缓解 |
|------|------|
| `libptxemu_device.so` 新增 3 个 T 符号 | 符号添加是 ABI 向后兼容 (by-name 解析), 旧 caller 不受影响 |
| `CPPTLM_MODULE_VERSION 1→2` 强制 consumer 检查 | 文档同步告知下游 consumer (UsrLinuxEmu/TaskRunner) 在初始化时校验 |
| `ptxemu_image_execute_named` 替代 `kernels[0]` 硬编码 | 旧 `ptxemu_image_execute` 仍按 `kernels[0]` 运行 (backward-compat) |
| `ptxemu_image_kernel_name` 截断契约 | 文档明确: `buf_size=0` 返回 -1 表示需重试; `buf_size < len+1` 截断但不写 NUL (由 caller 验证) |

## Acceptance

### 量化指标

1. **v2 writer round-trip 测试** — ≥6 测试用例 (单 entry / 多 entry / 空 vector / 大端 / 异常 / fixture)
2. **Multi-entry fixture 完整度** — ≥3 个 kernel (vec_add + mat_mul + reduce_sum)
3. **cuModuleGetFunction 集成测试** — ≥3 测试场景 (name 查找 / 重名 / 不存在)
4. **cpptlm_module multi-entry API** — ≥4 测试 (load + get_handle + execute + unload per kernel)
5. **ptxemu_image_kernel_name 多 kernel API** — ≥2 测试 (返回首个 + 遍历全部)
6. **test_multi_kernel_selection.cpp 升级** — placeholder 全部替换, ≥3 真实测试
7. **ABI baseline 回归** — v1 binary 加载单元测试 + mutation regression 集成测试
8. **e2e multi-kernel drain** — 顺序 launch 3 kernel 确定性测试

### 质量门

- [ ] cmake --build build && ctest --output-on-failure **0 failed**
- [ ] ./scripts/sanity.sh **0 errors** (per `ptx-lessons-learned` §5)
- [ ] ./scripts/regression.sh **0 failures**
- [ ] `nm -D build/lib/libptxemu_device.so` **无 removed/modified T 符号** (commit 6 ABI 验证) — 新增 3 个允许 (`ptxemu_image_kernel_count`/`_kernel_name_at`/`_execute_named`) + `CPPTLM_MODULE_VERSION 1→2` bump
- [ ] `nm -D build/lib/libcudart.so` 仍含 4 个 T 符号 (`cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`/`cuModuleUnload`)
- [ ] cpptlm_bridge.h diff **空** (5 byte-identical gates hold) — **不** 修改 cpptlm_bridge.h ABI
- [ ] per-commit git log 检查 (per Lesson §3): 6 个 commit, 每个独立可回退
- [ ] `ptxir_format.h` `PTXIR_VERSION` 仍为 4
- [ ] `Kernels` vector 在 v1 binary 加载后 synthesize 1 entry, `kernel_name` 一致
- [ ] `KernelEntry.arg_count` 与 `ManifestParam` vector size 数字一致 (source of truth 文档化)
- [ ] `cpptlm_module.h` `CPPTLM_MODULE_VERSION 1→2` 验证 (commit 6 必须 bump)
- [ ] `SC-5` unload-vs-enumerate race 测试通过 (新加测试用例, per Oracle Q3 扩展)
- [ ] `SC-8` within-module duplicate name first-match 测试通过 (per Oracle Q3 新增)

### 文档同步

- [ ] `roadmap.md` §Phase 12.5 状态: 4 P0 + 2 P1 + 2 P2 全部 ✅
- [ ] `multi-kernel-manifest-gaps-gap-analysis.md` §3 状态列: 全部 ⏳ → ✅
- [ ] `proposal-approved.md` "已批准" 段添加本提案
- [ ] `iteration.json` 添加本 change (status: planned → shipped)

### 任务完成度

| # | 任务 | 验收 | 关联 commit |
|---|------|------|-----------|
| 12.5-1 | v2 PTXIR writer | round-trip 6 测试通过 | C1 |
| 12.5-2 | Multi-entry fixture | fixture 加载成功, ≥3 kernel | C2 |
| 12.5-3 | cuModuleGetFunction handle 映射 | 3 测试场景通过 | C3 |
| 12.5-4 | cpptlm_module multi-entry handle | 4 测试通过 + `kernels[0]` fallback 移除 | C4 |
| 12.5-5 | test_multi_kernel_selection 升级 | placeholder 全部替换, ≥3 真实测试 | C5 |
| 12.5-6 | ptxemu_image_kernel_name 多 kernel | 2 测试通过 + ABI 兼容 | C6 |
| 12.5-7 | ABI baseline 回归 | v1 加载 + mutation 测试通过 | C6 |
| 12.5-8 | KernelEntry 数据冗余文档化 | source of truth 文档段落 | C6 (commit msg + ADR 注释) |

