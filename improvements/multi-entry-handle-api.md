# multi-entry-handle-api

**优先级**: P1 | **来源**: [docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md) §3 差距清单（4 P0 + 2 P1 + 2 P2）+ [roadmap.md](roadmap.md) §Phase 12.5 + 2026-08-11 实施状态审计
**阶段**: Phase 12.5 | **分类**: arch-design
**类型**: functional

## 架构依据

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

## 范围

### In Scope (8 gaps, 6 commits per gap analysis §4)

| # | Gap | Commit | 关键交付物 |
|---|-----|--------|-----------|
| 1 | `cpptlm_module.cpp:111` 用 `kernels[0]` 单 entry 路径 | Commit 4 | 完整 multi-entry handle API + 公开 `get_kernel_handle(name)` |
| 2 | `cuModuleGetFunction` 多 kernel name→handle 映射 | Commit 3 | `cuFunction` 注册表 (key = kernel_name) + 真实现 |
| 3 | v2 PTXIR writer (multi-entry 写入) | Commit 1 | writer 单元测试: 单/多 entry 双向 round-trip |
| 4 | Multi-entry PTXIR fixture | Commit 2 | `tests/fixtures/ptx/multi_kernel_*.ptx` + generator 脚本 |
| 5 | `test_multi_kernel_selection.cpp` 升级 (placeholder) | Commit 5 | 替换 `SUCCEED("placeholder")` 为实际 fixture 加载测试 |
| 6 | `ptxemu_image_kernel_name` 多 kernel 暴露 | Commit 6 | 遍历 `kernels` vector 公开 API + ABI 兼容性验证 |
| 7 | ABI baseline 回归验证 (v1 binary 加载) | Commit 6 | 单元测试: v1 single-kernel binary 加载无 regression |
| 8 | `KernelEntry` 数据冗余文档化 | 文档澄清 (合并入 Commit 6) | 明确 `arg_count`/`arg_byte_size` vs `ManifestParam` source of truth |

### Out Scope

- ❌ **HAL extension** (Phase 13 已 ship, 跨仓 UsrLinuxEmu/TaskRunner) — 不重复实施
- ❌ **`ptxir_build` CLI / `ptx-nvcc` wrapper** (Phase 12.3.B/C 仍待启动) — 独立 improvement
- ❌ **v1 → v2 PTXIR upgrade tool** — backward-compat 已保证, 无需升级工具
- ❌ **多 module 跨模块 kernel name 冲突解决** (Phase 12.6 远期) — 命名空间 + aliasing 推迟
- ❌ **KernelEntry extend-only 字段实施** (`ptx_version`/`sm_target`) — ADR-0028 注释预留, 未来扩展
- ❌ **D3 mutation bug 复检** — Phase 12.3.A 已 ship (commit `c46bdfcc` regenerate baseline)
- ❌ **cpptlm_bridge.h ABI 修改** — ADR-0029 D7 5 byte-identical gates 继续 hold

## 关键场景

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

### SC-4: ptxemu_image_kernel_name 遍历
- **GIVEN** 加载多 kernel image (含 N 个 kernel)
- **WHEN** 通过 `libptxemu_device.so` 5 函数查询 kernel 名
- **THEN** 返回首个 kernel name (向后兼容) + 提供额外 API 遍历全部 N 个

### SC-5: 错误路径 (stale handle)
- **GIVEN** module 已 unload, function handle 仍被持有
- **WHEN** 调用 `cuLaunchKernel` 携带 stale handle
- **THEN** 返回 `CUDA_ERROR_INVALID_HANDLE` (per 架构 §7 7 类 error mapping)

### SC-6: Concurrent thread 隔离
- **GIVEN** 2 host thread 同时调用 `cuModuleGetFunction` (同名 / 不同名)
- **WHEN** 高并发场景 (per ModuleRegistry `std::mutex` 线程安全要求)
- **THEN** 无 data race, 两个 handle 独立有效

### SC-7: Multi-kernel drain e2e
- **GIVEN** 加载多 kernel module 后顺序 launch 3 个 kernel (kernel_a → kernel_b → kernel_c)
- **WHEN** 完成所有 launch + 验证输出
- **THEN** co-sim advance ceiling 满足 + drain 行为确定性

## 技术约束

### MUST (硬约束)

1. **不修改 `cpptlm_bridge.h` ABI** — per ADR-0029 D7 5 byte-identical gates 继续 hold
2. **不破坏 `libptxemu_device.so` 5 函数 ABI** — 仅扩展, 不修改签名
3. **backward-compat 100%** — 旧 v1 single-kernel binary 在新 runtime 必须可加载 (per ADR-0028 Decision 3)
4. **extend-only 字段预留** — `KernelEntry` 未来字段 (`ptx_version`/`sm_target`) 不在本次实施
5. **每 commit 独立可回退** — per `ptx-lessons-learned` §3
6. **测试先行 (TDD)** — Red → Green → sanity.sh 验证 (per `ptx-lessons-learned` §5)
7. **commit 顺序固定** — writer (C1) → fixture (C2) → handle API (C3-C4) → test 升级 (C5) → API 暴露 (C6), 不允许乱序
8. **多 entry fixture 包含 ≥3 个 kernel** — 验证 list 遍历 + 索引访问

### MUST NOT (禁止)

1. ❌ **不** 在 v2 writer 中删除 `kernel_name` 字段 (backward-compat)
2. ❌ **不** 修改 ANTLR 解析路径 (复用 `PTXIRLoader::deserializeForCubin()` 扩展)
3. ❌ **不** 修改 `PTXIR_VERSION` (4 已 ship, 不再 bump)
4. ❌ **不** 在 `cuFunction` 注册表中暴露 `kernels[0]` 之外的隐式 fallback
5. ❌ **不** 引入新 thread-safety 原语 (复用 `ModuleRegistry` 已有 `std::mutex`)
6. ❌ **不** 创建跨仓 commit (UsrLinuxEmu/TaskRunner 不在本仓范围)
7. ❌ **不** 在 `ptxemu_image_*` API 中硬编码 kernel index (提供 name API)

### SHOULD (推荐)

1. **writer round-trip 单元测试** — 单 entry / 多 entry / 空 kernels vector 三种输入
2. **`std::lock_guard` 模式** — 与 Phase 12.3.A `ModuleRegistry` 风格一致
3. **错误信息包含 kernel name** — 便于 e2e 调试 (per `ptx-debug` 技能)
4. **fixture generator 脚本 Python** — 复用 `tests/scripts/` 现有模式
5. **commit 6 包含 ABI 兼容性验证** — `nm -D libptxemu_device.so` 验证无符号增删

## 验收标准

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
- [ ] `nm -D build/lib/libptxemu_device.so` 无新 T 符号 (commit 6 ABI 验证)
- [ ] `nm -D build/lib/libcudart.so` 仍含 4 个 T 符号 (`cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`/`cuModuleUnload`)
- [ ] cpptlm_bridge.h diff **空** (5 byte-identical gates hold)
- [ ] per-commit git log 检查 (per Lesson §3): 6 个 commit, 每个独立可回退
- [ ] `ptxir_format.h` `PTXIR_VERSION` 仍为 4
- [ ] `Kernels` vector 在 v1 binary 加载后 synthesize 1 entry, `kernel_name` 一致
- [ ] `KernelEntry.arg_count` 与 `ManifestParam` vector size 数字一致 (source of truth 文档化)

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

## 参考资料

### 直接引用

- [multi-kernel-manifest-gaps-gap-analysis.md](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md) — 本提案的完整差距分析 + commit 拆分
- [ADR-0028 (multi-kernel-manifest)](docs/adr/ADR-0028-multi-kernel-manifest.md) — schema + backward-compat 已 ship
- [ADR-0029 §D4 (ptxemu-image-executor)](docs/adr/ADR-0029-ptxemu-image-executor.md) — D8 HAL 已 ship, D4 multi-kernel runtime 待补
- [ADR-0023 (ptxir-binary-format)](docs/adr/ADR-0023-ptxir-binary-format.md) — Extend-Only 版本管理
- [roadmap.md §Phase 12.5](roadmap.md#phase-125) — 延后登记的任务列表

### 实施模式参考

- Phase 12.3.A `ModuleRegistry` (`include/cudart/module_registry.h` + `src/cudart/module_registry.cpp`) — `std::mutex` 线程安全模式
- Phase 12.3.A `image_classifier` (`src/cudart/image_classifier.cpp`) — 纯函数单测模式
- Phase 12.3.A `cuModuleGetFunction` stub 替换 (`src/cudart/cudart_sim.cpp:514-521`) — 函数签名参考

### 教训沉淀

- [ptx-lessons-learned §3](docs/dev-process/lessons-learned.md) — 复杂迁移分 Phase commit
- [ptx-lessons-learned §5](docs/dev-process/lessons-learned.md) — 测试先行 (Red → Green)
- [ptx-debug](.opencode/skills/ptx-debug/SKILL.md) — 屏障/状态修改需行级 diff + 多模块交叉验证

### 关键 commit (Phase 12.4 基础, 不在本提案范围)

- `05504d0c` — feat(ptxir): extend ManifestSection to vector<kernel_entry> + bump PTXIR_VERSION
- `cd277e13` — docs(adr,architecture): ADR-0028 BLOCKING DEPENDENCY upgrade + toolchain-stack v1.3
- `c6ac1176` — test(cudart): add multi-kernel selection placeholder test
- `c46bdfcc` — fix(ptxemu): regenerate ABI baseline (stale after multi-kernel schema bump)
