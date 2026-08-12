## Context

### 背景
[ADR-0028 (Multi-Kernel Manifest)](docs/adr/ADR-0028-multi-kernel-manifest.md) 已 ship 2026-08-11 (commit `multi-kernel-manifest-adr-0028`)。Phase 12.4 schema ship 后状态：
- ✅ `ManifestSection.kernels` vector 扩展（`include/ptx_ir/ptxir_format.h:53`）
- ✅ `PTXIR_VERSION` 3→4 bump（`include/ptx_ir/ptxir_format.h:14`）
- ✅ backward-compat synthesis (`kernels` 空 + `kernel_name` 非空 → synthesize 单 entry)（`src/cudart/ptxir_loader.cpp`）

### 现状问题
[multi-kernel-manifest-gaps-gap-analysis §3](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md) 揭示 8 个 runtime 缺口：

| # | 缺口 | 严重度 |
|---|------|--------|
| 1 | `cpptlm_module.cpp:120` 用 `kernels[0]` 单 entry 路径 | P0 |
| 2 | `cuModuleGetFunction` 多 kernel name→handle 映射 | P0 |
| 3 | v2 PTXIR writer (multi-entry 写入) | P0 |
| 4 | Multi-entry PTXIR fixture (真实多 kernel 测试源) | P0 |
| 5 | `test_multi_kernel_selection.cpp` 结构性占位符 | P1 |
| 6 | `ptxemu_image_kernel_name` 仅返回首个 | P1 |
| 7 | ABI baseline 回归验证 (v1 binary 加载) | P2 |
| 8 | `KernelEntry` 数据冗余文档化 | P2 |

### 关键约束
- **cpptlm_bridge.h ABI 不变** — 5 byte-identical fallback gates 继续 hold（per [ADR-0029 §D7](docs/adr/ADR-0029-ptxemu-image-executor.md)）
- **HAL extension 不重复实施** — Phase 13 已 ship
- **`CPPTLM_MODULE_VERSION 1→2` bump 必须发生** — 新增 3 个 `extern "C"` 符号需 version gate
- **Per [ptx-lessons-learned](.opencode/skills/ptx-lessons-learned/SKILL.md)**：
  - 复杂迁移分 Phase commit — 6 个 commit，每个独立可回退（C1-C6）
  - 递归锁死锁 — `exec_mu_`/`mu_` 顺序固定（`execute()` 先 `exec_mu_` 再 `mu_`）

## Goals / Non-Goals

**Goals:**
1. **完整 multi-entry PTXIR 端到端支持**: writer → reader → runtime → e2e drain
2. **`cuModuleGetFunction` 真实实现**: 替换 `cudart_sim.cpp:556-570` stub
3. **libptxemu_device.so 多 kernel 枚举 API**: 3 个新 `extern "C"` 函数
4. **真实多 kernel fixture**: ≥3 kernel 端到端测试源
5. **ABI 向后兼容**: `CPPTLM_MODULE_VERSION 1→2` bump + 旧 v1 binary 加载验证

**Non-Goals:**
- ❌ `ptxir_build` CLI / `ptx-nvcc` wrapper 实施（独立 improvement）
- ❌ v1 → v2 PTXIR upgrade tool（backward-compat 已保证）
- ❌ 多 module 跨模块 kernel name 冲突解决（Phase 12.6 远期）
- ❌ `KernelEntry` extend-only 字段实施（ADR-0028 注释预留）
- ❌ D3 mutation bug 复检（Phase 12.3.A 已 ship）
- ❌ cpptlm_bridge.h ABI 修改（5 byte-identical gates 继续 hold）
- ❌ Within-module duplicate kernel name 详细语义（Oracle Q3 first-match wins 由本 change SC-8 明确, 后续独立 change）

## Decisions

### 决策 1: cuModuleGetFunction 注册表数据结构

**选择**: `std::unordered_map<std::string, CUfunction>` per module（in `ModuleRegistry`）

**Rationale**:
- 模块作用域隔离 — 不同 module 独立注册表，避免跨模块 kernel name 冲突
- `unordered_map` O(1) 平均查找（典型 N < 100 kernels per module）
- 已有 `ModuleRegistry` 模式（`include/cudart/module_registry.h`），扩展而非新增

**替代方案**:
- 全局 `unordered_map<CUmodule, unordered_map<...>>` — 增加模块句柄管理复杂度
- `std::map` — O(log N) 查找，对典型规模无优势

### 决策 2: libptxemu_device.so 多 kernel API 签名

**选择**: 3 个新 `extern "C"` 函数 + `CPPTLM_MODULE_VERSION 1→2` bump

```c
// include/cudart/cpptlm_module.h (新增)
int ptxemu_image_kernel_count(uint64_t handle);  // 返回 N
int ptxemu_image_kernel_name_at(uint64_t handle, uint32_t idx, char* buf, size_t buf_size);
int ptxemu_image_execute_named(uint64_t handle, const char* kernel_name,
                               uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                               uint32_t block_x, uint32_t block_y, uint32_t block_z,
                               size_t shared_mem_bytes,
                               void** kernel_args, size_t args_count);
```

**Rationale**:
- 名称遵循 `ptxemu_image_*` 既有命名空间
- 截断契约: `buf_size=0` 返回 -1 (查询需要长度)；`buf_size < len+1` 截断但不写 NUL（caller 验证）
- 新增符号是 ABI 向后兼容 (by-name 解析，旧 caller 不受影响)
- Version bump 强制下游 consumer (UsrLinuxEmu/TaskRunner) 在初始化时校验

### 决策 3: cpptlm_module.cpp 锁顺序保留

**选择**: `execute_named` 保持 `exec_mu_` → `mu_` 顺序（与 `execute()` 一致）

**Rationale** (per ptx-lessons-learned §3):
- 持锁方法调用同锁其他 public 方法 = deadlock
- `unload()` 用 `try_lock(exec_mu_)` 检测 in-flight execute()
- 新 `execute_named` 必须遵循相同顺序，避免引入新的 race window

### 决策 4: v2 PTXIR writer 多 entry 写入策略

**选择**: 保留 `kernel_name` 字段（向后兼容）+ 同时写 `kernels` vector

```cpp
// src/ptx_ir/ptxir_writer.cpp (修改)
ManifestSection ms;
// ... 填充 ms.kernels[i] for each kernel
ms.kernel_name = ms.kernels.empty() ? "" : ms.kernels[0].name;
writeManifestSection(ms);
```

**Rationale**:
- Reader 端 backward-compat synthesis 已 ship (Phase 12.4)
- 双字段保证 v1 reader 仍能读 v2 binary（虽然 `kernels[0]` fallback）
- v2 reader 优先读 `kernels`，无 fallback 时降级到 `kernel_name`

### 决策 5: Multi-entry fixture 生成策略

**选择**: Python generator (`tests/scripts/gen_multi_kernel_ptxir.py`) + 静态 PTX 文件 (≥3 kernel)

**Rationale**:
- 静态 PTX 保证 reproducible 测试
- Generator 脚本允许扩展更多 kernel 组合
- 测试运行时不依赖外部工具链（无 nvcc 编译）

### 决策 6: 测试覆盖策略

**选择**: 5 测试层 + 1 e2e drain

1. **单元测试**: KernelEntry / ManifestSection 结构（`tests/unit/cudart/test_multi_kernel_selection.cpp`）
2. **PTXIR round-trip**: v2 writer → reader（`tests/unit/ptxir/test_multi_entry_roundtrip.cpp` 新增）
3. **cudart 集成**: cuModuleGetFunction 3 场景（`tests/integration/cudart/test_cuda_driver_api.cpp`）
4. **cpptlm 集成**: 4 场景 (load + get_handle + execute + unload)（`tests/integration/cudart/test_in_memory_mutation.cpp`）
5. **API 集成**: ptxemu_image_kernel_name 多 kernel（`tests/integration/cudart/test_libptxemu_device.cpp` 新增）
6. **e2e drain**: 顺序 launch 3 kernel 确定性（`tests/e2e/multi_kernel_drain.cpp` 新增）

**Rationale**: 6 层测试覆盖 proposal.md §Capabilities 全部 3 项 + ABI baseline 回归

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| `libptxemu_device.so` 新增 3 个 T 符号破坏下游 binary | 符号添加是 ABI 向后兼容 (by-name 解析)；`CPPTLM_MODULE_VERSION` bump 强制 consumer 检查 |
| `cuModuleGetFunction` 注册表并发 race | `ModuleRegistry` 已用 `std::mutex`（per `include/cudart/module_registry.h`）；新增 per-module `unordered_map` 复用同一 mutex |
| `execute_named` 锁顺序错误引入 deadlock | 复制 `execute()` 的 `exec_mu_` → `mu_` 顺序；单元测试覆盖 SC-6 并发场景 |
| v2 writer 写入 `kernels` 但漏写 `kernel_name` 导致 v1 reader 加载失败 | 双字段强制一致性 (`kernel_name = kernels[0].name`)；unit test 验证 |
| Multi-entry fixture 与真实 nvcc 编译产物差异 | Generator 脚本从 PTX → PTXIR 与 `cudart` runtime 路径一致 (复用 `ptxir_loader.cpp`) |
| SC-8 first-match wins 行为与未来 ADR-0025/0027 冲突 | 在 proposal §What Changes 已明确 "后续独立 change"；SC-8 测试仅验证当前契约 |
| Per-Phase commit 顺序错误导致依赖倒置 | 6 个 commit 顺序 C1 (writer) → C2 (fixture) → C3 (cuModuleGetFunction) → C4 (cpptlm_module) → C5 (test upgrade) → C6 (kernel_name + ABI) |

## Migration Plan

### 6 Phase 部署步骤（每步独立可回退）

**Phase C1: v2 PTXIR writer (P0)**
1. 在 `src/ptx_ir/ptxir_writer.cpp` 添加 `writeMultiKernels()` 函数
2. 单元测试: round-trip 6 用例 (单 entry / 多 entry / 空 vector / 大端 / 异常 / fixture)
3. **回退策略**: `git revert C1 commit` — reader 已 ship backward-compat synthesis，v1 writer 仍可用

**Phase C2: Multi-entry fixture (P0)**
1. 创建 `tests/fixtures/ptx/multi_kernel_basic.ptx` (≥3 kernel)
2. 创建 `tests/scripts/gen_multi_kernel_ptxir.py` (generator)
3. 单元测试: fixture 加载成功，验证 ≥3 kernel
4. **回退策略**: 删除 fixture + 调整测试 fixture 引用

**Phase C3: cuModuleGetFunction handle 映射 (P0)**
1. 在 `ModuleRegistry` 添加 per-module `unordered_map<name, CUfunction>` (in `include/cudart/module_registry.h` + `src/cudart/cuda_driver.cpp`)
2. 替换 `cudart_sim.cpp:556-570` stub 为真实实现
3. 集成测试: 3 场景 (name 查找 / 重名 / 不存在)
4. **回退策略**: `git revert C3 commit` — registry 修改独立

**Phase C4: cpptlm_module multi-entry handle (P0)**
1. 修改 `include/cudart/cpptlm_module.h`: 添加 3 函数 + `CPPTLM_MODULE_VERSION 1→2`
2. 修改 `src/cudart/cpptlm_module.cpp`: 实现 3 函数 + 替换 `kernels[0]` fallback
3. 集成测试: 4 场景 (load + get_handle + execute + unload per kernel)
4. **回退策略**: `git revert C4 commit` — `CPPTLM_MODULE_VERSION` bump 同步回退

**Phase C5: test_multi_kernel_selection 升级 (P1)**
1. 修改 `tests/unit/cudart/test_multi_kernel_selection.cpp`: placeholder 全部替换
2. 单元测试: ≥3 真实测试
3. **回退策略**: `git revert C5 commit` — 测试独立

**Phase C6: ptxemu_image_kernel_name 多 kernel + ABI baseline (P1+P2)**
1. 修改 `ptxemu_image_kernel_name` (in `src/cudart/cpptlm_module.cpp`): 遍历 `kernels`
2. 单元测试: ABI baseline (v1 binary 加载) + mutation regression
3. 文档: `KernelEntry` 数据冗余段落 (in `docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md`)
4. **回退策略**: `git revert C6 commit` — kernel_name 修改独立

### 部署顺序

```
C1 (writer) → C2 (fixture) → C3 (cudart) → C4 (cpptlm) → C5 (test) → C6 (api+abi)
```

### 验证门

每 Phase 后必须通过：
- `cmake --build build && ctest --output-on-failure` — 0 failed
- `./scripts/sanity.sh` — 0 errors
- `./scripts/regression.sh` — 0 failures
- `nm -D build/lib/libptxemu_device.so` — 新增 3 符号 (允许), 无 modified/removed T 符号
- `nm -D build/lib/libcudart.so` — 仍含 4 个 T 符号 (`cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`/`cuModuleUnload`)
- `cpptlm_bridge.h` diff — 空 (5 byte-identical gates hold)
- Per-commit git log 检查 — 6 个 commit, 每个独立可回退

## Open Questions

| 问题 | 状态 | 决策依据 |
|------|------|----------|
| Within-module duplicate kernel name 详细语义 (Oracle Q3) | 已明确: first-match wins | proposal §SC-8 + §Out of Scope |
| `KernelEntry` 未来 extend-only 字段 (`ptx_version`/`sm_target`) | 已延后 | ADR-0028 注释预留，独立 improvement |
| `ptxir_build` CLI 是否需要 Phase 12.5 支持 multi-entry | 已延后 | `ptxir-toolchain-stack.md` v1.4 已移除 BLOCKING DEPENDENCY |
| 下游 consumer (UsrLinuxEmu/TaskRunner) 同步升级 `CPPTLM_MODULE_VERSION` 校验 | 跨仓 | Phase 13 已 ship，需独立跨仓 PR |
