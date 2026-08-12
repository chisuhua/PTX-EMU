# hal-extension-ptxemu-usrlinu-emu-taskrunner

**优先级**: P1 | **来源**: [docs/adr/ADR-0029-ptxemu-image-executor.md](docs/adr/ADR-0029-ptxemu-image-executor.md) §D8（HAL 方案 D8 修订）+ [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) v1.3 §2 CP 端跨仓集成节点表 + §12 future work
**阶段**: Phase 13 | **分类**: arch-design
**类型**: integration

## 架构依据

ADR-0029 §D8 描述 HAL 方案 D8 的 CP 端跨仓集成路径：TaskRunner 仓的 CUDA driver LD_PRELOAD shim（`libcuda_shim`）通过 `IGpuDriver` → `GpuDriverClient` → System C ioctl 间接调 UsrLinuxEmu 仓的 GPU 驱动，UsrLinuxEmu 仓的 HAL 层（`hal_user.cpp`）dlsym `libptxemu_device.so` 暴露的 `ptxemu_image_*` 函数，最终调 PTX-EMU。

3 个仓的耦合关系：
- **TaskRunner** — CUDA driver LD_PRELOAD shim；CUDA Driver API 用户；调 `cuModuleLoadData`/`cuLaunchKernel`/`cuModuleUnload`
- **UsrLinuxEmu** — GPU 驱动 + HAL；System C ioctl + HAL fn-ptr 边界；**HAL 是 drv ↔ sim 唯一桥**（per UsrLinuxEmu ADR-036 三区分架构硬约束）
- **PTX-EMU** — device-side executor；暴露 `libptxemu_device.so` 5 `extern "C"` ABI 入口（已 ship）

**关键架构约束（D8.1）**：**TaskRunner 仓零 PTX-EMU 链接依赖**。所有 PTX-EMU 调用经 UsrLinuxEmu HAL 边界封装。这意味着 TaskRunner 不能 `#include` PTX-EMU 任何头文件、不能 link `libptxemu_device.so`、不能在 build system 引入 PTX-EMU include 路径。

**为什么需要 improvement 提案先行**（而非直接 OpenSpec）：
1. **跨 3 仓协调**——任一仓的破坏性变更都会波及其他仓
2. **不可逆的契约变更**——一旦 UsrLinuxEmu 加 ioctl + HAL fn-ptr #66-68 + TaskRunner 加 IGpuDriver 3 方法，撤回成本高
3. **需要架构师 review**（design-done gate）后再投入实施，避免 3 仓的反复返工
4. **PTX-EMU 仓只需保证 `libptxemu_device.so` ABI 稳定**——具体跨仓细节由 UsrLinuxEmu + TaskRunner 决定，PTX-EMU 仓不能越界

## 范围

**In Scope（PTX-EMU 仓）**:
- 保证 `libptxemu_device.so` 5 `ptxemu_image_*` ABI 入口签名稳定（已 ship，需冻结）
- 保证 `libptxemu_device.so` SONAME / symlinks 不变
- 新增 1 个 PTX-EMU 仓内跨仓 RFC 文档（引用 TaskRunner ADR-035 R5.1 原文，确认跨仓 commit 顺序）
- DL-isolated 测试 + in-flight unload 边界测试（已部分存在，需扩展覆盖）

**In Scope（跨仓 — 仅记录，不在本仓实施）**:
- **UsrLinuxEmu 仓**：
  - 新增 3 个 ioctl：`GPU_IOCTL_LOAD_KERNEL_MODULE`/`LAUNCH_KERNEL_MODULE`/`UNLOAD_KERNEL_MODULE`（编号 39/40/41）
  - 新增 3 个 HAL fn-ptr #66/#67/#68（`kernel_module_load`/`execute`/`unload`）
  - `hal_user.cpp` 新增 dlsym `libptxemu_device.so` 的 `ptxemu_image_*` 实现
- **TaskRunner 仓**：
  - `libcuda_shim` 实现 `cuModuleLoadData`/`cuLaunchKernel`/`cuModuleUnload` 经 `IGpuDriver`
  - `IGpuDriver` 新增 3 个纯虚方法（`load_kernel_module`/`launch_kernel_module`/`unload_kernel_module`）

**Out Scope（PTX-EMU 仓）**:
- 不引入 TaskRunner 或 UsrLinuxEmu include 路径
- 不修改 `cpptlm_bridge.h` ABI（与 Phase 12.3.A 共享同一约束）
- 不实现 `cuModuleLoadData` 等 CUDA Driver API（Phase 12.3.A 范围）
- 不实现新的 HAL ioctl handler（UsrLinuxEmu 仓范围）
- 不修改 `libcudart.so` 任何 entry（HAL 集成路径在 UsrLinuxEmu 仓独立）

## 关键场景

### 场景 1：TaskRunner UMD 端到端

- **GIVEN** TaskRunner 仓 `libcuda_shim` 加载，`cuModuleLoadData(image)` 被 CUDA 应用代码调用（image 是 standalone PTXIR bytes）
- **WHEN** `cuModuleLoadData` → `IGpuDriver::load_kernel_module` → `GpuDriverClient` → System C ioctl → UsrLinuxEmu drv → UsrLinuxEmu HAL fn-ptr #66 → `hal_user.cpp` dlsym `ptxemu_image_load` → PTX-EMU executor → 返回 handle
- **THEN** `cuModuleLoadData` 成功返回；handle 可用于 `cuLaunchKernel`/`cuModuleUnload`；kernel 实际执行

### 场景 2：跨仓 commit 顺序（per TaskRunner ADR-035 R5.1）

- **GIVEN** 3 仓需要按依赖顺序 ship：先 UsrLinuxEmu 加 ioctl + HAL fn-ptr → 再 PTX-EMU 验证兼容性 → 最后 TaskRunner `IGpuDriver` 扩展 + `libcuda_shim` 集成
- **WHEN** 任意一仓未就绪时启动下一仓
- **THEN** 下一仓的编译/测试会失败；防止提前 commit 引入死锁/未解析符号

### 场景 3：in-flight unload returns busy

- **GIVEN** 同一 `CUmodule` 的 kernel 正在执行（in-flight）
- **WHEN** `cuModuleUnload(module)` 被调用（TaskRunner `libcuda_shim` 路径）
- **THEN** 返回 `CUDA_ERROR_INVALID_HANDLE`（busy）——沿链路直到 PTX-EMU executor 的 `ptxemu_image_unload`（架构 §10 item 24）

## 技术约束

### MUST（PTX-EMU 仓）

- **`libptxemu_device.so` 5 ABI 入口签名不变**：`ptxemu_image_load` / `ptxemu_image_kernel_name` / `ptxemu_image_execute` / `ptxemu_image_unload` / `ptxemu_module_version`（已 ship，需冻结）
- **`libptxemu_device.so` SONAME / symlinks 不变**：`libptxemu_device.so.12` 主版本号不 bump；`libptxemu_device.so → libptxemu_device.so.12 → libptxemu_device.so.12.0` 链式保留
- **`CPPTLM_MODULE_VERSION` 保持 1**：除非新 ABI 字段需要 bump（与 ADR-0029 D7 一致）
- **DL-isolated 测试保留**：`dlopen libptxemu_device.so` 无 libcudart.so 依赖下可独立调用所有 API（架构 §10 item 20）
- **in-flight unload 边界**：`ptxemu_image_unload` 对 in-flight handle 返回非 0 错误码（架构 §10 items 16/24）
- **跨仓 RFC 文档**：建立 `openspec/changes/<TBD>/rfc-hal-extension.md` 引用 TaskRunner ADR-035 R5.1 原文（roadmap v1 未核实 R5.1 实际内容，需先确认）

### MUST NOT（PTX-EMU 仓）

- **不引入跨仓依赖**：`CMakeLists.txt` 不加 TaskRunner 或 UsrLinuxEmu include path
- **不暴露内部 PTX-EMU 类型给 HAL**：HAL 只能调 `extern "C"` ABI；不能传 `PtxContext` 等 C++ 类型
- **不修改 `cpptlm_bridge.h`**（与 Phase 12.3.A 共享约束）
- **不修改 `libcudart.so` 任何 entry**（HAL 集成走 UsrLinuxEmu 仓独立路径）
- **不在 PTX-EMU 仓实施跨仓 commit 顺序**——本提案只规定 PTX-EMU 仓应做的兼容性验证

### MUST（跨仓 — 仅记录约束）

- **TaskRunner 仓零 PTX-EMU 链接依赖**（架构 §2 D8.1）：所有 PTX-EMU 调用经 UsrLinuxEmu HAL 边界封装
- **HAL 是 drv ↔ sim 唯一桥**（UsrLinuxEmu ADR-036）：TaskRunner → UsrLinuxEmu drv 的 System C ioctl 路径不能绕过 HAL
- **跨仓 commit 顺序**（per ADR-035 R5.1）：具体顺序由 RFC 确认后记录

### SHOULD

- 跨仓 RFC 包含 3 仓的 PR 链接 / commit hash（实施时填）
- 跨仓 RFC 包含 5 byte-identical gate 类比（PTX-EMU 仓侧验证 UsrLinuxEmu 调用不破坏 `libptxemu_device.so` ABI）
- PTX-EMU 仓侧提供 compatibility matrix 给 UsrLinuxEmu HAL 实现者参考

## 验收标准（架构层）

提案被批准后，guide-design → openspec proposal.md → tasks.md 时应明确：

1. **`libptxemu_device.so` 5 ABI 入口字节级不变**：`nm -D build/lib/libptxemu_device.so | grep ptxemu_` 输出与 Phase 1 ship 时完全一致
2. **DL-isolated 测试 PASS**：`tests/integration/test_cpptlm_module_dlopen.cpp` 在 Phase 13 完成后仍全部通过（无 libcudart.so 依赖）
3. **in-flight unload 边界 PASS**：`tests/integration/test_cpptlm_module_inflight.cpp` 覆盖并发 launch + in-flight unload 场景（架构 §10 items 17/24）
4. **跨仓 RFC 落地**：`openspec/changes/<TBD>/rfc-hal-extension.md` 引用 TaskRunner ADR-035 R5.1 原文 + UsrLinuxEmu ADR-036 三区分架构约束 + UsrLinuxEmu ADR-023 §D4
5. **PTX-EMU 仓 build system 0 跨仓污染**：`grep -r "UsrLinuxEmu\|TaskRunner" src/ include/ CMakeLists.txt` 输出为空
6. **CP 端 HAL 集成 acceptance（跨仓）**：TaskRunner 端到端 `cuModuleLoadData(image)` → `cuLaunchKernel` → `cuModuleUnload` 通过（架构 §10 item 23）

---

**注**：本提案只回答 PTX-EMU 仓的"为什么"和"什么"。跨仓实施细节由 UsrLinuxEmu 仓 + TaskRunner 仓各自的 improvement 提案 + OpenSpec change 维护。本提案的产物是 PTX-EMU 仓侧的兼容性保证 + 跨仓 RFC 文档。

详细实施 tasks 由 guide-design 评审通过后创建的 openspec `proposal.md` → tasks.md 维护。
