# hal-extension-ptxemu-usrlinu-emu-taskrunner

> **Oracle 评审结果（2026-08-10）**: ✅ APPROVE-WITH-CONDITIONS — 风险 LOW
> **3 个硬性条件**: 在 RFC 中明示非 in-scope 工作 + grep 验证 + 明确 Out of Scope 行

## Why

ADR-0029 §D8 描述 HAL 方案 D8 的 CP 端跨仓集成路径：TaskRunner 仓的 CUDA driver LD_PRELOAD shim（`libcuda_shim`）通过 `IGpuDriver` → `GpuDriverClient` → System C ioctl 间接调 UsrLinuxEmu 仓的 GPU 驱动，UsrLinuxEmu 仓的 HAL 层（`hal_user.cpp`）dlsym `libptxemu_device.so` 暴露的 `ptxemu_image_*` 函数，最终调 PTX-EMU。

3 个仓的耦合关系：
- **TaskRunner** — CUDA driver LD_PRELOAD shim；CUDA Driver API 用户；调 `cuModuleLoadData`/`cuLaunchKernel`/`cuModuleUnload`
- **UsrLinuxEmu** — GPU 驱动 + HAL；System C ioctl + HAL fn-ptr 边界；**HAL 是 drv ↔ sim 唯一桥**（per UsrLinuxEmu ADR-036 三区分架构硬约束）
- **PTX-EMU** — device-side executor；暴露 `libptxemu_device.so` 5 `extern "C"` ABI 入口（已 ship）

**关键架构约束（D8.1）**：**TaskRunner 仓零 PTX-EMU 链接依赖**。所有 PTX-EMU 调用经 UsrLinuxEmu HAL 边界封装。

## What Changes

**In Scope（PTX-EMU 仓）**:
- 保证 `libptxemu_device.so` 5 `ptxemu_image_*` ABI 入口签名稳定
- 保证 `libptxemu_device.so` SONAME / symlinks 不变
- 新增 1 个 PTX-EMU 仓内跨仓 RFC 文档（引用 TaskRunner ADR-035 R5.1 原文）
- DL-isolated 测试 + in-flight unload 边界测试（扩展覆盖）

**Out of Scope（PTX-EMU 仓）**:
- **PTX-EMU 仓不引入 TaskRunner 或 UsrLinuxEmu include 路径**（Oracle 条件 #1：grep 验证）
- **PTX-EMU 仓不修改 `cpptlm_bridge.h` ABI**（与 Phase 12.3.A 共享同一约束）
- **PTX-EMU 仓不实现 `cuModuleLoadData` 等 CUDA Driver API**（Phase 12.3.A 范围）
- **PTX-EMU 仓不实现新的 HAL ioctl handler**（UsrLinuxEmu 仓范围）
- **PTX-EMU 仓不修改 `libcudart.so` 任何 entry**
- **PTX-EMU 仓不拥有跨仓协调责任**（Oracle 条件 #3：commit 顺序是 integrator 责任，PTX-EMU 仅在 RFC 中记录）

### 关键场景

#### 场景 1：TaskRunner UMD 端到端

- **GIVEN** TaskRunner 仓 `libcuda_shim` 加载，`cuModuleLoadData(image)` 被 CUDA 应用代码调用
- **WHEN** cuModuleLoadData → IGpuDriver::load_kernel_module → GpuDriverClient → System C ioctl → UsrLinuxEmu drv → UsrLinuxEmu HAL fn-ptr #66 → hal_user.cpp dlsym ptxemu_image_load → PTX-EMU executor → 返回 handle
- **THEN** handle 可用于 cuLaunchKernel/cuModuleUnload；kernel 实际执行

#### 场景 2：跨仓 commit 顺序（per TaskRunner ADR-035 R5.1）

- **GIVEN** 3 仓按依赖顺序 ship：UsrLinuxEmu → PTX-EMU 验证 → TaskRunner
- **WHEN** 任意一仓未就绪时启动下一仓
- **THEN** 下一仓编译/测试会失败；防止提前 commit 引入死锁/未解析符号

#### 场景 3：in-flight unload returns busy

- **GIVEN** 同一 CUmodule 的 kernel 正在执行（in-flight）
- **WHEN** cuModuleUnload(module) 被调用
- **THEN** 返回 CUDA_ERROR_INVALID_HANDLE（busy）

## Capabilities

- **`libptxemu_device.so` 5 ABI 入口签名不变**：`ptxemu_image_load` / `ptxemu_image_kernel_name` / `ptxemu_image_execute` / `ptxemu_image_unload` / `ptxemu_module_version`
- **`libptxemu_device.so` SONAME 不变**：`libptxemu_device.so.12` 主版本号不 bump
- **`CPPTLM_MODULE_VERSION` 保持 1**
- **DL-isolated 测试保留**
- **in-flight unload 边界**：`ptxemu_image_unload` 对 in-flight handle 返回非 0 错误码
- **跨仓 RFC 文档**：建立 `openspec/changes/<this>/rfc-hal-extension.md`

## Impact

- **`libptxemu_device.so` ABI 不变**（already shipped）
- **`cpptlm_bridge.h` ABI 不变**
- **`libcudart.so` 不变**（HAL 集成走 UsrLinuxEmu 仓独立路径）

## Acceptance

### Oracle 评审通过条件
- [ ] **C1**: grep 验证 `CMakeLists.txt` / `src/` / `include/` 无 TaskRunner/UsrLinuxEmu include 路径（archive 前 re-verify）
- [ ] **C2**: RFC 引用 ADR-0029 §D8、TaskRunner ADR-035 R5.1 和 UsrLinuxEmu ADR-036 作为**外部依赖**，非 in-scope 工作
- [ ] **C3**: RFC 与 proposal 含显式 **"PTX-EMU 仓不拥有跨仓协调责任；commit 顺序是 integrator 责任"** 一行

### PTX-EMU 仓交付物
- [ ] **`libptxemu_device.so` 5 ABI 入口字节级不变**：`nm -D build/lib/libptxemu_device.so | grep ptxemu_` 输出与 Phase 1 ship 时完全一致
- [ ] **DL-isolated 测试 PASS**：`tests/integration/test_cpptlm_module_dlopen.cpp` 在 Phase 13 完成后仍全部通过
- [ ] **in-flight unload 边界 PASS**：`tests/integration/test_cpptlm_module_inflight.cpp` 覆盖并发 launch + in-flight unload 场景
- [ ] **跨仓 RFC 落地**：`openspec/changes/hal-extension-ptxemu-usrlinu-emu-taskrunner/rfc-hal-extension.md`
- [ ] **PTX-EMU 仓 build system 0 跨仓污染**：`grep -r "UsrLinuxEmu\|TaskRunner" src/ include/ CMakeLists.txt` 输出为空

### 跨仓 acceptance（不在本仓实施）
- [ ] TaskRunner 端到端 cuModuleLoadData(image) → cuLaunchKernel → cuModuleUnload 通过（由 TaskRunner 仓负责）
