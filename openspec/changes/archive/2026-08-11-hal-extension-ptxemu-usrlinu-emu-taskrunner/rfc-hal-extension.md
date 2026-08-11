# RFC: HAL Extension Cross-Repo Integration (D8)

> **PTX-EMU 仓不拥有跨仓协调责任；commit 顺序是 integrator 责任**（per Oracle C3）。
> 本 RFC 仅记录 PTX-EMU 仓侧的 ABI 冻结承诺与跨仓协议指针，**不**包含 in-scope 工作。
> 跨仓 commit 顺序由 TaskRunner 仓 integrator 决策。

## 背景

ADR-0029 §D8 描述 HAL 方案 D8：CUDA Driver API 用户（TaskRunner）通过
UsrLinuxEmu HAL 边界调用 PTX-EMU executor。

```
TaskRunner (libcuda_shim)        ← CUDA Driver API 调用方
   ↓ cuModuleLoadData/cuLaunchKernel
UsrLinuxEmu (drv + HAL hal_user.cpp)   ← System C ioctl + HAL fn-ptr #66
   ↓ dlsym ptxemu_image_*
PTX-EMU (libptxemu_device.so)   ← device-side executor
```

3 仓耦合关系 + commit 顺序约束见 TaskRunner ADR-035 §R5.1（canonical）。

## 外部依赖（PTX-EMU 仓**不**实施，仅引用）

- **TaskRunner ADR-035 §R5.1**：canonical 跨仓协议 + commit 顺序
  (UsrLinuxEmu → PTX-EMU 验证 → TaskRunner)
- **UsrLinuxEmu ADR-036**：HAL 是 drv ↔ sim 唯一桥（三区分架构硬约束）
- **ADR-0029 §D8**：CP 端跨仓集成路径

## PTX-EMU 仓承诺（freeze）

| 项目 | 承诺 | 验证 |
|------|------|------|
| `libptxemu_device.so` 5 ABI 入口 | 签名不变 | nm -D baseline 比对 |
| `libptxemu_device.so.12` SONAME | 不 bump | `nm -D ... | grep SO` |
| `CPPTLM_MODULE_VERSION` | 保持 1 | `grep CPPTLM_MODULE_VERSION` |
| DL-isolated 测试 | 全部 PASS | `ctest -R dlopen` |
| in-flight unload | 非 0 错误码 | `ctest -R inflight` |

## 跨仓污染约束（Oracle C1）

PTX-EMU 仓**不**引入 TaskRunner 或 UsrLinuxEmu include 路径：

```bash
grep -r "UsrLinuxEmu\|TaskRunner" src/ include/ CMakeLists.txt
# 必须输出为空
```

## 范围边界

**PTX-EMU 仓不**：
- 实现 `cuModuleLoadData` 等 CUDA Driver API（Phase 12.3.A 范围）
- 实现 HAL ioctl handler（UsrLinuxEmu 仓范围）
- 修改 `cpptlm_bridge.h` ABI（与 Phase 12.3.A 共享冻结约束）
- 修改 `libcudart.so` 任何 entry
- 拥有跨仓协调责任

**PTX-EMU 仓**：
- 冻结 `libptxemu_device.so` 5 ABI 入口字节级不变
- 维护 DL-isolated 测试 + in-flight unload 边界测试
- 维护 nm baseline 防止 ABI 意外变更

## 跨仓 e2e acceptance（不在本仓）

TaskRunner 仓端到端 cuModuleLoadData(image) → cuLaunchKernel → cuModuleUnload
由 TaskRunner 仓实施 + 验证。

## 历史

| 版本 | 日期 | 改动 |
|------|------|------|
| v1.0 | 2026-08-11 | 初版（Phase 13 ship） |
