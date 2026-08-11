# Design: hal-extension-ptxemu-usrlinu-emu-taskrunner

## 现状问题

ADR-0029 §D8 描述 HAL 方案 D8 的 CP 端跨仓集成路径：TaskRunner 仓的 CUDA driver LD_PRELOAD shim（`libcuda_shim`）通过 `IGpuDriver` → `GpuDriverClient` → System C ioctl 间接调 UsrLinuxEmu 仓的 GPU 驱动，UsrLinuxEmu 仓的 HAL 层（`hal_user.cpp`）dlsym `libptxemu_device.so` 暴露的 `ptxemu_image_*` 函数，最终调 PTX-EMU。

PTX-EMU 仓目前**没有**显式的跨仓 RFC；`libptxemu_device.so` 5 ABI 已 ship 但需要冻结以避免破坏跨仓兼容性；DL-isolated 测试 + in-flight unload 边界测试部分存在需要扩展覆盖。

## 目标状态

PTX-EMU 仓侧保证：
- `libptxemu_device.so` 5 ABI 入口字节级不变
- SONAME / symlinks 不变
- DL-isolated 测试 + in-flight unload 边界测试覆盖完备
- 新增跨仓 RFC 文档引用 ADR-0029 §D8、TaskRunner ADR-035 R5.1、UsrLinuxEmu ADR-036
- CMake build system 0 跨仓污染（无 TaskRunner / UsrLinuxEmu include 路径）

PTX-EMU 仓**不**实现跨仓 commit 顺序；不修改 `cpptlm_bridge.h` ABI；不修改 `libcudart.so` 任何 entry。

## 影响范围

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `include/cudart/cpptlm_module.h` | **不变** | 5 ABI 入口已 ship，需冻结 |
| `libptxemu_device.so` ABI | **不变** | 已 ship |
| `libptxemu_device.so` SONAME | **不变** | `libptxemu_device.so.12` |
| `openspec/changes/hal-extension-ptxemu-usrlinu-emu-taskrunner/rfc-hal-extension.md` | 新增 | 跨仓 RFC 文档 |
| `tests/integration/test_cpptlm_module_dlopen.cpp` | 扩展 | DL-isolated 测试覆盖增强 |
| `tests/integration/test_cpptlm_module_inflight.cpp` | 扩展 | in-flight unload 边界测试 |
| `CMakeLists.txt` / `src/` / `include/` | **不变** | 无跨仓 include |
| `cpptlm_bridge.h` | **不变** | 与 Phase 12.3.A 共享约束 |
| `libcudart.so` | **不变** | HAL 集成路径在 UsrLinuxEmu 仓独立 |

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| `libptxemu_device.so` ABI 意外变更 | 低 | nm 审计 + git diff byte-level 验证 |
| 跨仓污染（意外引入 TaskRunner/UsrLinuxEmu 路径） | 中 | **Oracle C1**: archive 前 `grep -r "UsrLinuxEmu\|TaskRunner" src/ include/ CMakeLists.txt` 必须为空 |
| RFC 内容缺失（未引用 ADR-035 R5.1 / ADR-036） | 低 | **Oracle C2**: RFC 必须显式引用 ADR-0029 §D8 + ADR-035 R5.1 + ADR-036 |
| PTX-EMU 仓越界承担跨仓协调 | 中 | **Oracle C3**: RFC + proposal 含 "PTX-EMU 仓不拥有跨仓协调责任" 显式声明 |
| 跨仓 commit 顺序未对齐 | 中 | RFC 引用 ADR-035 R5.1（具体顺序由 TaskRunner 仓 integrator 决定） |

## 关键约束 (MUST)

- 5 ABI 入口字节级不变
- SONAME 不 bump（`libptxemu_device.so.12`）
- DL-isolated: dlopen 无 libcudart.so 依赖可调 5 API
- in-flight unload 返回非 0 错误码
- 不引入跨仓 include path
- 不暴露 C++ 类型给 HAL（仅 `extern "C"` ABI）
- 不修改 `cpptlm_bridge.h`
- 不修改 `libcudart.so`

## 测试策略

- DL-isolated: 已有 `tests/integration/test_cpptlm_module_dlopen.cpp`（扩展覆盖）
- in-flight unload: 已有 `tests/integration/test_cpptlm_module_inflight.cpp`（扩展覆盖）
- 5 ABI nm 验证: `nm -D build/lib/libptxemu_device.so | grep ptxemu_` 输出与 baseline 比对
- 跨仓 e2e: 由 TaskRunner 仓实施（不在本仓范围）