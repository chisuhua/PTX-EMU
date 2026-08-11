# Design: ptxir-driver-api-front-door

## 现状问题

当前 `libcudart.so` 只有**单一 legacy front door**——`__cudaRegisterFatBinary` 处理链接后的可执行文件。架构文档 §2 §4.2 设计的"in-memory module loading front door"（4 个 Driver API 入口）在 `libcudart.so` 侧**完全未实现**——Oracle 实测 `nm -D build/lib/libcudart.so` 仅导出 `cuModuleLoad`（stub）与 `cuModuleGetFunction`（stub, line 513），缺 `cuModuleLoadData` / `cuModuleUnload` / 真 `cuLaunchKernel(CUfunction,...)` Driver API 版本。

副作用：CUDA Driver API 用户（TaskRunner、动态加载场景、CP 端跨仓集成）无法使用 PTX-EMU；`ptx_interpreter.cpp:100-140` mutation bug 在并发 launch 同一 image 时复发（ADR-0029 §触发事件-4）。

## 目标状态

`libcudart.so` 暴露 4 个 CUDA Driver API 入口（`cuModuleLoadData` / `cuModuleGetFunction` 真版本 / `cuLaunchKernel(CUfunction,...)` / `cuModuleUnload`），由 `ModuleRegistry`（线程安全 + mutex lock order）管理不透明 handle。复用 `PTXIRLoader::deserializeForCubin()` + `PtxContextAdapter`；per-launch fresh `PtxContext` 修复 mutation bug；6 类 image classifier + 7 类 error mapping 完备。

`libcudart.so` 与 `libptxemu_device.so` 路径解耦但执行后端共享（架构 §2）。`cpptlm_bridge.h` ABI 5 byte-identical gates 继续 PASS（ADR-0029 D7）。

## 影响范围

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `libcudart.so` 导出符号 | 新增 4 个 T 符号 | `cuModuleLoadData` / `cuModuleGetFunction`（替换 stub）/ `cuLaunchKernel(CUfunction,...)` / `cuModuleUnload` |
| `include/cudart/module_registry.h` | 新增 | `ModuleRecord` + `FunctionRecord` + `ModuleRegistry` 接口 |
| `src/cudart/module_registry.cpp` | 新增 | Registry 实现 + `std::mutex` + lock order 文档 |
| `src/cudart/image_classifier.cpp` | 新增 | 6 类纯函数 classifier |
| `src/cudart/cudart_sim.cpp` | 修改 | 新增 4 入口 + 错误映射（保持 legacy path 不变） |
| `include/cudart/cpptlm_bridge.h` | **不变** | ADR-0029 D7 — `CPPTLMBRIDGE_VERSION=2` |
| `libptxemu_device.so` 5 ABI | **不变** | `ptxemu_image_*` 已 ship |
| `__cudaRegisterFatBinary` legacy | **不变** | 架构 §4.1 保持独立 |
| `WarpContext` / `ThreadContext` / `GPUContext` 核心路径 | **不变** | 不添加新依赖（per `improvements/implement-ptxir-cubin-embed-extension.md`） |
| 新测试文件 | 新增 5 个 | `tests/unit/cudart/image_classifier_test.cpp` + `test_module_registry.cpp` + `test_module_registry_mode_independence.cpp` + `tests/integration/test_cuda_driver_api.cpp` + `test_in_memory_mutation.cpp` |

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 递归锁死锁（per `ptx-lessons-learned.md` §1） | 中 | **Oracle C1**: `ModuleRegistry` mutex 范围明确 + lock order vs per-`PtxContext` 锁文档化；任何持锁方法禁止调同锁其他 public 方法 |
| mutation bug 回归 | 中 | **Oracle C3**: 并发 launch 同一 image 1000 次，断言 SHA-256 不变 + 无 barrier-mask corruption |
| `cpptlm_bridge.h` ABI 破坏 | 低 | **Oracle C7**: archive 前 `git diff cpptlm_bridge.h` 必须为空 + `CPPTLMBRIDGE_VERSION=2` 测试 |
| driver-api 入口意外引入反序列化多路径 | 中 | **Oracle C4**: `grep -r "deserializeForCubin\|deserialize" src/cudart/` 仅 1 处 `deserializeForCubin` |
| 阻塞 Phase 12.4 multi-kernel manifest（`deserializeForCubin` 签名） | 低 | **multi-kernel C2**: 本 change 不改 `deserializeForCubin` 签名；Phase 12.4 硬串行启动 |
| `cuInit` / `cuCtx*` / packed-extra 缺失导致下游无法使用 | 中 | **Oracle C5**: 建 follow-up proposal task 显式记录 |
| 与 `libptxemu_device.so` 边界不清 | 低 | **Oracle C9 (Issue 3 fix)**: 场景 2 同进程双路径共存测试 + A8b `PTXIR_MODE=off` 独立性测试共同验证边界 |

## 关键约束 (MUST)

- 复用 `PTXIRLoader::deserializeForCubin()` 作为唯一反序列化入口
- 不读 `/proc/self/exe` / 不调 `cuobjdump` / 不读 `PTXIR_MODE`
- 失败路径不抛异常（per archive change `2026-08-07` 约束）
- 线程安全：`std::mutex` 覆盖 4 个 Driver API 入口
- per-launch fresh `PtxContext`（架构 §5.4 + ADR-0029 D3）
- DL-isolated：`libptxemu_device.so` 5 ABI 字节级不变
- WAR/CONTESTED 文件路径在 commit 拆分时**不跨 commit 触碰核心执行路径**

## 测试策略

按 `ptx-lessons-learned` §3，分 Phase commit：
- Commit 1: `ModuleRecord`/`FunctionRecord`/`Registry` + 单元测试
- Commit 2: `cuModuleLoadData` + `cuModuleGetFunction` + 6 类 image classifier
- Commit 3: `cuLaunchKernel(CUfunction)` + `cuModuleUnload` + 错误映射
- Commit 4: D3 mutation bug 复检 + ABI 稳定性回归
- Commit 5: integration tests + nm -D verify
- 单元测试 ≥13 项（A8-1..A8-13 枚举已写入 `proposal.md` Acceptance 段）