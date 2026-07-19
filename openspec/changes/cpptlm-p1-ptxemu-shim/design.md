## Context

### 现状问题

CppTLM P1 Phase 4 Wave 1 已完成 `IPtxEmuDriver` 窄接口定义（CppTLM 侧），当前 PTX-EMU 侧存在以下未闭合的协同仿真 seam：

1. **Bridge path 不生成 PTX 任务**：`cudaLaunchKernel` bridge 路径（`g_cpptlm_bridge != nullptr`）仅将 kernel 存到 `g_pending_kernels`（`cudart_sim.cpp:531-543`），不调用 `g_gpu_context->submit_kernel_request()`，kernel 不进入 `GPUContext::task_queue` → SM 不执行 → `poll_kernel` 立即返回 0。

2. **缺少驱动回调机制**：`cudaLaunchKernel` bridge 路径不设置 `KernelLaunchRequest::on_complete`，CppTLM 端无法获知 kernel 真实完成时间。

3. **跨仓库驱动接口未闭环**：CppTLM 定义 `IPtxEmuDriver` 但 PTX-EMU 侧无实现（`PtxEmuDriverShim`），`advance()`/`inject_*()`/`is_kernel_complete` 无后端。

4. **ABI 注册入口缺失**：`cpptlm_set_driver` 未声明，`initialize_environment()` 不创建 `PtxEmuDriverShim`。

5. **构建配置未就绪**：CppTLM `cpptlm_core` 未启用 PIC，PTX-EMU `CPPTLM_COMMIT_HASH` 未 pin 到已验证提交。

### 目标状态

```
PTX-EMU initialize_environment()
  ├─ 创建 GPUContext (已有)
  ├─ 创建 PtxEmuDriverShim (新增)
  ├─ cpptlm_set_driver(shim) (新增 ABI)
  └─ ...

CppTLM KernelLaunchTLM::tick()
  └─ driver_->advance(N, actual)
       └─ PtxEmuDriverShim::advance()
            └─ GPUContext::exe_once()
                 ├─ task_queue → SMContext 分配
                 ├─ SMContext::exe_once()
                 │    ├─ Step A: sb->allocate() (CppTLM ScoreboardTLM)
                 │    ├─ execute_warp_instruction()
                 │    ├─ Step B: pipeline->get_frac() (CppTLM PipelineTLM)
                 │    └─ Step C: sb->release() (CppTLM ScoreboardTLM)
                 └─ on_complete → mark completion map

cudaLaunchKernel (bridge path)
  ├─ kernel_id = generate_kernel_id()
  ├─ g_cpptlm_bridge->submit_kernel(kernel_id, ...)  (保持)
  ├─ GPUContext::submit_kernel_request({              (新增: 双路径 enqueue)
  │    on_complete = [kid]{ shim->mark_complete(kid); }
  │  })
  └─ return cudaSuccess
```

## Goals / Non-Goals

**Goals:**
- 实现 `PtxEmuDriverShim`：`IPtxEmuDriver` 的 PTX-EMU 端完整实现
- 修复 bridge path：`cudaLaunchKernel` 同时 enqueue 到 `GPUContext::task_queue`
- 新增 `cpptlm_set_driver` ABI 入口：跨 `.so` 边界驱动注册
- 构建修复：`cpptlm_core` PIC + pin commit

**Non-Goals:**
- 不修改 `SMContext::exe_once()` 的 3-step 注入逻辑（HSK-5 `367fd6a5` 已落地）
- 不修改 `IPtxEmuDriver` 接口定义（CppTLM 侧 `ptx_emu_driver.hh` 已完成）
- 不涉及 Phase 4 Wave 2 验证（G-D2/G-D3/G-D5/G-D8）
- 不处理 IAsyncCompletion（Phase 9+）
- 不修改 PTX-EMU 侧已归档的 `cpptlm-phase8b-injection-points` change

## Decisions

### D1: PtxEmuDriverShim 位置

**决定**: `src/cudart/cpptlm_bridge/PtxEmuDriverShim.{h,cpp}`

**理由**: 与 `cudart_sim.cpp` 同属 `src/cudart/`（CUDA Runtime 替代实现），与 bridge 基础设施共目录。CppTLM 端 `IPtxEmuDriver` 头文件通过 `ExternalProject_Add` 的 `cpptlm-install/include/` 可用。

**替代方案**: `src/ptxsim/core/PtxEmuDriverShim` — 但 `IPtxEmuDriver` 是 cudart 层概念（驱动执行引擎），非模拟器核心逻辑。

### D2: advanced() 实现

**决定**: 循环调用 `GPUContext::exe_once()` 直到 `max_cycles` 或 `get_state() == EXIT`。

```cpp
AdvanceResult advance(uint32_t max_cycles, uint32_t& actual) override {
    if (!ctx_) return AdvanceResult::Error;
    try {
        while (actual < max_cycles && ctx_->get_state() != EXIT) {
            ctx_->exe_once();
            ++actual;
        }
        if (ctx_->get_state() == EXIT) {
            std::lock_guard lock(mu_);
            for (auto& [kid, done] : completion_) done = true;
        }
    } catch (...) { return AdvanceResult::Error; }
    return actual > 0 ? AdvanceResult::Executed : AdvanceResult::NoOp;
}
```

**理由**:
- 1:1 cycle 映射：1 `exe_once()` = 1 PTX-EMU cycle（所有 SM 各推进 1 步）
- 异常安全：捕获 `exe_once()` 异常 → `AdvanceResult::Error`
- EXIT 检测：kernel 完成时自动标记所有 completion_

### D3: Kernel 注入 + on_complete 回调

**bridge path 改造**:
```cpp
// cudart_sim.cpp cudaLaunchKernel bridge 路径:
if (g_cpptlm_bridge) {
    uint64_t kernel_id = generate_kernel_id();
    // ... args copy + bridge submit ...
    g_cpptlm_bridge->submit_kernel(kernel_id, ...);

    // 新增: enqueue 到 GPUContext 以驱动真实 PTX 执行
    KernelLaunchRequest ptx_req;
    ptx_req.kernel_name  = kernel_name;
    ptx_req.grid_dim     = gridDim3;
    ptx_req.block_dim    = blockDim3;
    ptx_req.shared_mem   = sharedMem;
    ptx_req.args         = std::move(args_deep_copy);
    ptx_req.on_complete  = [kernel_id, driver = g_ptx_emu_driver]() {
        if (driver) driver->mark_complete(kernel_id);
    };
    g_gpu_context->submit_kernel_request(std::move(ptx_req));
}
```

**理由**:
- 统一 `kernel_id`：bridge 路径与 GPUContext 使用相同 ID（通过 `PTXInterpreter::launchPtxInterpreter` 自动获取 `func2name`）
- `shared_ptr` 单次深拷贝：args_copy 在 bridge 路径完成深拷贝后 move 到 `KernelLaunchRequest`
- `on_complete` 回调：`GPUContext` 在 kernel 执行结束时调用 → `PtxEmuDriverShim::mark_complete()` → CppTLM `is_kernel_complete` 返回 true

### D4: cpptlm_set_driver ABI 入口

**决定**: 与 `cpptlm_attach_bridge` 同模式 — `extern "C" PTXEMU_BRIDGE_API`，全局指针 `g_ptx_emu_driver` 定义在 `cudart_sim.cpp`，声明在 `cpptlm_bridge.h`。

```cpp
// cpptlm_bridge.h 新增:
namespace tlm { class IPtxEmuDriver; }
extern "C" PTXEMU_BRIDGE_API void cpptlm_set_driver(tlm::IPtxEmuDriver* driver);
```

`initialize_environment()` 调用时机：在 `g_gpu_context` 创建后，`PtxEmuDriverShim` 构造完成立即调用。

**理由**:
- `IPtxEmuDriver` 是 C++17 纯虚接口（`unique_ptr`/`uint32_t`/`uint64_t`），跨 `.so` ABI 安全
- 与 `cpptlm_attach_bridge` 同生命周期，同 TU 定义确保初始化顺序

### D5: SM 数量决定注入方式

**决定**: `main.cpp` 通过 `driver->num_sms()` 查询 SM 数，逐个 `inject_scoreboard(sm_id, unique_ptr)`。PTX-EMU 侧 shim 内部维护 `vector<unique_ptr<IScoreboard>>`。

**理由**: 
- `SMContext` 的 3 setter 已在 `sm_context.h:67-75` 定义（inline），shim 仅做转发
- `SMContext` 仅持原始指针（`scoreboard_` `pipeline_provider_` `tensor_core_timing_`），shim 负责生命周期（`vector<unique_ptr>`）

### D6: CppTLM 构建修复

```cmake
# PTX-EMU CMakeLists.txt:
set(CPPTLM_COMMIT_HASH "73e5422" CACHE STRING "CppTLM git tag/commit to pin")
# ExternalProject_Add 新增: CppTLM 侧启用 PIC
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON
```

**理由**: 
- `73e5422` 是 CppTLM P0 + P1 Phase 1 已验证提交（包含 `MemoryBridge` + 3 核心模块 + 12 端点 static_assert）
- PIC 是链接 `.so` 的必要条件

## 端到端数据流

```
cudaLaunchKernel (PTX-EMU bridge path)
  ├─ kernel_id = generate_kernel_id()               # 统一 ID
  ├─ shared_ptr deep-copy args (单次)
  ├─ g_cpptlm_bridge->submit_kernel(kernel_id, ...)  # CppTLM 异步提交
  ├─ g_gpu_context->submit_kernel_request({           # PTX-EMU 任务入队
  │    .args         = 已深拷 args,
  │    .grid_dim     = gridDim,
  │    .block_dim    = blockDim,
  │    .shared_mem   = sharedMem,
  │    .on_complete  = [kid]{ g_ptx_emu_driver->mark_complete(kid); }
  │  })
  └─ return cudaSuccess

CppTLM KernelLaunchTLM::tick()
  └─ driver_->advance(MAX_PTX_STEPS, actual)
       └─ PtxEmuDriverShim::advance(max, &actual)
            └─ while actual < max && ctx_->get_state() != EXIT:
                 ctx_->exe_once()
                   ├─ task_queue → SM admission
                   ├─ for each SM in RUN: sm->exe_once()
                   │    ├─ Step A: sb->allocate(reg, warp)
                   │    ├─ execute_warp_instruction()
                   │    ├─ Step B: pipeline->get_fractional_cycles()
                   │    └─ Step C: sb->release(reg, warp)
                   ├─ kernel done → on_complete() → mark_complete()
                   └─ ++actual
            └─ return Executed/NoOp/Error

CppTLM KernelLaunchTLM::tick() (续)
  └─ while pending_ not empty:
       if driver_->is_kernel_complete(front.kernel_id)
         pending_.pop_front()
```

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| R1: bridge path args 深拷贝 double-copy | 中 | 低 | `args_copy` 在 bridge 路径深拷后 move 到 `KernelLaunchRequest`，`GPUContext` 持所有权 |
| R2: `on_complete` 回调时序（exe_once 返回前触发的回调） | 低 | 中 | `mark_complete` 仅设置 atomic bool，CppTLM 在 next tick 才检查，无 race |
| R3: `g_ptx_emu_driver` 初始化为空导致 bridge path 跳过 enqueue | 低 | 高 | `initialize_environment()` 中确保 driver 创建在 bridge setup 之前；空 driver 检查防止空指针 |
| R4: CppTLM `CPPTLM_COMMIT_HASH` 未 pin 导致 ABI 不匹配 | 低 | 高 | pin 到 `73e5422`（已验证 P0+P1 Phase 1）；CppTLM 12 端点 static_assert 编译期拦截 |
| R5: `inject_*()` 在 sm_id 越界时的行为 | 低 | 中 | shim 检查 `sm_id < num_sms()`，越界 `return` 不崩溃 |
| R6: unique_ptr 所有权转移后 CppTLM main.cpp 生命周期 | 低 | 低 | shim 在 `vector<unique_ptr>` 中持有，shim 析构时自动释放 |