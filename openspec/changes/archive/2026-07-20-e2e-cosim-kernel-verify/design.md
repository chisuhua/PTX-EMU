## Context

### 现状

cpptlm-p1-ptxemu-shim 已归档（commit `63f10703`），shim + bridge dual-enqueue + `cpptlm_set_driver` ABI 已就绪，co-sim smoke test 通过。但缺少用**真实 CUDA kernel** 验证全链路的 E2E 测试。

### 目标

添加一个 CUDA vectorAdd kernel E2E 测试，通过 `BUILD_LIB_CPPTLM_CUDART=ON` 构建，验证：
1. `cudaLaunchKernel` bridge 路径 → `prepareKernelLaunchRequest()` → IR 正确
2. `GPUContext::exe_once()` 真实执行 PTX 指令
3. `on_complete` → `mark_complete` 回调链
4. GPU 内存输出与 CPU golden value 一致

## Decisions

### D1: test 位置

**决定**: `tests/e2e/cosim/` — 独立子目录，与现有 e2e 测试隔离。

**理由**: 这是 CppTLM 协同仿真专用测试，不应与普通 E2E 测试混在一起。`BUILD_LIB_CPPTLM_CUDART=OFF` 时此目录跳过编译。

### D2: kernel 选择

**决定**: `vectorAdd` — 最简单的有输入/输出数据的 kernel。

**理由**:
- 最小化 kernel 复杂度（~30 LOC CUDA）
- 有明确的 golden value（CPU `A[i] + B[i]`）
- 足够验证 LD/ST 指令 + 寄存器运算 + barrier sync
- 不依赖 tcgen05 或特殊指令

### D3: 构建条件

**决定**: `BUILD_LIB_CPPTLM_CUDART=ON` 时编译并运行；`OFF` 时测试目标不创建（`ctest -R` 无匹配）。

```cmake
if(BUILD_LIB_CPPTLM_CUDART)
    add_catch_test(e2e_cosim_vector_add
        cosim/test_cosim_vector_add.cpp
        cosim/kernel_vector_add.cu
    )
    set_tests_properties(e2e_cosim_vector_add PROPERTIES LABELS "e2e;cosim;cpptlm")
endif()
```

### D4: 测试流程

```
1. cudaMalloc host/device 内存
2. 初始化输入数据（CPU golden value 同步计算）
3. cudaMemcpy H→D
4. cudaLaunchKernel (bridge 路径: g_cpptlm_bridge != nullptr)
5. shim->advance(N) 驱动 GPUContext::exe_once() 执行 PTX 指令
6. cudaDeviceSynchronize (poll_kernel while-loop，等待 bridge 确认完成)
7. cudaMemcpy D→H
8. REQUIRE(host_output[i] == golden[i]) for all i
```

**关键约定**：
- step 5 的 `advance()` 是测试代码显式调用 `g_ptx_emu_driver_shim->advance()`——bridge 路径下 `cudaDeviceSynchronize` **只执行 poll_kernel 循环，不驱动 exe_once**。详见 [D6 - 执行驱动模型](#d6-执行驱动模型)。
- step 6 的 `cudaDeviceSynchronize` 仅等 bridge 确认 kernel 已从 pending queue 清除。
- kernel 完成性通过两个维度验证：(a) `cudaDeviceSynchronize` 返回（bridge 路径耗尽），(b) golden value 匹配（PTX 确实执行了）。不使用 `is_kernel_complete(kernel_id)` 直接断言，因为 `kernel_id` 由 `cudaLaunchKernel` 内部生成，测试代码无法获取（见 [I3 - kernel_id 不可达](#i3---kernel_id-不可达)）。

### D5: 验收标准

- `BUILD_LIB_CPPTLM_CUDART=ON` 构建通过
- ctests 全量 PASS（含新增 `e2e_cosim_vector_add`）
- `BUILD_LIB_CPPTLM_CUDART=OFF` 时测试目标不存在（`ctest -R e2e_cosim_vector_add` 返回 "No tests were found"）

### D6: 执行驱动模型

**决定**: 测试使用 **mock bridge + 测试内显式驱动 `advance()`** 模式。不依赖 CppTLM 外部仓库。

**理由**（`cudaDeviceSynchronize` bridge 路径的实际行为——`src/cudart/cudart_sim.cpp:936-972`）：

```
cudaDeviceSynchronize (bridge path): 仅 poll_kernel 循环，不调用 exe_once
  └─ while (g_pending_kernels not empty)
      └─ g_cpptlm_bridge->poll_kernel(id)
          若返回 0 → 从 pending 清除 → 循环终止
```

PTX 指令真实执行链：

```
PtxEmuDriverShim::advance()  ← 唯一调用 exe_once() 的入口
  └─ ctx_->exe_once()        ← 驱动 PTX 指令逐条执行
      └─ on_complete → g_ptx_emu_driver_shim->mark_complete(kernel_id)
```

**因此**，若 mock bridge 的 `poll_kernel` 立即返回 0，则 `cudaDeviceSynchronize` 立即退出，但 PTX 从未真正执行——golden value 必然不匹配。测试必须**在 `cudaDeviceSynchronize` 之前**显式调用 `g_ptx_emu_driver_shim->advance()` 驱动执行。

**测试调用链**：

> **前置条件**：`__cudaRegisterFatBinary`（由 `.cu` kernel 的 CUDA 启动序列触发）必须在 `cudaLaunchKernel` 之前执行，以初始化 `g_ptx_emu_driver_shim` 和 `g_gpu_context`。E2E 测试编译真实 `.cu` 文件，CUDA 驱动语义天然保证此顺序——但在理解 D6 模型时需显式认知。

```
test:
  1. cpptlm_attach_bridge(&mock)          // 设置 g_cpptlm_bridge = &mock
  2. cudaLaunchKernel(...)               // dual-enqueue: submit_kernel + GPUContext
  3. g_ptx_emu_driver_shim->advance(N)   // ★ 测试驱动 exe_once 循环
      → on_complete → mark_complete(kernel_id)
  4. cudaDeviceSynchronize()             // poll_kernel 确认完成
  5. cudaMemcpy D→H → golden compare     // 验证执行结果
```

**生产代码变更量**（支持此模式的前提）：

为让测试能访问 `g_ptx_emu_driver_shim`（当前为 `src/cudart/cudart_sim.cpp:137` 的 `static`），需做一处最小变更：

| 文件 | 变更 | LOC |
|------|------|:---:|
| `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` | 新增 `extern PtxEmuDriverShim* g_ptx_emu_driver_shim;` | +1 |
| `src/cudart/cudart_sim.cpp` | 移除第 137 行的 `static` 关键字 | ±0 |

总计 +1 行，零行为变更。`PtxEmuDriverShim` 已提供 `get_gpu_context()` + `advance()` 公开接口——此变更仅将指针可见性从 `static`（文件内）提升至 `extern`（全程序），不改变 `cudart` 库内任何逻辑。详见 [Non-Goals](#non-goals)。

> **注意**：`PtxEmuDriverShim` 的 `advance()` 内部 `while` 循环会在 kernel 完成后返回。但测试不应依赖此循环自动终止——当 `GPUContext::task_queue` 为空且状态非 RUNNING 时，`exe_once()` 可能 noop，导致死循环。因此 `advance(N)` 的 `max_cycles` 上限既是安全措施，也作为循环次数上限。实际实现需在 advance 后检查 GPUContext 状态确认 kernel 确实执行完毕。

### I3 - kernel_id 不可达

**问题**：`kernel_id` 由 `cudaLaunchKernel` 内部 `generate_kernel_id()` 生成（`cudart_sim.cpp:571`），不对外暴露。`PtxEmuDriverShim::is_kernel_complete(kernel_id)` 要求传入 `kernel_id`，测试代码无法获取。

**决策**：不在测试中直接断言 `is_kernel_complete`。Kernel 完成性通过以下方式间接验证：
1. `cudaDeviceSynchronize` 正常返回（bridge polling loop 耗尽 `g_pending_kernels`）
2. `cudaMemcpy D→H` 后的 golden value 完全匹配（证明 PTX 确实执行了）

这两个条件同时成立，等价于 kernel 正确完成。

## Non-Goals

- 不修改核心桥接逻辑（`cudaLaunchKernel` bridge 路径、`cudaDeviceSynchronize` bridge 路径、`PtxEmuDriverShim` 类行为）
- 仅做一处符号可见性变更（将 `g_ptx_emu_driver_shim` 从 `static` 提升为 `extern`，使测试可访问），详见 [D6 - 执行驱动模型](#d6-执行驱动模型)
- 不涉及 CppTLM 侧变更
- 不处理 multi-kernel / multi-stream 场景