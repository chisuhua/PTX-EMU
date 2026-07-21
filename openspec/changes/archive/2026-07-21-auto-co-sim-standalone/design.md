## Context

### 现状

三个已知缺陷阻止标准 CUDA 程序在 bridge 路径下正常运行：

1. **arg-count segfault**（ADR-0021 L2）：bridge path `cudaLaunchKernel:574` 的 `count_kernel_args` 哨兵遍历假设全指针 args
2. **2-cycle completion**（ADR-0021 L1）：`gpu_context.cpp:246-336` 在 bridge 路径下 admit kernel + 判 EXIT 同 cycle
3. **手动管理 gap**：`g_cpptlm_bridge` 必须手动 attach，`advance()` 必须显式调用

### 目标

修复以上三点 + 增加 bridge 自动初始化 + `cudaDeviceSynchronize` 自动 advance，使标准 CUDA 程序（纯 `cudaMalloc`/`cudaMemcpy`/`<<<>>>`/`cudaDeviceSynchronize`）在 `BUILD_LIB_CPPTLM_CUDART=ON` 下自动走 bridge 路径完成 PTX 执行。

### 适用范围

仅修改 bridge 路径，不修改 `cpptlm_bridge.h` ABI、不修改同步路径、不修改 `prepareKernelLaunchRequest` IR 构造逻辑。

## Decisions

### D1: StubBridge 内部实例

**决定**：在 `initialize_environment()`（`cudart_sim.cpp:292-310`）创建 `PtxEmuDriverShim` 的同时，创建内部 `StubBridge` 实例并设置 `g_cpptlm_bridge = &stub`。

```cpp
// PtxEmuDriverShim 初始化后（cudart_sim.cpp:310之后）
#ifdef BUILD_LIB_CPPTLM_CUDART
    // 仅在用户未显式 override 时启用 StubBridge（允许测试注入 mock bridge）
    if (!g_bridge_user_override) {
        static StubBridge stub_bridge;
        g_cpptlm_bridge = &stub_bridge;
    }
#endif
```

**`StubBridge` 类体**：

```
class StubBridge : public CppTLMBridge {
    mutable std::mutex mu_;  // 保护 submitted_ids_（与 PtxEmuDriverShim 一致的线程安全模式）
    std::unordered_set<uint64_t> submitted_ids_;  // 记录所有 submit 过的 kernel_id
    int version() const override { return CPPTLMBRIDGE_VERSION; }
    int submit_kernel(...) override {
        std::lock_guard<std::mutex> lock(mu_);
        submitted_ids_.insert(kernel_id);
        return 0;
    }
    uint64_t poll_kernel(uint64_t kid) override {
        std::lock_guard<std::mutex> lock(mu_);
        return submitted_ids_.count(kid) ? 0 : UINT64_MAX;  // 未知 id → 错误码
    }
    int synchronize_stream(uint64_t) override { return 0; }
    uint64_t global_access(uint64_t, uint64_t, uint8_t) override { return 0; }
};
```

**`g_bridge_user_override` 标志**：
- `cpptlm_attach_bridge` 设置 `g_bridge_user_override = true; g_cpptlm_bridge = bridge;`
- `cpptlm_detach_bridge` 设置 `g_bridge_user_override = false; g_cpptlm_bridge = nullptr;`
- `initialize_environment` 检查 `!g_bridge_user_override` 后才 auto-attach StubBridge

**理由**：
- `submitted_ids_` 确保 `poll_kernel(UINT64_MAX)` 在未知 kernel_id 时语义正确（`cpptlm_bridge.h:113`）
- `g_bridge_user_override` 允许测试注入 mock bridge 而不被 `initialize_environment` 覆盖（Oracle Q2 Issue 3）
- `static` 单例生命周期覆盖整个程序，安全
- `mutable std::mutex mu_` 保护 `submitted_ids_`，与 `PtxEmuDriverShim`（`PtxEmuDriverShim.h:55`）保持一致的线程安全模式。当前 host 端单线程模型下无害，但预防未来 CppTLM 多线程调用场景的 UB
- 无外部 CppTLM 依赖（`BUILD_LIB_CPPTLM_CUDART=ON` 时不依赖 `libcpptlm_cudart.so`）
- 标准 CUDA 程序无需 `cpptlm_attach_bridge`
- `poll_kernel` 返回 0 为正确语义（`advance()` 已在 `cudaDeviceSynchronize` 中先行执行完毕）
- `global_access` 返回 0 为零延迟（stub 不建模 NoC）

**替代方案**（拒绝）：
- **方案 B**：CppTLM 侧提供默认 `StubBridge` → PTX-EMU 单边即可，无需 CppTLM 仓库参与
- **方案 C**：`__cudaRegisterFatBinary` 时 attach → `initialize_environment()` 更早调用，为避免 race condition 选同一点

### D2: cudaDeviceSynchronize + cudaStreamSynchronize auto advance

**决定**：在 bridge path polling loop 前新增 `advance()` 调用。扩展至 `cudaStreamSynchronize(0)`（默认 stream）。

```cpp
// cudart_sim.cpp:936 — cudaDeviceSynchronize / cudaStreamSynchronize bridge path 入口
if (g_cpptlm_bridge) {
    // ★ Phase D: auto-advance — 驱动 PTX 执行
    if (g_ptx_emu_driver_shim) {
        uint32_t actual = 0;
        uint32_t max_cycles = get_max_advance_cycles();  // configurable ceiling
        g_ptx_emu_driver_shim->advance(max_cycles, actual);
    }

    // then poll
    while (true) {
        // ... 原有 poll_kernel 循环 ...
    }
}
```

**advance ceiling 安全机制**：`get_max_advance_cycles()` 返回可配置上限，防止病态 kernel（无限循环 / barrier 死锁）导致永久挂起。

- 环境变量 `PTX_EMU_MAX_ADVANCE_CYCLES`（默认 10,000,000，约等价于 10M GPU 周期）
- 若 advances 耗尽上限且 GPUContext 仍非 EXIT，**先清理执行状态**：清空 GPUContext 的 `executing_requests`、擦除 `g_pending_kernels` 中对应该 kernel 的条目、重置 SM 状态为 IDLE。然后记录 PTX_ERROR_EMU 并 `return cudaErrorUnknown`
- 每次 `cudaDeviceSynchronize` / `cudaStreamSynchronize` 调用独立计算 ceiling（不累计跨调用）
- 合法 kernel 可多次 sync 累积推进：第一次 sync 跑 N cycles 后返回 error，第二次 sync 从 N 继续推进到完成
- 默认 10M cycles 覆盖正常 kernel（实测 vectorAdd ~100 cycles，GEMM ~10k cycles）

**理由**：
- 标准 CUDA 程序不显式调用 `advance()`
- `advance(max_cycles)` 内部 while 循环 `ctx_->get_state() != EXIT` 硬终止条件 + ceiling 上限双重保障
- `cudaStreamSynchronize(0)` 必须同样 auto-advance——标准 CUDA 程序常调用 stream sync 而非 device sync（Oracle Q2 Issue 4）
- 保持对称性：同步路径 `wait_for_completion()` 也是在调用点内驱动执行

**替代方案**（拒绝）：
- **方案 B**：`cudaLaunchKernel` bridge 路径自动 advance → 破坏异步语义（bridge 路径设计目标就是不阻塞 launch）
- **方案 C**：新增 `cpptlm_advance_and_sync` API → 增加 ABI，违反"标准 CUDA 程序"目标

### D3: count_kernel_args 修复

**决定**：bridge 路径 args deep-copy 前从 PTX context 的 `kernelParams.size()` 获取权威参数计数，用 `SIZE_MAX` 作为"未找到"哨兵以区分"无参 kernel"和"PTX context 查询失败"。

```cpp
// 修复前
size_t arg_count = count_kernel_args(args);

// 修复后
size_t arg_count = SIZE_MAX;  // sentinel: PTX context 未找到
if (g_ptx_interpreter) {
    auto& ptx = g_ptx_interpreter->get_ptx_context();
    for (auto& kc : ptx.ptxKernels) {
        if (kc.kernelName == kernel_name) {
            arg_count = kc.kernelParams.size();
            break;
        }
    }
}
if (arg_count == SIZE_MAX) {
    arg_count = count_kernel_args(args);  // fallback: PTX context 查询失败
}
```

**理由**：`kernelParams` 是 PTX 解析器提供的权威参数签名，不依赖运行时 args 数组布局。`SIZE_MAX` 哨兵区分"无参 kernel"（`kernelParams.size()==0` 是合法值）与"PTX context 未初始化/kernel 未找到"（应走 fallback），避免无参 kernel 误触发 sentinel walk。Fallback 保持向后兼容 PTX context 未初始化场景。

### D4: 2-cycle completion 修复（待 Phase 0 调试定位）

**决定**：分 Phase 0 调试定位 + Phase 2 实施修复。

**候选方案**（按可能性排序，待 Phase 0 确认）：

- **方案 A**（可能性最高）：`exe_once()` 分离 admit 与 execute——`execute_kernel_internal` 后设 flag "just_admitted=true"，跳过同 cycle 的 `all_finished` 判定。下次 `exe_once()` 时 flag 复位
- **方案 B**：修复 `all_warps_finished()` 误判——确认 `add_thread` 后 thread `is_exited()` 为 false，`active_count > 0`
- **方案 C**：`update_state() -> sm_state = RUN` 在 `add_block` 后立即对 SM 生效，但 `exe_once()` 应确保至少执行一个 cycle

**Phase 0 步骤**：
1. 在 `gpu_context.cpp:246` / `sm_context.cpp:350` / `warp_context.cpp:414` 加运行时日志
2. 复现（bridge 路径 + vectorAdd 64 线程）
3. Oracle 咨询确认根因
4. 输出诊断报告

### D5: 测试退化为标准 CUDA 程序

**决定**：`test_cosim_vector_add.cu` 删除 `MockBridge` 类 + `cpptlm_attach_bridge` + `advance()` 代码，退化为仅含 `vectorAdd<<<>>>` + `cudaDeviceSynchronize` + golden compare 的标准 CUDA 程序。

**理由**：D1（auto-attach）+ D2（auto-advance）消除了测试手动管理的需要。测试变为纯 CUDA，不依赖任何 PTX-EMU 内部 API。

### D6: 移除 `extern g_ptx_emu_driver_shim`

**决定**：恢复 `cudart_sim.cpp:137` 的 `static PtxEmuDriverShim* g_ptx_emu_driver_shim`，移除 `PtxEmuDriverShim.h` 的 `extern` 声明。

**理由**：测试不再直接访问 `g_ptx_emu_driver_shim`（auto-advance 替代了显式 `advance()` 调用）。恢复原封装状态。

### D7: Multi-Phase 推进

| Phase | 内容 | 依赖 | 预计 |
|-------|------|------|------|
| 0 | 2-cycle bug 根因定位（Oracle 协助） | 无 | 2-4h |
| 1 | D3 + D1 + D2 + D5 + D6 实施（测试自动化 + StubBridge + auto-advance） | Phase 0（日志输出确定后可并行） | 2-3h |
| 2 | D4 实施（修复 2-cycle bug） | Phase 0（结论输出后启动） | 2-4h |
| 3 | 回归验证 + 文档同步 | Phase 1+2 | 1-2h |

**关键依赖链**：Phase 0 输出确定 D4 修复方案后，Phase 2 才能启动。Phase 1 与 Phase 0 可部分重叠（Phase 0 的日志代码修改与 Phase 1 的 `cudart_sim.cpp` 修改触及不同文件，无冲突），但 Phase 1 的完成验证（`tasks.md:1.24-1.25` 回归测试）需等 Phase 0 日志就绪。建议 git workflow：Phase 0/1 并行开发（不同文件），Phase 2 在 Phase 0 结论输出后 rebase 到 Phase 1 之上。

## Non-Goals

- 不修改 `cpptlm_bridge.h` ABI（5 虚方法 + 版本号不变）
- 不集成 CppTLM 真实 TLM（仅提供零延迟 StubBridge）
- 不修改同步路径（`launchPtxInterpreter` + `wait_for_completion`）
- 不暴露 `g_ptx_emu_driver_shim` 为 public API（恢复 `static`）
- 不修改 `prepareKernelLaunchRequest` IR 构造
- 不修复 Event/Stream API / Multi-kernel / Multi-stream（独立议题）
- 不连线 `g_pending_kernels[kid].completed` 字段（当前死代码——`poll_kernel == 0` 直接 erase，`completed` 检查 `pk.completed` 从未被写入，跳过则 fallback 到 `poll_kernel` 调用。修复需独立 change `fix-pending-kernel-completed-field`）
- 不处理非零 stream 的 auto-advance（`cudaStreamSynchronize(non_default)` 不触发 advance；用户可使用 `cudaDeviceSynchronize` 替代）