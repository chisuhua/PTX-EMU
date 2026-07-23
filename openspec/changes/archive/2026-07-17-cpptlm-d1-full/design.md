# Design: CppTLM F12b-LD MemoryBridge 集成（D-PTX-1~6 决策 + HSK-1/2/3）

> **Status**: Proposed
> **Parent**: `proposal.md` (cpptlm-d1-full)
> **ADR**: [docs/adr/ADR-0021-cpptlm-d1-full-integration.md](../../../docs/adr/ADR-0021-cpptlm-d1-full-integration.md)（D-PTX-1~6 决策记录）
> **Triggered by**: CppTLM `2026-07-14-ptxemu-comprehensive-modification-plan.md §2` + `PTX-EMU-README §10`
> **Companion**: `cpptlm-phase8b-injection-points`（姊妹 change，§3 D1-Full 注入点）

---

## 1. 现状问题

### 1.1 PTX-EMU 当前 API（基于 2026-07-15 实证审查）

#### A. `cudaLaunchKernel` 同步阻塞（`src/cudart/cudart_sim.cpp:332-386`）

```cpp
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
    // ...args 检查 + sharedMem limit check...

    try {
        g_ptx_interpreter->launchPtxInterpreter(...);
        g_gpu_context->wait_for_completion();  // ← 同步阻塞！
    } catch (const PtxEmuException& e) { ... }
    return cudaSuccess;
}
```

- ❌ 无异步路径：每次 `cudaLaunchKernel` 必须等 kernel 完成才返回
- ❌ CppTLM 接入点不存在（`g_cpptlm_bridge` 全局指针缺失）

#### B. 全局单例（`include/cudart/cudart_sim.h:13-14` + `src/cudart/cudart_sim.cpp:55`）

```cpp
extern std::unique_ptr<GPUContext> g_gpu_context;
extern std::unique_ptr<PtxInterpreter> g_ptx_interpreter;
// cudart_sim.cpp:107-125 初始化入口（CudaDriver::instance + HardwareMemoryManager::instance 类似）
```

- ❌ F12b-LD 文档 §10.1 明确指 PTX-EMU 单例在多实例仿真中导致**静默状态损坏**
- ❌ 当前**零**重复初始化检测

#### C. GLOBAL 访存（`src/ptxsim/instructions/memory.cpp` 的 `LdHandler::processOperation()`）

- ❌ 当前直接走 PTX-EMU 内部 `SimpleMemory::read/write()`
- ❌ 无 `g_cpptlm_bridge->global_access()` 桥接路径
- ❌ 无 timing-only NoC 路由延迟注入

#### D. ANTLR4 版本一致性冲突（**D-PTX-4 关键发现**）

| 来源 | 声称版本 |
|------|---------|
| `AGENTS.md` §已知限制 | "ANTLR 版本：4.13.2（antlr-4.13.2-complete.jar）" |
| 根 `README.md` | "ANTLR 版本：4.13.2 完全 vendored" |
| `.github/copilot-instructions.md` | "ANTLR 运行时来自 antlr4/antlr4-cpp-runtime-**4.13.1**-source" |
| 实际 vendored 目录 | `antlr4/antlr4-cpp-runtime-**4.13.2**-source` |

- **冲突**：`AGENTS.md` + 实际目录 = 4.13.2；`copilot-instructions.md` = 4.13.1（**过时/错误**）
- **决策必要性**（D-PTX-4）：必须在 change 启动前澄清 + 修复 `copilot-instructions.md`

### 1.2 问题归纳

| # | 问题 | 影响 |
|---|------|------|
| **P1** | `cudaLaunchKernel` 同步阻塞 | CppTLM 无法成为时钟真相源（clock-of-truth） |
| **P2** | `cudaStreamSynchronize` 立即返回 | 多 stream 同步无法工作（任务书 §2.1 Task #3） |
| **P3** | GLOBAL LD/ST 不经 NoC 路由 | 失去 MemoryBridge timing 注入价值（任务书 §2.1 Task #4） |
| **P4** | 4 个全局单例无 SingletonGuard | 多实例仿真静默状态损坏（F12b-LD 文档 §10.1） |
| **P5** | ANTLR4 版本文档/实际不一致 | CppTLM CI 风险传递（HSK-2 阻塞） |
| **P6** | `cpptlm_bridge.h` 接口未定义 | CppTLM 端 #C1 MemoryBridge 实现编译失败（任务书 §2.1） |
| **P7** | 无错误码映射（PTX-EMU 内部 ↔ `cudaError_t`） | 异步路径错误传递不明（D-PTX-5） |

---

## 2. 目标状态

### 2.1 MemoryBridge F12b-LD 架构

```
CUDA kernel ──→ libcudart.so (PTX-EMU) ──┬─→ Bridge 路径（异步）
                                          │    ├─ cudaLaunchKernel → bridge->submit_kernel
                                          │    ├─ cudaStreamSynchronize → bridge->poll_kernel
                                          │    └─ LD/ST → bridge->global_access
                                          └─→ 原路径（fallback，bridge == nullptr）
```

- 行为切换：`g_cpptlm_bridge == nullptr` → 字节级回退到原行为（向后兼容）
- 时钟同步：CppTLM EventQueue 主动 tick → KernelLaunchTLM::tick() → PTX-EMU::exe_once()（PTX-EMU 失去时钟主控）

### 2.2 D-PTX-1~6 决策表（详见 ADR-0021）

| 决策 | 结论 | 文档位置 |
|------|------|---------|
| **D-PTX-1** `g_cpptlm_bridge` 位置与初始化 | 静态全局指针 + first-cuda-call 懒初始化 | ADR-0021 §3 |
| **D-PTX-2** 全局单例共存策略 | `SingletonGuard` 在 `__cudaRegisterFatBinary` 入口检测，重复时 FATAL | ADR-0021 §4 |
| **D-PTX-3** `exe_once()` 注入代码定位 | sm_context.cpp:222 (A) + 253/338 (B+C) 三段式注入（依赖姊妹 change `cpptlm-phase8b-injection-points` design.md §7.1）| ADR-0021 §5 |
| **D-PTX-4** ANTLR4 版本策略 | Pin 4.13.2（与实际一致），修复 `copilot-instructions.md` 错误 | ADR-0021 §6 |
| **D-PTX-5** 错误码映射表 | 5 类条件 + 返回值（任务书 §5.1 一致性表） | ADR-0021 §7 |
| **D-PTX-6** 性能预算策略 | vtable 优化 + 编译期内联（依赖 PTX-EMU 编译流程） | ADR-0021 §8 |

### 2.3 Handshake 信号产出（HSK-1/2/3）

| # | 内容 | 产出形式 | 时间 |
|---|------|---------|------|
| **HSK-1** | `cpptlm_bridge.h` 初始 commit hash | git commit hash（`CPPTLMBRIDGE_VERSION=1`） | D1 开工前 |
| **HSK-2** | ANTLR4 版本号 + CI yml 截图 | 版本号 = 4.13.2 + `.github/workflows/*.yml` 不安装 ANTLR4 | D1 开工前 |
| **HSK-3** | `libcpptlm_cudart.so` CMake 暴露方式 | CMake 草案（`ExternalProject_Add` 默认） | D5 EOD 前 |

---

## 3. CppTLMBridge 接口设计（PTX-EMU ABI 真值源）

### 3.1 文件：`include/cudart/cpptlm_bridge.h`

来源：[综合任务书 §2.1 Task #1]（完整代码 ~70 行）

**关键属性**：
- 5 个纯虚方法：`version()` + `submit_kernel()` + `poll_kernel()` + `synchronize_stream()` + `global_access()`
- `CPPTLMBRIDGE_VERSION` 宏：编译期 ABI 断言（每次接口变更同步 bump）
- `g_cpptlm_bridge` 全局指针：`nullptr` = 独立模式（字节级兼容）
- `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))`：防止未来 cudaStream_t 宽度变化导致隐式截断

**约束验证**：`grep '#include' include/cudart/cpptlm_bridge.h` → 仅 `<cstddef>` + `<cstdint>` + `<cuda_runtime.h>`

### 3.2 ABI 修订流程（D-PTX-1 关联）

- PTX-EMU commit 修改 `cpptlm_bridge.h` → bump `CPPTLMBRIDGE_VERSION` (e.g. 1 → 2)
- CppTLM 端消费方收到通知 → 同步 rebase → 通过 ExternalProject_Add 引用新版本
- 编译期 `static_assert` 在 CppTLM 端验证：`static_assert(CppTLMBRIDGE_VERSION >= 1)`

---

## 4. cudaLaunchKernel 异步化（D-PTX-1/2）

### 4.1 修改 `src/cudart/cudart_sim.cpp`

**位置**：`cudaLaunchKernel` 函数（行 332-386）+ 新增 helper 函数 + 数据结构

```cpp
// cudart_sim.cpp 顶部（修改点 1）
static std::atomic<uint64_t> next_kernel_id{1};
uint64_t generate_kernel_id() { return next_kernel_id.fetch_add(1); }

struct PendingKernel {
    uint64_t kernel_id;
    uint64_t stream_id;       // ★ NEW — 0 = 默认 stream
    const void* func;
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
};
std::unordered_map<uint64_t, PendingKernel> g_pending_kernels;  // F12b-LD 单线程

std::unordered_set<uint64_t> g_active_streams{0};  // 默认 stream 始终存在

void register_pending_kernel(uint64_t id, uint64_t stream_id,
                             const void* func, void** args,
                             dim3 grid, dim3 block, size_t shared_mem);
```

**位置**：`cudaLaunchKernel` 函数体（行 332-386）

```cpp
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
    // ... 原有 args 检查、共享内存限制检查 ...

    if (g_cpptlm_bridge) {
        // ★ NEW: 异步路径
        uint64_t kernel_id = generate_kernel_id();
        uint64_t stream_id = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(stream));
        const char* kernel_name = func2name[(uint64_t)func].c_str();
        const void** args_ptr = reinterpret_cast<const void**>(args);
        size_t args_count = count_kernel_args(args);  // PTX-EMU 端 helper

        int ret = g_cpptlm_bridge->submit_kernel(
            kernel_id, kernel_name,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            args_ptr, args_count, sharedMem, stream_id);
        if (ret != 0) return cudaError_t(ret);  // ★ 错误码直接转发（D-PTX-5）

        register_pending_kernel(kernel_id, stream_id, func, args,
                                gridDim, blockDim, sharedMem);
        return cudaSuccess;  // 立即返回！
    }

    // ★ 原有路径 — fallback 当 g_cpptlm_bridge 为 nullptr
    try {
        g_ptx_interpreter->launchPtxInterpreter(...);
        g_gpu_context->wait_for_completion();
    } catch (const PtxEmuException& e) { ... }
    return cudaSuccess;
}
```

---

## 5. Stream 同步原语（D-PTX-2 关联）

### 5.1 修改 `cudaStreamSynchronize` + `cudaDeviceSynchronize`

```cpp
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (!g_cpptlm_bridge) {
        return cudaSuccess;  // ★ 原有 fallback
    }

    uint64_t target_stream = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(stream));

    while (true) {
        bool stream_empty = true;
        std::vector<uint64_t> completed_ids;  // ★ 先收集

        for (const auto& [id, info] : g_pending_kernels) {
            if (info.stream_id != target_stream) continue;
            uint64_t remaining = g_cpptlm_bridge->poll_kernel(id);
            if (remaining == 0) {
                completed_ids.push_back(id);  // ★ 不直接 erase
            } else if (remaining != UINT64_MAX) {
                stream_empty = false;
            }
        }

        for (uint64_t id : completed_ids) {
            g_pending_kernels.erase(id);  // ★ 统一删除
        }

        if (stream_empty) break;
        // CppTLM 主动推进（在 host 端事件循环外部）
    }
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    if (!g_cpptlm_bridge) {
        g_gpu_context->wait_for_completion();
        return cudaSuccess;
    }
    for (uint64_t stream_id : g_active_streams) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_id));
    }
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    uint64_t id = next_kernel_id.fetch_add(1);
    g_active_streams.insert(id);
    *pStream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(id));
    return cudaSuccess;
}
```

**关键约束**：
- **迭代器失效修复**：先 `completed_ids.push_back(id)` → 循环外统一 erase（替代直接 range-for 中 erase）
- **stream_id 过滤**：只轮询匹配 stream 的 kernels
- **F12b-LD 单线程假设**：`g_pending_kernels`/`g_active_streams` 无锁（Phase 9+ 加 mutex）
- **无 `bridge->tick()`**：cppTLM EventQueue 主动时钟推进方，PTX-EMU 不反向驱动

---

## 6. GLOBAL LD/ST 桥接（D-PTX-3 关联）

### 6.1 修改 `src/ptxsim/instructions/memory.cpp`

**位置**：`LdHandler::processOperation()` + `StHandler::processOperation()`

```cpp
uint64_t LdHandler::processOperation(StatementContext& stmt,
                                     ThreadContext* thread) {
    uint64_t device_addr = compute_effective_address(stmt, thread);

    // ★ NEW: 如果有 bridge 且为 GLOBAL 空间，走 CppTLM NoC
    if (g_cpptlm_bridge && is_global_space(device_addr)) {  // is_global_space() 遍历全部 qualifier（§经验 #5）
        uint64_t latency = g_cpptlm_bridge->global_access(
            device_addr, 0, /*LD=*/0);
        if (latency != UINT64_MAX) {
            uint64_t value = 0;
            SimpleMemory::read(device_addr, &value);  // Phase 8.B bypass cache
            thread->write_register(stmt.dest_registers[0], value);
            return latency;  // NoC 路由延迟（用于设置 blocked_cycles）
        }
        // UINT64_MAX = 地址未映射 → fallback
    }

    return LdHandler::processOperation_internal(stmt, thread);
}
```

**关键约束**：
- **timing-only 语义**：返回 latency 仅用于设置 `blocked_cycles_remaining`；数据读写立即完成
- **Phase 8.B cache bypass**：直接读写 SimpleMemory，不经过 CppTLM CacheTLM
- **地址映射**：由 PTX-EMU 端 `is_global_space()` 判定（CUDA 虚拟地址 → GLOBAL/LOCAL/SHARED），`global_access()` 传入 CUDA device address
- **是 qualifier back() 教训**：`is_global_space()` 实现必须遍历整个 qualifier 列表而非仅末尾（参考 lessons-learned #5）

---

## 7. CMake 集成 + SingletonGuard（D-PTX-2 / D-PTX-6）

### 7.1 CMakeLists.txt 修改

```cmake
# CMakeLists.txt 末尾追加（实际实现，per commit d0803a09）

option(BUILD_LIB_CPPTLM_CUDART "Build libcpptlm_cudart.so bridge (requires CppTLM repo)" OFF)

if(BUILD_LIB_CPPTLM_CUDART)
    include(ExternalProject)

    # CppTLM commit hash (HSK-3: 待 CppTLM 团队确认后替换)
    set(CPPTLM_COMMIT_HASH "main" CACHE STRING "CppTLM git tag/commit to pin")

    ExternalProject_Add(cpptlm
        GIT_REPOSITORY  https://github.com/chisuhua/CppTLM.git
        GIT_TAG         ${CPPTLM_COMMIT_HASH}
        CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/cpptlm-install
                        -DBUILD_TESTING=OFF
                        -DCPPTLM_BUILD_CUDART_BRIDGE=ON
        UPDATE_DISCONNECTED TRUE  # 不自动 fetch，避免 ABI 漂移
        BUILD_ALWAYS FALSE
    )

    # 链接 CppTLM 到 cudart 目标（不是 ptxemu_runtime）
    add_dependencies(cudart cpptlm)
    target_link_libraries(cudart PRIVATE ${CMAKE_BINARY_DIR}/cpptlm-install/lib/libcpptlm_cudart.so)
    target_include_directories(cudart PRIVATE ${CMAKE_BINARY_DIR}/cpptlm-install/include)
endif()
```

**B5 (Metis second-pass review)**：本节原先描述 `find_package(cpptlm) + add_subdirectory(src/cudart/cpptlm_bridge) + target_link_libraries(ptxemu_runtime PRIVATE cpptlm::core)`，但实际 `CMakeLists.txt`（per commit `d0803a09`）使用 `ExternalProject_Add` 从 CppTLM 仓库拉取 + 构建到 `${CMAKE_BINARY_DIR}/cpptlm-install/`，然后直接链接 `libcpptlm_cudart.so` 路径到 `cudart` 目标。本节已更新以匹配实现。

**HSK-3 草案**：默认 `ExternalProject_Add`（任务书 §2.1 Task #5 建议），可配置切换 `find_library` / `pkg-config`（未来扩展，当前未实现）。

### 7.2 SingletonGuard（D-PTX-2）

```cpp
// cudart_sim.cpp:107 附近（现有 __cudaRegisterFatBinary 入口）

class SingletonGuard {
public:
    SingletonGuard() {
        if (initialized_) {
            std::cerr << "FATAL: PTX-EMU global singleton already initialized";
            std::abort();  // ★ 重复时立即崩溃
        }
        initialized_ = true;
    }
    static bool initialized_;
};
bool SingletonGuard::initialized_ = false;

// 在 4 个全局单例的初始化入口前加 SingletonGuard guard;
```

**约束**：F12b-LD 阶段假设 host 端单线程调用 CUDA API；Phase 9+ 加 mutex 保护

---

## 8. ANTLR4 版本一致性修复（D-PTX-4）

### 8.1 修复路径

| 文档 | 当前 | 修正 |
|------|------|------|
| `.github/copilot-instructions.md` | "ANTLR 运行时来自 antlr4/antlr4-cpp-runtime-**4.13.1**-source" | 改为 "**4.13.2**-source" |
| AGENTS.md | "ANTLR 版本：4.13.2" | (保留，与实际一致) |
| README.md | "ANTLR 版本：4.13.2 完全 vendored" | (保留) |

### 8.2 版本策略

- **Pin 4.13.2**（与实际 vendored 一致）
- **CI yml**（HSK-2 证据）：验证 `.github/workflows/*.yml` 不安装 ANTLR4（vendored 目录已包含）
- **升级路径**：半年 review 一次；升级时需 d-PTX 同步给 CppTLM

---

## 9. 影响范围（组件 | 影响类型）

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `docs/adr/ADR-0021-cpptlm-d1-full-integration.md` | **新增** | D-PTX-1~6 决策记录（~200 行） |
| `include/cudart/cpptlm_bridge.h` | **新增** | CppTLMBridge 抽象接口（~70 LOC） |
| `include/cudart/cpptlm_bridge_impl.h` | **新增**（optional） | Bridge 默认 stub 实现（~20 LOC） |
| `include/cudart/AGENTS.md` | **修改** | Bridge ABI 真值源地位记录 |
| `src/cudart/cudart_sim.cpp` | **修改** | cudaLaunchKernel 异步 + Stream sync + SingletonGuard + PendingKernel 数据结构（~150 LOC 增量） |
| `src/ptxsim/instructions/memory.cpp` | **修改** | LdHandler/StHandler bridge 分支（~30 LOC 增量） |
| `CMakeLists.txt` | **修改** | libcpptlm_cudart.so 集成（~10 LOC 增量） |
| `.github/copilot-instructions.md` | **修改** | ANTLR4 版本 4.13.1 → 4.13.2 修正 |
| `tests/unit/cpptlm/test_cpptlm_bridge.cpp` | **新增** | 7 个 Bridge stub 测试（~200 LOC） |
| `tests/integration/cpptlm/test_async_launchkernel.cpp` | **新增** | 异步路径测试（~150 LOC） |
| `tests/integration/cpptlm/test_ld_st_bridge.cpp` | **新增** | GLOBAL 桥接测试（~150 LOC） |
| `tests/integration/cpptlm/test_singleton_guard.cpp` | **新增** | 重复初始化 FATAL 测试（~80 LOC） |
| `tests/CMakeLists.txt` | **修改** | 4 个 `add_catch_test` 注册 |
| `AGENTS.md` | **修改** | §已知限制增加 §F12b-LD 状态 |
| `docs/adr/README.md` | **修改** | 索引追加 ADR-0021 |
| `docs/dev-process/lessons-learned.md` | **修改** | 新增 §F12b-LD 经验条目 |
| **合计** | | **~1100 LOC 增量（16 个文件）** |

---

## 10. 风险与缓解

| # | 风险 | 概率 | 影响 | 缓解措施 |
|---|------|:---:|:---:|---------|
| **R1** | `cudaLaunchKernel` 异步路径破坏现有同步测试 | 中 | 高 | 基线 worktree + Phase 1 全量回归 + `bridge == nullptr` 字节级回退 |
| **R2** | 迭代器失效在 `cudaStreamSynchronize` 中再现 | 低 | 中 | 修复模式已应用：先收集 completed_ids 再统一 erase（避免 range-for 中 `unordered_map::erase` 触发 UB）|
| **R3** | `is_global_space()` 实现走 back() 路径导致误判 | 中 | 高 | Lessons Learned #5 强制：必须遍历整个 qualifier 列表 |
| **R4** | SingletonGuard 与现有初始化路径冲突 | 低 | 中 | Phase 0 走 `__cudaRegisterFatBinary` 入口，与 `g_gpu_context` 初始化并列放置 |
| **R5** | ANTLR4 版本双声明导致 CppTLM CI 困惑 | 中 | 中 | D-PTX-4 强制：1 个权威源（实际 vendored 4.13.2），其他文档同步修正 |
| **R6** | `libcpptlm_cudart.so` CMake 暴露方式三选一导致 lock-in | 低 | 低 | HSK-3 草案提供 `ExternalProject_Add` 默认值 + `option(BUILD_LIB_CPPTLM_CUDART)` 可切换 |
| **R7** | 错误码映射不一致（PTX-EMU 内部 ↔ cudaError_t）| 中 | 中 | D-PTX-5 表格强制 5 类条件 + 返回值；任务书 §5.1 一致性表 |
| **R8** | 性能：vtable 调用开销导致 submit/poll 退化 | 低 | 低 | D-PTX-6 vtable 优化 + 编译期内联（依赖 PTX-EMU 编译流程） |
| **R9** | 实施过程中与姊妹 change `cpptlm-phase8b-injection-points` 冲突 | 低 | 高 | 主代码完全并行（本 change touch cudart_sim.cpp + memory.cpp；姊妹 change touch sm_context.cpp + warp_context.cpp）；唯一共享点为 `tests/CMakeLists.txt` 的追加式 `add_catch_test` 行（追加而非覆盖，不阻塞）|

---

## 11. 与 PTX-EMU 现有架构的协调

### 11.1 与 ADR-0010（Fake CUDA Runtime）协调

- `cudaLaunchKernel` 当前实现（行 332-386）→ 改为 bridge 优先 + fallback
- `cudaStreamSynchronize` 当前实现（`return cudaSuccess;`）→ bridge 路径真实轮询

### 11.2 与 ADR-0019（ThreadContext 瘦身）协调

- `block_cycles_remaining` 当前是 `ThreadState` 字段 — 本 change 不修改
- 与 `god-class-refactor-thread-context-phase3` 并行实施时需关注：
  - 若字段迁移，本 change 的 `is_global_space()` 需重新定位
  - 通过 CPUT 层 `MemoryAccessor` 调用

### 11.3 与 ADR-0020（cpptlm-injection-points）协调

- 本 change **不**修改 `exe_once()`（姊妹 change 处理）
- 但需注意：`exe_once()` Step B 调用 `InstructionLatencyTable::instance().get(stmt.type).cycles` → 当 `pipeline_provider_` 提供 LD/ST 延迟时，本 change 的 `g_cpptlm_bridge->global_access()` 应保持**单一来源**（避免双重 timing 注入）
- 解决：`global_access()` timing 设置的 `blocked_cycles` 与 `pipeline_provider_` 设置的 `blocked_cycles` 取最大值（max-of-two 语义）

---

## 12. 验证策略

### 12.1 Phase 0 基线

```bash
cd /workspace/project/PTX-EMU
git worktree add ../ptxemu-baseline-f12b main
cd ../ptxemu-baseline-f12b
. env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest -L "unit;integration;e2e" --output-on-failure
```

### 12.2 Phase 1-3 单元测试

- Phase 1（cpptlm_bridge.h）：`unit_cpptlm_bridge` 7 个 stub 测试
- Phase 2（cudaLaunchKernel）：`integration_async_launchkernel` 真实 kernel 异步路径
- Phase 3（cudaStreamSynchronize）：`integration_cudart_sync` stream 过滤 + iterator 修复

### 12.3 Phase 4-5 集成测试

- Phase 4（memory.cpp）：`integration_ld_st_bridge` GLOBAL 走 bridge + 数据正确性对比 baseline
- Phase 5（SingletonGuard）：`integration_singleton_guard` 重复初始化 FATAL

### 12.4 Phase 6 全量回归

```bash
cd /workspace/project/PTX-EMU
./scripts/sanity.sh  # 含 PTX-EMU 600+ 测试 + 新增 4 个测试目标
```

**合格标准**：100% PASS + `bridge == nullptr` 字节级回退到 baseline 行为 + Handler 单元测试 0 回归

---

## 13. 关联 spec 章节

- `CppTLM/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md` §2 (PTX-EMU 端 #1~#5)
- `CppTLM/docs/superpowers/specs/PTX-EMU-README.md` §10 (6 项决策 + 3 个 handshake)
- `CppTLM/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md` §4 + §5 (Bridge 接口 + 错误语义)
- `CppTLM/docs/adr/ADR-NV-02-phase8b-d1-strategy.md` Status Update 2026-07-14
- `docs/dev-process/lessons-learned.md` §1 跨模块状态翻译 + §4 分 Phase commit + §5 基线 worktree + §16 qualifier back()（注：迭代器失效模式为 PTX-EMU 内部约定，未独立编号）
- `openspec/changes/cpptlm-phase8b-injection-points/design.md` §7.1 `exe_once()` 注入参考
