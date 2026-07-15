## ADDED Requirements

### Requirement: cpptlm-bridge-interface

The system MUST provide a `CppTLMBridge` abstract class in `include/cudart/cpptlm_bridge.h` with exactly five public pure virtual methods: `version() const`, `submit_kernel(uint64_t, const char*, uint32_t×6, const void**, size_t, size_t, uint64_t)`, `poll_kernel(uint64_t)`, `synchronize_stream(uint64_t)`, and `global_access(uint64_t, uint64_t, uint8_t)`. The header MUST define a `CPPTLMBRIDGE_VERSION` macro with integer value 1. A global raw pointer `g_cpptlm_bridge` of type `CppTLMBridge*` MUST be declared with `extern` and have default value `nullptr`. The header MUST include a `static_assert` verifying `sizeof(cudaStream_t) <= sizeof(uint64_t)`. The header MUST include only `<cstdint>` and `<cuda_runtime.h>`.

#### Scenario: ABI 真值源 — header is the canonical source
- **WHEN** CppTLM `ExternalProject_Add` 引用 PTX-EMU 仓库的 `include/cudart/cpptlm_bridge.h`
- **THEN** 字节级相同（双方编译 `static_assert(CPPTLMBRIDGE_VERSION == 1)` 通过）
- **AND** `version() == 1` 返回值与 CppTLM 端 MemoryBridge 实现一致

#### Scenario: nullptr 全局指针 = 字节级向后兼容
- **WHEN** PTX-EMU 默认初始化，`g_cpptlm_bridge` 未被赋值
- **THEN** 值为 `nullptr`
- **AND** `cudaLaunchKernel` / `cudaStreamSynchronize` / `LdHandler` 走 fallback 路径
- **AND** 现有 600+ PTX-EMU 测试零回归（与 baseline 字节级一致）

#### Scenario: cudaStream_t 宽度 static_assert 拦截未来溢出
- **WHEN** 未来 `cuda_runtime.h` 更新使 `sizeof(cudaStream_t) > sizeof(uint64_t)`
- **THEN** 编译期 `static_assert` 失败，CI 中止
- **AND** 阻塞后续 release，防止截断 bug 上线

#### Scenario: 接口签名演进触发版本号 bump
- **WHEN** `cpptlm_bridge.h` 中任一虚方法签名变更
- **THEN** `CPPTLMBRIDGE_VERSION` 必须 bump（如 1 → 2）
- **AND** 通知 CppTLM 同步 rebase
- **AND** CppTLM 端 `MemoryBridge::version()` 返回同步的新版本号

---

### Requirement: cudart-async-launchkernel

The `cudaLaunchKernel` function in `src/cudart/cudart_sim.cpp` MUST branch on `g_cpptlm_bridge == nullptr`. When non-null, the function MUST call `bridge->submit_kernel()` with all 12 parameters (kernel_id, kernel_name, grid×3, block×3, args, args_count, shared_mem, stream_id) generated from the input arguments, register a `PendingKernel` entry into a global `unordered_map` keyed by `kernel_id` with a `stream_id` field, and return `cudaSuccess` immediately without blocking. The `kernel_id` MUST be generated via a thread-safe `std::atomic<uint64_t>` counter starting at 1. The `stream_id` MUST be `static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream))`. When `g_cpptlm_bridge == nullptr`, the function MUST execute the original synchronous path unchanged (byte-identical with baseline).

#### Scenario: 异步路径立即返回
- **WHEN** `g_cpptlm_bridge != nullptr` 且调用 `cudaLaunchKernel(...)`
- **THEN** `bridge->submit_kernel()` 被调用一次，参数与 CUDA runtime API 对应
- **AND** 完成后立即 `return cudaSuccess`，不阻塞
- **AND** `PendingKernel{ kernel_id, stream_id, func, grid, block, shared_mem }` 注册到 `g_pending_kernels`

#### Scenario: nullptr 路径字节级回退
- **WHEN** `g_cpptlm_bridge == nullptr` 且调用 `cudaLaunchKernel(...)`
- **THEN** 调用 `g_ptx_interpreter->launchPtxInterpreter(...)`
- **AND** 调用 `g_gpu_context->wait_for_completion()`（同步）
- **AND** 输出与改造前**字节级相同**

#### Scenario: kernel_id 唯一性保证
- **WHEN** 连续调用 `cudaLaunchKernel(...)` N 次（N ≥ 1000）
- **THEN** `g_pending_kernels` 中所有 `kernel_id` 唯一
- **AND** 通过 `std::atomic<uint64_t>::fetch_add(1)` 顺序生成
- **AND** 不存在重复 id

#### Scenario: bridge submit 失败 → 错误码传递
- **WHEN** `bridge->submit_kernel()` 返回非 0（错误码）
- **THEN** `cudaLaunchKernel` 返回 `cudaError_t(ret)` 直接转发
- **AND** **NOT** 注册到 `g_pending_kernels`

#### Scenario: kernel_name 来自 func2name 表（PTX-EMU 内部）
- **WHEN** 异步路径查找 `kernel_name`
- **THEN** 取 `func2name[(uint64_t)func].c_str()`
- **AND** 该字符串在 PTX-EMU 内部长期存储，无需 deep-copy

---

### Requirement: cudart-stream-synchronization

The `cudaStreamSynchronize`, `cudaDeviceSynchronize`, and `cudaStreamCreate` functions in `src/cudart/cudart_sim.cpp` MUST implement real synchronization semantics when `g_cpptlm_bridge != nullptr`. `cudaStreamSynchronize` MUST poll `bridge->poll_kernel(kernel_id)` for all `PendingKernel` entries whose `stream_id` matches the target stream, mark completed entries (return value 0) into a local `vector<uint64_t>` first, then perform all `g_pending_kernels.erase(id)` calls in a separate loop after iteration completes (iterator-invalidation fix). `cudaDeviceSynchronize` MUST iterate over `g_active_streams` set and call `cudaStreamSynchronize` for each. `cudaStreamCreate` MUST allocate a 64-bit unique ID, insert into `g_active_streams`, and return it cast to `cudaStream_t`. When `g_cpptlm_bridge == nullptr`, all three MUST preserve original semantics (byte-identical with baseline).

#### Scenario: 按 stream_id 过滤，避免跨 stream 干扰
- **WHEN** `g_pending_kernels` 含 `[id=1, stream=0; id=2, stream=7; id=3, stream=0]`
- **AND** 调用 `cudaStreamSynchronize(stream_default)`
- **THEN** `poll_kernel(id=1)` 和 `poll_kernel(id=3)` 被调用
- **AND** `poll_kernel(id=2)` **NOT** 被调用（不同 stream）

#### Scenario: 迭代器失效修复 — 完成 id 先收集再统一 erase
- **WHEN** 在 `cudaStreamSynchronize` 内部 range-for 迭代 `g_pending_kernels`
- **AND** 检测到 `bridge->poll_kernel(id)` 返回 0
- **THEN** `id` 推入 `std::vector<uint64_t> completed_ids`
- **AND** **NOT** 在 range-for 中立即调用 `g_pending_kernels.erase(id)`
- **AND** range-for 结束后在第二个循环统一 `g_pending_kernels.erase(id)`

#### Scenario: cudaDeviceSynchronize 遍历所有活跃 stream
- **WHEN** `g_active_streams = {0, 7, 12}` 且调用 `cudaDeviceSynchronize()`
- **THEN** `cudaStreamSynchronize(stream=0)` 被调用
- **AND** `cudaStreamSynchronize(stream=7)` 被调用
- **AND** `cudaStreamSynchronize(stream=12)` 被调用
- **AND** 所有 stream 上的 pending kernels 在返回前完成

#### Scenario: cudaStreamCreate 分配 64-bit 唯一 ID
- **WHEN** 调用 `cudaStreamCreate(&pStream)`
- **THEN** `*pStream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(id))`
- **AND** `id` 通过 `next_kernel_id.fetch_add(1)` 生成（64-bit atomic 计数器）
- **AND** `id` 插入 `g_active_streams`
- **AND** `id` 与现有 kernel_id / stream_id 无冲突（unique-by-construction）

#### Scenario: nullptr 路径原始行为保留
- **WHEN** `g_cpptlm_bridge == nullptr`
- **THEN** `cudaStreamSynchronize` 返回 `cudaSuccess` 立即（原有行为）
- **AND** `cudaDeviceSynchronize` 调用 `g_gpu_context->wait_for_completion()`
- **AND** `cudaStreamCreate` 行为不变

#### Scenario: pending kernel 完成后清理
- **WHEN** `bridge->poll_kernel(id)` 返回 0
- **THEN** `g_pending_kernels.erase(id)` 被调用
- **AND** `g_active_streams` 中的 stream_id **NOT** 被移除（stream 句柄生命周期独立）

---

### Requirement: ptx-global-ld-st-bridge

The `LdHandler::processOperation()` and `StHandler::processOperation()` functions in `src/ptxsim/instructions/memory.cpp` MUST branch on `(g_cpptlm_bridge != nullptr) && is_global_space(device_addr)`. When both true, the handlers MUST call `bridge->global_access(device_addr, val, type)` with `type=0` for LD and `type=1` for ST. If the return value is not `UINT64_MAX`, the handler MUST perform the data read/write against `SimpleMemory` (Phase 8.B cache-bypass semantics) and return the latency value. If the return value IS `UINT64_MAX` (address not mapped), handlers MUST fall back to the original PTX-EMU internal path. The `is_global_space()` function MUST iterate over the entire qualifier list (not just `qualifiers.back()`) to determine if the address is in the GLOBAL space.

#### Scenario: GLOBAL 地址走 bridge 时序路径
- **WHEN** `device_addr` 为 GLOBAL 空间地址
- **AND** `g_cpptlm_bridge != nullptr`
- **THEN** `bridge->global_access(device_addr, 0, /*LD=*/0)` 被调用
- **AND** 返回的 latency（非 UINT64_MAX）被返回给调用方
- **AND** 数据**仍**从 `SimpleMemory::read()` 读取（Phase 8.B 语义）
- **AND** 返回 latency 用于设置 `blocked_cycles_remaining`

#### Scenario: bridge 返回 UINT64_MAX → fallback
- **WHEN** `bridge->global_access()` 返回 `UINT64_MAX`（地址未映射）
- **THEN** fallback 到 `processOperation_internal(stmt, thread)`
- **AND** **NOT** 直接读写 SimpleMemory（避免破坏现有 PTX-EMU 内部路径）

#### Scenario: 非 GLOBAL 地址不触发 bridge
- **WHEN** `device_addr` 为 LOCAL/SHARED 空间
- **THEN** **NOT** 调用 `bridge->global_access()`
- **AND** fallback 到原有 PTX-EMU 内部路径

#### Scenario: nullptr bridge 字节级回退
- **WHEN** `g_cpptlm_bridge == nullptr`
- **THEN** handler 行为与改造前**字节级相同**
- **AND** 现有 `[unit;memory]` `[e2e;memory]` 测试 0 回归

#### Scenario: is_global_space() 必须遍历整个 qualifier 列表
- **WHEN** `device_addr` 来源的 `StatementContext.qualifiers` 是多元素列表（如 `{Q_U32, Q_GLOBAL}` 或 `{Q_F32, Q_GLOBAL}`）
- **AND** Q_GLOBAL **NOT** 在末尾
- **THEN** `is_global_space()` 仍正确识别为 GLOBAL 空间
- **AND** bridge 路径被触发
- **AND** **NOT** 误判为非 GLOBAL（避免 Lessons Learned #5 复发）

---

### Requirement: libcpptlm-cudart-integration

The `CMakeLists.txt` MUST add an `option(BUILD_LIB_CPPTLM_CUDART "Build libcpptlm_cudart.so bridge" OFF)` option. When `option(BUILD_LIB_CPPTLM_CUDART)=ON` AND `cpptlm_FOUND` is true, the build MUST add subdirectory `src/cudart/cpptlm_bridge` and link `ptxemu_runtime` against `cpptlm::core`. The default OFF state MUST guarantee zero-regression (existing PTX-EMU tests pass byte-identically with baseline).

#### Scenario: 默认 OFF — 现有构建字节级不变
- **WHEN** 用户运行 `cmake --build build` 默认配置
- **THEN** `BUILD_LIB_CPPTLM_CUDART=OFF`（默认）
- **AND** `cpptlm_bridge` 子目录**NOT**被包含
- **AND** `libcpptlm_cudart.so` **NOT** 被构建
- **AND** `g_cpptlm_bridge == nullptr`（默认）
- **AND** 现有 600+ PTX-EMU 测试 PASS，与 baseline 字节级一致

#### Scenario: ON + CppTLM found → 链接 libcpptlm_cudart.so
- **WHEN** `cmake -DBUILD_LIB_CPPTLM_CUDART=ON ..` + `find_package(cpptlm)` 成功
- **THEN** `src/cudart/cpptlm_bridge` 子目录被 add
- **AND** `libcpptlm_cudart.so` 被构建到 `build/lib/`
- **AND** `ptxemu_runtime` 链接到 `cpptlm::core`
- **AND** `CPPTLMBRIDGE_VERSION` 编译期断言在双方都通过

#### Scenario: HSK-3 CMake 暴露方式 — 草案生效
- **WHEN** 实施 §Phase 5 (D5 EOD 前)
- **THEN** HSK-3 草案发出给 CppTLM（含 3 选项对比）
- **AND** 草案默认选择 `ExternalProject_Add`，备选 `find_library` / `pkg-config`
- **AND** PTX-EMU README §9 命令参考包含 `./build/bin/ptxemu_tests` 验证入口

#### Scenario: SingletonGuard 拒绝重复初始化
- **WHEN** `__cudaRegisterFatBinary` 被第二次调用（重复初始化检测）
- **THEN** `SingletonGuard::initialized_ == true`
- **AND** `std::cerr << "FATAL: PTX-EMU global singleton already initialized"`
- **AND** `std::abort()` 立即中止
- **AND** **NOT** 进入静默状态损坏路径（防止 F12b-LD 文档 §10.1 R1 风险）

#### Scenario: ANTLR4 版本声明一致性（HSK-2 证据）
- **WHEN** 检查 PTX-EMU 仓库 ANTLR4 版本声明
- **THEN** `AGENTS.md` = "4.13.2"
- **AND** 根 `README.md` = "4.13.2"
- **AND** `antlr4/antlr4-cpp-runtime-4.13.2-source/` 实际存在
- **AND** `.github/copilot-instructions.md` 修正为 "4.13.2"（原 "4.13.1" 错误）
- **AND** `.github/workflows/*.yml` 不安装 ANTLR4（vendored 覆盖）
