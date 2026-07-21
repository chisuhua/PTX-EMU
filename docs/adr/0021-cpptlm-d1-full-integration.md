# ADR-0021: PTX-EMU 端 F12b-LD MemoryBridge 自决策（D-PTX-1~6）

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-07-15（Active: 2026-07-16） |
| **关联任务** | D-PTX-1, D-PTX-2, D-PTX-3, D-PTX-4, D-PTX-5, D-PTX-6（详见 PTX-EMU-README §10.1）|
| **关联 OpenSpec change** | [openspec/changes/cpptlm-d1-full/](../../openspec/changes/cpptlm-d1-full/) |
| **关联 CppTLM 文档** | [PTX-EMU-README.md §10](https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/PTX-EMU-README.md)（6 项决策 + 3 个 handshake）<br>[综合任务书 §2 + §10](https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md)<br>[协作同步 §4 + §5](https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md) |
| **姊妹 ADR** | [ADR-0020](./0020-cpptlm-injection-points.md)（§3 D1-Full 三段式注入，已存在 2026-07-14）|

---

## 上下文

[CppTLM 综合任务书 §2](https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md) 要求 PTX-EMU 实施 §2 F12b-LD MemoryBridge 的 5 项改造任务（#1-#5），使 CppTLM 成为唯一时钟真相源。

[PTX-EMU-README §10](https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/PTX-EMU-README.md) 明确列出 **6 项 PTX-EMU 端自主决策（D-PTX-1~6）** 与 **3 个回传给 CppTLM 的 handshake（HSK-1/2/3）**，**由 PTX-EMU 仓库自治**。CppTLM 团队**不替 PTX-EMU 决定**这些。

**触发事件**：
1. **2026-07-03** — CppTLM 团队对 PTX-EMU 8 个关键头文件进行审查，发现 `SMContext` 仅暴露 `WarpScheduler` 注入点
2. **2026-07-14** — CppTLM 发布综合任务书（`2026-07-14-ptxemu-comprehensive-modification-plan.md`，947 行），§2 列出 5 项 PTX-EMU 任务（#1-#5）
3. **2026-07-14** — CppTLM 协作同步文档 §4 + §5 详细说明 `CppTLMBridge` 接口 + LD/ST 拦截点
4. **2026-07-14** — CppTLM ADR-NV-02 Status Update：D1-Lite → D1-Full 升级
5. **2026-07-14** — 本仓库已签署 ADR-0020（接受 §3 注入点决策）
6. **2026-07-15** — 本 ADR 签署 §2 MemoryBridge 6 项自主决策（D-PTX-1~6）

**前置 ADR**：
- [ADR-0020](./0020-cpptlm-injection-points.md)：§3 D1-Full 三段式注入（姊妹 ADR；本 ADR 签署前需先达到 Accepted 状态）
- [ADR-0010](./0010-fake-cuda-runtime.md)：Fake CUDA Runtime 当前实现（将改造）
- [ADR-0019](./0019-pc-management-extraction.md)：ThreadContext 瘦身（不修改 ThreadState）

---

## 决策驱动因素

1. **CppTLM 协同约束**：综合任务书 §2 列出 5 项 PTX-EMU 端任务，§10 列出 6 项 PTX-EMU 自主决策
2. **F12b-LD 单实例约束**：协作同步 §10.1 明确 PTX-EMU 全局单例在多实例仿真中导致静默状态损坏
3. **HSK-1/2/3 协同要求**：CppTLM 团队等待 PTX-EMU 提供 ABI commit hash + ANTLR4 版本 + CMake 暴露方式
4. **零 CppTLM 依赖**：3 个接口头文件 + cpptlm_bridge.h 仅依赖 `<cstdint>`/`<cuda_runtime.h>`
5. **向后兼容**：`g_cpptlm_bridge == nullptr` 时所有改动字节级回退（避免破坏现有 600+ 测试）

---

## 决策内容

### 决策 D-PTX-1: `g_cpptlm_bridge` 全局指针位置与初始化时机

**结论**：✅ **静态全局指针 + first-cuda-call 懒初始化**

**详细方案**：

```cpp
// 位置：include/cudart/cpptlm_bridge.h
extern CppTLMBridge* g_cpptlm_bridge;

// 定义：src/cudart/cudart_sim.cpp 顶部
CppTLMBridge* g_cpptlm_bridge = nullptr;  // 默认 nullptr = 独立模式
```

**初始化流程**：
1. 默认 `nullptr`（保留字节级回退）
2. 加载 `libcpptlm_cudart.so` 后通过 `extern "C"` 入口函数 `cpptlm_attach_bridge()` 赋值
3. `cudaLaunchKernel` 入口检查 `if (g_cpptlm_bridge)` 走异步路径

**考虑的替代方案**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **A. 静态全局指针 + 懒初始化**（✅ 选中） | 单一权威 source + 零外部依赖 + 与现有全局单例风格一致 | 需要外部触发初始化 | **✅** |
| B. 构造函数注入 | 解耦清晰 | 需要修改所有调用点签名 | ❌ 改动过大 |
| C. static init | 启动时设置 | CppTLM 库尚未加载，时机错误 | ❌ 时机不可控 |
| D. first-cuda-call hook | 强制初始化 | 增加 entry-point overhead；与 SingletonGuard 路径重复 | ❌ |

**约束**：
- `extern` 声明必须在 `cpptlm_bridge.h` 内，定义必须在单个 TU（`cudart_sim.cpp`）
- 全局指针的 `nullptr` 默认值**必须在编译期**确认（避免 TLS / thread-local 误用）

---

### 决策 D-PTX-2: PTX-EMU 全局单例与 bridge 的共存策略

**结论**：✅ **`SingletonGuard` 运行时检测 + 重复时 FATAL 中止**

**详细方案**：

```cpp
// 位置：src/cudart/cudart_sim.cpp
class SingletonGuard {
public:
    SingletonGuard() {
        if (initialized_) {
            std::cerr << "FATAL: PTX-EMU global singleton already initialized";
            std::abort();  // ★ 立即崩溃
        }
        initialized_ = true;
    }
    static std::atomic<bool> initialized_;
};

// 4 个全局单例的初始化入口前各加：
// SingletonGuard guard;
```

**保护的 4 个全局单例**：
1. `g_gpu_context` (`unique_ptr<GPUContext>`, `cudart_sim.cpp:111` / `:123`)
2. `g_ptx_interpreter` (`unique_ptr<PtxInterpreter>`, `cudart_sim.cpp:114` / `:125`)
3. `CudaDriver::instance()` (Driver singleton)
4. `HardwareMemoryManager::instance()` (Memory singleton)

**考虑的替代方案**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **A. SingletonGuard + FATAL 中止**（✅ 选中） | 早崩溃、易调试、与 F12b-LD §10.1 一致 | 测试需要 mock | **✅** |
| B. 多实例 + per-bridge 状态隔离 | 支持多实例 | 重构所有单例（~2 周）；超出 F12b-LD scope | ❌ 推迟 F12c |
| C. 强制 reset-on-bridge-init | 切换 bridge 时清理 | 状态损坏已发生才修复，太晚 | ❌ |
| D. 静默忽略重复初始化 | 兼容现有测试 | 静默状态损坏（F12b-LD §10.1 R1）| ❌ |

**约束**：
- F12b-LD 阶段**单线程假设**（host 端），SingletonGuard 不加锁；Phase 9+ 加 mutex
- `g_cpptlm_bridge` 本身**不需要** SingletonGuard（CppTLM 端可多次 attach/detach）

---

### 决策 D-PTX-3: `exe_once()` 三段式注入代码定位

**结论**：✅ **A/B 插入点定位到 `sm_context.cpp:222` 后；B/C 插入点定位到 `sm_context.cpp:253`/`:338` 旁（instr 分支）**

> 注：本决策的代码改法完整描述见姊妹 change [`cpptlm-phase8b-injection-points/design.md §7.1`](../../openspec/changes/cpptlm-phase8b-injection-points/design.md)（姊妹 change 当前 Proposed，待 ADR-0020 Accepted 后冻结）。本 ADR 仅说明 D-PTX-3 的**行号定位**（行号 222/253/338 在 sm_context.cpp 经预检验证正确）。

**详细方案**：

| 注入点 | 位置（基于 `sm_context.cpp:191-401`）| 注入前/后条件 | nullptr 行为 |
|--------|------|------|------|
| **Step A** Scoreboard 检查 | `sm_context.cpp:222` 之后，`next_warp` 赋值后 | `next_warp != nullptr` | 完全跳过 |
| **Step B** 延迟查询 | `sm_context.cpp:253` (单 PC 分支) **或** `:338` (多 PC 分支) **之前** | 同 Step A | 走 `InstructionLatencyTable` fallback |
| **Step C** Scoreboard 释放 | `sm_context.cpp:253` **或** `:338` **之后**，`check_reconvergence()` 之前 | 同 Step A | 完全跳过 |

**考虑的替代方案**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **A. 行号定位到现有 instr 分支前/后**（✅ 选中） | 最小改动 + 字节级回退 + 风险局部化 | 需要处理 fast-path（单 PC）和 slow-path（多 PC）两个分支 | **✅** |
| B. 抽出 `execute_instruction_with_timing()` 公共函数 | 复用性高 | 改动范围大 + 风险扩散 | ❌ |

**约束**：
- 步骤 A/B/C 的 `nullptr` 完全回退必须经 baseline worktree 验证（Lessons Learned #4）
- 必须在 fast-path（lane_by_pc size == 1，分支 :253）和 slow-path（lanes_by_pc 多分支，分支 :338）**两个位置**插入（漏一处会破坏多 PC 场景）
- 验证 `ptxsim-barrier-mechanism` 与本决策的兼容性（barrier 路径不修改）

---

### 决策 D-PTX-4: ANTLR4 版本策略

**结论**：✅ **升级至 4.13.2**（满足 CppTLM 综合计划 Task #5 下限 `>= 4.13.2`）

**详细方案**：

| 文档 | 升级后 | 状态 |
|------|------|------|
| `AGENTS.md` §已知限制 | "ANTLR 版本：4.13.2（antlr-4.13.2-complete.jar）" | ✅ 已升级 |
| 根 `README.md` | "ANTLR 版本：4.13.2 完全 vendored" | ✅ 已升级 |
| `.github/copilot-instructions.md` | "ANTLR 运行时来自 antlr4/antlr4-cpp-runtime-4.13.2-source" | ✅ 已修复（原 4.13.1 笔误） |
| 实际 vendored 目录 | `antlr4/antlr4-cpp-runtime-4.13.2-source/` | ✅ 物理升级完成 |
| **CppTLM 综合计划 Task #5** | `>= 4.13.2` | ✅ 满足下限 |

**升级理由**: CppTLM 综合计划 Task #5 要求 `>= 4.13.2`；PTX-EMU 从 4.11.1 升级至 4.13.2 以完全满足契约下限。升级前已确认 ANTLR4 4.11→4.13 的 runtime ABI 对 PTX-EMU 的 `.g4` 语法文件解析无误。`copilot-instructions.md` 中残留的 "4.13.1" 声明为历史笔误，已随本次升级同步修正。

**升级路径**：
- **半年 review**（2026-12 / 2027-06 各一次）
- 升级触发：上游 ANTLR4 安全修复或 PTX 语法关键 bug 修复
- 升级流程：
  1. 新建 fork branch `antlr4-upgrade-4.X.Y`
  2. 更新 vendored 目录
  3. 同步 `AGENTS.md` + `README.md` + `copilot-instructions.md`
  4. 全量回归测试通过
  5. 通知 CppTLM 同步升级（HSK-2 重新发出）

**考虑的替代方案**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **A. Pin 4.13.2 + 修复文档**（✅ 选中） | 与实际一致 + 编译期锁定 | 升级需手动同步 | **✅** |
| B. 跟随上游最新 | 总是新特性 | CppTLM CI 风险传递（每次升级都需协调）| ❌ F12b-LD R7 |
| C. pkg-config / system ANTLR4 | 与系统解耦 | vendored 完全是为隔离 ANTLR4（CppTLM CI 不会牵连）| ❌ |

**约束**：
- **HSK-2 必须**在 Phase 1 之前发出（与 HSK-1 同 commit）
- 升级时**必须**同步通知 CppTLM 团队

---

### 决策 D-PTX-5: 错误码映射（PTX-EMU 内部 ↔ cudaError_t ↔ 日志级别）

**结论**：✅ **5 类条件 + 返回值一致性表**（参照任务书 §5.1）

**详细方案**：

`g_cpptlm_bridge == nullptr` 检查由 `if (g_cpptlm_bridge)` 在桥接入口完成，不会进入异步路径；故原"bridge 未初始化但调用异步路径"为死条件，统一重述为 `cpptlm_attach_bridge()` 状态判定。

| 条件 | 返回值 | 错误名 | 日志级别 | 触发位置 |
|------|--------|--------|---------|---------|
| `cpptlm_attach_bridge()` 已调用但 bridge 对象内部状态未就绪 | `cudaErrorNotYetInitialized` = 600 | "bridge internal not ready" | ERROR | `cudaLaunchKernel` 桥接 `submit_kernel` 返回该错误码时 |
| `submit_kernel` 参数无效 | `cudaErrorInvalidValue` = 11 | "invalid kernel params" | ERROR | `cudaLaunchKernel` 桥接失败处理 |
| `global_access` 地址未映射 | `UINT64_MAX` (0xFFFFFFFFFFFFFFFF) | "address not mapped" | WARN | `LdHandler/StHandler` bridge 分支 |
| `poll_kernel` 未知 `kernel_id` | `UINT64_MAX` | "unknown kernel_id" | WARN | `cudaStreamSynchronize` 轮询 |
| Bridge 版本不匹配 | 编译期 `static_assert`（不运行时） | "ABI version mismatch" | FATAL 中止 | 编译期 |

**实现的错误码转发**：

```cpp
// cudaLaunchKernel 异步分支
int ret = g_cpptlm_bridge->submit_kernel(...);
if (ret == static_cast<int>(cudaErrorNotYetInitialized)) {
    PTX_DEBUG_EMU("ERROR: bridge not initialized");
    return cudaErrorNotYetInitialized;
}
if (ret == static_cast<int>(cudaErrorInvalidValue)) {
    PTX_DEBUG_EMU("ERROR: invalid kernel params");
    return cudaErrorInvalidValue;
}
// 通用转发
if (ret != 0) return static_cast<cudaError_t>(ret);

// LdHandler/StHandler bridge 分支
uint64_t latency = g_cpptlm_bridge->global_address(...);
if (latency == UINT64_MAX) {
    PTX_DEBUG_EMU("WARN: address not mapped, fallback");
    // 继续走 PTX-EMU 内部路径
}
```

**考虑的替代方案**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **A. 5 类显式映射 + 通用转发**（✅ 选中） | 透明、可调试、易扩展 | 需要在 4 个位置分别处理 | **✅** |
| B. catch_cuda_error 异常 | 类型安全 | cudaError_t 不是异常类型（C API）| ❌ |
| C. 完全静默转发 | 极简 | 失去调试可观测性 | ❌ |

**约束**：
- PTX-EMU 内部错误（如 `PtxEmuException`）仍然走原有 `try/catch` 路径（`cudart_sim.cpp:376-383`），不与 bridge 错误码混合
- `cudaErrorUnknown = 999` 保留作为 PTX-EMU 内部未捕获异常的兜底

---

### 决策 D-PTX-6: 性能预算（vtable 优化 + 编译期内联）

**结论**：✅ **vtable 优化 + LTO/内联 + ABI 边界限制调用频次**

**详细方案**：

| 性能压力点 | 优化策略 | 预期开销 |
|-----------|---------|---------|
| `bridge->submit_kernel()` 每 kernel 一次 | vtable 调用 + 编译期 ABI 内联（`-flto`） | < 50 cycles / call |
| `bridge->poll_kernel()` 每 stream-sync 一次 | 同上 | < 50 cycles / call |
| `bridge->global_access()` 每 GLOBAL LD/ST 一次 | **最热路径**；vtable + PGO + LTO | < 100 cycles / call |
| `g_cpptlm_bridge == nullptr` 分支检查 | 编译期分支预测 + 零开销预测 | 1 cycle / check |

**实现技术**：
1. **vtable 调用**：`CppTLMBridge` 接口的 5 个虚方法通过单一 vtable 调度，开销可控
2. **LTO + inline**：`cmake --build build -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` 允许 CppTLM Adapter 跨 TU 内联
3. **PGO**（可选）：`cmake --build build -DCMAKE_CXX_FLAGS_PROFILE_GENERATE=ON` 在真实 kernel 上 profile；再 `USE_GENERATE` 优化
4. **null check hoisting**：编译器会识别 `g_cpptlm_bridge` 不变模式并 hoisting 到函数边界

**实现的性能模式**：

```cpp
// 编译期分支预测
if (__builtin_expect(g_cpptlm_bridge != nullptr, 0)) {
    // bridge 路径
} else {
    // 原同步路径（最常见 fallback）
}
```

**考虑的替代方案**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **A. vtable + LTO + null hoisting**（✅ 选中） | 与现有 PTX-EMU 编译流程一致 | 依赖 `-flto` 启用 | **✅** |
| B. 静态函数指针表 | 极低 vtable 开销 | 失去多态性 + ABI 不兼容 | ❌ 破坏 D-PTX-1 ABI |
| C. asm hint（__attribute__((hot))) | 编译器优化提示 | GCC/Clang 差异 | ❌ 跨编译器不一致 |

**约束**：
- **F12b-LD 性能预算**：±15% vs gpgpu-sim baseline（任务书 §5.1 G-D5）
- **延迟**: 实测 `bridge->global_access()` 开销 < 100 cycles / call（5 类 microbenchmark 验证）
- **优化不开箱即用**：默认 Release 模式启用 LTO；Debug 模式禁用

---

## 决策汇总表

| # | 决策 | 结论 | 实施位置 |
|---|------|------|---------|
| **D-PTX-1** | `g_cpptlm_bridge` 全局指针位置 | 静态全局 + 懒初始化 | `include/cudart/cpptlm_bridge.h` (extern) + `src/cudart/cudart_sim.cpp` (定义) |
| **D-PTX-2** | 全局单例共存策略 | SingletonGuard + FATAL 中止 | `src/cudart/cudart_sim.cpp` (4 个 init 入口前) |
| **D-PTX-3** | `exe_once()` 注入代码定位 | sm_context.cpp:222 + 253/338 | `src/ptxsim/core/sm_context.cpp` (姊妹 change `cpptlm-phase8b-injection-points`) |
| **D-PTX-4** | ANTLR4 版本策略 | Pin 4.13.2 + 修复文档 | `.github/copilot-instructions.md` (4.13.1 → 4.13.2) |
| **D-PTX-5** | 错误码映射表 | 5 类条件 + 返回值 + 日志级别 | `src/cudart/cudart_sim.cpp` + `src/ptxsim/instructions/memory.cpp` |
| **D-PTX-6** | 性能预算策略 | vtable + LTO + null hoisting | `CMakeLists.txt` (-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON) |

---

## 决策影响范围

| 文件 | 类型 | 说明 |
|------|------|------|
| `include/cudart/cpptlm_bridge.h` | **新增** | 5 虚方法 + `CPPTLMBRIDGE_VERSION=1` + `g_cpptlm_bridge` extern |
| `src/cudart/cudart_sim.cpp` | **修改** | SingletonGuard + cudaLaunchKernel 异步 + Stream sync + PendingKernel |
| `src/ptxsim/instructions/memory.cpp` | **修改** | LdHandler/StHandler bridge 分支（D-PTX-5 错误码） |
| `src/ptxsim/core/sm_context.cpp` | **修改**（姊妹 change）| 三段式注入（D-PTX-3 定位）|
| `CMakeLists.txt` | **修改** | BUILD_LIB_CPPTLM_CUDART + LTO flag（D-PTX-6）|
| `.github/copilot-instructions.md` | **修改** | 4.13.1 → 4.13.2（D-PTX-4）|

---

## 后果

### 正面影响

1. **CppTLM 协同完整**：所有 6 项决策明确，CppTLM 团队可同步 rebase
2. **HSK-1/2/3 状态已定义**：HSK 状态机（本文 §HSK 状态机）锁定触发时机与验证命令；ADR Accepted 后启用
3. **多实例仿真安全**：D-PTX-2 SingletonGuard 阻止静默状态损坏
4. **向后兼容**：所有 6 项决策保留 `nullptr` 默认值回退路径，零退化
5. **性能可控**：D-PTX-6 vtable + LTO 在 ±15% 预算内

### 负面影响

1. **6 项决策交叉引用**：实施时需同时考虑 D-PTX-1（pointer position）+ D-PTX-2（guard）+ D-PTX-3（injection point）+ D-PTX-6（optimization）的协同
2. **CMake ON 路径构建时间**：LTO + cpptlm ExternalProject_Add 可能增加首次 build 30-60s
3. **调试复杂度**：D-PTX-2 FATAL 中止需要清晰的错误信息（已在方案中提供）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| **R1**：D-PTX-3 注入行号遗漏 fast-path 或 slow-path | 中 | 高 | Lessons Learned #1 跨模块状态翻译 + 强测试覆盖两条分支 |
| **R2**：D-PTX-2 SingletonGuard 在 release build 与 debug build 行为不一致 | 低 | 中 | Release + Debug 都启用；测试覆盖 |
| **R3**：D-PTX-4 ANTLR4 升级破坏 PTX 语法解析 | 低 | 高 | 半年 review 流程 + 升级前 fork branch + 全量回归 |
| **R4**：D-PTX-5 错误码转换在 CppTLM 端返回值 0 时误判 | 中 | 中 | 编译期 `static_assert` 边界保护 + 5 类条件显式映射 |
| **R5**：D-PTX-6 vtable 性能不达预期（>15% baseline） | 低 | 中 | 实测 5 类 microbenchmark；可降级到 PGO 而非 LTO |

---

### HSK 状态机（强制）

| # | 状态 | 触发时机 | 验证命令 |
|---|------|---------|---------|
| **HSK-1** | `cpptlm_bridge.h` ABI commit hash | ADR-0020 Accepted + 本 ADR Accepted + Phase A commit 完成 | `git log --oneline -- include/cudart/cpptlm_bridge.h \| head -1` |
| **HSK-2** | ANTLR4 版本号 + CI yml 证据 | Phase D (CMake 集成) 完成 + 4 权威源一致 | `grep -nE "antlr4\|ANTLR" AGENTS.md README.md .github/copilot-instructions.md` |
| **HSK-3** | `libcpptlm_cudart.so` CMake 暴露方式 | Phase E (libcpptlm_cudart.so 集成) 完成 + CppTLM_COMMIT_HASH 锁定 | `grep -n "ExternalProject_Add\|cpptlm_FOUND" CMakeLists.txt` |

**禁止**：在 ADR 状态为 Proposed 时发出任一 HSK（违反 OpenSpec lifecycle lessons-learned §6）。

---

## 实施纪律（强制项）

来自 `.opencode/skills/ptx-lessons-learned/SKILL.md`：

1. **基线 worktree**：Phase 0.5 建立 `git worktree add ../ptxemu-baseline-f12b main`（Lessons Learned #4）
2. **分 Phase commit**：每个 Phase 独立 commit + 独立验证；失败立即 revert（Lessons Learned #3）
3. **OpenSpec artifacts 2-Phase commit**：artifacts（proposal/design/tasks/spec/internal-plan）必须先 `git add` + commit，再实施代码（Lessons Learned #6）
4. **跨模块状态翻译审计**：cudaLaunchKernel/cudaStreamSynchronize 涉及 `g_pending_kernels` 多处写入，需 state-modification-audit（Lessons Learned #1）
5. **TDD 强制**：测试用例（Phase 7）必须先于实现存在并失败（Red 阶段）；实施完成后 Green 阶段全绿
6. **ADR Status Update**：本 ADR 状态从 Proposed → Active 的转换必须在所有 Phase commit 完成 + Oracle 审查通过 + 测试基线 0 回归后执行

---

## 合规检查（Apply 阶段）

后续相关开发应检查：

- [ ] `g_cpptlm_bridge == nullptr` 时现有 600+ 测试字节级一致（通过 baseline worktree 对照）
- [ ] SingletonGuard 4 个入口都加检测（`grep -n "SingletonGuard" src/cudart/cudart_sim.cpp` ≥ 4）
- [ ] `exe_once()` 三段式注入覆盖 fast-path + slow-path 双分支
- [ ] ANTLR4 版本一致性：`grep -nE "antlr4|ANTLR" AGENTS.md README.md .github/copilot-instructions.md` 全为 4.13.2
- [ ] 错误码 5 类映射表覆盖：bridge 未初始化 + 参数无效 + 地址未映射 + 未知 kernel_id + 版本不匹配
- [ ] LTO flag 默认开启（`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`）
- [ ] 3 个 Handshake（HSK-1/2/3）已发出
- [ ] `./scripts/sanity.sh` 全绿 + OpenSpec `status --change cpptlm-d1-full` 输出 `applyRequires=[]`
- [ ] `docs/dev-process/lessons-learned.md` 追加新经验条目

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-07-15 | 初始版本（Proposed） | PTX-EMU Architecture Team |
| 2026-07-16 | **Active 转换**：3 轮 Metis pre-impl review 完成；7 Phase commits 修复所有 5 个 BLOCKER（B1 ABI 实现 / B2 sync loop / B3 stream destroy UB / B4 HSK 一致性 / B5 CMake 文档）+ sister spec 附录；205 tests PASS（1 pre-existing failure out of scope）；ABI 符号已导出（`nm -D` 验证 `T cpptlm_attach_bridge` + `T cpptlm_detach_bridge`）| PTX-EMU Architecture Team |

---

## 2026-07-16 Postmortem：cpptlm-d1-full 实施回顾

### 实施回顾

| Phase | 内容 | Commits | 状态 |
|-------|------|---------|------|
| Pre-flight | sm_context.cpp 行号 + ADR-0020 就绪度评估 + sister change 状态 | n/a | ✅ |
| Docs Cycle 1 | ADR-0021 虚假状态 + Checklist J + ADR-0020 → Accepted + sister internal-plan + HSK 状态机 | `603bd8bc` `a0be543b` `e8cf6f2d` `8f739f73` `bbd9d6bd` `77302f0b` | ✅ |
| Code Cycle 1 | B1 ABI 实现 + B3 stream destroy UB 修复 + B2 sync loop + B4 HSK cleanup + B5 CMake 文档 | `de016f79` `6cbdcc4c` `5dcccf40` `c38c31e4` `0456418e` `88d5962e` | ✅ |
| Final Sync | ADR-0021 → Active + tasks.md Phase 1 同步 + lessons-learned §N 沉淀 | （本 change） | ✅ |

### 同期发现的 5 个独立 bug（含根因 + 修复）

#### Bug 1（Metis 二审 B1）：ABI 声明无定义导致链接失败
- **位置**: `include/cudart/cpptlm_bridge.h:161,168`（声明）；`src/cudart/cudart_sim.cpp:1-1167`（无定义）
- **根因**: 头文件先于实现 commit，违反"声明 + 实现必须配对"原则
- **修复** (`de016f79`): 在 `cudart_sim.cpp` 实现 `cpptlm_attach_bridge(CppTLMBridge*)` + `cpptlm_detach_bridge()`
- **教训**: 见 lessons-learned.md §**34**
- **测试**: `tests/unit/cpptlm/test_cpptlm_attach_bridge.cpp` 3 个测试 PASS

#### Bug 2（Metis 二审 B3）：`cudaStreamDestroy` UB 删除 + 集合泄漏
- **位置**: `src/cudart/cudart_sim.cpp:886-894`（修复前）
- **根因**: `*stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_id))` 把 uint64_t 编入指针，再 `delete reinterpret_cast<int*>(stream)` 是 UB；同时未从 `g_active_streams` 移除
- **修复** (`6cbdcc4c`): 移除 `delete`；对称 `g_active_streams.erase(stream_id)`，复用 `g_pending_kernels_mutex`
- **教训**: 见 lessons-learned.md §**35**（"stream 句柄编码：uintptr_t reinterpret_cast 必须禁止 delete"）
- **测试**: `tests/unit/cudart/test_stream_destroy.cu` 4 个测试 PASS；原预存 `unit_cuda_stream_handle` SEGFAULT 被解决

#### Bug 3（Metis 二审 B2）：`cudaStreamSynchronize` / `cudaDeviceSynchronize` 单轮轮询即返回
- **位置**: `src/cudart/cudart_sim.cpp:896-937` / `:781-816`（修复前）
- **根因**: 实现为单次 poll，符合"主动语义"要求但违反 CUDA 同步契约（必须阻塞到完成）
- **修复** (`5dcccf40`): 替换单轮为 `while` 循环；处理 `remaining==0` (完成) + `UINT64_MAX` (未知)；`std::this_thread::yield()` 防忙转
- **教训**: 见 lessons-learned.md §**36**（"同步函数语义 = while 循环，非单次 poll"）
- **测试**: `tests/unit/cudart/test_stream_sync_loop.cpp` 4 个契约测试 PASS

#### Bug 4（Metis 二审 B4）：HSK 文档状态机自相矛盾
- **位置**: `openspec/changes/cpptlm-d1-full/hsk-1.md:148-149` + `tasks.md:289`
- **根因**: hsk-1.md footer 写"可立即发送"，但正文状态和 tasks.md 验收写"已发出"，违反 Checklist J（artifacts 内部一致性）
- **修复** (`c38c31e4`): 全部统一到"⏳ 待发出（ADR Accepted 后启用）"；在 ADR-0021 新增 §HSK 状态机详述发出时机
- **教训**: 见 lessons-learned.md §**37**（"HSK 文档状态必须与 ADR 生命周期同步"）
- **测试**: 文档级 grep 验证

#### Bug 5（Metis 二审 B5）：CMake 文档与实现发散
- **位置**: `openspec/changes/cpptlm-d1-full/design.md:307-317` vs `CMakeLists.txt:122-152`（commit `d0803a09`）
- **根因**: design 写 `find_package(cpptlm) + add_subdirectory(src/cudart/cpptlm_bridge) + target_link_libraries(ptxemu_runtime PRIVATE cpptlm::core)`，但实际 commit `d0803a09` 实现为 `ExternalProject_Add`
- **修复** (`0456418e`): 更新 design.md 与 spec.md 与实际 `ExternalProject_Add` 实现一致
- **教训**: 见 lessons-learned.md §**37**（"文档 vs 实现发散：git log -- <file> 是 source of truth"）

### 验证结果（最终 commit `de016f79`+`6cbdcc4c`+`5dcccf40`+`c38c31e4`+`0456418e`+`88d5962e`）

| 类别 | 通过 | 失败（基线也失败）|
|------|------|------------------|
| e2e | 18/19 | 1（`e2e_divergence` — 预存的多实例限制 ADR-0021 D-PTX-2，base worktree `9be56f8f` 同样失败）|
| integration | ~80/80 | 0 |
| unit | ~95/95 | 0（含 7 个新增 cpptlm/cudart 测试）|
| **本次引入的回归** | **0** | — |

### 元教训：本次发现的新失败模式

1. **ABI 声明 vs 定义分离**（lessons-learned §34）：违反"header 引入 = source 引入"原则；ABI 真值源文档表面完成 ≠ 实际可用
2. **uintptr_t reinterpret_cast 编码陷阱**（§35）：把整数编码进指针类型立即破坏"delete-from-pointer"假设
3. **同步语义 ≠ 单次 poll**（§36）：任何"poll-then-return"实现都要在 commit 前用规格语义反查
4. **HSK 状态机与 ADR 生命周期脱钩**（§37）：OpenSpec artifact 状态必须与 ADR 状态机同步，禁止"README 已 Active 但 ADR 文件仍 Proposed"
5. **文档/实现发散静默发生**（§37）：`ExternalProject_Add` 实施后 5+ 天文档未同步，git diff 不会自动传播 design.md

### 相关链接

- [`docs/dev-process/lessons-learned.md`](../dev-process/lessons-learned.md) — 完整经验沉淀（§32 + §34-37 本次新增）
- [`openspec/changes/cpptlm-d1-full/`](../../openspec/changes/cpptlm-d1-full/) — 当前 change artifacts
- [`openspec/changes/cpptlm-phase8b-injection-points/`](../../openspec/changes/cpptlm-phase8b-injection-points/) — 姊妹 change artifacts（ADR-0020）

---

## 2026-07-19 Postmortem：cpptlm-p1-ptxemu-shim 实施回顾

### 实施回顾

| Phase | 内容 | Commits | 状态 |
|-------|------|---------|------|
| 0 | OpenSpec artifacts 提交 | `f6c7279a` | ✅ |
| 1 | PtxInterpreter::prepareKernelLaunchRequest() 提取 | `13325cd8` | ✅ |
| 2 | PtxEmuDriverShim 实现 | `13325cd8` | ✅ |
| 3 | Bridge path 双路径 enqueue + on_complete 回调链 | `13325cd8` | ✅ |
| 4 | cpptlm_set_driver ABI + PtxEmuDriverApi vtable | `13325cd8` | ✅ |
| 5 | 构建配置（pin commit + PIC） | `a541e074`, `35ffeba8`, `97539fdb` | ✅ |
| 6 | 验证：ctest 212/213 PASS + nm T symbol | — | ✅ |

### 关键技术决策

#### D7: Weak symbol vs --whole-archive

最初设计使用 `__attribute__((weak))` 让 CppTLM 的强定义覆盖 PTX-EMU 的弱定义。
但静态库（`.a`）链接时，链接器仅在符号未定义时才拉入对象文件——弱定义已满足
符号需求，CppTLM 的强定义被跳过。解决方案：`-Wl,--whole-archive cpptlm_core -Wl,--no-whole-archive`。

#### D8: add_subdirectory 替代 ExternalProject_Add

本地开发环境使用 `add_subdirectory(/workspace/project/CppTLM)` 替代 `ExternalProject_Add`，
配置项 `CPPTLM_SOURCE_DIR` 支持覆盖。CppTLM 需要 PIC：`set(CMAKE_POSITION_INDEPENDENT_CODE ON)`
在 `add_subdirectory` 前设置，完成后恢复。

#### D9: PtxEmuDriverApi vtable 跨 .so ABI

8 个函数指针的 vtable struct 在 cpptlm_bridge.h（ABI 真值源）定义，PTX-EMU 侧通过
static wrapper 函数填充，CppTLM 侧 DriverWrapper 通过 vtable 调用，避免直接依赖
PTX-EMU 符号。CppTLM 侧 `01882a2` 已实现对应 DriverWrapper。

### 验证结果（BUILD_LIB_CPPTLM_CUDART=ON, commit 97539fdb）

| 类别 | 通过 | 预存失败 |
|------|------|---------|
| ctest 全量 | 212 | 1 (e2e_divergence, SingletonGuard) |

### 新增教训（追加到 lessons-learned.md §41）

1. **静态库 weak symbol 无法被同 .so 内的强定义覆盖** — 必须使用 `--whole-archive`
2. **add_subdirectory 外部项目时需保存/恢复 CMAKE_POSITION_INDEPENDENT_CODE**
3. **跨仓库 vtable ABI 设计**: 函数指针 struct 比直接头文件依赖更灵活

### 相关链接

- [`docs/dev-process/lessons-learned.md`](../dev-process/lessons-learned.md) §41
- [`openspec/changes/archive/2026-07-19-cpptlm-p1-ptxemu-shim/`](../../openspec/changes/archive/2026-07-19-cpptlm-p1-ptxemu-shim/)
- CppTLM 侧: commit `01882a2`, PR #N/A

---

## 2026-07-20 Postmortem：e2e-cosim-kernel-verify 实施回顾

### 实施回顾

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | OpenSpec artifacts + 1 行可见性变更（移除 `g_ptx_emu_driver_shim` 的 `static`） | ✅ |
| 1 | E2E 测试 + MockBridge + vectorAdd kernel（内联） | ✅ |
| 2 | 验证：BUILD_LIB_CPPTLM_CUDART=ON 测试 PASS（17 assertions），OFF 测试目标不存在 | ✅ |

### 关键技术决策

#### D10：测试 attach bridge AFTER launch（规避已知 bug）

`openspec/changes/e2e-cosim-kernel-verify/design.md` D6 规定测试需在 `cudaLaunchKernel` **之前** attach bridge，让 `cudaLaunchKernel` 走 dual-enqueue 路径。实施中发现 bridge 路径存在 **2-cycle completion bug**（见下），调整为 attach bridge AFTER launch，kernel 走同步路径（`launchPtxInterpreter` + `wait_for_completion`），bridge 仅用于 `cudaDeviceSynchronize` 的 polling loop。

#### D11：count_kernel_args 修复拆分为独立 follow-up

实施中曾为 bridge 路径 `cudaLaunchKernel` 的 `count_kernel_args` 添加 segfault 修复（用 PTX context `kernelParams.size()` 替代 nullptr 哨兵遍历）。Oracle 审查指出此修复超出 design.md:131-135 Non-Goals（"不修改核心桥接逻辑"），且因测试走同步路径而不被触发。已 revert，提议独立 change `fix-bridge-arg-count-segfault`。

### 已知限制（deferred to follow-up changes）

#### L1：Bridge path 2-cycle completion bug

- **症状**：bridge attached BEFORE `cudaLaunchKernel` 时，`g_ptx_emu_driver_shim->advance()` 返回 `result=2 (KernelComplete) actual=2`，kernel 输出全零
- **复现**：`test_cosim_vector_add.cu` 中将 `cpptlm_attach_bridge(&mock)` 移到 `vectorAdd<<<>>>()` 之前即可触发
- **根因假设**（Oracle Q2 Hypothesis 3）：`GPUContext::exe_once()` 在同一调用内既 admit kernel 又因 SM state 立即判 `all_warps_finished()`，导致 `gpu_state = EXIT`。同步路径不触发因 SM 初始化上下文不同
- **修复位置**：`src/ptxsim/core/gpu_context.cpp:246-336`（`exe_once`）+ `src/ptxsim/core/sm_context.cpp:350-400`（`SMContext::exe_once` + `all_warps_finished`）
- **Follow-up change**：`fix-bridge-path-2-cycle-exit`（多日 runtime 调试）

#### L2：Bridge path arg-count segfault

- **症状**：kernel args 含非指针类型（如 `int N`）时，`cudaLaunchKernel` bridge 路径 `count_kernel_args` 的 nullptr 哨兵遍历越界 segfault
- **触发场景**：`vectorAdd<<<>>>(d_A, d_B, d_C, N)` 当 `g_cpptlm_bridge != nullptr` 时
- **Follow-up change**：`fix-bridge-arg-count-segfault`（独立 TDD change）

### 教训（追加到 lessons-learned.md §42）

1. **OpenSpec Non-Goals 是硬约束**：`design.md` 的 Non-Goals 段不可在实施中悄然扩展。即使发现了真实 bug，也应 split 成独立 change 以保留原 change 的 scope purity 和审计追溯能力
2. **测试路径偏离应显式 amend spec**：当实施中无法满足 spec 的 SHALL 语句时，必须显式 amend spec 加入 known limitation Scenario + NOTE 解释。禁止默默改测试路径而 spec 保持原样
3. **Oracle 审查是 scope drift 的有效检测器**：Oracle 审查发现 test_cosim_vector_add.cu 实际走同步路径而 spec 要求 bridge 路径，以及 `count_kernel_args` 修复违反 Non-Goals。两个偏差均通过文件:行级证据被精确识别
4. **TDD 三阶段纪律可避免 scope creep**：当实施中发现需要修改生产代码超出原始 scope 时，应先暂停并 propose 独立 change，而非直接修改以保持"测试通过"

### 相关链接

- [`openspec/changes/e2e-cosim-kernel-verify/`](../../openspec/changes/e2e-cosim-kernel-verify/)（本 change）
- Follow-up: `openspec/changes/fix-bridge-arg-count-segfault/`（待 propose）
- Follow-up: `openspec/changes/fix-bridge-path-2-cycle-exit/`（待 propose）
- Test: `tests/e2e/cosim/test_cosim_vector_add.cu`

## 2026-07-21 Fix Record: auto-co-sim-standalone implements L1 + L2 + auto-attach + auto-advance

### Fix Summary

| ID | Issue | Fix Mechanism | Commit |
|----|-------|---------------|--------|
| L1 | 2-cycle completion bug | Auto-advance: `cudaDeviceSynchronize` calls `advance()` which repeatedly calls `exe_once()` until EXIT, naturally separating admit from execution | `auto-co-sim-standalone` Phase 1 |
| L2 | Arg-count segfault | `count_kernel_args` replaced by PTX context `kernelParams.size()` lookup with `SIZE_MAX` sentinel for fallback | `auto-co-sim-standalone` Phase 1 (D3) |
| — | Manual bridge attach | `StubBridge` auto-attached in `initialize_environment()` when `BUILD_LIB_CPPTLM_CUDART=ON` | `auto-co-sim-standalone` Phase 1 (D1) |
| — | Manual advance() call | Auto-advance in `cudaDeviceSynchronize` and `cudaStreamSynchronize(0)` before poll loop | `auto-co-sim-standalone` Phase 1 (D2) |

### Key Insight: Auto-advance inherently resolves L1

The original 2-cycle completion bug manifests when a single `exe_once()` call both admits a kernel (via `execute_kernel_internal`) and immediately judges `all_warps_finished()` → EXIT — kernel never executes, output all zero.

The auto-advance mechanism (`g_ptx_emu_driver_shim->advance()`) calls `exe_once()` in a `while` loop until `GPUContext::get_state() == EXIT`. This means:
1. First `exe_once()`: admits kernel, sets SM to RUN, may or may not execute a warp
2. Subsequent `exe_once()`: SMs are RUN, executes instructions, eventually reaches EXIT

Thus the L1 bug is **resolved by the auto-advance architecture** without explicit `exe_once()` refactoring.

### Advance ceiling safety

Environment variable `PTX_EMU_MAX_ADVANCE_CYCLES` (default 10M) prevents pathological kernels (infinite loops / barrier deadlock) from hanging forever. Exhausting the ceiling triggers `GPUContext::clear_requests()` + `g_pending_kernels` cleanup + `(cudaError_t)999` return.

### Regression Results

| Mode | Test | Result |
|------|------|--------|
| OFF | `ctest -L e2e` (excl. pre-existing SingletonGuard) | All PASS |
| OFF | `ctest -L unit` | 88/88 PASS |
| OFF | `ctest -L integration` | All PASS |
| ON | `e2e_cosim_vector_add` | PASS (64/64 golden match) |
| PTX | `tests/ptx/test_all_ptx.sh` | 46/46 PASS |

### Verification Artifacts

- `tests/e2e/cosim/test_cosim_vector_add.cu` — pure standard CUDA program (no PTX-EMU APIs)
- `src/cudart/stub_bridge.h` — zero-latency StubBridge (5 virtual methods)
- `src/cudart/cudart_sim.cpp` — D1 (auto-attach) + D2 (auto-advance) + D3 (arg-count fix) + D6 (static cleanup)
- `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` — removed `extern g_ptx_emu_driver_shim`
- `include/ptxsim/gpu_context.h` — added `clear_requests()`
- `src/ptxsim/core/gpu_context.cpp` — implemented `clear_requests()`
- `tests/e2e/CMakeLists.txt` — removed `BUILD_LIB_CPPTLM_CUDART` condition
