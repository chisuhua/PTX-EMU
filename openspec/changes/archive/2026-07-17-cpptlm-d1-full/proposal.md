## Why

[CppTLM 综合任务书 §2](https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md) 要求 PTX-EMU 实施 §2 (P0) F12b-LD MemoryBridge 的 5 项改造 (#1~#5) 以建立 CppTLM 作为唯一时钟真相源（clock-of-truth）的基础设施。同时 [PTX-EMU-README §10](https://github.com/chisuhua/PTX-EMU/blob/main/docs/superpowers/specs/PTX-EMU-README.md) 明确列出 **6 项 PTX-EMU 端自主决策 (D-PTX-1~6)** 与 **3 个回传给 CppTLM 的 handshake (HSK-1/2/3)**，由本仓库自治。

本 change 完成：
- **§2 MemoryBridge F12b-LD 基础设施**（5 项 PTX-EMU 端任务 #1-#5，覆盖 3 天工时）
- **6 项 PTX-EMU 端自主决策 + 3 个 handshake**（D-PTX-1~6 + HSK-1/2/3，PTX-EMU 自治决策点）
- **PTX-EMU 内部 ADR-0021**（D-PTX-1~6 决策记录）
- **ABI 真值源 `include/cudart/cpptlm_bridge.h`**（PTX-EMU 是 ABI 提供方，CppTLM 通过 `ExternalProject_Add` + path include 消费）

姊妹 change [`cpptlm-phase8b-injection-points`](../../cpptlm-phase8b-injection-points/)（ADR-0020）已并行覆盖 §3 D1-Full 三段式注入（IScoreboard + IPipelineLatencyProvider + ITensorCoreTiming），与本 change 并行实施。

**前置 change 假设**：
- CppTLM `openspec/changes/2026-06-24-gpu-soc-phase8b-core/` 已修订为 D1-Full ✅
- CppTLM 任务书 `2026-07-14-ptxemu-comprehensive-modification-plan.md` §2 已签发 ✅
- 协作同步 `2026-07-01-f12b-ld-ptxemu-collaboration-sync.md` §4 `CppTLMBridge` 接口已对齐 ✅
- 姊妹 change `cpptlm-phase8b-injection-points` (`openspec/changes/cpptlm-phase8b-injection-points/`) 已 Proposed ✅

**关联 PTX-EMU 经验沉淀**（来自 `docs/dev-process/lessons-learned.md`）：
- **#4 基线 worktree**：实施前 1 分钟建立 baseline，节省数小时争论
- **#3 分 Phase commit**：每个 Phase 独立可回退；任何已有测试回归 → 立即 revert
- **#6 OpenSpec artifacts 2-Phase commit**：artifacts（proposal/design/tasks/spec）必须先 `git add` + commit，再实施代码
- **#1 跨模块状态翻译**：`g_pending_kernels`/`cudaStreamSynchronize` 多处写入需集中审计（用 `state-modification-audit` skill）
- **#7 Pre-implementation Review**：实施前跑 Metis 审计（避免 ANTLR4 版本策略决策假设错误）
- **#5 类型判断依赖 `qualifiers.back()`**：`is_global_space()` 必须遍历整个 qualifier 列表而非仅末尾（LD/ST handler 调用栈）

**关联 ADR**：
- [ADR-0020](../../docs/adr/ADR-0020-cpptlm-injection-points.md)：姊妹 change 决策依据（§3 注入点）
- [ADR-0021](../../docs/adr/ADR-0021-cpptlm-d1-full-integration.md)（本 change 创建）：D-PTX-1~6 自主决策
- [ADR-0009](../../docs/adr/ADR-0009-xmacro-instruction-dispatch.md)：StatementType 枚举来源
- [ADR-0010](../../docs/adr/ADR-0010-fake-cuda-runtime.md)：Fake CUDA Runtime（`cudart_sim.cpp` 当前实现）

**关联 Skill**：
- `ptx-lessons-learned`：经验沉淀快速决策树 + 4 个 checklist
- `ptx-instruction-pipeline`：指令执行流水线（`exe_once()` 上下文）
- `ptx-barrier-mechanism`：`blocked_cycles` 扩展至全指令的影响范围
- `state-modification-audit`：`g_pending_kernels` 跨模块写入审计
- `regression-bisect`：测试回归定位

## What Changes

### 新增产物

- **新增** `docs/adr/ADR-0021-cpptlm-d1-full-integration.md`：D-PTX-1~6 决策记录（PTX-EMU 自治）
- **新增** `include/cudart/cpptlm_bridge.h`：`CppTLMBridge` 抽象接口（5 个虚方法 + `CPPTLMBRIDGE_VERSION` 编译期断言 + `g_cpptlm_bridge` 全局指针 + `cudaStream_t` 宽度 `static_assert`）
- **新增** `include/cudart/cpptlm_bridge_impl.h` (optional)：Bridge 默认实现 stub（方便测试 fallback 路径）
- **修改** `src/cudart/cudart_sim.cpp`：
  - `cudaLaunchKernel`：当 `g_cpptlm_bridge != nullptr` 时走异步路径（submit + register pending kernel + 立即返回）；否则原有同步路径
  - `cudaStreamSynchronize`：按 `stream_id` 过滤 + 迭代器失效修复（先收集完成 id 再统一 erase）+ 调用 `bridge->poll_kernel()`
  - `cudaDeviceSynchronize`：遍历所有活跃 stream 同步
  - `cudaStreamCreate`：分配 64-bit 唯一 ID，插入 `g_active_streams`
  - 新增 `next_kernel_id` atomic counter + `PendingKernel` 数据结构（含 `stream_id` 字段） + `g_pending_kernels` map + `g_active_streams` set
- **修改** `src/ptxsim/instructions/memory.cpp`：`LdHandler::processOperation()` + `StHandler::processOperation()` 当 `g_cpptlm_bridge != nullptr && is_global_space(addr)` 时调用 `global_access()`（timing-only），数据仍读/写 `SimpleMemory`
- **修改** `CMakeLists.txt`：当 `cpptlm_FOUND` + `BUILD_LIB_CPPTLM_CUDART=ON` 时构建 `libcpptlm_cudart.so` + 链接 `cpptlm::core`

### 新增测试

- **新增** `tests/unit/cpptlm/test_cpptlm_bridge.cpp`：7 个 Bridge stub 测试（version + submit/poll/synchronize/global_access + nullptr fallback + cudaStream_t static_assert）
- **新增** `tests/integration/cpptlm/test_async_launchkernel.cpp`：真实 kernel 路径走异步 bridge
- **新增** `tests/integration/cpptlm/test_ld_st_bridge.cpp`：GLOBAL LD/ST 通过 bridge global_access 返回延迟 + 数据正确性对比 baseline
- **新增** `tests/integration/cpptlm/test_singleton_guard.cpp`：重复初始化时 FATAL 中止
- **修改** `tests/CMakeLists.txt`：注册 4 个新测试目标（`unit_cpptlm_bridge` + 3 个 integration_）

### 配置 / 文档更新

- **修改** `AGENTS.md`：在已知限制章节添加 §F12b-LD MemoryBridge 状态（桥接到 §2 任务清单）
- **修改** `docs/adr/README.md`：索引追加 ADR-0021
- **修改** `docs/dev-process/lessons-learned.md`：新增 §"Bridge 接口 6 项决策 + SingletonGuard 强制"经验条目
- **修改** `include/cudart/AGENTS.md`：记录 `cpptlm_bridge.h` 的 ABI 真值源地位 + bump `CPPTLMBRIDGE_VERSION` 流程

## Capabilities

### New Capabilities

- `cpptlm-bridge-interface`: `CppTLMBridge` 抽象接口 + `CPPTLMBRIDGE_VERSION=1` 编译期断言 + `cudaStream_t` 宽度 `static_assert` + `g_cpptlm_bridge` 全局指针（PTX-EMU 是 ABI 提供方）
- `cudart-async-launchkernel`: `cudaLaunchKernel` 异步路径（`submit_kernel` 立即返回 + 注册 PendingKernel 含 stream_id）
- `cudart-stream-synchronization`: `cudaStreamSynchronize` 按 `stream_id` 过滤 + 迭代器失效修复 + `cudaDeviceSynchronize` + `cudaStreamCreate` 句柄编码
- `ptx-global-ld-st-bridge`: `LdHandler`/`StHandler` 走 `global_access()` timing-only 路径（数据保留 SimpleMemory）
- `libcpptlm-cudart-integration`: CMake `libcpptlm_cudart.so` 集成构建（含 `CPPTLMBRIDGE_VERSION` 断言 + ANTLR4 version guard）

### Modified Capabilities

（空 — 本 change 不修改现有 openspec/specs/ 下的 capability，仅新增上 5 个）

## Impact

| 文件 | 类型 | 工时 | 验证 |
|------|------|:---:|------|
| `docs/adr/ADR-0021-cpptlm-d1-full-integration.md` | **新增** | 0.2d | Oracle 审阅 |
| `openspec/changes/cpptlm-d1-full/{proposal,design,specs/cpptlm-d1-full,tasks,internal-plan}.md` | **新增** | 0.3d | `openspec status --change cpptlm-d1-full` 输出 `applyRequires=[]` |
| `include/cudart/cpptlm_bridge.h` | **新增** | 0.2d | 编译通过 + `CPPTLMBRIDGE_VERSION == 1` + `static_assert(cudaStream_t <= uint64_t)` |
| `include/cudart/cpptlm_bridge_impl.h` (optional) | **新增** | 0.1d | 编译通过 + 默认 nullptr 实现 |
| `src/cudart/cudart_sim.cpp` | **修改** | 1.0d | `[unit;cudart] [integration;cudart] [e2e;cudart]` 0 回归 |
| `src/ptxsim/instructions/memory.cpp` | **修改** | 0.3d | `[unit;memory] [e2e;memory]` 0 回归 |
| `CMakeLists.txt` | **修改** | 0.2d | `cmake --build build --target cudart` PASS + `find_package(cpptlm)` mock |
| `tests/unit/cpptlm/test_cpptlm_bridge.cpp` | **新增** | 0.4d | 7 个 Mock 测试 PASS |
| `tests/integration/cpptlm/test_async_launchkernel.cpp` | **新增** | 0.4d | 异步路径 kernel 正确执行 |
| `tests/integration/cpptlm/test_ld_st_bridge.cpp` | **新增** | 0.4d | LD/ST 通过 bridge + 数据正确性 |
| `tests/integration/cpptlm/test_singleton_guard.cpp` | **新增** | 0.2d | 重复初始化 FATAL |
| `tests/CMakeLists.txt` | **修改** | 0.1d | 4 个 `add_catch_test` 注册 |
| `AGENTS.md` | **修改** | 0.1d | §已知限制章节更新 |
| `docs/adr/README.md` | **修改** | 0.1d | 索引追加 ADR-0021 |
| `docs/dev-process/lessons-learned.md` | **修改** | 0.2d | §F12b-LD 经验条目 |
| `include/cudart/AGENTS.md` | **修改** | 0.1d | Bridge ABI 真值源地位 |
| **合计** | | **~5d** | **~1100 LOC 增量（16 个文件，含测试）** |

**影响类别**：
- **新增公共 API 表面**：`CppTLMBridge`（5 个 public 虚方法）+ `g_cpptlm_bridge` 全局指针（PTX-EMU ↔ CppTLM ABI 真值源）
- **现有行为变更**：`cudaLaunchKernel` 异步路径（在 bridge == nullptr 时**字节级相同**）+ `cudaStreamSynchronize` 真实等待（不再立即返回）+ `LdHandler/StHandler` GLOBAL 分支（fallback 到现有 SimpleMemory）
- **依赖关系**：`cpptlm_bridge.h` 仅依赖 `<cstddef>` + `<cstdint>` + `<cuda_runtime.h>`（`size_t` + `cudaStream_t` 来源）；零 CppTLM 依赖
- **握手信号**：3 个 Handshake（HSK-1 ABI commit hash、HSK-2 ANTLR4 版本 + CI yml、HSK-3 libcpptlm_cudart.so CMake 暴露方式草案）

## References

- **CppTLM 综合任务书**：`CppTLM/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md` §2 + §10（947 行）
- **CppTLM 协作同步**：`CppTLM/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md` §4 + §5（332 行）
- **CppTLM ADR**：`CppTLM/docs/adr/ADR-NV-02-phase8b-d1-strategy.md` Status Update 2026-07-14
- **PTX-EMU 入口**：`CppTLM/docs/superpowers/specs/PTX-EMU-README.md` §10（317 行，6 项决策 + 3 个 handshake）
- **PTX-EMU 姊妹 change**：`openspec/changes/cpptlm-phase8b-injection-points/`（§3 D1-Full 三段式注入）
- **PTX-EMU 现有 OpenSpec changes**：
  - `cpptlm-phase8b-injection-points`（**并行**，§3 注入点）
  - `god-class-refactor-thread-context-phase3`（**并行**，关注字段迁移 — 但本 change 不修改 ThreadContext）
  - `migrate-bar-warp-sync-to-barrier-module`（**并行**，关注 barrier 交互 — 本 change 不修改 barrier 路径）

## ⚠️ 风险与历史教训（来自 `docs/dev-process/lessons-learned.md`）

1. **#4 基线 worktree 强制**：实施 PTX-#1 前 1 分钟建立 `../ptxemu-baseline-2026-07-XX`，节省数小时争论
2. **#3 分 Phase commit**：每个 Phase 独立可回退；任何已有测试回归 → 立即 revert
3. **#6 OpenSpec artifacts 2-Phase commit**：本 change 5 个 artifacts 必须先 `git add` + commit，再实施代码
4. **#1 跨模块状态翻译**：`cudaLaunchKernel`/`cudaStreamSynchronize` 涉及 `g_pending_kernels`/`g_active_streams` 多处写入，必须用 `state-modification-audit` skill 集中审计
5. **#7 Pre-implementation Review**：实施 PTX-#1 前跑 Metis 审计（避免 D-PTX-4 ANTLR4 版本策略假设错误 — README 与 copilot-instructions 一致性需澄清）
6. **#5 类型判断依赖 `qualifiers.back()`**：`is_global_space()` 实现必须遍历整个 qualifier 列表（与 cute_rmsnorm float 类型判断 bug 同源）

## ⚠️ 序列化考虑

- **依赖关系**：本 change 与 `cpptlm-phase8b-injection-points` 完全**并行**（主代码互不冲突；唯一共享点为 `tests/CMakeLists.txt` 的追加式 `add_catch_test` 行，追加而非覆盖，不阻塞）。两个 change 互不阻塞，可任意顺序启动。
- **互斥字段**：本 change 新增 `g_cpptlm_bridge` 全局指针 + `g_pending_kernels`/`g_active_streams` 不与现有字段冲突
- **测试基线**：实施前必须确认 `[unit;cudart]` `[integration;cudart]` `[e2e;cudart]` 测试基线 100% PASS
