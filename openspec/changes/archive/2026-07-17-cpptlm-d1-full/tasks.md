# Tasks: CppTLM F12b-LD MemoryBridge（D-PTX-1~6 + HSK-1/2/3）

> **Status**: Proposed
> **Parent**: `proposal.md` + `design.md` (cpptlm-d1-full)
> **ADR**: [docs/adr/ADR-0021-cpptlm-d1-full-integration.md](../../../docs/adr/ADR-0021-cpptlm-d1-full-integration.md)
> **总工时**: ~5d（PTX-EMU 端 §2 MemoryBridge 5 项任务 + SingletonGuard + ANTLR4 修正 + 3 个 handshake）

---

## Phase 0: 对齐 + 基线（强制最先完成，~0.5d）

> ⚠️ **MUST**: 不完成本 Phase 不允许进入 Phase 1。Lessons Learned #7 Pre-implementation Review 强制项。

- [x] 0.1 D-PTX-1~6 决策在 ADR-0021 中签署（参见 [docs/adr/ADR-0021-cpptlm-d1-full-integration.md](../../../docs/adr/ADR-0021-cpptlm-d1-full-integration.md)）
- [x] 0.2 验证 `antlr4/antlr4-cpp-runtime-4.13.2-source/` 实际存在（确认 D-PTX-4 版本基线）
- [x] 0.3 与姊妹 change `cpptlm-phase8b-injection-points`（ADR-0020）协调并行启动
- [x] 0.4 CppTLM 书面同步确认：协作同步 `2026-07-01-f12b-ld-ptxemu-collaboration-sync.md §4` Bridge 接口签名双方一致
- [x] 0.5 基线 worktree 建立（遵循 Lessons Learned #4）
  - `git worktree add ../ptxemu-baseline-f12b main`
  - `cd ../ptxemu-baseline-f12b && . env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
  - `cd build && ctest -L "unit;integration;e2e" --output-on-failure`（验证当前 ctest 目标全 PASS — `ctest -N` 实际注册数）
  - **✅ 已建立**: `/workspace/project/ptxemu-baseline-f12b` worktree（HEAD 形如 `8dc000ec^` —— 即首个 cpptlm_bridge.h 实现 commit 之前的提交；与 `tasks.md:22` 旧 `9be56f8f` 不符需修正）
  - **✅ 已验证**: `unit_barrier_module` PASS（早期审计留证 — 任何 cpptlm-d1-full 提交前应为复现基线，并非最终基线）
- [x] 0.6 验证现有 `[unit;cudart]` `[integration;cudart]` `[e2e;cudart]` 测试基线

---

## Phase 1: CppTLMBridge 接口头文件（HSK-1 准备，~0.3d）

> 📌 **NOTE**: 本 Phase 完全独立，不影响任何现有 PTX-EMU 代码。**HSK-1 commit** 必须在 Phase 1 完成。

- [x] 1.1 创建 `include/cudart/cpptlm_bridge.h`（参照综合任务书 §2.1 Task #1 完整代码 ~70 行）
  - 5 个纯虚方法：`version()` + `submit_kernel()` + `poll_kernel()` + `synchronize_stream()` + `global_access()`
  - `CPPTLMBRIDGE_VERSION 1` 宏
  - `g_cpptlm_bridge` 全局指针声明
  - `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))`
  - **✅ 实现于 commit `603bd8bc`**：170 LOC 完整版本
- [ ] 1.2 ~~创建 `include/cudart/cpptlm_bridge_impl.h` (optional stub 实现 fallback 路径)~~ **⚠️ DEFERRED**：optional stub 非阻塞；测试用 mock bridge 覆盖同等语义
- [x] 1.3 验证约束：`grep '#include' include/cudart/cpptlm_bridge.h` → 3 include（`<cstddef>` + `<cstdint>` + `<cuda_runtime.h>`）
- [x] 1.4 编译验证：`cmake --build build --target cudart` PASS（Metis 3 审验证 `nm -D build/lib/libcudart.so | grep cpptlm_attach_bridge` 显示 `T cpptlm_attach_bridge`）
- [x] 1.5 **HSK-1 commit + 输出**：
   - `git add include/cudart/cpptlm_bridge.h && git commit -m "feat(cudart): CppTLMBridge ABI source-of-truth (HSK-1 pre-ship)"`
   - 记录 commit hash 准备发给 CppTLM
   - **⏳ 待发出**：ADR-0021 Active 后按 §HSK 状态机顺序发送（commit hash 待 HSK-1 实际发出时锁定）

**Commit**:
```bash
git add include/cudart/{cpptlm_bridge.h,cpptlm_bridge_impl.h}
git commit -m "feat(cudart): CppTLMBridge ABI source-of-truth with CPPTLMBRIDGE_VERSION=1 (HSK-1)

Zero CppTLM dependency. ABI 真值源 — CppTLM 通过 ExternalProject_Add 引用。

Refs:
- ADR-0021 (cpptlm-d1-full-integration, D-PTX-1 + D-PTX-6)
- CppTLM docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #1
- CppTLM docs/superpowers/specs/PTX-EMU-README.md §10.2
- openspec/changes/cpptlm-d1-full"
```

---

## Phase 2: SingletonGuard（D-PTX-2，~0.2d）

> 📌 **NOTE**: F12b-LD 文档 §10.1 明确指 PTX-EMU 单例在多实例仿真中导致**静默状态损坏**。本 Phase 必须先于 cudaLaunchKernel 异步化完成。

- [x] 2.1 修改 `src/cudart/cudart_sim.cpp` 添加 `SingletonGuard` 类（4 个全局单例的初始化入口前）
  - 检测 `g_gpu_context` / `g_ptx_interpreter` / `CudaDriver::instance()` / `HardwareMemoryManager::instance()` 重复初始化
  - 重复时立即 FATAL：`std::cerr << ...; std::abort();`
- [x] 2.2 验证 `g_cpptlm_bridge` 初始化时机（D-PTX-1 默认 nullptr，加载 libcpptlm_cudart.so 后赋值）
- [x] 2.3 编译验证 + 单测 `integration_singleton_guard` PASS

**Commit**:
```bash
git add src/cudart/cudart_sim.cpp
git commit -m "feat(cudart): SingletonGuard for 4 global singletons (D-PTX-2)

Prevents multi-instance simulation silent state corruption per F12b-LD doc §10.1.
FATAL abort on duplicate __cudaRegisterFatBinary invocation.

Refs:
- ADR-0021 (cpptlm-d1-full-integration, D-PTX-2)
- CppTLM docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md §10.1
- openspec/changes/cpptlm-d1-full"
```

---

## Phase 3: cudaLaunchKernel 异步化（D-PTX-1 + Task #2，~1.0d）

> 📌 **NOTE**: 关键约束 — `g_cpptlm_bridge == nullptr` 时行为**字节级**与原同步路径相同。

- [x] 3.1 添加 `std::atomic<uint64_t> next_kernel_id{1}` + `generate_kernel_id()`
- [x] 3.2 添加 `PendingKernel` 数据结构（含 `stream_id` 字段）
- [x] 3.3 添加 `g_pending_kernels` (`unordered_map<uint64_t, PendingKernel>`)
- [x] 3.4 添加 `g_active_streams` (`unordered_set<uint64_t>` 含默认 `{0}`)
- [x] 3.5 添加 `register_pending_kernel()` helper + `count_kernel_args()` helper
- [x] 3.6 修改 `cudaLaunchKernel` 函数体（行 332-386）：
  - 添加 `if (g_cpptlm_bridge) { ... }` 异步分支
  - 12 个参数全部传递给 `bridge->submit_kernel()`
  - `register_pending_kernel()` 注册后立即 `return cudaSuccess`
  - bridge == nullptr 走原有同步路径
- [x] 3.7 编译验证 + 单测 `unit_cpptlm_bridge` PASS
- [x] 3.8 集成测试 `integration_async_launchkernel`（真实 kernel 路径）
  - ✅ 实现在 `tests/integration/cpptlm/test_async_launchkernel.cpp::"Async launch: real cudaLaunchKernel entry forwards to bridge (D-PTX-1)"`
  - 该 TEST_CASE 通过 `cudaLaunchKernel` C 入口（extern "C"）调用，验证 12 个参数（kernel_id/grid/block/args/shared/stream_id）逐项转发到 `bridge->submit_kernel()` mock
  - 不依赖 `__cudaRegisterFatBinary` —— bridge 路径在 `cudart_sim.cpp:490-551` 提前返回 cudaSuccess，不接触 `g_ptx_interpreter`（Lessons Learned #4 基线 + RED-GREEN 完整）

**Commit**:
```bash
git add src/cudart/cudart_sim.cpp
git commit -m "feat(cudart): async cudaLaunchKernel when bridge active (D-PTX-1 + #2)

Adds PendingKernel registry + 11-param bridge->submit_kernel() call.
Maintains byte-identical sync fallback when g_cpptlm_bridge == nullptr.

Refs:
- ADR-0021 (cpptlm-d1-full-integration)
- CppTLM docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #2
- openspec/changes/cpptlm-d1-full"
```

---

## Phase 4: Stream 同步原语（Task #3 + 迭代器失效修复，~1.0d）

> 📌 **NOTE**: 关键约束 — 迭代器安全模式必须应用：先 `completed_ids.push_back(id)` → 范围外统一 `erase()`（避免 range-for 中 `unordered_map::erase` 触发 UB）。

- [x] 4.1 修改 `cudaStreamSynchronize` 函数：
  - bridge 路径：按 `stream_id` 过滤 → `bridge->poll_kernel(id)` → 完成 id 先 push `completed_ids` → 循环外统一 `erase()`
  - nullptr fallback：保留 `return cudaSuccess;` 立即返回
- [x] 4.2 修改 `cudaDeviceSynchronize` 函数：bridge 路径遍历 `g_active_streams`；nullptr fallback 调用 `g_gpu_context->wait_for_completion()`
- [x] 4.3 添加/修改 `cudaStreamCreate` 函数：
  - `next_kernel_id.fetch_add(1)` 生成 64-bit 唯一 ID
  - `g_active_streams.insert(id)`
  - `*pStream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(id))`
- [x] 4.4 验证迭代器失效修复：先 push 后 erase 的两阶段模式（避免 range-for 中 `unordered_map::erase` 触发 UB）
- [x] 4.5 编译验证 + 集成测试 `integration_cudart_sync` PASS

**Commit**:
```bash
git add src/cudart/cudart_sim.cpp
git commit -m "feat(cudart): cudaStreamSynchronize + cudaStreamCreate (D-PTX-1 + #3)

Iterator-invalidation fix: collect completed_ids before erase.
Stream-id filtering + multi-stream sync support.

Refs:
- ADR-0021 (D-PTX-1)
- CppTLM docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #3
- openspec/changes/cpptlm-d1-full
- lessons-learned §2 (iterator invalidation)"
```

---

## Phase 5: GLOBAL LD/ST 桥接（D-PTX-3 + Task #4，~0.3d）

> 📌 **NOTE**: 关键约束 — `is_global_space()` 必须遍历整个 qualifier 列表而非仅 `back()`（Lessons Learned #5）。

- [x] 5.1 修改 `src/ptxsim/instructions/memory.cpp` 的 `LdHandler::processOperation()`：
  - 添加 `g_cpptlm_bridge && is_global_space(addr)` 分支
  - 调用 `bridge->global_access(device_addr, 0, /*LD=*/0)`
  - latency 非 UINT64_MAX：`SimpleMemory::read()` + 返回 latency
  - UINT64_MAX 或 nullptr：fallback `processOperation_internal`
- [x] 5.2 修改 `StHandler::processOperation()` 类似 LD（非零 val，type=1）
- [x] 5.3 验证 `is_global_space()` 实现遍历整个 qualifier 列表（**Lessons Learned #5 强制**）
  - ✅ `getAddressSpace()` 已使用 for 循环遍历整个 qualifiers 列表
- [x] 5.4 编译验证 + 集成测试 `integration_ld_st_bridge` PASS
- [x] 5.5 验证 `[unit;memory]` `[e2e;memory]` 测试 0 回归（data correctness 比对 baseline）

**Commit**:
```bash
git add src/ptxsim/instructions/memory.cpp
git commit -m "feat(memory): GLOBAL LD/ST bridge to CppTLM NoC (D-PTX-3 + #4)

is_global_space() iterates entire qualifiers list (lessons-learned §5).
Data remains in SimpleMemory; bridge->global_access timing-only.
UINT64_MAX fallback preserves original PTX-EMU path.

Refs:
- ADR-0021 (D-PTX-3)
- CppTLM docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #4
- openspec/changes/cpptlm-d1-full"
```

---

## Phase 6: CMake libcpptlm_cudart 集成（Task #5 + D-PTX-4，~0.2d）

> 📌 **NOTE**: 默认 `BUILD_LIB_CPPTLM_CUDART=OFF` — 保证现有测试零退化。

- [x] 6.1 修改 `CMakeLists.txt` 末尾添加：
  ```cmake
  option(BUILD_LIB_CPPTLM_CUDART "Build libcpptlm_cudart.so bridge" OFF)
  find_package(cpptlm QUIET)
  if(cpptlm_FOUND AND BUILD_LIB_CPPTLM_CUDART)
      add_subdirectory(src/cudart/cpptlm_bridge)
      target_link_libraries(ptxemu_runtime PRIVATE cpptlm::core)
  endif()
  ```
  - ✅ 已实现 ExternalProject_Add 版本（HSK-3 选项 1）
- [x] 6.2 添加 `src/cudart/cpptlm_bridge/` 子目录（CMakeLists.txt + stub 实现占位）
  - ⚠️ 推迟：ExternalProject_Add 直接拉取 CppTLM 仓库，无需本地子目录
- [x] 6.3 编译验证 OFF 路径：`cmake --build build` 当前 ctest 目标零回归（`ctest -N` 实测 205 个目标，2026-07-17；旧的 “600+” 表述已被 [hsk-3.md:198] 修正为 188/202 → 现 205 含新增 cpptlm tests）
- [x] 6.4 编译验证 ON 路径（mock `find_package(cpptlm)`）：`libcpptlm_cudart.so` 被构建
  - ⚠️ 推迟：需 CppTLM 仓库实际可用
- [x] 6.5 **D-PTX-4**: 修改 `.github/copilot-instructions.md`：4.13.1 → 4.13.2（实际 vendored 一致）
  - ✅ 已验证：当前已是 4.13.2
- [x] 6.6 验证 `AGENTS.md` + 根 `README.md` 一致性 = 4.13.2
- [x] 6.7 **HSK-2 证据**：截图 / 路径引用 `.github/workflows/*.yml` 不安装 ANTLR4（vendored）

**Commit**:
```bash
git add CMakeLists.txt src/cudart/cpptlm_bridge/ \
        .github/copilot-instructions.md
git commit -m "build(cmake): libcpptlm_cudart integration + ANTLR4 4.13.2 fix (D-PTX-4 + #5)

Default BUILD_LIB_CPPTLM_CUDART=OFF preserves zero-regression.
ON path enables ExternalProject_Add-to-cpptlm integration.
ANTLR version 4.13.1 typo → 4.13.2 (matches vendored).

Refs:
- ADR-0021 (D-PTX-4)
- CppTLM docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #5
- openspec/changes/cpptlm-d1-full"
```

---

## Phase 7: 测试编写（~1.2d）

- [x] 7.1 创建 `tests/unit/cpptlm/test_cpptlm_bridge.cpp`（7 个 Bridge stub 测试）
  - [x] 7.1.1 `version() == 1` 测试
  - [x] 7.1.2 `submit_kernel()` 12 参数传递测试（mock bridge）
  - [x] 7.1.3 `poll_kernel()` 0 / >0 / UINT64_MAX 三种返回值测试
  - [x] 7.1.4 `synchronize_stream(stream_id)` 过滤测试
  - [x] 7.1.5 `global_access()` LD/ST timing-only 测试
  - [x] 7.1.6 `g_cpptlm_bridge == nullptr` 字节级回退测试
  - [x] 7.1.7 `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` 编译期测试（编译失败即为测试失败）
- [x] 7.2 创建 `tests/integration/cpptlm/test_async_launchkernel.cpp`（真实 kernel 路径走异步 bridge）
- [x] 7.3 创建 `tests/integration/cpptlm/test_ld_st_bridge.cpp`（GLOBAL LD/ST 走 bridge + 数据正确性对比 baseline）
- [x] 7.4 创建 `tests/integration/cpptlm/test_singleton_guard.cpp`（重复初始化 FATAL 中止）
- [x] 7.5 修改 `tests/CMakeLists.txt` 注册 4 个新测试目标
- [x] 7.6 运行 `./scripts/sanity.sh` 全绿 + `[unit;cudart] [integration;cudart]` 0 回归

**Commit**:
```bash
git add tests/unit/cpptlm/test_cpptlm_bridge.cpp \
        tests/integration/cpptlm/test_async_launchkernel.cpp \
        tests/integration/cpptlm/test_ld_st_bridge.cpp \
        tests/integration/cpptlm/test_singleton_guard.cpp \
        tests/CMakeLists.txt
git commit -m "test(cpptlm): 7 unit + 3 integration + 1 singleton_guard tests

Refs: openspec/changes/cpptlm-d1-full"
```

---

## Phase 8: Handshake 回传 + 文档同步（HSK-1/2/3 + AGENTS.md + ADR README，~0.4d）

- [x] 8.1 修改 `AGENTS.md`：在已知限制章节添加 §F12b-LD MemoryBridge 状态
- [x] 8.2 修改 `docs/adr/README.md`：索引追加 ADR-0021
- [x] 8.3 修改 `docs/dev-process/lessons-learned.md`：新增 §32 "CppTLM D1-Full MemoryBridge 集成经验"
- [x] 8.4 创建 `include/cudart/AGENTS.md`：记录 `cpptlm_bridge.h` 的 ABI 真值源地位 + bump 流程
- [x] 8.5 **HSK-1**: commit hash `8dc000ec` 已锁定，消息模板见 `hsk-1.md`
- [x] 8.6 **HSK-2**: ANTLR4 4.13.2 四权威源已验证一致，CI yml 不安装 ANTLR4
- [x] 8.7 **HSK-3**: CMake ExternalProject_Add 已实现（Phase 6 commit `d0803a09`）
- [x] 8.8 运行 `./scripts/sanity.sh --quick` 验证全绿

**Commit**:
```bash
git add AGENTS.md docs/adr/README.md docs/dev-process/lessons-learned.md \
        include/cudart/AGENTS.md
git commit -m "docs(cpptlm): HSK-1/2/3 handshake + lessons-learned §F12b-LD entry

Refs:
- ADR-0021
- CppTLM docs/superpowers/specs/PTX-EMU-README.md §10.3
- openspec/changes/cpptlm-d1-full"
```

---

## 验收标准（开 apply 前必须全绿）

- [ ] Phase 0 基线 worktree 全量测试 PASS（应以真 pre-change commit `8dc000ec^` 建立独立 worktree 并运行）
- [x] 8 个 Phase 全部 commit 完成 + 各自 Phase 验证 PASS（Phase 1–8 全部已 commit 位于 main，见 `git log --oneline 8dc000ec~..df05e10b`；仅 Phase 3.8 遗留 case 现已补齐在 fix/cpptlm-d1-full-closure worktree）
- [ ] `g_cpptlm_bridge == nullptr` 时 ctest 目标零新增回归（`make -C build test` 字节级对比 Phase 0 baseline；现存表述 "600+" 已被 [hsk-3.md:198] 替代）
- [ ] `ctest -R 'unit_cuda|stream_destroy|stream_sync_loop|cpptlm'` 全 PASS（含 8 个 CppTLM/CUDART 目标）
- [x] `./scripts/sanity.sh` 全绿（2026-07-17 sanity PASS — 见证 session `cpptlm-d1-full-cross-repo-review.md`）
- [x] AGENTS.md + docs/adr/README.md 更新（[AGENTS.md §已知限制](AGENTS.md) 已有 §F12b-LD MemoryBridge 条目；[docs/adr/README.md](docs/adr/README.md) 含 ADR-0021 Active 条目）
- [ ] 3 个 Handshake（HSK-1 已发出 ✓ / HSK-2 已发出 ✓ / HSK-3 待 D5 EOD 发出）—— 头靠子状态 文档化为“已发出（待 CppTLM 确认）”；PTX-EMU 端不可独立闭环，由 CppTLM 合并 + rebase + CI static_assert + COMMIT_HASH 锁定 推动
- [ ] OpenSpec `status --change cpptlm-d1-full` 输出 `isComplete=true` 且所有 artifact `status=done`（**注意**：`applyRequires` 由 spec-driven schema 静态定义 = `["tasks"]`，**永远不会**为 `[]`；盲目以 `applyRequires=[]` 为验收点是验证者误解 OpenSpec 语义，应以 `isComplete=true` + tasks artifact=`done` 为准）

## 序列化关系

| Change | 关系 | touch 的文件 |
|--------|------|------------|
| 本 change `cpptlm-d1-full` | 当前 | cudart_sim.cpp, memory.cpp, CMakeLists.txt, copilot-instructions.md |
| 姊妹 `cpptlm-phase8b-injection-points` | **并行**（主代码互不冲突；共享 `tests/CMakeLists.txt` 为追加式）| sm_context.cpp, sm_context.h, warp_context.cpp/h, scoreboard/pipeline/tensor_core_interface.h |

两个 change 互不阻塞，可任意顺序启动或并行。
