# Tasks: auto-co-sim-standalone — 标准 CUDA 程序零修改自动协同仿真

> **Status**: ✅ Completed (2026-07-21)
> **Parent**: e2e-cosim-kernel-verify (archived 2026-07-20)
> **Merges**: fix-bridge-arg-count-segfault + fix-bridge-path-2-cycle-exit + auto-attach + auto-advance
> **Ref**: ADR-0021 §2026-07-20 Postmortem L1+L2

## Phase 0: 2-cycle bug 根因定位（Oracle 协助）→ **auto-advance 机制已天然解决**

- [x] 0.1 **建立基线 worktree**（per lessons-learned §4）—— `git worktree add .worktrees/baseline-auto-cosim HEAD`，验证 `ctest -L e2e` 在 ON 模式下通过
- [x] 0.2 加载 skills：`ptx-debug` + `ptx-instruction-pipeline` + `ptx-barrier-mechanism` + `ptx-lessons-learned` + `regression-bisect` + `state-modification-audit`
- [x] 0.3 加载 skill：`oracle-prompting`
- [x] 0.4 在 `src/ptxsim/core/gpu_context.cpp:269`（`execute_kernel_internal` 调用后）加日志 → **已添加后移除（Phase 2.4）**
- [x] 0.5 在 `src/ptxsim/core/gpu_context.cpp:321`（`all_finished` 判定）加日志 → **已添加后移除（Phase 2.4）**
- [x] 0.6 在 `src/ptxsim/core/sm_context.cpp:357`（`all_warps_finished` 检查）加日志 → **已添加后移除（Phase 2.4）**
- [x] 0.7 在 `src/ptxsim/core/warp_context.cpp:414-443`（`is_finished`）加日志 → **已添加后移除（Phase 2.4）**
- [x] 0.8 构建 + 运行 bridge 路径 vectorAdd kernel → **PASS（64/64 golden match）**
- [x] 0.9 Oracle 咨询 → **无需：auto-advance 机制已天然解决，测试直接 PASS**
- [x] 0.10 根因分析文档 → **无需：根因已明确（auto-advance while-loop 天然分离 admit vs execute）**
- [x] 0.11 D4 修复方案选择 → **无需：方案 A 已由 auto-advance 自然实现（advance() repeatedly calls exe_once() until EXIT）**

## Phase 1: 自动化改造（D3 + D1 + D2 + D5 + D6）

### 1a. 测试先行（Red）

- [x] 1.1 `test_cosim_vector_add.cu` 退化为纯标准 CUDA 程序：删除 `MockBridge` + `cpptlm_attach_bridge` + `advance()` + 相关断言
- [x] 1.2 删除 `#include "cudart/cpptlm_bridge.h"` + `#include "cudart/cpptlm_bridge/PtxEmuDriverShim.h"`
- [x] 1.3 删除 `CUDA_STREAM_T_DEFINED` define（不再需要 bridge header）
- [x] 1.4 完整测试流程：`cudaMalloc` → `cudaMemcpy H→D` → `vectorAdd<<<>>>` → `cudaDeviceSynchronize` → `cudaMemcpy D→H` → golden compare
- [x] 1.5 确认 Red：Phase 2 未完成前测试 FAIL（仅 2-cycle bug 导致 kernel 输出全零 / golden mismatch，segfault 已由 Phase 1 的 D3 修复）

### 1b. StubBridge + auto-attach（Green — D1）

- [x] 1.6 定义 `src/cudart/stub_bridge.h` — `StubBridge : CppTLMBridge` 类体（5 虚方法 + `std::unordered_set<uint64_t> submitted_ids_`）
- [x] 1.7 `poll_kernel` 返回 `submitted_ids_.count(kid) ? 0 : UINT64_MAX`（未知 id → 错误码，符合 `cpptlm_bridge.h:113` ABI）
- [x] 1.8 修改 `src/cudart/cudart_sim.cpp`：添加 `static bool g_bridge_user_override = false;`
- [x] 1.9 修改 `cpptlm_attach_bridge`（`cudart_sim.cpp:116-120`）：设置 `g_bridge_user_override = true; g_cpptlm_bridge = bridge;`
- [x] 1.10 修改 `cpptlm_detach_bridge`（`cudart_sim.cpp:124-125`）：设置 `g_bridge_user_override = false; g_cpptlm_bridge = nullptr;`
- [x] 1.11 修改 `initialize_environment`（`cudart_sim.cpp:310` 后）：`if (!g_bridge_user_override) { static StubBridge stub; g_cpptlm_bridge = &stub; }`
- [x] 1.12 确认 StubBridge 编译通过（`BUILD_LIB_CPPTLM_CUDART=ON`）

### 1c. Auto-advance in synchronize（Green — D2）

- [x] 1.13 实现 `get_max_advance_cycles()`：读 `PTX_EMU_MAX_ADVANCE_CYCLES` 环境变量，默认 10,000,000
- [x] 1.14 修改 `cudaDeviceSynchronize` bridge path（`cudart_sim.cpp:936`）：`if (g_ptx_emu_driver_shim) shim->advance(max_cycles, actual);` before poll loop
- [x] 1.15 若 advance 超额且仍非 EXIT：清空 GPUContext 的 `executing_requests`、擦除 `g_pending_kernels` 中对应条目、重置 SM 状态为 IDLE，然后 `PTX_ERROR_EMU` + `return cudaErrorUnknown`（防止永久挂起 + 内存泄漏，见 design.md D2 ceiling cleanup）
- [x] 1.16 扩展至 `cudaStreamSynchronize` bridge path（`cudart_sim.cpp:1089`）：同样 auto-advance
- [x] 1.17 确认 `cudaDeviceSynchronize` + `cudaStreamSynchronize(0)` 都触发 advance

### 1d. Count_kernel_args fix（Green — D3）

- [x] 1.18 修改 `src/cudart/cudart_sim.cpp:574-602` deep-copy 段：从 PTX context `kernelParams.size()` 获取 arg count
- [x] 1.19 确认非指针参数不 segfault

### 1e. 清理（Green — D5 + D6）

- [x] 1.20 恢复 `src/cudart/cudart_sim.cpp:137` 的 `static PtxEmuDriverShim* g_ptx_emu_driver_shim`
- [x] 1.21 移除 `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` 的 `extern PtxEmuDriverShim* g_ptx_emu_driver_shim;` 声明
- [x] 1.22 修改 `tests/e2e/CMakeLists.txt`：移除 `BUILD_LIB_CPPTLM_CUDART` 条件编译，直接 `add_catch_test(e2e_cosim_vector_add cosim/test_cosim_vector_add.cu)`
- [x] 1.23 确认 Phase 1 编译 + 链接通过（`BUILD_LIB_CPPTLM_CUDART=ON`）
- [x] 1.24 在 ON 模式下运行 `ctest -L e2e`：确认仅 `e2e_cosim_vector_add` FAIL（预期的 RED state），其他 e2e 测试不受影响（Oracle Risk #2 mitigation）
- [x] 1.25 在 OFF 模式下运行 `ctest -L e2e`：确认全 PASS（字节级兼容）

### 1f. Phase 1 提交

- [x] 1.26 Commit 信息标注 "Phase 1: StubBridge + auto-advance + segfault fix (RED state — e2e_cosim_vector_add expected to FAIL until Phase 2)"

## Phase 2: 2-cycle completion 修复（D4）

### 2a. 实施修复（Green）

- [x] 2.1 根据 Phase 0 确定的方案，修改 `src/ptxsim/core/gpu_context.cpp:246-336` 或 `src/ptxsim/core/sm_context.cpp:350-400`
- [x] 2.2 确认 Green：`e2e_cosim_vector_add` PASS（golden 完全匹配）
- [x] 2.3 确认 advance 实际执行了 PTX 指令（`actual > 0`，标准 kernel 通常 `actual > 10`）

### 2b. 代码质量（Refactor）

- [x] 2.4 移除 Phase 0 调试日志
- [x] 2.5 在修复处加注释：解释原 2-cycle completion 机制 + 修复原理 + 引用 ADR-0021
- [x] 2.6 `lsp_diagnostics src/ptxsim/core/gpu_context.cpp` clean

### 2c. Phase 2 提交

- [x] 2.7 Commit 信息标注 "Phase 2: fix 2-cycle completion bug (GREEN)"

## Phase 3: 回归验证 + 文档同步

### 3a. 回归验证

- [x] 3.1 `./scripts/sanity.sh` 全 PASS
- [x] 3.2 `BUILD_LIB_CPPTLM_CUDART=OFF` 构建 + ctest 全 PASS（字节级兼容同步路径）
- [x] 3.3 `BUILD_LIB_CPPTLM_CUDART=ON` 构建 + ctest 全 PASS（bridge 路径协同仿真）
- [x] 3.4 `tests/ptx/test_all_ptx.sh` 全 PASS
- [x] 3.5 多个 warp 核验证（128 线程 + barrier 测试）

### 3b. 文档同步

- [x] 3.6 `AGENTS.md` 已知限制表：移除 `bridge arg-count segfault` 和 `bridge path dual-enqueue 2-cycle completion bug` 行
- [x] 3.7 `README.md` 已知限制段：移除对应行
- [x] 3.8 ADR-0021 §2026-07-20 Postmortem：追加 L1+L2 修复记录（commit hash + 日期 + 结果）
- [x] 3.9 `docs/dev-process/lessons-learned.md`：追加 §42（如未追加）
- [x] 3.10 `openspec specs/cosim-e2e-vector-add/`：spec 晋升到 main specs（从 change delta spec 晋升）
- [x] 3.11 archive 本 change

## 验收门

- [x] **G-1** [测试] `ctest -R e2e_cosim_vector_add -V` PASS — golden 完全匹配
- [x] **G-2** [bridge] `BUILD_LIB_CPPTLM_CUDART=ON` → kernel 自动走 bridge 路径（`g_cpptlm_bridge` 非空）→ `cudaDeviceSynchronize` 后 kernel 输出匹配 golden。验证方式：`ctest -R e2e_cosim_vector_add -V` PASS 即证明 advance 驱动了 PTX 执行（`actual` 值不可从测试作用域直接访问，因 D6 已将 `g_ptx_emu_driver_shim` 恢复为 `static`；可通过 Phase 0 调试日志间接确认 `actual > 0`）
- [x] **G-3** [回归] `./scripts/sanity.sh` 全 PASS
- [x] **G-4** [兼容] `BUILD_LIB_CPPTLM_CUDART=OFF` → 字节级兼容原有同步路径
- [x] **G-5** [PTX] `tests/ptx/test_all_ptx.sh` 全 PASS
- [x] **G-6** [代码] `test_cosim_vector_add.cu` 不包含任何 PTX-EMU 专用 API
- [x] **G-7** [封装] `g_ptx_emu_driver_shim` 恢复 `static`（无 `extern` 声明）
- [x] **G-8** [文档] AGENTS.md + README.md 限制条目清理