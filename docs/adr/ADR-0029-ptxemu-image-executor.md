# ADR-0029: PTX-EMU Image Executor（in-memory Driver API + 反向依赖符号搬迁）

| 属性 | 值 |
|------|-----|
| **状态** | Proposed |
| **日期** | 2026-08-09 |
| **关联任务** | TBD（待 propose 阶段确定 `openspec/changes/` 名） |
| **关联 PR** | TBD |
| **作者** | PTX-EMU Architecture Team |
| **审核人** | Oracle（架构 review，已完成两轮，见 §上下文）、Metis（实施前 review，实施启动时强制） |

---

## 上下文

### 问题背景

`docs/architecture/ptxir-toolchain-stack.md` v1.1 §11 明确声明:

> *ADR-XXXX (TBD) — in-memory Driver API path (`cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`/`cuModuleUnload`): 格式、registry lifetime、错误映射详见 §5、§7；ADR 待 propose*

本 ADR 就是填平这个 TBD 缺口。同时承担由两轮 Oracle review 揭示的 4 个架构风险（见 §触发事件），将其升级为正式架构决策。

### 触发事件

1. **2026-08-08**：`ptxir-toolchain-stack.md` v1.1 ship，明确列出 in-memory Driver API ADR 待 propose（§11）。
2. **2026-08-09 Oracle Round 1**：发现 `PtxEmuDriverShim.h:32-45` 7 个方法（`advance`/`inject_scoreboard`/`inject_pipeline`/`inject_tensor_core`/`is_kernel_complete`/`mark_complete`/`num_sms`）+ vtable 中 `destroy`（`cpptlm_bridge.h:205`），**无 launch 入口**，今天 launch 由 PTX-EMU 内部 `cudaLaunchKernel` 发起，新栈下 launch 由 CP（UsrLinuxEmu）外部发起，**方向错位**。
3. **2026-08-09 Oracle Round 1**：发现 3 个反向依赖符号必须搬迁:（*注:Oracle Round 3+F1 后确认实际搬迁 2 组 5 个全局符号,`CudaDriver` 保留理由见 D2 行 1;此处 3 指 Round 1 发现的 reverse-dep 现象数,不是搬迁数*）
   - `src/ptxsim/core/gpu_context.cpp:3,7` `#include "cudart/cuda_driver.h"`（ptxsim 依赖 CudaDriver）
   - `src/ptxsim/instructions/memory.cpp:8` `#include "cudart/cpptlm_bridge.h"` (D-PTX-3,引用 `g_cpptlm_bridge`)
   - `src/utils/logger.cpp:2` `#include "cudart/ptx_interpreter.h"` 访问 `g_gpu_context`，`logger.cpp` 编译进 `libptxsim`，但 `g_gpu_context` 定义在 `cudart_sim.cpp:92`
4. **2026-08-09 Oracle Round 2**：发现 `src/cudart/ptx_interpreter.cpp:100-140` launch 时会 **mutate stored KernelContext**:
   - S_SHARED 全局声明插入到 `kernelContext->kernelStatements`（guarded by `already_inserted`）
   - barrier 参与 mask 被 launch 时 blockDim **覆盖**
   - 顺序 launch 自我修复（每次重新覆盖），**并发 launch 同一 image → data race + corruption**
   - 这是 `ptx-lessons-learned.md` §1 跨模块状态 mutation 的具体实例

### 技术约束

- **`cpptlm_bridge.h` ABI v2 不变**: `include/cudart/cpptlm_bridge.h:18-21` 显式 governance rule "任何对本接口的修改必须同步 bump CPPTLMBRIDGE_VERSION + 通知 CppTLM rebase"；`include/cudart/AGENTS.md` 反模式: 不要静默 bump。
- **默认 LD_PRELOAD 路径字节级不变**: 任何迁移必须按 `ptx-lessons-learned.md` §14 "byte-identical fallback 必须由直接单元测试锁定"。
- **cpptlm_bridge.h 零外部依赖**: `include/cudart/AGENTS.md` 反模式 #1 "不要向 cpptlm_bridge.h 添加 CppTLM 头文件 include"；新 ABI 必须独立 header。
- **in-memory 路径 PTXIR dispatch 始终 ON**: 与 `PTXIR_MODE` 配置无关（已在 `ptxir-toolchain-stack.md` §4.2 定义）。
- **MANIFEST 单 kernel v1**: `ptxir_format.h:36-41` 的 `ManifestSection` 只有单 `kernel_name`，multi-kernel 仍 defer（ADR-0028 预留）。

### 利益相关方

- **CP 端（UsrLinuxEmu/TaskRunner）**：新增可调用的 in-memory 模块入口（"通过 cuModuleLoadData 加载二进制 → 通过 cuLaunchKernel 触发执行"语义）
- **PTX-EMU 内部**：ptxsim core 必须保持纯执行语义，**2 组反向依赖符号搬迁**（D2 行 2: 4 个 bridge 符号 `g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override`;行 3: `g_gpu_context`;`CudaDriver` 保留理由见 D2 行 1）
- **CppTLM 端**：零 ABI 影响（cpptlm_bridge.h v2 不变；新 ABI 走独立 header）
- **现有 LD_PRELOAD 用户**：零行为变化（**5 gates 锁定**，D7）
- **ADR 治理委员会**：amendment 检查（ADR-0024 v1.1 不动；本决策走新 ADR-0029，遵循 Checklist G）

---

## 决策驱动因素

1. **factor 1 — CP 端集成可调用性**：UsrLinuxEmu/TaskRunner 软件栈（UMD → ioctl → KMD → CP → device-side PTX 仿真）需要 PTX-EMU 暴露 in-memory 模块加载与 kernel 执行入口
2. **factor 2 — 字节级兼容**：默认 `__cudaRegisterFatBinary` 路径、cudart 符号表面、e2e 输出必须零变化
3. **factor 3 — ABI 不 bump**：新 ABI 必须独立 header（`cpptlm_module.h`），不污染 `cpptlm_bridge.h`
4. **factor 4 — Mutation bug 修复**：stored `KernelContext` 在 launch 时被 mutate（`src/cudart/ptx_interpreter.cpp:100-140`）必须解决，否则并发 launch 同一 image 会 corruption
5. **factor 5 — Load/launch 分离**：匹配真实 CUDA 架构（`cuModuleLoadData` 一次性解析 + `cuLaunchKernel` 廉价触发）
6. **factor 6 — Phase 化实施**：`ptx-lessons-learned.md` §3 "复杂迁移必须分 Phase commit"，每个 Phase 独立可回退

---

## 考虑的替代方案

### 方案 A: cpptlm_bridge.h vtable 加新方法（❌ 未采用）

**描述**：在 `CppTLMBridge` 5 虚方法之外加 `load_module` / `execute_kernel` / `unload_module` 等

**优点**：
- CppTLM 端通过现有 vtable 调用，接口一致

**缺点**：
- ❌ **必须 bump `CPPTLMBRIDGE_VERSION` v2 → v3**，触发 `cpptlm_bridge.h:18-21` governance 全套流程
- ❌ 必须通知 CppTLM rebase（HSK-1 重新发出），CI 双重 `static_assert` 失败
- ❌ 与 `include/cudart/AGENTS.md` 反模式 "不要静默 bump CPPTLMBRIDGE_VERSION" 直接冲突

**未采用理由**：ABI 治理成本远超收益，且语义边界（driver shim vs image executor）不应混入同一 vtable。

### 方案 B: cpptlm_bridge.h 加新 `extern "C"` 函数（❌ 未采用）

**描述**：在 `cpptlm_bridge.h` 末尾添加 `extern "C" cpptlm_load_module(...)` 等

**优点**：
- 不进 vtable，不触发 12 端点 `static_assert`
- 零 ABI 表面变化

**缺点**：
- ❌ `cpptlm_bridge.h:18-21` governance rule 严格解读: "任何对本接口的修改必须 bump"（包括添加新符号）
- ❌ CppTLM 通过 `ExternalProject_Add` 引用此 header，`git diff` 会触发其 CI rebuild
- ❌ 命名混淆（cpptlm_bridge.h 包含 cpptlm 弱符号 + cpptlm_module 的 extern "C"，未来维护混乱）

**未采用理由**：governance 风险与命名可读性都输。

### 方案 C: 新独立 header `cpptlm_module.h`（✅ 选中）

**描述**：新建 `include/cudart/cpptlm_module.h`，自带 `CPPTLM_MODULE_VERSION 1` 宏；声明 `ptxemu_image_load/execute/unload/is_complete/synchronize` 等 `extern "C"` 函数

**优点**：
- ✅ 零 ABI 风险（`cpptlm_bridge.h` 不修改，CppTLM 无需 rebase）
- ✅ 语义清晰（image executor 命名空间独立于 driver shim）
- ✅ governance 兼容（新 header 自带版本宏，类似 ADR-0023 PTXIR 版本治理）
- ✅ 与 `cpptlm_set_driver` 弱符号模式可并存（新 ABI 可选 weak fallback）
- ✅ 未来扩展（multi-kernel 等）走 `CPPTLM_MODULE_VERSION 2`

**缺点**：
- ⚠️ 多一个 header 需要维护 ABI 表面
- ⚠️ TaskRunner/UsrLinuxEmu 端需要 link `libptxemu_device.so` 解析新符号

**选择理由**：唯一同时满足 ABI 治理、语义清晰、可演进的方案。

---

## 决策内容

### D1: 新 ABI header — `cpptlm_module.h`（C1 命名 + 独立版本）

#### Header 文件结构

```cpp
// include/cudart/cpptlm_module.h (新文件)
#ifndef CPPTLM_MODULE_H
#define CPPTLM_MODULE_H

#include <cstddef>
#include <cstdint>

/// CPPTLM_MODULE ABI 版本号 — 编译期断言双端一致
/// 每次接口签名变更必须同步递增此值
/// 与 cpptlm_bridge.h 的 CPPTLMBRIDGE_VERSION 独立
#define CPPTLM_MODULE_VERSION 1

/// PTX-EMU image executor C-API
/// 设计原则:
///   - 零外部依赖（不 include 任何 PTX-EMU 内部类型）
///   - handle 是 opaque uint64_t（PTX-EMU 生成，跨 .so 安全）
///   - 所有方法同步阻塞调用；launch 完成后才返回
///   - 默认 LD_PRELOAD 路径完全不感知此 ABI

#ifdef __cplusplus
extern "C" {
#endif

/// 加载 image bytes 到 image memory,返回 handle
/// image_bytes 接受:
///   - standalone PTXIR (前 4 字节 = "PTXI")
///   - PTXIR-Embedded CUBIN (末尾含 PTXIR_EMBED_MAGIC, ADR-0024)
/// 返回 0 = 失败;非 0 = opaque handle
uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size);

/// 查询已加载 image 的 kernel 名（v1 限制: 每个 image 仅 1 个 kernel）
/// buf 至少 PtxEmuKernelNameMax (256) 字节
/// 返回 0 = 成功;-1 = handle 无效
int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size);

/// 卸载 image;in-flight kernel 引用时返回 -1 (busy),否则返回 0
int ptxemu_image_unload(uint64_t handle);

/// 同步执行 kernel;返回时 kernel 已完成
/// 返回 0 = 成功;非 0 = cudaError_t 错误码
int ptxemu_image_execute(uint64_t handle,
                         uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                         uint32_t block_x, uint32_t block_y, uint32_t block_z,
                         size_t shared_mem_bytes,
                         void** kernel_args, size_t args_count);

/// 查询版本（必须等于 CPPTLM_MODULE_VERSION）
int ptxemu_module_version(void);

#ifdef __cplusplus
}
#endif

#endif // CPPTLM_MODULE_H
```

#### 命名约定（C1）

- 命名空间前缀 `ptxemu_`（区别于 CppTLM 侧 `cpptlm_*`）
- 操作动词 `image_load` / `image_execute` / `image_unload` / `image_kernel_name`
- 强调 "image"（与真实 CUDA device code cache 语义对齐）而非 "module"（避免与 cudaModuleLoadData 的 module handle 混淆）

### D2: 2 反向依赖符号搬迁 + CudaDriver 保留理由

`ptx-lessons-learned.md` §1 强调"迁移必须行级 diff"。`Oracle Round 3` 实地核实确认:ptxsim→cudart 的反向依赖中,**`CudaDriver` 类不需要搬迁**(保留理由见行 1),**实际只需搬迁 2 组共 5 个全局符号**(D2 行 2: `g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override`;行 3: `g_gpu_context`),且搬迁必须遵守 ADR-0021 D-PTX-1 的已确认约束(见 BLOCKER 段)。

| 符号 | 现位置 | 搬迁目标 | 搬迁后访问路径 |
|---|---|---|---|
| `CudaDriver` 类 | `include/cudart/cuda_driver.h` + `src/cudart/cuda_driver.cpp` | **不搬迁**（保留理由:依赖仅为 `SimpleMemory` + `SimpleMemoryAllocator`,无 cudart 内部耦合;搬迁到 core/libptxemu_common 收益微薄但增加库数量） | ptxsim 保持 `#include "cudart/cuda_driver.h"`；device lib 与 cudart layer 共享同一份,链接时不会冲突 |
| `g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override`（**4 个符号必须一起搬**） | 全部位于 `cudart_sim.cpp`（`:104`, `:126-134`） | 全部搬迁到 `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp`（同 TU as `cpptlm_set_driver`,后者已在新 TU 定义） | 由 `cpptlm_bridge.h:147` 注释 "编译期定义于 src/cudart/cudart_sim.cpp" 改为新 TU；`cudart_sim.cpp:121-124` 的 "same TU as `g_cpptlm_bridge` per D-PTX-1" 不变量在搬迁后由 PtxEmuDriverShim.cpp 维持 |
| `g_gpu_context` | `cudart_sim.cpp:92` 定义 | 搬迁到 `src/cudart/ptx_interpreter.cpp`（同 TU as `PtxInterpreter`） | 由 `ptx_interpreter.h:19` extern 声明保持 |

**🔴 BLOCKER: ADR-0021 D-PTX-1 已确认约束（不是 SPECULATIVE）**。`Oracle Round 3` 实地核实 `ADR-0021:75-77`:

> "**约束**：- extern 声明必须在 `cpptlm_bridge.h` 内，**定义必须在单个 TU（`cudart_sim.cpp`）**"

这是 Active ADR 的硬约束,D2 计划搬迁 `g_cpptlm_bridge` 直接违反它。**Phase 0 Step 0 = amend ADR-0021**（per Checklist G:Active ADR 可 amend,需走标准 amendment 流程）— 解锁 `g_cpptlm_bridge` 可在 `PtxEmuDriverShim.cpp` 定义。amendment 通过前**不得搬迁**。D2 行 2 之所以要求 attach/detach/`g_bridge_user_override` 一起搬,是为了维持 `cudart_sim.cpp:121-124` "same TU as `g_cpptlm_bridge`" 的不变量(指针只能通过 ABI 入口 mutate)。

### D3: Image bytes 私有保存 + launch 时重 deserialize（A2 修复 mutation bug）

**问题**：Oracle Round 2 Q2C 发现 `src/cudart/ptx_interpreter.cpp:100-140` launch 时 mutate stored `KernelContext`，并发 launch 同一 image 会 corruption。

**修复方案**（A2）：

| 行为 | 实现 |
|---|---|
| Image bytes 私有保存 | `PtxEmuImageExecutor` 持有 `std::vector<uint8_t> image_bytes_`（来自 `ptxemu_image_load` 的 deep copy） |
| 不缓存 PtxContext | 不预存 `unique_ptr<PtxContext>`；每次 `ptxemu_image_execute` 重新调 `PTXIRLoader::deserializeForCubin(image_bytes_)` + `PtxContextAdapter::fromEmbedded()` |
| Deserialize 成本 | PTXIR 二进制解码 O(bytes)，不是 ANTLR parse；Oracle 建议对 `cute_rmsnorm` benchmark < 10% 执行时间 |
| 副作用 | Image 真正不可变（每次 launch 都是 fresh `PtxContext`），符合"image memory"心智模型 |
| 多 launch 串行 | 同一 handle 的并发 launch 由 executor mutex 串行化（D6 SINGLE-LAUNCH 假设） |

**为什么不选其他修复**：
- **A1（launch 时 deep-copy kernelStatements）**：O(N) per launch, N 大时不可忽略
- **A3（executor mutex 串行化）**：弱方案，stored state 仍会被 mutate,只是不并发

### D4: v1 单 kernel per image（B1）

**Scope 限制**：
- 每个 `ptxemu_image_load` 返回的 image handle **仅含 1 个 kernel**
- 调用方通过 `ptxemu_image_kernel_name(handle, buf, sz)` 查询该 kernel 名
- `ptxemu_image_execute(handle, ...)` 不接受 kernel 名（kernel 名已 baked into handle）

**Multi-kernel 延期**：
- `ptxir_format.h:36-41` 的 `ManifestSection` 当前为单 `kernel_name`
- multi-kernel 需要扩展 `ManifestSection` 为 `vector<kernel_entry>`，bump `PTXIR_VERSION`
- 预留 ADR-0028（multi-kernel manifest + runtime selection），本决策明确 defer
- `ptxir-toolchain-stack.md` §11 现有 ADR-0028 引用保留

### D5: 3-Phase 实施分解

按 `ptx-lessons-learned.md` §3 "复杂迁移必须分 Phase commit，每个 Phase 独立可回退"：

#### Phase 0: 反向依赖符号搬迁 + ADR-0021 D-PTX-1 验证

**目标**：**2 组共 5 个全局符号**搬迁完成（`g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override` → `PtxEmuDriverShim.cpp`；`g_gpu_context` → `ptx_interpreter.cpp`），默认 LD_PRELOAD 路径字节级不变

**改动文件**：
- `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` — 新增 4 个 bridge 符号定义（`g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override`，维持 `cudart_sim.cpp:121-124` same-TU 不变量）
- `src/cudart/ptx_interpreter.cpp` — 新增 `g_gpu_context` 定义
- 移除 `cudart_sim.cpp:92`（`g_gpu_context`）、`:104`（`g_cpptlm_bridge`）、`:126-134`（`cpptlm_attach_bridge` + `cpptlm_detach_bridge`）等定义（line-level diff）
- `CudaDriver` 类保持 `cudart/` 目录不动（D2 行 1 保留理由）

**验证**：
- **5 gates 全部通过**（D7）：`nm -D` diff 空, SONAME/symlink 保持, e2e stdout 字节 diff 空, g_cpptlm_bridge==nullptr 单元测试, logger→g_gpu_context 单元测试
- ADR-0021 D-PTX-1 amendment 已 merged（Phase 0 Step 0 hard gate）

**独立 commit,失败可 revert**

#### Phase 1: 新 ABI header + `libptxemu_device.so` + Image executor

**目标**：`ptxemu_image_load/execute/unload/kernel_name/version` 可调用，DL-isolated test 通过

**新增文件**：
- `include/cudart/cpptlm_module.h` — ABI 头（约 70 行）
- `src/cudart/cpptlm_module.cpp` — image executor 实现（约 350 行）
  - `PtxEmuImageExecutor` 类（持有 `std::vector<uint8_t> image_bytes_` + 每次 launch 临时 PtxContext + executor mutex）
  - `g_image_executor` 全局单例
  - 5 个 `extern "C"` 函数实现
- `src/cudart/CMakeLists.txt` — 新增 `add_library(ptxemu_device SHARED ...)` 链接 `ptxsim` `ptx_ir` `ptxir`
- `include/cudart/cpptlm_bridge.h` — **零修改**（governance 验证）

**测试**：
- `tests/unit/cudart/test_cpptlm_module.cpp` — roundtrip + handle invalidation + concurrent launch serialization
- `tests/unit/cudart/test_image_executor_mutation.cpp` — 验证 D3（A2 修复有效），并发 launch 同一 image 不 corruption

**Default path 影响**：零（纯加法新 .so + 新 header）

**独立 commit,失败可 revert**

#### Phase 2: TaskRunner 集成

**目标**：TaskRunner UMD `cuModuleLoadData`/`cuLaunchKernel` 走 PTX-EMU image executor

**改动**：
- `UsrLinuxEmu/external/TaskRunner/src/umd/libcuda_shim/cu_module.cpp` — `cuModuleLoadData` 调用 `ptxemu_image_load`（取代现有 `CUDA_ERROR_NOT_IMPLEMENTED`）
- `UsrLinuxEmu/external/TaskRunner/src/umd/libcuda_shim/cu_launch.cpp` — `cuLaunchKernel` 通过 `func_to_module[f]` 反查 image handle 调用 `ptxemu_image_execute`
- TaskRunner `libcuda_shim` link `libptxemu_device.so`

**独立 commit（跨 repo，PTX-EMU 端零影响）**,失败可 revert

### D6: [SINGLE-LAUNCH ASSUMPTION] 显式标注

`ptx-lessons-learned.md` §10 模板强制：helper 的 single-instance 假设必须显式注释。本决策要求 `PtxEmuImageExecutor` 类头注释包含 7 个 [SINGLE-LAUNCH / SINGLE-INSTANCE] 标记：

1. **`g_gpu_context` 全局唯一**: 同进程内所有 image 共享一个 `GPUContext`（一个模拟 GPU）
2. **`CudaDriver::instance()` 单例**: 共享全局内存池（所有 image 的 global/local/param memory 同池）
3. **`g_cpptlm_bridge` 单指针**: standalone 模式（device lib 持本地 nullptr 定义，不接 CppTLM）
4. **`PtxEmuImageExecutor` 单例** (`g_image_executor`): 进程内所有 image 共享
5. **executor mutex**: 同一 handle 的并发 launch 串行化（D3 A2 配套）
6. **`PtxInterpreter` 状态非重入**: `src/cudart/ptx_interpreter.cpp:19-36` 缓存 `ptxContext/kernelContext/kernelArgs/param_space` 为成员；每 launch 构造一个新 `PtxInterpreter`（D3 A2 通过 deserialize 路径自然实现）
7. **不接 SingletonGuard**: `__cudaRegisterFatBinary` 的 FATAL guard 不影响 image executor 路径（device lib 不调 `__cudaRegisterFatBinary`）

**Falsification 测试**: 构造两个 `PtxEmuImageExecutor` 实例必须显式失败（不是 silent corruption）；同 handle 并发 launch 必须由 mutex 串行（性能正确性正确性均可验证）

### D7: byte-identical fallback 5 gates（Lesson §14）

Phase 0 完成后必须 5 个 gate 全部通过,默认 LD_PRELOAD 路径才算"零行为变化":

1. **导出符号表面相同**：`nm -D --defined-only libcudart.so` 前后 diff 必须为空
2. **SONAME/symlink 保持**：`linux-so-version.txt` 的 SOVERSION 12 + POST_BUILD symlink 命令保留
3. **e2e 套件 stdout 字节级相同**：对同一组 e2e fixture 跑 monolithic 与 split build，stdout/stderr 字节 diff 为空
4. **`g_cpptlm_bridge == nullptr` standalone 单元测试通过**：`cpptlm_bridge.h:61` "nullptr = 独立模式，字节级兼容" 必须 test-lock
5. **logger→`g_gpu_context` 单元测试通过**：搬迁后 `logger.cpp:8` extern `get_gpu_clock_from_context` 仍能正确读取时钟（这是触发搬迁 #3 的动机,但 Phase 0 自身不能保证 logger 路径仍 OK — 必须 test-lock）

**D7 + Perf 扩展（Phase 1 验证）**:

6. **D3 deserialize cost 性能验收（实测非估算）**：`bench/cute/cute_rmsnorm.ptx` PTXIR 在 `ptxemu_image_load + 100 次 image_execute` 与"load + execute × 1 + 复用 PtxContext"的 wall time 比 < 1.10（10% 阈值）；超标触发 A1 fallback 决策点（launch 时 deep-copy `kernelStatements`）

**Lesson §4 强制**：Phase 0 前建立基线 worktree，monolithic 跑一次全量 ctest，存档为 oracle

### D8: TaskRunner 集成约定

**约束**：
- TaskRunner `libcuda_shim` link `libptxemu_device.so`（**非** libcudart.so；避免命名冲突）
- `cuModuleLoadData(image)` → `ptxemu_image_load(image, size)` → 存 handle keyed by `CUmodule`
- `cuLaunchKernel(f, grid, block, args, ...)` → `func_to_module[f]` 反查 → `ptxemu_image_execute(handle, ...)`
- `cuModuleUnload(m)` → `ptxemu_image_unload(handle)`；in-flight kernel 时返回 `CUDA_ERROR_INVALID_HANDLE`（busy）
- KMD/CP 集成不在本 ADR 范围（属于 UsrLinuxEmu 端架构）

---

## 后果

### 正面影响

1. **TBD 缺口填平**：`ptxir-toolchain-stack.md` §11 "ADR-XXXX (TBD)" 解析为 ADR-0029
2. **CP 端可集成**：UsrLinuxEmu/TaskRunner 可通过标准 C-API 加载 PTXIR 并执行 kernel
3. **Mutation bug 修复**：D3（A2）通过 image bytes 重 deserialize 真正实现 image 不可变
4. **ABI 零变化**：`cpptlm_bridge.h` 不修改，CppTLM 端零 rebuild
5. **Phase 化解耦**：D5 三 Phase 各自独立可回退，per Lesson §3

### 负面影响

1. **3 Phase 工期**：估算 2-3 周（与 Oracle Round 1 估算一致）
2. **PtxContextAdapter 等命名空间迁移**：从 `cudart` 命名空间到 `ptxemu`（语义更准确，但需要改名）
3. **新 ABI 表面维护**：`cpptlm_module.h` 需独立 governance（类似 ADR-0023 PTXIR 版本治理）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| ADR-0021 D-PTX-1:76 已确认强制 `g_cpptlm_bridge` 定义与 ABI 入口同 TU 约束 | **确定（不是风险,是约束）** | 高 | **Phase 0 Step 0 = amend ADR-0021**（per Checklist G:Active ADR 可 amend）；attach/detach/`g_bridge_user_override` 必须一起搬迁维持 same-TU 不变量（D2 行 2） |
| D3 重 deserialize 性能开销 > 10% 执行时间 | 中 | 中 | Phase 1 对真实 kernel（`bench/cute/cute_rmsnorm.ptx`）benchmark（**实测非估算**）作为 acceptance gate；超标触发 A1 fallback 决策点 |
| Phase 0 搬迁破坏默认 LD_PRELOAD 路径 | 低 | 高 | 5 gates (D7) 全部通过才允许 merge |
| `ptxir-toolchain-stack.md` 文档与本 ADR 长期漂移 | 中 | 中 | 归档本 ADR 时同步更新 v1.2 → v1.3（如有） |
| `PtxInterpreter` statefulness 引入并发 launch 之外的 corruption | 低 | 中 | D6 SINGLE-LAUNCH 假设 + 单元测试覆盖 |
| UsrLinuxEmu 端 C-API 调用习惯与 `cudaLaunchKernel` 不同 | 中 | 低 | D8 集成约定 + Phase 2 TaskRunner 端 e2e 测试 |

---

## 合规检查

后续相关开发应检查：

- [ ] **Phase 0 Step 0**: ADR-0021 amendment merged（解除 D-PTX-1:76 同 TU 约束）— 这是 **hard gate**，未通过不得搬迁任何符号
- [ ] **Phase 0 Step 1**: 4 个全局符号（`g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override`）一起从 `cudart_sim.cpp` 搬到 `PtxEmuDriverShim.cpp`；`g_gpu_context` 从 `cudart_sim.cpp` 搬到 `ptx_interpreter.cpp`
- [ ] **Phase 0 完成**: 5 gates (D7) 全部通过: `nm -D` diff = 空, SONAME/symlink 保持, e2e stdout 字节 diff = 空, g_cpptlm_bridge==nullptr 单元测试通过, logger→g_gpu_context 单元测试通过
- [ ] **Phase 1 完成 (perf)**: cute_rmsnorm D3 deserialize cost 实测 < 10%（D7 gate 6）
- [ ] **Phase 1 完成**: `cpptlm_bridge.h` `git diff` 为空（governance 验证）
- [ ] **Phase 1 完成**: `tests/unit/cudart/test_cpptlm_module.cpp` 覆盖 5 个 ABI 入口的 roundtrip + invalid handle + concurrent serialization
- [ ] **Phase 1 完成**: `tests/unit/cudart/test_image_executor_mutation.cpp` 验证 D3 修复（同一 image 并发 launch 无 corruption）
- [ ] **Phase 1 完成**: `PtxEmuImageExecutor` 类头包含 7 个 [SINGLE-LAUNCH / SINGLE-INSTANCE] 标记（D6）
- [ ] **Phase 2 完成**: TaskRunner `libcuda_shim` link `libptxemu_device.so` 不冲突
- [ ] **Phase 2 完成**: `cuModuleLoadData`/`cuLaunchKernel`/`cuModuleUnload` 端到端 e2e 测试通过
- [ ] **Phase 2 完成**: `cuModuleUnload` in-flight kernel 返回 `CUDA_ERROR_INVALID_HANDLE` (busy)
- [ ] **后续**: 任何 `cpptlm_module.h` 接口签名变更必须先 bump `CPPTLM_MODULE_VERSION`
- [ ] **后续**: multi-kernel 支持必须新 ADR（ADR-0028 预留，本 ADR 不承诺）

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-08-09 | 初始版本（A2+B1+C1 决策固化 + 3 Phase 分解 + 4 gates + 7 SINGLE-LAUNCH 标记） | PTX-EMU Architecture Team |
| 2026-08-09 | **F1 hardening（Oracle Round 3 review）**: D2 retitled "2 反向依赖符号搬迁 + CudaDriver 保留理由" + 4-symbol 一起搬 (attach/detach/override) + Phase 0 Step 0 = amend ADR-0021 hard gate + D7 升 5 gates (加 logger→`g_gpu_context`) + D3 perf gate 6 (cute_rmsnorm < 1.10 实测) + §10.5 item 18 重写 3 子项 + 全 doc 引用路径统一 (`src/cudart/ptx_interpreter.cpp`) | PTX-EMU Architecture Team |
| 2026-08-09 | **F1+1 hardening（Oracle Round 4 review）**: 数字漂移清扫 — 利益相关方 + Phase 0 目标 + 风险表 + 本行所有 "3 符号"/"4 gates" stale hits 统一为 "2 组 5 个全局符号" / "5 gates"；本行追加 | PTX-EMU Architecture Team |
| 2026-08-09 | **F1+2 hardening（Oracle Round 5 review）**: D2 preamble "2 个全局符号" 字面歧义修正为 "2 组共 5 个全局符号"（含 4 bridge + g_gpu_context）；本行追加 | PTX-EMU Architecture Team |

---

## 参考

- [ADR-0021 CppTLM D1-Full Integration](./ADR-0021-cpptlm-d1-full-integration.md) — D-PTX-1~6 决策 + `cpptlm_bridge.h` ABI 真值源；本 ADR 的 D-PTX-1 co-location 约束源
- [ADR-0024 PTXIR-Embedded CUBIN](./ADR-0024-ptxir-cubin-embed-extension.md) — `PTXIR_EMBED_MAGIC` 检测 + `PtxContextAdapter::fromEmbedded` 复用入口
- [ADR-0025 ptxir_build CLI](./ADR-0025-ptxir-build-cli.md) — build-time 工具，与本 ADR 正交
- [ADR-0026 PTXIR dispatch default auto](./ADR-0026-ptxir-default-mode-auto.md) — legacy front door 配置；本 ADR 的 in-memory 路径 PTXIR dispatch 始终 ON 独立于此
- [ADR-0027 ptx-nvcc wrapper](./ADR-0027-ptx-nvcc-wrapper.md) — build-time wrapper，Phase 2 TaskRunner 集成对照参考
- [ADR-0028 (预留) 多 kernel manifest + runtime selection](./README.md) — 本 ADR 的 D4 明确 defer
- [docs/architecture/ptxir-toolchain-stack.md](../architecture/ptxir-toolchain-stack.md) v1.1 §4.2/§5/§11 — 本 ADR 是 §11 "TBD Driver API path" 的填平
- [include/cudart/cpptlm_bridge.h](../../include/cudart/cpptlm_bridge.h) — ABI v2 governance 真值源；本 ADR 不修改
- [include/cudart/AGENTS.md](../../include/cudart/AGENTS.md) — ABI governance anti-patterns（"不要静默 bump CPPTLMBRIDGE_VERSION"）
- [src/cudart/cudart_sim.cpp](../../src/cudart/cudart_sim.cpp) — `g_gpu_context:92` + `g_cpptlm_bridge:104` 现位置（D2 搬迁源）
- [src/cudart/cpptlm_bridge/PtxEmuDriverShim.h](../../src/cudart/cpptlm_bridge/PtxEmuDriverShim.h) — 7 方法 + vtable destroy (`cpptlm_bridge.h:205`)，无 launch（D1 决策依据）
- [src/cudart/ptx_interpreter.cpp](../../src/cudart/ptx_interpreter.cpp) — mutation bug @ :100-140 + statefulness @ :19-36（D3+D6 决策依据）
- [.opencode/skills/ptx-lessons-learned/SKILL.md](../../.opencode/skills/ptx-lessons-learned/SKILL.md) — §1 跨模块状态 mutation, §3 Phase commits, §4 基线 worktree, §7 pre-impl review, §10 single-instance assumption, §14 byte-identical fallback testing
