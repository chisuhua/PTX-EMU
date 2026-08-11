# ADR-0029: PTX-EMU Image Executor（in-memory Driver API + 反向依赖符号搬迁）

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
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
- **MANIFEST 单 kernel v1**: `ptxir_format.h:36-41` 的 `ManifestSection` 只有单 `kernel_name`，multi-kernel 仍 defer（ADR-0028 预留）。**v2 状态 (2026-08-11)**: 已由 ADR-0028 解除；详见 ADR-0028 §Decision 1。

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
| 多 launch 串行 | 同一 handle 的并发 launch 由 executor mutex 串行化（D6 SINGLE-GPU-INSTANCE 假设） |

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
- ADR-0028 已 ship（§Decision 1）；本 D4 限制已解除。

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

### D6: [SINGLE-GPU-INSTANCE ASSUMPTION] 显式标注

> **命名变更（2026-08-09 修订）**：原标签 [SINGLE-LAUNCH] 改名 [SINGLE-GPU-INSTANCE]，更准确反映假设边界。原标签描述的是 "进程内一个 GPU 仿真器"，而非 "一个 kernel launch"。两者的并发含义截然不同。

`ptx-lessons-learned.md` §10 模板强制：helper 的 single-instance 假设必须显式注释。本决策要求 `PtxEmuImageExecutor` 类头注释包含 7 个 [SINGLE-GPU-INSTANCE] 标记：

1. **`g_gpu_context` 全局唯一**: 同进程内所有 image 共享一个 `GPUContext`（一个模拟 GPU）
2. **`CudaDriver::instance()` 单例**: 共享全局内存池（所有 image 的 global/local/param memory 同池）
3. **`g_cpptlm_bridge` 单指针**: standalone 模式（device lib 持本地 nullptr 定义，不接 CppTLM）
4. **`PtxEmuImageExecutor` 单例** (`g_image_executor`): 进程内所有 image 共享
5. **executor mutex**: 同一 handle 的并发 launch 串行化（D3 A2 配套）
6. **`PtxInterpreter` 状态非重入**: `src/cudart/ptx_interpreter.cpp:19-36` 缓存 `ptxContext/kernelContext/kernelArgs/param_space` 为成员；每 launch 构造一个新 `PtxInterpreter`（D3 A2 通过 deserialize 路径自然实现）
7. **不接 SingletonGuard**: `__cudaRegisterFatBinary` 的 FATAL guard 不影响 image executor 路径（device lib 不调 `__cudaRegisterFatBinary`）

**对 TaskRunner 并发模型的隐含影响**（HAL 方案 D8 后已缓解，但本节仍记录 standalone 路径下行为）：

TaskRunner 的 `CmdProcessor` worker pool 采用 work-stealing 并发模型，多 worker 可同时调 `cuLaunchKernel`。**在原 D8 直链方案下**：
- 同一 handle 的并发 launch 由 executor mutex 串行（标记 #5）
- **跨 handle 的并发 launch 也会因 `g_gpu_context`（标记 #1）和 `CudaDriver::instance()`（标记 #2）的全局单例状态被间接互斥**，实际退化为串行执行
- 这意味着 TaskRunner 的并发优势在 PTX-EMU standalone 路径下被完全抹平
- 即使采用 D3 per-launch re-deserialize 策略，每个 worker 仍需获取 executor mutex，wall-clock cost ≈ N × (deserialize + execute)

**HAL 方案（D8 采纳）下缓解**：
- 所有 launch 经由 UsrLinuxEmu GpgpuDevice ioctl 派发表 → `HardwarePullerEmu::submitBatch()` → `GlobalScheduler::enqueue()`
- GlobalScheduler 已有 fence_id 异步跟踪 + 命令队列并发处理（Stage 4 ✅ ship 阶段）
- PTX-EMU 作为 HAL backend，仅在 `hal_user.cpp` 单入口被调用，并发由 UsrLinuxEmu 上层调度处理
- SINGLE-GPU-INSTANCE 假设在 HAL 边界外不直接暴露给 TaskRunner

**Falsification 测试**（standalone 路径适用，HAL 路径下不适用）：
- 构造两个 `PtxEmuImageExecutor` 实例必须显式失败（不是 silent corruption）
- 同 handle 并发 launch 必须由 mutex 串行（性能正确性均可验证）
- 跨 handle 并发 launch **预期**互斥（标记 #1/#2 全局状态），不应作为性能假设

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

### D8: CP 端集成约定 — **HAL 扩展方案**（UsrLinuxEmu ↔ PTX-EMU 跨仓契约）

> **方向变更（2026-08-09 修订）**：原 D8 提议 TaskRunner `libcuda_shim` 直链 `libptxemu_device.so`。该方案虽然技术上正确，但会绕过 UsrLinuxEmu HAL 边界（ADR-036 三区分架构硬约束：HAL 是 drv ↔ sim 唯一桥），导致两个并行 GPU 仿真器实例（PTX-EMU `GPUContext` + UsrLinuxEmu `HardwarePullerEmu`）无同步竞争。**经跨仓评审**，采用 HAL 扩展方案作为 v1 集成路径；原直接 link 提案保留为 **D8-Alt**（见末尾）作为技术对照。

#### D8.1 集成拓扑

```
TaskRunner UMD (cu_module.cpp / cu_launch.cpp)
    │ (现状 zero change)
    ▼ cuModuleLoadData / cuLaunchKernel / cuModuleUnload
TaskRunner CudaRuntimeApi
    │ (新增 IGpuDriver 方法, 无需 shim 改动)
    ▼ IGpuDriver::load_kernel_module / launch_kernel_module / unload_kernel_module
GpuDriverClient
    │ (新增 System C ioctl, 不改现有 38 个)
    ▼ GPU_IOCTL_LOAD_KERNEL_MODULE / LAUNCH_KERNEL_MODULE / UNLOAD_KERNEL_MODULE
UsrLinuxEmu GpgpuDevice (ioctl 派发表新增 3 行)
    │ (新增 HAL fn-ptr, append-only per ADR-023 §D4)
    ▼ hal_user.cpp → ptxemu_image_load / execute / unload (via dlsym)
libptxemu_device.so (PTX-EMU 作为 HAL implementation detail)
```

**关键不变量**：
- TaskRunner 仓**零改动**（`cu_module.cpp`/`cu_launch.cpp`/`CMakeLists.txt` 无需 link PTX-EMU）
- UsrLinuxEmu 仓改动：1 个新 ioctl + 1 个新 HAL fn-ptr + `hal_user.cpp` 实现
- PTX-EMU 仓改动：实现 `libptxemu_device.so` + `cpptlm_module.h`，**作为 HAL backend**（类似 `hal_user.cpp` vs `hal_mock.cpp` 二选一关系）
- 单一 GPU 状态来源：所有 kernel 状态、device memory、module handle 都在 UsrLinuxEmu 的 `HardwarePullerEmu` + `GpgpuDevice` 内；PTX-EMU 仅作为 ISA 执行后端
- PTX-EMU 与 TaskRunner 不直接通信；所有跨边界调用走 UsrLinuxEmu IOCTL 通道

#### D8.2 新增 IGpuDriver / GpuDriverClient 接口

```cpp
// include/shared/igpu_driver.hpp (TaskRunner 端)
class IGpuDriver {
  // ... 现有 47 方法 ...
  virtual int load_kernel_module(const void* image, size_t size,
                                 uint64_t* out_module_handle) = 0;
  virtual int launch_kernel_module(uint64_t module_handle,
                                   uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                   uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                   size_t shared_mem_bytes,
                                   void** kernel_args, size_t args_count) = 0;
  virtual int unload_kernel_module(uint64_t module_handle) = 0;
};
```

#### D8.3 新增 System C ioctl

> **canonical source**: UsrLinuxEmu [adr-076 §D1](../../../../../UsrLinuxEmu/docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md) `plugins/gpu_driver/shared/gpu_ioctl.h`(TaskRunner 通过 symlink 访问)。
> 本节代码片段**镜像** ADR-076 §D1 canonical 定义,自包含性保留;任何分歧必须先 amend ADR-076 后同步此处(per ADR-035 §R5.1 mirror 协议)。

```c
// plugins/gpu_driver/shared/gpu_ioctl.h (UsrLinuxEmu canonical source, TaskRunner 通过 symlink 访问)
enum {
  // ... 现有 38 个 GPU_IOCTL_* ...
  GPU_IOCTL_LOAD_KERNEL_MODULE   = _IOWR('G', 0x27, gpu_load_kernel_module_args),
  GPU_IOCTL_LAUNCH_KERNEL_MODULE = _IOWR('G', 0x28, gpu_launch_kernel_module_args),
  GPU_IOCTL_UNLOAD_KERNEL_MODULE = _IOWR('G', 0x29, gpu_unload_kernel_module_args),
};

struct gpu_load_kernel_module_args {
  uint64_t image_ptr;            // 用户态 image buffer (PTXIR 或 PTXIR-Embedded CUBIN)
  uint64_t image_size;
  uint64_t out_module_handle;    // 输出:image executor 返回的 handle
  char     kernel_name[256];     // 输出:image 内 kernel 名(ADR-0028 已解除 v1 限制)
};

struct gpu_launch_kernel_module_args {
  uint64_t module_handle;
  uint32_t grid_x, grid_y, grid_z;
  uint32_t block_x, block_y, block_z;
  uint64_t shared_mem_bytes;
  uint64_t args_ptr;             // 用户态 void** kernel_args
  uint64_t args_count;
  int32_t  launch_status;        // 输出:0 成功;-EINVAL/-EBUSY 等 cudaError_t 转换后 errno
};

struct gpu_unload_kernel_module_args {
  uint64_t module_handle;
  int32_t  unload_status;        // 输出:0 成功;-EBUSY (in-flight kernel);其他错误码
};
```

#### D8.4 新增 HAL fn-ptr + 实现

> **canonical source**: UsrLinuxEmu [adr-076 §D2](../../../../../UsrLinuxEmu/docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md) `plugins/gpu_driver/hal/gpu_hal.h`(HAL fn-ptr 65→68 append-only per ADR-023 §D4)。
> 本节代码片段**镜像** ADR-076 §D2 canonical 定义;`void *ctx` first param 与 ADR-023 HAL 约定一致。
> `kernel_module_load` 合并 PTX-EMU `ptxemu_image_load` + `ptxemu_image_kernel_name` 两次 ABI 调用为 1 次 HAL 调用(在 `hal_user.cpp` 内部分别调两次 cpptlm_module.h 函数)。

```c
// plugins/gpu_driver/hal/gpu_hal.h (append-only per ADR-023 §D4)
struct gpu_hal_ops {
  void *ctx;  // HAL ctx,per ADR-023 HAL 约定(现有 65 fn-ptrs 共享此字段)

  // ... 现有 65 个 fn-ptrs ...

  /* --- ADR-076 扩展(2026-08-09, PTX-EMU Image Executor HAL backend)--- */
  int (*kernel_module_load)(void *ctx,
                            const uint8_t* image_bytes, size_t image_size,
                            uint64_t* out_module_handle,
                            char* out_kernel_name, size_t kernel_name_buf_size);   // 新增 #66
  int (*kernel_module_execute)(void *ctx,
                               uint64_t module_handle,
                               uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                               uint32_t block_x, uint32_t block_y, uint32_t block_z,
                               size_t shared_mem_bytes,
                               void** kernel_args, size_t args_count);                   // 新增 #67
  int (*kernel_module_unload)(void *ctx, uint64_t module_handle);                   // 新增 #68
};

// plugins/gpu_driver/hal/hal_user.cpp (真机/仿真实现)
// 仅展示 kernel_module_load_via_ptxemu 核心逻辑;完整实现见 ADR-076 §D4
static int kernel_module_load_via_ptxemu(void *ctx,
                                          const uint8_t* image_bytes, size_t image_size,
                                          uint64_t* out_module_handle,
                                          char* out_kernel_name, size_t kernel_name_buf_size) {
  (void)ctx;
  // dlsym libptxemu_device.so (dlopen 一次,缓存句柄;三级 fallback: PTXEMU_ROOT -> /opt/ptxemu -> RTLD_DEFAULT)
  static uint64_t (*image_load)(const uint8_t*, size_t) =
      (uint64_t(*)(const uint8_t*, size_t))dlsym(RTLD_DEFAULT, "ptxemu_image_load");
  static int (*image_kernel_name)(uint64_t, char*, size_t) =
      (int(*)(uint64_t, char*, size_t))dlsym(RTLD_DEFAULT, "ptxemu_image_kernel_name");
  if (!image_load || !image_kernel_name) {
    // dlsym 三级 fallback 全失败
    return -ENOSYS;
  }
  // ABI version 检查(per ADR-0029 D1 governance)
  int (*module_version)(void) = (int(*)(void))dlsym(RTLD_DEFAULT, "ptxemu_module_version");
  if (module_version && module_version() != CPPTLM_MODULE_VERSION) return -EPROTO;

  uint64_t handle = image_load(image_bytes, image_size);
  if (handle == 0) return -EINVAL;
  *out_module_handle = handle;
  // kernel_name 读取失败回滚(handle 仍分配但 kernel name 不可用 -> caller 决定是否 unload)
  if (image_kernel_name(handle, out_kernel_name, kernel_name_buf_size) < 0) {
    // ADR-076 §D4 选择:handle 已分配但 kernel name 失败 -> 返回 -EINVAL 让 caller 调 unload
    return -EINVAL;
  }
  return 0;
}
```

#### D8.5 TaskRunner 集成约定（按 HAL 方案）

| TaskRunner 端 | 行为 |
|---|---|
| `cuModuleLoadData(module, image)` | 通过 `runtime()->load_kernel_module(image, ...)` → `IGpuDriver` → UsrLinuxEmu ioctl；**无需 link `libptxemu_device.so`** |
| `cuModuleGetFunction(func, mod, name)` | 维持现状：纯 handle 表 + 名字查询 |
| `cuLaunchKernel(f, grid, block, args, ...)` | 通过 `runtime()->launch_kernel_module(handle, ...)` → `IGpuDriver` → UsrLinuxEmu ioctl |
| `cuModuleUnload(m)` | 通过 `runtime()->unload_kernel_module(handle)`；in-flight kernel 时返回 `CUDA_ERROR_INVALID_HANDLE`（busy） |
| TaskRunner `CMakeLists.txt` | **零 PTX-EMU 依赖**（仅依赖 UsrLinuxEmu canonical System C header） |
| TaskRunner CI 构建 | 不需要 PTX-EMU build artifact；构建解耦 |

#### D8.6 优势（HAL 方案 vs 原 D8 直链方案）

| 维度 | D8-Alt 直链 | **D8 HAL 扩展（采纳）** |
|------|------------|------------------------|
| 符合 PTX-EMU 边界 | ✅ | ✅（PTX-EMU 仍独立） |
| 符合 UsrLinuxEmu 3 区分 | ❌ 破坏 ADR-036 | ✅ HAL 仍是唯一桥 |
| TaskRunner 改动 | 中（CMake + shim） | **零**（IGpuDriver 自动扩展） |
| GPU 状态来源 | 双（UsrLinuxEmu + PTX-EMU） | **单**（UsrLinuxEmu 唯一） |
| PTXIR 复用 System C ioctl 基础设施 | ❌ 需新建 | ✅ 复用 fence_id/va_space |
| 跨仓构建依赖 | 强（TaskRunner 需 link PTX-EMU） | 弱（PTX-EMU 是 UsrLinuxEmu HAL 的 impl detail） |
| TaskRunner 并发 launch 与 PTX-EMU SINGLE-GPU-INSTANCE 假设冲突 | 高（D8-Alt 多 worker 触发 mutex 序列化） | 低（HAL 单入口，UsrLinuxEmu 已有 fence/queue 调度） |
| 测试隔离 | 需 mock libptxemu_device.so | 需 mock libptxemu_device.so（同等） |

#### D8.7 实施分解（替换原 D5 Phase 2 — 概要，详细见各仓 ADR）

> **详细设计分散到跨仓 ADR**（per ADR-035 §R3 治理 + 用户审查可并行）：
>
> - **canonical source**：[UsrLinuxEmu adr-076-gpgpu-kernel-module-ioctl.md](../../../../../UsrLinuxEmu/docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md)
>   System C ioctl 编号（0x27/0x28/0x29）+ 结构体字段 + HAL fn-ptr #66/#67/#68 完整定义 + 跨仓 commit 顺序协议
> - **consumer-side 对偶**：[TaskRunner tadr-307-igpu-driver-kernel-module-extension.md](../../../../../UsrLinuxEmu/external/TaskRunner/docs/shared/adr/tadr-307-igpu-driver-kernel-module-extension.md)
>   IGpuDriver 扩展契约 + shim 调用链改动 + MockGpuDriver 更新 + e2e 测试要求
>
> 本 ADR 仅保留概要，详细实施归各仓 ADR owner。

**概要**：

| 仓 | 范围 | 工时估算 |
|----|------|---------|
| **PTX-EMU**（本仓） | 实现 `libptxemu_device.so` + `cpptlm_module.h`（D1, D3），**不变** | 已计入 ADR-0029 Phase 1 |
| **UsrLinuxEmu** | 新 Phase 5.x：`gpu_ioctl.h` 新增 3 个 ioctl + `GpgpuDevice::ioctl` 派发表新增 3 行 + `gpu_hal.h` 新增 3 个 fn-ptr + `hal_user.cpp` 新增 3 个 fn-ptr 实现（dlsym 加载 `libptxemu_device.so`）+ 单元测试 + e2e 测试（mock libptxemu_device.so） | ~200 行 |
| **TaskRunner** | 新 Phase x：`IGpuDriver` 新增 3 个纯虚方法（#48-#50）+ `GpuDriverClient` 新增 3 个 wrapper + `CudaRuntimeApi` 新增 3 个方法 + `cu_module.cpp::cuModuleLoadData` 替换 `CUDA_ERROR_NOT_IMPLEMENTED` + `cu_module.cpp::cuModuleUnload` 新增 busy 检查 + `cu_launch.cpp::cuLaunchKernel` 新增 image-executor fast-path + 测试：mock IGpuDriver，验证 IOCTL 调用契约 | ~150 行 |

**跨仓 commit 顺序**（canonical in [UsrLinuxEmu adr-076 §Migration](../../../../../UsrLinuxEmu/docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md#migration--实施步骤--跨仓-commit-顺序)，per ADR-035 §R5.1）：
```
i. PTX-EMU 仓 ship libptxemu_device.so + cpptlm_module.h (Phase 1) + tag v0.1.0+
ii. UsrLinuxEmu 仓 ship HAL extension (per adr-076) + bump external/TaskRunner submodule pointer
iii. TaskRunner 仓 ship IGpuDriver extension (per tadr-307) + push
iv. UsrLinuxEmu 仓 bump external/TaskRunner submodule pointer + final integration + adr-076 status 升 Accepted
```

#### D8.8 风险与缓解（HAL 方案特有）

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| HAL fn-ptr 增加破坏 append-only 原则 | 极低 | 中 | 按 ADR-023 §D4 编号续 #66/#67/#68；不修改现有 65 fn-ptrs（详见 adr-076 §D2）|
| ioctl 编号冲突（TaskRunner 未来需求） | 低 | 中 | 新增 0x27/0x28/0x29 预留（System C magic 'G' 8-bit 范围，与现有 0x01~0x26 连续）；0x2A~0x3F 给未来扩展；CI 加 ioctl 编号唯一性测试 |
| dlsym `libptxemu_device.so` 失败 | 中 | 中 | HAL 实现三级 fallback：`PTXEMU_ROOT` env → `/opt/ptxemu` 默认路径 → `cuobjdump` 路径（与 legacy front door 一致，详见 adr-076 §D4.1）|
| 跨仓版本错位（PTX-EMU ABI v2 升级 vs UsrLinuxEmu HAL v1） | 中 | 中 | HAL 实现内部 `static_assert(CPPTLM_MODULE_VERSION == ptxemu_module_version())`（详见 adr-076 §D4 + cpptlm_module.h 治理规则）|
| Phase 5.x 工时与原 D5 Phase 2 估算偏差 | 中 | 低 | HAL 方案对 PTX-EMU 仓 0 工时；UsrLinuxEmu 仓 ~200 行；TaskRunner 仓 ~150 行（详见 adr-076 §Migration + tadr-307 §Acceptance Items）|
| TaskRunner `cuModuleLoadData` 缺 image_size 参数 | 中 | 低 | tadr-307 §D4.1 调研：从 PTXIR header magic 前 4 字节推断 size 或保留 0；建议 task list 包含此调研 |

---

### D8-Alt: 直接 link `libptxemu_device.so` 方案（**原提案，记录备查**）

> 以下为 6 轮 Oracle 评审通过的原 D8 提案。本次修订保留作为技术对照记录。**该方案不作为 v1 集成路径**。

**约束**：
- TaskRunner `libcuda_shim` link `libptxemu_device.so`（**非** libcudart.so；避免命名冲突）
- `cuModuleLoadData(image)` → `ptxemu_image_load(image, size)` → 存 handle keyed by `CUmodule`
- `cuLaunchKernel(f, grid, block, args, ...)` → `func_to_module[f]` 反查 → `ptxemu_image_execute(handle, ...)`
- `cuModuleUnload(m)` → `ptxemu_image_unload(handle)`；in-flight kernel 时返回 `CUDA_ERROR_INVALID_HANDLE`（busy）
- KMD/CP 集成不在本 ADR 范围（属于 UsrLinuxEmu 端架构）

**不采用理由**（2026-08-09 修订）：
1. 绕过 UsrLinuxEmu HAL 边界（破坏 ADR-036 三区分架构）
2. 两个并行 GPU 仿真器实例（PTX-EMU `GPUContext` + UsrLinuxEmu `HardwarePullerEmu`），无同步机制
3. TaskRunner CMake 必须 link PTX-EMU，跨仓构建依赖
4. TaskRunner `CmdProcessor` 并发模型与 PTX-EMU D6 SINGLE-GPU-INSTANCE 假设直接冲突

---

## 后果

### 正面影响

1. **TBD 缺口填平**：`ptxir-toolchain-stack.md` §11 "ADR-XXXX (TBD)" 解析为 ADR-0029
2. **CP 端可集成**（HAL 方案，按 D8）：UsrLinuxEmu 通过新增 3 个 ioctl + 3 个 HAL fn-ptr 集成 PTX-EMU；TaskRunner 通过现有 System C ioctl 链路 **零改动** 获得 PTXIR execution 能力
3. **Mutation bug 修复**：D3（A2）通过 image bytes 重 deserialize 真正实现 image 不可变
4. **ABI 零变化**：`cpptlm_bridge.h` 不修改，CppTLM 端零 rebuild
5. **Phase 化解耦**：D5 三 Phase 各自独立可回退，per Lesson §3
6. **HAL 方案架构正确性**（2026-08-09 修订）：
   - PTX-EMU 作为 HAL backend（impl detail），符合 UsrLinuxEmu ADR-036 三区分架构
   - 单一 GPU 状态来源（UsrLinuxEmu 唯一），无 PTX-EMU / UsrLinuxEmu 双仿真器分裂
   - 跨仓构建依赖降低：TaskRunner 不再直接 link PTX-EMU
   - TaskRunner 并发模型与 PTX-EMU SINGLE-GPU-INSTANCE 假设的冲突被 HAL 边界隔离

### 负面影响

1. **3 Phase 工期**（PTX-EMU 仓）：估算 2-3 周（与 Oracle Round 1 估算一致）；HAL 方案额外引入 UsrLinuxEmu 仓 Phase 5.x 工作量（~200 行 + 1 个 ADR）
2. **PtxContextAdapter 等命名空间迁移**：从 `cudart` 命名空间到 `ptxemu`（语义更准确，但需要改名）
3. **新 ABI 表面维护**：`cpptlm_module.h` 需独立 governance（类似 ADR-0023 PTXIR 版本治理；治理规则同 `cpptlm_bridge.h:18-21`，2026-08-09 修订）
4. **HAL 方案的跨仓协调成本**（2026-08-09 修订）：需要 UsrLinuxEmu 仓 owner 评审 HAL fn-ptr 编号续 #66/#67/#68 + 3 个新 ioctl 编号 **0x27/0x28/0x29** 的可接受性；若被拒绝则退回 D8-Alt 直链方案
5. **HAL 方案的并行实现负担**：`hal_user.cpp` 需新增 `dlsym` 加载 `libptxemu_device.so` 的运行时解析逻辑（含 fallback 路径）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| ADR-0021 D-PTX-1:76 已确认强制 `g_cpptlm_bridge` 定义与 ABI 入口同 TU 约束 | **确定（不是风险,是约束）** | 高 | **Phase 0 Step 0 = amend ADR-0021**（per Checklist G:Active ADR 可 amend）；attach/detach/`g_bridge_user_override` 必须一起搬迁维持 same-TU 不变量（D2 行 2） |
| D3 重 deserialize 性能开销 > 10% 执行时间 | 中 | 中 | Phase 1 对真实 kernel（`bench/cute/cute_rmsnorm.ptx`）benchmark（**实测非估算**）作为 acceptance gate；超标触发 A1 fallback 决策点 |
| Phase 0 搬迁破坏默认 LD_PRELOAD 路径 | 低 | 高 | 5 gates (D7) 全部通过才允许 merge |
| `ptxir-toolchain-stack.md` 文档与本 ADR 长期漂移 | 中 | 中 | 归档本 ADR 时同步更新 v1.2 → v1.3（如有） |
| `PtxInterpreter` statefulness 引入并发 launch 之外的 corruption | 低 | 中 | D6 SINGLE-GPU-INSTANCE 假设 + 单元测试覆盖 |
| UsrLinuxEmu 端 C-API 调用习惯与 `cudaLaunchKernel` 不同 | 中 | 低 | D8 集成约定 + Phase 2 TaskRunner 端 e2e 测试 |

---

## 合规检查

> **Acceptance gate 关系（2026-08-09 修订）**：本 ADR 由 Proposed → Accepted 必须满足两个前置 gate：
>
> 1. **Phase 0 Step 0 gate**（**HARD gate**，未通过 → ADR 退回 Proposed 并重设计）：
>    ADR-0021 v1.1 amendment merged，解除 D-PTX-1:76 同 TU 约束
> 2. **D8 HAL 方案接受 gate**（**SOFT gate**，未通过 → ADR 仍可 Accepted，但 D8 退回 D8-Alt）：
>    UsrLinuxEmu 仓评审确认 HAL 扩展方案可实施（HAL 65→68 fn-ptrs append-only + 3 个新 ioctl 编号预留无冲突）
>
> 两个 gate 各自独立，但都必须在 ship 任意 Phase 前完成。如 Phase 0 Step 0 gate 与 ADR-0029 Accepted 同时段评审，**建议 Phase 0 Step 0 先通过**，避免 Accepted 后 D2 方案无法执行。

后续相关开发应检查：

- [x] **Phase 0 Step 0**（**HARD GATE，未通过不得进入 Phase 1**）：ADR-0021 v1.1 amendment merged（解除 D-PTX-1:76 同 TU 约束）
- [x] **Phase 0 Step 1**: 4 个全局符号（`g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override`）一起从 `cudart_sim.cpp` 搬到 `PtxEmuDriverShim.cpp`；`g_gpu_context` 从 `cudart_sim.cpp` 搬到 `ptx_interpreter.cpp`
- [x] **Phase 0 完成**: 5 gates (D7) 全部通过: `nm -D` diff = 空, SONAME/symlink 保持, g_cpptlm_bridge==nullptr 单元测试通过, logger→g_gpu_context 单元测试通过
- [x] **Phase 1 完成 (perf)**: cute_rmsnorm D3 deserialize cost 实测 0.183x（D7 gate 6 PASS, 81% margin under 1.10 threshold）
- [x] **Phase 1 完成**: `cpptlm_bridge.h` `git diff` 为空（governance 验证）
- [x] **Phase 1 完成**: `tests/unit/cudart/test_cpptlm_module.cpp` 覆盖 5 个 ABI 入口的 roundtrip + invalid handle + concurrent serialization
- [x] **Phase 1 完成**: `tests/unit/cudart/test_image_executor_mutation.cpp` 验证 D3 修复（同一 image 并发 launch 无 corruption）
- [x] **Phase 1 完成**: `PtxEmuImageExecutor` 类头包含 7 个 [SINGLE-GPU-INSTANCE] 标记（v1 SINGLE-GPU-INSTANCE assumption tracked in code comments）
- [ ] **Phase 2 完成**（HAL 方案，按 D8.7）：UsrLinuxEmu 仓新增 3 个 ioctl + 3 个 HAL fn-ptr + `hal_user.cpp` dlsym 实现 + e2e 测试通过；TaskRunner `libcuda_shim` **零 PTX-EMU link 依赖**
- [ ] **Phase 2 完成**（D8-Alt 备选，若 UsrLinuxEmu 仓拒绝 HAL 方案）：TaskRunner `libcuda_shim` link `libptxemu_device.so` 不冲突 + D6 SINGLE-GPU-INSTANCE 对 TaskRunner 并发影响文档化
- [x] **后续**: 任何 `cpptlm_module.h` 接口签名变更必须先 bump `CPPTLM_MODULE_VERSION`；治理规则同 `cpptlm_bridge.h:18-21`
- [ ] **后续**: multi-kernel 支持见 ADR-0028 §Decision 1（已 ship）

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-08-09 | 初始版本（A2+B1+C1 决策固化 + 3 Phase 分解 + 4 gates + 7 [SINGLE-LAUNCH] 标记） | PTX-EMU Architecture Team |
| 2026-08-09 | **F1 hardening（Oracle Round 3 review）**: D2 retitled "2 反向依赖符号搬迁 + CudaDriver 保留理由" + 4-symbol 一起搬 (attach/detach/override) + Phase 0 Step 0 = amend ADR-0021 hard gate + D7 升 5 gates (加 logger→`g_gpu_context`) + D3 perf gate 6 (cute_rmsnorm < 1.10 实测) + §10.5 item 18 重写 3 子项 + 全 doc 引用路径统一 (`src/cudart/ptx_interpreter.cpp`) | PTX-EMU Architecture Team |
| 2026-08-09 | **F1+1 hardening（Oracle Round 4 review）**: 数字漂移清扫 — 利益相关方 + Phase 0 目标 + 风险表 + 本行所有 "3 符号"/"4 gates" stale hits 统一为 "2 组 5 个全局符号" / "5 gates"；本行追加 | PTX-EMU Architecture Team |
| 2026-08-09 | **F1+2 hardening（Oracle Round 5 review）**: D2 preamble "2 个全局符号" 字面歧义修正为 "2 组共 5 个全局符号"（含 4 bridge + g_gpu_context）；本行追加 | PTX-EMU Architecture Team |
| 2026-08-09 | **F2 跨仓评审修订（PTX-EMU Architecture Team + UsrLinuxEmu/TaskRunner review）**: (a) D8 替换为 HAL 扩展方案，新增 D8.1-D8.8 子节 + ioctl 编号预留 **0x27/0x28/0x29** + HAL fn-ptr #66/#67/#68 + `hal_user.cpp` dlsym 设计；原 D8 直链方案保留为 **D8-Alt** 记录备查。(b) D6 标签 [SINGLE-LAUNCH] → [SINGLE-GPU-INSTANCE]，新增 TaskRunner 并发影响段落。(c) §合规 检查新增两个 Acceptance gate（Phase 0 Step 0 HARD gate + D8 HAL 方案 SOFT gate）。(d) §后果 + §负面影响 反映 HAL 方案后调整。(e) ADR-0028 multi-kernel manifest 升 BLOCKING DEPENDENCY。**(f) F3 跨仓文档契约化（2026-08-09）**：§D8.7 概要化 + 跨仓 ADR 引用分工；canonical source 落地到 UsrLinuxEmu [adr-076](../../../../../UsrLinuxEmu/docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md)；consumer-side 对偶到 TaskRunner [tadr-307](../../../../../UsrLinuxEmu/external/TaskRunner/docs/shared/adr/tadr-307-igpu-driver-kernel-module-extension.md)。ioctl 编号 39/40/41 → **0x27/0x28/0x29**（System C magic 'G' 8-bit 范围修正）| PTX-EMU Architecture Team |
| 2026-08-09 | **F4 D8.3/D8.4 canonical 同步（Oracle Round 7 review）**: D8.3 ioctl struct 镜像 ADR-076 §D1 canonical — `gpu_load_kernel_module_args` 新增 `char kernel_name[256]` 输出字段；`gpu_launch_kernel_module_args` 新增 `int32_t launch_status` 输出字段；`gpu_unload_kernel_module_args` 新增 `int32_t unload_status` 输出字段；修正 `_IOWR ('G'` trailing-space typo。D8.4 HAL fn-ptr 镜像 ADR-076 §D2 canonical — `struct gpu_hal_ops` 新增 `void *ctx` first field（per ADR-023 HAL 约定）；3 个 fn-ptr 全部加 `void *ctx` first param；`kernel_module_load` 新增 `out_kernel_name` + `kernel_name_buf_size` 参数；参数命名统一（`image_bytes`/`image_size`/`out_module_handle`/`kernel_args`/`module_handle`）。新增 `kernel_module_load_via_ptxemu` inline 实现展示合并 2 ABI call 为 1 HAL call + 版本检查 + rollback 语义。两条 canonical 标注均正确链接到 `../../../../../UsrLinuxEmu/docs/00_adr/adr-076-...`（修正 Round 7 G1 hyperlink drift）| PTX-EMU Architecture Team |

---

## 参考

- [ADR-0021 CppTLM D1-Full Integration](./ADR-0021-cpptlm-d1-full-integration.md) — D-PTX-1~6 决策 + `cpptlm_bridge.h` ABI 真值源；本 ADR 的 D-PTX-1 co-location 约束源
- [ADR-0024 PTXIR-Embedded CUBIN](./ADR-0024-ptxir-cubin-embed-extension.md) — `PTXIR_EMBED_MAGIC` 检测 + `PtxContextAdapter::fromEmbedded` 复用入口
- [ADR-0025 ptxir_build CLI](./ADR-0025-ptxir-build-cli.md) — build-time 工具，与本 ADR 正交
- [ADR-0026 PTXIR dispatch default auto](./ADR-0026-ptxir-default-mode-auto.md) — legacy front door 配置；本 ADR 的 in-memory 路径 PTXIR dispatch 始终 ON 独立于此
- [ADR-0027 ptx-nvcc wrapper](./ADR-0027-ptx-nvcc-wrapper.md) — build-time wrapper，Phase 2 TaskRunner 集成对照参考
- [ADR-0028 多 kernel manifest + runtime selection](./ADR-0028-multi-kernel-manifest.md) — 本 ADR 的 D4 已解除
- [docs/architecture/ptxir-toolchain-stack.md](../architecture/ptxir-toolchain-stack.md) v1.1 §4.2/§5/§11 — 本 ADR 是 §11 "TBD Driver API path" 的填平
- [include/cudart/cpptlm_bridge.h](../../include/cudart/cpptlm_bridge.h) — ABI v2 governance 真值源；本 ADR 不修改
- [include/cudart/AGENTS.md](../../include/cudart/AGENTS.md) — ABI governance anti-patterns（"不要静默 bump CPPTLMBRIDGE_VERSION"）
- [src/cudart/cudart_sim.cpp](../../src/cudart/cudart_sim.cpp) — `g_gpu_context:92` + `g_cpptlm_bridge:104` 现位置（D2 搬迁源）
- [src/cudart/cpptlm_bridge/PtxEmuDriverShim.h](../../src/cudart/cpptlm_bridge/PtxEmuDriverShim.h) — 7 方法 + vtable destroy (`cpptlm_bridge.h:205`)，无 launch（D1 决策依据）
- [src/cudart/ptx_interpreter.cpp](../../src/cudart/ptx_interpreter.cpp) — mutation bug @ :100-140 + statefulness @ :19-36（D3+D6 决策依据）
- [.opencode/skills/ptx-lessons-learned/SKILL.md](../../.opencode/skills/ptx-lessons-learned/SKILL.md) — §1 跨模块状态 mutation, §3 Phase commits, §4 基线 worktree, §7 pre-impl review, §10 single-instance assumption, §14 byte-identical fallback testing
