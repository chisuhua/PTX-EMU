# PTXIR Toolchain Stack Architecture

> **版本**: 1.3
> **日期**: 2026-08-09
> **状态**: Proposed
> **作者**: PTX-EMU Architecture Team
> **关联 ADRs**: [ADR-0024](../adr/ADR-0024-ptxir-cubin-embed-extension.md), [ADR-0025](../adr/ADR-0025-ptxir-build-cli.md), [ADR-0026](../adr/ADR-0026-ptxir-default-mode-auto.md), [ADR-0027](../adr/ADR-0027-ptx-nvcc-wrapper.md), [**ADR-0029**](../adr/ADR-0029-ptxemu-image-executor.md) — in-memory Driver API 与 image executor (HAL 方案 D8 修订)

**v1.3 修订摘要**（2026-08-09 跨仓评审）：
- §2 Components 新增 **CP 端跨仓集成节点表**（UsrLinuxEmu GpgpuDevice + hal_user.cpp + gpu_hal.h + TaskRunner libcuda_shim + IGpuDriver），明示 HAL 方案 D8 集成路径
- §11 Related ADRs：**ADR-0028 升级为 BLOCKING DEPENDENCY**（从 "预留占位" 升级），新增下游 ADR 须遵守的契约
- §12 Future work：新增 "UsrLinuxEmu ↔ PTX-EMU HAL extension" 高优先级条目；ADR-0029 实施进度跟踪更新为 HAL 方案 Phase 分解

**v1.2 修订摘要**:
- 新增 §5.4 *Image Bytes Ownership & Per-launch Re-deserialization* — 填平 ADR-0029 的 D3 决策（image bytes 私有保存 + launch 重 deserialize 修复 `ptx_interpreter.cpp:100-140` mutation bug）
- §2 Components 新增 `libptxemu_device.so` 与 `cpptlm_module.h` 行
- §10 Testing 新增 §10.5 *Image executor acceptance items*
- §11 Related ADRs 移除 "ADR-XXXX (TBD)" 占位，加入 ADR-0029
- §12 Future work 移除 "TBD Driver API" 占位，加入 3-Phase 实施进度跟踪

---

## 1. Goals

- 文档定义两个并列的 runtime front door：legacy linked executable registration 通过 `__cudaRegisterFatBinary` 处理链接后的 executable；in-memory module loading 通过 `cuModuleLoadData` 接收调用方提供的 image buffer。
- legacy front door 继续支持用户用 `ptx-nvcc` 编译 `.cu`，再直接运行生成的 executable。嵌入 PTXIR 的 executable 自动加载 PTX-EMU `libcudart.so` 并尝试 PTXIR dispatch。
- in-memory front door 以 `cuModuleLoadData`、`cuModuleGetFunction`、`cuLaunchKernel`、`cuModuleUnload` 组成显式的 module lifecycle。它复用现有 PTXIR loader 和 `PtxInterpreter` / `GPUContext` 执行后端，不改变本轮文档之外的产品实现。
- 两条 front door 的 image discovery 和 gate 语义保持分离：legacy registration 继续受 `PTXIR_MODE` precedence 约束；in-memory module loading 的 PTXIR dispatch 在本架构范围内始终启用，不读取 `/proc/self/exe`，也不回退到 `cuobjdump`。
- 未嵌入 PTXIR 的普通 binary 在 legacy front door 未发现 footer 时继续走 cuobjdump 路径。auto 模式增加一次 executable-tail probe，不承诺 byte-level unchanged。
- 工具链保持可分步调用，便于调试和验收。本节和 §2 是架构文档描述，不代表本轮已经实现新的 Driver API 行为。

## 2. Components

| Tool / Lib | 角色 | 路径 | 关联 ADR |
|---|---|---|---|
| `ptx-nvcc` | nvcc wrapper，编译、链接、提取、序列化、嵌入 | `tools/ptx-nvcc` | ADR-0027 |
| `nvcc` | `.cu` → `.o`，再由 `.o` → executable | CUDA toolkit | — |
| `cuobjdump` | executable → `.ptx` | CUDA toolkit | — |
| `ptxir_build` | `.ptx` → `.ptxir` | `tools/ptxir_build` | ADR-0025 |
| `ptxir_embed` | executable + `.ptxir` → embedded executable | `tools/ptxir_embed` | ADR-0024 |
| `ptxir_extract` | embedded executable → cubin / PTXIR | `tools/ptxir_extract` | ADR-0024 |
| `libcudart.so.12` | PTX-EMU runtime，由 `DT_RUNPATH` 加载 | `build/lib/` | ADR-0024 |
| `config::isPTXIRModeEnabled` | legacy runtime 模式决策 | `src/cudart/ptxir_config.cpp` | ADR-0026 |
| `__cudaRegisterFatBinary` | legacy linked executable registration front door，发现 executable-tail PTXIR 或走 cuobjdump | `src/cudart/cudart_sim.cpp` | ADR-0024, ADR-0026, ADR-0027 |
| `cuModuleLoadData` | in-memory module loading front door，接收 NULL-terminated PTX 或 standalone PTXIR image | `src/cudart/cudart_sim.cpp` | ADR-0024, ADR-0029 |
| `cuModuleGetFunction` | 从已加载 module 按名称取得 function record | `src/cudart/cudart_sim.cpp` | ADR-0029 |
| `cuLaunchKernel` | 使用 function record 提交 in-memory module kernel | `src/cudart/cudart_sim.cpp` | ADR-0029 |
| `cuModuleUnload` | 释放 module record，并使其 function handles 失效 | `src/cudart/cudart_sim.cpp` | ADR-0029 |
| `PTXIRLoader` | 从 byte buffer 检测、提取、反序列化 PTXIR | `include/cudart/ptxir_loader.h` | ADR-0024 |
| `libptxemu_device.so` | device-side executor 库（CP 端可直接 dlopen / link，或作为 UsrLinuxEmu HAL backend） | `build/lib/` | ADR-0029 |
| `cpptlm_module.h` | image executor C-API ABI header（与 `cpptlm_bridge.h` 独立） | `include/cudart/cpptlm_module.h` | ADR-0029 |

> **CP 端跨仓集成节点**（2026-08-09 增补，HAL 方案 D8）：
>
> | 节点 | 角色 | 位置（外部仓） | 关联 |
> |---|---|---|---|
> | `UsrLinuxEmu GpgpuDevice` | 模拟 GPU 驱动入口；通过 System C ioctl 接收 TaskRunner 请求；HAL 65→68 fn-ptrs append-only 集成 `kernel_module_load/execute/unload` | `UsrLinuxEmu/plugins/gpu_driver/drv/gpgpu_device.cpp` | UsrLinuxEmu AGENTS.md ADR-036 |
> | `UsrLinuxEmu hal_user.cpp` | HAL 真机/仿真实现；新增 dlsym `libptxemu_device.so` 的 `ptxemu_image_*` 函数 | `UsrLinuxEmu/plugins/gpu_driver/hal/hal_user.cpp` | UsrLinuxEmu AGENTS.md ADR-023 |
> | `UsrLinuxEmu gpu_hal.h` | HAL 接口契约；新增 fn-ptr #66/#67/#68（kernel_module_load/execute/unload） | `UsrLinuxEmu/plugins/gpu_driver/hal/gpu_hal.h` | UsrLinuxEmu ADR-023 §D4 |
> | `TaskRunner libcuda_shim` | CUDA driver LD_PRELOAD shim；`cuModuleLoadData`/`cuLaunchKernel`/`cuModuleUnload` 通过 `IGpuDriver` → `GpuDriverClient` → System C ioctl 间接调 PTX-EMU | `TaskRunner/src/umd/libcuda_shim/cu_module.cpp` + `cu_launch.cpp` | TaskRunner ADR-035 |
> | `TaskRunner IGpuDriver` | 抽象 GPU 驱动接口；新增 3 个纯虚方法 `load_kernel_module` / `launch_kernel_module` / `unload_kernel_module` | `TaskRunner/include/shared/igpu_driver.hpp` | TaskRunner TADR-301 |
>
> **重要约束**（HAL 方案 D8.1）：**TaskRunner 仓零 PTX-EMU 链接依赖**。所有 PTX-EMU 调用经 UsrLinuxEmu HAL 边界封装。这维持 UsrLinuxEmu 三区分架构（ADR-036）的硬约束——HAL 是 drv ↔ sim 唯一桥。

## 3. Build-time data flow

`ptx-nvcc` 按 ADR-0027 执行以下顺序：

```text
源文件和 nvcc 参数
  → nvcc compile-only，生成临时 object
  → nvcc link，生成带 libcudart 和 DT_RUNPATH 的 executable
  → cuobjdump --ptx executable，生成临时 PTX
  → 自动检测一个 kernel 名，或使用显式 --kernel-name
  → ptxir_build --in PTX --kernel-name K --out PTXIR
  → ptxir_embed --in-exe executable --in-ptxir PTXIR --kernel-name K --out executable
  → 删除临时 object、PTX、PTXIR 和临时目录
```

wrapper 为 object、PTX、PTXIR 创建明确的临时目录和文件名，不使用 shell wildcard。任何步骤失败都进入清理路径。`--no-embed` 在 compile-only 和 link 成功后停止，不执行后续三步。`--ptxemu-root` 只影响新构建 binary 的 `DT_RUNPATH`、库路径和工具路径，不改变已有 binary。

v1 只接受单 kernel。自动检测不到 `.entry`、检测到多个 entry，或显式 kernel 名不存在时，wrapper 以 kernel 数据错误退出，不生成嵌入结果。ADR-0028 文件目前不存在，多 kernel manifest 和 runtime selection 语义延期，待语义明确后再提出。

## 4. Runtime data flow

运行时有两个并列的 front door。两者在 image discovery、PTXIR gate 和 module 生命周期上保持独立，但最终都汇合到同一个执行后端：`cudaLaunchKernel` → `SMContext::exe_once` → `WarpContext`。

### 4.1 Legacy linked executable registration

```text
./myapp
  → dynamic loader 按 DT_RUNPATH 加载 PTX-EMU libcudart.so.12
  → __cudaRegisterFatBinary
  → 按 PTXIR_MODE precedence 决定是否尝试 PTXIR
       → off：直接走 cuobjdump
       → auto/on：通过 /proc/self/exe 检查 executable tail
            → 缺少 footer：fallback 到 cuobjdump
            → footer 存在且 PTXIR/manifest 有效：反序列化并 dispatch PTXIR
            → malformed embedded PTXIR 或 manifest mismatch：报告错误，不当作缺少 footer
  → cudaLaunchKernel → SMContext::exe_once → WarpContext
```

legacy front door 继续以 `__cudaRegisterFatBinary` 注册链接后的 executable。其 tail probe 读取 `/proc/self/exe`，`PTXIR_MODE` 的 precedence 语义与 §6 一致。`PTXIR_MODE=off` 跳过 executable-tail PTXIR 检测并使用 `cuobjdump`，`auto/on` 则先尝试检测；未发现 footer 时正常 fallback 到 `cuobjdump`。malformed embedded PTXIR 或 manifest mismatch 仍按嵌入数据错误处理，而不是静默 fallback。

### 4.2 In-memory module loading

```text
应用代码
  → cuModuleLoadData(module, image)
       → 复制调用方提供的 image
       → eager parse：立即检测并解析 standalone PTXIR 或 NULL-terminated PTX
       → 建立 module record 和深拷贝后的 kernel/function 数据
  → cuModuleGetFunction(function, module, name)
  → cuLaunchKernel(function, ...)
       → cudaLaunchKernel → SMContext::exe_once → WarpContext
  → cuModuleUnload(module)
       → 释放 module record，使关联 function handles 失效
```

in-memory front door 由应用代码通过 `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cuModuleUnload` 的 Driver API 调用序列进入。`cuModuleLoadData` 对调用方提供的 image 执行 eager parse，并深拷贝解析结果及后续执行所需的 module/function 数据，因此调用返回后不依赖 caller-owned pointer 继续作为不透明 handle 存活。

该路径的 PTXIR dispatch 始终 ON，与 `PTXIR_MODE` 无关，也不读取 `/proc/self/exe`，不使用 executable-tail probe，且不回退到 `cuobjdump`。本架构范围内，输入限定为 standalone PTXIR image 或 NULL-terminated PTX，不宣称 cubin、fatbin 或 Tile IR 由该 front door 支持。解析或 module/function 选择失败时，该路径报告自身的加载或调用错误，不转入 legacy executable discovery。

两条 front door 都只在各自的加载和选择阶段处理输入差异，kernel 提交后统一进入 `cudaLaunchKernel`、`SMContext::exe_once` 和 `WarpContext` 后端。本文描述的是运行时架构边界；in-memory 路径的具体 Driver C-API（`ptxemu_image_load` / `ptxemu_image_execute` / `ptxemu_image_unload` 等）、2 反向依赖符号搬迁（CudaDriver 保留理由见 ADR-0029 D2 行 1）、image bytes 重 deserialize 修复 `src/cudart/ptx_interpreter.cpp:100-140` mutation bug、3-Phase 实施分解与 5 byte-identical gates 详见 [ADR-0029](../adr/ADR-0029-ptxemu-image-executor.md)。

## 5. In-memory module loading

本节定义 `cuModuleLoadData` 入口的内存模块加载边界。该入口只承诺 PTX 文本和 standalone PTXIR，其他 image 格式明确拒绝，不把未实现的格式支持写成产品能力。

### 5.1 Image classification matrix

| Image format | Detected by | Status | Behavior |
|---|---|---|---|
| PTX text (NULL-terminated) | leading bytes classified as PTX text | SUPPORTED | Eagerly parse the null-terminated PTX text and materialize a fresh module record |
| standalone PTXIR | PTXIR magic / leading bytes | SUPPORTED | Route the image to the PTXIR deserializer and materialize a fresh module record |
| executable-tail PTXIR suffix | executable-tail magic / suffix probe | REJECTED | Defer to the legacy front door; do not treat an executable tail as an in-memory module image |
| NVIDIA cubin | NVIDIA cubin magic / leading bytes | NOT SUPPORTED | Return `CUDA_ERROR_INVALID_IMAGE` |
| NVIDIA fatbin | NVIDIA fatbin magic / leading bytes | NOT SUPPORTED | Return `CUDA_ERROR_INVALID_IMAGE` |
| Tile IR | Tile IR magic / leading bytes | NOT SUPPORTED | Return `CUDA_ERROR_INVALID_IMAGE` |

Image classification first peeks at the magic or other leading bytes to determine the format, then routes a supported format to the corresponding parser. If no classifier matches, the loader rejects the image as invalid and returns `CUDA_ERROR_INVALID_IMAGE`. The in-memory path does not read `/proc/self/exe`, invoke `cuobjdump`, or use an executable-tail fallback.

### 5.2 ModuleRecord and FunctionRecord ownership

`ModuleRecord` 是由 PTX-EMU registry 拥有的不透明 module handle。它持有已解析的 IR statements、kernel metadata、深拷贝后的 image bytes，以及 eager-parse 状态。registry 负责 module record 的索引和销毁，调用方只能通过 NVIDIA Driver API contract names 使用 `CUmodule` handle。

`FunctionRecord` 是由所属 `ModuleRecord` 拥有的不透明 function handle。它持有 kernel symbol reference 和 argument descriptors，生命周期绑定到父 `ModuleRecord`。`cuModuleGetFunction` 只从仍然有效的 module record 中选择 function record，不为 function handle 建立独立的隐式所有权。

### 5.3 Lifetime and materialization

`cuModuleLoadData` 在返回前复制 image bytes，并完成 eager parse。解析结果、kernel metadata 和 function 所需的参数描述都在加载时物化并深拷贝，因此 caller-owned pointers 在调用返回后不会作为 handles 继续存活。registry 只在 `cuModuleUnload` 时释放 module handle。父 module 卸载后，关联的 function handles 立即失效，不存在 implicit lifetime extension。

in-memory front door 与现有 legacy interpreter 的关系是共享执行后端，而不是共享已注册对象。两条 front door 都把 kernel 提交到同一个 `PtxInterpreter` / `GPUContext` execution backend。in-memory 路径为每个加载的 image 产生一个 fresh `PtxContext`（**v1.2 修订：fresh 的实际粒度是 per-launch 而非 per-image，详见 §5.4**），不会修改 legacy front door 已注册的 `PtxContext`。

### 5.4 Image bytes ownership & per-launch re-deserialization（ADR-0029 D3）

本节说明 in-memory 路径的 image bytes 持有与 launch 时执行模型。这是 [ADR-0029](../adr/ADR-0029-ptxemu-image-executor.md) D3 决策的架构级固化，对应 mutation bug 的根本修复。

**问题**: `src/cudart/ptx_interpreter.cpp:100-140` 在 launch 时会 mutate 已存储的 `KernelContext`：
- S_SHARED 全局声明插入到 `kernelContext->kernelStatements`（guarded by `already_inserted`）
- barrier 参与 mask 被 launch 时 blockDim 覆盖

这意味着"image memory = 不可变字节流"的心智模型在现有实现下并不成立——顺序 launch 自我修复（每次重新覆盖），但并发 launch 同一 image 会 data race + corruption。

**v1.2 决策**: Image executor (`PtxEmuImageExecutor`) 持有 `std::vector<uint8_t> image_bytes_`（来自 `ptxemu_image_load` 的 deep copy），**不缓存 `unique_ptr<PtxContext>`**。每次 `ptxemu_image_execute` 重新调 `PTXIRLoader::deserializeForCubin(image_bytes_)` + `PtxContextAdapter::fromEmbedded()` 构造一个 fresh `PtxContext`，launch 完成后该 `PtxContext` 析构。

| 维度 | v1.2 行为 |
|---|---|
| Image bytes 持有 | executor 私有 `std::vector<uint8_t>`（不变） |
| PtxContext 持有 | 每次 launch 构造 fresh，launch 后析构 |
| Deserialize 成本 | PTXIR 二进制解码 O(bytes)（不是 ANTLR parse）；目标 < 10% 执行时间 |
| 并发 launch | executor mutex 串行化同 handle 的 launch（[SINGLE-LAUNCH ASSUMPTION]，ADR-0029 D6） |
| Unload 语义 | 引用计数 0 时释放 image bytes；in-flight kernel 时 unload 返回 busy |

**为什么不缓存 PtxContext**: 缓存方案要求 launch 时 deep-copy `kernelStatements`（O(N) per launch，N 大时不可忽略）或仅 executor mutex（弱方案，stored state 仍被 mutate）。重 deserialize 是唯一同时满足"image 真正不可变"与"每次 launch fresh state"的方案。

**为什么 image bytes 由 executor 持有（而非映射调用方 pointer）**: 本架构 §5.3 的 eager parse + 深拷贝契约要求 caller-owned pointer 在调用返回后不作为 handle 存活；executor 必须独立持有 bytes 才能在 unload 之前完整保留可执行性。

## 6. Configuration precedence

legacy front door 的配置优先级如下：

```text
env PTXIR_MODE > INI [ptxir] mode > default auto
```

| env `PTXIR_MODE` | INI `[ptxir] mode` | `isPTXIRModeEnabled()` |
|---|---|---|
| `auto` | (any) | `true` |
| `off` | (any) | `false` |
| (unset) | `on` / `auto` | `true` |
| (unset) | `off` | `false` |
| (unset) | 未设置或无段 | `true` |

上表只描述 legacy linked executable registration 的模式决策。完整 precedence matrix、默认 auto 的兼容性说明和 fallback 契约见 [ADR-0026](../adr/ADR-0026-ptxir-default-mode-auto.md)，本文不另行扩展该决策表。

in-memory `cuModuleLoadData` 的 PTXIR dispatch 始终 ON，独立于 `PTXIR_MODE` 和 INI 配置。也就是说，legacy front door 可以通过 `PTXIR_MODE=off` 保留 `cuobjdump` 语义，但该设置不会关闭 in-memory front door 的 standalone PTXIR 识别与加载，两条路径的配置边界不能互相推导。

ADR-0026 是 ADR-0024 v1.1 的 amendment，形成 v1.2。其承诺是 dispatch semantics 和最终执行路径兼容，并在 unset 时增加一次 executable-tail probe，不是 byte-level unchanged 或 zero behavior change。

## 7. Error mapping

以下错误映射只适用于 in-memory module loading path，即 `cuModuleLoadData`、`cuModuleGetFunction` 和 `cuLaunchKernel` 组成的调用链：

| Condition | CUDA Error | Notes |
|---|---|---|
| Unsupported image format (cubin/fatbin/Tile IR) | `CUDA_ERROR_INVALID_IMAGE` | Reject at image classifier; see §5.1 |
| Malformed PTX text | `CUDA_ERROR_INVALID_PTX` | Eager parse failure in `cuModuleLoadData` |
| Malformed standalone PTXIR | `CUDA_ERROR_INVALID_IMAGE` | PTXIR deserializer failure |
| Unknown `CUmodule` handle passed to `cuModuleGetFunction` / `cuLaunchKernel` | `CUDA_ERROR_INVALID_HANDLE` | Handle not in registry |
| Unknown `CUfunction` handle passed to `cuLaunchKernel` | `CUDA_ERROR_INVALID_HANDLE` | Function record not valid |
| Missing kernel symbol in module | `CUDA_ERROR_NOT_FOUND` | `cuModuleGetFunction` lookup miss |
| Module unloaded while function handles still held | `CUDA_ERROR_INVALID_HANDLE` | Stale function handle detection |

legacy front door 有自己的错误语义，例如 `cuobjdump` 失败和 embedded PTXIR manifest mismatch，相关说明见 §4.1 和 ADR-0026。本节映射仅适用于 in-memory module loading path，不改变 legacy executable registration 的错误处理。

## 8. CLI/error contract

`ptxir_build` 使用统一退出码：0 成功，1 用法或参数错误，2 PTX/kernel 数据错误，3 I/O、内部或工具失败。wrapper 传播 `nvcc`、`cuobjdump`、`ptxir_build` 和 `ptxir_embed` 的失败码，无法分类的子工具失败映射为 3。

缺少 footer 是正常 fallback。malformed embedded PTXIR 和 manifest mismatch 是数据或选择错误，不得静默 fallback。单 kernel 失败不会留下临时文件，也不会把未完成的 PTXIR 当作成功结果。

## 9. Constraints and compatibility

| 约束 | 说明 |
|---|---|
| 单 kernel per binary | `ManifestSection.kernel_name` 为单值，wrapper 只接受一个 kernel |
| Linux / POSIX | runtime executable-tail detection 依赖 `/proc/self/exe` |
| `DT_RUNPATH` | 现行 `-Wl,-rpath` 链接行为统一以 `DT_RUNPATH` 术语描述 |
| nvcc passthrough | 除 wrapper 自有选项外，参数按原顺序透传 |
| Python 3 | wrapper 使用 `#!/usr/bin/env python3` |

| binary 类型 | 模式 | runtime 行为 |
|---|---|---|
| 嵌入 PTXIR | default auto | 检测 footer，成功后 PTXIR dispatch |
| 嵌入 PTXIR | env off | cuobjdump 路径 |
| 普通 binary | default auto | 未发现 footer，fallback 到 cuobjdump |
| 普通 binary | env off | cuobjdump 路径 |

in-memory module path 具有独立的 image compatibility matrix，不受 legacy `PTXIR_MODE` precedence 影响：

| In-memory image | PTXIR_MODE | Status | runtime 行为 |
|---|---|---|---|
| In-memory module + standalone PTXIR | 任意 | SUPPORTED | 始终 ON，独立于 `PTXIR_MODE`，执行 standalone PTXIR 解析并建立 module record |
| In-memory module + NULL-terminated PTX | 任意 | SUPPORTED | 始终 ON，解析 NULL-terminated PTX 并建立 module record |
| In-memory module + cubin / fatbin / Tile IR | 任意 | NOT SUPPORTED | 返回 `CUDA_ERROR_INVALID_IMAGE`（见 §5.1） |
| In-memory module + executable-tail PTXIR suffix | 任意 | REJECTED | 拒绝作为 in-memory image，defer to legacy front door |

## 10. Testing and acceptance

验收至少覆盖：

| # | Acceptance item | 验收内容 |
|---|---|---|
| 1 | `ptxir_build` roundtrip | `ptxir_build` PTXIR roundtrip，以及 exit code 0、1、2、3。 |
| 2 | ADR-0026 precedence matrix | ADR-0026 的五行 precedence matrix。 |
| 3 | wrapper passthrough | wrapper 的 nvcc passthrough、明确临时文件和 object cleanup。 |
| 4 | 输入与工具错误 | no-entry、multi-entry、显式 kernel 不存在和 tool failure propagation。 |
| 5 | `DT_RUNPATH` | `DT_RUNPATH` 注入和 `--ptxemu-root` 对新 binary 的作用。 |
| 6 | malformed fallback | malformed footer、malformed embedded PTXIR、manifest mismatch，以及缺少 footer fallback。 |
| 7 | end-to-end | end-to-end compile, link, embed, and run。 |
| 8 | In-memory module happy path, NULL-terminated PTX | `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cuModuleUnload`。 |
| 9 | In-memory module happy path, standalone PTXIR | standalone PTXIR image 上执行 `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cuModuleUnload`。 |
| 10 | Multi-kernel selection | image with multiple entry symbols，`cuModuleGetFunction` 按名称选择正确的 entry。<br>**注**：v1 显式 single-kernel（§3 + ADR-0029 D4），本 item **推迟到 v2 / ADR-0028** 范围；v1.2 标记保留而非删除以备扩展。 |
| 11 | Error mapping verification | cubin → `CUDA_ERROR_INVALID_IMAGE`；malformed PTX → `CUDA_ERROR_INVALID_PTX`；unknown module handle → `CUDA_ERROR_INVALID_HANDLE`；missing symbol → `CUDA_ERROR_NOT_FOUND`；unloaded module's stale function handle → `CUDA_ERROR_INVALID_HANDLE`。 |
| 12 | `PTXIR_MODE=off` independence | `PTXIR_MODE=off` 不会 disable in-memory module loading。 |

### 10.5 Image executor acceptance items（ADR-0029 Phase 1 验证）

[ADR-0029](../adr/ADR-0029-ptxemu-image-executor.md) 的 image executor 在 Phase 1 完成后必须通过以下 acceptance items：

| # | Acceptance item | 验收内容 |
|---|---|---|
| 13 | ABI 头不动 | `git diff include/cudart/cpptlm_bridge.h` 为空；`CPPTLMBRIDGE_VERSION` 保持 2 |
| 14 | 新 ABI header 可读 | `include/cudart/cpptlm_module.h` 存在，`CPPTLM_MODULE_VERSION 1`，5 个 `extern "C"` 函数声明齐全 |
| 15 | Roundtrip | `ptxemu_image_load(PTXIR bytes)` → `ptxemu_image_kernel_name(handle, buf)` → `ptxemu_image_execute(handle, grid, block, args)` → `ptxemu_image_unload(handle)` 全部成功 |
| 16 | Invalid handle rejection | 对 0 handle / 已 unload handle 调 `execute` 返回非 0 错误码 |
| 17 | **Concurrent launch serialization**（D3 + D6 配套） | 同一 handle 的 N 个并发 launch 由 executor mutex 串行执行，全部成功，stored image bytes 不被 mutate（**仅覆盖正确性，性能见 item 19**） |
| 18 | Mutation bug 修复验证（D3 修复，3 子项） | (a) 同 bytes 两次 deserialize → byte-identical `kernelStatements`；(b) 顺序 launch N 次（不同 blockDim）→ 输出确定，无累积；(c) image bytes hash 经 N 次 launch 不变（**注：post-D3 无 stored `kernelStatements`，本项验证 deserialize 的纯净性 + image bytes 的不可变性**） |
| 19 | **D3 deserialize cost 性能验收**（实测非估算） | `bench/cute/cute_rmsnorm.ptx` PTXIR 在 "load + 100 × execute（每 launch 重 deserialize）" 与 "load + execute × 1 + 复用 PtxContext" 的 wall time 比 < 1.10（10% 阈值，**实测**）；超标触发 A1 fallback 决策点（launch 时 deep-copy `kernelStatements`） |
| 20 | DL-isolated test | `dlopen libptxemu_device.so` 独立测试（无 libcudart.so 加载），所有 image executor API 可调用 |
| 21 | Default LD_PRELOAD path 零影响 | Phase 1 完成后 **5 gates**（ADR-0029 D7）全部通过：`nm -D` diff 空, SONAME/symlink 保持, e2e stdout 字节 diff 空, g_cpptlm_bridge==nullptr 单元测试通过, logger→g_gpu_context 单元测试通过 |

**Phase 2 TaskRunner 集成 acceptance items**（在 UsrLinuxEmu 端实施）:

| # | Acceptance item | 验收内容 |
|---|---|---|
| 22 | link 不冲突 | TaskRunner `libcuda_shim` link `libptxemu_device.so` 不与 libcudart.so 命名冲突 |
| 23 | cuModuleLoadData → cuLaunchKernel e2e | TaskRunner UMD `cuModuleLoadData(image)` → `cuLaunchKernel(f, ...)` → kernel 实际执行 → `cuModuleUnload(m)` 端到端通过 |
| 24 | in-flight unload 返回 busy | kernel 在飞时调 `cuModuleUnload` 返回 `CUDA_ERROR_INVALID_HANDLE`（busy） |

## 11. Related ADRs

- [ADR-0024](../adr/ADR-0024-ptxir-cubin-embed-extension.md) — PTXIR-Embedded CUBIN 格式与 runtime dispatch，格式 source of truth
- [ADR-0025](../adr/ADR-0025-ptxir-build-cli.md) — `ptxir_build` CLI
- [ADR-0026](../adr/ADR-0026-ptxir-default-mode-auto.md) — default auto 与 fallback/error 契约
- [ADR-0027](../adr/ADR-0027-ptx-nvcc-wrapper.md) — wrapper 编排和 passthrough 契约
- [**ADR-0029**](../adr/ADR-0029-ptxemu-image-executor.md) — in-memory Driver API（image executor）、2 反向依赖符号搬迁（CudaDriver 保留）、image bytes 重 deserialize 修复 mutation bug、3-Phase 实施分解与 5 byte-identical gates、CP 端 HAL 扩展集成方案（2026-08-09 修订）
- **ADR-0028（**[BLOCKING DEPENDENCY]**）**：多 kernel manifest + runtime selection 设计
  > **状态升级（2026-08-09）**：从 "预留占位" 升级为 **BLOCKING DEPENDENCY**。原因：ADR-0025 §v1 单 kernel 限制、ADR-0027 §v1 单 kernel 限制、ADR-0029 D4 v1 单 kernel per image 限制，三者的根因都是 `ptxir_format.h:36-41` 的 `ManifestSection` 只有单 `kernel_name` 字段。在 ADR-0028 未 ship 前，所有相关 ADR 都受 v1 单 kernel 限制拖累。
  >
  > **下游 ADR 必须遵守的契约**：
  > 1. ADR-0025/0027/0029 §v1 限制段落须明示 "等待 ADR-0028 解除"
  > 2. ADR-0028 ship 时必须 bump `PTXIR_VERSION`（继承 ADR-0023 Extend-Only 原则）
  > 3. backward-compat 策略：旧 v1 单 kernel binary 在 ADR-0028 后运行时仍可加载（manifest 格式向后可读）

## 12. Future work

| 主题 | 说明 | 优先级 |
|---|---|---|
| **ADR-0029 实施进度跟踪** | Phase 0（**Step 0 = amend ADR-0021** + 2 反向依赖符号搬迁 + 5 byte-identical gates）→ Phase 1（`libptxemu_device.so` + `cpptlm_module.h` + image executor + D3 perf 验证）→ Phase 2（**HAL 方案 D8** UsrLinuxEmu 仓 Phase 5.x + TaskRunner 仓 IGpuDriver 扩展）。每个 Phase 独立 commit，失败可 revert。<br>**进度 SSOT**：实施 detail 与 tasks checklist 由 `openspec/changes/<TBD>/tasks.md` 维护（per OpenSpec lifecycle, Lesson §6），本文 §12 仅作 orientation 不作追踪 | 高 |
| **UsrLinuxEmu ↔ PTX-EMU HAL extension**（HAL 方案 D8 跨仓协作） | UsrLinuxEmu 仓新增 3 个 ioctl（GPU_IOCTL_LOAD_KERNEL_MODULE/LAUNCH_KERNEL_MODULE/UNLOAD_KERNEL_MODULE 编号 39/40/41）+ 3 个 HAL fn-ptr（#66/#67/#68 kernel_module_load/execute/unload）+ `hal_user.cpp` 新增 dlsym `libptxemu_device.so` 的 `ptxemu_image_*` 实现 + TaskRunner 仓 `IGpuDriver` 新增 3 个纯虚方法 + 跨仓 commit 顺序 per ADR-035 R5.1。详见 ADR-0029 §D8.1-D8.8 | 高 |
| **ADR-0028 多 kernel manifest**（**[BLOCKING DEPENDENCY]**，详见 §11） | 多个 `.entry` binary 支持；`ManifestSection` 扩展为 `vector<kernel_entry>`；bump `PTXIR_VERSION` per ADR-0023 Extend-Only 原则。解除 ADR-0025/0027/0029 的 v1 单 kernel 限制 | 高 |
| `$ORIGIN` 相对路径 | 减少安装路径限制（ADR-0027 §v1 限制 DT_RUNPATH 绝对路径缓解未根本解决） | 中 |
| CMake wrapper 集成 | 提供 `ptxemu_add_executable()` (per ADR-0027 方案 C) | 中 |
| macOS / Windows 支持 | 适配各平台动态库搜索路径；ADR-0027 现状 Linux-only | 低 |
| `cuInit` / `cuCtx*` context management | 当前架构把 current-context 作为前置条件；full context 管理需后续 | 中 |
| Packed `extra` argument buffer | 当前 `cuLaunchKernel` 接受 packed 参数，完整 packed `extra` argument buffer 支持待定 | 中 |

## 13. Glossary

| 术语 | 含义 |
|---|---|
| PTXIR | PTX-EMU 中间表示二进制格式，见 ADR-0023 |
| PTXIR-Embedded CUBIN | 标准 binary 末尾追加 PTXIR section 和 magic footer 的混合格式，见 ADR-0024 |
| PTXEMB magic | ADR-0024 定义的嵌入 footer magic |
| MANIFEST section | PTXIR 内部 manifest，具体 section type 和布局以 ADR-0024 为准 |
| DT_RUNPATH | ELF dynamic section 中由现行 `-Wl,-rpath` 行为生成的运行时搜索路径 |
| fallback | 未发现 footer 时回到 cuobjdump 路径 |
| `ModuleRecord` | in-memory module loading path 的 module handle 持有者；包含已解析 IR、深拷贝 image bytes、kernel metadata |
| `FunctionRecord` | in-memory module loading path 的 function handle 持有者；绑定到 `ModuleRecord` |
| in-memory front door | `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cuModuleUnload` 的 Driver API 调用路径，与 legacy linked executable registration 并列 |
| legacy front door | `__cudaRegisterFatBinary` 处理链接后的 executable；受 `PTXIR_MODE` precedence 控制 |
| image executor | ADR-0029 引入的设备侧执行抽象，持有 PTXIR image bytes 并通过 `ptxemu_image_*` C-API 暴露给 CP（UsrLinuxEmu/TaskRunner）调用。每次 launch 重新反序列化 image bytes 为 fresh `PtxContext`（mutation bug 修复），同 handle 并发 launch 由 executor mutex 串行化 |
| `cpptlm_module.h` | ADR-0029 新增的 ABI header，声明 `ptxemu_image_load` / `ptxemu_image_execute` / `ptxemu_image_unload` 等 `extern "C"` 函数；与 `cpptlm_bridge.h` ABI 独立，自带 `CPPTLM_MODULE_VERSION 1` 宏 |
| `libptxemu_device.so` | ADR-0029 新增的 device-side 库，CP 端可直接 dlopen / link；包含 `PtxEmuImageExecutor` 实现 |

---

**最后更新**: 2026-08-09（v1.3: §2 CP 端跨仓集成节点表 + §11 ADR-0028 BLOCKING DEPENDENCY 升级 + §12 HAL extension future work；v1.2: 新增 ADR-0029 关联 + §5.4 image bytes ownership + §10.5 image executor acceptance items + §11 ADR-0029 替换 TBD 占位 + §12 Phase 跟踪）
**维护者**: PTX-EMU Architecture Team
