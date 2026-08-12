# ptxir-driver-api-front-door

**优先级**: P0 | **来源**: [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) v1.3 §2 §4.2 + [roadmap.md](roadmap.md) Phase 12.3.A + 2026-08-10 实施状态审计
**阶段**: Phase 12.3 | **分类**: arch-design
**类型**: functional

## 架构依据

当前 `libcudart.so` 只有**单一 legacy front door**——`__cudaRegisterFatBinary` 处理链接后的可执行文件。架构文档 §2 Components 表与 §4.2 Runtime data flow 设计的"in-memory module loading front door"（`cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cuModuleUnload`）在 `libcudart.so` 侧**完全未实现**——`nm -D build/lib/libcudart.so` 实测仅导出 `cuModuleLoad`（stub）与 `cuModuleGetFunction`（stub, line 514-521），缺 `cuModuleLoadData` / `cuModuleUnload` / 真 `cuLaunchKernel(CUfunction,...)` Driver API 版本。

这造成两个架构性缺陷：

1. **CUDA Driver API 用户（TaskRunner、动态加载场景、CP 端跨仓集成）无法使用 PTX-EMU**——他们调 `cuModuleLoadData` 时要么拿不到，要么拿到 stub 返回的假 handle
2. **现有 front door 之间缺乏清晰边界**——legacy 与 in-memory 必须共存且互不污染（架构 §4.2：in-memory 路径不读 `/proc/self/exe`、不调 `cuobjdump`、不读 `PTXIR_MODE`）

参考 ADR-0029 §D8（image executor）已经为 `libptxemu_device.so` 建立了"image bytes + per-launch re-deserialize"的范式，但那是给 UsrLinuxEmu HAL 用的；本提案为 `libcudart.so` 自身的 Driver API front door 建立对等的能力，且**与 `libptxemu_device.so` 路径解耦但执行后端共享**（架构 §2 边界说明）。

## 范围

**In Scope**:
- 在 `libcudart.so` 新增 4 个 Driver API 入口：`cuModuleLoadData` / `cuModuleGetFunction`（真版本替换 stub）/ `cuLaunchKernel(CUfunction,...)` / `cuModuleUnload`
- 新增不透明 handle 数据结构：`ModuleRecord` + `FunctionRecord` + `ModuleRegistry`（架构 §5.2 §5.3 契约）
- 新增 image classifier（架构 §5.1 6 类）
- 新增 7 类 error mapping（架构 §7）
- Registry 线程安全（Driver API 可从多 host thread 调用）

**Out Scope**:
- `cpptlm_bridge.h` ABI 不变（ADR-0029 D7 5 byte-identical gates 继续 hold）
- `libptxemu_device.so` ABI 不变（5 `ptxemu_image_*` 入口已 ship）
- `__cudaRegisterFatBinary` legacy front door 不变（架构 §4.1 保持独立）
- `cuInit` / `cuCtx*` context management（架构 §12 Future-4 远期）
- Packed `extra` argument buffer（架构 §12 Future-5 远期）

## 关键场景

### 场景 1：端到端 in-memory module loading

- **GIVEN** 应用代码持有 standalone PTXIR image bytes
- **WHEN** 调 `cuModuleLoadData(module, image)` → `cuModuleGetFunction(func, module, name)` → `cuLaunchKernel(func, grid, block, args, shared, stream)` → `cuModuleUnload(module)`
- **THEN** 4 个调用全部成功；kernel 实际执行；image bytes 经 N 次 launch 不被 mutate；模块卸载后 function handle 立即失效

### 场景 2：legacy / in-memory front door 边界独立性

- **GIVEN** `PTXIR_MODE=off`（legacy front door 关闭 PTXIR dispatch，走 `cuobjdump`）
- **WHEN** 应用代码同时走 `__cudaRegisterFatBinary` legacy 路径与 `cuModuleLoadData` in-memory 路径
- **THEN** legacy 路径行为不变（cuobjdump 语义）；in-memory 路径 PTXIR dispatch **仍然 ON**（架构 §4.2 明确：与 `PTXIR_MODE` 无关）

### 场景 3：多 host thread 并发 cuLaunchKernel

- **GIVEN** 同一 `CUmodule` handle 的 `CUfunction` 被 N 个 host thread 并发 launch
- **WHEN** 全部 thread 同时调 `cuLaunchKernel`
- **THEN** 所有 launch 串行执行（registry mutex）；无 data race；无 stored state mutation；最终结果确定

## 技术约束

### MUST

- **image bytes deep copy**（架构 §5.3）：`cuModuleLoadData` 返回前必须把 image bytes 拷贝到 `ModuleRecord` 私有存储，caller-owned pointer 在调用返回后不作为 handle 存活
- **eager parse**（架构 §5.3）：`cuModuleLoadData` 返回前必须完成解析（不 lazy）
- **6 类 image classifier**（架构 §5.1）：PTX text / standalone PTXIR / executable-tail PTXIR suffix / NVIDIA cubin / NVIDIA fatbin / Tile IR；前 2 类 SUPPORTED，第 3 类 REJECTED（defer legacy），后 3 类 NOT SUPPORTED → `CUDA_ERROR_INVALID_IMAGE`
- **Registry 线程安全**：所有 `ModuleRegistry` 路径必须 `std::mutex` 保护（Driver API 可从多 host thread 调用）
- **per-launch fresh `PtxContext`**（架构 §5.4 + ADR-0029 D3）：不缓存 `kernelStatements`；每次 `cuLaunchKernel` 重新 deserialize；修复 `ptx_interpreter.cpp:100-140` mutation bug
- **复用 `PTXIRLoader` + `PtxContextAdapter`**（架构 §2）：不重新发明二进制解析；`PTXIRLoader::deserializeForCubin()` 是唯一入口
- **`std::optional` / `nullptr` 失败路径**（per archive change `2026-08-07-implement-ptxir-cubin-embed-extension` 约束）：所有失败路径不抛异常
- **in-flight unload 返回 busy**（架构 §10 item 24）：kernel 在飞时调 `cuModuleUnload` 返回 `CUDA_ERROR_INVALID_HANDLE`（busy）

### MUST NOT

- **不读取 `/proc/self/exe`**（架构 §4.2）：in-memory 路径只用 caller-provided image
- **不调 `cuobjdump`**：in-memory 路径不走 cuobjdump fallback
- **不读 `PTXIR_MODE`**（架构 §4.2）：in-memory 路径 PTXIR dispatch 始终 ON
- **不修改 `cpptlm_bridge.h` ABI**（ADR-0029 D7）：`CPPTLMBRIDGE_VERSION` 保持 2
- **不修改 `libptxemu_device.so` 5 `ptxemu_image_*` ABI**（已 ship）
- **不修改 `__cudaRegisterFatBinary` legacy front door**（架构 §4.1 保持独立）
- **不在 WarpContext / ThreadContext / GPUContext 核心执行路径添加新依赖**（per `improvements/implement-ptxir-cubin-embed-extension.md` 约束）

### SHOULD

- 新建文件 `include/cudart/module_registry.h` + `src/cudart/module_registry.cpp`（`cuda_driver.h` 是内存分配器，**不混合职责**）
- 新建文件 `src/cudart/image_classifier.cpp`（cudart_sim.cpp 已 ~1478 行，分类器是纯函数易单测）
- 与 `libptxemu_device.so` 路径解耦但执行后端 `PtxInterpreter` / `GPUContext` 共享
- Registry 单例模式（per cpptlm_bridge.h precedent）

## 验收标准（架构层）

提案被批准后，guide-design → openspec proposal.md → tasks.md 时应明确：

1. **`libcudart.so` 导出 4 个新 Driver API T 符号**：`cuModuleLoadData` + `cuModuleGetFunction`（替换 stub）+ `cuLaunchKernel(CUfunction,...)` + `cuModuleUnload`（架构 §10 item 8）
2. **in-memory 与 legacy front door 行为字节级独立**（架构 §4.1 §4.2 边界）：`PTXIR_MODE=off` 不影响 in-memory 路径（架构 §10 item 12）
3. **mutation bug 不再发生**：per-launch fresh `PtxContext` 经 1000 次 launch 后 image bytes SHA-256 不变（架构 §5.4 + ADR-0029 D3）
4. **`cpptlm_bridge.h` ABI 5 byte-identical gates 继续 PASS**：`git diff cpptlm_bridge.h` 为空 + `CPPTLMBRIDGE_VERSION` 保持 2 + SONAME 保持 + symlinks 保留 + g_cpptlm_bridge 单测通过（ADR-0029 D7 + 架构 §10 item 21）
5. **与 `libptxemu_device.so` 边界清晰**：两条路径可独立调用，互不依赖；可同进程共存

---

**注**：本提案只回答"为什么"和"什么"，不维护详细 tasks/tasks.md。详细实施 tasks 由 guide-design 评审通过后创建的 openspec `proposal.md` → tasks.md 维护（per OpenSpec lifecycle, Lesson §6）。
