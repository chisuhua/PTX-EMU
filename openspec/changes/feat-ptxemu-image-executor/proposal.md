# feat-ptxemu-image-executor

## Why

PTX-EMU 现状 (`__cudaRegisterFatBinary` @ `src/cudart/cudart_sim.cpp:354-386`) 仅支持**单 LD_PRELOAD front door**:通过 `readlink("/proc/self/exe")` + `cuobjdump` 提取 PTX 文本 → ANTLR 解析 → `PtxContext` 装载。这种架构对**嵌入式 binary 部署**和**非 NVIDIA 硬件真机部署**(UsrLinuxEmu + TaskRunner 软件栈)完全不友好:
- 嵌入式 binary 需手动 `ptxir_embed` + 末尾 magic 检测,断点流程多
- UsrLinuxEmu/TaskRunner 端 KMD/CP 无法直接接收 PTXIR image bytes 作为输入(只能接收文件名)
- 跨仓 HAL 集成方案([UsrLinuxEmu ADR-076](https://example.com/adr-076))需要 PTX-EMU 提供可调用的 C-API,而当前架构没有

本提案实现 **PTX-EMU in-memory image executor**(per ADR-0029):通过 `libptxemu_device.so` + `cpptlm_module.h` 提供 5 个 `extern "C"` ABI 函数,允许外部 caller(UsrLinuxEmu HAL、TaskRunner shim、其他 future consumers)以 opaque handle 方式加载 PTXIR bytes 并执行 kernel,而无需 ANTLR 解析路径。

**前置治理已完成**(per CheckList G + ADR-0021 v1.1 amendment,commit `8d05f35f` + `100afdc4`):
- ADR-0029 (image executor architecture, D1-D8 决策固化,Proposed) ✅ ship
- ADR-0021 v1.1 amendment (D-PTX-1:76 同 TU 约束解除) ✅ merged (commit `8d05f35f`)
- 6 轮 Oracle review + 3 轮 F1 hardening + 1 轮 F4 canonical sync ✅ 通过

**待实现**(本 change 范围):
- Phase 0 Step 1:5 个全局符号实际搬迁(`g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override` → `PtxEmuDriverShim.cpp`;`g_gpu_context` → `ptx_interpreter.cpp`)
- Phase 1:`cpptlm_module.h` + `PtxEmuImageExecutor` 实现 + `libptxemu_device.so` CMake target + D3 perf acceptance + 7 个 SINGLE-GPU-INSTANCE 标记
- 5 byte-identical fallback gates (D7) 实测
- git tag `v0.1.0+` 触发 ADR-076 §Migration Step 1 完结

**不在本 change 范围**(per ADR-076 跨仓 commit 顺序 + ADR-035 §R5.1 mirror 协议):
- Phase 2 UsrLinuxEmu HAL extension(ADR-076 §Migration Step 2,跨仓,独立 ADR)
- Phase 2 TaskRunner `cu_module.cpp::cuModuleLoadData` 改造(tadr-307,跨仓,独立 ADR)
- PTX-EMU 仓**零 link 依赖**(per ADR-0029 §D8.1 关键不变量)

## What Changes

### Phase 0 Step 1:5 全局符号搬迁

- **修改** `src/cudart/cudart_sim.cpp` — 移除 `g_cpptlm_bridge` 定义 (`:104`)、`cpptlm_attach_bridge`/`cpptlm_detach_bridge` 定义 (`:126-134`)、`g_bridge_user_override` 定义;移除 `g_gpu_context` 定义 (`:92`)。**不修改** 任何 `g_cpptlm_bridge`/`g_gpu_context` 调用点的逻辑(line-level diff 锁)
- **新增定义** `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` — `CppTLMBridge* g_cpptlm_bridge = nullptr;` + `extern "C"` `cpptlm_attach_bridge(CppTLMBridge*)` + `cpptlm_detach_bridge()` + `g_bridge_user_override` 定义(同 TU as `PtxEmuDriverShim` 类,维持 `cudart_sim.cpp:121-124` same-TU 不变量)
- **新增定义** `src/cudart/ptx_interpreter.cpp` — `std::unique_ptr<GPUContext> g_gpu_context;`(同 TU as `PtxInterpreter` 类)
- **零修改** `include/cudart/cpptlm_bridge.h`(ABI 真值源 governance 严守,per AGENTS.md 反模式)
- **零修改** `CPPTLMBRIDGE_VERSION`(保持 = 2)

### Phase 1:cpptlm_module.h + PtxEmuImageExecutor + libptxemu_device.so

- **新增** `include/cudart/cpptlm_module.h` — 公共 ABI 头(per ADR-0029 D1):
  - `#define CPPTLM_MODULE_VERSION 1`(独立版本治理,类似 `CPPTLMBRIDGE_VERSION` + `PTXIR_VERSION`)
  - `extern "C"` 5 个函数:
    - `uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size)` — 接受 standalone PTXIR 或 PTXIR-Embedded CUBIN,返回 opaque handle
    - `int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size)` — 查询 handle 内 kernel 名(v1 单 kernel)
    - `int ptxemu_image_execute(uint64_t handle, uint32_t grid_x, grid_y, grid_z, uint32_t block_x, block_y, block_z, size_t shared_mem_bytes, void** kernel_args, size_t args_count)` — 同步 launch(阻塞至完成)
    - `int ptxemu_image_unload(uint64_t handle)` — 卸载;in-flight kernel 返回 busy
    - `int ptxemu_module_version(void)` — 返回 `CPPTLM_MODULE_VERSION`(调用方启动时校验)
  - 零 PTX-EMU 内部类型暴露(governance per ADR-0029 §D1)
- **新增** `src/cudart/cpptlm_module.cpp` — `PtxEmuImageExecutor` 类(per ADR-0029 §D2 + §D3 + §D6):
  - **持有** `std::vector<uint8_t> image_bytes_`(私有,来自 `ptxemu_image_load` 的 deep copy)
  - **每次** `ptxemu_image_execute` 调用 `PTXIRLoader::deserializeForCubin(image_bytes_)` + `PtxContextAdapter::fromEmbedded()` 构造 fresh `PtxContext`(per D3 mutation bug 修复);launch 后析构
  - **executor mutex** 串行化同 handle 并发 launch([SINGLE-GPU-INSTANCE ASSUMPTION] 标记 #5, per ADR-0029 §D6)
  - **global singleton**(`g_image_executor` 单例,标记 #4) — 进程内一个 simulated GPU(`g_gpu_context` 标记 #1 + `CudaDriver::instance()` 标记 #2 + `g_cpptlm_bridge` 标记 #3 共享)
  - **每 launch 构造新 `PtxInterpreter`**(标记 #6,状态非重入)
  - **不接 SingletonGuard**(`__cudaRegisterFatBinary` 的 FATAL guard 不影响 image executor 路径,标记 #7)
- **新增** `src/cudart/CMakeLists.txt` 修改 — `add_library(ptxemu_device SHARED ...)` 链接 `ptxsim` + `ptx_ir` + `ptxir` 三个 existing 共享库 + `cpptlm_module.cpp`(per ADR-0029 D5 Phase 1)
- **新增** `src/cudart/cpptlm_module_export.h`(可选) — 显式导出符号(visibility default)

### 测试(unit + integration)

- **新增** `tests/unit/cudart/test_cpptlm_module.cpp` — 5 ABI 入口覆盖:
  - `ptxemu_image_load` roundtrip(standalone PTXIR + PTXIR-Embedded CUBIN)
  - `ptxemu_image_kernel_name` 输出验证
  - `ptxemu_image_execute` 同步等待 + grid/block/args 透传
  - `ptxemu_image_unload` happy path + in-flight busy
  - `ptxemu_module_version` 返回正确值
  - 0 handle / 已 unload handle 拒绝
- **新增** `tests/unit/cudart/test_image_executor_mutation.cpp` — D3 mutation bug 修复验证:
  - (a) 同 bytes 两次 deserialize → byte-identical `kernelStatements`
  - (b) 顺序 launch N 次(不同 blockDim)→ 输出确定,无累积
  - (c) image bytes hash 经 N 次 launch 不变
- **新增** `tests/integration/test_cpptlm_module_inflight.cpp` — 并发 launch 串行化验证(executor mutex 行为)
- **新增** `tests/integration/test_cpptlm_module_dlopen.cpp` — DL-isolated 测试:`dlopen("libptxemu_device.so")` 无 libcudart.so 依赖下独立调用

### Phase 1 验收(D7 5 gates + D7 gate 6 perf)

- **新增** `tests/integration/test_phase0_byte_identical_gates.cpp` — 5 gates 实测:
  1. `nm -D --defined-only libcudart.so` 前后 diff 为空
  2. SONAME `libcudart.so.12` 保持 + POST_BUILD symlink 保持
  3. e2e 套件 stdout 字节 diff 为空(monolithic vs split build)
  4. `g_cpptlm_bridge==nullptr` 单元测试通过
  5. `logger.cpp → g_gpu_context` 时钟路径单元测试通过
- **新增** `tests/performance/test_ptxir_deserialize_cost.cpp` — D3 perf gate:
  - 使用 `bench/cute/cute_rmsnorm.ptx` PTXIR
  - 测量 "load + 100 × execute(每 launch 重 deserialize)" vs "load + execute × 1 + 复用 PtxContext" wall time 比
  - **阈值**:< 1.10(10% overhead 容差,实测非估算)
  - 超标触发 A1 fallback 决策点(launch 时 deep-copy `kernelStatements`)

### Tag + release

- **新增** git tag `v0.1.0`(或 `v0.1.0+` per ADR-065 version policy)
- **新增** `CHANGELOG.md` entry(per ptx-lessons-learned §8 — 重大功能交付必须同步更新)

### 文档同步(per Lesson §8 重大功能交付清单)

- **修改** 根 `README.md` — §已实现功能新增 "PTX-EMU Image Executor" + §已知限制移除 "in-memory Driver API TBD" 描述(per `ptxir-toolchain-stack.md §11` 填平)
- **修改** `docs/dev-process/lessons-learned.md` — 新增章节:Image executor per-launch re-deserialize(D3 mutation bug 修复 pattern)
- **修改** `docs/adr/ADR-0029-ptxemu-image-executor.md` §合规检查 — Phase 0 / Phase 1 checkboxes 全部勾选
- **修改** `docs/adr/ADR-0021-cpptlm-d1-full-integration.md` — 如果 Phase 0 Step 1 实际搬迁过程中发现 v1.1 amendment 需微调(governance check)

## Capabilities

### New Capabilities

- **`ptxemu-image-executor`**: PTX-EMU in-memory image executor C-API + 共享库。具体契约:
  - 5 个 `extern "C"` 函数(`ptxemu_image_load/execute/unload/kernel_name/module_version`)签名与语义
  - `CPPTLM_MODULE_VERSION 1` 版本治理(独立于 `CPPTLMBRIDGE_VERSION` + `PTXIR_VERSION`)
  - 接受 input image bytes 类型:standalone PTXIR(前 4 字节 = "PTXI") + PTXIR-Embedded CUBIN(末尾 magic `PTXEMB\x01\x00`)
  - 拒绝 NVIDIA cubin / fatbin / Tile IR(返回错误码)
  - v1 单 kernel per image 限制(per `ptxir_format.h:36-41` `ManifestSection`)
  - **D3 mutation bug 修复契约**:image bytes 私有保存,每次 launch 重 deserialize,无 stored state mutation
  - **D6 SINGLE-GPU-INSTANCE 假设**:进程内一个 simulated GPU,executor mutex 串行化同 handle launch
  - **byte-identical 默认 LD_PRELOAD 路径**(per D7 5 gates)

- **`phase-0-symbol-relocation`**: PTX-EMU 仓 5 全局符号搬迁契约。具体:
  - `g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override` 同 TU 迁出 `cudart_sim.cpp` 到 `PtxEmuDriverShim.cpp`(per ADR-0021 v1.1 amendment,line-level diff 锁)
  - `g_gpu_context` 迁出 `cudart_sim.cpp` 到 `ptx_interpreter.cpp`(同 TU as `PtxInterpreter`)
  - `cpptlm_bridge.h` ABI 头零修改(governance 严守,per `include/cudart/AGENTS.md` 反模式)
  - `CPPTLMBRIDGE_VERSION = 2` 不变
  - 5 byte-identical fallback gates(D7)全部通过

### Modified Capabilities

_None._ 默认 LD_PRELOAD 路径(`__cudaRegisterFatBinary` + `cudaLaunchKernel` + `cudaModuleLoadData` 不动)在字节级行为兼容(`PTXIR_MODE=off` 等价现状)。

## Impact

- **受影响 Specs**:
  - `ptxir-format-compliance` — 间接引用 `PTXIRLoader::deserializeForCubin` 复用,无需修改
  - `cpptlm-d1-full` — `cpptlm_bridge.h` 治理保持,但 `g_cpptlm_bridge` 实际 TU 变更(same ABI surface)
- **受影响代码(PTX-EMU 仓)**:
  - `src/cudart/cudart_sim.cpp` — 移除 5 全局符号定义(line-level diff 锁,不改 logic)
  - `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` — 新增 4 bridge 符号定义
  - `src/cudart/ptx_interpreter.cpp` — 新增 `g_gpu_context` 定义
  - `src/cudart/cpptlm_module.{h,cpp}` — 新建(image executor 主入口)
  - `src/cudart/CMakeLists.txt` — 新增 `libptxemu_device` target
- **ABI 影响**:
  - **零破坏性变更**(per D7 5 gates 字节级兼容):默认 LD_PRELOAD 路径行为完全等价
  - 新增公共 ABI:`include/cudart/cpptlm_module.h`(`CPPTLM_MODULE_VERSION = 1`,独立治理)
- **跨仓影响**:
  - 触发 UsrLinuxEmu `adr-076` Migration Step 1 完结(可启动 Step 2 HAL extension)
  - 触发 TaskRunner `tadr-307` consumer-side 改造前置条件
- **性能影响**:
  - 启动路径:`g_cpptlm_bridge` 全局查找需 1 次额外 indirect call(在 cudart_sim.cpp 通过 `extern` 调用 `PtxEmuDriverShim.cpp` 中定义)
  - Image executor 路径:D3 per-launch deserialize O(bytes) — gate 6 实测需 < 10% overhead
- **文档影响**:
  - 根 `README.md` §已实现功能 / §已知限制 同步(per ptx-lessons-learned §8)
  - `docs/dev-process/lessons-learned.md` 新增章节(D3 pattern)
  - `docs/adr/ADR-0029-ptxemu-image-executor.md` §合规检查 全部勾选
  - `CHANGELOG.md` v0.1.0 entry
- **风险**:
  - **HARD**:Phase 0 Step 1 实施过程中若发现 ADR-0021 v1.1 amendment 需微调(governance check 失败)→ 触发 ADR-0021 v2 二次 amendment(独立 change)
  - **MEDIUM**:D3 perf gate 6 超标(>10% overhead)→ 触发 A1 fallback 决策(launch 时 deep-copy `kernelStatements`,per ADR-0029 §D3)
  - **MEDIUM**:并发 launch 测试中发现的 mutex overhead 性能问题 → 评估是否需要升级到 lock-free queue(per ADR-0029 §D6 SINGLE-GPU-INSTANCE 假设边界讨论)
  - **LOW**:CMake 链接依赖膨胀 → libptxemu_device.so 大小 < 5MB(估算,基于现有 ptxsim + ptx_ir + ptxir 共享库 ~3MB 总和)

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性(per Lesson §1)

- [ ] baseline 函数所有 `set_*`/`commit_*`/`force_*`/`lock_*` 调用已列出 — N/A,本 change 是新功能 + 符号搬迁
- [ ] `cudart_sim.cpp` 移除 5 全局符号时,**不修改**任何调用点(line-level diff 锁)— Phase 0 Step 1 任务清单必须逐行核对

### 多 Phase 推进(per Lesson §3)

- [x] Phase 拆分方案 + 独立 commit 粒度已说明(Commit 1: Phase 0 搬迁;Commit 2: Phase 1 cpptlm_module;Commit 3: D3 perf;Commit 4: tag)
- [x] 基线 worktree 命令已记录(per Lesson §4)
- [x] 失败处理策略(revert 该 Phase,不混入后续 commit)已说明

### Pre-impl Review(per Lesson §7)

- [x] 10 轮 Oracle review 已完成(Round 1-6 + Round 7 D8 canonical sync) — docs 全部 ✓ ACCEPTED-ready
- [x] 实施前最后 Metis pre-impl review 强制(进入 `openspec/apply-change` 时)

### 文档同步(per Lesson §8 重大功能交付清单)

- [x] 根 `README.md` §已实现功能 / §已知限制 同步项已列出
- [x] AGENTS.md 同步项已列出(无新增 — 仅 state transition docs)
- [x] ADR 追加段落已规划(ADR-0029 §合规检查 勾选)
- [x] tasks.md Phase 状态变更已说明
- [x] CHANGELOG.md entry 已规划
- [x] `docs/dev-process/lessons-learned.md` 新增章节已规划(D3 mutation bug 修复 pattern)

### 单一实例 / Single-Instance Assumption(per Lesson §10)

- [ ] `PtxEmuImageExecutor` 类头注释包含 7 个 [SINGLE-GPU-INSTANCE] 标记(per ADR-0029 §D6):
  1. `g_gpu_context` 全局唯一
  2. `CudaDriver::instance()` 单例
  3. `g_cpptlm_bridge` 单指针
  4. `PtxEmuImageExecutor` 单例(`g_image_executor`)
  5. executor mutex(并发 launch 串行化)
  6. `PtxInterpreter` 状态非重入(每 launch 新构造)
  7. 不接 SingletonGuard

### Byte-Identical Fallback(per Lesson §14)

- [x] Phase 0 Step 1 完成后 5 gates 全部由直接单元测试锁定(D7 + 4 gates):
  1. `nm -D --defined-only libcudart.so` diff 为空
  2. SONAME `libcudart.so.12` 保持
  3. e2e 套件 stdout 字节 diff 为空
  4. `g_cpptlm_bridge==nullptr` 单元测试
  5. `logger→g_gpu_context` 单元测试
- [x] D3 perf gate 6(cute_rmsnorm < 1.10)由 perf benchmark 锁定
