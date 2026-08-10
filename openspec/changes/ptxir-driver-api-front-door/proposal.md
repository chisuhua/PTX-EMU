# ptxir-driver-api-front-door

> **Oracle 评审结果（2026-08-10）**: ✅ APPROVE-WITH-CONDITIONS — 风险 MEDIUM-HIGH
> **关键发现**: 当前 Driver API front door 比提案描述的更残缺——4 个函数中只有 `cuModuleGetFunction` 是 stub，另 3 个**完全不存在**
> **杠杆最大**: per-launch fresh `PtxContext` 直接修复 `ptx_interpreter.cpp:100-140` mutation bug（ADR-0029 §触发事件-4）

## Why

当前 `libcudart.so` 只有**单一 legacy front door**——`__cudaRegisterFatBinary` 处理链接后的可执行文件。架构文档 §2 §4.2 设计的"in-memory module loading front door"（`cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cuModuleUnload`）在 `libcudart.so` 侧**完全未实现**——Oracle 实测 `nm -D build/lib/libcudart.so` 仅导出 `cuModuleLoad`（stub）与 `cuModuleGetFunction`（stub, line 514-521），缺 `cuModuleLoadData` / `cuModuleUnload` / 真 `cuLaunchKernel(CUfunction,...)` Driver API 版本。

这造成两个架构性缺陷：
1. **CUDA Driver API 用户（TaskRunner、动态加载场景、CP 端跨仓集成）无法使用 PTX-EMU**
2. **现有 front door 之间缺乏清晰边界**——legacy 与 in-memory 必须共存且互不污染（架构 §4.2）

参考 ADR-0029 §D8 已经为 `libptxemu_device.so` 建立了"image bytes + per-launch re-deserialize"的范式；本提案为 `libcudart.so` 自身的 Driver API front door 建立对等的能力，**与 `libptxemu_device.so` 路径解耦但执行后端共享**。

## What Changes

**In Scope**:
- 在 `libcudart.so` 新增 4 个 Driver API 入口：`cuModuleLoadData` / `cuModuleGetFunction`（替换 stub）/ `cuLaunchKernel(CUfunction,...)` / `cuModuleUnload`
- 新增不透明 handle 数据结构：`ModuleRecord` + `FunctionRecord` + `ModuleRegistry`
- 新增 image classifier（架构 §5.1 6 类，Oracle 条件 #2：单测 TDD）
- 新增 7 类 error mapping（架构 §7）
- Registry 线程安全（Oracle 条件 #1：明确 mutex 范围与 lock order）

**Out of Scope**:
- `cpptlm_bridge.h` ABI 不变
- `libptxemu_device.so` ABI 不变（5 `ptxemu_image_*` 入口已 ship）
- `__cudaRegisterFatBinary` legacy front door 不变
- `cuInit` / `cuCtx*` context management（架构 §12 Future-4 远期；Oracle 条件 #5：建 follow-up proposal）
- Packed `extra` argument buffer（架构 §12 Future-5 远期；Oracle 条件 #5）

### 关键场景

#### 场景 1：端到端 in-memory module loading
- **GIVEN** 应用代码持有 standalone PTXIR image bytes
- **WHEN** `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → `cuModuleUnload`
- **THEN** 4 个调用全部成功；image bytes 经 N 次 launch 不被 mutate（修复 mutation bug）

#### 场景 2：legacy / in-memory front door 边界独立性
- **GIVEN** `PTXIR_MODE=off`
- **WHEN** 同时走 legacy 与 in-memory 路径
- **THEN** legacy 路径行为不变；in-memory 路径 PTXIR dispatch 仍 ON

#### 场景 3：多 host thread 并发 cuLaunchKernel
- **GIVEN** 同一 `CUmodule` 的 `CUfunction` 被 N 个 host thread 并发 launch
- **WHEN** 全部 thread 同时调 `cuLaunchKernel`
- **THEN** 所有 launch 串行执行（registry mutex）；无 data race；无 stored state mutation

## Capabilities

- **image bytes deep copy**
- **eager parse**
- **6 类 image classifier**（架构 §5.1）
- **Registry 线程安全**（Oracle 条件 #1：mutex 范围明确 + lock order 定义）
- **per-launch fresh `PtxContext`**（Oracle 条件 #3：regression test 验证 barrier-mask corruption）
- **复用 `PTXIRLoader` + `PtxContextAdapter`**（Oracle 条件 #4：grep 验证仅此入口）
- **`std::optional` / `nullptr` 失败路径**
- **in-flight unload 返回 busy**
- **不读 `/proc/self/exe`**、**不调 `cuobjdump`**、**不读 `PTXIR_MODE`**
- **不修改 `cpptlm_bridge.h` ABI**
- **不修改 `libptxemu_device.so` 5 ABI**
- **不在 WarpContext / ThreadContext / GPUContext 核心执行路径添加新依赖**（per `improvements/implement-ptxir-cubin-embed-extension.md` 约束，避免污染 SIMT 调度热路径）

## Impact

- **`libcudart.so` 新增 4 个 Driver API T 符号**
- **legacy / in-memory 行为字节级独立**
- **`ptx_interpreter.cpp:100-140` mutation bug 不再发生**
- **`cpptlm_bridge.h` ABI 5 byte-identical gates 继续 PASS**

## Acceptance

### Oracle 评审通过条件（HARD）
- [ ] **C1**: `ModuleRegistry` mutex 范围明确定义（覆盖 cuModuleLoadData / cuModuleGetFunction / cuLaunchKernel / cuModuleUnload）；定义 lock order vs per-`PtxContext` 锁（per `ptx-lessons-learned.md` 递归锁教训）
- [ ] **C2**: 6 类 image classifier 必须有 `tests/unit/cudart/image_classifier_test.cpp` 单元测试（TDD per `ptx-grammar-modification` 纪律）
- [ ] **C3**: Per-launch fresh `PtxContext` 修复必须有回归测试——并发 launch 同一 image 两次，断言无 barrier-mask corruption（这是 ADR-0029 §触发事件-4 引用的 bug）
- [ ] **C4**: grep 回归检查：`grep -r "deserializeForCubin\|deserialize" src/cudart/` 仅 1 处 `deserializeForCubin` 调用，无其他反序列化路径
- [ ] **C5**: 新建 follow-up proposal task（记入 `openspec/changes/` 或 issues）：`cuInit` / `cuCtx*` context management 与 packed-extra buffer

### A8 单元测试枚举（roadmap.md:135，≥13 测试 = 6 + 5 + 2）

**A8-1 至 A8-6**（image classifier 6 类，对应 Oracle C2）：
- [ ] PTX text → SUPPORTED
- [ ] standalone PTXIR → SUPPORTED
- [ ] executable-tail PTXIR suffix → REJECTED (defer legacy)
- [ ] NVIDIA cubin → `CUDA_ERROR_INVALID_IMAGE`
- [ ] NVIDIA fatbin → `CUDA_ERROR_INVALID_IMAGE`
- [ ] Tile IR → `CUDA_ERROR_INVALID_IMAGE`

**A8-7 至 A8-11**（架构 §10 item 11，5 类 error mapping 端到端）：
- [ ] cubin 输入 → `CUDA_ERROR_INVALID_IMAGE`
- [ ] malformed PTX → `CUDA_ERROR_INVALID_PTX`
- [ ] unknown module handle → `CUDA_ERROR_INVALID_HANDLE`
- [ ] missing kernel symbol → `CUDA_ERROR_NOT_FOUND`
- [ ] stale handle → `CUDA_ERROR_INVALID_HANDLE`

**A8-12 至 A8-13**（stale handle 边界 2 类）：
- [ ] stale module handle after `cuModuleUnload` → `CUDA_ERROR_INVALID_HANDLE`
- [ ] stale function handle after parent module unload → `CUDA_ERROR_INVALID_HANDLE`

### 标准交付物
- [ ] **`libcudart.so` 导出 4 个新 Driver API T 符号**（架构 §10 item 8）
- [ ] **in-memory 与 legacy front door 行为字节级独立**（架构 §4.1 §4.2；验证：A8b 的 `PTXIR_MODE=off` 不影响 in-memory 路径测试 + 场景 2 的同进程双路径共存）
- [ ] **mutation bug 不再发生**：per-launch fresh `PtxContext` 经 1000 次 launch 后 image bytes SHA-256 不变
- [ ] **`cpptlm_bridge.h` ABI 5 byte-identical gates 继续 PASS**：`git diff cpptlm_bridge.h` 为空 + `CPPTLMBRIDGE_VERSION` 保持 2 + SONAME 不变
- [ ] **与 `libptxemu_device.so` 边界清晰**：两条路径可独立调用，互不依赖；可同进程共存（验证：场景 2 的 legacy/in-memory 共存测试 + A8b 的 `PTXIR_MODE=off` 独立性测试共同保证此属性；两条路径无任何符号依赖）
