# Tasks: feat-ptxemu-image-executor

> **策略**:TDD 5 步结构(Write failing test → Verify fail → Implement → Verify pass → Commit),per `ptx-lessons-learned.md` §7
> **依赖**:Commit 0 (ADR-0021 v1.1 amendment) 已 ship(commit `8d05f35f` + `100afdc4`)
> **风险**:Phase 0 Step 1 实施必须严格 line-level diff 锁(per ADR-0029 §D2 论证);D3 perf gate 6 必须实测不可估算

---

## 0. Commit 0 — ADR-0021 v1.1 Amendment(✅ 已 ship)

> **状态**:✅ 已 commit `8d05f35f` + `100afdc4`(Oracle Round 6 ✓ ACCEPTED-ready)

- [x] 0.1 ADR-0021 v1.1 amendment merged(解除 D-PTX-1:76 同 TU 约束)
- [x] 0.2 ADR-0029 D8 expansion + canonical sync merged(commit `100afdc4`)
- [x] 0.3 ptxir-toolchain-stack.md v1.2 merged + adr/README index sync
- [x] 0.4 6 轮 Oracle review 全 PASS(Round 1-6 + Round 7 canonical sync)
- [x] 0.5 **Phase 0 Step 0 hard gate CLEARED**(per ADR-0029 §合规检查)

---

## 1. Commit 1 — Phase 0 Step 1(5 全局符号搬迁)+ 5 Byte-Identical Gates

> **策略**:TDD 5 步 — 先写 5 gates 单元测试 → 验证 fail → 实施搬迁 → 验证 pass → commit
> **依赖**:Commit 0 (ADR-0021 v1.1 amendment merged)
> **风险**:搬迁过程中可能发现 ADR-0021 v1.1 amendment 需微调 → 触发 v2 二次 amendment

### 1.1 5 byte-identical gates 测试失败骨架(Step 1-2: Red)

- [ ] 1.1.1 创建 `tests/integration/test_phase0_byte_identical_gates.cpp` 骨架,5 个 test cases
- [ ] 1.1.2 失败测试 gate_1_nm_D_libcudart_so_symbol_diff_empty(`nm -D --defined-only libcudart.so | sort > before; ... > after; diff before after`)
- [ ] 1.1.3 失败测试 gate_2_soname_libcudart_so_12_preserved(`objdump -p libcudart.so | grep SONAME`)
- [ ] 1.1.4 失败测试 gate_3_post_build_symlinks_preserved(`ls -la lib/libcudart.so* | grep -E '\.12$|libcudart\.so$'`)
- [ ] 1.1.5 失败测试 gate_4_g_cpptlm_bridge_nullptr_unit_test(单元测试,引用 `if (g_cpptlm_bridge == nullptr)` 路径)
- [ ] 1.1.6 失败测试 gate_5_logger_g_gpu_context_clock_path(单元测试,引用 `get_gpu_clock_from_context`,验证搬迁后 logger.cpp:8 extern 仍解析)
- [ ] 1.1.7 验证测试 fail:`cmake --build build && ctest -R test_phase0_byte_identical_gates --output-on-failure`(expect 5 failures)

### 1.2 Phase 0 Step 1 实施(Step 3: Implement)

- [ ] 1.2.1 **`src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp`** — 新增 4 bridge 符号定义:
  - `CppTLMBridge* g_cpptlm_bridge = nullptr;`(顶层 TU 全局,非 static)
  - `extern "C"` `cpptlm_attach_bridge(CppTLMBridge* bridge)` — 实现内容从 `cudart_sim.cpp:153-180` 迁移(零逻辑修改)
  - `extern "C"` `cpptlm_detach_bridge()` — 实现内容从 `cudart_sim.cpp:182-195` 迁移(零逻辑修改)
  - `static thread_local bool g_bridge_user_override = false;`(per cpptlm_bridge.h 上下文推断)
- [ ] 1.2.2 **`src/cudart/ptx_interpreter.cpp`** — 新增 `g_gpu_context` 定义:
  - `std::unique_ptr<GPUContext> g_gpu_context;`(顶层 TU 全局,与 `PtxInterpreter` 类同 TU)
- [ ] 1.2.3 **`src/cudart/cudart_sim.cpp`** — 移除 5 全局符号定义(**仅定义移除,调用点逻辑零修改**):
  - 移除 `g_gpu_context` 定义(`:92`)
  - 移除 `g_cpptlm_bridge` 定义(`:104`)
  - 移除 `cpptlm_attach_bridge` 实现(`:153-180`,但函数声明保留 by `cpptlm_bridge.h:162`)
  - 移除 `cpptlm_detach_bridge` 实现(`:182-195`,但函数声明保留 by `cpptlm_bridge.h:169`)
  - 移除 `g_bridge_user_override` 定义
- [ ] 1.2.4 验证所有 `g_cpptlm_bridge` / `g_gpu_context` 调用点 **未修改**(grep + diff):
  - `grep -n "g_cpptlm_bridge" src/cudart/*.cpp` 排除定义点后,引用站点计数与搬迁前一致
  - `grep -n "g_gpu_context" src/cudart/*.cpp src/ptxsim/*.cpp` 排除定义点后,引用站点计数与搬迁前一致
- [ ] 1.2.5 验证 **logger.cpp:8** `extern size_t get_gpu_clock_from_context()` extern 声明在 `ptx_interpreter.h` 未变(搬迁不破坏解析)

### 1.3 5 gates 验证 pass(Step 4: Verify)

- [ ] 1.3.1 gate_1 PASS:`diff <(nm -D --defined-only build/lib/libcudart.so | sort) <(nm -D --defined-only build-baseline/lib/libcudart.so | sort)` 为空(基线 worktree per Lesson §4)
- [ ] 1.3.2 gate_2 PASS:`objdump -p build/lib/libcudart.so | grep SONAME` = `libcudart.so.12`
- [ ] 1.3.3 gate_3 PASS:`ls -la build/lib/libcudart.so*` 显示 `.12` + 主符号链接
- [ ] 1.3.4 gate_4 PASS:`ctest -R "test_cpptlm_bridge_nullptr"` exit 0
- [ ] 1.3.5 gate_5 PASS:`ctest -R "test_logger_clock_path"` exit 0(单元测试调用 `get_gpu_clock_from_context()` 返回递增时钟值)
- [ ] 1.3.6 额外 baseline 对比:跑现有 230+ ctest 全集,确认 0 regression(per Lesson §14 字节级 fallback 约束)

### 1.4 Commit

- [ ] 1.4.1 `git add` Phase 0 Step 1 涉及的所有文件(无 ADR/无 doc 改动)
- [ ] 1.4.2 commit message:`refactor(cudart): Phase 0 Step 1 - relocate 5 global symbols per ADR-0029 D2 / ADR-0021 v1.1 amendment. cpptlm_bridge.h ABI unchanged. 5 byte-identical gates verified.`
- [ ] 1.4.3 push + 验证 CI 全绿(若适用)

---

## 2. Commit 2 — Phase 1:cpptlm_module.h + PtxEmuImageExecutor + libptxemu_device.so

> **策略**:TDD 5 步 — 先写 5 ABI 入口 roundtrip 测试 → 验证 fail → 实现 PtxEmuImageExecutor → 验证 pass → commit
> **依赖**:Commit 1 (Phase 0 Step 1 + 5 gates PASS)
> **风险**:D3 perf gate 6 实测超 10% 阈值 → 触发 A1 fallback(launch 时 deep-copy `kernelStatements`)

### 2.1 cpptlm_module.h 5 ABI 入口测试失败骨架(Step 1-2: Red)

- [ ] 2.1.1 创建 `tests/unit/cudart/test_cpptlm_module.cpp` 骨架,5 个 test cases + invalid handle 边界
- [ ] 2.1.2 失败测试 `ptxemu_image_load_standalone_ptxir_returns_valid_handle`(使用 `tests/ptxir/fixtures/cute_rmsnorm.ptxir` 测试 fixture)
- [ ] 2.1.3 失败测试 `ptxemu_image_load_ptxir_embedded_cubin_returns_valid_handle`(使用嵌入 binary fixture)
- [ ] 2.1.4 失败测试 `ptxemu_image_load_zero_size_returns_zero`(零字节 image)
- [ ] 2.1.5 失败测试 `ptxemu_image_load_corrupt_magic_returns_zero`(前 4 字节 ≠ "PTXI" + 末尾无 `PTXEMB`)
- [ ] 2.1.6 失败测试 `ptxemu_image_kernel_name_valid_handle_returns_kernel_string`(验证返回 "cute_rmsnorm_kernel")
- [ ] 2.1.7 失败测试 `ptxemu_image_execute_valid_handle_returns_zero`(同步 launch + 验证 grid/block/args 透传)
- [ ] 2.1.8 失败测试 `ptxemu_image_execute_zero_handle_returns_einval`(0 handle)
- [ ] 2.1.9 失败测试 `ptxemu_image_unload_valid_handle_returns_zero`(正常卸载)
- [ ] 2.1.10 失败测试 `ptxemu_image_unload_inflight_returns_ebusy`(执行中调 unload)
- [ ] 2.1.11 失败测试 `ptxemu_module_version_returns_CPPTLM_MODULE_VERSION`(返回 1)
- [ ] 2.1.12 验证测试 fail:`cmake --build build && ctest -R test_cpptlm_module --output-on-failure`(expect 10 failures)

### 2.2 PtxEmuImageExecutor + cpptlm_module.cpp 实施(Step 3: Implement)

- [ ] 2.2.1 **`include/cudart/cpptlm_module.h`** — 新增(per ADR-0029 §D1):
  - `#define CPPTLM_MODULE_VERSION 1`
  - `extern "C"` 5 个函数声明(无 PTX-EMU 内部类型)
  - `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` 等编译期守卫(per cpptlm_bridge.h precedent)
  - `#include` 守卫:`<cstddef>` `<cstdint>`
- [ ] 2.2.2 **`src/cudart/cpptlm_module.cpp`** — `PtxEmuImageExecutor` 类(单例)实现:
  ```cpp
  class PtxEmuImageExecutor {
  public:
    static PtxEmuImageExecutor& instance();

    // 5 个公开方法对应 5 个 ABI 入口
    uint64_t load_image(const uint8_t* bytes, size_t size);
    int      get_kernel_name(uint64_t handle, char* buf, size_t buf_size);
    int      execute(uint64_t handle, uint32_t gx, uint32_t gy, uint32_t gz,
                    uint32_t bx, uint32_t by, uint32_t bz,
                    size_t shared_mem, void** args, size_t args_count);
    int      unload(uint64_t handle);
    int      version() const { return CPPTLM_MODULE_VERSION; }

  private:
    PtxEmuImageExecutor() = default;
    std::mutex exec_mu_;                                          // [SINGLE-GPU-INSTANCE] #5
    std::unordered_map<uint64_t, std::vector<uint8_t>> images_;   // handle -> image bytes (private copy)
    std::atomic<uint64_t> next_handle_{1};
  };
  // 全局单例 [SINGLE-GPU-INSTANCE] #4
  static PtxEmuImageExecutor* g_image_executor = &PtxEmuImageExecutor::instance();
  ```
- [ ] 2.2.3 `load_image` 实现:
  - 检测 image 类型(standalone PTXIR / PTXIR-Embedded CUBIN / NVIDIA cubin / fatbin / Tile IR — 后 3 类拒绝)
  - 若是 embedded:调 `PTXIRLoader::extractPureCubin()` 拿到纯 cubin + `PTXIRLoader::extractPTXIR()` 拿 section
  - **deep copy** image bytes 到 `images_[handle]`
  - 返回 `handle`(next_handle_.fetch_add(1))
- [ ] 2.2.4 `execute` 实现(**核心 — D3 mutation bug 修复**):
  - 加 `exec_mu_` lock
  - 从 `images_[handle]` 拿 image bytes,**每次都重新调** `PTXIRLoader::deserializeForCubin(image_bytes)` + `PtxContextAdapter::fromEmbedded()` 构造 fresh `PtxContext`(per D3)
  - 构造新 `PtxInterpreter`(per D6 标记 #6 — statefulness 非重入)
  - 调 `PtxInterpreter::launchPtxInterpreter(...)` 同步阻塞
  - 析构 fresh `PtxContext` + `PtxInterpreter`
  - 解锁 `exec_mu_`
- [ ] 2.2.5 `unload` 实现:
  - 尝试 erase `images_[handle]`
  - 若 in-flight(`exec_mu_` 忙),返回 `-EBUSY`
  - 否则返回 0
- [ ] 2.2.6 5 个 `extern "C"` 函数实现(薄 wrapper):
  - `ptxemu_image_load` → `g_image_executor->load_image(...)`
  - `ptxemu_image_kernel_name` → `g_image_executor->get_kernel_name(...)`
  - `ptxemu_image_execute` → `g_image_executor->execute(...)`
  - `ptxemu_image_unload` → `g_image_executor->unload(...)`
  - `ptxemu_module_version` → `g_image_executor->version()`
- [ ] 2.2.7 类头注释包含 7 个 [SINGLE-GPU-INSTANCE] 标记(per ADR-0029 §D6 + ptx-lessons-learned §10):
  - [SINGLE-GPU-INSTANCE] #1:`g_gpu_context` 全局唯一 — 引用 `extern std::unique_ptr<GPUContext> g_gpu_context;`
  - [SINGLE-GPU-INSTANCE] #2:`CudaDriver::instance()` 单例 — 引用 `CudaDriver::instance().malloc(...)`
  - [SINGLE-GPU-INSTANCE] #3:`g_cpptlm_bridge` 单指针 — standalone 模式 `nullptr`
  - [SINGLE-GPU-INSTANCE] #4:`g_image_executor` 单例
  - [SINGLE-GPU-INSTANCE] #5:executor mutex 串行化同 handle launch
  - [SINGLE-GPU-INSTANCE] #6:`PtxInterpreter` 状态非重入(每 launch 新构造)
  - [SINGLE-GPU-INSTANCE] #7:不接 SingletonGuard(`__cudaRegisterFatBinary` 路径)

### 2.3 libptxemu_device.so CMake target(Step 3 续)

- [ ] 2.3.1 **`src/cudart/CMakeLists.txt`** 修改 — 新增 `ptxemu_device` 共享库 target:
  ```cmake
  add_library(ptxemu_device SHARED
      cpptlm_module.cpp
  )
  target_link_libraries(ptxemu_device
      PUBLIC ptxsim ptx_ir ptxir
  )
  set_target_properties(ptxemu_device PROPERTIES
      VERSION ${PROJECT_VERSION}
      SOVERSION ${PROJECT_VERSION_MAJOR}
      POSITION_INDEPENDENT_CODE ON
  )
  install(TARGETS ptxemu_device LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  ```
- [ ] 2.3.2 顶层 `CMakeLists.txt` 修改 — `add_subdirectory(cudart)` 后挂 `ptxemu_device` target(若 cudart CMakeLists 已统一管理)
- [ ] 2.3.3 验证 build:
  - `cmake --build build --target ptxemu_device` 0 error
  - `ls -la build/lib/libptxemu_device.so` 存在 + SONAME 正确
  - `nm -D --defined-only build/lib/libptxemu_device.so | grep -E "ptxemu_"` 显示 5 个 ABI 符号

### 2.4 5 ABI 测试 + DL-isolated + Mutation 测试(Step 4: Verify)

- [ ] 2.4.1 `tests/unit/cudart/test_cpptlm_module.cpp` 10 个测试 PASS
- [ ] 2.4.2 新增 `tests/integration/test_cpptlm_module_dlopen.cpp`:
  - `dlopen("libptxemu_device.so")` 无 libcudart.so 依赖下独立调用
  - 验证 `dlsym` 5 个 ABI 符号 + `ptxemu_module_version() == 1`
  - `dlclose` 清理
- [ ] 2.4.3 新增 `tests/integration/test_cpptlm_module_inflight.cpp`:
  - 多线程并发 `ptxemu_image_execute` 同 handle → mutex 串行化(全部 success,无 corruption)
  - `std::async(std::launch::async, ...)` + `future.wait_for(30s)` 防 deadlock
- [ ] 2.4.4 新增 `tests/unit/cudart/test_image_executor_mutation.cpp` (D3 修复验证):
  - (a) 同 bytes 两次 deserialize → byte-identical `kernelStatements`
  - (b) 顺序 launch N=1000 次(不同 blockDim)→ 输出确定,无累积
  - (c) image bytes SHA-256 hash 经 N 次 launch 不变
- [ ] 2.4.5 跑全集 ctest 验证 0 regression

### 2.5 Commit

- [ ] 2.5.1 `git add` 新增 `include/cudart/cpptlm_module.h` + `src/cudart/cpptlm_module.cpp` + `src/cudart/CMakeLists.txt` 修改 + 测试
- [ ] 2.5.2 commit message:`feat(cudart): Phase 1 - cpptlm_module C-API + libptxemu_device.so. 5 ABI entry (load/execute/unload/kernel_name/version), CPPTLM_MODULE_VERSION=1, D3 mutation fix via per-launch re-deserialize. cpptlm_bridge.h unchanged.`
- [ ] 2.5.3 push + CI

---

## 3. Commit 3 — D3 Performance Gate 6 实测

> **依赖**:Commit 2 (libptxemu_device.so ship)
> **风险**:实测超过 10% 阈值 → 触发 A1 fallback 决策点(launch 时 deep-copy `kernelStatements`,独立 change)

### 3.1 cute_rmsnorm perf benchmark 实施

- [ ] 3.1.1 新增 `tests/performance/test_ptxir_deserialize_cost.cpp`
- [ ] 3.1.2 加载 `bench/cute/cute_rmsnorm.ptx` → 编译为 `cute_rmsnorm.ptxir`(`ptxir_build` 工具,per ADR-0025)
- [ ] 3.1.3 测量组 A:`ptxemu_image_load + ptxemu_image_execute × 1`(单次 launch,baseline PtxContext 缓存)
- [ ] 3.1.4 测量组 B:`ptxemu_image_load + ptxemu_image_execute × 100`(100 次 launch,每次重 deserialize,per D3)
- [ ] 3.1.5 计算 wall time 比 `B/A`
- [ ] 3.1.6 阈值断言:比 < 1.10(10% overhead 容差)
- [ ] 3.1.7 输出结构化报告:`deserialize_cost=1.05x  PASS` / `deserialize_cost=1.15x  FAIL (触发 A1 fallback)`

### 3.2 perf 结果处理

- [ ] 3.2.1 **若 PASS(< 1.10)**:commit perf benchmark + ADR-0029 §合规检查 gate 6 勾选
- [ ] 3.2.2 **若 FAIL(≥ 1.10)**:**触发 A1 fallback**(独立 change):
  - 暂停 Commit 3
  - 创建 `fix-ptxemu-image-executor-a1-fallback` change
  - 修改 `cpptlm_module.cpp::execute`:launch 时 deep-copy `kernelStatements`(O(N) per launch)
  - 重测 D3 perf(应 < 1.10)
  - 顺序:本 change 标记 blocked,等 fix change ship 后再 resume

### 3.3 Commit (perf PASS 路径)

- [ ] 3.3.1 `git add` perf benchmark + 结果
- [ ] 3.3.2 commit message:`test(perf): D3 deserialize cost gate 6 — cute_rmsnorm wall time ratio 1.05x (PASS < 1.10 threshold). Lock ADR-0029 compliance gate 6.`

---

## 4. Commit 4 — 文档同步 + git tag v0.1.0

> **依赖**:Commit 1 + 2 + 3 全 PASS
> **触发**:ADR-076 §Migration Step 1 完结 → UsrLinuxEmu/TaskRunner 可启动 Step 2/3

### 4.1 文档同步(per Lesson §8 重大功能交付清单)

- [ ] 4.1.1 根 `README.md` 修改:
  - §已实现功能 新增 "PTX-EMU Image Executor (libptxemu_device.so + cpptlm_module.h) — D3 mutation bug 修复 + 7 [SINGLE-GPU-INSTANCE] 标记 + 5 byte-identical gates + D3 perf acceptance"
  - §已知限制 移除 "in-memory Driver API TBD"(per `ptxir-toolchain-stack.md §11` 填平描述)
  - §快速开始 表格更新:`build/lib/libptxemu_device.so` row
- [ ] 4.1.2 `CHANGELOG.md` 新增 `v0.1.0` entry:
  ```markdown
  ## v0.1.0 (2026-08-XX)

  ### Added
  - PTX-EMU Image Executor (cpptlm_module.h + libptxemu_device.so) — 5 ABI entry for in-memory PTXIR module loading and execution
  - 5 global symbol relocation per ADR-0021 v1.1 amendment
  - D3 mutation bug fix via per-launch re-deserialize

  ### Changed
  - Default LD_PRELOAD path: byte-level unchanged (5 gates verified)

  ### Migration
  - Cross-repo: UsrLinuxEmu ADR-076 §Migration Step 1 complete; TaskRunner tadr-307 unblocked
  ```
- [ ] 4.1.3 `docs/dev-process/lessons-learned.md` 新增章节 §44:
  - 标题:"PTX-EMU Image Executor: per-launch re-deserialize vs cached PtxContext"
  - 内容:`src/cudart/ptx_interpreter.cpp:100-140` mutation bug 修复 pattern;`exec_mu_` mutex + fresh PtxContext per launch;性能 trade-off(D3 perf gate 6)
- [ ] 4.1.4 `docs/adr/ADR-0029-ptxemu-image-executor.md` §合规检查 全部勾选:
  - [x] Phase 0 Step 0 (Commit 0)
  - [x] Phase 0 Step 1 (Commit 1)
  - [x] Phase 0 完成 5 gates (Commit 1)
  - [x] Phase 1 完成 (perf) (Commit 3)
  - [x] Phase 1 完成 (5 个 ABI 入口测试) (Commit 2)
  - [x] Phase 1 完成 (mutation fix) (Commit 2)
  - [x] PtxEmuImageExecutor 7 SINGLE-GPU-INSTANCE 标记 (Commit 2)
- [ ] 4.1.5 `docs/adr/ADR-0021-cpptlm-d1-full-integration.md` §合规检查:
  - [x] ADR-0021 v1.1 amendment merged(v1.1 已 ship)
  - 5 byte-identical gates(若 Phase 0 实施后有 ADR-0021 v1.1 微调需求,触发 v2 amendment)

### 4.2 git tag v0.1.0

- [ ] 4.2.1 `git tag -a v0.1.0 -m "feat: PTX-EMU Image Executor (libptxemu_device.so + cpptlm_module.h). ADR-0029 + ADR-0021 v1.1 amendment ship. Cross-repo: UsrLinuxEmu ADR-076 Step 1 complete; TaskRunner tadr-307 unblocked."`
- [ ] 4.2.2 `git push --tags`(若已 push 习惯) 或仅本地 tag
- [ ] 4.2.3 触发跨仓通知:在 UsrLinuxEmu ADR-076 §Migration Step 2 owner 评审,TaskRunner tadr-307 owner 评审

### 4.3 Commit (docs-only)

- [ ] 4.3.1 `git add` README + CHANGELOG + lessons-learned + ADR-0029 + ADR-0021
- [ ] 4.3.2 commit message:`docs: PTX-EMU Image Executor v0.1.0 ship — README, CHANGELOG, lessons-learned §44 (D3 mutation pattern), ADR compliance checkboxes`

---

## 5. Final State — ADR-0029 → Accepted

- [ ] 5.1 `openspec/changes/feat-ptxemu-image-executor/tasks.md` 全部 checkbox 勾选
- [ ] 5.2 ADR-0029 状态:`Proposed` → `Accepted`(per OpenSpec workflow)
- [ ] 5.3 触发 ADR-076 §Migration Step 2(UsrLinuxEmu 仓独立推进)

---

## 跨仓 Handoff(不在本 change scope)

> 以下为 UsrLinuxEmu/TaskRunner 侧独立 ADR / change,本 change 仅触发其启动前置条件。

| 仓 | 触发条件 | 链接 |
|---|---|---|
| UsrLinuxEmu | PTX-EMU v0.1.0 tag ship | [adr-076 §Migration Step 2](../../../../../UsrLinuxEmu/docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md) |
| TaskRunner | UsrLinuxEmu `libptxemu_device.so` consumer 实现 ship | [tadr-307](../../../UsrLinuxEmu/external/TaskRunner/docs/shared/adr/tadr-307-igpu-driver-kernel-module-extension.md) |
