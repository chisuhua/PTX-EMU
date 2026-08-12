# fix-path-coverage-gaps

**优先级**: P0 | **来源**: Oracle review (User reported Path 1B e2e gap; verified Path 1C/2D gaps)
**阶段**: default | **分类**: core-test
**类型**: debt

## 架构依据

### 1. 当前覆盖矩阵 (来源：实测 grep, 经 Oracle 验证)

| 路径 | unit | integration | e2e 真实执行 | 缺口 |
|---|---|---|---|---|
| 1A Legacy PTX | ⚠️ 故意 skip | ⚠️ SingletonGuard only | ✅ test_blackwell_gemm.cu + test_tcgen05_*.cu + tests/e2e/divergence/*.cu | 靠现有 e2e 间接覆盖 |
| 1B PTXIR fat-binary | ✅ test_ptxir_config | ✅ test_ptxir_cubin_loader (调 dispatch 函数) | **❌ 缺失** | proposal 写有 "PTX-EMU 加载" 但实际只验证格式 |
| 1C Driver API | ✅ test_cuda_driver_api + helpers | ✅ test_cuda_driver_api + mutation | **❌ 缺失** | test_cuda_driver_api 测 load/get_function/unload; `cuLaunchKernel` 仅在 test_error_mapping.cpp 覆盖错误路径 (NULL/stale handle)，**从未成功执行 kernel** |
| 2D Image Executor | ✅ test_cpptlm_module (rc==0 only) | ✅ test_libptxemu_device + 5 个 | **⚠️ 部分** | 缺输出正确性验证 (cute_rmsnorm baseline 仅有 ABI baseline，无 output baseline) |

### 2. 关联 ADR / 已存档 OpenSpec

- **ADR-0024** (PTXIR-Embedded CUBIN, 2026-08-06 Accepted) — Risk 1: NVIDIA cuobjdump 必须容忍尾部 PTXIR。test_ptxir_cubin_embed.cpp 验证 Risk 1 成立，但**未验证 PTX-EMU 真的能加载并执行**该 embedded binary
- **ADR-0029** (PTX-EMU Image Executor, 2026-08-10 ship) — D6: SINGLE-GPU-INSTANCE 假设。test_cpptlm_module.cpp 仅验证 API 调用成功，未验证 RMSNorm 输出正确
- **已存档 OpenSpec `implement-ptxir-cubin-embed-extension`** (2026-08-07 ship) — proposal §Capabilities 声称 e2e 测试 "PTX-EMU 加载 + ptxir_extract" 但**实际只验证了格式 round-trip**。交付文件 `.cu` → `.cpp` 后缀变化是 silent descoping 的证据
- **OpenSpec `multi-entry-handle-api`** (2026-08-12 ship) — Phase C1-C6 实施；⚠️ archived tasks.md 中 **C1 (line 14) 和 C2 (line 25) commit checkboxes 未勾选**，需确认是 archive 时漏更新还是确实未完成

### 3. Oracle review 真实阻断历史 (为什么这成为 P0)

`test_ptxir_cubin_embed.cpp:82-84` 显式 skip 了 `e2e_cuModuleLoadData_noDriver_explicitSkip`：

> "Phase 12.2 R5: Oracle review scenarios for ADR-0024 §风险 risk 1.
>  Validates the core architectural assumption of PTXIR-Embedded CUBIN format:
>  NVIDIA's cuobjdump MUST tolerate trailing PTXIR section + footer in the
>  embedded cubin. If this fails, the whole PTXIR-Embedded CUBIN story breaks."

也就是说：**该 e2e 仅验证 "NVIDIA 工具能否容忍 PTXIR 追加"**，并未验证 **"PTX-EMU 真的能加载并执行 embedded binary"**。这是一个**关键的架构假设尚未被任何 e2e 验证**：假设 `try_ptxir_dispatch_from_memory` 在生产场景中真的能从 `/proc/self/exe` 正确反序列化 PTXIR 并 dispatch 到 `g_ptx_interpreter`。

### 4. 结构性约束 (Oracle 补充)

`cuModuleLoadData` **显式拒绝 `kExecutableTailPtxir`** (`cudart_sim.cpp:532-537`) — driver API 路径拒绝恰好是 Path 1B 接受的 fat-binary 形式。意味着 **Path 1C coverage 无法用作 Path 1B 的覆盖替身**，3 个缺口是结构性独立问题，必须各自补齐。

### 5. 重构风险 (Oracle 修正归属)

cudart 持续重构横跨全部 4 条路径：

- sm_context god-class refactor → **Path 1A** (已有 1A e2e 守护)
- cpptlm VERSION 1→2 bump → **Path 2D** (现有 2D 测试仅 rc==0)
- multi-kernel manifest → **Path 1B + 1C** (均无 e2e guard)

其中 **Path 1B/1C/2D 均无 output-correctness 级 e2e guard**，重构回归只能靠 format-level 测试兜底。下一步 Blackwell tcgen05 重构、多仓 HAL extension 都依赖 PTXIR 路径稳定。

### 6. 隐含债务 (Oracle 修正 D-PTX 编号)

ADR-0021 已定义 D-PTX-1 (`g_cpptlm_bridge` 全局指针位置与初始化时机) 至 D-PTX-6。新债务须编号 D-PTX-7/D-PTX-8 避免冲突：

- D-PTX-1 (`g_cpptlm_bridge`, ADR-0021) — covered
- D-PTX-2 (SingletonGuard) — covered
- **D-PTX-7 (proposed)**: PTXIR fat-binary 端到端未验证
- **D-PTX-8 (proposed)**: Driver API 真实成功 kernel 执行未验证

## 范围

### In Scope (5 个分阶段执行的工作)

**Phase 1 — Path 1B (PTXIR fat-binary) 真实 e2e**

- 新建 `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp` (e2e_ 前缀 + ctest)
- **Standalone binary 构建**:
  1. nvcc 编译 `path_1B_kernels.cu` → cubin (≥3 kernels: vector_add, matmul, reduction)
  2. **`build/bin/ptxir_embed` 多次调用** (Oracle 修订 — 每次只接受一个 `--kernel-name`):

     ```
     ptxir_embed --in-cubin kernels.o --in-ptx vector_add.ptx --kernel-name vec_add --out kernels_v1.o
     ptxir_embed --in-cubin kernels_v1.o --in-ptx matmul.ptx --kernel-name matmul --out kernels_v2.o
     ptxir_embed --in-cubin kernels_v2.o --in-ptx reduction.ptx --kernel-name reduce --out kernels_final.o
     ```

     (or 现有 ptxir_embed 已支持多段追加 — 待 implementer 验证工具支持范围)
  3. link PTX-EMU 的 `lib/libcudart.so` + `PTXIR_MODE=auto` env
- fork+exec 该 standalone binary，触发 `__cudaRegisterFatBinary` → `try_ptxir_dispatch_from_memory` → `g_ptx_interpreter` → `cudaLaunchKernel` 全链路
- **Anti-fallback guard (Oracle 修订 — 顺序调整)**:
  - **(推荐)** PATH 操纵 — 在 fork+exec 前设置 `PATH=""` (or unset cuobjdump location)，使 `extract_ptx_with_cuobjdump` 子进程调用失败；若 PTXIR dispatch 也失败，test_ptxir_fatbinary_exec.cpp 应收到 FATAL 或空输出 (证明 fallback 没生效)
  - (备选) Dispatch marker — 需要 PTX-EMU 的 `libcudart.so` 导出 dispatch hit counter (`__attribute__((constructor))` 在 test binary 中无法直接观测库内 dispatch)；Phase 5 改进 `cpptlm_module.cpp` 暴露 `extern "C" uint32_t ptxemu_ptxir_dispatch_hits()` ABI 后再启用
- 验证 kernel 输出与 Legacy 路径 (Path 1A) 字节级一致

**Phase 2 — Path 1C (Driver API) 真实 e2e**

- 新建 `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp` (e2e_ 前缀)
- 编译 .cu → cubin (or 直接 prepare PTXIR bytes)
- 调用 `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel` → 内部转 `cudaLaunchKernel`
- 验证 kernel 输出正确 (与 Path 1A/1B 字节级一致)
- ≥3 scenarios: lookup成功 + duplicate handle + not-found error

**Phase 3 — Path 2D (Image Executor) 输出正确性验证**

- 增强 `tests/integration/cudart/test_libptxemu_device.cpp` (新增 test case)
- 读 `tests/ptxir/fixtures/cute_rmsnorm.ptxir` fixture
- 调 `ptxemu_image_load + ptxemu_image_execute`
- **新增 output baseline**: `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` (reference output bytes)
- 验证输出 buffer 与 baseline 字节级一致

**Phase 4 — tests/e2e/ 重组织 (路径化目录)**

新建 4 个子目录，每个有独立 `CMakeLists.txt` (新模式 — 现有 divergence/ 没有自己的 CMakeLists.txt):

| 目标子目录 | 移入文件 | 来源 |
|---|---|---|
| `path_1A_legacy_ptx/` | `test_blackwell_gemm.cu`, `test_tcgen05_*.cu` | `git mv tests/e2e/kernel/` |
| `path_1A_legacy_ptx/` | `test_divergence*.cu` | `git mv tests/e2e/divergence/` (整目录内容) |
| `path_1B_ptxir_fatbinary/` | `test_ptxir_cubin_embed.cpp` (Oracle 补: 格式验证 + e2e 共处) | `git mv tests/e2e/kernel/` |
| `path_1B_ptxir_fatbinary/` | Phase 1 新建文件 | 新增 |
| `path_1C_driver_api/` | Phase 2 新建文件 | 新增 |
| `path_2D_image_executor/` | Phase 3 增强文件 | 增强 |

**保留不动 (Oracle 建议)**: `tests/e2e/kernel/` 内的非 4-path 测试 (`test_test3_cfg_full`, `test_barrier_warp_sync`, `test_ldglobal_simple`, 3 个 shared_memory 测试, `test_flashattention_mini`, `test_printf`) + 整个 `tests/e2e/cosim/`。这些不属于 4-path 范畴，不动。

修改 `tests/e2e/CMakeLists.txt`:

- 删除被移走的 `add_catch_test()` 调用
- 新增 4 个 `add_subdirectory(path_X/)` 调用
- 给所有 path-related e2e 测试加 `LABELS "e2e;path_1X;..."` (Oracle 修订 — 必须含 `e2e` 段以保证 `regression.sh -L e2e` 覆盖)
- 新建 4 个 `path_X/CMakeLists.txt` (新模式，每个含 `add_catch_test(e2e_XXX ...)`)

ctest label 一致化: `ctest -L path_1B` 直接定位单路径全部测试

**Phase 5 — Proposal 描述修正 (一致性)**

- 修改 `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md`
- §Capabilities 中 `tests/e2e/test_ptxir_cubin_embed.cu` 描述补全 disclaimer:
  > "**Note [修正: 2026-08-12, see fix-path-coverage-gaps]**: 此 e2e 验证 PTXIR-Embedded CUBIN 格式兼容性 (Phase 12.2 R5 / ADR-0024 Risk 1)，**不验证 PTX-EMU 真实加载执行**。完整端到端执行验证见 `openspec/changes/path-1b-ptxir-fatbinary-e2e/` (Phase 1 of `fix-path-coverage-gaps` improvement)"
- 不重命名已存档 change (避免 archive history 篡改)；可考虑建 errata 文档 (project 有先例: `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md`)

### Out of Scope (明确不做的事)

- **不修改 Path 1A/1B/1C/2D 的实现代码** — 仅补测试，不动 cudart_sim.cpp / cpptlm_module.cpp / ptxir_loader.cpp 等生产代码
- **不修复 `multi-entry-handle-api` 任务未勾选状态** — 这是 archive gate 的 process gap，需另立 improvement (`archive-gate-incomplete-tasks`)
- **不引入新测试框架** — 沿用 Catch2 + add_catch_test (现有)
- **不创建新的 PTXIR fixture 生成工具** — Phase 1 用现有 nvcc 编译简单 kernels；Phase 3 用现有 `cute_rmsnorm.ptxir` fixture (5294 B)
- **不修改 openspec CLI / openspec validate 规则** — 测试 failure 不应被 openspec 误判
- **不修复 Path 1A 现有 e2e 的 SingletonGuard 问题** — 现有 test_divergence.cu 已 inline kernel 避免该问题，足够覆盖
- **不做黑well tcgen05 路径 1B/1C 集成** — 现有 test_tcgen05_*.cu 已经走 Path 1A 间接覆盖足够；Phase 4 仅把它们移到 path_1A/ 子目录但保留 Path 1A 守护
- **不修改 ctest 标签体系** — 仅添加新 label，不破坏现有 LABELS
- **不动 PTX-EMU 整体测试目录结构** — 仅修改 tests/e2e/ 子树，不动 tests/unit/ 或 tests/integration/
- **不修改 anti-fallback guard 的实现细节** — Phase 1 仅规定行为契约，dispatch marker 实现细节由 implementer 决定

### 边界冲突 (4 条)

- **vs. `tests/integration/cudart/test_ptxir_cubin_loader.cpp`** (Phase 12.2 R5): 该文件测 dispatch 函数，保留不动；Phase 1 新建 e2e 是更高层级
- **vs. `tests/unit/cudart/test_cuda_driver_api.cpp`**: 测 driver API 函数本身，保留不动；Phase 2 新建 e2e 是真实 kernel 执行
- **vs. `tests/unit/cudart/test_cpptlm_module.cpp`**: 测 API 调用成功 (rc==0)，保留不动；Phase 3 在 integration 层加 output correctness
- **vs. `tests/e2e/kernel/test_ptxir_cubin_embed.cpp`** (Oracle 补): 格式 round-trip，保留不动 (format-level)；Phase 1 新建 e2e 是 execution-level 在 path_1B/ 子目录内与该文件**共存**

## 关键场景

### Phase 1 — Path 1B (PTXIR fat-binary)

**Scenario 1.1: 标准 PTXIR dispatch 流程**

- GIVEN:
  - standalone CUDA binary `kernel_exec_ptxir` 含 PTXIR section (通过 `ptxir_embed` 嵌入 binary 末尾)
  - binary link PTX-EMU `lib/libcudart.so`
  - 执行环境 `PTXIR_MODE=auto` + `PATH=""` (anti-fallback)
  - **前置验证 (Oracle 修订)**: `generate_ptxir` 对多 entry PTX 序列化全部 statements 且 manifest 填充 v1 `kernel_name` (否则 `try_ptxir_dispatch_from_memory` 返回 kMalformedManifest)。**若不支持，Phase 1 降级为单 kernel binary**
- WHEN: 用户执行 `kernel_exec_ptxir`，触发 `__cudaRegisterFatBinary` → `try_ptxir_dispatch_from_memory` → `g_ptx_interpreter->set_ptx_context` → `cudaLaunchKernel<<<grid,block>>>(vec_add_kernel, args)`
- THEN:
  - binary stdout 输出 `RESULT: vec_add=<expected> matmul=<expected> reduce=<expected>`
  - binary exit code = 0
  - PTXIR dispatch 命中 (PATH="" 保证 fallback cuobjdump 失败；若 PTXIR 也失败 binary 会输出错误而非 RESULT 行)

**Scenario 1.2: PTXIR footer 缺失**

- GIVEN: standalone CUDA binary **无** PTXIR footer (但 `PTXIR_MODE=auto` 已设置)
- WHEN: `__cudaRegisterFatBinary` 调用
- THEN:
  - `try_ptxir_dispatch_from_memory` 返回 `kNoFooter`
  - 控制流转入 `extract_ptx_with_cuobjdump`，但 PATH="" 使其失败
  - **`__cudaRegisterFatBinary` 返回 nullptr; stderr 输出 `Error: Could not extract PTX code`** (Oracle 修订 — emulator 不直接控制 exit code)

**Scenario 1.3: PTXIR footer 损坏**

- GIVEN: standalone CUDA binary 含 magic 但 footer body 损坏 (e.g., u32 size 超过 binary 长度)
- WHEN: `__cudaRegisterFatBinary` 调用
- THEN:
  - `try_ptxir_dispatch_from_memory` 返回 `kMalformedPtxir`
  - **`__cudaRegisterFatBinary` 返回 nullptr; stderr 输出 `malformed embedded PTXIR: footer present but deserialize failed`** (Oracle 修订)

**Scenario 1.4: Path 1B vs Path 1A 字节级一致**

- GIVEN: 同一 `path_1B_kernels.cu` 编译两份 binary: `kernel_exec_ptxir` (含 PTXIR footer, 走 Path 1B) + `kernel_exec_legacy` (无 PTXIR footer, 走 Path 1A)
- WHEN: 两 binary 各执行一次
- THEN: 输出 stdout 字节级一致 (验证 PTXIR fast-path 与 ANTLR parse 路径语义等价)

**Scenario 1.5: kMalformedManifest** (Oracle 新增)

- GIVEN: standalone CUDA binary 含 valid PTXIR footer + valid statements，但 `manifest.kernel_name` 为空 (v1 manifest 缺失必填字段)
- WHEN: `__cudaRegisterFatBinary` 调用
- THEN:
  - `try_ptxir_dispatch_from_memory` 返回 `kMalformedManifest`
  - `__cudaRegisterFatBinary` 返回 nullptr; stderr 输出 `manifest mismatch: kernel_name is empty`

### Phase 2 — Path 1C (Driver API)

**Scenario 2.1: 完整 cuModule* 流程**

- GIVEN: PTXIR image bytes (含 `ManifestSection` + statements)
- WHEN: cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel → cudaLaunchKernel
- THEN: 三个调用均 CUDA_SUCCESS, mod != nullptr, func != nullptr, 输出 buffer 与 Path 1B 字节级一致

**Scenario 2.2: Duplicate handle**

- GIVEN: 同一 PTXIR image 两次 cuModuleLoadData
- WHEN: 第二次调用
- THEN: 第二次 CUDA_SUCCESS 但生成新 mod handle, 两次 mod 不相等

**Scenario 2.3: Not-found error**

- GIVEN: cuModuleLoadData 后的 module **fixture manifest 为 v2 格式 (`kernels[]` 非空)** (Oracle 修订 — v1 manifest 下任意 name 均返回 SUCCESS)
- WHEN: `cuModuleGetFunction(&func, mod, "nonexistent_kernel")`
- THEN: 返回 `CUDA_ERROR_NOT_FOUND`，`func` 保持未修改

**Scenario 2.4: cuLaunchKernel 错误路径 (回归)**

- GIVEN: func == nullptr 或 params == nullptr
- WHEN: cuLaunchKernel
- THEN: 返回 CUDA_ERROR_INVALID_VALUE (per cudart_sim.cpp:607)

### Phase 3 — Path 2D (Image Executor)

**Scenario 3.1: cute_rmsnorm 输出正确性**

- GIVEN: `tests/ptxir/fixtures/cute_rmsnorm.ptxir` (5294 B)；**Phase 3 首先生成 `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` (golden capture)** (Oracle 修订)
- WHEN: `ptxemu_image_load(bytes, size)` → `ptxemu_image_execute(handle, grid, block, args)` → 读取输出 buffer
- THEN: 输出 buffer 与 baseline 字节级一致 (`memcmp == 0`)

**Scenario 3.2: 重复 load 同一 fixture (D3 mutation 回归)**

- GIVEN: 同一 fixture 加载 2 次
- WHEN: load + execute 两次
- THEN: 两 handle 不同, 两 output 字节级一致, 两 unload 成功

**Scenario 3.3: ABI baseline (回归)**

- GIVEN: libptxemu_device.so 已构建
- WHEN: tests/integration/cpptlm/test_libptxemu_abi_baseline.cpp
- THEN: ABI symbols 与 libptxemu_abi_baseline.txt 字节级一致

### Phase 4 — tests/e2e/ 重组织

**Scenario 4.1: ctest label 过滤**

- GIVEN: 重组织后所有 path-related 测试有 LABELS "e2e;path_1X;..."
- WHEN: ctest -L path_1B
- THEN: 仅 path_1B_ptxir_fatbinary/ 子目录内测试运行

**Scenario 4.2: 全量回归**

- GIVEN: 重组织后所有测试
- WHEN: ctest --output-on-failure
- THEN: 现有所有测试通过

**Scenario 4.3: per-subdir CMakeLists 独立构建**

- GIVEN: 4 个 path_X/CMakeLists.txt
- WHEN: 在 path_1B_ptxir_fatbinary/ 子目录 cmake build
- THEN: 该子目录独立编译

### Phase 5 — Proposal 修正

**Scenario 5.1: 文档一致性**

- GIVEN: 修改后的 proposal.md
- WHEN: 阅读 §Capabilities
- THEN: 看到 disclaimer 明确 test_ptxir_cubin_embed 只验证格式兼容性

**Scenario 5.2: Archive history 不篡改**

- GIVEN: 已存档的 proposal
- WHEN: Phase 5 修改
- THEN: 文件名/tasks.md 不变, 仅 proposal.md 文案补全, git log 显示追加

## 技术约束

### 通用约束 (跨所有 Phase)

- **MUST**: 测试独立可重复运行 — 不依赖执行顺序，不共享全局状态
- **MUST**: 测试代码遵循 PTX-EMU 编码规范 (clang-format LLVM + IndentWidth=4 + ColumnLimit=80, 文件 snake_case, 函数 camelCase)
- **MUST**: 维持 3 层测试隔离 — unit/integration/e2e 三类不混
- **MUST**: 所有新测试 ctest target 保留 `e2e_` 前缀 (commit ab55e06 约定)
- **MUST**: 所有新测试加 `LABELS "e2e;path_1X;..."` (新 path label + 保留 `e2e` 段以保证 regression.sh 覆盖)
- **MUST** (Oracle 修订): 新增 `.ptx` fixture 必须在 `.gitignore` 加白名单 (`!tests/e2e/path_X/**`); 否则 `*.ptx` 全局 ignore 规则会静默 untrack
- **MUST**: 临时生成文件 (e.g., `*_tmp.ptxir`) 必须被 ignore (`.gitignore` 已有 `tests/ptxir/*_tmp.ptxir`)
- **MUST**: committed baseline (`tests/ptxir/baselines/*.bin`) 不得被 ignore
- **MUST NOT**: 引入新第三方依赖 (C++20 + Catch2 + ANTLR4 + PTX-EMU 现有依赖已足够)
- **MUST NOT**: 引入新测试框架 (沿用 Catch2 + add_catch_test)
- **MUST NOT**: 修改 `cudart_sim.cpp` / `cpptlm_module.cpp` / `ptxir_loader.cpp` 等生产代码 (本改进仅补测试)
- **MUST NOT**: 修改 `tests/unit/` 和 `tests/integration/` 子树结构 (本改进仅动 tests/e2e/)
- **SHOULD**: 测试有清晰 `[type][path_X][feature]` tag 三段式 (现有 `[cudart][sync]` 风格扩展)
- **SHOULD** (新增): 新 e2e 测试设置 `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` + `ENVIRONMENT "PTX_EMU_GPU_CONFIG=ampere_a100.json;PTX_EMU_CONFIG=perf_config.ini"` (per existing basic e2e test pattern, tests/CMakeLists.txt:312)
- **SHOULD** (新增): 若新测试是 regression test for coverage gap，加 RED PHASE header comment (per tests/AGENTS.md:91)

### Phase 1 专属约束 (Path 1B)

- **MUST**: standalone CUDA binary 必须 fork+exec 启动 (避免 SingletonGuard 二次调用 FATAL abort, cudart_sim.cpp:329-335)
- **MUST**: standalone binary link PTX-EMU 的 `lib/libcudart.so` (不能用 nvcc 自带的 cuda runtime)
- **MUST**: binary 末尾嵌入 PTXIR section (通过 `build/bin/ptxir_embed --in-cubin/--in-ptx/--kernel-name/--out`)
- **MUST**: Anti-fallback guard (推荐: PATH="" 阻止 cuobjdump 子进程)
- **MUST**: 每个 kernel PTXIR 必须含 valid `manifest.kernel_name` (v1 manifest, 触发 kMalformedManifest 会失败)
- **MUST**: 验证 output 与 Path 1A 字节级一致 (per `tests/ptxir/...` 已有 precedent: 5 byte-identical gates verified)
- **MUST NOT**: 修改 `ptxir_embed` 工具源码 (Phase 1 仅消费现有工具)
- **MUST NOT**: 修改 `try_ptxir_dispatch_from_memory` 函数签名 (Phase 1 是消费者)
- **SHOULD**: ≥3 个不同复杂度 kernels (vector_add, matmul, reduction)
- **SHOULD**: 全部 4 个 dispatch 状态都覆盖 (kSuccess, kNoFooter, kMalformedPtxir, kMalformedManifest)
- **SHOULD**: 若 `generate_ptxir` 不支持多 entry 序列化，降级为单 kernel binary

### Phase 2 专属约束 (Path 1C)

- **MUST**: PTXIR image fixture 使用 **v2 manifest** (`kernels[]` 非空, 满足 NOT_FOUND 测试要求)
- **MUST**: 真实调用 `cuLaunchKernel` (不是仅测 error path — 当前 test_error_mapping.cpp 已覆盖 error path)
- **MUST**: 验证 kernel output buffer 内容正确 (不能只验证 rc == 0)
- **MUST NOT**: 修改 `ModuleRegistry::insert` 语义 (依赖现有重复 handle 行为)
- **MUST NOT**: 跳过 output correctness 验证 (Phase 2 必须新增与 Path 1A/1B 字节级对比)
- **SHOULD**: ≥3 scenarios (lookup success + duplicate handle + not-found error)
- **SHOULD**: 显式覆盖 `cuModuleUnload` 后 func2name 失效 (per cudart_sim.cpp:573-592)

### Phase 3 专属约束 (Path 2D)

- **MUST**: 生成 golden output baseline 文件 (`tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin`)
- **MUST**: baseline 仅在确认 simulator 正确后 commit (避免 baseline 错误被固化)
- **MUST**: 测试运行时若 baseline 不存在则明确报错 (而非自动生成或 skip)
- **MUST NOT**: 修改 ptxemu_image_* ABI (7 符号) + ptxemu_module_version, 共 **8 extern "C" 符号** (cpptlm_module.cpp:227-262); 任何 ABI 修改必须经过 ADR 流程 (Oracle 修订)
- **MUST NOT**: 修改 `cpptlm_module.cpp` 的 SINGLE-GPU-INSTANCE 假设 (per ADR-0029 D6)
- **SHOULD** (Oracle 修订): 定义 baseline 文件含 magic header (e.g., `PTXR_OUT\0\0` 8-byte magic + 4-byte LE size + bytes) 便于将来版本迁移

### Phase 4 专属约束 (重组织)

- **MUST**: 4 个新 `path_X/CMakeLists.txt` 各自独立 (新模式 — divergence/ 现有无子目录 CMakeLists)
- **MUST**: 每个子目录 CMakeLists.txt 复用父目录 CUDA flags (CMAKE_CUDA_PTX_COMPILATION ON, ARCHITECTURES 100 等)
- **MUST**: 现有 kernel/cosim 测试不变 (test_test3_cfg_full, test_barrier_warp_sync 等保留在原位)
- **MUST NOT** (Oracle 修订): 修改现有测试的 ctest labels; 新 labels 遵循 `<type>;<subject>` 约定
- **MUST NOT**: 修改已存档 change 文件名 (避免 archive history 篡改)
- **MUST NOT**: 修改 ctest 标签体系 — 仅添加新 label, 现有 label 不变
- **SHOULD**: 每个 path_X/CMakeLists.txt 含 1-2 行 header 注释说明该子目录的覆盖路径
- **SHOULD**: 重组织采用 `git mv` 而非 rm+add (保留 file history, 便于 git blame)

### Phase 5 专属约束 (Proposal 修正)

- **MUST**: 仅修改 `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` 的 §Capabilities 文案
- **MUST**: disclaimer 必须显式说明 test_ptxir_cubin_embed 只验证格式兼容性，不验证 PTX-EMU 加载执行
- **MUST** (Oracle 修订): disclaimer 须含 inline 修正标记 `[修正: 2026-08-12, see fix-path-coverage-gaps]` (对齐 ERRATA inline-merge 惯例, per HEALTH-AUDIT-2026-06-21-ERRATA.md)
- **MUST**: disclaimer 交叉引用 fix-path-coverage-gaps improvement 和 Phase 1 新测试位置
- **MUST NOT**: 修改 tasks.md (任何 checkbox 状态变化都是 archive gate violation)
- **MUST NOT**: 修改 change 目录名 (避免 archive history 篡改)
- **MUST NOT**: 添加新 test target 到该 archived change (已 archived 不应有 active work)
- **SHOULD** (Oracle 修订): 考虑另建 `docs/audits/implement-ptxir-cubin-embed-extension-ERRATA.md` (项目有先例: HEALTH-AUDIT ERRATA)

### 依赖 / 风险

- **依赖**: ptxir_embed 工具 (tools/CMakeLists.txt:1) 必须先 build 才能用于 Phase 1
- **依赖**: cute_rmsnorm.ptxir fixture (5294 B) 必须先 commit (已存在 ✅)
- **依赖**: libptxemu_device.so 必须先 build (依赖现有构建系统)
- **风险**: Phase 1 standalone binary 构建时间可能拖慢 ctest (建议标记 TIMEOUT 60 property)
- **风险**: cute_rmsnorm output baseline 可能因 simulator 微小变化失效 (per ADR-0029 D6 SINGLE-GPU-INSTANCE 假设 — 确定性强, 但需 baseline 更新流程)
- **风险**: Phase 4 重组织涉及 ~10 文件 git mv, 若有未提交修改会冲突
- **风险** (Oracle 修订): `cuModuleUnload` 修改 func2name 的实现位置在 `cudart_sim.cpp:573-592` (Oracle 行号修正) — Phase 2 测试依赖此行为不要变

## 验收标准

### 全局验收 (跨所有 Phase, 必须全部满足)

- **AC-G1**: `./scripts/sanity.sh` 通过 (分层健康检查)
- **AC-G2** (Oracle 修订): `ctest --output-on-failure -L "e2e;integration;unit"` 100% pass (与 regression.sh 范围对齐; 当前 e2e_divergence 实测全过, latent 一致性问题待 Phase 1/4 后实测验证)
- **AC-G3**: `./scripts/regression.sh` 通过 (全量回归)
- **AC-G4** (Oracle 修订): `clang-format --dry-run --Werror <changed-files>` 返回 0 (之前未定义 command; 先例: archived simt-architecture-fix plan)
- **AC-G5**: 5 个 Phase 全部 ship (openspec status=archived), iteration.json 同步更新
- **AC-G6** (Oracle 修订 — 降为流程说明): "下次 debt audit (预计 2026-09-02) 无新增条目（人工复核）" (原 AC 不可度量, no counter script)

### 命名 / Label 集成 (Oracle 新增 — critical)

- **AC-N1** (新增): 新测试 ctest target 必须保留 `e2e_` 前缀 (commit ab55e06 约定)
- **AC-N2** (新增, Oracle critical): 新测试 LABELS 必须含 `e2e` 段 (e.g., `e2e;path_1B`), 保证 `regression.sh -L e2e` 覆盖 — 否则回归脚本静默 skip 新测试

### Phase 1 验收 (Path 1B PTXIR fat-binary)

- **AC-1.1**: `ctest -L "path_1B"` 100% pass
- **AC-1.2**: test_ptxir_fatbinary_exec 覆盖 **全部 4 个 dispatch 状态** (Scenario 1.1/1.2/1.3/1.5)
- **AC-1.3**: Scenario 1.4 — Path 1B 与 Path 1A 输出**字节级一致** (`diff kernel_exec_ptxir.out kernel_exec_legacy.out` 返回 0)
- **AC-1.4**: Anti-fallback guard 验证 — `PATH=""` 阻止 cuobjdump, PTXIR 失败时 binary 输出空结果
- **AC-1.5**: ≥3 个不同复杂度 kernels 各自正确执行 (vector_add, matmul, reduction)
- **AC-1.6** (Oracle 修订): `ldd <binary> | grep lib/libcudart.so` 显示 PTX-EMU libcudart (原 nm -D 仅证明 dynamic reference, ldd 验证 RPATH 实际路径)
- **AC-1.7**: PTXIR footer 嵌入验证 — `xxd kernel_exec_ptxir | tail -12` 显示 8-byte PTXIR_EMBED_MAGIC
- **AC-1.8** (Oracle 修订): `set_tests_properties(<test> PROPERTIES TIMEOUT 60)` (原 wall-clock ≤60s 易受系统负载影响)

### Phase 2 验收 (Path 1C Driver API)

- **AC-2.1**: `ctest -L "path_1C"` 100% pass
- **AC-2.2**: test_cuda_driver_exec 覆盖 ≥3 个 scenarios (Scenario 2.1/2.2/2.3)
- **AC-2.3**: cuLaunchKernel 实际执行 kernel (Scenario 2.1) — output buffer 与 Path 1B 字节级一致
- **AC-2.4**: cuLaunchKernel 错误路径回归测试 (Scenario 2.4) — NULL/params 返回 CUDA_ERROR_INVALID_VALUE
- **AC-2.5**: cuModuleUnload func2name 失效验证 (per cudart_sim.cpp:573-592)
- **AC-2.6**: cuModuleLoadData negative paths 覆盖 — null args, non-PTXIR magic, cubin/fatbin image

### Phase 3 验收 (Path 2D Image Executor)

- **AC-3.1**: `ctest -L "path_2D"` 100% pass
- **AC-3.2**: cute_rmsnorm output 与 baseline 字节级一致 (Scenario 3.1) — `memcmp(output, baseline, size) == 0`
- **AC-3.3**: 重复 load 同一 fixture (D3 mutation, Scenario 3.2) — 两 handle 不同, output 一致, unload 成功
- **AC-3.4**: ABI baseline 回归 (Scenario 3.3) — `diff <(nm -D libptxemu_device.so | grep ptxemu_ | sort) libptxemu_abi_baseline.txt` 返回 0
- **AC-3.5**: baseline 文件 commit 验证 — `git ls-files tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` 存在
- **AC-3.6**: ≥4 个 new error path tests (load garbage, execute invalid handle, unload invalid handle, kernel_name 不存在)
- **AC-3.7** (Oracle 修订): Phase 3 定义 baseline 格式 (8-byte magic `PTXR_OUT\0\0` + 4-byte LE size + bytes) 并提交 baseline 文件 + 验证 (当前 repo 无 `PTXR_OUT` magic, Phase 3 必须先定义后验证)
- **AC-3.3-RED** (Oracle 新增): D3 mutation 回归测试 (Scenario 3.2) 加 RED PHASE header comment (per tests/AGENTS.md:91)

### Phase 4 验收 (重组织)

- **AC-4.1**: 4 个 path_X/ 子目录全部存在
- **AC-4.2**: 每个 path_X/CMakeLists.txt 含 ≥1 个 add_catch_test
- **AC-4.3**: `ctest -L "path_1B"` 仅运行 path_1B 子目录内测试
- **AC-4.4**: `ctest -L "path_1C"` 仅运行 path_1C 子目录内测试
- **AC-4.5**: `ctest -L "path_1X"` (4 个 X) 各自运行
- **AC-4.6**: `ctest --output-on-failure` 全量通过
- **AC-4.7**: 现有 kernel/cosim 测试不变
- **AC-4.8**: 重组织采用 `git mv` — `git log --follow` 验证 file history 保留

### Phase 5 验收 (Proposal 修正)

- **AC-5.1**: `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` §Capabilities 含 disclaimer
- **AC-5.2**: disclaimer 含 inline 修正标记 `[修正: 2026-08-12, see fix-path-coverage-gaps]` (Oracle MUST)
- **AC-5.3**: disclaimer 交叉引用 Phase 1 新测试位置
- **AC-5.4**: `git log -- openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` 显示本次修改为后续追加
- **AC-5.5**: archive 目录名 `2026-08-07-implement-ptxir-cubin-embed-extension` 不变
- **AC-5.6**: tasks.md 不变

### 度量指标 (项目健康度改进, Oracle 修订 counting rule)

- **AC-M1** (修订): cudart 路径测试覆盖率从 **3/4 (75%)** 提升到 **4/4 (100%)**
  - counting rule: `coverage(4-path) = (Path 1A ✅ + Path 1B ✅ + Path 1C ✅ + Path 2D ✅) / 4`
  - 现状 = 3/4 (Path 1B 工具级, Path 1C 缺失)
  - 目标 = 4/4 (Phase 1/2/3 完成后)
- **AC-M2** (修订): e2e output correctness 覆盖率 **2/4 → 4/4 (100%)**
  - counting rule: `output-correctness(4-path) = (Path 1A ✅ + Path 1B ✅ + Path 1C ✅ + Path 2D ✅) / 4`
  - 现状 = 1/4 (Path 1A ✅, Path 2D rc==0 但无 output baseline)
  - 目标 = 4/4
- **AC-M3**: openspec 文档一致性 (proposal ↔ implementation) 修复 1 处 (Phase 5)
- **AC-M4**: `ctest -L path_1X` 可作为单路径回归命令, 便于将来 cudart 重构时快速定位回归