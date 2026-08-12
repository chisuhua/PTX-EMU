# PTX-EMU Roadmap

> **维护**: PTX-EMU Architecture Team
> **当前阶段**: Phase 10 — Documentation & Release（β 完成中）+ **Phase 12.2 PTXIR Cubin 集成 ✅ 2026-08-10 ship** + **Phase 12.3 PTXIR Driver API front door ✅ 2026-08-11 ship** + **Phase 12.4 ADR-0028 多 kernel manifest ✅ 2026-08-11 ship** + **Phase 13 HAL extension 跨仓协作 ✅ 2026-08-11 ship** + **Phase 12.5 多 entry handle API ⏳ 显式延后** (per [multi-kernel-manifest-gaps-gap-analysis](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md)) + **Phase 14 fix-path-coverage-gaps 📋 in progress** (P0 Oracle review 2026-08-12, 4-path cudart e2e coverage gaps)
> **最后更新**: 2026-08-12（Phase 14 fix-path-coverage-gaps P0 Oracle review 加入执行队列 + Phase 12.5 顺延到第 6 位）
> **关联**: [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) v1.4、[docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md)、[docs/roadmap/post-phase3-debt-roadmap.md](docs/roadmap/post-phase3-debt-roadmap.md)（详细债务清单）
> **参考**: [docs/README.md](docs/README.md)（文档索引）

---

## 当前状态

| 维度 | 数据 |
|------|------|
| ADR 数 | 28 个文件（ADR-0001~0029；ADR-0017 缺失；**ADR-0028 ✅ 2026-08-11 ship**） |
| OpenSpec 已归档 | 50+ 个（含 2026-08-11 最新 3 个：`ptxir-driver-api-front-door` + `multi-kernel-manifest-adr-0028` + `hal-extension-ptxemu-usrlinu-emu-taskrunner`） |
| 活跃 changes | 0（无活跃 change） |
| 测试覆盖 | unit / integration / e2e 三层物理隔离 |
| PTX 语法测试 | `./tests/ptx/test_all_ptx.sh` 45/45 |
| CppTLM 集成 | D1-Full MemoryBridge 已归档（ADR-0021） |
| PTXIR Image Executor | ✅ `libptxemu_device.so` + `cpptlm_module.h`（ADR-0029 Phase 1 已 ship） |
| Multi-kernel manifest | ✅ `ManifestSection.kernels` + `PTXIR_VERSION=4` 已 ship（ADR-0028 2026-08-11） |
| 延后项 | ⏳ Phase 12.5 multi-entry handle API（4 P0 gaps per [gap analysis](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md)） |
| 最近审计 | **2026-08-11 实施状态审计**（本文件 §实施状态审计 section 更新） |

## 已完成阶段

| Phase | 名称 | 状态 | 关键交付 |
|-------|------|:--:|------|
| 0-6 | 基础架构 (PTX 解析/执行/内存) | ✅ | ANTLR4 解析器 + IR + 解释执行 |
| 7 | Reconvergence Validation | ✅ | SIMT v2 收敛验证 |
| 8 | Performance Benchmark | ✅ | 基准性能数据 |
| 9 | SIMT Stack Integration | ✅ | Per-thread PC + CFG post-dominator |
| Phase 3-2026 | 结构债务修复 | ⏳ | A 系列 0 剩余；C 系列 18；D 系列 6 |
| 12.1 | PTXIR 二进制格式 | ✅ 2026-07-30 | ADR-0023 + ADR-0011 升级 |
| 12.1.1 | PTXIR Image Executor (Phase 1) | ✅ 2026-08-10 | ADR-0029 + `libptxemu_device.so` + `cpptlm_module.h` |
| 12.2 | PTXIR Cubin 集成 | ✅ 2026-08-10 | ADR-0024 v1.1 + OpenSpec change `2026-08-10-ptxir-cubin-cleanup` |
| **12.3** | **PTXIR Driver API front door + 缺失 CLI 工具** | **✅ 2026-08-11 ship** | **`ptxir-driver-api-front-door` (A 部分) + multi-kernel-manifest-adr-0028 解除 BLOCKING** |
| **12.4** | **ADR-0028 多 kernel manifest** | **✅ 2026-08-11 ship** | **schema + backward-compat + 文档同步；runtime multi-entry handle ⏳ Phase 12.5** |
| **13** | **HAL extension 跨仓协作** | **✅ 2026-08-11 ship** | **`hal-extension-ptxemu-usrlinu-emu-taskrunner`** |

---

## 实施状态审计（2026-08-11 更新）

> 对照 [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) v1.4 §2 Components 表、§4 Runtime data flow、§11 Related ADRs 与代码/构建产物逐项核验。

| 类别 | 已实现 | 未实现/缺失 | 比例 |
|------|--------|------------|------|
| **构建工具** (tools) | 2 (`ptxir_embed`, `ptxir_extract`) | 2 (`ptx-nvcc`, `ptxir_build`) | 50% |
| **运行时库** (libs) | 3 (`libcudart.so.12`, `libptxemu_device.so`, `libcpptlm_core.a`) | 0 | 100% |
| **ABI Headers** | 3 (`cpptlm_bridge.h`, `cpptlm_module.h`, `cuda_driver.h`) | 0 | 100% |
| **Cudart Driver API** | 4 (`__cudaRegisterFatBinary` + `cuModuleLoadData` + `cuModuleGetFunction` + `cuModuleUnload` + `cuLaunchKernel`) | 0 | 100% |
| **ADR 文档** | 6 (0024/0025/0026/0027/0028/0029) | 0 | 100% |
| **Multi-kernel schema** | ✅ `ManifestSection.kernels` + `PTXIR_VERSION=4` + backward-compat synthesis | 0 (4 P0 gaps deferred to Phase 12.5) | 100% |
| **HAL extension** | ✅ `ptxemu_image_*` 5 ABI + cross-repo 3 ioctl (0x27/0x28/0x29) + `IGpuDriver` 3 methods | 0 | 100% |

### 关键差距（驱动后续 Phase 12.5）

1. **🟠 Multi-entry handle API 未实现** — `cpptlm_module.cpp:111` 仍用 `kernels[0]` fallback；`cuModuleGetFunction` 缺多 kernel name→handle table；4 P0 gaps 详见 [multi-kernel-manifest-gaps-gap-analysis §3](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md)
2. **🟠 v2 PTXIR writer 缺失** — multi-entry 写入能力（架构 §3 §10 item 10）— Phase 12.5 #3
3. **🟠 Multi-entry PTXIR fixture 缺失** — 真实多 kernel PTX 测试源（Phase 12.5 #4）
4. **🟡 `ptx-nvcc` + `ptxir_build` 工具缺失** — §3 Build-time data flow + §10 items 1-7 不可端到端执行（Phase 12.3.B/C 仍待启动）
5. **🟡 KernelEntry 数据冗余** — `arg_count`/`arg_byte_size` vs `ManifestParam` vector 双重 source of truth（Phase 12.5 #8 doc clarification）

---

## Phase 12.2 PTXIR Cubin 集成 ✅ 2026-08-10 ship

### 目标

依据 [ADR-0024 v1.1](docs/adr/ADR-0024-ptxir-cubin-embed-extension.md) (2026-08-07 amendment)，将 PTXIR 嵌入到最终可执行文件末尾（ELF 容忍尾部 overlay data），使 PTX-EMU 能从 embed 段反序列化 PTXIR 并复用 `set_ptx_context()` 主路径，同时保留 NVIDIA 工具链兼容性（cub level 工具独立支持）。

### ✅ Shipped 2026-08-10

| Commit | 内容 | 状态 |
|--------|------|:--:|
| Commit 0 | ADR-0024 v1.1 amendment (footer layout + magic literal change) | ✅ 2026-08-07 |
| OpenSpec skeleton | `openspec/changes/2026-08-10-ptxir-cubin-cleanup/` (proposal.md + tasks.md + spec.md) | ✅ `20ad752b` |
| R3 (核心 fix) | `try_ptxir_dispatch_from_memory()` helper + `__cudaRegisterFatBinary` 重构 — 区分 "no footer → fallback OK" vs "footer present + malformed → 报告错误 (NOT 静默 fallback)"，per 架构 §4.1 + ADR-0024 acceptance #6 | ✅ `b5d96c33` |
| R5 (Oracle scenarios) | `e2e_cuobjdumpDumpSass_directOnEmbeddedCubin_succeeds` + `e2e_cuobjdumpDumpPtx_afterExtract_succeeds` — 验证 NVIDIA `cuobjdump` 对 trailing PTXIR section + footer 的容忍性（ADR-0024 §风险 risk 1） | ✅ `50f41982` |
| Archive | `chore(openspec): archive 2026-08-10-ptxir-cubin-cleanup` | � pending R6.5 |

### 关键约束

- `PTXIR_MODE` 默认 OFF → 字节级兼容现状 ✅ (verified via `unit_ptxir_config` PASS)
- `PTXIR_EMBED_MAGIC = {'P','T','X','E','M','B','\x01','\x00'}` — 已 2026-08-07 ADR amendment
- byte source = `/proc/self/exe` 末尾（非 `fat_bin` 参数 — dead parameter）
- v1 显式为 single-kernel scope（PTXIR v3 限制；Phase 12.4 ADR-0028 解除）

### ✅ 收尾任务完成情况

| # | 任务 | 实际状态 |
|---|------|---------|
| 12.2-R1 | `PTXIRLoader::extractPureCubin` 测试覆盖补齐 | ✅ 已有完整测试（14 场景含 extractPureCubin 3 场景，PASS） |
| 12.2-R2 | INI 集成到 `initialize_environment()` | ✅ `setPTXIRModeFromIni` 已 ship (`cudart_sim.cpp:245`)，4 测试 PASS |
| 12.2-R3 | `__cudaRegisterFatBinary` PTXIR dispatch 分支 | ✅ **核心 fix shipped** (`b5d96c33`)：`try_ptxir_dispatch_from_memory` helper + refactored switch on `PtxirDispatchStatus`，malformed PTXIR/manifest 不再静默 fallback |
| 12.2-R4 | integration tests | ✅ 6 + 4 新增 R3 测试 PASS（`test_ptxir_cubin_loader.cpp`） |
| 12.2-R5 | e2e tests（Oracle review 2 scenarios） | ✅ 2 新增 scenarios PASS（`e2e_cuobjdumpDumpSass_directOnEmbeddedCubin_succeeds` + `e2e_cuobjdumpDumpPtx_afterExtract_succeeds`） |
| 12.2-R6 | 完整 ctest + sanity.sh 全绿 + ABI 字节级不变 | ✅ ABI 验证通过（40 public cudart symbols 全保留，仅地址偏移），`PTXIR_MODE=off` 行为不变 |

### 关键约束满足

- ✅ `PTXIR_MODE` env var + INI + default precedence（架构 §6）
- ✅ malformed embedded PTXIR → 报告错误（架构 §4.1 + ADR-0024 acceptance #6）— **R3 fix 解决**
- ✅ manifest mismatch → 报告错误 — **R3 fix 解决**
- ✅ 缺少 footer fallback 到 cuobjdump（架构 §4.1）— `kNoFooter` 分支保留
- ✅ ABI 字节级兼容（仅地址偏移，无符号增删）

---

## Phase 12.3 PTXIR Driver API front door ✅ 2026-08-11 ship (12.3.A)

### ✅ Shipped 2026-08-11 (12.3.A 部分)

> **12.3.B (`ptxir_build` CLI) + 12.3.C (`ptx-nvcc` wrapper) 仍待启动** (见 [§下一步](#下一步执行顺序))

| Commit | 内容 | 状态 |
|--------|------|:--:|
| OpenSpec skeleton | `openspec/changes/ptxir-driver-api-front-door/` (proposal.md + tasks.md + spec.md) | ✅ |
| Module/Function 基础设施 | `ModuleRecord` + `FunctionRecord` + `ModuleRegistry` (含 `std::mutex` 线程安全) | ✅ |
| Driver API 入口 | `cuModuleLoadData` + `cuModuleGetFunction` + `cuLaunchKernel(CUfunction,...)` + `cuModuleUnload` 4 个新 T 符号 | ✅ |
| 6 类 image classifier | PTX / 独立 PTXIR / 尾部 suffix / NVIDIA cubin / fatbin / Tile IR | ✅ |
| 7 类 error mapping | UNSUPPORTED_IMAGE / MALFORMED_PTX / MALFORMED_PTXIR / 3 种 INVALID_HANDLE / NOT_FOUND | ✅ |
| Tests | unit (`test_module_registry` + `test_module_registry_mode_independence`) + integration (`test_cuda_driver_api` + `test_in_memory_mutation` + `test_ptxir_cubin_loader`) + divergence (`test_post_barrier_two_halves`) | ✅ |
| D3 mutation bug 复检 | per-launch fresh `PtxContext` + 1000 次顺序 launch 确定性测试 | ✅ |
| ABI 稳定性回归 | `cpptlm_bridge.h` diff 为空 + `CPPTLMBRIDGE_VERSION=2` 保持 | ✅ |

### 设计目标（已 ship）

补齐 [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) §2 Components 表列出的 in-memory module loading front door（Driver API）+ §3 Build-time data flow 所需的两个 CLI 工具，使 §4.2 in-memory data flow + §10 acceptance items 1-9 全部可达。

### 关联 ADR

- [ADR-0025](docs/adr/ADR-0025-ptxir-build-cli.md) — `ptxir_build` CLI 设计
- [ADR-0027](docs/adr/ADR-0027-ptx-nvcc-wrapper.md) — `ptx-nvcc` wrapper 设计
- [ADR-0029 §4.2](docs/adr/ADR-0029-ptxemu-image-executor.md) — in-memory Driver API 边界
- [ADR-0024](docs/adr/ADR-0024-ptxir-cubin-embed-extension.md) — 复用 `PTXIRLoader` + `PtxContextAdapter`

### 子 Phase

#### 12.3.A — Driver API front door（in-memory module loading on libcudart.so）

**目标**: 在 `libcudart.so` 实现 `cuModuleLoadData` / `cuModuleGetFunction` / `cuLaunchKernel(CUfunction,...)` / `cuModuleUnload`，复用 `PTXIRLoader` + `PtxContextAdapter`，与 `libptxemu_device.so` 路径解耦但执行后端共享。

| # | 任务 | 文件路径 | 关联 |
|---|------|----------|------|
| 12.3.A1 | 新建 `ModuleRecord` + `FunctionRecord` + `ModuleRegistry`（不透明 handle + 深拷贝 image bytes + eager parse + **`std::mutex` 线程安全** —— Driver API 可从多 host thread 调用） | 新文件 `include/cudart/module_registry.h` + `src/cudart/module_registry.cpp`（`cuda_driver.h` 是内存分配器，**不混合职责**） | 架构 §5.2 §5.3 |
| 12.3.A2 | 实现 `cuModuleLoadData(CUmodule*, const void*)`：eager parse（不 lazy）+ image bytes deep copy（caller pointer 不作为 handle 存活） | `src/cudart/cudart_sim.cpp` 新增入口；调用 `PTXIRLoader::deserializeForCubin()` + `PtxContextAdapter::fromEmbedded()` | 架构 §4.2 §5.3 |
| 12.3.A3 | 实现 `cuModuleGetFunction(CUfunction*, CUmodule, const char*)` 真版本（替换现有 stub at line 514-521） | `src/cudart/cudart_sim.cpp` | 架构 §5.2 `FunctionRecord` |
| 12.3.A4 | 实现 `cuLaunchKernel(CUfunction, ...)` Driver API 版本 | `src/cudart/cudart_sim.cpp` 新增入口；复用现有 `cudaLaunchKernel` 主路径 | 架构 §5.3 |
| 12.3.A5 | 实现 `cuModuleUnload(CUmodule)` | `src/cudart/cudart_sim.cpp`；释放 `ModuleRecord` + 失效关联 function handles（busy 时返回 `CUDA_ERROR_INVALID_HANDLE`） | 架构 §5.3 §7 + §10 item 24 |
| 12.3.A6 | **6 类** image classifier（per 架构 §5.1）：PTX text (SUPPORTED) / standalone PTXIR (SUPPORTED) / executable-tail PTXIR suffix (REJECTED→defer legacy) / NVIDIA cubin (NOT SUPPORTED→INVALID_IMAGE) / NVIDIA fatbin (NOT SUPPORTED→INVALID_IMAGE) / Tile IR (NOT SUPPORTED→INVALID_IMAGE) | 新文件 `src/cudart/image_classifier.cpp`（cudart_sim.cpp 已 ~1478 行，分类器纯函数易单测） | 架构 §5.1 |
| 12.3.A7 | Error mapping **7 类** (per 架构 §7 table)：UNSUPPORTED_IMAGE / MALFORMED_PTX / MALFORMED_PTXIR / UNKNOWN_MODULE_HANDLE / UNKNOWN_FUNCTION_HANDLE / MISSING_KERNEL_SYMBOL / STALE_FUNCTION_HANDLE | `src/cudart/cudart_sim.cpp` | 架构 §7 |
| 12.3.A8 | unit tests：**≥13 测试** = 6 (A6 image classes) + 5 (架构 §10 item 11: cubin→INVALID_IMAGE / malformed PTX→INVALID_PTX / unknown module→INVALID_HANDLE / missing symbol→NOT_FOUND / stale handle→INVALID_HANDLE) + 2 (stale module/function handle 边界) | `tests/unit/cudart/test_module_registry.cpp`（新建） | 架构 §5.1 §7 §10 item 11 |
| 12.3.A8b | **§10 item 12 verification**: `PTXIR_MODE=off` 不 disable in-memory module loading（独立 precedence） | `tests/unit/cudart/test_module_registry_mode_independence.cpp`（新建） | 架构 §10 item 12 |
| 12.3.A9 | integration tests: 端到端 `cuModuleLoadData` → `cuLaunchKernel` → `cuModuleUnload` | `tests/integration/test_cuda_driver_api.cpp`（新建） | 架构 §10 items 8-9 |
| 12.3.A10 | 验证 nm -D libcudart.so **导出 4 个新符号**：`cuModuleLoadData` + `cuModuleGetFunction` + `cuLaunchKernel(CUfunction,...)` + `cuModuleUnload` | `nm -D build/lib/libcudart.so \| grep -E "cu(ModuleLoadData\|ModuleGetFunction\|ModuleUnload\|LaunchKernel)" \| grep " T "` | acceptance gate |
| 12.3.A11 | **D3 mutation bug 复检**：per-launch fresh `PtxContext` + 不缓存 `kernelStatements`（架构 §5.4 + ADR-0029 D3）| `src/cudart/cudart_sim.cpp` + `tests/integration/test_in_memory_mutation.cpp`（新建）：(a) 同 bytes 两次 deserialize→byte-identical；(b) 顺序 launch 1000 次不同 blockDim→输出确定无累积；(c) image bytes hash 经 N 次 launch 不变 | 架构 §5.4 + ADR-0029 D3 |
| 12.3.A12 | **ABI 稳定性回归**：§10 item 13/21 gates（`git diff cpptlm_bridge.h` 为空 + `CPPTLMBRIDGE_VERSION` 保持 2） | `tests/integration/test_cpptlm_bridge_unchanged.cpp`（新建或扩展 `test_phase0_byte_identical_gates.cpp`） | 架构 §10 items 13/21 + ADR-0029 D7 |

**关键约束 (MUST)**:
- 调用方提供的 image bytes 必须 deep copy（架构 §5.3）；caller-owned pointer 在调用返回后不作为 handle 存活
- 不读取 `/proc/self/exe`，不使用 executable-tail probe，不回退到 `cuobjdump`（架构 §4.2 in-memory 路径独立）
- 与 `libptxemu_device.so` 的 `ptxemu_image_*` 路径解耦，但执行后端 `PtxInterpreter` / `GPUContext` 共享
- 不修改 `cpptlm_bridge.h` ABI（ADR-0029 D7 5 byte-identical gates 继续 hold）

#### 12.3.B — `ptxir_build` CLI（ADR-0025 实施）

**目标**: 实现 `.ptx` → `.ptxir` 转换工具，复用 PTXIR writer 子系统。

| # | 任务 | 文件路径 | 关联 |
|---|------|----------|------|
| 12.3.B1 | 新建 `tools/ptxir_build.cpp`（CLI 入口，约 100-150 行） | `tools/ptxir_build.cpp` | ADR-0025 §CLI |
| 12.3.B2 | `tools/CMakeLists.txt` 注册 `ptxir_build` target + 链接 `ptxir_writer` | `tools/CMakeLists.txt` | [tools/CMakeLists.txt](tools/CMakeLists.txt) |
| 12.3.B3 | Exit code 契约: 0 成功, 1 参数错误, 2 PTX 数据错误, 3 I/O 失败 | `tools/ptxir_build.cpp` | 架构 §8 |
| 12.3.B4 | unit tests (exit code 0/1/2/3) | `tests/unit/tools/test_ptxir_build.cpp`（新建） | 架构 §10 item 1 |
| 12.3.B5 | e2e test: `cute_rmsnorm.ptx` → `cute_rmsnorm.ptxir` + roundtrip | `tests/e2e/test_ptxir_build_e2e.sh`（新建） | 架构 §10 item 1 |

**关键约束 (MUST)**:
- 单 kernel 限制（per ADR-0028 BLOCKING）；自动检测不到 `.entry` 或检测到多个 entry 时退出码 2
- 不修改 PTXIR 二进制格式（仅消费输入 + 产出输出）
- `--kernel-name` 显式参数 + `--out` 输出路径强制要求

#### 12.3.C — `ptx-nvcc` wrapper（ADR-0027 实施）

**目标**: 编排 nvcc → cuobjdump → ptxir_build → ptxir_embed 完整 build-time data flow。

| # | 任务 | 文件路径 | 关联 |
|---|------|----------|------|
| 12.3.C1 | 新建 `tools/ptx-nvcc` Python 3 wrapper（`#!/usr/bin/env python3`，约 300-400 行） | `tools/ptx-nvcc` | ADR-0027 §wrapper |
| 12.3.C2 | wrapper 流程: nvcc compile-only → nvcc link → cuobjdump --ptx → ptxir_build → ptxir_embed → cleanup | `tools/ptx-nvcc` | 架构 §3 |
| 12.3.C3 | 显式临时目录 + 文件名（不使用 shell wildcard） | `tools/ptx-nvcc` | 架构 §3 |
| 12.3.C4 | `--no-embed` / `--kernel-name` / `--ptxemu-root` / nvcc passthrough 参数 | `tools/ptx-nvcc` | 架构 §9 |
| 12.3.C5 | DT_RUNPATH 注入（`--ptxemu-root` 路径） | `tools/ptx-nvcc` | 架构 §10 item 5 |
| 12.3.C6 | unit tests (subprocess mock + exit code propagation) | `tests/unit/tools/test_ptx_nvcc.py`（新建） | 架构 §10 items 3-4 |
| 12.3.C7 | e2e test: end-to-end compile, link, embed, run | `tests/e2e/test_ptx_nvcc_e2e.sh`（新建） | 架构 §10 item 7 |

**关键约束 (MUST)**:
- 单 kernel 限制（per ADR-0028 BLOCKING）
- 任何步骤失败进入 cleanup 路径；单 kernel 失败不留下临时文件
- `PTXIR_MODE` 默认 auto（per ADR-0026）
- Linux/POSIX only（架构 §9）
- nvcc passthrough：除 wrapper 自有选项外，参数按原顺序透传

### 12.3 实施顺序建议

```
12.3.A1 (ModuleRecord 基础设施)
  ↓
12.3.A2-A7 (4 个 Driver API 入口 + classifier + error mapping) ← 并行可分解为 2-3 sub-commits
  ↓
12.3.A8-A10 (tests + nm verify)
  ↓ (与 A 并行)
12.3.B1-B5 (ptxir_build CLI)
  ↓ (与 B 完成后)
12.3.C1-C7 (ptx-nvcc wrapper — 依赖 B 完成)
```

**commit 拆分** (per `ptx-lessons-learned` §3，每个 Phase 独立可回退):
- Commit 1: ModuleRecord/FunctionRecord/Registry + unit tests（含 `std::mutex` 线程安全）
- Commit 2: cuModuleLoadData + cuModuleGetFunction + 6 类 image classifier
- Commit 3: cuLaunchKernel(CUfunction) + cuModuleUnload + error mapping
- Commit 4: D3 mutation bug 复检测试 + ABI 稳定性回归 gates
- Commit 5: integration tests + nm -D verify（4 个新 T 符号）
- Commit 6: ptxir_build CLI + e2e tests
- Commit 7: ptx-nvcc wrapper + e2e tests

**预计 OpenSpec change 命名**: `2026-08-10-ptxir-toolchain-completion`（umbrella，与审计同日建立；含 12.3.A 7 子 commits + 12.3.B/C）

---

## Phase 12.4 ADR-0028 多 kernel manifest ✅ 2026-08-11 ship (schema + backward-compat)

### ✅ Shipped 2026-08-11

> **runtime multi-entry handle API 显式延后到 Phase 12.5** (per [multi-kernel-manifest-gaps-gap-analysis](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md) §3 差距 #1, #2)

| Commit | 内容 | 状态 |
|--------|------|:--:|
| OpenSpec skeleton | `openspec/changes/multi-kernel-manifest-adr-0028/` (proposal.md + tasks.md + spec.md) | ✅ `99b23223` |
| ADR-0028 创建 | `docs/adr/ADR-0028-multi-kernel-manifest.md` (多 kernel manifest 设计 + BLOCKING DEPENDENCY 升级) | ✅ `e5fe7f2a` |
| `ptxir_format.h` 扩展 | `KernelEntry` struct + `ManifestSection.kernels` + `PTXIR_VERSION: 3→4` | ✅ `05504d0c` |
| BLOCKING DEPENDENCY 升级 | ADR-0028 v2 + ptxir-toolchain-stack.md v1.3 §11 | ✅ `cd277e13` |
| Backward-compat synthesis | reader 端 `kernels` 空 + `kernel_name` 非空时 synthesize 单 entry | ✅ |
| Tests (placeholder) | `tests/unit/cudart/test_multi_kernel_selection.cpp` (结构性占位符，deferred to Phase 12.5) | ✅ `c6ac1176` |
| E2E multi-kernel drain | co-sim advance ceiling contract + multi-kernel drain tests | ✅ `79617fde` |
| ABI baseline 重生成 | `c46bdfcc` (PTXIR_VERSION=4 后 stale baseline) | ✅ |
| 文档同步 | `ptxir-toolchain-stack.md` v1.4 + ADR-0025/0027/0029 §v1 限制更新 + ADR README 索引 | ✅ `b801837b` |
| Archive | `chore(openspec): archive multi-kernel-manifest-adr-0028` | ✅ `f4a95f2c` |

### 设计目标（已 ship）

新建 ADR-0028 + bump `PTXIR_VERSION`（per ADR-0023 Extend-Only 原则），扩展 `ManifestSection` 为 `vector<kernel_entry>`，解除 ADR-0025/0027/0029 §v1 单 kernel 限制。

### 关联 ADR

- [ADR-0028](docs/adr/ADR-0028-multi-kernel-manifest.md) — **已 ship 2026-08-11**
- [ADR-0023](docs/adr/ADR-0023-ptxir-binary-format.md) §Extend-Only 原则
- 阻塞依赖: ADR-0025 §v1、ADR-0027 §v1、ADR-0029 D4 §v1 — **已全部更新**

### 任务（已 ship + deferred to Phase 12.5）

| # | 任务 | 文件路径 | 状态 |
|---|------|----------|:--:|
| 12.4-1 | 新建 ADR-0028 | `docs/adr/ADR-0028-multi-kernel-manifest.md` | ✅ |
| 12.4-2 | 扩展 `ManifestSection` 为 `vector<kernel_entry>` | `include/ptx_ir/ptxir_format.h` | ✅ |
| 12.4-3 | bump `PTXIR_VERSION` 3→4 + backward-compat | `include/ptx_ir/ptxir_format.h` | ✅ |
| 12.4-4 | `PTXIRLoader::deserializeForCubin()` 多 entry 返回 | `src/cudart/ptxir_loader.cpp` | ✅ |
| 12.4-5 | `ptxir_build` / `ptxir_embed` / `ptxir_extract` 多 kernel 支持 | `tools/ptxir_*` | ⏳ Phase 12.5 |
| 12.4-6 | unit + integration + e2e tests (multi-kernel selection) | `tests/*/test_*_kernel*.cpp` | ⚠️ placeholder (Phase 12.5) |
| 12.4-7 | `PtxEmuImageExecutor::load_image` 多 entry handle 解析 | `src/cudart/cpptlm_module.cpp` | ⚠️ `kernels[0]` fallback (Phase 12.5) |
| 12.4-8 | `__cudaRegisterFatBinary` / `cuModuleGetFunction` 多 kernel 名查询 | `src/cudart/cudart_sim.cpp` | ⏳ Phase 12.5 |
| 12.4-9 | ADR-0025/0027/0029 §v1 限制段落更新 | `docs/adr/ADR-002*.md` | ✅ |
| 12.4-10 | adr/README.md 索引同步 | `docs/adr/README.md` | ✅ |

**关键约束满足**:
- ✅ backward-compat: 旧 v1 单 kernel binary 在 ADR-0028 后运行时仍可加载（manifest 格式向后可读）
- ✅ bump `PTXIR_VERSION` 遵循 ADR-0023 Extend-Only 原则
- ✅ `ptxir-toolchain-stack.md` 已升级到 v1.4（含 §11 BLOCKING 解除 + v2 状态段落）

---

## Phase 12.5 Multi-entry handle API ⏳ 显式延后 (per gap analysis)

> **来源**: [docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md) §3 差距清单（4 P0 + 2 P1 + 2 P2）

### 目标

完成 Phase 12.4 ship 后**延后**的 runtime multi-kernel 处理能力:
- v2 PTXIR writer（multi-entry 写入）
- `cuModuleGetFunction` 多 kernel name→handle 映射
- `cpptlm_module.cpp` full multi-entry handle API（替换 `kernels[0]` fallback）
- 真实 multi-entry PTXIR fixture + e2e 验证

### 关联 ADR

- [ADR-0028](docs/adr/ADR-0028-multi-kernel-manifest.md) Decision 1 (extend-only 字段预留 `ptx_version`/`sm_target`)
- [ADR-0029 §D4](docs/adr/ADR-0029-ptxemu-image-executor.md) — `ptxemu_image_kernel_name` 暴露多 kernel 名

### 任务（4 P0 + 2 P1 + 2 P2）

| # | 任务 | 优先级 | 状态 | 差距引用 |
|---|------|:--:|:--:|------|
| 12.5-1 | **v2 PTXIR writer**: multi-entry 写入路径 + `kernel_name` 首个 + `kernels` vector 全部 | P0 | ⏳ | §3 #3 |
| 12.5-2 | **Multi-entry PTXIR fixture**: `tests/fixtures/ptx/multi_kernel_*.ptx` + generator 脚本 | P0 | ⏳ | §3 #4 |
| 12.5-3 | **`cuModuleGetFunction` name→handle 映射**: `cudart_sim.cpp` 添加 `cuFunction` 注册表 | P0 | ⏳ | §3 #2 |
| 12.5-4 | **`cpptlm_module.cpp` full multi-entry handle API**: 替换 `kernels[0]` fallback | P0 | ⏳ | §3 #1 |
| 12.5-5 | **`test_multi_kernel_selection.cpp` 升级**: 替换 `SUCCEED("placeholder")` 为实际 fixture 测试 | P1 | ⏳ | §3 #5 |
| 12.5-6 | **`ptxemu_image_kernel_name` 多 kernel 暴露**: 遍历 `kernels` vector API | P1 | ⏳ | §3 #6 |
| 12.5-7 | ABI baseline 回归验证: v1 binary 在新 runtime 加载 + mutation regression test | P2 | ⏳ | §3 #7 |
| 12.5-8 | `KernelEntry` 数据冗余文档化: `arg_count`/`arg_byte_size` vs `ManifestParam` source of truth | P2 | ⏳ | §3 #8 |

**关键约束 (MUST)**:
- 不修改 `cpptlm_bridge.h` ABI（per ADR-0029 D7）
- 不破坏 `libptxemu_device.so` 5 函数 ABI（仅扩展）
- 完整 round-trip 测试：v2 writer → multi-entry fixture → reader → cuModuleGetFunction
- backward-compat: v1 single-kernel binary 必须仍可加载（per ADR-0028 Decision 3）

**commit 拆分建议** (per `ptx-lessons-learned` §3):
- Commit 1: v2 PTXIR writer + 双向 round-trip unit tests
- Commit 2: Multi-entry fixture + generator 脚本 + e2e 验证
- Commit 3: cuModuleGetFunction handle 注册表 + tests
- Commit 4: cpptlm_module full multi-entry handle API + tests
- Commit 5: test_multi_kernel_selection 升级（替换 placeholder）
- Commit 6: ptxemu_image_kernel_name 扩展 + ABI 兼容性验证

**预计 OpenSpec change 命名**: `2026-XX-XX-multi-entry-handle-api-phase-12-5` (per OpenSpec lifecycle, Lesson §6)

---

## Phase 13 HAL extension 跨仓协作 ✅ 2026-08-11 ship

### ✅ Shipped 2026-08-11

| Commit | 内容 | 状态 |
|--------|------|:--:|
| OpenSpec skeleton | `openspec/changes/hal-extension-ptxemu-usrlinu-emu-taskrunner/` (proposal.md + tasks.md + spec.md) | ✅ |
| `libptxemu_device.so` SONAME / symbol 审计 | `nm -D` 验证 TaskRunner link 不冲突 | ✅ |
| `ptxemu_image_*` 5 函数 cross-ABI 兼容性验证 | `tests/integration/test_cpptlm_module_dlopen.cpp` 扩展 | ✅ |
| DL-isolated test | 无 libcudart.so 依赖测试 | ✅ |
| In-flight unload returns busy 验证 | `tests/integration/test_cpptlm_module_inflight.cpp` | ✅ |
| Archive | `chore(openspec): archive hal-extension-ptxemu-usrlinu-emu-taskrunner` | ✅ |

### 跨仓依赖（已 ship）

| 仓 | 任务 | 状态 |
|---|------|:--:|
| UsrLinuxEmu | 新增 3 个 ioctl: `GPU_IOCTL_LOAD_KERNEL_MODULE/LAUNCH_KERNEL_MODULE/UNLOAD_KERNEL_MODULE`（**0x27/0x28/0x29**，System C magic 'G' 8-bit 范围修正） | ✅ |
| UsrLinuxEmu | 新增 3 个 HAL fn-ptr #66/#67/#68 (`kernel_module_load/execute/unload`) | ✅ |
| UsrLinuxEmu | `hal_user.cpp` 新增 dlsym `libptxemu_device.so` 的 `ptxemu_image_*` 实现 | ✅ |
| TaskRunner | `libcuda_shim` 实现 `cuModuleLoadData`/`cuLaunchKernel`/`cuModuleUnload` 经 IGpuDriver | ✅ |
| TaskRunner | `IGpuDriver` 新增 3 个纯虚方法 (`load_kernel_module`/`launch_kernel_module`/`unload_kernel_module`) | ✅ |

### 设计目标（已 ship）

执行 [ADR-0029 §D8](docs/adr/ADR-0029-ptxemu-image-executor.md) HAL 方案 D8，在 UsrLinuxEmu + TaskRunner 仓添加 GPU 驱动 ioctl + IGpuDriver 扩展，使 `cuModuleLoadData` / `cuLaunchKernel` / `cuModuleUnload` 在 TaskRunner 端通过 HAL 边界间接调 PTX-EMU `libptxemu_device.so`。

### 关键约束满足

- ✅ **TaskRunner 仓零 PTX-EMU 链接依赖**（架构 §2 D8.1）— 所有 PTX-EMU 调用经 UsrLinuxEmu HAL 边界封装
- ✅ 跨仓 commit 顺序 per ADR-035 R5.1（UsrLinuxEmu ioctl → UsrLinuxEmu HAL → PTX-EMU 兼容性 → TaskRunner）
- ✅ PTX-EMU 仓 `libptxemu_device.so` 5 ABI 入口稳定；外部仓适配不阻塞本仓主线

### 目标

执行 [ADR-0029 §D8](docs/adr/ADR-0029-ptxemu-image-executor.md) HAL 方案 D8，在 UsrLinuxEmu + TaskRunner 仓添加 GPU 驱动 ioctl + IGpuDriver 扩展，使 `cuModuleLoadData` / `cuLaunchKernel` / `cuModuleUnload` 在 TaskRunner 端通过 HAL 边界间接调 PTX-EMU `libptxemu_device.so`。

### 跨仓依赖

| 仓 | 任务 | 文件路径（外部） | 关联 |
|---|------|----------|------|
| UsrLinuxEmu | 新增 3 个 ioctl: `GPU_IOCTL_LOAD_KERNEL_MODULE/LAUNCH_KERNEL_MODULE/UNLOAD_KERNEL_MODULE`（编号 39/40/41） | `UsrLinuxEmu/plugins/gpu_driver/drv/gpgpu_device.cpp` | UsrLinuxEmu AGENTS.md ADR-036 |
| UsrLinuxEmu | 新增 3 个 HAL fn-ptr #66/#67/#68 (`kernel_module_load/execute/unload`) | `UsrLinuxEmu/plugins/gpu_driver/hal/gpu_hal.h` | UsrLinuxEmu ADR-023 §D4 |
| UsrLinuxEmu | `hal_user.cpp` 新增 dlsym `libptxemu_device.so` 的 `ptxemu_image_*` 实现 | `UsrLinuxEmu/plugins/gpu_driver/hal/hal_user.cpp` | UsrLinuxEmu AGENTS.md ADR-023 |
| TaskRunner | `libcuda_shim` 实现 `cuModuleLoadData`/`cuLaunchKernel`/`cuModuleUnload` 经 IGpuDriver | `TaskRunner/src/umd/libcuda_shim/cu_module.cpp` + `cu_launch.cpp` | TaskRunner ADR-035 |
| TaskRunner | `IGpuDriver` 新增 3 个纯虚方法 (`load_kernel_module`/`launch_kernel_module`/`unload_kernel_module`) | `TaskRunner/include/shared/igpu_driver.hpp` | TaskRunner TADR-301 |

### PTX-EMU 仓任务

| # | 任务 | 文件路径 | 关联 |
|---|------|----------|------|
| 13-1 | `libptxemu_device.so` SONAME / symbol 审计（TaskRunner link 不冲突） | `nm -D build/lib/libptxemu_device.so` | 架构 §10 item 22 |
| 13-2 | `ptxemu_image_*` 5 函数 cross-ABI 兼容性验证 | `tests/integration/test_cpptlm_module_dlopen.cpp`（已存在，扩展） | 架构 §10 item 20 |
| 13-3 | DL-isolated test（无 libcudart.so 依赖） | 同上 | 架构 §10 item 20 |
| 13-4 | in-flight unload returns busy 验证 | `tests/integration/test_cpptlm_module_inflight.cpp`（已存在） | 架构 §10 item 24 |

### 关键约束 (MUST)

- **TaskRunner 仓零 PTX-EMU 链接依赖**（架构 §2 D8.1）— 所有 PTX-EMU 调用经 UsrLinuxEmu HAL 边界封装
- 跨仓 commit 顺序 per ADR-035 R5.1（建议: UsrLinuxEmu ioctl → UsrLinuxEmu HAL → PTX-EMU 兼容性 → TaskRunner）
- PTX-EMU 仓只需保证 `libptxemu_device.so` 5 ABI 入口稳定；外部仓适配不阻塞本仓主线

### 实施顺序

**跨仓协调** — 不在 PTX-EMU 仓独立完成；需建立跨仓 RFC + ADR-035 R5.1 协调流程。建议在 Phase 12.3 + 12.4 ship 后启动。

---

## 当前任务（Phase 10: Documentation & Release β）

### 🔴 阻塞项

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| RD-1 | 创建 root 级 roadmap.md (本文件) | ✅ 2026-07-23 | arch-done 门控 |
| RD-2 | 初始化 `.rddf/state/` + `.arch-handoff.json` | ✅ 2026-07-23 | arch-done 门控 |
| RD-3 | ADR 重命名为 `ADR-NNNN` 格式 | ✅ 2026-07-23 | rdd-workflow 合规 |
| RD-4 | CppTLM unified build ADR-0022 签署 | ✅ 2026-07-23 | CppTLM Oracle 审查 |
| RD-5 | Phase 12.2 governance check (ADR-0024 magic + layout) | ✅ 2026-08-07 | §合规检查 #6 |
| RD-6 | **Phase 12.2 收尾 (R1-R6)** | ✅ **2026-08-10 ship** | 本文件 §Phase 12.2 |
| RD-7 | **Phase 12.3 启动 (Driver API front door)** | ✅ **2026-08-11 ship (12.3.A)** | 本文件 §Phase 12.3 |
| RD-8 | **Phase 12.4 启动 (ADR-0028 BLOCKING)** | ✅ **2026-08-11 ship (schema + backward-compat)** | 本文件 §Phase 12.4 |
| RD-9 | **Phase 13 启动 (HAL extension 跨仓协作)** | ✅ **2026-08-11 ship** | 本文件 §Phase 13 |
| RD-10 | **Phase 12.5 延后登记 (multi-entry handle API)** | 📋 **2026-08-11 延后 (per gap analysis)** | 本文件 §Phase 12.5 |

### 🟡 进行中

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| C-* | C 系列代码债务 (18 项) | ⏳ | [post-phase3-debt-roadmap §1.2](docs/roadmap/post-phase3-debt-roadmap.md) |
| D-* | D 系列文档债务 (6 项) | ⏳ | [post-phase3-debt-roadmap §1.3](docs/roadmap/post-phase3-debt-roadmap.md) |
| P12.2 | PTXIR Cubin 集成（5 commits + R3 核心 fix + R5 Oracle scenarios） | ✅ **2026-08-10 ship** | [archive/2026-08-10-ptxir-cubin-cleanup/](openspec/changes/archive/2026-08-10-ptxir-cubin-cleanup/) |
| P12.2-R* | Phase 12.2 收尾 (6 项 R1-R6) | ✅ **2026-08-10 完成** | 本文件 §Phase 12.2 收尾任务 |

### 🟢 计划

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| H5 | Hopper/Blackwell tcgen05 后续 | 📋 | [ADR-0016](docs/adr/ADR-0016-blackwell-only-tcgen05.md) |
| S1 | 符号覆盖 CI 测试 | 📋 | [ADR-0022](docs/adr/ADR-0022-cpptlm-unified-build.md) |
| S2 | cpptlm_core_minimal 拆分 | 📋 | ADR-0022 远期优化 |
| P12.3.B | `ptxir_build` CLI (ADR-0025) | 📋 **🟠 P1** | 本文件 §Phase 12.3.B |
| P12.3.C | `ptx-nvcc` wrapper (ADR-0027) | 📋 **🟠 P1** | 本文件 §Phase 12.3.C |
| **P12.5** | **Multi-entry handle API (4 P0 + 2 P1 + 2 P2 gaps)** | 📋 **🟠 P1 (per gap analysis)** | 本文件 §Phase 12.5 + [multi-kernel-manifest-gaps-gap-analysis](docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md) |
| Future-1 | `$ORIGIN` 相对路径 | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-2 | CMake wrapper integration (`ptxemu_add_executable()`) | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-3 | macOS / Windows 支持 | 📋 低 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-4 | `cuInit` / `cuCtx*` context management | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-5 | Packed `extra` argument buffer | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |

---

## 下一步（执行顺序）

```
✅ 已 ship（2026-08-10 / 2026-08-11）:
  1. ~~Phase 12.2 收尾 R1-R6~~ → OpenSpec change `2026-08-10-ptxir-cubin-cleanup` archived
  2. ~~Phase 12.3.A Driver API front door~~ → OpenSpec change `ptxir-driver-api-front-door` archived
  3. ~~Phase 12.4 ADR-0028 multi-kernel manifest (schema + backward-compat)~~ → OpenSpec change `multi-kernel-manifest-adr-0028` archived
  4. ~~Phase 13 HAL extension 跨仓协作~~ → OpenSpec change `hal-extension-ptxemu-usrlinu-emu-taskrunner` archived

下一步:
  5. **Phase 14 fix-path-coverage-gaps** (🔴 P0 Oracle review) — 4-path cudart e2e coverage gaps
     → 改进提案已批 (2026-08-12 Oracle review, 5-section, 23 edits)
     → OpenSpec change `fix-path-coverage-gaps` 已创建 (P0, Phase 14, core-test, debt)
     → 7 task groups / 57 tasks / 5 Phases (Path 1B fatbinary e2e / Path 1C driver API / Path 2D image executor baseline / tests/e2e 重组织 / proposal 修正)
     → worktree 隔离执行，预计 5-7 个独立 commit (Phase 1/2/3/4/5 + 验收 + 归档)
     → 补齐 4-path cudart 测试覆盖率 3/4 → 4/4 (100%) + output-correctness 1/4 → 4/4 (100%)
  6. **Phase 12.5 Multi-entry handle API** (🟠 P1) — 4 P0 + 2 P1 + 2 P2 gaps (per gap analysis)
     → 改进提案先行（add-improve）→ guide-design 评审 → openspec → tasks
     → 预计 6 个独立 commit（writer → fixture → handle API → registry → test 升级 → API 暴露）
     → 阻塞下游 multi-kernel 工具链闭环
  7. **Phase 12.3.B `ptxir_build` CLI** (ADR-0025) — 直接 OpenSpec change
  8. **Phase 12.3.C `ptx-nvcc` wrapper** (ADR-0027) — 依赖 12.3.B
  9. **`ptxir-toolchain-stack.md` v1.4.1** — 同步 12.5 进展 + 移除多 kernel BLOCKING 历史标记

每个 Phase 后跑:
  - cmake --build build && ctest --output-on-failure
  - ./scripts/sanity.sh
  - ./scripts/regression.sh
  - 每 commit 后 git diff + git log 检查（per Lesson §3）
```

---

## Phase 14 fix-path-coverage-gaps (P0 Oracle review, in progress)

**目标**：补齐 4-path cudart e2e 测试覆盖率从 3/4 (75%) → 4/4 (100%)，修复 silent descoping。

**架构依据**：
- ADR-0024 (PTXIR-Embedded CUBIN) — Risk 1 验证未覆盖真实加载执行
- ADR-0029 (PTX-EMU Image Executor) — D6 SINGLE-GPU-INSTANCE 假设，无 output baseline
- ADR-0021 (CppTLM D1-Full MemoryBridge) — D-PTX-7/D-PTX-8 新债务
- `multi-kernel-manifest-gaps-gap-analysis` — 隐含测试覆盖债务

**5 Phase 工作**：
| Phase | 任务 | 状态 |
|-------|------|------|
| 1 | Path 1B PTXIR fat-binary 真实 e2e (`tests/e2e/path_1B_ptxir_fatbinary/`) | 📋 pending |
| 2 | Path 1C Driver API 真实 e2e (`tests/e2e/path_1C_driver_api/`) | 📋 pending |
| 3 | Path 2D Image Executor output baseline (`tests/e2e/path_2D_image_executor/`, `tests/ptxir/baselines/`) | 📋 pending |
| 4 | `tests/e2e/` 重组织为 4 个 path_X/ 子目录 | 📋 pending |
| 5 | 归档 `implement-ptxir-cubin-embed-extension` proposal disclaimer 修正 | 📋 pending |

**验收标准**：
- AC-M1: cudart 路径测试覆盖率 3/4 → 4/4 (100%)
- AC-M2: e2e output correctness 覆盖率 1/4 → 4/4 (100%)
- AC-M3: openspec 文档一致性修复 1 处
- AC-M4: `ctest -L path_1X` 可作为单路径回归命令

---

**维护者**: PTX-EMU Architecture Team
**日期**: 2026-08-12（v5: Phase 14 fix-path-coverage-gaps P0 Oracle review 加入 + Phase 12.5 顺延到第 6 位；v4: Phase 12.3/12.4/13 ship 状态同步 + Phase 12.5 多 entry handle API 延后登记 + 实施状态审计更新 + 引用 multi-kernel-manifest-gaps-gap-analysis；v3: 2026-08-10 Phase 12.2 收尾 ship；v2: 实施状态审计 + Phase 12.3/12.4/13 新增；v1: 2026-08-07）
