# PTX-EMU Roadmap

> **维护**: PTX-EMU Architecture Team
> **当前阶段**: Phase 10 — Documentation & Release（β 完成中）+ **Phase 12.2 PTXIR Cubin 集成 ✅ 2026-08-10 ship** + **Phase 12.3 PTXIR Driver API front door + 缺失 CLI 工具（待启动）** + **Phase 12.4 ADR-0028 多 kernel manifest（BLOCKING DEPENDENCY）** + Phase 13 HAL extension 跨仓协作（待启动）
> **最后更新**: 2026-08-10（实施状态审计 + 后续任务梳理）
> **关联**: [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) v1.3、[docs/roadmap/post-phase3-debt-roadmap.md](docs/roadmap/post-phase3-debt-roadmap.md)（详细债务清单）
> **参考**: [docs/README.md](docs/README.md)（文档索引）

---

## 当前状态

| 维度 | 数据 |
|------|------|
| ADR 数 | 27 个文件（ADR-0001~0027 + ADR-0029；**ADR-0017 + ADR-0028 双缺失**，其中 0028 BLOCKING） |
| OpenSpec 已归档 | 50+ 个（含 2026-08-07 / 2026-08-10 最新两个） |
| 活跃 changes | 0（无活跃 change；需建立 `2026-08-10-ptxir-toolchain-completion` 跟踪 Phase 12.3+） |
| 测试覆盖 | unit / integration / e2e 三层物理隔离 |
| PTX 语法测试 | `./tests/ptx/test_all_ptx.sh` 45/45 |
| CppTLM 集成 | D1-Full MemoryBridge 已归档（ADR-0021） |
| PTXIR Image Executor | ✅ `libptxemu_device.so` + `cpptlm_module.h`（ADR-0029 Phase 1 已 ship） |
| 最近审计 | **2026-08-10 实施状态审计**（本文件 §实施状态审计 section） |

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
| **12.2** | **PTXIR Cubin 集成** | **📋 2026-08-07 实施中** | **ADR-0024 v1.1 + OpenSpec change** |
| **12.3** | **PTXIR Driver API front door + 缺失 CLI 工具** | **🆕 待启动（🔴 P0）** | **ADR-0025/0027/0029 §4.2 实施补齐** |
| **12.4** | **ADR-0028 多 kernel manifest** | **🆕 待启动（🟠 BLOCKING P1）** | **ADR-0028 新建 + v1 单 kernel 限制解除** |
| **13** | **HAL extension 跨仓协作** | **🆕 待启动（🟠 P1）** | **UsrLinuxEmu + TaskRunner 仓联动** |

---

## 实施状态审计（2026-08-10）

> 对照 [docs/architecture/ptxir-toolchain-stack.md](docs/architecture/ptxir-toolchain-stack.md) v1.3 §2 Components 表、§4 Runtime data flow、§11 Related ADRs 与代码/构建产物逐项核验。本审计作为 OpenSpec change `2026-08-10-ptxir-toolchain-completion` 的 §实施背景 归档至 [openspec/changes/2026-08-10-ptxir-toolchain-completion/background/2026-08-10-audit.md](openspec/changes/2026-08-10-ptxir-toolchain-completion/background/2026-08-10-audit.md)（change 建立后）。

| 类别 | 已实现 | 未实现/缺失 | 比例 |
|------|--------|------------|------|
| **构建工具** (tools) | 2 (`ptxir_embed`, `ptxir_extract`) | 2 (`ptx-nvcc`, `ptxir_build`) | 50% |
| **运行时库** (libs) | 3 (`libcudart.so.12`, `libptxemu_device.so`, `libcpptlm_core.a`) | 0 | 100% |
| **ABI Headers** | 3 (`cpptlm_bridge.h`, `cpptlm_module.h`, `cuda_driver.h`) | 0 | 100% |
| **Cudart Driver API** | 1 (`__cudaRegisterFatBinary`) | 3 (`cuModuleLoadData`, `cuModuleUnload`, 真 `cuLaunchKernel(CUfunction,...)`) | 25% |
| **ADR 文档** | 5 (0024/0025/0026/0027/0029) | 1 (**ADR-0028** BLOCKING) | 83% |

### 关键差距（驱动后续 Phase）

1. **🔴 In-memory front door 未实现** — `cuModuleLoadData` / `cuModuleUnload` 在 `libcudart.so` 未导出；`cuModuleGetFunction` 仅 stub；架构 §4.2 / §5 / §10 items 8-12 不可达
2. **🔴 `ptx-nvcc` + `ptxir_build` 工具缺失** — §3 Build-time data flow + §10 items 1-7 不可端到端执行
3. **🟠 ADR-0028 BLOCKING DEPENDENCY 缺失** — 不解除则 v1 单 kernel 限制持续拖累 ADR-0025/0027/0029
4. **🟡 活跃 OpenSpec change tracker 未建立** — `openspec/changes/` 目录无活跃 change；archive 中 `2026-08-07-implement-ptxir-cubin-embed-extension` tasks 1.1-1.7 仍有大量 `[ ]` 未完成

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

## Phase 12.3 PTXIR Driver API front door + 缺失 CLI 工具（🆕 🔴 P0）

### 目标

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

## Phase 12.4 ADR-0028 多 kernel manifest（🆕 🟠 BLOCKING P1）

### 目标

新建 ADR-0028 + bump `PTXIR_VERSION`（per ADR-0023 Extend-Only 原则），扩展 `ManifestSection` 为 `vector<kernel_entry>`，解除 ADR-0025/0027/0029 §v1 单 kernel 限制。

### 关联 ADR

- **新 ADR-0028**（待创建）
- [ADR-0023](docs/adr/ADR-0023-ptxir-binary-format.md) §Extend-Only 原则
- 阻塞依赖: ADR-0025 §v1、ADR-0027 §v1、ADR-0029 D4 §v1

### 任务

| # | 任务 | 文件路径 | 关联 |
|---|------|----------|------|
| 12.4-1 | 新建 ADR-0028（多 kernel manifest + runtime selection 设计） | `docs/adr/ADR-0028-multi-kernel-manifest.md` | 架构 §11 BLOCKING DEPENDENCY |
| 12.4-2 | 扩展 `ManifestSection` 为 `vector<kernel_entry>` | `include/ptx_ir/ptxir_format.h:36-41`（**注意目录是 `ptx_ir` 不是 `ptxir`**，架构 §11 引用） | ADR-0028 §决策 |
| 12.4-3 | bump `PTXIR_VERSION` + 维护 backward-compat（v1 单 kernel binary 仍可加载） | `include/ptx_ir/ptxir_format.h` | ADR-0023 §决策 6 |
| 12.4-4 | 更新 `PTXIRLoader::deserializeForCubin()` 支持多 entry 返回 | `src/cudart/ptxir_loader.cpp` | ADR-0028 §决策 |
| 12.4-5 | 更新 `ptxir_build` / `ptxir_embed` / `ptxir_extract` 支持多 kernel | `tools/ptxir_*` | 架构 §3 §10 item 10 |
| 12.4-6 | unit + integration + e2e tests (multi-kernel selection) | `tests/unit/test_ptxir_loader.cpp` + `tests/integration/test_multi_kernel.cpp` + `tests/e2e/test_multi_kernel.cu` | 架构 §10 item 10 |
| 12.4-7 | 更新 `PtxEmuImageExecutor::load_image` 支持多 entry handle 解析 | `src/cudart/cpptlm_module.cpp` | ADR-0029 §D4 |
| 12.4-8 | 更新 `__cudaRegisterFatBinary` + `cuModuleGetFunction` 支持多 kernel 名查询 | `src/cudart/cudart_sim.cpp` | 架构 §4.1 §4.2 |
| 12.4-9 | ADR-0025/0027/0029 §v1 限制段落更新为 "等待 ADR-0028 解除" → 解除后改为 "已支持" | `docs/adr/ADR-002*.md` | 架构 §11 下游契约 #1 |
| 12.4-10 | adr/README.md 索引同步 | `docs/adr/README.md` | 维护规范 |

**关键约束 (MUST)**:
- backward-compat: 旧 v1 单 kernel binary 在 ADR-0028 后运行时仍可加载（manifest 格式向后可读）
- bump `PTXIR_VERSION` 必须遵循 ADR-0023 Extend-Only 原则
- 实施前必须先 amend `ptxir-toolchain-stack.md` v1.4（更新 §11 + §12 + §3）

**commit 拆分**:
- Commit 1: ADR-0028 创建 + adr/README 同步
- Commit 2: `ptxir_format.h` 多 entry 扩展 + bump version
- Commit 3: `PTXIRLoader` + `PtxEmuImageExecutor` 多 entry 支持
- Commit 4: `__cudaRegisterFatBinary` / `cuModuleGetFunction` 多 kernel 名查询
- Commit 5: tools/ 多 kernel 支持 + tests
- Commit 6: ADR-0025/0027/0029 §v1 限制段落更新 + ptxir-toolchain-stack.md v1.4

**预计 OpenSpec change 命名**: `2026-08-XX-multi-kernel-manifest-adr-0028`（per OpenSpec lifecycle, Lesson §6）

---

## Phase 13 HAL extension 跨仓协作（🆕 🟠 P1）

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
| RD-7 | **Phase 12.3 启动 (Driver API front door)** | 📋 **待启动（🔴 P0）** | 本文件 §Phase 12.3 |
| RD-8 | **Phase 12.4 启动 (ADR-0028 BLOCKING)** | 📋 **待启动（🟠 P1）** | 本文件 §Phase 12.4 |

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
| P12.3 | PTXIR Driver API + 缺失 CLI 工具 | 📋 **🔴 P0** | 本文件 §Phase 12.3 |
| P12.4 | ADR-0028 多 kernel manifest | 📋 **🟠 BLOCKING P1** | 本文件 §Phase 12.4 |
| P13 | HAL extension 跨仓协作 | 📋 **🟠 P1** | 本文件 §Phase 13 |
| 13-0 | **建立跨仓 RFC**：获取并引用 TaskRunner ADR-035 R5.1 原文到 Phase 13 RFC（roadmap v1 未验证 R5.1 实际内容） | `openspec/changes/<TBD>/rfc-hal-extension.md`（与 UsrLinuxEmu/TaskRunner 协同） | Phase 13 启动前置 |
| Future-1 | `$ORIGIN` 相对路径 | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-2 | CMake wrapper integration (`ptxemu_add_executable()`) | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-3 | macOS / Windows 支持 | 📋 低 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-4 | `cuInit` / `cuCtx*` context management | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |
| Future-5 | Packed `extra` argument buffer | 📋 中 | [架构 §12](docs/architecture/ptxir-toolchain-stack.md) |

---

## 下一步（执行顺序）

```
✅ 已 ship（2026-08-10）:
  1. ~~Phase 12.2 收尾 R1-R6~~ → 实际范围比预期小（R1/R2 已存在，仅 R3 + R5 是新工作）
     → commits: `20ad752b` (skeleton) + `b5d96c33` (R3 fix) + `50f41982` (R5)
     → OpenSpec change `2026-08-10-ptxir-cubin-cleanup` 待 R6.5 archive

立即（🔴 P0）:
  2. **guide-design 评审 3 个 improvement 提案**（2026-08-10 已建）：
     - `improvements/ptxir-driver-api-front-door.md` (Phase 12.3.A, P0)
     - `improvements/multi-kernel-manifest-adr-0028.md` (Phase 12.4, P1 BLOCKING)
     - `improvements/hal-extension-ptxemu-usrlinu-emu-taskrunner.md` (Phase 13, P1)
     → 评审通过 → 移入 `proposal-approved.md` → guide-plan 创建对应 OpenSpec change

后续（🟠 P1，**改进提案先行 → guide-design → openspec → tasks**）:
  3. Phase 12.3.A: 评审通过后 → openspec `proposal.md` 创建 → tasks.md 创建 → guide-ship worktree 实施
     → 预计 5-7 个独立 commit，10-15 天
  4. Phase 12.3.B (ptxir_build CLI) — 直接 OpenSpec change（无 improvement 提案，ADR-0025 已 ship）
  5. Phase 12.3.C (ptx-nvcc wrapper) — 直接 OpenSpec change（无 improvement 提案，ADR-0027 已 ship）
  6. Phase 12.4 (ADR-0028) — **严格在 12.3.A ship 之后启动**（PTXIRLoader 签名变化冲突）
  7. Phase 13 HAL: 先建跨仓 RFC 引用 TaskRunner ADR-035 R5.1 原文 → 评审 → 实施
  8. ADR-0025/0027/0029 §v1 限制段落更新（Phase 12.4 ship 后）
  9. ptxir-toolchain-stack.md 升级到 v1.4（同步 12.3 + 12.4 ship 状态）

每个 Phase 后跑:
  - cmake --build build && ctest --output-on-failure
  - ./scripts/sanity.sh
  - ./scripts/regression.sh
  - 每 commit 后 git diff + git log 检查（per Lesson §3）
```

---

**维护者**: PTX-EMU Architecture Team
**日期**: 2026-08-10（v3: Phase 12.2 收尾 ship — R3 silent-fallback fix + R5 Oracle scenarios + 文档同步；v2: 实施状态审计 + Phase 12.3/12.4/13 新增 + Phase 12.2 收尾任务拆分；v1: 2026-08-07）
