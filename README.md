# PTX-EMU

> **状态**：SIMT v2.0 完成；Blackwell tcgen05 完整实施；H5 规划中
> **核心特性**：C++20/CUDA PTX 模拟器，ANTLR4 解析 PTX，fake libcudart.so 拦截 CUDA runtime
> **文档入口**：[docs/README.md](./docs/README.md)

PTX-EMU 是一个 PTX（Parallel Thread Execution）指令级模拟器，用于在无 NVIDIA GPU 环境下仿真执行 CUDA 程序。

## 快速开始

```bash
# 1. 设置环境（必须！）
. env.sh

# 2. 配置 + 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. 跑测试
cd build && ctest --output-on-failure
```

## 架构概览

- **执行层次**：GPUContext → SMContext → CTAContext → WarpContext → ThreadContext
- **PTX 解析**：ANTLR4 → IR (StatementContext) → 解释执行
- **测试三类物理隔离**：
  - `tests/unit/` — 直接单元测试（数据结构/算法）
  - `tests/integration/` — 指令序列集成测试（通过 `execute_warp_instruction`）
  - `tests/e2e/` — CUDA Kernel 端到端测试（nvcc 编译 + 拦截）

## 文档导航

| 类别 | 路径 |
|---|---|
| 项目总入口 | [AGENTS.md](./AGENTS.md) |
| 文档索引 | [docs/README.md](./docs/README.md) |
| SIMT 架构 | [docs/architecture/SIMT-ARCHITECTURE-V2.md](./docs/architecture/SIMT-ARCHITECTURE-V2.md) |
| 开发指南 | [docs/developer-guide/](./docs/developer-guide/) |
| ADR 索引 | [docs/adr/README.md](./docs/adr/README.md) |
| Blackwell tcgen05 架构 | [docs/adr/ADR-0016-blackwell-only-tcgen05.md](./docs/adr/ADR-0016-blackwell-only-tcgen05.md) |
| tcgen05 实施 roadmap | [docs/dev-process/post-tcgen05-roadmap.md](./docs/dev-process/post-tcgen05-roadmap.md) |
| 健康审计 | [docs/audits/HEALTH-AUDIT-2026-06-21.md](./docs/audits/HEALTH-AUDIT-2026-06-21.md) |
| 审计勘误 | [docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md](./docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md) |
| Roadmap | [openspec/changes/](./openspec/changes/)（活跃 changes 即未来 roadmap 项）|

## 已实现功能

- **PTX-EMU Image Executor** (`libptxemu_device.so` + `cpptlm_module.h`): 5 `extern "C"` ABI entry (`ptxemu_image_load`/`execute`/`unload`/`kernel_name`/`module_version`) for in-memory PTXIR module loading and execution. D3 mutation bug fix via per-launch re-deserialize. 7 [SINGLE-GPU-INSTANCE] assumptions documented. 5 byte-identical fallback gates verified + D3 perf acceptance (`cute_rmsnorm` 0.183x wall-time ratio, well below 1.10 threshold). See [ADR-0029](./docs/adr/ADR-0029-ptxemu-image-executor.md).
- **PTX-EMU Public Device API** (`include/ptxemu/device_api.h` + `ptxemu_core` STATIC lib): CppTLM 消费入口 (`IPtxEmuDevice` 接口 + `PTXEMU_API_VERSION=1` 冻结 + 6 HSK-8 invariants via `drift_check.yml`)。`cpp 不暴露` 约束: CppTLM 侧 0 PTX-EMU 内部 header includes (commit `09c27d5`)。跨仓协议 HSK-8 ✅ ACCEPTED (PR #14 merged `fcdad151` + CppTLM submodule bump `beb3db8`)。**Phase 2.2/2.3 + Phase 2.2.1/2.3.1 delegation 完成** (12/12 methods, commits `4f6b5e1a` + `488fe75e` on `feat/device-api-delegation` + Phase 2.2.1/2.3.1 follow-up on `feat/phase-2-2-1-3-1-followup`): 12 stubbed methods wired — `set_scoreboard` (R7 验证 SMContext + IScoreboard 注册) / `set_active_mask` (overwrite 语义 per BUG-RETHANG 守护) / `set_next_pc` (`set_pc`+`commit_pc` per AGENTS.md L85) / `attach_timing` (Decision 6 命名空间桥接 `static_cast<void*>` round-trip) / `warp_exe_once` (per-warp scheduler via `WarpContext::execute_warp_instruction`) / `get_thread_state` (delegates to `thread->get_state()` + `map_state`) / `get_warp_status` (populates existing 5-field struct via `warp->get_warp_state().threads[]`)。详见 [HSK-8 audit](./docs/audits/2026-08-13-hsk8-ptxemu-public-api.md) + [follow-up plan](./docs/superpowers/plans/2026-08-24-hsk8-followup-task-path.md) + OpenSpec change `openspec/changes/phase-2-2-1-3-1-followup/`。
- **Gate 1 byte-identical fallback** (per ADR-0029 §D7): `libcudart.so` 链接行不含 `cpptlm_core` (`commit 09786635`),Gate 1 leak 物理消除。详见 [postmortem](./docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md)。
- **PTXIR-Embedded CUBIN/EXE**: 标准可执行文件末尾追加 PTXIR section，PTX-EMU 通过 O(1) tail detection 加载（ADR-0024 v1.1）。提供 `ptxir_embed`/`ptxir_extract` 工具，`PTXIR_MODE=auto` 控制 `__cudaRegisterFatBinary` 分发。
- **Blackwell tcgen05**：完整实现 `.mma` / `.ld` / `.st` / `.commit` / `.wait`（commit `4151268` Fix #14）— 详见 [docs/adr/ADR-0016-blackwell-only-tcgen05.md](./docs/adr/ADR-0016-blackwell-only-tcgen05.md)
- **TMA descriptors**：异步拷贝 descriptor 解析（commit `ad527f5` Fix #5）
- **TMEM**：per-CTA Tensor Memory（commit `758edb0` Fix #6）
- **Cluster arrive/wait**：分布式 shared memory 同步（commit `e513235` Fix #7）
- **TcQueue**：commit-group + wait-aware scheduling（commit `c0fa43f` Fix #8）

## 已知限制

- **PTX 指令覆盖**：参考 [docs/audits/debt-audit-2026-07-02.md](./audits/debt-audit-2026-07-02.md) 自动统计（避免硬编码）
- **CppTLM D1-Full MemoryBridge**：**已归档（2026-07-17）** — ABI 真值源 `include/cudart/cpptlm_bridge.h`（5 虚方法 + `g_cpptlm_bridge` 全局指针）+ 异步 `cudaLaunchKernel` + `cudaStreamSynchronize` 真实轮询 + GLOBAL LD/ST timing-only 桥接。默认 `g_cpptlm_bridge == nullptr` 时字节级兼容原有同步路径。详见 [ADR-0021](./docs/adr/ADR-0021-cpptlm-d1-full-integration.md) + `openspec/specs/cpptlm-d1-full/`。
- **CppTLM bridge auto-co-sim**：标准 CUDA 程序在 `BUILD_LIB_CPPTLM_CUDART=ON` 下零修改自动协同仿真（commit `auto-co-sim-standalone`）。详见 [ADR-0021](docs/adr/ADR-0021-cpptlm-d1-full-integration.md)。
- **pre-Blackwell tcgen05**：永久抛 `UnsupportedInstructionException`（c5 Fix #1 + [ADR-0016](./docs/adr/ADR-0016-blackwell-only-tcgen05.md)）
- **ANTLR 版本**：4.13.2 完全 vendored
- **CUDA Toolkit**：环境自适应（`env.sh` 自动检测 `$(which nvcc)`）

## 贡献指南

新增 PTX 指令时，遵循三步流程：

1. 在 `include/ptx_ir/ptx_op.def` 添加 X-Macro 条目
2. 在 `src/ptxsim/instructions/<category>.cpp` 实现 handler
3. 添加测试（unit + integration + e2e 三层）

详见 [docs/developer-guide/](./docs/developer-guide/)。

## 相关参考

- [PTX ISA 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- [GPGPU-Sim](https://github.com/accel-sim/gpgpu-sim_distribution)

## 许可证

[按项目实际情况填写]
