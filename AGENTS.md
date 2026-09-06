# PTX-EMU Agent Instructions

**Generated:** 2026-07-25 | **Commit:** be746976 | **Branch:** main

PTX ISA 指令级模拟器 — C++20/CUDA, ANTLR4 解析, fake libcudart.so 拦截 CUDA runtime. sm_100 虚拟架构.

## 🛑 最高优先级

- **ptx-lessons-learned**: 任何 OpenSpec change 前必读 `.opencode/skills/ptx-lessons-learned/SKILL.md`（14 核心经验 + 12 checklist）
- **跨模块间接状态翻译**: 迁移函数必须行级 diff, 不漏看似冗余的 `set_state()` — 下模块的 `sync_to_warp_state()` 才翻译为 `is_blocked`
- **递归锁死锁**: 持锁方法调用同锁其他 public 方法 = deadlock. 集中审计所有使用同一锁的代码路径
- **复杂迁移分 Phase commit**: 每个 Phase 独立可回退. 测试回归 → revert 该 Phase, 不混入后续 commit
- **基线 worktree**: 重大重构前 1 分钟建立 `.worktrees/`, 节省数小时争论

## HSK Cross-Repo Protocol Chain (2026)

PTX-EMU 通过 HSK (HandShake) protocol 与 CppTLM / UsrLinuxEmu 协同. 跨仓契约 14 天窗口, 超时 = 默认 ack.

| HSK | 状态 | 关键 commit | 文档 |
|-----|------|-------------|------|
| HSK-6 | ✅ ACCEPTED | PTX-EMU `25e36f60` + CppTLM `369cf71` | `docs/superpowers/specs/2026-08-18-hsk-6-cpptlm-bridge-deprecation.md` |
| HSK-7 | 🔵 预留 (未签发) — 仅 ABI 解冻 CPPTLMBRIDGE_VERSION 触发 | — | — |
| **HSK-8** | ✅ **ACCEPTED** (PR #14 merged `fcdad151` + CppTLM bump `beb3db8`) | PTX-EMU `fcdad151` (squash merge of 12 impl commits) + CppTLM `beb3db8` (submodule pin at `530bd6ca`) | `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` + `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` §Postmortem |
| **HSK-9** | 📤 已发布 — 触发条件: **ICOMPUTE_API_VERSION=1 引入 + SM-owns-state 契约** (非 PTXEMU_API_VERSION bump; HSK-9 实际语义是 CppTLM 端新接口 + 镜像头, 不是 PTX-EMU 公共头变更) | TBD (PR 创建后回填 PTX-EMU/CppTLM 端 commit SHA) | `docs/superpowers/specs/2027-02-09-hsk-9-icompute-api-v1-sm-rewrite.md` (镜像) |

HSK-8 实施进度 (per OpenSpec `openspec/changes/ptxemu-public-device-api/`):
- ✅ Phase 0: 2 污染点净化 (4 commits: 0.6 + 0.8/0.8b + 0.3a-d)
- ✅ Phase 1: `include/ptxemu/ir/` 头文件 scaffolding (commit 564174f7)
- ✅ Phase 2: `device_api.h` + `device_api_impl.cc` + `ptxemu_core` 静态库 (commit d281a21e)
- ✅ Phase 3: `PROJECT_IS_TOP_LEVEL` 隔离 + `PTXEMU_BUILD_TESTING` option + install rules (commit c225780e)
- ✅ Phase 4: `.github/workflows/drift_check.yml` 5 invariants 验证 (commit ae86c816)
- ✅ Phase 5: doc sync (`include/ptxemu/AGENTS.md` + root AGENTS.md HSK chain + audit doc) — commit `3678a0d7`
- ✅ Phase 6 (本 session): archive prep + postmortem + 4 main specs create — commits `d5600e89` / `8aa72f1d` / `530bd6ca`
- ✅ CppTLM bump PR — commits `6f408b5` / `09c27d5` / `d035551` + submodule pin `beb3db8`

HSK-8 follow-up (Phase 2.2/2.3 delegation + cpptlm bridge cleanup, per `docs/superpowers/plans/2026-08-24-hsk8-followup-task-path.md`):
- ✅ **`device-api-delegation`** (per OpenSpec `openspec/changes/archive/2026-08-25-device-api-delegation/`) — 4 stubbed `IPtxEmuDevice` 方法委托实施 (`set_scoreboard` / `set_active_mask` / `set_next_pc` / `attach_timing`) + drift_check Invariant 6 + `ptxemu-device-api-delegation` / `delegation-thread-pc-invariants` main specs 创建。**Phase 2.2 commit `4f6b5e1a`** (set_scoreboard + set_active_mask + set_next_pc) + **Phase 2.3 commit `488fe75e`** (attach_timing) squash merge 进 **PR #17 → commit `183a6ada`** (2026-08-25T10:40:22+08:00 by `chisuhua`, `feat/device-api-delegation` branch archived)。**实测 249/249 ctest PASS** + drift_check 6 invariants PASS。**1 task DEFERRED to Phase 2.2.1/2.3.1 follow-up** (task 3.4 `test_device_api_delegation_e2e.cc` — 4 方法各自 unit/integration 已分别覆盖,e2e 需要 deep warp setup,与现有 3 个 deferred stubs `warp_exe_once` / `get_thread_state` / `get_warp_status` 合并到 Phase 2.2.1/2.3.1 change)。**无需新 HSK** — Phase 2.2/2.3 delegation 为 HSK-8 ack 后追加,public ABI 未变更 (`PTXEMU_API_VERSION=1` 仍冻结)。
- ✅ **`cleanup-cudart-cpptlm-bridge-coupling`** (per OpenSpec `openspec/changes/archive/2026-08-25-cleanup-cudart-cpptlm-bridge-coupling/`) — PTX-EMU 端 cpptlm bridge 物理消除 4-phase refactor:**Phase 2a commit `292022a3`** (cudart_sim.cpp bridge 删除) + **Phase 2b commit `e4d7e369`** (memory.cpp GLOBAL LD/ST bridge 删除) + **Phase 3 commit `09786635`** (CMakeLists + 4 文件删除 + abi_guards.h 创建 + 3 AGENTS.md 同步),合计 **3 个 commits**。**14 PtxEmuDriverShim/cpptlm bridge 符号消失** (vs proposal 估算 16,差异源于 GCC ctor/dtor 复制可见性),**`libptxemu_device.so` 8 ABI 符号保留** (`ptxemu_image_*` per `OUT-OF-SCOPE` 设计)。审计追踪:`docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21-cleanup.txt` (1609 行)。Postmortem:`docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`。同步主 specs:**`cudart-sync-only-runtime`** (新建) + **`cpptlm-d1-full`** SUPERSEDED (4 旧 requirement 删除:`cpptlm-bridge-interface` / `cudart-async-launchkernel` / `ptx-global-ld-st-bridge` / `libcpptlm-cudart-integration`;`cudart-stream-synchronization` → `cudart-stream-api` MODIFIED) + **`auto-co-simulation`** 整删除 (`StubBridge`/`g_ptx_emu_driver_shim`/`EMU_COSIM`/`PTX_EMU_MAX_ADVANCE_CYCLES` 机制全部不再存在)。`cpp-tlm-consumes-ptxemu-device` 反向桥接留待未来 change (HSK-9 准入或独立 change)。
- ✅ **`antlr4-path-hardcoding-fix`** (per OpenSpec `openspec/changes/antlr4-path-hardcoding-fix/`) — `${CMAKE_SOURCE_DIR}` → `${PROJECT_SOURCE_DIR}` for `ANTLR_EXECUTABLE` + `ANTLR4_RUNTIME_SOURCE_DIR` (2-line change)。CppTLM-side chained builds via `add_subdirectory`/`ExternalProject_Add` 现在成功。drift_check **Invariant 7** added (guards against `${CMAKE_SOURCE_DIR}/antlr4` regression). Resolves Doc2 §8 follow-up item 4 (per `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md` §7.2 item 4). 无需新 HSK — PTX-EMU 单仓构建系统修复，public ABI 未变。

跨仓协调顺序 (HSK-8 spec §"跨仓协调顺序" + follow-up):
1. ✅ PTX-EMU HSK-8 ack commit (738b412c) + issue #22 评论 #5381166580
2. ✅ PTX-EMU Phase 2 PR — PR #14 merged `fcdad151`(2026-08-24T03:55:14Z by `chisuhua`, 分支已 archive + 删除)
3. ✅ PTX-EMU CI 全绿 — ctest 246/246 + drift_check 5 invariants PASS
4. ✅ PTX-EMU Phase 2 PR 合入 main — origin/main HEAD = `530bd6ca`,**ahead of 2026-09-19 target by 26 天**
5. ✅ CppTLM bump PR — submodule pin `beb3db8` → PTX-EMU `530bd6ca`, 5 HSK-8 commits merged (`6f408b5` + `09c27d5` + `d035551` + `12b9e0f` + `beb3db8`)
6. ✅ Gate 1 fix archive (2026-08-25) — PR #16 (`chore/archive-gate1-fix-2026-08-25` branch, 3 commits `13f55bbe` / `3a3f8a93` / `098c50fc`),Gate 1 leak 物理消除 by 4-phase refactor (`09786635`),详见 `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`
7. ✅ **HSK-8 follow-up Phase 2.2/2.3** (2026-08-25) — PR #17 merged `183a6ada` (squash of `4f6b5e1a` + `488fe75e`),`feat/device-api-delegation` branch archived。4 个 delegated method 全部 wired (set_scoreboard / set_active_mask / set_next_pc / attach_timing),`src/ptxemu/device_api_impl.cc` 无 empty-body stubs。drift_check **6 invariants** PASS (新增 Invariant 6)。ctest **249/249 PASS** (246 baseline + 2 unit + 1 integration)。无需 HSK 触发 — public ABI 未变。
8. ✅ **cpptlm bridge cleanup** (2026-08-21 → 2026-08-25) — 3 commits `292022a3` + `e4d7e369` + `09786635` 已合入 main (在 PR #16/17 之前)。`libcudart.so` 现在是 sync-only runtime shim,**0 CppTLM 符号**,**`cpp 不暴露`** 约束在反向方向上进一步强化。**`libptxemu_device.so` 8 ABI 符号保留**供未来 reversal direction (`cpp-tlm-consumes-ptxemu-device` change) 使用。无需 HSK 触发 — PTX-EMU 单仓 refactor。
9. ✅ **HSK-8 follow-up Phase 2.2.1/2.3.1** (2026-08-25) — OpenSpec change `phase-2-2-1-3-1-followup` completes the remaining 3 deferred stubs (`warp_exe_once` / `get_thread_state` / `get_warp_status`) + e2e test via `WarpContext::execute_warp_instruction` + drift_check Invariant 6 exemption list EMPTY。12/12 IPtxEmuDevice methods 全部 wired。drift_check **6 invariants** PASS。ctest **251/251 PASS** (249 baseline + 2 新 integration 测试: `integration_warp_status_snapshot` + `integration_device_api_delegation_e2e`)。无需 HSK 触发 — public ABI 未变 (`PTXEMU_API_VERSION=1` 冻结),`WarpStatus` 5-field struct layout preserved。详见 [HSK-8 audit](./docs/audits/2026-08-13-hsk8-ptxemu-public-api.md) §Postmortem + OpenSpec change `openspec/changes/phase-2-2-1-3-1-followup/`。
- ✅ **Phase 1.5 namespace migration**: 22 atomic commits (1.5c+d → 1.5k); ctest 254/254; PTX 46/46; scanner fully clean; Invariant 8 wired.

HSK protocol 文档: `docs/superpowers/specs/HSK-PROTOCOL-NOTES.md`

## STRUCTURE

```
src/                  # 核心源码
├── cudart/           # fake libcudart.so (__cudaRegisterFatBinary → cudaLaunchKernel)
├── grammar/          # ANTLR4 .g4 (5 split files)
├── ptx_ir/           # IR types + ptx_op.def X-Macro (106 entries)
├── ptx_parser/       # PtxVisitor + CFGBuilder
├── ptxsim/           # 执行引擎
│   ├── barrier/      # BarrierModule + WarpBarrier + CTABarrier
│   ├── core/         # GPU/SM/CTA/Warp/Thread 上下文
│   ├── instructions/ # PTX 指令 handlers (~20 files)
│   └── memory/       # TMA descriptor, TMEM, TmemAllocator
├── ptxir/            # PTXIR 二进制序列化
├── memory/           # SimpleMemory, SharedMemoryManager
└── register/         # RegisterBankManager
include/              # 公共头文件 (mirrors src/)
tests/                # 3 层: unit/integration/e2e (~181 文件, Catch2)
configs/              # GPU JSON + debug INI
bench/                # CUDA 基准测试 (含 CUTLASS/cute headers)
docs/                 # ADR, 架构文档, audits (173 文件)
openspec/             # proposal/design/tasks/specs (413 文件)
scripts/              # sanity.sh, regression.sh, debug-run.sh
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| CUDA runtime 拦截 | `src/cudart/cudart_sim.cpp` |
| PTX 解析 | `src/ptx_parser/` + `src/grammar/` |
| IR + 新指令注册 | `include/ptx_ir/ptx_op.def` (X-Macro) |
| 指令实现 | `src/ptxsim/instructions/` |
| 执行层次 | `src/ptxsim/core/` (GPU→SM→CTA→Warp→Thread) |
| 屏障同步 | `src/ptxsim/barrier/` (BarrierModule) |
| 测试 | `tests/{unit,integration,e2e}/` 对应子目录 |
| PTX 语法测试 | `tests/ptx/test_all_ptx.sh` (NOT ctest!) |
| 架构决策 | `docs/adr/` (23 文件) |
| 技能索引 | `.opencode/skills/README.md` |

## CODE MAP

| Symbol | Type | File | Role |
|--------|------|------|------|
| `__cudaRegisterFatBinary` | Func | `src/cudart/cudart_sim.cpp` | 初始化入口 |
| `cudaLaunchKernel` | Func | `src/cudart/cudart_sim.cpp` | 内核启动 |
| `GPUContext::exe_once()` | Func | `src/ptxsim/core/gpu_context.cpp` | 时钟周期循环 |
| `SMContext::exe_once()` | Func | `src/ptxsim/core/sm_context.cpp` | SM 级执行 |
| `WarpContext::execute_warp_instruction()` | Func | `src/ptxsim/core/warp_context.cpp` | Warp 指令分发 |
| `ThreadContext::execute_thread_instruction()` | Func | `src/ptxsim/core/thread_context.cpp` | 线程级执行 |
| `InstructionFactory::get_handler()` | Func | `src/ptxsim/instruction_factory.cpp` | X-Macro 分派 |
| `BarrierModule` | Class | `include/ptxsim/barrier/barrier_module.h` | 屏障状态机 |
| `WarpBarrier` | Class | `include/ptxsim/barrier/warp_barrier.h` | Per-warp 屏障 |
| `PtxVisitor::visitFunctionDecl()` | Func | `src/ptx_parser/ptx_visitor.cpp` | PTX 解析入口 |

## CONVENTIONS

- **格式化**: clang-format (BasedOnStyle=LLVM, IndentWidth=4, ColumnLimit=80)
- **命名**: 文件 snake_case | 函数 camelCase | 类 PascalCase | 变量 camelCase
- **PTX 指令**: 全小写 (mov, add, ld, st)
- **头文件**: `#ifndef`/`#define`/`#endif` 守卫
- **X-Macro**: `#define X(name) #include "ptx_op.def" #undef X`
- **TDD 强制**: 测试先行 (Red) → 实现 (Green) → sanity.sh 验证
- **ctest 命名**: 必须前缀 `unit_`/`integration_`/`e2e_` (commit `ab55e06`)
- **测试标签**: `<type>;<subject>`, 如 `unit;barrier`

## ANTI-PATTERNS

- ❌ `ctest` 代替 `test_all_ptx.sh` 做 PTX 语法测试
- ❌ 不加载 `ptx-grammar-modification` 技能就改 `.g4`
- ❌ `force_set_pc()` — 用 `set_pc()` + `commit_pc()`
- ❌ 从 WarpContext 调 ThreadContext 方法无锁
- ❌ `set_active_mask()` 做 OR 合并 (ret handler 依赖 overwrite 语义)
- ❌ 新增 `Wbar` struct 使用 — 已全部迁移至 BarrierModule
- ❌ 不加 struct+variant 就加 `ptx_op.def` 条目
- ❌ 不改 reader 就改 PTXIR writer 格式

## COMMANDS

```bash
. env.sh                                                    # 设置环境 (必须!)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build.sh                                                  # Debug 快捷
cmake --build build --target GenerateParser                 # 改 .g4 后重生成
cd build && ctest                                           # 全量测试
./tests/ptx/test_all_ptx.sh                                 # PTX 语法测试 (NOT ctest!)
./scripts/sanity.sh                                         # 分层健康检查
./scripts/regression.sh                                     # 全量回归
```

## CHILD AGENTS.MD

| File | Coverage |
|------|----------|
| `configs/AGENTS.md` | GPU arch JSON + debug INI schema |
| `include/cudart/AGENTS.md` | ABI guards + public headers (abi_guards.h, cuda_driver.h, cudart_intrinsics.h) |
| `src/cudart/AGENTS.md` | CUDA runtime interception |
| `src/grammar/AGENTS.md` | ANTLR4 grammar modification |
| `src/ptx_ir/AGENTS.md` | IR types + X-Macro + PTXIR |
| `src/ptx_parser/AGENTS.md` | PTX parser + CFG builder |
| `src/ptxsim/AGENTS.md` | Execution engine overview |
| `src/ptxsim/core/AGENTS.md` | GPU→SM→CTA→Warp→Thread contexts |
| `src/ptxsim/instructions/AGENTS.md` | PTX instruction handlers |
| `src/ptxsim/barrier/AGENTS.md` | Barrier state machine |
| `tests/AGENTS.md` | 3-tier test suite |