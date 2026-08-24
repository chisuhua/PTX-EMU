## Why

CppTLM 仓 S1 deep-integration 交付通过 `cmake/PTXEmuCore.cmake` shim 实现编译防火墙,但 S1 facade.cc 直接 include 12 个 PTX-EMU 实现头并完整持有 `GPUContext`/`SMContext`/`WarpContext`/`ThreadContext` 类型 — 头文件漂移照样炸 CppTLM 编译。

PTX-EMU 远端 main 已主动清理 cpptlm bridge Phase 1-4 (1018 行删除,commits `a9a14e1d`/`292022a3`/`e4d7e369`/`09786635`),HSK-6 已接受桥梁废弃。HSK-3 (ExternalProject_Add) 旧方向已被 HSK-6 废止,新方向需要 HSK-8 handshake。

本次 change 实施 [HSK-8 spec `3b8f7a5`](https://github.com/chisuhua/CppTLM/commit/3b8f7a5) (PTX-EMU 仓主 ack [`738b412c`](https://github.com/chisuhua/PTX-EMU/commit/738b412c)),新增 PTX-EMU 端**公共设备 API 契约** `ptxemu::device_api.h`,通过 CMake `target_include_directories` PUBLIC/PRIVATE 拆分确保 CppTLM 侧编译时看不到 PTX-EMU 内部头。

**Oracle 闭包审计** (session `ses_fd5ef471cffeWvINOBm5E1GMYd`): StatementContext 实际闭包 5 文件 (~1053 LOC) 通过 CppTLM Decision 5 "pure data, no implementation" 结构门槛,但发现 2 个非纯数据污染点必须 Phase 0 净化。

## What Changes

### 新增 (NEW)

- `include/ptxemu/device_api.h` (~200 行) — 公共设备 API 契约,包含 `IPtxEmuDevice` 抽象接口 + `DeviceConfig`/`WarpStatus`/`LaneStatus`/`ThreadState` DTO + `create_device`/`destroy_device` 工厂 + `PTXEMU_API_VERSION=1` 宏
- `include/ptxemu/ir/statement.h` — StatementContext 公共晋升 (Phase 0 净化后)
- `src/ptxemu/device_api_impl.cc` (~400 行) — 薄适配层,对内调 PTX-EMU 核心 API,对外实现 `IPtxEmuDevice` 虚方法表
- `src/ptxemu/cmake/[ptxemu_core.cmake]` — CMake 库目标 `add_library(ptxemu_core STATIC ...)` 显式源清单 + PUBLIC/PRIVATE include 拆分
- `.github/workflows/drift_check.yml` — header 一致性 check workflow (~5min,Phase 2 PR 即纳入)

### 修改 (MODIFY)

- `CMakeLists.txt` (root): 顶部加 `option(PTXEMU_BUILD_TESTING "Build PTX-EMU tests" OFF)` + `if(PROJECT_IS_TOP_LEVEL OR PTXEMU_BUILD_TESTING)` 隔离 `tests/` + install 规则
- `include/ptx_ir/operand_context.h` — **Phase 0 净化**: `mutable void *operand_phy_addr = nullptr` (line 59) 移出值类型,改用 `unordered_map<OperandContext*, void*>` runtime cache
- `include/ptx_ir/statement_context.h` — **Phase 0 净化**: `InstructionState state = InstructionState::READY` (line 310) 移出值类型,改 `unordered_map<StatementContext*, InstructionState>` runtime side-table;`BarWarpSyncInstr::reconvergenceLabel` (line 229) 删除(dead code,参见 HSK-8 spec Oracle session `ses_fd5ef471cffeWvINOBm5E1GMYd`)
- `include/ptxir/ptx_qualifier.def` (332 行) + `include/ptxir/ptx_op.def` (203 行) — 路径迁移至 `include/ptxemu/ir/` (晋升闭包一部分),旧路径保留 forwarding header 一个 release 周期
- 5 文件 namespace 包装加 `ptxemu::ir` (`statement_context.h`/`operand_context.h`/`ptx_types.h`/`execution_types.h`/`statement_factory.h`)
- `include/ptx_ir/` 目录结构调整为 `include/ptxemu/ir/`(新公共布局),旧目录保留兼容 forwarding

### 不变 (UNCHANGED)

- CppTLM 端不修改 (后续由 CppTLM 仓独立 bump PR 处理,见 HSK-8 spec §"跨仓协调顺序" Step 5)
- PTX-EMU 内部 `ExecutionContext`/`gpu_context.cpp`/`sm_context.cpp`/`warp_context.cpp`/`thread_context.cpp` 实现不变 (HSK-8 锁定 "PTX-EMU 内部重构不影响 device_api.h" 承诺,见 spec §3 Risk 1 Mitigation)
- PTXIR 二进制格式 (`ptxir_writer.h`) 不变 (HSK-8 spec §7 锁)

### 验证清单

- 5 条验收条件 (per HSK-8 ack body [§4](https://github.com/chisuhua/PTX-EMU/blob/738b412c/docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md)):
  - [ ] #1 `include/ptxemu/device_api.h` 已新增
  - [ ] #2 `add_library(ptxemu_core STATIC ...)` 可被 `add_subdirectory(external/PTX-EMU)` 消费
  - [ ] #3 `consumer_smoke` 测试 PASS (**下期 HSK-9 准入,Phase 2 PR 仅含 `drift_check`**)
  - [ ] #4 `drift_check` workflow PASS
  - [ ] #5 PTX-EMU maintainer 在 #22 评论 +1 ack (**已完成**,comment 5381166580 @ 2026-08-22)

## Capabilities

### New Capabilities

- `public-device-api`: PTX-EMU 端公共设备 API 契约 (`ptxemu/device_api.h`) — 抽象接口 + DTO + 工厂 + 版本守卫宏;CppTLM 唯一 include 入口
- `ptxemu-core-library`: CMake `ptxemu_core` STATIC 库目标 — PUBLIC include `include/ptxemu/` + PRIVATE `src/ptxemu/`;支持 `add_subdirectory(external/PTX-EMU)` 消费
- `statement-ir-public`: `StatementContext` 公共 IR 头晋升 (Phase 0 净化后零非纯数据污染) + 5 文件闭包晋升 + `ptxemu::ir` namespace
- `ci-drift-check`: `.github/workflows/drift_check.yml` — **local-only invariants**: `PTXEMU_API_VERSION=1` 守卫宏保留 + `IPtxEmuDevice` 虚方法数量 >= 12 (覆盖 S1 facade 12 callsites 1:1), 本地编译无关 (无需 CppTLM submodule, 避免 PTX-EMU CI 依赖 CppTLM 构建链)

### Modified Capabilities

无 — 本 change 不修改 spec-level behavior,仅新增。

## Impact

### 受影响代码

- **新增公共面 (~5-7 文件, 600 LOC)**:
  - `include/ptxemu/device_api.h`
  - `include/ptxemu/ir/statement.h` (从 `include/ptx_ir/statement_context.h` 晋升)
  - `include/ptxemu/ir/{operand_context,ptx_types,execution_types}.h` (闭包晋升)
  - `include/ptxemu/ir/{ptx_qualifier,ptx_op}.def` (X-Macro 表头)
  - `src/ptxemu/device_api_impl.cc` (薄适配层)
- **修改内部面 (~5-7 文件, ~150 LOC diff)**:
  - `include/ptx_ir/operand_context.h` (Phase 0 净化)
  - `include/ptx_ir/statement_context.h` (Phase 0 净化 + dead code 删)
  - `CMakeLists.txt` (root + include 子目录)
- **新增 CI 工作流**: `.github/workflows/drift_check.yml`

### 受影响 API

- **PTX-EMU 内部 API**: `gpu_context.cpp`/`sm_context.cpp`/`warp_context.cpp` 不直接暴露 — `device_api_impl.cc` 作为唯一适配层
- **HSK-4 vendored interface 复用**: `IPtxEmuDevice::attach_timing()` 接收已 vendored 3 接口 (`IScoreboard*` / `IPipelineLatencyProvider*` / `ITensorCoreTiming*`),不重复定义
- **HSK-6 不变**: `CPPTLMBRIDGE_VERSION` 冻结于 2,任何解冻触发 HSK-7 (与本 change 无交集)

### 受影响 ADR

- **[ADR-0028 BLOCKING DEPENDENCY upgrade](docs/adr/)**: 升级路径锁定 — PTX-EMU PR 必须基于 origin/main (post `09786635`),严禁基于 `c2038a93` 或更早 (保留 `g_cpptlm_bridge` 引用)
- **[ADR-0029 image executor](docs/adr/)**: 间接相关 — `IPtxEmuDevice` DTO `WarpStatus` 应避免与 ADR-0029 的 image executor HAL 接口命名冲突 (Phase 1 design review 检查)

### 受影响 skill

- **ptx-lessons-learned §1 (跨模块状态翻译)**: Phase 0 净化 `state` 字段正是这条教训 — 不能让 IR 值类型携带执行态
- **ptx-lessons-learned §3 (Phase commit 纪律)**: 本 change 至少 4 phases (Phase 0 净化 / Phase 1 布局迁移 / Phase 2 facade 层 / Phase 3 CI + 集成),每 phase 独立 commit
- **ptx-lessons-learned §4 (基线 worktree)**: 已建 `.worktrees/phase2-baseline` at `738b412c`
- **ptx-lessons-learned §7 (Metis pre-impl review)**: 本 proposal 完成后必跑 Metis 审计 4 artifacts + 2 污染点下游影响

### 风险与回退

- **Risk 1**: Phase 0 净化发现 `state` 字段被 5+ 处执行引擎深度依赖无法剥离 → **降级路径 (b)** opaque `StatementHandle` + `decode_ptxir` 字节流 (HSK-8 spec §7 已预留)
- **Risk 2**: 4 artifacts 内部不一致 → 严格执行 Checklist J (4 个 artifact 同债务项范围对齐)
- **Risk 3**: Phase 2 PR 未合入前 CppTLM 仓先开 bump PR → 跨仓协调顺序锁定 (Step 5 仅在 Step 4 后启动)
