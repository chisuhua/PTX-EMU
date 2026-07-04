# Implement Blackwell tcgen05 (skip pre-Blackwell WMMA)

> **架构决策**：本 change scope 由 [ADR-0016](../docs/adr/0016-blackwell-only-tcgen05.md)
> 锁定 — 仅实现 Blackwell (sm_100 / sm_120) 的 `tcgen05.*` 指令集，不实现
> pre-Blackwell (`wmma.mma.sync`, `wgmma.async` 等)。
>
> **前置依赖**：TMA descriptors + Tensor Memory (TMEM) + cluster mode + async
> tensor core queue 必须先建（Phase 0）。这些是 tcgen05 的硬前置，不是"可选优化"。

## Why

`replace-silent-stub-failures` (archived 2026-07-04) 把
`WmmaHandler::processWmmaOperation` 的 silent no-op 改为
`throw UnsupportedInstructionException`，建立了"未实现 stub 必须显式失败"的合约。
本 change 是该合约的"补完"——把 Blackwell 现代张量核心指令集**真正实现**出来。

**核心动机**：

1. **vision purity**：放弃 pre-Blackwell legacy 兼容性，全力投入 Blackwell 现代范式
2. **scope discipline**：cutlass / cute 模板矩阵太大，全做 = 永远做不完
3. **infrastructure first**：`tcgen05.mma` 需要 TMA + TMEM + cluster + async queue
   这套新基础设施，先建避免后期架构返工
4. **future-readiness**：Blackwell 是 NVIDIA 2024-2026 主推架构，下一代仍在同一范式

**前置 commit / state**：
- HEAD `e83ec94`（在 main 上刚合并 `replace-silent-stub-failures` + propose 本 change）
- ADR-0016（Accepted 2026-07-04，本 change 的架构依据）
- `UnsupportedInstructionException` 基础设施已就绪（`include/ptxsim/ptx_exceptions.h:97`）
- cute_rmsnorm 等 e2e 测试已通过（实测不依赖 WMMA — `grep -r "wmma\." tests/ bench/` 零匹配）

## What Changes

### Phase 0：基础设施（TMA / TMEM / cluster / async queue）

**TMA descriptors**（`src/ptxsim/memory/tma_descriptor.{h,cpp}`）：
- 解析 `cuda::tma::desc` 二进制布局（TensorMap header + swizzle + strides）
- 提供 `tma_descriptor::load(...)` / `store(...)` 抽象
- 拦截 fake `cudaMemcpy` 时识别 descriptor 拷贝
- ~800-1200 LoC

**Tensor Memory (TMEM)**（`src/ptxsim/memory/tmem.{h,cpp}`）：
- per-CTA 的新存储层，pre-Blackwell 没有
- 256 个 32-bit slot × 128 byte / slot = 32 KB per CTA
- 与 shared memory 平行，不互通
- ~600-800 LoC

**Cluster mode**（`src/ptxsim/cluster/{h,cpp}`）：
- `cta_cluster_arrive` / `cta_cluster_wait` 同步原语
- **Deferred**: distributed shared memory（`cta_group::1` 不需要，仅 `cta_group::2` 需要）
- ~300-400 LoC（Oracle review 简化：从 800-1200 削减至 arrive/wait only）

**Async tensor core queue**（`src/ptxsim/async/tc_queue.{h,cpp}`）：
- per-CTA 的命令队列（commit-group counter）
- `tcgen05.commit` / `tcgen05.wait` 同步原语
- 与现有 `WarpState` 集成（独立抽象层，不冲突 per-thread PC）
- ~800-1200 LoC

### Phase 1：tcgen05.mma fragment arithmetic

`src/ptxsim/instructions/wmma.cpp`：
- 解析 `tcgen05.mma.cta_group::1.kind::f16` 指令变体
- 实现真实 fragment 算术（m64nNk 等）
- 复用 `include/ptxsim/utils/half_utils.h`
- 委托给 Phase 0 的 async queue
- ~500-800 LoC

### Phase 2：tcgen05.ld / st + commit / wait

`src/ptxsim/instructions/wmma.cpp`：
- `tcgen05.ld` / `tcgen05.st` → TMA descriptor + TMEM
- `tcgen05.commit` → commit-group counter++
- `tcgen05.wait` → 等到 commit-group 完成
- ~600-1000 LoC

### Phase 3：e2e GEMM kernel + AGENTS/spec 同步

`tests/e2e/kernel/test_blackwell_gemm.cu`：
- cutlass 3.x GEMM kernel，target sm_100
- 16×16 GEMM，验证 fragment 算术正确

`src/ptxsim/instructions/AGENTS.md` + 根 `AGENTS.md`：
- 移除 WMMA stub 描述
- 标注 Blackwell tcgen05 已实现

`openspec/specs/wmma-tensor-core/spec.md` → publish 到 `openspec/specs/`

## Non-Goals

### 显式拒绝（ADR-0016 锁定）

- **不实现 `wmma.mma.sync.*`**（sm_70 / sm_75 / sm_80 / sm_86）
- **不实现 `wgmma.async.*`**（sm_90）
- **不实现 `mma.sync.*`**（sm_70+ 通用路径）
- **不实现 pre-Blackwell cute / cutlass template instantiation**

这些路径继续抛 `UnsupportedInstructionException`（与 `replace-silent-stub-failures`
合约一致），不视为 bug。

### 范围限制

- **sm_120 sparse / FP4 / mxfp8**：落在同一范式，但单独 propose，每个特性一个 change
- **性能对标**：仅 functional correctness，不追求 cycle-accurate
- **SASS 层语义**：仍 PTX-level interpretation
- **mma.sp 稀疏变种**：Phase 3 之后单独 change
- **TMA host API 拦截策略**：Phase 0 用 fake descriptor（手工填值），host API 拦截
  后续单独 propose（候选 ADR-0017）

### 不修改

- `UnsupportedInstructionException` / `ExecutionStateException` 类定义本身
- X-Macro `ptx_op.def`（仍 `S_WMMA` → `WmmaHandler::processWmmaOperation`）
- `replace-silent-stub-failures` 已建立的合约
- ADR-0001..0015 任何既有决策

## Goals

### Phase 0 目标

1. TMA descriptor 解析单元测试覆盖 ≥ 10 种典型 swizzle/stride 组合
2. TMEM 单元测试验证 256 slot × 128 byte 容量 + 读写一致性
3. cluster mode 单元测试验证 cta_cluster_arrive/wait 正确性
4. async queue 单元测试验证 commit-group counter + wait-aware 调度
5. 全部单元测试 PASS，无回归

### Phase 1 目标

1. `tcgen05.mma.cta_group::1.kind::f16` 真实 fragment arithmetic
2. 32 lane 输出片段元素正确（per NVIDIA PTX ISA §9.7.13）
3. 集成测试覆盖 uniform warp + commit-after-mma 序列
4. 单元测试 + 集成测试 PASS

### Phase 2 目标

1. `tcgen05.ld` / `st` 与 TMA descriptor + TMEM 集成
2. `tcgen05.commit` / `wait` 同步原语正确
3. 完整 mma 序列（ld → mma → commit → wait → st）端到端通过

### Phase 3 目标

1. cutlass 3.x GEMM kernel e2e 通过
2. AGENTS.md 同步反映新行为
3. spec 从 change 移到 main specs

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| TMA descriptor 解析与 NVIDIA 实际二进制布局不匹配 | **Critical** (Oracle review: 无硬件交叉验证) | Phase 0 用 NVIDIA PTX ISA §9.7 TensorMap 字段定义手工构造 descriptor 字节；标注为 unverified-against-hardware。Phase 3 e2e 测试若失败则追溯 descriptor 解析 |
| cluster mode 与现有 CTAContext 集成复杂 | High | Phase 0 子系统先 unit test 验证隔离行为，再 e2e 集成 |
| async queue 与 WarpState 集成产生 invariant 冲突 | High | `ptx-lessons-learned` §1 cross-module state translation 强制审计 |
| cute 模板 sm_100 fallback 到 sm_90 wgmma 代码 | Medium | 不保证 cute 编译时不引用 sm_90 头文件；只保证 emit 的 PTX 走 Blackwell |
| Phase 0 工程量大（~3000-4000 LoC，9 commits） | Medium | 9 个独立 commit（Oracle review 修正：5→9），每个子系统 1 commit + 4 个逐子系统集成微 commit，独立可 revert |
| sm_120 sparse variants 与 sm_100 fragment 不兼容 | Low | 预留 `cta_group::2` / `kind::*` 扩展点 |
| TMA host API 拦截策略不明确 | Medium | Phase 0 用 fake descriptor；ADR-0017 候选后续单独决策 |
| cute_rmsnorm 未来升级到 tcgen05 触发依赖 | Low | Phase 0-2 完成后才能升级 cute_rmsnorm |

## Design-Time Checklist (Lessons-Learned)

### 多 Phase 推进
- [x] Phase 拆分：4 个 Phase（0=基础设施，1=mma，2=ld/st+commit/wait，3=e2e+AGENTS）
- [x] 基线 worktree: 复用 `.worktrees/fix-pre-p0-baseline`
- [x] 失败处理: 任何已有测试回归 → 立即 revert 该 Phase，不混入后续 commit
- [x] Phase 0 子系统分独立 commit（5 commits = TMA / TMEM / cluster / async queue / 集成）

### 函数迁移完整性
- `tensor.cpp::WmmaHandler::processWmmaOperation` 当前实现：抛异常
- 本 change 把"抛异常"行替换为真实 Blackwell tcgen05 路径
- 所有 set_state / commit_pc 调用需在 design.md Migration Plan 列出行级 diff
- AGENTS.md / SPEC.md / X-Macro `ptx_op.def` 中所有 `WmmaHandler` / `wmma` / `tcgen05` 引用需 grep 一致

### 文档同步
- [x] `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS: 移除 `tensor.cpp (WmmaHandler)` 异常说明
- [x] 根 `AGENTS.md` 已知限制表: WMMA 条目从"抛异常" → "Blackwell tcgen05 已实现；pre-Blackwell 永久抛异常（ADR-0016）"
- [x] `tests/ptx/parser/test_wmma.cpp` 顶部"Known broken"注释: 修复后可移除
- [x] `docs/architecture/sm90_100.md`: §4 引用 ADR-0016
- [x] ADR README 索引: 添加 0016 entry

## Capabilities

### New Capabilities
- `wmma-tensor-core`: Blackwell `tcgen05.*` instruction set, including TMA + TMEM
  + cluster + async queue infrastructure. Modifies the existing
  `replace-silent-stub-failures` contract for the implemented Blackwell variants.

### Modified Capabilities
- `stub-explicit-failure`: The WMMA-Stub-Throws-Exception MUST requirement is
  **permanent** for pre-Blackwell variants per ADR-0016; Blackwell variants
  follow the real-execution path.

## Impact

**新建文件**:
- `src/ptxsim/memory/tma_descriptor.{h,cpp}` (Phase 0)
- `src/ptxsim/memory/tmem.{h,cpp}` (Phase 0)
- `src/ptxsim/cluster/{h,cpp}` (Phase 0)
- `src/ptxsim/async/tc_queue.{h,cpp}` (Phase 0)
- `tests/unit/memory/test_tma_descriptor.cpp` (Phase 0)
- `tests/unit/memory/test_tmem.cpp` (Phase 0)
- `tests/unit/cluster/test_cluster_mode.cpp` (Phase 0)
- `tests/unit/async/test_tc_queue.cpp` (Phase 0)
- `tests/unit/ptx/test_tcgen05_mma.cpp` (Phase 1)
- `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` (Phase 1)
- `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp` (Phase 2)
- `tests/e2e/kernel/test_blackwell_gemm.cu` (Phase 3)

**修改文件**:
- `src/ptxsim/instructions/wmma.cpp` (Phase 1-2: 抛异常 → tcgen05 实现)
- `src/ptxsim/instructions/AGENTS.md` (Phase 3)
- 根 `AGENTS.md` 已知限制表 (Phase 3)
- `src/CMakeLists.txt` (Phase 0 新增 4 个 source)
- `tests/unit/CMakeLists.txt` (Phase 0-1 新增 test 注册)
- `docs/architecture/sm90_100.md` (引用 ADR-0016)
- `docs/adr/README.md` (索引 0016 entry — done)
- `openspec/specs/stub-explicit-failure/spec.md` (delta spec)
- `openspec/specs/wmma-tensor-core/spec.md` (publish from change)

**影响范围**:
- cute_rmsnorm 等 e2e 测试 (无变化 — 不依赖 WMMA)
- cute / cutlass 模板编译 (无变化 — 已有 sm_90 fallback 代码抛异常是预期)
- 现有 165 ctest (无变化 — Phase 0 单元测试独立于现有测试)
- Multi-PTX warning (`PTX_WARN_EMU` Fix #3) (无变化)