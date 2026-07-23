# Phase 1-3: Blackwell tcgen05 Handler Implementation

> **架构决策**: 本 change scope 由 [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) 锁定。
>
> **前置 change**: `implement-wmma-tensor-core-phase-0-infra` (archived 2026-07-04)
> **Ref**: archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/
>
> **本 change 交付**: Phase 1 (mma fragment arithmetic) + Phase 2 (ld/st + commit/wait async) +
> Phase 3 (e2e GEMM + AGENTS sync + spec publish) = **5 commits**
>
> **前置依赖**:
> - `replace-silent-stub-failures` (archived 2026-07-04)：建立 `WmmaHandler` 抛 `UnsupportedInstructionException` 合约
> - ADR-0016（Accepted 2026-07-04）：架构依据
> - `implement-wmma-tensor-core-phase-0-infra` (archived 2026-07-04)：TMA + TMEM + cluster + TcQueue 基础设施

## Why

`replace-silent-stub-failures` (archived 2026-07-04) 把
`WmmaHandler::processWmmaOperation` 的 silent no-op 改为
`throw UnsupportedInstructionException`，建立了"未实现 stub 必须显式失败"的合约。
基础设施前置 change `implement-wmma-tensor-core-phase-0-infra` 交付了 4 个 Blackwell
新范式子系统 (TMA + TMEM + cluster + TcQueue)。

本 change 是该合约的"补完"——把 Blackwell 现代张量核心指令集**真正实现**出来，
基于已 archive 的基础设施：

1. **vision purity**：放弃 pre-Blackwell legacy 兼容性，全力投入 Blackwell 现代范式
2. **infrastructure first**：Phase 0 基础设施先建，避免后期架构返工
3. **future-readiness**：Blackwell 是 NVIDIA 2024-2026 主推架构，下一代仍在同一范式

## What Changes

### Phase 1：tcgen05.mma fragment arithmetic

`src/ptxsim/instructions/wmma.cpp`（从 `tensor.cpp` rename）：
- 解析 `tcgen05.mma.cta_group::1.kind::f16` 指令变体
- 实现真实 fragment 算术（m64nNk 等）
- 复用 `include/ptxsim/utils/half_utils.h`
- 委托给 Phase 0 archive 的 `TcQueue::enqueue_mma`
- 每个 fragment 输出元素 `// UNVERIFIED-AGAINST-HARDWARE` 注释 + PTX ISA §9.7.13 行号引用

### Phase 2：tcgen05.ld / st + commit / wait

`src/ptxsim/instructions/wmma.cpp`：
- `tcgen05.ld` / `tcgen05.st` → TMA descriptor + TMEM
- `tcgen05.commit` → `TcQueue::commit(group_id)`
- `tcgen05.wait` → `TcQueue::wait(group_id)` (复用 BAR_SYNC path per design.md Decision 7)

### Phase 3：e2e GEMM kernel + AGENTS/spec 同步

`tests/e2e/kernel/test_blackwell_gemm.cu`：
- Cute tcgen05 风格 16×16 GEMM kernel, target sm_100
- 使用 vendored Cute headers (`bench/cute/include/`) — `tests/e2e/kernel/CMakeLists.txt` 添加 include path
- 验证 fragment 算术正确：16×16 矩阵乘 `C[i][j] = sum_k A[i][k] * B[k][j]`, host 端对比, f32 rounding tolerance

`src/ptxsim/instructions/AGENTS.md` + 根 `AGENTS.md`：
- 移除 WMMA stub 描述
- 标注 Blackwell tcgen05 已实现
- pre-Blackwell 永久抛异常（ADR-0016）

`openspec/specs/wmma-tensor-core/spec.md` → publish 到 `openspec/specs/`

## Non-Goals

### 显式拒绝（ADR-0016 锁定）

- **不实现 `wmma.mma.sync.*`**（sm_70 / sm_75 / sm_80 / sm_86）— 永久抛异常
- **不实现 `wgmma.async.*`**（sm_90）— 永久抛异常
- **不实现 `mma.sync.*`**（sm_70+ 通用路径）— 永久抛异常
- **不实现 pre-Blackwell cute / cutlass template instantiation**

这些路径继续抛 `UnsupportedInstructionException`（与 `replace-silent-stub-failures`
合约一致），不视为 bug。

### 范围限制

- **sm_120 sparse / FP4 / mxfp8**：留在同一范式，但单独 propose，每个特性一个 change
- **性能对标**：仅 functional correctness，不追求 cycle-accurate
- **SASS 层语义**：仍 PTX-level interpretation
- **mma.sp 稀疏变种**：本 change 完成后单独 change
- **TMA host API 拦截**：留给 ADR-0017
- **distributed_smem**：留给 cta_group::2 后续 change（候选 ADR-0018）
- **TMA / TMEM / cluster / TcQueue**：本 change **不实现**，已在 phase-0-infra archive

### 不修改

- `UnsupportedInstructionException` / `ExecutionStateException` 类定义本身
- X-Macro `ptx_op.def`（仍 `S_WMMA` → `WmmaHandler::processWmmaOperation`）
- `replace-silent-stub-failures` 已建立的合约
- `implement-wmma-tensor-core-phase-0-infra` 已 archive 的基础设施

## Goals

### Phase 1 目标
1. `tcgen05.mma.cta_group::1.kind::f16` 真实 fragment arithmetic
2. 32 lane 输出片段元素正确（per NVIDIA PTX ISA §9.7.13，每个 `// UNVERIFIED-AGAINST-HARDWARE` 注释）
3. 集成测试覆盖 uniform warp + 验收 mma → TMEM 写入（直接读 TMEM, no commit/wait）
4. 单元测试 + 集成测试 PASS

### Phase 2 目标
1. `tcgen05.ld` / `st` 与 TMA descriptor + TMEM 集成
2. `tcgen05.commit` / `wait` 同步原语正确（复用 Phase 0.4 TcQueue 框架）
3. 完整 mma 序列（ld → mma → commit → wait → st）端到端通过

### Phase 3 目标
1. Cute tcgen05 风格 16×16 GEMM kernel e2e 通过
2. AGENTS.md 同步反映新行为
3. spec 从 change 移到 main specs
4. Phase 1-3 archive 永久生效（per ADR-0016，pre-Blackwell 抛异常行为不变）

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| TMA fragment layout 解读错误无硬件交叉验证 | **Critical** (Oracle review: hand-computed reference correctness is self-consistency) | 每个 fragment 元素 `// UNVERIFIED-AGAINST-HARDWARE` 注释 + 手工对照 PTX ISA §9.7.13 latest；本项目为 Blackwell tcgen05 fragment layout 解释的 primary source（GPGPU-Sim / MGPUSim / Accel-Sim 尚未实现 sm_100 tcgen05） |
| cute 模板 sm_100 fallback 到 sm_90 wgmma | Medium | 不保证 cute 编译时不引用 sm_90 头；只保证 emit 的 PTX 走 Blackwell（emit 后由 WMMA handler per-qualifier 分发） |
| TcQueue BAR_SYNC reuse 集成 bug | High | 复用 Phase 0 archive 已审计的 BAR_SYNC path；只新增 `tcgen05.wait` handler 调用 `TcQueue::wait` 即可 |
| cutlass 3.x GEMM kernel e2e 编译失败 | Medium | cute header spike（前置 change Open Question #5）已验证可行性；失败则 propose `fix-cute-sm100-headers` change |

## Quality Gates (Phase 1-3 内部)

> **Phase 1-3 archive 的硬门** = 下列 Gate 全部通过

| Gate | 适用 | 命令 | 阈值 |
|------|------|------|------|
| **P1-3.G1** | Phase 1 → Phase 2 | `grep -c "UNVERIFIED-AGAINST-HARDWARE" src/ptxsim/instructions/wmma.cpp` | **`≥ 256`** (32 lane × 8x4 矩阵 per `tasks.md:153`) |
| **P1-3.G2** | Phase 1 → Phase 2 | `ctest -R "tcgen05_mma_sync"` | PASS |
| **P1-3.G3** | Phase 2 → Phase 3 | `ctest -R "tcgen05_ld_st_commit"` | PASS (per tasks.md:192) |
| **P1-3.G4** | Phase 3 之前 | `./scripts/sanity.sh` | 0 unexpected FAIL |
| **P1-3.G5** | Phase 3 之前 | `./tests/ptx/test_all_ptx.sh` | 0 FAIL (严禁 ctest 代替) |
| **P1-3.G6** | Phase 3 archive | baseline worktree 对比 | 0 new FAIL |
| **P1-3.G7** | Phase 3 archive | `git ls-files openspec/changes/implement-wmma-tensor-core-tcgen05/` | 非空 |

## Capabilities

### New Capabilities

- `wmma-tensor-core`: Blackwell `tcgen05.*` instruction set feature 层。包括:
  - `tcgen05.mma.cta_group::1.kind::f16` fragment arithmetic
  - `tcgen05.ld/st` 与 TMA + TMEM 集成
  - `tcgen05.commit/wait` 异步流（基于 Phase 0-archive TcQueue）

  基础设施层（TMA + TMEM + cluster + TcQueue）由
  `implement-wmma-tensor-core-phase-0-infra` 交付，本 change 在 archive 后依赖之。

### Modified Capabilities

- `stub-explicit-failure`: 修改 — pre-Blackwell 抛异常永久（per ADR-0016）;
  Blackwell `tcgen05.*` 走真实执行路径。delta spec in `specs/stub-explicit-failure/spec.md`。

> **未在此列出的 capability**：phase-0-archive 交付的 `wmma-tensor-core-infrastructure`
> (TMA + TMEM + cluster + TcQueue 4 子系统类型) 不被本 change 修改。本 change **依赖**
> 其 API 但不修改其 spec。本 change 实施前 phase-0-archive 必须先完成。

## Impact

**新建文件**:
- `tests/unit/ptx/test_tcgen05_mma.cpp` (Phase 1.1)
- `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` (Phase 1.2)
- `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp` (Phase 2.2)
- `tests/e2e/kernel/test_blackwell_gemm.cu` (Phase 3.1)

**修改文件**:
- `src/ptxsim/instructions/wmma.cpp` (rename `tensor.cpp` → `wmma.cpp`, Phase 1.1):
  - 抛异常 → tcgen05 实现
  - 每 fragment 元素 `// UNVERIFIED-AGAINST-HARDWARE` 注释
- `src/CMakeLists.txt:103` (rename target `ptxsim/instructions/tensor.cpp` → `ptxsim/instructions/wmma.cpp`)
- `src/ptxsim/instructions/AGENTS.md` (Phase 3.3): 移除 WMMA stub 异常说明
- 根 `AGENTS.md` 已知限制表 (Phase 3.4): WMMA 条目从"抛异常" → "Blackwell tcgen05 已实现"
- `tests/e2e/kernel/CMakeLists.txt` (Phase 3.2): 添加 `bench/cute/include/` include path
- `openspec/specs/wmma-tensor-core/spec.md` (Phase 3 archive): publish 到 main specs

**影响范围**:
- cute_rmsnorm 等 e2e 测试（无变化 — 不依赖 WMMA, 也不依赖 cute headers per phase-0 Open Question #5）
- 现有 123 labeled ctest (73 unit + 42 integration + 8 e2e；无变化 — Phase 1-3 仅修改 wmma.cpp handler，不动现有 test paths)
- Multi-PTX warning (`PTX_WARN_EMU` Fix #3)（无变化）

## Open Questions

(继承自 phase-0 change，归档时由 phase-1-3 change 跟踪)
1. **TMA host API 拦截策略** — separate ADR-0017 (deferred)
2. **sm_120 sparse / FP4 / mxfp8** — separate changes (deferred)
3. **cute_rmsnorm 升级到 tcgen05** — blocked until phase-1-3 archive (Phase 3 后 follow-up)
4. **async queue priority vs scheduler** — 需硬件数据 calibrate (deferred)
