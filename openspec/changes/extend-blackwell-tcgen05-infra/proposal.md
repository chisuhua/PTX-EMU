# Extend Blackwell tcgen05 Infrastructure (TMA + TMEM + Cluster + TcQueue Audit)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 change**: `archive/2026-07-06-implement-tcgen05-syntax-ir` (Change-1, archived)
> **4-Change 拆分**: 本 change 是第 2 步(共 4 步),审计并补全 Blackwell tensor core 所需的 4 个底层子系统
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §6(artifacts-first)+ §7(Pre-impl review)

## Why

Change-1 建立了独立 tcgen05 命名空间(grammar + IR),但**未触及底层基础设施**。`src/ptxsim/memory/tma_descriptor.{h,cpp}`、`src/ptxsim/memory/tmem.{h,cpp}`、`src/ptxsim/cluster/cluster_context.{h,cpp}`、`src/ptxsim/async/tc_queue.{h,cpp}` 4 个子系统虽然在 `archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/` 中已 archive,但:

1. **TmaDescriptor 128 字节布局**(`tma_descriptor.h:1-29`)有 32+ 处 `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13` 注释
2. **TcQueue 集成**在 commit `c0fa43f` archive,但 `tcgen05.wait` handler 仍复用 BAR_SYNC path(per archive `archive/2026-07-04-implement-wmma-tensor-core-tcgen05/tasks.md`)
3. **Cluster context** 集成在 commit `eb52af4`,但仅 `tcgen05.commit/wait` 涉及(其他指令未测试)
4. **0 个基础设施子系统的端到端 PTX 验证**

handler 实施(Change-3)前必须先验证基础设施可工作,否则 handler 调试会与基础设施 bug 混淆(per `ptx-lessons-learned` §3 "每个 Phase 独立可 revert")。

## What Changes

### 审计 4 个子系统(无代码改动,纯验证)

| 子系统 | 文件 | 审计项 |
|--------|------|--------|
| TmaDescriptor | `src/ptxsim/memory/tma_descriptor.{h,cpp}` | 128 字节布局验证(对 cuobjdump -xptx 输出)、descriptor store 隔离性、≥10 swizzle/stride 组合覆盖 |
| Tmem | `src/ptxsim/memory/tmem.{h,cpp}` | 256 slot × 128 byte 布局、CTA 隔离、partial write no-clobber |
| ClusterContext | `src/ptxsim/cluster/cluster_context.{h,cpp}` | arrive/wait 同步、distributed_smem(若 cta_group::2) |
| TcQueue | `src/ptxsim/async/tc_queue.{h,cpp}` | commit-group counter、wait-aware 调度、BAR_SYNC 集成 |

### 补全缺口(per 审计发现)

预计可能的缺口:
- TmaDescriptor 偏移需硬件 dump 验证(从 NVIDIA docs 推断)
- TcQueue 与现有 `WarpState::set_warp_state(BAR_SYNC)` 路径的 invariant 需 `state-modification-audit` skill 验证
- Cluster context 缺 `cta_group::2` distributed_smem 场景(per ADR-0016 Open Question #2)
- TMEM 与 SMEM 边界检查(防止越界)

### 三套测试覆盖

| 类型 | 文件 | 范围 |
|---|---|---|
| 单元 | `tests/unit/memory/test_tma_descriptor.cpp`(已有,18 TEST_CASE)+ `test_tmem.cpp`(已有,18 TEST_CASE)+ 新增 `test_cluster_context.cpp` + `test_tc_queue.cpp` |
| 集成 | `tests/integration/memory/test_tma_descriptor_*.cpp` + `test_tmem_*.cpp` + `test_cluster_*.cpp` + `test_tc_queue_*.cpp` |
| E2E | `tests/e2e/kernel/test_tma_descriptor_e2e.cu` + `test_tmem_e2e.cu`(用真实 cuobjdump 提取的 PTX) |

## Non-Goals

### 显式拒绝(per ADR-0016 锁定)

- ❌ 不实现 `cp.async.bulk.tensor.*`(TMA 加载指令,留待 change-3 实施 handler)
- ❌ 不实现 `tensormap.create/replace` host API 拦截(候选 ADR-0017)
- ❌ 不实现 `cuTensorMapEncodeTiled` host-side 拦截
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8
- ❌ 不修改 `tcgen05.*` grammar/IR(handler 在 change-3)
- ❌ 不修改 `wmma.cpp`(留待 change-4 删除)

### 范围限制

- 仅在已有 `tests/unit/memory/test_*.cpp` 基础上**扩展**测试,不动现有实现
- 若发现基础设施 bug,只修测试或文档,**不修实现**(避免与 change-3 冲突)
- 性能对标不要求(仅 functional correctness)

### 不修改

- `src/ptxsim/memory/tma_descriptor.{h,cpp}`(已有,18 个 TEST_CASE 已 PASS)
- `src/ptxsim/memory/tmem.{h,cpp}`(已有,18 个 TEST_CASE 已 PASS)
- `src/ptxsim/cluster/cluster_context.{h,cpp}`(已有,arrive/wait 已实现)
- `src/ptxsim/async/tc_queue.{h,cpp}`(已有,commit/wait 已实现)
- `src/ptxsim/core/cta_context.{h,cpp}`(已有 TmaDescriptorStore + Tmem 集成)

## Goals

### Phase A: 审计(无代码改动)

1. 跑 `ctest -L "unit;memory|unit;barrier"` 全量通过
2. 阅读 4 个子系统的 `.h` 头文件,记录 `// UNVERIFIED-AGAINST-HARDWARE` 注释位置
3. 跑 `state-modification-audit` skill 验证 TcQueue 与 WarpState 的 invariant
4. 输出 `docs/audits/2026-07-XX-tcgen05-infra-audit.md` 报告

### Phase B: 补全(per 审计发现)

1. 若 TmaDescriptor 偏移需修正:`docs/dev-process/lessons-learned.md` 追加 §23
2. 若 TcQueue 与 BAR_SYNC 冲突:refactor(独立 commit,per `ptx-lessons-learned` §3)
3. 若 ClusterContext 缺 `cta_group::2`:补全
4. 若 TMEM 边界检查缺失:补全

### Phase C: 测试覆盖

1. 单元测试:每个子系统 ≥20 TEST_CASE
2. 集成测试:跨子系统协作(如 `TmaDescriptor → Tmem → TcQueue` 完整 pipeline)
3. E2E 测试:用真实 cuobjdump 提取的 PTX(per `ptx-grammar-modification` skill)

## Capabilities

### New Capabilities

- `tcgen05-infra-audit`: 4 个 Blackwell 子系统的审计报告 + 验证
- `tcgen05-infra-tests`: 单元/集成/E2E 三套测试覆盖 4 个子系统
- `tcgen05-infra-pipeline`: 跨子系统 pipeline 集成测试(若审计发现需要)

### Modified Capabilities

- `wmma-tensor-core`: 审计后,可能需要更新 spec(若发现子系统功能 gap)

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `docs/audits/2026-07-XX-tcgen05-infra-audit.md` | 新增 | +200 |
| `docs/dev-process/lessons-learned.md` | 追加 §23 | +50 |
| `tests/unit/memory/test_tma_descriptor.cpp` | 扩展(若发现 bug) | +50 |
| `tests/unit/memory/test_tmem.cpp` | 扩展(若发现 bug) | +30 |
| `tests/unit/cluster/test_cluster_context.cpp` | 新增(若缺) | +100 |
| `tests/unit/async/test_tc_queue.cpp` | 新增(若缺) | +150 |
| `tests/integration/memory/test_tma_pipeline.cpp` | 新增 | +200 |
| `tests/e2e/kernel/test_tma_descriptor_e2e.cu` | 新增 | +100 |
| `src/ptxsim/cluster/cluster_context.{h,cpp}` | 修改(若 cta_group::2 缺) | +200 |
| `src/ptxsim/async/tc_queue.{h,cpp}` | 修改(若 BAR_SYNC 冲突) | +100 |
| **总计** | | **+1180** |

### 影响的依赖

- `state-modification-audit` skill(per ptx-lessons-learned §1,跨模块状态翻译)
- `ptx-grammar-modification` skill(若 E2E 需 cuobjdump 提取)
- `oracle-prompting` skill(若 TmaDescriptor 偏移需硬件验证)

### 不影响的依赖(本 change 范围外)

- `src/ptxsim/instructions/wmma.cpp`(change-3 scope)
- grammar/IR 命名空间(Change-1 已完成,不动)
- `src/ptxsim/core/cta_context.{h,cpp}`(已有集成,不动)

### 影响的文档

- `docs/audits/2026-07-XX-tcgen05-infra-audit.md`(新增)
- `docs/dev-process/lessons-learned.md`(追加 §23)
- 根 `AGENTS.md` 已知限制表(更新 cluster 状态)
- `src/ptxsim/cluster/AGENTS.md`(若补全 cta_group::2)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性(类比 Checklist A)

- [x] Baseline 函数清单:`tma_descriptor.cpp` 8 个 public function + `tmem.cpp` 4 个 + `cluster_context.cpp` 6 个 + `tc_queue.cpp` 8 个
- [x] 锁点审计:`grep -n "lock_guard\|unique_lock" src/ptxsim/memory/ src/ptxsim/cluster/ src/ptxsim/async/`(已确认 4 个子系统都遵循 lessons-learned §2 无递归锁)
- [x] 跨模块状态翻译:`TcQueue::wait` → `WarpState::set_warp_state(BAR_SYNC)`(per ADR-0016 Decision 7)需 audit
- [x] invariant 清单:per-warp ordering、CTA 隔离、commit-group counter 原子性

### 多 Phase 推进

- [x] Phase 拆分:A 审计 → B 补全 → C 测试,每 Phase 独立 commit
- [x] 基线 worktree 计划:`.worktrees/baseline-tcgen05-infra`(per `ptx-lessons-learned` §4)
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 Phase

### 文档同步

- [x] 审计报告路径已列出
- [x] lessons-learned §23 预留
- [x] AGENTS.md 同步项已列出

### 实施前必跑

- [ ] **Metis pre-implementation review**:验证审计范围、LoC 估算、test gap 数字
- [ ] 验证 `wc -l tests/unit/memory/test_tma_descriptor.cpp tests/unit/memory/test_tmem.cpp` 数字
- [ ] 验证 4 个子系统的 baseline 测试通过(per change-1 baseline)
- [ ] 验证 `ctest -L "unit;memory" --output-on-failure` 全绿
