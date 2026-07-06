# Audit Blackwell tcgen05 Infrastructure (TMA + TMEM + Cluster + TcQueue)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 change**: `archive/2026-07-06-implement-tcgen05-syntax-ir` (Change-1, archived)
> **4-Change 拆分**: 本 change 是第 2 步(共 4 步),**仅审计 4 个 Blackwell 底层子系统**(无源码修改)
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §6(artifacts-first)+ §7(Pre-impl review)+ Metis MR-3(Phase B 反应式设计已移除)

## Why

Change-1 建立了独立 tcgen05 命名空间(grammar + IR),但**未触及底层基础设施**。`src/ptxsim/memory/tma_descriptor.{h,cpp}`、`src/ptxsim/memory/tmem.{h,cpp}`、`src/ptxsim/cluster/cluster_context.{h,cpp}`、`src/ptxsim/async/tc_queue.{h,cpp}` 4 个子系统虽然在 `archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/` 中已 archive,但:

1. **TmaDescriptor 128 字节布局**有 29 处 `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13` 注释(17 在 .h,12 在 .cpp)
2. **TcQueue 集成**在 commit `c0fa43f` archive,**明确不调用 `set_state(BAR_SYNC)`**(`tc_queue.h:16-17` / `tc_queue.cpp:13-14` 注释:`NO set_state(BAR_SYNC) — TcQueue is not a CTA-level barrier`);`tcgen05.wait` handler 通过 `cta->tc_queue().wait(warp, 0, 1)`(`wmma.cpp:556`)使用独立 `is_blocked` + `status=Blocked` 路径,**不涉及** BAR_SYNC(per Change-2 Metis MR-1 修正)
3. **Cluster context 集成**在 commit `eb52af4 feat(cluster): wire ClusterContext into tcgen05 commit/wait (Fix #2)`(基础原语在更早的 `e513235 feat(sim): cluster arrive/wait primitives (Fix #7, simplified—no distributed smem)`);`tcgen05.commit/wait` 通过 `wmma.cpp:526-528` opt-in 调用 `cta_cluster_arrive`(其他指令未测试)
4. **跨子系统 pipeline 端到端 PTX 验证** = 0(每个子系统孤立测试)
5. **wmma.cpp handler-level UNVERIFIED** 9 处(`wmma.cpp:427, 449, 455, 467, 489, 506, 522, 538, 554`)直接关联 `tcgen05.ld/st/commit/wait` 正确性,Change-3 前置 blocker — per Change-2 Metis MR-5 扩 scope 包含

handler 实施(Change-3)前必须先**审计**基础设施可工作,否则 handler 调试会与基础设施 bug 混淆(per `ptx-lessons-learned` §3 "每个 Phase 独立可 revert")。

**本 change 只审计不修复**(Metis MR-3 修复):任何审计发现的问题通过**独立 follow-up change** 修复(如 `fix-tcgen05-tma-descriptor-offsets`),不在本 change 内修改源码 — 这样 Change-2 始终是 pure read-only 审计,diff 归零,易 revert。

## What Changes

### 审计 4 个子系统(无代码改动,纯验证)

| 子系统 | 源文件 | 行数 | 现有测试 | UNVERIFIED 注释数(实现级) |
|--------|--------|------|---------|--------------------------|
| TmaDescriptor | `src/ptxsim/memory/tma_descriptor.{h,cpp}` | 168+206 | 36 TEST_CASE(`tests/unit/memory/test_tma_descriptor.cpp`)| **29** (17 .h + 12 .cpp) |
| Tmem | `src/ptxsim/memory/tmem.{h,cpp}` | 49+61 | 19 TEST_CASE(`tests/unit/memory/test_tmem.cpp`)| 0 |
| ClusterContext | `src/ptxsim/cluster/cluster_context.{h,cpp}` | 54+82 | 16 TEST_CASE(`tests/unit/cluster/test_cluster_mode.cpp`)| 0 |
| TcQueue | `src/ptxsim/async/tc_queue.{h,cpp}` | 74+108 | 15 TEST_CASE(`tests/unit/async/test_tc_queue.cpp`)| 0 |
| wmma.cpp handlers | `src/ptxsim/instructions/wmma.cpp` (L320-565,handler 段) | 246 | 0 独立 TEST_CASE(handler 通过 Change-1 grammar/integration 覆盖)| **9**(`wmma.cpp:427, 449, 455, 467, 489, 506, 522, 538, 554`;排除 L62-317 的 256-entry reference table per D5 排除规则) |
| 跨子系统集成 | (无独立文件) | — | **2 TEST_CASE**(`tests/unit/cluster/test_cluster_tcgen05_integration.cpp`,**不是** `tests/integration/` per Metis MR-3 fix)| — |

> **MR-5 改动**:原列 4 子系统 → 现列 5 子系统 + 跨集成。wmma.cpp handlers 段是 Change-3 的直接前置依赖(`tcgen05.ld/st/commit/wait` 实现),必须纳入 readiness 评估。L62-317 的 256 fragment element reference entries 按 D5 排除规则归类为 Verified-Ref,不计入 UNVERIFIED 等级。

### 审计项

| 子系统 | 审计内容 | 输出 |
|--------|---------|------|
| TmaDescriptor | 128 字节布局与 PTX ISA §9.7.13 对齐、29 处 UNVERIFIED 分级(P0/P1/P2)、descriptor store 隔离性、≥10 swizzle/stride 组合覆盖 | 报告 §2.1 |
| Tmem | 256 slot × 128 byte 布局、CTA 隔离性、partial write no-clobber、越界检查 | 报告 §2.2 |
| ClusterContext | arrive/wait 同步语义、CTA 隔离、与 BarrierModule 集成 | 报告 §2.3 |
| TcQueue | commit-group counter 原子性、wait-aware 调度、**`NO set_state(BAR_SYNC)` 设计契约**(`tc_queue.h:16-17` / `tc_queue.cpp:13-14`) | 报告 §2.4 |
| wmma.cpp handlers | 9 处 handler-level UNVERIFIED 分级(`wmma.cpp:427, 449, 455, 467, 489, 506, 522, 538, 554`)、`tcgen05.ld/st/commit/wait` × 4 子系统的集成路径 | 报告 §2.5 |
| 跨子系统 pipeline | TmaDescriptor → Tmem → TcQueue 完整 pipeline 路径(含 wmma.cpp handler 调用链) | 报告 §2.6 |

### 跨 Change 协调

- **与 Change-3 (handlers)** 的契约:本 change 输出 "基础设施 readiness report"(L1/L2/L3 等级),Change-3 handler 实施前需先满足 ≥L2
- **与 Change-4 (cleanup)** 的契约:本 change 不修改源码,与 Change-4 互不干扰

## Non-Goals

### 显式拒绝(per ADR-0016 锁定)

- ❌ **不修改任何源码文件**(Metis MR-3 修复)— 若审计发现 bug,通过独立 `fix-*` change 修复
- ❌ 不实现 `cp.async.bulk.tensor.*`(TMA 加载指令)— 这与 `tcgen05.ld` 是不同指令,留待**独立 follow-up change**(`implement-cp-async-bulk-tensor`),非本 change 或 Change-3 范围
- ❌ 不实现 `tensormap.create/replace` host API 拦截(候选 ADR-0017)
- ❌ 不实现 `cuTensorMapEncodeTiled` host-side 拦截
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8
- ❌ 不修改 `tcgen05.*` grammar/IR(handler 在 Change-3)
- ❌ 不修改 `wmma.cpp`(留待 Change-4 删除) — **审计性只读**:`wmma.cpp` L320-565 handler 段纳入本 change 审计(per MR-5),但**不修改任何 handler 代码**;审计发现的 handler-level UNVERIFIED 不确定项通过独立 `fix-*` change 修复

### 范围限制

- **纯 read-only 审计**(无 commit 触达 `src/`)
- 仅在 `docs/audits/2026-07-XX-tcgen05-infra-audit.md` 输出报告
- 不修复 bug、不添加测试、不重构代码
- 性能对标不要求(仅 functional correctness)

### 不修改(明确列出)

- `src/ptxsim/memory/tma_descriptor.{h,cpp}`(只读)
- `src/ptxsim/memory/tmem.{h,cpp}`(只读)
- `src/ptxsim/cluster/cluster_context.{h,cpp}`(只读)
- `src/ptxsim/async/tc_queue.{h,cpp}`(只读)
- `src/ptxsim/core/cta_context.{h,cpp}`(只读)
- `src/ptxsim/instructions/wmma.cpp`(只读 — 审计 L320-565 handler 段 UNVERIFIED,但不修改)
- 任何 `tests/` 文件(只读 + 运行)
- 任何 `include/` 头文件(只读)

## Goals

### Phase A: 审计(无代码改动,1 个 commit)

1. 跑 `cmake --build build` 确保 baseline 编译通过
2. 跑 `ctest -R "unit_tma_descriptor|unit_tmem|unit_cluster_mode|unit_cluster_tcgen05_integration|unit_tc_queue" --output-on-failure` 记录 baseline(per Metis MR-2 修正:`ctest -L` 用 AND 语义,`-L "unit;memory"` 实际返回 0 tests;改用 `-R` 正则 OR 一次性枚举 5 个 ctest targets)
3. 阅读 **5 个子系统**(per MR-5 扩 scope:4 + wmma.cpp handlers L320-565)的 `.h`/`.cpp`,统计 UNVERIFIED 注释位置和数量(L62-317 fragment reference table 按 Decision 5 排除规则**不计入**)
4. 跑 `state-modification-audit` skill 验证 **`NO set_state(BAR_SYNC)` 设计契约**(`tc_queue.h:16-17` / `tc_queue.cpp:13-14` — **不是** ADR-0016 Decision 7,而是 tc_queue 模块内部 Decision 7,per MR-1 修正);验证 `wmma.cpp:556` 的 `tc_queue.wait` 调用不通过 `set_state(BAR_SYNC)`
5. 阅读 5 个子系统的测试,识别覆盖空白
6. 输出 `docs/audits/2026-07-XX-tcgen05-infra-audit.md` 报告

### Phase A Acceptance Criteria

- 报告包含 5 个子系统的 **readiness 等级**(L1=可工作/L2=需关注/L3=阻塞 Change-3,per Decision 4)
- 报告列出 38 个 UNVERIFIED 注释的 **分级**(29 TmaDescriptor + 9 wmma.cpp handlers,P0=影响 handler 正确性/P1=影响精度/P2=边缘 case,per Decision 5)
- 报告明确 "哪些 UNVERIFIED 必须 Change-3 实施 handler 前先修"(P0 列表)
- 报告明确 "哪些可推迟到 Change-3 handler 实施时一并验证"(P1 列表)
- 报告列出 "跨子系统 pipeline 覆盖空白"(TmaDescriptor → Tmem → TcQueue 端到端测试缺口)

### 不实施任何 Phase B / Phase C

- ❌ 不实施 "Phase B: 补全缺口"(已移除,任何修改通过独立 change)
- ❌ 不实施 "Phase C: 测试覆盖"(已移除,新增测试属于 `fix-*` 或 follow-up change)

## Capabilities

### New Capabilities

- `tcgen05-infra-audit`: 4 个 Blackwell 子系统的审计报告 + readiness 等级 + UNVERIFIED 分级

### Modified Capabilities

- 无

## Impact

### 影响的文件(预计,纯新增文档)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `docs/audits/2026-07-XX-tcgen05-infra-audit.md` | 新增 | +400 |
| `openspec/changes/extend-blackwell-tcgen05-infra/` | 新增(proposal + design + tasks + spec) | +600 |
| **总计** | | **+1000** |

### 影响的依赖

- `state-modification-audit` skill(per ptx-lessons-learned §1,跨模块状态翻译)— 纯 read-only 分析
- `ptx-grammar-modification` skill(若 E2E 需 cuobjdump 提取)— 审计报告可选引用

### 不影响的依赖(本 change 范围外)

- `src/ptxsim/instructions/wmma.cpp`(change-3 scope)
- grammar/IR 命名空间(Change-1 已完成,不动)
- `src/ptxsim/core/cta_context.{h,cpp}`(已有集成,不动)
- 任何 `tests/` 文件(本 change 不新增/修改测试)

### 影响的文档

- `docs/audits/2026-07-XX-tcgen05-infra-audit.md`(主要交付物)
- 根 `AGENTS.md` 已知限制表(更新 cluster 状态,若 readiness = L3)
- `docs/adr/0016-blackwell-only-tcgen05.md`(追加审计 commit 引用)

## Design-Time Checklist (Lessons-Learned)

### 审计完整性

- [x] 4 个子系统源文件路径已列出
- [x] 现有测试数量已修正(TMA 36,Tmem 19,cluster 16,tc_queue 15,**不是**原 proposal 声称的 18)
- [x] 跨子系统集成测试仅 2 TEST_CASE(覆盖率不足,审计报告需明确)
- [x] UNVERIFIED 注释总数 29(17 in .h + 12 in .cpp)
- [x] 不修改源码(纯 read-only 审计)— 违反则立即 abort

### 跨 Change 协调

- [x] 与 Change-3 的契约:本 change 输出 readiness report,Change-3 实施前需 ≥L2
- [x] `cp.async.bulk.tensor` 归 `implement-cp-async-bulk-tensor` 独立 follow-up(非 Change-3)
- [x] 任何审计发现的问题通过独立 `fix-*` change 修复,不在本 change 内

### 多 Phase 推进

- [x] 仅 1 个 Phase(Phase A 审计)— 简化 commit 粒度
- [x] 不需要多 Phase 拆分(本 change 是 pure read-only)
- [x] 基线 worktree 计划:`.worktrees/baseline-tcgen05-audit`(per `ptx-lessons-learned` §4)
- [x] 失败处理策略:若有 1 个子系统 readiness = L3(阻塞),在报告中明确并建议新建 `fix-*` change

### 文档同步

- [x] 审计报告路径已列出
- [x] AGENTS.md 同步项已规划
- [x] ADR 追加段落已规划

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证审计范围、文件路径、测试数字(MR-1~5 全部解决后再 apply)
- [ ] 跑 `wc -l src/ptxsim/memory/tma_descriptor.cpp src/ptxsim/memory/tmem.cpp src/ptxsim/cluster/cluster_context.cpp src/ptxsim/async/tc_queue.cpp src/ptxsim/instructions/wmma.cpp` 确认行数
- [ ] 跑 `grep -c "UNVERIFIED-AGAINST-HARDWARE" src/ptxsim/memory/tma_descriptor.{h,cpp}` 确认 **29**
- [ ] 跑 `awk 'NR>=320 && NR<=565 && /UNVERIFIED-AGAINST-HARDWARE/' src/ptxsim/instructions/wmma.cpp | wc -l` 确认 **9**(handler-level,排除 L62-317 reference)
- [ ] 跑 `ctest -R "unit_tma_descriptor|unit_tmem|unit_cluster_mode|unit_cluster_tcgen05_integration|unit_tc_queue" --output-on-failure` 记录 baseline 通过数(per MR-2 修正:用 `-R` 而非 `-L`)
