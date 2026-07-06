## Context

Change-1 (archived) 建立了独立 tcgen05 命名空间,handler 实施(Change-3)前需先**审计** **5 个子系统**(原 4 + wmma.cpp handlers per **MR-5 scope 扩展**)。本 change 是 pure read-only 审计,无源码改动(Metis MR-3 修复),任何发现通过独立 `fix-*` change 处理。

当前状态(per Change-1 Metis MR-4 + Change-2 Metis MR-1/2/7 修正):TmaDescriptor 36 TEST_CASE(wc 168+206)、Tmem 19(wc 49+61)、Cluster 16(wc 54+82)、TcQueue 15(wc 74+108)、wmma.cpp handlers 0 独立(Change-1 grammar 覆盖)、跨子系统集成 2。**29 处**实现级 UNVERIFIED-AGAINST-HARDWARE(TmaDescriptor 17 .h + 12 .cpp)+ **9 处** handler-level UNVERIFIED(`wmma.cpp:427, 449, 455, 467, 489, 506, 522, 538, 554`)。L62-317 的 256 fragment reference table per **Decision 5 排除规则** 不计入分级。

目标:1 commit 输出 `docs/audits/2026-07-XX-tcgen05-infra-audit.md` 报告,标注每子系统 readiness(L1/L2/L3)+ UNVERIFIED 分级(P0/P1/P2)+ aggregate readiness,不修改任何源码。

## Goals / Non-Goals

**Goals**: 5 子系统审计 + 38 UNVERIFIED 实现级注释分级(29 TmaDescriptor + 9 wmma.cpp handlers) + cross-subsystem pipeline 覆盖空白识别 + aggregate readiness 判定。

**Non-Goals**: 不修改任何源码(per Metis MR-3)、不实施任何 fix(独立 `fix-*` change)、不实施 handler(Change-3)。`wmma.cpp` 仅审计性只读(L320-565 handler 段纳入审计,但**不修改**任何代码);L62-317 fragment reference table 不计入分级(per Decision 5 排除规则)。

## Decisions

### D1: 审计策略 — 阅读 + 跑测试 + 跑 `state-modification-audit` skill,无实验

**采纳**: 5 子系统的 `.h`/`.cpp` 静态阅读 + 跑 `ctest` baseline(用 `-R` 正则,per MR-2)+ `state-modification-audit` skill 验证 **`NO set_state(BAR_SYNC)` 不变量**(`tc_queue.h:16-17` / `tc_queue.cpp:13-14` — 这是 tc_queue 模块内部 Decision 7,**不是** ADR-0016 Decision 7,per **MR-1 修正**)。

**拒绝**: 不需要真实 GPU 实验(无访问,`cuobjdump -xptx` 不可用)。

### D2: 报告结构 — per 子系统章节 + readiness 等级 + UNVERIFIED 分级 + cross-subsystem pipeline

**采纳**: **5 章节**(per 子系统) + 1 章节(cross-subsystem pipeline) + 1 章节(aggregate readiness + 发现的问题 + 推荐 `fix-*` change)。

### D3: 1 个 commit(纯文档,无需分 Phase)

**采纳**: 1 个 commit = 审计报告。

**理由**: per Metis F.1,本 change 极简,避免 Change-1 的 "3 atomic commits 变 6 个" 反模式。

### D4: L1 / L2 / L3 Readiness Rubric(per Metis MR-3,2026-07)

每个子系统审计后给一个 **readiness 等级**,用于 Change-3(handlers)评估前置依赖:

| 等级 | 含义 | 判定标准(全部满足) | Change-3 决策 |
|------|------|---------------------|---------------|
| **L1** | working / 可工作 | (a) 相关 ctest target 全绿;(b) **零** P0 UNVERIFIED;(c) 代码路径覆盖所有公共 API | 可直接依赖;无需额外审计 |
| **L2** | needs-attention / 需关注 | (a) 相关 ctest target 全绿;(b) **1-2 处** P0 UNVERIFIED;(c) P0 项已写入独立 `fix-*` change backlog 且有 owner | 可依赖;**必须并行** fix-* work;Change-3 完成前确认 fix-* 有进展 |
| **L3** | blocks / 阻塞 Change-3 | (a) 相关 ctest target **有失败**;或 (b) **≥3 处** P0 UNVERIFIED;或 (c) P0 UNVERIFIED 涉及根本 invariant 缺失(如 `set_state` 漏调用) | **必须 wait for** `fix-*` change 完成 |

**L2 → L3 升级规则**:若 L2 子系统超过 2 个 Phase 未推进 `fix-*` change,自动升级 L3(per lessons-learned §3 跨 Phase invariant 冲突)。

> **Phase 定义**(per NI-6 澄清):**Phase = OpenSpec change 生命周期**(从 propose → accepted → active → archived 视为一个 Phase)。当前 Phase = Change-2 (extend-blackwell-tcgen05-infra);Change-3 archive 之后 L2 仍未推进 → 自动升级 L3。

**aggregate readiness** = 取 4+1 子系统中最低等级(min-rule)。Change-3 可开始 = aggregate ≥ L2。

### D5: P0 / P1 / P2 UNVERIFIED 判定准则(per Metis MR-4,2026-07)

每个 UNVERIFIED 注释给一个 **优先级**,决定是否阻塞 readiness:

| 级别 | 影响维度 | 判定准则(满足任一即为此级) | 修复时序 |
|------|---------|---------------------------|---------|
| **P0** | handler 正确性 | (a) UNVERIFIED 涉及 handler 直接调用的 invariant 缺失(e.g. `cta->tc_queue().commit(1)` 的 `group_id=1` 硬编码);或 (b) UNVERIFIED 涉及核心数据结构 size/offset(e.g. 128-byte transfer);或 (c) UNVERIFIED 涉及同步原语(e.g. `arrive/wait` 计数) | **必须** Change-3 之前修 |
| **P1** | 数据精度 | (a) UNVERIFIED 涉及 fragment element 位置/索引准确性(e.g. "fragment element lane X C[Y][Z]");或 (b) UNVERIFIED 涉及位字段布局但不影响功能正确性;或 (c) UNVERIFIED 涉及 swizzle/stride 组合但有 fallback 路径 | Change-3 可并行 fix;不留技术债 |
| **P2** | 边缘 case | (a) UNVERIFIED 涉及罕见路径(zero-stride, border region);或 (b) UNVERIFIED 涉及罕见 type 组合(f16→f32 mixed);或 (c) UNVERIFIED 涉及测试覆盖空白而非实现 gap | 可延后到 Change-4 或独立 cleanup |

**排除规则**:以下情形**不计入** P0/P1/P2 等级(自动归 "Reference / Verified-Ref"):

- **(a)** 自动生成的参考数据表(e.g. `wmma.cpp:62-317` 的 256-entry `fragment element lane X C[Y][Z]` table — 这是 PTX ISA §9.7.13 fragment 布局的静态参考,非实现 UNVERIFIED)
- **(b)** 注释直接引用 PTX ISA section 而无具体 invariant 缺失**且位于 reference/table 上下文(非 handler 实现代码体内)**(per NI-5 fix:`wmma.cpp:427, 467, 506, 538` 四处 bare ISA-reference UNVERIFIED 位于 handler 函数体内,**仍计入**实现级分级;只有位于 reference data 表(如 L62-317)的 bare ISA-reference 才排除)
- **(c)** 测试 fixture / golden values

**实现级 vs Reference 区分规则**:UNVERIFIED 注释所在的代码上下文是关键——
- 注释紧邻 `void execute_*` 函数体开头之后(handler 实现代码)→ 实现级(无论注释描述详略)
- 注释位于数组初始化、struct member 列表、或独立 reference 段 → Reference(自动归 Verified-Ref,不计入分级)

**L2/L3 判定中的 P0 阈值**:L2 = ≤2 P0,L3 = ≥3 P0(per D4)。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: 审计发现架构级 bug(需跨文件修改) | 报告标注为 "需独立 `fix-*` change",本 change 不修 |
| **R2**: 报告过长(>500 LoC) | 控制在 400 LoC 内,详细数据用表格 |
| **R3**(per MR-5):wmma.cpp handler 段纳入审计可能增加 Change-3 阻塞面 | L62-317 fragment reference table 按 Decision 5 排除规则不计入分级;仅 9 处 handler-level UNVERIFIED 评估;若 readiness = L3,推荐新建 `fix-wmma-tcgen05-handler-unverified` change |
| **R4**(per MR-1):TcQueue ↔ BAR_SYNC 关系误解 | `tc_queue.h:16-17` 是事实声明,审计验证 grep 0 match + 文档 contract 不是 invariant 验证 |
| **R5**(per MR-7):cluster commit hash 误记录风险 | `eb52af4`(Fix #2 cluster 集成)+ `e513235`(Fix #7 基础原语)是两个独立 commit,proposal 已区分;审计需覆盖**两个** commit 引入的代码路径 |

## Migration Plan

1. 跑 baseline `ctest -R "unit_tma_descriptor|unit_tmem|unit_cluster_mode|unit_cluster_tcgen05_integration|unit_tc_queue" --output-on-failure` 记录(per Metis MR-2 修正:`-L` 用 AND 语义,改用 `-R` 正则枚举 5 个 targets)
2. 跑 `state-modification-audit` skill(per ptx-lessons-learned §1)
3. 写 `docs/audits/2026-07-XX-tcgen05-infra-audit.md`
4. `git add docs/audits/` + commit
5. `openspec archive` + commit archive

### 回退策略

`git revert HEAD` 回到 good state。

## Open Questions

无(per MR-3/4:L1/L2/L3 与 P0/P1/P2 判定标准已在 D4/D5 定义;per MR-1/5/7:BAR_SYNC 契约、wmma.cpp 扩 scope、cluster commit hash(`eb52af4` 集成 + `e513235` 基础原语)已在 proposal Why 段落标注)。
