# CppTLM D1-Full × PTX-EMU cpptlm-d1-full 跨仓库评审

> **日期**: 2026-07-16
> **范围**: CppTLM `feature/d1-full-impl`（worktree `feature-d1-full-impl`） vs PTX-EMU `cpptlm-d1-full` change（main HEAD `73e09d97`）
> **目的**: 在 PTX-EMU 端把 PR/合并决策前，CppTLM/PTX-EMU 维护者可独立审计此报告
>
> ⚠️ **快照过期（SNAPSHOT STALE）**（[PATCH 2026-07-16 in-session]）：本评审**原始快照**为 CppTLM worktree HEAD `abd7dd5`（2 commits / 8 files / +1691 lines）。**当前** HEAD 已推进至 `57f50667ad0e70de7a02f00c978c8838a23b0f70`（5 commits / 15 files / +2519 lines）。新增 3 个 commit 直接解决了 4 个 BLOCKER 中的 2 个：
>
> | Commit | 解决的问题 | 对应章节 |
> |:------:|------------|----------|
> | `e9014de` docs(superpowers): outbound 4 RFCs to PTX-EMU team | RFC 文件 untracked → **已提交** | §1 / §B1 / §11.3 |
> | `505a7c9` feat(openspec): cpptlm-d1-p1-pipeline-scoreboard P1 artifacts | pipeline-scoreboard untracked → **已提交** | §4.2 / §9.2 H8 |
> | `57f5066` feat(openspec): cpptlm-d1-p1-pipeline-scoreboard initial proposal/design/tasks | Day 3.5 cleanup | §4.2 |
>
> **有效的结论**：本评审所有事实层声明 / 文件级行号 / 测试结果 / SHA / HEAD 哈希仍准确；唯一失效的是"untracked"判定。**建议把 RFC / pipeline-scoreboard 提交后的新状态当作"§1 修订记录"读**。
>
> **关键结论**: 2026-07-16 RFC 在评审快照（`abd7dd5`）时是 **untracked** design-time questions（4 项 RFC-001~004 均 🟡 Pending）；评审快照后由 CppTLM 端通过 commit `e9014de` 完成提交。本评审**不能**对 RFC 合规性下结论 — 见 §1
> **状态**: 评审完成；PTX-EMU 端 cpptlm-d1-full tasks.md 自身标记 `Phase 1 / Phase 3.8 / 验收` 等子项未完成，因此"PTX-EMU 端 F12b-LD MemoryBridge 已完整"也是**不能**声明的
> **续审**: 在 worktree 临时态下发现 RFC 文件后，本评审已重新校准 §1、§2、§3.3、§11.3、§12.3、§13.1、§13.2；Verdict C 从 NO-GO 升级为 CONDITIONAL，Verdict A/B 保持原结论；二次验证（[PATCH 2026-07-16 in-session]）增加 snapshot-stale 警告 + HEAD `57f5066` 修订记录
> **基准事实（已实测，分两阶段）**:
> - CppTLM main HEAD `c89d9966f24b4ee6339dee52703c348805793cce`
> - CppTLM feature worktree **评审快照** HEAD `abd7dd5ebd39c6c2b05489eec5d61111a3a1b471`（branch `feature/d1-full-impl`，**已过期**）
> - CppTLM feature worktree **当前** HEAD `57f50667ad0e70de7a02f00c978c8838a23b0f70`（**已快照过期**）
> - PTX-EMU main HEAD `73e09d97610404f44106c47e427f74ca1d452df9`
> - CppTLM worktree committed diff (相对于 c89d996, **原始快照**): 2 commits / 8 files / +1691 lines
> - CppTLM worktree committed diff (相对于 c89d996, **当前**): **5 commits / 15 files / +2519 lines**
> - CppTLM worktree 评审快照后无 untracked；RFC 已在 `e9014de` 提交 + pipeline-scoreboard 已在 `505a7c9`/`57f5066` 提交

---

## 1. 关键结论 — 2026-07-16 RFC 已发现（worktree 暂存、main 缺位）

**事实（FACT, 高置信度）** — 后续二次审查（2026-07-16 in-session 续审）推翻原"未发现"结论：

- 文件存在：`/workspace/project/CppTLM/.worktrees/feature-d1-full-impl/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu.md`（165 行，9,392 字节，mtime 2026-07-16 15:31）
- `git status --short` 显示 `??`（**untracked**）— 文件在工作树中但**未 git add、未 commit**
- `git log --oneline -5` 显示 HEAD `abd7dd5`（即原"committed diff 2 commits"快照之上**未增加**新提交）；最新预期提交 `e9014de docs(superpowers): outbound 4 RFCs to PTX-EMU team (Day 2 / Metis Round 4)` 仍是 Metis 描述，**未在 worktree 提交**
- CppTLM `main` HEAD `c89d9966f24b4ee6339dee52703c348805793cce`、feature/d1-full-impl HEAD `abd7dd5ebd39c6c2b05489eec5d61111a3a1b471`、PTX-EMU main HEAD `73e09d97610404f44106c47e427f74ca1d452df9` 三处**均不包含该文件**（git object 未引用）；主分支与姊妹 change 链路**完全看不见**这份 RFC

**RFC 内容（4 项跨端契约请求，由 CppTLM Metis Round 4 发出）**:

| RFC | 主题 | 推荐选项 | PTX-EMU 回执 | 状态 |
|:---:|------|----------|-------------|------|
| **RFC-001** | 12 端点 `static_assert` 时机 | **A** — 从 P0 推迟到 P1；C1 侧 `cpptlm-f12b-ld-impl` 不包含 12 端点，仅 P1 `cpptlm-d1-pipeline-scoreboard` 含 | ⏳ 待回复 | 🟡 Pending |
| **RFC-002** | `synchronize_stream` ABI 冗余（双 sync loop） | **B** — `cudaStreamSynchronize` delegate 到 `bridge->synchronize_stream(target_stream)`；不 bump `CPPTLMBRIDGE_VERSION` | ⏳ 待回复（48h SLA） | 🟡 Pending |
| **RFC-003** | `kernel_args` lifecycle gap | **A** — `PendingKernel` 加 `std::vector<std::vector<uint8_t>> copied_args` | ⏳ 待回复 | 🟡 Pending |
| **RFC-004** | Cross-repo 链接不对称 | **A** — 在 3 个 `proposal.md` 加 `Cross-Project Counterparts` 表 | ⏳ 待回复 | 🟡 Pending |

**结论（INFERENCE, 高置信度）**:
1. RFC **评审快照（`abd7dd5`）时未 git-tracked** — PTX-EMU 端 + CppTLM main 端 + 评审快照均看不到此文件。[PATCH 2026-07-16 in-session]：评审快照后由 CppTLM 端通过 commit `e9014de docs(superpowers): outbound 4 RFCs to PTX-EMU team` 完成提交；当前 HEAD `57f5066` 中 git-tracked 可读。
2. 文件**不是 RFC 模板**，而是 **4 项跨端契约询问**（"Ask" 模式） — 等同于 "design-time questions"，**不是**已批准的设计规范。
3. **不能用其他文档替代**（综合任务书 2026-07-14、协作同步 2026-07-01）做 RFC 合规性判断；二者的覆盖范围、签名、验收与 RFC 询问无 1:1 对应。
4. 任何"符合 2026-07-16 RFC"的合规性主张在当前证据下**仍不能成立** — 因为 RFC 是"提问"而非"规范"，且 4 项均 🟡 Pending。
5. **RFC-002 / RFC-003 与本评审结论强相关**：
   - RFC-002 选项 B 正是本评审 §9.2 H6 指出的"production code 未调用 `synchronize_stream`"问题的 PTX 端修复路径。
   - RFC-003 选项 A 正是本评审 §9.2 H2（8 字节 deep-copy）相关 lifecycle 修复路径。
   - RFC-001 选项 A 改变了 H9 "P1 Compute 实施" 的入口条件（12 端点 static_assert 推迟）。

**修订后严禁做出的声明**:
- ❌ "本 change 符合 2026-07-16 RFC X.Y.Z 节要求" — RFC 是设计询问，不是已批准规范
- ❌ "本 change 通过 RFC §A 验收" — 同上
- ❌ "RFC 2026-07-16 不存在" — **本节已修正**为 "RFC 存在但 untracked，非规范"（[PATCH 2026-07-16 in-session]：评审快照后 RFC 已在 `e9014de` 提交；建议表述为"RFC 已发现且已 git-tracked，但仍是未批准的设计询问"）
- ❌ 用综合任务书（2026-07-14）或协作同步（2026-07-01）替代 RFC 审查

---

## 2. 执行摘要（Executive Summary）

| 维度 | 结论 | 严重度 |
|------|------|:-----:|
| **CppTLM D1-Full 实现（除 ABI 头文件）** | **不存在** — worktree 仅含 ABI vendor + 6 个 OpenSpec artifacts，无 MemoryBridge、无 D1 KernelLaunchTLM、无 4 Adapter、无 Scoreboard/Pipeline/TC 模块、无 D1 测试、无 CMake 集成 | BLOCKER |
| **PTX-EMU 端 cpptlm-d1-full tasks.md 自身状态** | 多个 Phase 任务仅声明 `[x]` 但 PTX-EMU 端 `cpptlm-d1-full` tasks.md 注明部分 Phase 1 / 3.8 / 验收项待实际发出 | HIGH |
| **PTX-EMU 端 cpptlm-d1-full 是否"100% 完成"** | **不能声明** — `openspec list` 实测 51/61，10 项 unchecked（含 8 项验收门 + Phase 3.8 + Phase 1.2 DEFERRED） | BLOCKER |
| **ABI 兼容性** | PTX 端 `ca716a81...` 与 CppTLM 端 vendored `c19e66a3...` **不一致** — PTX 端额外加入 `PTXEMU_BRIDGE_API`、`CUDA_STREAM_T_DEFINED`、`cpptlm_attach_bridge`、`cpptlm_detach_bridge` 4 项 | HIGH |
| **RFC 合规性** | **仍不能成立**（RFC 在评审快照时存在但 untracked；[PATCH in-session] 后已在 `e9014de` 提交；4 项（RFC-001~004）均 🟡 Pending，PTX-EMU 未回执；RFC-002/003 与本评审 §9 H6/H2 强相关） | BLOCKER |
| **PTX-EMU 端 `cpptlm-d1-full` tasks.md 数字** | `openspec list` 实测 51/61 checked（10 unchecked：8 项验收门 + Phase 1.2 DEFERRED + Phase 3.8 集成测试） | HIGH |
| **RFC 2026-07-16 实际状态** | RFC 文件**已存在**于 CppTLM worktree；**评审快照时 untracked** (`??`)，**[PATCH 2026-07-16 in-session]：已在 commit `e9014de` 提交**；内容是 4 项 design-time questions（RFC-001~004），均 🟡 Pending；非已批准规范 | BLOCKER |
| **PTX-EMU 端实现路径真实情况** | `cudart_sim.cpp:154-159`、`494-506`、`819-853`、`972-1017` 等位置确实存在 bridge 异步代码（实测），但路径**不完整** | HIGH |
| **OpenSpec 跨仓库一致性** | 双端 changes 各自包含，但范围数字 / 路径 / 接口描述**可能存在**内部不一致（见 §7） | MEDIUM |

---

## 3. 范围与基线（Scope & Baselines）

### 3.1 仓库与提交基线（实测）

| 仓库 | 分支 / Worktree | HEAD | 状态 |
|------|----------------|------|------|
| CppTLM | `main` | `c89d9966f24b4ee6339dee52703c348805793cce` | 干净 working tree |
| CppTLM | `feature/d1-full-impl` worktree（**评审快照 HEAD，过期**） | `abd7dd5ebd39c6c2b05489eec5d61111a3a1b471` | 干净工作区 + 1 untracked dir（**评审快照时**；见顶部 snapshot-stale 警告） |
| CppTLM | `feature/d1-full-impl` worktree（**当前 HEAD**） | `57f50667ad0e70de7a02f00c978c8838a23b0f70` | 干净工作区（**已无 untracked**；RFC + pipeline-scoreboard 已在 `e9014de`/`505a7c9`/`57f5066` 提交） |
| PTX-EMU | `main` | `73e09d97610404f44106c47e427f74ca1d452df9` | 干净 working tree |

### 3.2 已审计的已验证文件

| 文件 | 大小 / 行数 | 用途 |
|------|-------------|------|
| `/workspace/project/CppTLM/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md` | 949 行 | 综合任务书（**非 RFC**，仅设计参考） |
| `CppTLM/.worktrees/feature-d1-full-impl/include/cudart/cpptlm_bridge.h` | 161 行 | ABI 头文件 vendored 副本 |
| `CppTLM/.worktrees/feature-d1-full-impl/include/tlm/gpu/kernel_launch_tlm.hh` | 62 行 | **Phase 8.A 旧 stub**（非 D1 实现） |
| `CppTLM/.worktrees/feature-d1-full-impl/src/tlm/gpu/kernel_launch_tlm.cc` | 22 行 | **Phase 8.A 旧 stub**（按 interval 计数，非 D1 路径） |
| `CppTLM/.worktrees/feature-d1-full-impl/src/CMakeLists.txt` | 88 行 | 无 `memory_bridge`/`kernel_launch_tlm`/Adapter 目标 |
| `CppTLM/.worktrees/feature-d1-full-impl/openspec/changes/cpptlm-f12b-ld-impl/` | 6 文件 + .openspec.yaml | P0 F12b-LD OpenSpec artifacts（**全部 proposed 状态**） |
| `CppTLM/.worktrees/feature-d1-full-impl/openspec/changes/cpptlm-d1-p1-pipeline-scoreboard/` | 4 文件 | **评审快照时 untracked；[PATCH in-session]：已在 commit `505a7c9` + `57f5066` 提交** |
| `PTX-EMU/include/cudart/cpptlm_bridge.h` | 184 行 | PTX 端 ABI 真值源（含 PTXEMU_BRIDGE_API + cpptlm_attach_bridge 等） |
| `PTX-EMU/src/cudart/cudart_sim.cpp` | 1255 行 | 异步化 cudaLaunchKernel + sync 原语 |
| `PTX-EMU/src/ptxsim/instructions/memory.cpp` | 196 行 | LD/ST bridge 分支 |
| `PTX-EMU/CMakeLists.txt` | 152 行 | ExternalProject_Add + commit pin |
| `PTX-EMU/openspec/changes/cpptlm-d1-full/tasks.md` | 300 行 | 端到端实施 tasks |
| `PTX-EMU/openspec/changes/cpptlm-phase8b-injection-points/` | 7 文件 | 姊妹 change（D1 接口 + SMContext 注入） |

### 3.3 已审 **未通过** 的内容

- CppTLM `docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu.md` — **评审快照时 untracked**（worktree 暂存）; **[PATCH in-session]：已在 commit `e9014de` 提交** — §1 已修订
- CppTLM `docs/superpowers/specs/PTX-EMU-README.md` — 不在已验证文件列表中，未独立审计

---

## 4. 源可用性 / 实现完整性（Source Availability）

### 4.1 CppTLM worktree committed diff（实测）

```
$ git diff c89d996 HEAD --stat
 include/cudart/AGENTS.md                           |  16 +
 include/cudart/cpptlm_bridge.h                     | 161 +++++++
 .../changes/cpptlm-f12b-ld-impl/.openspec.yaml     |   2 +
 openspec/changes/cpptlm-f12b-ld-impl/design.md     | 431 +++++++++++++++++
 .../changes/cpptlm-f12b-ld-impl/internal-plan.md   | 528 +++++++++++++++++++++
 openspec/changes/cpptlm-f12b-ld-impl/proposal.md   | 114 ++++++
  .../specs/cpptlm-f12b-ld-impl/spec.md              | 155 ++++++
 openspec/changes/cpptlm-f12b-ld-impl/tasks.md      | 284 +++++++++++
 8 files changed, 1691 insertions(+)
```

**FACT, 高置信度**（评审快照 `abd7dd5`，已过期 — 见顶部 snapshot-stale 警告）:
- 2 commits（`627ecb4` feat + `abd7dd5` BLOCK fixes）
- 8 文件 / +1691 行
- **全部为 OpenSpec artifacts + vendored ABI header + AGENTS.md** — **零实现代码**
- **[PATCH 2026-07-16 in-session]**: 当前 HEAD `57f5066` 已扩展至 5 commits / 15 files / +2519 lines（追加 `e9014de` RFC outbound + `505a7c9`/`57f5066` P1 pipeline-scoreboard artifacts）

### 4.2 CppTLM worktree untracked 内容（评审快照 `abd7dd5`，已过期 — 见顶部 snapshot-stale 警告）

```
$ git status
?? openspec/changes/cpptlm-d1-p1-pipeline-scoreboard/
```

**FACT, 高置信度（评审快照时）**: `cpptlm-d1-p1-pipeline-scoreboard` 整个目录**未跟踪**（评审快照 `abd7dd5` 时），因此：
- 未包含在 committed diff 统计内
- 未被 PR/CI 验证

**[PATCH 2026-07-16 in-session]**: 当前 HEAD `57f5066` 时 `git status --short` **为空**（整个目录已在 commit `505a7c9` + `57f5066` 中提交）。未跟踪状态已解除，**未实现内容仍未实现**。
- 与 `cpptlm-f12b-ld-impl` 的关系在 `git status` 中无法追溯

### 4.3 CppTLM 端 D1 实施搜索（实测）

```bash
find ... -name "memory_bridge*"
# → 空（无 memory_bridge.hh / .cc）

find ... -name "*adapter*"  # 在 include/ 与 src/ 下
# → 仅 stream_adapter_base.hh（Phase 7 已有），无 cpptlm_*_adapter

find ... -name "*scoreboard*" "*pipeline_tlm*" "*tensorcore*"
# → 仅 openspec/changes/cpptlm-d1-p1-pipeline-scoreboard/ 计划目录
#   （在 src/ 与 include/ 下无 .hh/.cc 实现）
```

**INFERENCE, 高置信度**: CppTLM worktree 在 committed 状态下**零 D1 实施**：
- ❌ `include/tlm/gpu/memory_bridge.hh` + `src/tlm/gpu/memory_bridge.cc` — **不存在**
- ❌ 4 Adapter (`cpptlm_{warp_scheduler,scoreboard,pipeline,tensor_core}_adapter.hh`) — **不存在**
- ❌ `include/tlm/gpu/{scoreboard,pipeline,tensorcore}_tlm.hh` — **不存在**
- ❌ `async_completion_adapter.hh` — **不存在**
- ❌ `tests/unit/cpptlm/test_memory_bridge.cc` — **不存在**
- ❌ `tests/integration/cpptlm/test_f12b_integration.cc` — **不存在**
- ❌ `tests/python/test_f12b_smoke.py` — **不存在**
- ❌ `CMakeLists.txt` 中 `memory_bridge` 目标 — **不存在**

唯一存在的实现代码是 **Phase 8.A 旧 `KernelLaunchTLM` stub**（62 + 22 行，`cppTLM-` author 注释为 2026-06-24；只按 interval 计数 + `kernels_launched_++`）— **与 D1-Full Compute/Pipeline/Scoreboard 完全无关**。

### 4.4 PTX-EMU 端实施搜索（实测）

| 文件 | 行数 | 内容 | 状态 |
|------|------|------|------|
| `include/cudart/cpptlm_bridge.h` | 184 | ABI 真值源 + `PTXEMU_BRIDGE_API` + `cpptlm_attach_bridge/detach_bridge` | ✅ 提交 |
| `src/cudart/cudart_sim.cpp` | 1255 | `SingletonGuard`、`PendingKernel`、`cudaLaunchKernel` 异步分支、`cudaStreamSynchronize` 真 poll、`cudaDeviceSynchronize` 真 poll、`cudaStreamCreate/Destroy`、`cpptlm_attach/detach_bridge` | ✅ 提交 |
| `src/ptxsim/instructions/memory.cpp` | 196 | LD handler bridge 分支（`g_cpptlm_bridge && space == GLOBAL`），ST handler 类似 | ✅ 提交 |
| `CMakeLists.txt` | 152 | `BUILD_LIB_CPPTLM_CUDART` option + ExternalProject_Add + `CPPTLM_COMMIT_HASH "main"` 默认值 | ✅ 提交 |
| `openspec/changes/cpptlm-d1-full/` | 6 文件 | tasks.md 完整 | ✅ 提交 |

**INFERENCE, 高置信度**: PTX-EMU 端 F12b-LD 异步路径**实际有提交**，但 tasks.md 内部对完成度的标记存在矛盾（见 §6.1）。

---

## 5. RFC 合规性矩阵（RFC Compliance Matrix）

> ⚠️ **范围声明**: 由于 2026-07-16 RFC 是 **design-time questions**（非已批准规范）且 4 项均 🟡 Pending，本矩阵**不能填写** RFC 节号。下表转而对照**实际存在的参考文档**（2026-07-14 综合任务书与 2026-07-01 协作同步），**仅作执行参考**，不构成 RFC 合规性结论。
>
> ⚠️ **RFC 2026-07-16 摘要**（用于 §9 严重度校准）:
>
> | RFC | 主题 | 推荐选项 | 与本评审 §9 关联 |
> |:---:|------|----------|----------------|
> | **001** | 12 端点 `static_assert` 时机 | A: 推迟到 P1 | H9 P1 入口条件 |
> | **002** | `synchronize_stream` 冗余 | B: `cudaStreamSynchronize` delegate | **H6 直接闭合** |
> | **003** | `kernel_args` lifecycle | A: `PendingKernel` 加 `copied_args` | H2 范围缩窄 |
> | **004** | Cross-repo 链接 | A: 3 个 `proposal.md` 加 Counterparts 表 | §11.1 部分闭合 |

### 5.1 综合任务书 §2 (P0 F12b-LD MemoryBridge) 对照

| Task | 来源（综合任务书） | CppTLM 端实施状态 | PTX-EMU 端实施状态 | 一致性 |
|------|--------------------|------------------|--------------------|--------|
| **#1** `CppTLMBridge` 接口 | §2.1 Task #1, lines 132-202 | ✅ ABI header vendored | ✅ header 在 main | 一致 |
| **#2** 异步 `cudaLaunchKernel` | §2.1 Task #2, lines 218-293 | n/a | ✅ `cudart_sim.cpp:490-551` | 一致 |
| **#3** Stream 同步原语 | §2.1 Task #3, lines 302-367 | n/a | ⚠️ `cudaStreamSynchronize` 真 poll 在 `cudart_sim.cpp:972-1017`（B2 fix）；`cudaDeviceSynchronize` 在 `819-855`；`cudaStreamCreate` 在 `911-927`；`cudaStreamDestroy` 在 `929-953` | 一致 |
| **#4** `hardware_memory_manager` GLOBAL 桥接 | §2.1 Task #4, lines 378-422 | n/a | ⚠️ `memory.cpp:35-56`（LD）；ST handler 在文件后续行（task 写 memory.cpp） | 一致（修正：实际修改点在 `memory.cpp` 而非综合任务书原 §2.1 Task #4 假设的 `hardware_memory_manager.cpp`，符合 §9 v1.1 修订说明） |
| **#5** `libcpptlm_cudart.so` 集成构建 | §2.1 Task #5, lines 426-453 | n/a | ✅ `CMakeLists.txt:121-149` ExternalProject_Add | 一致（但 `CPPTLM_COMMIT_HASH` 默认 `"main"`） |
| **#C1** MemoryBridge 实现 | §2.2 Task #C1, lines 458-523 | ❌ **未实施**（`memory_bridge.hh` 不存在） | n/a | **不一致 — 双端无法集成** |
| **#C2** KernelLaunchTLM EventQueue 集成 | §2.2 Task #C2, lines 525-582 | ❌ **未实施**（`kernel_launch_tlm.hh/cc` 仅为 Phase 8.A 旧 stub） | n/a | **不一致 — 双端无法集成** |

**INFERENCE, 高置信度**:
- PTX-EMU 端 #1-#5 已实质实施（实测代码可见）
- CppTLM 端 #C1/#C2 **完全缺失** — 与 §1 RFC 缺失合并起来，等于"F12b-LD MemoryBridge 集成"在 CppTLM 侧**仅是计划/文档，未实现**

### 5.2 综合任务书 §3 (P1 D1-Full Compute) 对照

| Task | 来源 | CppTLM 端 | PTX-EMU 端 |
|------|------|-----------|------------|
| **#6** 3 纯虚接口头文件 | §3.1 Task #6 | n/a | ❌ 未在 cpptlm-d1-full 实测范围；姊妹 change `cpptlm-phase8b-injection-points` 包含（未独立审计） |
| **#7** SMContext 头文件修改 | §3.1 Task #7 | n/a | ❌ 同上 |
| **#8** `set_blocked_cycles_for_active` | §3.1 Task #8 | n/a | ❌ 同上 |
| **#9** `exe_once()` 三段式注入 | §3.1 Task #9 | n/a | ❌ 同上 |
| **#C3** 4 Adapter | §3.2 Task #C3 | ❌ **未实施** | n/a |
| **#C4** 3 核心模块 + `tlm::I*Internal` | §3.2 Task #C4 | ❌ **未实施** | n/a |
| **#C5** `IAsyncCompletion` 占位 | §3.2 Task #C5 | ❌ **未实施** | n/a |

**INFERENCE, 高置信度**: P1 D1-Full Compute 在双端都**未实质实施**。姊妹 change `cpptlm-phase8b-injection-points` 的具体状态需独立审计（不在本评审范围）。

---

## 6. Worktree 实施审计（Implementation Audit）

### 6.1 PTX-EMU `openspec/changes/cpptlm-d1-full/tasks.md` 内部一致性

FACT, 高置信度（来自 `cpptlm-d1-full/tasks.md`）:

| 字段 | 标记 | 状态 |
|------|------|------|
| Phase 0.5 基线 worktree | `[x]` 0.5 | ✅ 完成（含 `9be56f8f` commit + `unit_barrier_module` PASS 验证） |
| Phase 1.1 创建 cpptlm_bridge.h | `[x]` 1.1 | ✅ 声称完成（但标注"⏳ 待发出 HSK-1"） |
| Phase 1.2 stub 实现 | `[ ]` 1.2 ⚠️ **DEFERRED** | ⏸️ 推迟（mock bridge 覆盖） |
| Phase 1.4 编译验证 + attach/detach | `[x]` 1.4 | ✅ 声称完成（`T cpptlm_attach_bridge` 符号可见） |
| Phase 1.5 HSK-1 commit + 输出 | `[x]` 1.5 | ✅ claim commit `603bd8bc`，但标注 "⏳ 待发出" |
| Phase 2 SingletonGuard | `[x]` 2.1-2.3 | ✅ 声称完成 |
| Phase 3 cudaLaunchKernel 异步化 | `[x]` 3.1-3.7 | ✅ 声称完成 |
| Phase 3.8 集成测试 `integration_async_launchkernel` | `[ ]` 3.8 | ❌ **未勾选** |
| Phase 4 cudaStreamSynchronize | `[x]` 4.1-4.5 | ✅ 声称完成 |
| Phase 5 GLOBAL LD/ST 桥接 | `[x]` 5.1-5.5 | ✅ 声称完成 |
| Phase 6 CMake libcpptlm_cudart | `[x]` 6.1-6.7 | ✅ 声称完成（HSK-3 候选 ExternalProject_Add） |
| Phase 7 测试编写 | `[x]` 7.1-7.6 | ✅ 声称完成（含 7 unit + 3 integration + 1 singleton_guard） |
| Phase 8 Handshake 回传 + 文档 | `[x]` 8.1-8.8 | ✅ 声称完成 |
| **验收** | `[ ]` 全 7 项 | ❌ **全未勾选** |

**INFERENCE, 高置信度**:
- `tasks.md` 自身承认：Phase 3.8 集成测试 + 7 项验收门**实际未完成**
- 即"PTX-EMU 端 cpptlm-d1-full 已完整"在 tasks.md 自证层面是**未达成的**
- 因此 §2 表中"PTX-EMU 端 F12b-LD MemoryBridge 已完整"是**不能声明**的

### 6.2 CppTLM `cpptlm-f12b-ld-impl/tasks.md` 状态

FACT, 高置信度（来自 `cpptlm-f12b-ld-impl/tasks.md`）:

| Phase | 标记 | 状态 |
|-------|------|------|
| Phase 0 (vendor + baseline 验证) | `[ ]` 0.1-0.4 | ❌ 全部未勾选 |
| Phase 1 P0 F12b-LD MemoryBridge | `[ ]` 1.1.1-1.3.3 | ❌ 全部未勾选 |
| Phase 2 P1 D1-Full Compute 注入 | `[ ]` 2.1.1-2.2.6 | ❌ 全部未勾选 |
| Phase 3 P2 Phase 9+ Async Seam | `[ ]` 3.1-3.2 | ❌ 全部未勾选 |
| Phase 4 P3 集成验证 | `[ ]` 4.1-4.6 | ❌ 全部未勾选 |
| 综合验收 Gates G0-G7 + G-F0 | `[ ]` 全 8 项 | ❌ 全部未勾选 |

**INFERENCE, 高置信度**: `cpptlm-f12b-ld-impl/tasks.md` 全部 `[ ]` — change **未实施**，仅文档/计划存在。

### 6.3 关键代码路径 — PTX-EMU 端实测

> 以下路径在 `cudart_sim.cpp` 中实测存在，但每条均有具体限制（见后）

**FACT, 高置信度** (源自 `cudart_sim.cpp:140-159`):
```cpp
140:    size_t shared_mem;
141:    std::vector<std::vector<uint8_t>> args_copy;  // deep-copy 的参数
142:    bool completed = false;
143: };
144:
145: static std::atomic<uint64_t> next_kernel_id{1};
146: static std::unordered_map<uint64_t, PendingKernel> g_pending_kernels;
147: static std::unordered_set<uint64_t> g_active_streams{0};  // 默认包含 stream 0
148: static std::mutex g_pending_kernels_mutex;
...
154: static size_t count_kernel_args(void** args) {
155:     if (!args) return 0;
156:     size_t count = 0;
157:     while (args[count] != nullptr) ++count;   // ← nullptr-sentinel 终止
158:     return count;
159: }
```

**⚠️ 限制 #1（INFERENCE, 高置信度）**: `count_kernel_args()` 用 nullptr 哨兵确定参数数量：
- 若调用方传入的 `args[]` 数组**本身不带 nullptr 哨兵**（很多 CUDA runtime 调用是这种情况），将**读取越界**直到撞到零字节或触发 UB
- 综合任务书 §2.1 Task #2 描述：`args_count` 由 PTX-EMU 端根据 PTX 元数据提供（CppTLM 不需要解析 PTX）— 但 `cudart_sim.cpp:154-159` 实现是 nullptr 哨兵方案，**与综合任务书描述不一致**

**FACT, 高置信度** (源自 `cudart_sim.cpp:494-506`):
```cpp
494:        // deep-copy kernel args
495:        std::vector<std::vector<uint8_t>> args_copy;
496:        if (args) {
497:            size_t arg_count = count_kernel_args(args);
498:            args_copy.reserve(arg_count);
499:            for (size_t i = 0; i < arg_count; ++i) {
500:                if (args[i]) {
501:                    // 假设每个参数最大 8 字节（指针或基本类型）
502:                    std::vector<uint8_t> arg_data(8);   // ← 固定 8 字节
503:                    std::memcpy(arg_data.data(), args[i], 8);
504:                    args_copy.push_back(std::move(arg_data));
505:                }
506:            }
507:        }
```

**⚠️ 限制 #2（INFERENCE, 高置信度）**: 固定 8 字节 deep-copy：
- 对 `struct` 或 `__int128` 等大于 8 字节参数**截断**（silently lose upper bytes）
- 对 `float`/`int`（≤4 字节）等小参数：`std::memcpy(arg_data.data(), args[i], 8)` 会**读取 4 字节之后**的 4 字节（未分配区/相邻栈数据），属**越界读 UB**（即使大多数平台不会 crash）
- 综合任务书 §2.1 Task #2 描述提到"PTX-EMU 在 submit_kernel 返回前保证 args 内存有效" — 未规定 PTX-EMU 必须 deep-copy args；`cudart_sim.cpp` 实施 deep-copy 是 P0 决策（task 中提到 `#C2` 的 CppTLM 端"必须 deep-copy"），但**固定 8 字节**这一具体策略**未在任务书或 ADR 中审查到**

**FACT, 高置信度** (源自 `cudart_sim.cpp:819-855`):
```cpp
819:    if (g_cpptlm_bridge) {
820:        while (true) {
821:            std::vector<uint64_t> completed_ids;
822:            {
823:                std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
824:                if (g_pending_kernels.empty()) {
825:                    break;  // all kernels drained - sync complete
826:                }
827:                for (const auto& [id, pk] : g_pending_kernels) {
828:                    if (!pk.completed) {
829:                        uint64_t remaining = g_cpptlm_bridge->poll_kernel(id);
830:                        if (remaining == 0 || remaining == UINT64_MAX) {
831:                            completed_ids.push_back(id);
832:                        }
833:                    }
834:                }
835:            }
836:            if (!completed_ids.empty()) {
837:                std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
838:                for (uint64_t id : completed_ids) {
839:                    g_pending_kernels.erase(id);
840:                }
841:            }
842:            ...
847:            {
848:                std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
849:                if (g_pending_kernels.empty()) {
850:                    break;
851:                }
852:            }
853:            std::this_thread::yield();
854:        }
855:        return cudaSuccess;
856:    }
```

**FACT, 高置信度** (源自 `cudart_sim.cpp:972-1017`):
```cpp
972:    if (g_cpptlm_bridge) {
973:        while (true) {
974:            std::vector<uint64_t> completed_ids;
975:            {
976:                std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
977:                bool stream_has_pending = false;
978:                for (const auto& [id, pk] : g_pending_kernels) {
979:                    if (pk.stream_id == stream_id) {
980:                        stream_has_pending = true;
981:                        if (!pk.completed) {
982:                            uint64_t remaining = g_cpptlm_bridge->poll_kernel(id);  // 简化引用
```

**⚠️ 限制 #3（INFERENCE, 高置信度）**: 外部 `poll_kernel` 调用在 mutex 持锁状态下进行：
- `cudart_sim.cpp:829` 与 `cudart_sim.cpp:982` 均在 `lock_guard(g_pending_kernels_mutex)` 作用域内调用 `g_cpptlm_bridge->poll_kernel(id)`
- 若 `poll_kernel` 实现（CppTLM 端 `MemoryBridge::poll_kernel`）反向调用任何**共享** `g_pending_kernels_mutex` 或其下游锁，则会触发**锁倒置 (lock inversion)** 或**重入死锁**
- 当前 CppTLM 端 `MemoryBridge::poll_kernel` 实现**不存在**，因此死锁**未被证实**
- 综合任务书 §2.1 Task #C1 的 `poll_kernel` 设计是查 `kernel_launch_->poll_completion(kernel_id)`，不反向持有 PTX-EMU 端 mutex — 但**实际实施前无法验证**

**⚠️ 限制 #4（FACT, 高置信度）**: `bridge->synchronize_stream()` 在生产同步路径中**未被调用**：
- `cpptlm_bridge.h:116-120` 定义 `synchronize_stream` 虚方法
- `cudart_sim.cpp:819-855`（cudaDeviceSynchronize）**只调用 `poll_kernel`**，**未调用 `synchronize_stream`**
- `cudart_sim.cpp:972-1017`（cudaStreamSynchronize）同样**只调用 `poll_kernel`**
- 桥接接口语义上 `synchronize_stream` 是 stream 级别同步原语（综合任务书 §2.1 Task #3 设计意图），但**PTX-EMU 端生产代码未使用** — 接口与生产实现存在语义脱节

### 6.4 `cpptlm_bridge.h` ABI 端 (PTX 端实测)

**FACT, 高置信度** (源自 `cpptlm_bridge.h:116-120`, PTX 端):
```cpp
116:    /// 同步等待 stream 上所有 pending kernels 完成
117:    ///
118:    /// @param stream_id stream 句柄（0 = 默认 stream）
119:    /// @return 0=成功, 非0=cudaError_t 错误码
120:    virtual int synchronize_stream(uint64_t stream_id) = 0;
```

**FACT, 高置信度** (源自 `cpptlm_bridge.h:162-169`, PTX 端):
```cpp
162: extern "C" PTXEMU_BRIDGE_API void cpptlm_attach_bridge(CppTLMBridge* bridge);
...
168: /// 实现位置：src/cudart/cudart_sim.cpp
169: extern "C" PTXEMU_BRIDGE_API void cpptlm_detach_bridge();
```

**FACT, 高置信度** (源自 `cpptlm_bridge.h:39-45`, PTX 端):
```cpp
39: #if defined(__CUDACC_RUNTIME_H__)
40: #include <cuda_runtime.h>
41: #elif !defined(CUDA_STREAM_T_DEFINED)
42: // 与 cudart_intrinsics.h 保持一致：cudaStream_t = void*
43: typedef void* cudaStream_t;
44: #define CUDA_STREAM_T_DEFINED
45: #endif
```

### 6.5 PTX-EMU `memory.cpp` LD bridge 实测

**FACT, 高置信度** (源自 `memory.cpp:35-56`):
```cpp
35:  if (g_cpptlm_bridge && space == MemorySpace::GLOBAL) {
36:    uint64_t device_addr = reinterpret_cast<uint64_t>(host_ptr);
37:    uint64_t latency = g_cpptlm_bridge->global_access(device_addr, 0, /*LD=*/0);
38:
39:    if (latency != UINT64_MAX) {
40:      // timing-only 预计算：数据仍在 SimpleMemory 完成
41:      WarpContext *warp_ctx = context->warp_context_;
42:      if (warp_ctx != nullptr && latency > 0) {
43:        auto &ws = warp_ctx->get_warp_state();
44:        for (auto &thread : ws.threads) {
45:          if (thread.is_active && !thread.is_exited) {
46:            thread.is_blocked = true;
47:            thread.blocked_cycles_remaining = static_cast<uint64_t>(latency);
48:          }
49:        }
50:      }
51:      HardwareMemoryManager::instance().access(host_ptr, dst, data_size,
52:                                               false, space);
53:      return;
54:    }
55:    // UINT64_MAX: fallback 到原有路径
56:  }
```

**INFERENCE, 中等置信度**: LD 路径在 `bridge != nullptr && space == GLOBAL && latency != UINT64_MAX` 时走 CppTLM NoC timing；其余 fallback。ST handler 类似（tasks.md §5.2 描述，未独立实测）。

---

## 7. ABI SHA 对比与差异（ABI Diff）

### 7.1 头文件 SHA-256（实测）

| 仓库 | 文件路径 | SHA-256 |
|------|----------|---------|
| PTX-EMU main | `include/cudart/cpptlm_bridge.h` | `ca716a8179841da6de76e0c54406c76d21e42ca3cb8e08a8cd48907f865fe5e7` |
| CppTLM worktree | `include/cudart/cpptlm_bridge.h` | `c19e66a32de398e6bba2042f3f19923ff89dbc02f10bbf310c073ad3a8ff3dbe` |

**INFERENCE, 高置信度**: 两者**字节级不一致**。

### 7.2 内容差异（实测对比）

| 项 | PTX 端（main） | CppTLM 端（vendored） | 影响 |
|----|---------------|----------------------|------|
| 总行数 | 184 行 | 161 行 | PTX 端多 23 行 |
| `PTXEMU_BRIDGE_API` 宏 (line 4-10) | ✅ 存在（含 visibility default + Windows dllexport） | ❌ 缺失 | CppTLM 端 vendor 时**丢弃** ABI 导出宏 — 链接时符号可能不导出 |
| `cudaStream_t` 定义 (line 39-45) | ✅ 完整（`#if __CUDACC_RUNTIME_H__` → `cuda_runtime.h`，否则 `typedef void*` + `#define CUDA_STREAM_T_DEFINED`） | ⚠️ 简化版（同样路径，但 `__CUDACC_RUNTIME_H__` 仍触发 cuda_runtime.h；typedef 不带 `CUDA_STREAM_T_DEFINED` guard） | CppTLM 端在 `cuda_runtime.h` 已 include 时会**冲突 typedef** |
| `synchronize_stream` | ✅ 虚方法 (line 120) | ✅ 虚方法 (line 112) | 一致 |
| `cpptlm_attach_bridge` extern "C" | ✅ 存在 (line 162) | ❌ 缺失 | CppTLM 端 vendor 时**丢弃** attach 入口符号声明 — CppTLM 编译时无法声明链接入口 |
| `cpptlm_detach_bridge` extern "C" | ✅ 存在 (line 169) | ❌ 缺失 | 同上，detach 入口缺失 |
| `g_cpptlm_bridge` extern 声明 | ✅ 存在 (line 154) | ✅ 存在 (line 146) | 一致 |
| `CPPTLMBRIDGE_VERSION = 1` | ✅ (line 55) | ✅ (line 47) | 一致 |
| `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` | ✅ (line 181-182) | ✅ (line 158-159) | 一致 |

### 7.3 影响评估

**INFERENCE, 高置信度**:
- CppTLM 端 vendored header **缺少 PTX 端独有的 4 项**：`PTXEMU_BRIDGE_API`、`CUDA_STREAM_T_DEFINED` guard、`cpptlm_attach_bridge`、`cpptlm_detach_bridge`
- 这些差异**理论上**对应不同的 build 场景（PTX 端有 cuda_runtime.h + ABI 导出；CppTLM 端走 fallback + 不需要导出符号）
- **但不声明无法编译**：未实际编译 CppTLM 端代码（`MemoryBridge`/`KernelLaunchTLM` 不存在） — 因此**不能**说"无法编译"
- 综合任务书 §2.2 Task #C1 的 `MemoryBridge` 设计**依赖** `cpptlm_bridge.h` 接口签名（已匹配 vendor 部分），但**未说明** attach/detach 入口在哪里实现（CppTLM 端）
- 这是 **HIGH 严重度**的 ABI 漂移前置条件 — 必须重新 vendor 才能对齐

### 7.4 `cpptlm-f12b-ld-impl/tasks.md` Phase 0 自验证与现实的矛盾

**INFERENCE, 高置信度**（来自 tasks.md Phase 0.1 验证要求）:
> 验证: `sha256sum` 与 PTX-EMU commit 8dc000ec 字节级一致（`c19e66a32de398e6bba2042f3f19923ff89dbc02f10bbf310c073ad3a8ff3dbe`）

**矛盾点**:
- CppTLM 端 tasks.md 自己**指定**了"vendor 验证 SHA = `c19e66a3...`"，**承认这是 PTX-EMU commit 8dc000ec 当时的字节级哈希**
- 但当前 PTX-EMU main HEAD `73e09d97` 的 `cpptlm_bridge.h` SHA 是 `ca716a81...`（**已漂移**）
- 这意味着 PTX-EMU main 自 `8dc000ec` 后**已修改** ABI header
- CppTLM 端 tasks.md 0.1 当前状态 `[ ]` 表明 vendor 动作未执行；执行后会**重新 vendor 到 `ca716a81...` 还是 `c19e66a3...`**？需 CppTLM 团队明确

---

## 8. PTX 端实施限制 — 详细清单（实测）

> 以下每条均标注行号 + 文件，便于追溯。

### 8.1 cudart_sim.cpp 限制

| 行号 | 内容 | 限制 | 严重度 |
|------|------|------|:-----:|
| `140-159` | `PendingKernel` + `next_kernel_id` + `g_pending_kernels` + `count_kernel_args` | `count_kernel_args()` 用 nullptr 哨兵遍历 args 数组；若调用方未提供哨兵将越界 | HIGH |
| `494-506` | `cudaLaunchKernel` 异步路径 deep-copy 8 字节 | 固定 8 字节；struct/`__int128` 截断；< 8 字节类型多读 | HIGH |
| `819-855` | `cudaDeviceSynchronize` 真 poll 循环 | 在 `lock_guard(g_pending_kernels_mutex)` 内调用 `bridge->poll_kernel()`；潜在锁倒置/重入 | HIGH（条件性，未证实） |
| `911-927` | `cudaStreamCreate` | 用 `next_kernel_id.fetch_add(1)` 生成 64-bit stream_id；与 kernel_id 共享空间可能冲突（极小概率） | LOW |
| `929-953` | `cudaStreamDestroy` | `delete reinterpret_cast<int*>(stream)` 已修（B3）；改为 erase stream_id（实际行为正确，但 BUG-WARN-STD-STREAM-DESTROY-NULLPTR 检查） | LOW |
| `972-1017` | `cudaStreamSynchronize` 真 poll | 同样在 mutex 内调用 `bridge->poll_kernel()` | HIGH（条件性） |
| `972-1017` 同 | 真 poll 循环使用 `std::this_thread::yield()` | busy-spin 风格；无 sleep；无超时 | MEDIUM |

### 8.2 cpptlm_bridge.h (PTX 端) 接口 vs 生产使用

**FACT, 高置信度**:

| 接口方法 | cpptlm_bridge.h 定义 | cudart_sim.cpp 实际调用 |
|---------|---------------------|----------------------|
| `version()` | line 84 | ❌ 未在 cudart_sim.cpp 实测调用 |
| `submit_kernel()` | line 100-106 | ✅ `cudart_sim.cpp:518` |
| `poll_kernel()` | line 114 | ✅ `cudart_sim.cpp:829, 982` |
| `synchronize_stream()` | line 120 | ❌ **未在 cudart_sim.cpp 调用** |
| `global_access()` | line 139 | ✅ `memory.cpp:37` |

**INFERENCE, 高置信度**:
- `version()` 在 PTX-EMU 端**未被运行时检查** — 仅 `static_assert` 在编译期锁定 `CPPTLMBRIDGE_VERSION`（实测存在）
- `synchronize_stream()` **未被生产代码调用** — 接口与生产语义脱节（综合任务书 §2.1 Task #3 设计意图是 `cudaStreamSynchronize` 转发到此方法，但 PTX-EMU 实际实现走 `poll_kernel` 轮询）

### 8.3 CMakeLists.txt 限制

**FACT, 高置信度** (源自 `CMakeLists.txt:121-149`):
```cmake
121: # ==============================
122: # 🔗 CppTLM Bridge 集成 (HSK-3 + D-PTX-6)
123: # ==============================
124: # 默认 OFF：保证现有测试零退化
125: # ON 路径：ExternalProject_Add 拉取 CppTLM 仓库 + 构建 libcpptlm_cudart.so
126: option(BUILD_LIB_CPPTLM_CUDART "Build libcpptlm_cudart.so bridge (requires CppTLM repo)" OFF)
127:
128: if(BUILD_LIB_CPPTLM_CUDART)
129:     include(ExternalProject)
130:
131:     # CppTLM commit hash (HSK-3: 待 CppTLM 团队确认后替换)
132:     set(CPPTLM_COMMIT_HASH "main" CACHE STRING "CppTLM git tag/commit to pin")
```

**INFERENCE, 高置信度**:
- 默认 `BUILD_LIB_CPPTLM_CUDART=OFF` 保证现有测试零退化 — 已实测（PTX-EMU build 全部目标 PASS）
- `CPPTLM_COMMIT_HASH` 默认 `"main"` — **不锁定任何 commit**，存在 ABI 漂移风险
- HSK-3 选项 1（ExternalProject_Add）已实现，但**未锁定 commit**；进入生产前**必须**改为固定 SHA（LOW 严重度 — 当前 opt-in 阶段可接受，但生产前必须固定）

---

## 9. 风险排序发现（Risk-Ranked Findings）

### 9.1 BLOCKER（必须先解决）

| # | 发现 | 证据 | 触发条件 |
|---|------|------|----------|
| **B1** | 2026-07-16 RFC 文件**存在但评审快照时 untracked** | §1 worktree `??` 状态（快照时）；[PATCH 2026-07-16 in-session]：已在 `e9014de` 提交 → **B1 缓解为 MEDIUM**（未变更严重度但阻塞解除）；RFC 仍是 design-time questions 非规范 | 任何 RFC 合规性声明前必须 (a) PTX-EMU 端回执 4 项（RFC-001~004）—— 2026-07-16 in-session commit `e9014de` 是收到端的 CppTLM 通知，下一步是 PTX-EMU 端回执 |
| **B2** | CppTLM 端 D1 实施**完全不存在**（除 ABI header） | §4.3 find 4 项核心模块 + 4 Adapter + MemoryBridge + 5 测试文件全部缺失 | 不能宣称"D1-Full MemoryBridge 已实施" |
| **B3** | PTX-EMU `cpptlm-d1-full/tasks.md` 验收门 8 项**全未勾选**（tasks.md:284-291） | §6.1 表 | 任何"100% 完成"声明前必须勾选验收；与 B2 共同构成"实现存在但 RFC 缺失 + 验收未闭"的双重 BLOCKER |
| **B4** | PTX-EMU `cpptlm-d1-full/tasks.md` Phase 3.8 集成测试**未勾选** | §6.1 行 `3.8` + `tasks.md:103` | 不能宣称"异步 launch kernel 集成已验证" |

### 9.2 HIGH（应立即修复）

| # | 发现 | 证据 | 修复方向 |
|---|------|------|----------|
| **H1** | ABI SHA 漂移：PTX `ca716a81...` vs CppTLM vendored `c19e66a3...` | §7.1 | CppTLM 重新 vendor PTX main HEAD `73e09d97` 后 SHA |
| **H2** | CppTLM vendored header 缺失 4 项（`PTXEMU_BRIDGE_API`/`CUDA_STREAM_T_DEFINED`/`cpptlm_attach_bridge`/`cpptlm_detach_bridge`） | §7.2 | 重新 vendor；或文档化"为何需要这 4 项是 PTX-EMU 独有" |
| **H3** | `cudart_sim.cpp:154-159` `count_kernel_args` 用 nullptr 哨兵 | §6.3 限制 #1 | 改用 PTX 元数据提供 args_count（综合任务书原设计意图） |
| **H4** | `cudart_sim.cpp:494-506` deep-copy 固定 8 字节 | §6.3 限制 #2 | 按 PTX 参数 type 表取真实 size（可能需 metadata 元数据） |
| **H5** | `cudart_sim.cpp:819-855`/`972-1017` 在 mutex 内调用 `poll_kernel` | §6.3 限制 #3 | 在锁外调用 `poll_kernel`，仅持锁做 erase/insert |
| **H6** | `synchronize_stream` 接口定义但**生产代码未调用** | §8.2 | 删除接口（保持接口最小化）或在生产路径使用（stream 级别同步） |
| **H7** | PTX-EMU `cpptlm-d1-full` 验收门 8 项 + Phase 3.8 + Phase 1.2 DEFERRED 总 10 项未完成（与 `openspec list` 51/61 一致） | §6.1 | 完成验收后再 claim 完成 |
| **H8** | `cpptlm-d1-p1-pipeline-scoreboard/` 目录**评审快照时 untracked** | §4.2 | **[PATCH in-session]：已在 `505a7c9` + `57f5066` 提交** — **H8 缓解为 RESOLVED**；建议补充：仍需在 `cpptlm-d1-full-compute` 中正式建立 change 后再追溯还原 |
| **H9** | P1 D1-Full Compute（任务书 §3）**完全未实施**（双端） | §5.2 | 启动 P1 实施（需独立 change） |

### 9.3 MEDIUM（应规划修复）

| # | 发现 | 证据 |
|---|------|------|
| **M1** | OpenSpec 跨仓库一致性可能存在内部冲突（4 个 artifacts 范围数字 / 路径策略） | §10.2 |
| **M2** | 缺少 P1 实施 spec/spec.md（仅 tasks.md 列验收门） | cpptlm-phase8b-injection-points/specs/（未独立审计） |
| **M3** | 测试覆盖限制：PTX-EMU 端仅 6 个 bridge/sync 测试通过；CppTLM 端 0 个 D1 测试 | §10.1 |
| **M4** | `cudaDeviceSynchronize` 与 `cudaStreamSynchronize` 使用 `yield()` busy-spin | §6.3 行 853/1016 — 无超时，无 sleep，可能 100% CPU |

### 9.4 LOW（生产前必须修复）

| # | 发现 | 证据 |
|---|------|------|
| **L1** | `CMakeLists.txt:132` `CPPTLM_COMMIT_HASH` 默认 `"main"` — ABI 漂移风险 | §8.3 |
| **L2** | `cudaStreamCreate` 复用 `next_kernel_id` 生成 stream_id，理论极小冲突概率 | §6.3 行 921 |

---

## 10. 测试证据（Test Evidence）

### 10.1 实测结果

#### CppTLM 端（实测于 `feature-d1-full-impl` worktree）

| 检查 | 命令 | 结果 |
|------|------|------|
| **Build** | `cmake --build build --target cpptlm_core` | exit 0 — `[100%] Built target cpptlm_core` |
| **ctest** | `cd build && ctest` | **No tests were found!!!** |
| **直接 binary** | `./bin/cpptlm_tests` | `All tests passed (15547 assertions in 764 test cases)` |
| **docs_sync** | `bash scripts/test/docs_sync_check.sh --strict` | 1 missing: `docs/ONBOARDING.md → .understand-anything/knowledge-graph.json` |

**FACT, 高置信度**:
- 直接运行 `./build/bin/cpptlm_tests` 输出 `All tests passed (15547 assertions in 764 test cases)`，无任何失败。`764 cases` 与 `15547 assertions` 与 `AGENTS.md:187` 的 `764/764 pass (15547 assertions)` 一致。
- `ctest` 报 `No tests were found!!!` 是因为 CppTLM `cpptlm_tests` 使用 Catch2 自定义 main 且未通过 `add_test` 注册到 CTest；这**不是 build 失败**，仅说明 CTest 不知有哪些测试可调度。
- `docs_sync_check.sh --strict` 的 1 个 missing 路径（`.understand-anything/knowledge-graph.json`）是**预存在**的 doc drift，与本 review 无因果关系。

#### PTX-EMU 端（实测于 main HEAD `73e09d97`）

| 检查 | 命令 | 结果 |
|------|------|------|
| **Build** | `cmake --build build` | exit 0 — `[100%] Built target cute_rmsnorm_debug` |
| **目标 bridge/sync 测试**（6 个） | `ctest -R "cpptlm\|bridge\|attach\|singleton\|stream_sync"` | **6/6 PASS**，总时间 **0.32 sec** |

6 个目标测试（实测 `100% tests passed, 0 tests failed out of 6`）:
1. `Test #105: unit_stream_sync_loop` — PASS 0.08 sec
2. `Test #115: unit_cpptlm_bridge` — PASS 0.02 sec
3. `Test #116: unit_cpptlm_attach_bridge` — PASS 0.04 sec
4. `Test #183: integration_cpptlm_singleton_guard` — PASS 0.02 sec
5. `Test #184: integration_cpptlm_async_launchkernel` — PASS 0.03 sec
6. `Test #185: integration_cpptlm_ld_st_bridge` — PASS 0.02 sec

**FACT, 高置信度**:
- 6 个目标测试全部 PASS，0 fail；实际耗时如上表，总耗时 0.32 sec。
- `cpptlm_tests` 内具体 assertion 计数未在 ctest 输出中显式列出，**不能精确断言**每个测试的 assertion 数量；上述 26 个 assertion 累计数属**推测**而非**实测**，需在最终发布版前删除或注明"待 PTX-EMU 端单测代码中重核"。
- 这些测试**只验证 mock bridge 路径** — 无真实 CppTLM MemoryBridge 集成测试（因 CppTLM 端 MemoryBridge 不存在）。

### 10.2 缺失的测试覆盖（推断）

**INFERENCE, 高置信度** — 以下场景在两端都**未被验证**:
- ❌ 真实 `libcpptlm_cudart.so` 加载 + `cpptlm_attach_bridge` + 真实 MemoryBridge 实现 E2E
- ❌ `cudaStreamSynchronize` 在多 stream 并发场景下的正确性（虽然 `cudaStreamCreate` 已实施，但无并发测试）
- ❌ `cudaDeviceSynchronize` 在多个 bridge-submitted kernel 跨 stream 时的同步正确性
- ❌ G-F0 `vector_add` 端到端测试（CppTLM 端 `tests/python/test_f12b_smoke.py` 未实施）
- ❌ G-F1 `g_cpptlm_bridge == nullptr` 字节级回退的回归对比
- ❌ G-F3 `global_access()` 延迟与 CppTLM NoC 路由延迟一致（≤ 5%）
- ❌ G-D1~G-D8 D1-Full Compute 验证（实施未启动）

---

## 11. OpenSpec 生命周期矩阵（Lifecycle Matrix）

### 11.1 双端 OpenSpec 状态对照

| Change | 仓库 | 当前状态 | 实施状态 | 归档状态 |
|--------|------|----------|----------|----------|
| `cpptlm-f12b-ld-impl` | CppTLM | **Proposed** | ❌ 0% (tasks.md 全 `[ ]`) | 未归档 |
| `cpptlm-d1-p1-pipeline-scoreboard` | CppTLM | **Untracked**（仅 spec/tasks/proposal/design） | ❌ 0% | 未提交 |
| `cpptlm-d1-full` | PTX-EMU | **Active**（`openspec list` 51/61；tasks.md 验收 8 项 + Phase 1.2 + Phase 3.8 共 10 项 `[ ]`） | ⚠️ 部分（tasks.md 自证） | 未归档 |
| `cpptlm-phase8b-injection-points` | PTX-EMU | 独立 change（未独立审计） | 未独立审计 | 未归档 |

### 11.2 双端 OpenSpec 内部一致性

**UNKNOWN / 需独立审计**: 以下事项**超出本评审范围**（未独立审计）：
- PTX-EMU `cpptlm-d1-full/{design.md,proposal.md,spec.md}` 与 tasks.md 的范围数字一致性
- CppTLM `cpptlm-f12b-ld-impl/{design.md,proposal.md,spec.md}` 与 tasks.md 的范围数字一致性
- 双端 design/spec 之间对 F12b-LD MemoryBridge 的接口签名一致性

**INFERENCE, 中等置信度**: 由于 CppTLM 端**未实施**，tasks.md 中"提案范围"与"实际范围"**默认一致**（均为零）。PTX-EMU 端由于 24 commits 的 B1-B5 系列 fix，可能存在内部一致性修补历史（commit `77302f0b fix(cpptlm-d1-full): finalize doc consistency` 暗示需独立审计）。

### 11.3 ADR 与综合任务书关系

**FACT, 高置信度**:
- `docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md` 标注 "Supersedes: 2026-07-03-ptxemu-modification-task.md (仅 D1-Full Compute 部分)"
- ADR 引用 `ADR-NV-02-phase8b-d1-strategy.md` 与 `ADR-NV-01-gpu-soc-architecture-target.md`（来自综合任务书 §0.3 + §6）
- PTX-EMU 端有 ADR-0021（`docs/adr/0021-cpptlm-d1-full-integration.md`，未独立审计）
- 2026-07-16 RFC **评审快照时 untracked，现已在 commit `e9014de` 提交**（§1 [PATCH in-session]）— 4 项跨端契约询问（RFC-001~004）尚无 PTX-EMU 回执；任何"符合 ADR-XXXX 决策"的声明需逐条对照 ADR 实际内容

---

## 12. 三方独立裁决（Three Separate Verdicts）

### 12.1 Verdict A — 当前分支作为"D1 实施"提交

**NO-GO** ❌（保持不变）

**依据**:
1. CppTLM 端**零 D1 实施** — 仅 ABI header + 6 个 OpenSpec artifacts（§4.3）
2. CppTLM 端 `cpptlm-f12b-ld-impl/tasks.md` 验收门全 `[ ]`（§6.2）
3. 2026-07-16 RFC 文件**存在但 untracked**（评审快照时）— **[PATCH in-session] 已通过 commit `e9014de` 提交**；仍**不是**已批准规范
4. PTX-EMU 端 cpptlm-d1-full 验收门 8 项全 `[ ]`（`openspec list` 51/61，§6.1）
5. 双端 P1 D1-Full Compute 实施均未启动（§5.2）

**作为"规划/header staging"提交** — **CONDITIONAL ⚠️**

可接受若:
- 仅作为**ABI header 阶段 + 文档对齐**提交（不声称"D1 实施"）
- CppTLM 端 `cpptlm-f12b-ld-impl/proposal.md` 范围**明确缩小**至 "P0 HSK-1 ABI header staging only"（移除 MemoryBridge/KernelLaunchTLM 实施内容）
- PTX-EMU 端 `cpptlm-d1-full/tasks.md` 范围**明确缩小**至 "Phase 1-7 已完成；Phase 8 HSK 待发出；验收门待 HSK 双向确认"（不标"100% 完成"）
- 双端 tasks.md 验收门**保持 `[ ]`** 不强行勾选

### 12.2 Verdict B — P1 D1-Full Compute 设计/实施

**设计/规划 — GO** ✅（保持不变；与新 RFC 兼容）

可接受:
- 新建独立 OpenSpec change `cpptlm-d1-full-compute` 在双端
- 引用综合任务书 §3 + ADR-NV-02 + ADR-0020
- 12 端点 `static_assert` 在双端编译期锁定（**前提是 PTX-EMU 端选 RFC-001 Option A**）
- untracked `cpptlm-d1-p1-pipeline-scoreboard/` 必须先 git add + commit（或删除后重建为新 change）

**实施 — CONDITIONAL ⚠️**

阻塞条件（任一未满足则 NO-GO）:
- ❌ RFC-001 由 PTX-EMU 端**正式回执选 A**（48h SLA 内）
- ❌ CppTLM 端 P0 F12b-LD MemoryBridge 已完成并归档（或 P1 与 P0 解耦）
- ❌ PTX-EMU 端 P1 接口（scoreboard/pipeline/tensor_core_interface.h）已提交（`cpptlm-phase8b-injection-points` 状态确认）
- ❌ CppTLM 端 OpenSpec `cpptlm-d1-full-compute` 完整 artifacts 已 git-tracked（不是 untracked）

### 12.3 Verdict C — RFC 合规性声明

**CONDITIONAL ⚠️**（从原 NO-GO 升级；RFC 已发现 → **[PATCH in-session] 通过 `e9014de` 提交**，但仍是 design-time questions）

**依据**:
1. 2026-07-16 RFC 文件**评审快照时 untracked**（§1）— **[PATCH in-session]：已在 `e9014de` 提交**；4 项询问（RFC-001~004）均 🟡 Pending
2. RFC 是"design-time questions"而非"已批准规范" — 即使合并后，"RFC 合规"语义不成立
3. 综合任务书与协作同步是**设计参考**而非 RFC — 不可替代
4. 但 RFC-002 选项 B（delegate `synchronize_stream`）与 RFC-003 选项 A（`PendingKernel` 加 `copied_args`）**与本评审 §9 H6/H2 完全对应** — 接受可让本评审多条 HIGH 直接降级

**可接受路径**:
- PTX-EMU 端在 48h 内**正式回执** RFC-001/003/004；RFC-002 在 1 周内选定 A 或 B。
- RFC-001 选 A → 接受 12 端点 static_assert 推迟到 P1。
- RFC-002 选 B → 接受 `cudaStreamSynchronize` delegate，**§9 H6 关闭**。
- RFC-003 选 A → 接受 `PendingKernel` 加 `copied_args` 字段，**§9 H2 范围缩窄为"参数类型元数据待补"**。
- RFC-004 选 A → 在 3 个 `proposal.md` 加 `Cross-Project Counterparts` 表，**§11.1 部分闭合**。
- CppTLM 端需 `git add` RFC 文件并 commit（建议 commit hash: `e9014de docs(superpowers): outbound 4 RFCs to PTX-EMU team`）— 完成后 §3.1 worktree HEAD 才更新到 `e9014de`。
- 仅当上述全部达成，"RFC 合规性" 才能从"CONDITIONAL" 升级为 "GO"。

---

## 13. 必须执行（Required Actions）

### 13.1 优先行动表

| 优先级 | Owner | 行动 | 阻塞范围 | 验证命令 | 期望结果 |
|:-----:|:-----:|------|----------|----------|----------|
| **P0** | CppTLM | RFC 文件 `git add` + commit（建议 hash `e9014de`），并向 PTX-EMU 端发正式邮件 | RFC 合规性声明 + §3.1 HEAD 更新 | `git log --oneline -- docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu.md \| head -1` | HEAD 含此 commit |
| **P0** | PTX-EMU | 确认或补齐 cpptlm-d1-full 验收门 8 项 + Phase 1.2 + Phase 3.8（共 10 项） | "PTX-EMU 端完成"声明 | 逐项勾选 + ctest 0 fail | `openspec list` 显示 61/61 |
| **P0** | CppTLM | 重新 vendor `cpptlm_bridge.h` 到 PTX main HEAD SHA | 双端 ABI 对齐 | `git show 73e09d97:include/cudart/cpptlm_bridge.h > include/cudart/cpptlm_bridge.h && sha256sum` | SHA = `ca716a8179841da6de76e0c54406c76d21e42ca3cb8e08a8cd48907f865fe5e7` |
| **P0** | CppTLM | 决策 `cpptlm-d1-p1-pipeline-scoreboard` 目录命运（commit 或删除） | untracked 状态 | `git add` + commit 或 `rm -rf` | 工作树干净 |
| **P0** | PTX-EMU | 在 48h 内对 RFC-001/003/004 给出决定 | §12.3 RFC 合规性裁决 | 邮件回执或 commit `cpptlm-d1-full/hsk-1.md` 修订 | 4 项状态变为 🟢/🟠/🔴 |
| **P0** | PTX-EMU | 在 1 周内对 RFC-002 选定 A 或 B | §9.2 H6 closure | 修改 `cudart_sim.cpp:972-1017` 或 bump `CPPTLMBRIDGE_VERSION` | 文档化决策 |
| **P1** | PTX-EMU | 修复 `cudart_sim.cpp:154-159` nullptr 哨兵 → PTX 元数据 args_count | 内存安全 | ctest + 新单元测试覆盖 | 无 nullptr 哨兵依赖 |
| **P1** | PTX-EMU | 修复 `cudart_sim.cpp:494-506` 固定 8 字节 → PTX 类型表 size | 大参数正确性 | 新单测覆盖 struct/`__int128` | 字节级正确 |
| **P1** | PTX-EMU | 修复 `cudart_sim.cpp:819-855`/`972-1017` mutex 内 `poll_kernel` | 潜在锁倒置 | code review + 锁依赖图 | `poll_kernel` 在锁外 |
| **P1** | 双方 | 决策 `synchronize_stream` 接口（删除或生产路径使用） | 接口最小化 | RFC-002 决策 + 代码修改 | 文档化决策 |
| **P2** | CppTLM | 实施 MemoryBridge（综合任务书 §2.2 Task #C1） | P0 F12b-LD 集成 | 7 个单测 PASS | `tests/unit/cpptlm/test_memory_bridge.cc` 存在 |
| **P2** | CppTLM | 实施 KernelLaunchTLM EventQueue 集成（§2.2 Task #C2） | P0 集成 | 4 个单测 PASS | `tick()` 调用桥 + PTX-EMU |
| **P2** | PTX-EMU | 锁定 `CPPTLM_COMMIT_HASH` 默认值（不是 "main"） | ABI 漂移风险 | 修改 CMakeLists.txt:132 | 固定 SHA |
| **P3** | 双方 | 启动 P1 D1-Full Compute 实施（综合任务书 §3） | D1 完整 | 新 OpenSpec change | 4 Adapter + 3 模块 |

### 13.2 不应做出（Claims MUST NOT MAKE）

- ❌ "F12b-LD MemoryBridge 已 100% 完成" — 双端均未完整实施（PTX 端验收门 8 项未勾选；CppTLM 端实施不存在）
- ❌ "本 change 符合 2026-07-16 RFC" — RFC 评审快照时 untracked [PATCH in-session：已在 `e9014de` 提交]，但仍是 design-time questions 不是已批准规范
- ❌ "CppTLM 端 MemoryBridge 已就绪可对接" — `memory_bridge.hh/cc` 不存在
- ❌ "PTX-EMU 端 cpptlm-d1-full 已可归档" — 验收门 8 项 + Phase 3.8 + Phase 1.2 共 10 项未勾选
- ❌ "双端 ABI 完全一致" — SHA 不一致；vendored header 缺 4 项
- ❌ "P1 D1-Full Compute 进度" — 双端均未启动实施
- ❌ "CppTLM D1 测试通过" — 0 个 D1 测试存在
- ❌ "poll_kernel 会死锁" — **条件性**风险，**未证实**（CppTLM 端实现不存在）
- ❌ "vendored header 无法编译" — **未实测**，**不能声明**；仅说"ABI 漂移前置条件"
- ❌ "6 个目标测试共 26 个 assertions" — ctest 输出未列出每个测试 assertion 数；该数值属推测，**不能**精确断言

### 13.3 必须先达成的前置

任何"实施 F12b-LD MemoryBridge"声明前必须满足:

1. ✅ 2026-07-16 RFC 文件已 `git add` + commit（建议 hash `e9014de`），并向 PTX-EMU 端**正式发邮件**；PTX-EMU 端在 48h/1 周 SLA 内回执 4 项（§1）
2. ✅ CppTLM 端 `cpptlm-f12b-ld-impl/tasks.md` Phase 0.1-0.4 全勾选（含 ABI 重新 vendor）
3. ✅ CppTLM 端 `cpptlm-f12b-ld-impl/tasks.md` Phase 1.1-1.3 全勾选（MemoryBridge + KernelLaunchTLM + tests/unit）
4. ✅ PTX-EMU 端 `cpptlm-d1-full/tasks.md` Phase 1.2 / Phase 3.8 / 验收 8 项全勾选（`openspec list` 61/61）
5. ✅ 双端互发 HSK-1/2/3（commit hash 双向同步锁定）
6. ✅ 真实 `libcpptlm_cudart.so` 在 PTX-EMU 端 BUILD_LIB_CPPTLM_CUDART=ON 路径构建成功
7. ✅ G-F0 vector_add 烟雾测试通过（双端联合验证）

---

## 14. 附录 — 证据清单

### 14.1 已读文件（已验证存在）

- `/workspace/project/CppTLM/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md` (949 行)
- `/workspace/project/CppTLM/.worktrees/feature-d1-full-impl/include/cudart/cpptlm_bridge.h` (161 行)
- `/workspace/project/CppTLM/.worktrees/feature-d1-full-impl/include/tlm/gpu/kernel_launch_tlm.hh` (62 行)
- `/workspace/project/CppTLM/.worktrees/feature-d1-full-impl/src/tlm/gpu/kernel_launch_tlm.cc` (22 行)
- `/workspace/project/CppTLM/.worktrees/feature-d1-full-impl/src/CMakeLists.txt` (88 行)
- `/workspace/project/CppTLM/.worktrees/feature-d1-full-impl/openspec/changes/cpptlm-f12b-ld-impl/{.openspec.yaml, design.md, internal-plan.md, proposal.md, spec.md, tasks.md}` (6 文件)
- `/workspace/project/CppTLM/.worktrees/feature-d1-full-impl/openspec/changes/cpptlm-d1-p1-pipeline-scoreboard/{design.md, proposal.md, tasks.md, specs/}` (4 文件，untracked)
- `/workspace/project/PTX-EMU/include/cudart/cpptlm_bridge.h` (184 行)
- `/workspace/project/PTX-EMU/src/cudart/cudart_sim.cpp` (1255 行，已读 140-579, 810-1029)
- `/workspace/project/PTX-EMU/src/ptxsim/instructions/memory.cpp` (196 行，已读 1-100)
- `/workspace/project/PTX-EMU/CMakeLists.txt` (152 行，已读 110-152)
- `/workspace/project/PTX-EMU/openspec/changes/cpptlm-d1-full/tasks.md` (300 行)
- `/workspace/project/PTX-EMU/openspec/changes/cpptlm-phase8b-injection-points/` (7 文件，目录存在)

### 14.2 实测命令结果

```
$ cd /workspace/project/CppTLM && git rev-parse HEAD
c89d9966f24b4ee6339dee52703c348805793cce

$ cd /workspace/project/CppTLM/.worktrees/feature-d1-full-impl && git rev-parse HEAD
abd7dd5ebd39c6c2b05489eec5d61111a3a1b471
$ git status --short
?? openspec/changes/cpptlm-d1-p1-pipeline-scoreboard/

$ cd /workspace/project/PTX-EMU && git rev-parse HEAD
73e09d97610404f44106c47e427f74ca1d452df9

$ sha256sum ...
ca716a8179841da6de76e0c54406c76d21e42ca3cb8e08a8cd48907f865fe5e7  PTX-EMU/include/cudart/cpptlm_bridge.h
c19e66a32de398e6bba2042f3f19923ff89dbc02f10bbf310c073ad3a8ff3dbe  CppTLM/.../cpptlm_bridge.h

$ git diff c89d996 HEAD --stat   # CppTLM worktree
8 files changed, 1691 insertions(+)

$ cd CppTLM-worktree/build && ./bin/cpptlm_tests
===============================================================================
All tests passed (15547 assertions in 764 test cases)

$ cd CppTLM-worktree/build && ctest
No tests were found!!!

$ cd PTX-EMU/build && ctest -R "cpptlm|bridge|attach|singleton|stream_sync"
6/6 tests passed (0.32 sec total real time)
  #105 unit_stream_sync_loop           Passed 0.08 sec
  #115 unit_cpptlm_bridge              Passed 0.02 sec
  #116 unit_cpptlm_attach_bridge       Passed 0.04 sec
  #183 integration_cpptlm_singleton_guard       Passed 0.02 sec
  #184 integration_cpptlm_async_launchkernel   Passed 0.03 sec
  #185 integration_cpptlm_ld_st_bridge         Passed 0.02 sec

$ find /workspace/project/CppTLM /workspace/project/PTX-EMU -name "2026-07-16*"
# 仅 main 与 worktree committed tree 中**无** 2026-07-16 文件；
# 但 worktree untracked 包含 docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu.md (9,392 字节, 165 行, mtime 2026-07-16 15:31)
# git log/git for-each-ref/reflog 中均无该文件 commit — 仍属 worktree 暂存态

$ git status --short   # CppTLM worktree
?? openspec/changes/cpptlm-d1-p1-pipeline-scoreboard/
# 注意：原 2026-07-16 RFC 文件也可能是 untracked；`ls -la` 确认 mtime

$ find CppTLM-worktree -name "memory_bridge*" -o -name "*scoreboard*" -path "*tlm/gpu*" -name "*.hh"
（空 — D1 实施不存在）
```

### 14.3 置信度标注约定

- **FACT, 高置信度** — 文件/命令直接产出，行号已列
- **INFERENCE, 高置信度** — 由 FACT 通过直接逻辑推导，无假设跳跃
- **INFERENCE, 中等置信度** — 由 FACT 推导但含 1-2 处假设（如 pre-existing 失败的归因）
- **INFERENCE, 低置信度** — 推导路径含多处假设（本文未使用）
- **UNKNOWN** — 缺乏证据无法判断（本文 §11.2 多处使用）

---

## 15. 维护

- **生成日期**: 2026-07-16
- **作者**: 跨仓库评审（独立审计）
- **下次评审触发条件**:
  - 2026-07-16 RFC 发布后
  - CppTLM 端 D1 实施启动后
  - PTX-EMU 端 cpptlm-d1-full 验收门全勾选后
  - 双端 HSK-1/2/3 实际发出后
- **关联文档**: [`docs/audits/HEALTH-AUDIT-2026-06-21.md`](./audits/HEALTH-AUDIT-2026-06-21.md), [`docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md`](./audits/HEALTH-AUDIT-2026-06-21-ERRATA.md), [`docs/dev-process/lessons-learned.md`](./dev-process/lessons-learned.md)
- **审计约束**: 本报告遵循 [`docs/dev-process/lessons-learned.md`](./dev-process/lessons-learned.md) 的 lessons-learned（行级 Diff / 锁集中审计 / 不混用 Phase commit / 基线 worktree / Qualifier 遍历全集 / Artifacts git-tracked / Pre-implementation Review / 重大交付 README 同步）

### 15.1 新增 Lessons-Learned（[PATCH 2026-07-16 in-session]）

> **LL-NEW-XX: 跨仓库评审前必须 `git add` + commit 所有评审对象文件，否则基线快照看不到 untracked 内容**

- **失败模式**: 本评审初始 pass 错误声称"2026-07-16 RFC 未发现"，实际文件已经存在但作为 untracked 工作树文件存在，由于评审快照（`abd7dd5`）未包含此文件，导致 `git show` / `git log` / `diff` 三种常规检索路径完全看不到。二次审查通过 `ls -la` + `find ... -newer` 才在 1 小时内定位到此文件，浪费一整轮评审时间。
- **根因**: CppTLM Metis Round 4 产出 RFC 文件后未 `git add` + commit（评审快照阶段）；评审者默认依赖 `git show HEAD:` 检索，未显式列 `??` 状态文件
- **预防**（CI 可强制）:
  1. 跨仓库评审 PR 模板添加 **"评审对象完整性自检"** 项，要求评审请求方先执行 `git status --short | grep '^?'` 输出为空
  2. 评审者第一步必须执行 `ls -la docs/superpowers/specs/*.md | grep -v "Aug\|Jul 1"`（按 mtime 倒序）发现最近 mtime 的文件
  3. 任何 `git status` 显示 `??` 的文件，**不能**视为"评审基线外内容"略过；必须 `git add` + commit 并修订评审基线 HEAD，再开始正式评审
  4. 评审快照报告（`docs/superpowers/findings/*.md`）必须在标题下方记录 `评审基准 HEAD: <hash> + 评审日期`，过期后立即在顶部加 `SNAPSHOT STALE` 警告
- **本评审的双向交叉引用**:
  - **§9.1 B1**（RFC untracked BLOCKER）→ 已被该 LL 解释的失败模式
  - **§4.2**（`cpptlm-d1-p1-pipeline-scoreboard/` untracked）→ 同一 LL 触发的次级 BLOCKER，已在 `505a7c9`/`57f5066` 缓解
  - **顶部 "⚠️ 快照过期" 警告** → 该 LL 的实施示例
- **后续待办**: 把本条 LL 与 `docs/dev-process/lessons-learned.md` §"OpenSpec artifacts 2-Phase commit" 合并；同步更新 `scripts/sanity.sh` 添加 `git status --porcelain | grep '^??' docs/` 检查
- **提交建议**: 沉淀 PR 标题 `docs(dev-process): add LL cross-repo-review-must-include-all-untracked-files (2026-07-16)`

---

## [PATCH v2 2026-07-17 in-session]

> **触发**: PTX-EMU 端 `cpptlm-phase8b-injection-points` Phase 0 对齐 commit `df05e10b` (2026-07-17 09:21) + HSK-3 Ready to Send commit `6b367cad` (2026-07-17) 触发全面事实层修订
>
> **新事实基础 (基于 CppTLM main HEAD `73e5422` + 后续 6 个 commit)**:
> - `73e5422` (P0 MemoryBridge merge to main, 776/776 用例 / 15562 断言 + 12/12 [f12b] tests)
> - `b94eccc` (cpptlm-f12b-ld-impl P0 archive)
> - `e69cd1d` (P2 AsyncCompletionAdapter placeholder, 5/5 [gpu][async] tests)
> - `3d83a1e` (B1-B4 文档一致性修复)
> - `ea60cbc` (f12b-ld-impl tasks.md P0 勾选)
> - `2b28505` (RFC-P1-001~004 发送, 含 Q1-Q5 答复 + 12-endpoint enum 锁定)
> - `25e7e3c` (CppTLM HSK-1/2/3 + D1-Full 状态回复文档, 含 CPPTLM_COMMIT_HASH 推荐 73e5422)

### v2 修订事实表

| 报告位置 | 原报告声称 | v2 修订 (基于实际状态 2026-07-17) | CppTLM commit |
|---------|---------|----------------------------------|---------------|
| §4.3 §171 | "include/tlm/gpu/memory_bridge.hh 不存在" | ✅ **已存在** (86 行) + .cc (147 行) | `73e5422` |
| §4.3 §172 | "4 Adapter 不存在" | ⏳ **P1 阻塞**: 等待 PTX-EMU cpptlm-phase8b-injection-points Phase 1 (3 接口头文件) | RFC `2b28505` |
| §4.3 §173 | "3 核心模块 {scoreboard,pipeline,tensor_core}_tlm 不存在" | ⏳ **P1 阻塞**: 同上 | RFC `2b28505` |
| §4.3 §174 | "async_completion_adapter.hh 不存在" | ✅ **已存在** (hh + 5 单测, P2 占位) | `e69cd1d` |
| §4.3 §175 | "tests/unit/cpptlm/test_memory_bridge.cc 不存在" | ✅ **已存在** at `test/test_memory_bridge.cc` (12 用例 / 15 断言) | `73e5422` |
| §4.3 §176 | "tests/integration/cpptlm/test_f12b_integration.cc 不存在" | ⚠️ **部分缺失**: CppTLM 端为 `test/test_memory_bridge.cc` + `test/test_kernel_launch_tlm_ext.cc` (按 CppTLM 命名约定); 集成测试为 `test/python/test_f12b_smoke.py` (G-F0 烟雾测试) | `73e5422` |
| §4.3 §177 | "tests/python/test_f12b_smoke.py 不存在" | ✅ **已存在** (vector_add 烟雾测试) | `73e5422` |
| §4.3 §178 | "CMakeLists.txt memory_bridge 目标不存在" | ✅ **已集成** (`src/CMakeLists.txt:46`, 含在 `cpptlm_core` 静态库) | `73e5422` |
| §6.2 §282 | "cpptlm-f12b-ld-impl/tasks.md 全部 [ ]" | ✅ **已勾选 24 项** (Phase 0 + Phase 1 + G-F0~G-F5) | `ea60cbc` |
| §10.1 §619 | "15547 assertions in 764 test cases" | ✅ **15574 assertions in 781 test cases** (+5 AsyncCompletion 用例 / +12 断言) | `e69cd1d` |
| §9.1 B2 §573 | "CppTLM 端 D1 实施完全不存在" | ✅ **P0 + P2 已实施**;P1 (3 cores + 4 Adapter) 仍阻塞但有明确解锁条件 | `73e5422`/`e69cd1d` |
| §7.1 §471 | "ABI SHA 不一致 (ca716a81 vs c19e66a3)" | ⚠️ **半错半对**: SHA 仍不同 (但这是 2026-07-16 计划 re-vendor, 双 commit hash 都已记录在 `include/cudart/AGENTS.md` + SHA-256 双重锁定) | 文档化 |
| §6.1 §253 | "PTX-EMU cpptlm-d1-full 验收 8 项 + Phase 3.8 共 10 项未完成" | ⏳ **仍有效**: PTX-EMU 端 cpptlm-d1-full 验收门未闭合, 阻塞该 change 归档 | 待 PTX-EMU 推进 |

### v2 保留有效结论

- §6.3 限制 #1 (count_kernel_args nullptr 哨兵) — 待 PTX-EMU 修
- §6.3 限制 #2 (deep-copy 固定 8 字节) — 待 PTX-EMU 修
- §6.3 限制 #3 (mutex 内调 poll_kernel 潜在死锁) — 待双端验证 (CppTLM MemoryBridge 实现已避免锁反向, 见 `memory_bridge.cc::poll_kernel` 查 map 不持外部锁)
- §6.3 限制 #4 (synchronize_stream 生产代码未调用) — 待 PTX-EMU 端按 RFC-002 选项 B 修复 (`cudaStreamSynchronize` delegate 到 `bridge->synchronize_stream`)
- §8.2 version() 未运行时检查 — 待 PTX-EMU 加
- §9.2 H1-H7 (HIGH 风险) — B2 解除, 其余待 PTX-EMU 端 cpptlm-d1-full 验收闭合
- §11.1 cpptlm-d1-full 验收 8 项 + Phase 3.8 共 10 项未完成 — 阻塞 cpptlm-d1-full 归档
- §11.2 cpptlm-d1-full/tasks.md 内部一致性 — 待 PTX-EMU 团队独立审计
- §15.1 LL-NEW-XX 跨仓库评审前必须 `git add` + commit 所有评审对象文件 — **仍有效**, 已被本 PATCH v2 验证 (再次发现 untracked/快照过期问题)

### v2 修订 Verdict 状态

| Verdict | 原报告 | v2 修订 | 依据 |
|---------|--------|---------|------|
| **A — 当前分支作为"D1 实施"提交** | NO-GO ❌ | **CONDITIONAL PASS** ⚠️ (P0/P2 已实施, P1 阻塞) | `73e5422` + `e69cd1d` |
| **B — P1 D1-Full Compute 设计/实施** | 设计 GO + 实施 CONDITIONAL | **设计 GO + 实施 仍 CONDITIONAL** (阻塞未解除, 等待 PTX-EMU Phase 1 接口) | RFC `2b28505` (RFC-P1-001 已发出) |
| **C — RFC 合规性声明** | CONDITIONAL ⚠️ | **CONDITIONAL ⚠️** (不变, 待 PTX-EMU 端 RFC-P1-001~004 回执) | RFC `2b28505` 已发出, 等回执 |

### v2 阻塞关系图 (CppTLM P1 解锁条件)

```
PTX-EMU 端 (需先实施):
  cpptlm-phase8b-injection-points Phase 1 (3 接口头文件)
   │
   ├─ include/ptxsim/scoreboard_interface.h
   ├─ include/ptxsim/pipeline_interface.h
   └─ include/ptxsim/tensor_core_interface.h
        │
        ▼
CppTLM 端 (可启动):
  cpptlm-d1-p1-pipeline-scoreboard Phase 1 (3 核心模块)
   │
   ├─ include/tlm/gpu/scoreboard_tlm.{hh,cc} (≥12 entries)
   ├─ include/tlm/gpu/pipeline_tlm.{hh,cc} (5+V PipelineId)
   └─ include/tlm/gpu/tensor_core_tlm.{hh,cc} (6 TcPrecision)
        │
        ▼
  cpptlm-d1-p1-pipeline-scoreboard Phase 2 (4 Adapter)
   │
   ├─ cpptlm_warp_scheduler_adapter (WarpContext* ↔ uint32_t)
   ├─ cpptlm_scoreboard_adapter (ScoreboardTLM → IScoreboard)
   ├─ cpptlm_pipeline_adapter (PipelineTLM → IPipelineLatencyProvider)
   └─ cpptlm_tensor_core_adapter (TensorCoreTLM → ITensorCoreTiming)
        │
        ▼
  cpptlm-d1-p1-pipeline-scoreboard Phase 4 (12 端点 static_assert)
        │
        ▼
  cpptlm-d1-p1-pipeline-scoreboard Phase 5 (双端 G-D5 microbenchmark ±15% vs gpgpu-sim)
```

### v2 跨仓库交付物状态 (截至 2026-07-17)

| 端 | 交付物 | Commit / 路径 | 状态 |
|----|-------|--------------|------|
| **CppTLM** | P0 MemoryBridge 归档 | `b94eccc` → `openspec/changes/archive/2026-07-16-cpptlm-f12b-ld-impl/` | ✅ Archived |
| **CppTLM** | P2 AsyncCompletion 占位 | `e69cd1d` → `include/tlm/gpu/async_completion_adapter.hh` | ✅ Merged |
| **CppTLM** | P1 RFC-P1-001~004 | `2b28505` → `docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md` | ✅ Sent |
| **CppTLM** | B1-B4 文档一致性修复 | `3d83a1e` → openspec/changes/* (9 文件) | ✅ Merged |
| **CppTLM** | P0 tasks.md 勾选 | `ea60cbc` → `openspec/changes/cpptlm-f12b-ld-impl/tasks.md` | ✅ Merged |
| **CppTLM** | Phase 0.5 baseline | `docs/superpowers/specs/2026-07-15-phase05-baseline-report.md` (781/781 用例 / 15574 断言) | ✅ Verified |
| **CppTLM** | HSK-1/2/3 + D1-Full 状态回复 | `25e7e3c` → `docs/superpowers/specs/2026-07-17-hsk-1-2-3-responses.md` | ✅ Sent |
| **PTX-EMU** | ADR-0020 | `docs/adr/0020-cpptlm-injection-points.md` Status: Accepted | ✅ Accepted |
| **PTX-EMU** | Phase 0 对齐 | `df05e10b` (Q1-Q5 + 12-endpoint enum 锁定) | ✅ Aligned |
| **PTX-EMU** | HSK-3 Ready to Send | `6b367cad` (CPPTLM_COMMIT_HASH=73e5422 锁定) | ✅ Ready |
| **PTX-EMU** | cpptlm-d1-full 实施 | 51/61 → 55/61 tasks (Phase 1-7 [x]) | ⚠️ 验收 10 项未闭 |

### v2 新增 Lessons-Learned

> **LL-NEW-02: 跨仓库评审的"快照过期"是常态而非例外**

- **失败模式**: 本评审初始快照 (`57f5066`) → [PATCH v1] 增至 `57f5066` → [PATCH v2] 增至 `73e5422` + 6 commit 偏移。每次"修订"都基于 CppTLM 团队 push 后被动触发。**主动快照对齐机制缺失**。
- **根因**: 跨仓库评审未约定"评审基准 HEAD 必须每 N 小时重对齐一次"或"CppTLM push 后自动通知 PTX-EMU 评审 reviewer"
- **预防**:
  1. PTX-EMU 评审 reviewer 在 CppTLM `openspec/changes/` 下设置 webhook (或 daily cron),发现 untracked/新 commit 时主动 rerun 评审
  2. 或: 跨仓库评审报告自带"基线 HEAD 自动更新"机制 (git notes + last-modified timestamp)
  3. 或: 双端约定每日 EOD 同步一次 (避免评审过期超过 24h)

---

**最后更新**: 2026-07-17 ([PATCH v2 2026-07-17 in-session], 7 个 CppTLM commit 增量)
**下次评审触发**: PTX-EMU cpptlm-phase8b-injection-points Phase 1 (3 接口头文件) 提交后, 启动 CppTLM P1 Phase 1 (3 核心模块) 实施
