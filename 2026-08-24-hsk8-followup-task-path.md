# HSK-8 Phase 2 Follow-up Task Path (PTX-EMU)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 HSK-8 Phase 2 已完整归档(PTX-EMU + CppTLM 双方)的基础上,落地 5 步推荐任务路径,覆盖 docs sync + Phase 2.2/2.3 delegation 实施 + 推迟 Phase 1.5/HSK-9 规划。

**Architecture:** 三层结构:
1. **立即层**(Quick): push main + docs-sync 单 commit(AGENTS.md 7 stale + README §已实现功能补 IPtxEmuDevice 条目)
2. **近期层**(Medium 1-2d): 新 OpenSpec change `device-api-delegation` 实施 Phase 2.2/2.3 stub delegation,§7 Metis pre-impl review + §8 四件套
3. **延后层**(`[~] DEFERRED`): Phase 1.5 namespace 迁移 + HSK-9 consumer_smoke,触发条件驱动

**Tech Stack:** C++20、ANTLR4 PTX、OpenSpec、Catch2、ctest、CMake、HSK protocol

**Prerequisites:**
- 当前分支 `main`(HEAD = `530bd6ca`,ahead of origin/main by 3 commits)
- PTX-EMU OpenSpec change 已 archived 到 `openspec/changes/archive/2026-08-24-ptxemu-public-device-api/`
- CppTLM main HEAD = `beb3db8`(`external/PTX-EMU` submodule pinned at `530bd6ca`)
- `PTXEMU_API_VERSION=1` 冻结(drift_check Invariant 1 + `device_api.h:117` static_assert)
- 已加载 skill: `.opencode/skills/ptx-lessons-learned/`、`ptx-barrier-mechanism/`(set_active_mask overwrite 语义)、`test-coverage-enforcer/`(2.2/2.3 验证)

**Oracle consultation session ID**(可续接 Oracle): `ses_fcbc066a6ffeXfv5kIgmE8UwrB`

**Postmortem 关联:** `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` §"2026-08-24 Postmortem"(commit `8aa72f1d`)

---

## Oracle 分析综合 (验证通过 2026-08-24)

### 3 大 Hypotheses

| # | 假设 | 置信度 | 关键证据 |
|---|---|---|---|
| H1 | **docs-sync 单 commit**(非 2 commits) | HIGH | `tasks.md:92` Phase 5 `3678a0d7` 单 commit 先例 + ptx-lessons-learned §3 同 phase 同 commit 原则 |
| H2 | **Phase 2.2/2.3 不阻塞于 Phase 1.5,优先做** | HIGH | `device_api_impl.cc:91-137` stub 用内部类型(SMContext/WarpContext/ThreadContext)不经公共头;Phase 1.5 历史级联 build 失败(tasks.md:34 记录) |
| H3 | **Phase 1.5 + task 9.4 捆绑延后** | MEDIUM | `tasks.md:125` 9.4 前置 3.7;触发条件: CppTLM 侧 `ptx_ir/` include 需求 或 下个 release 窗口 |

### 验证门禁已通过(oracle-prompting §规则 4)

| 引用 | 验证结果 |
|---|---|
| `device_api_impl.cc:91-137` stub status | ✅ 5 stubs confirmed(set_scoreboard, set_active_mask, set_next_pc, get_warp_status, is_finished) + attach_timing stub |
| `tasks.md:34,67,125` DEFERRED status | ✅ 3 items `[~] DEFERRED` confirmed |
| `README.md` §已实现功能 missing IPtxEmuDevice | ✅ 0 matches for `IPtxEmuDevice\|ptxemu_core\|public device`(§21 违规) |
| `device_api.h:117` static_assert | ✅ `PTXEMU_API_VERSION==1` + HSK-9 trigger message |
| `AGENTS.md:23,30,33,37-40` 7 stale markers | ✅ 全部验证 |
| **`SMContext::set_scoreboard(IScoreboard*)` 存在性** | ✅ **`include/ptxsim/sm_context.h:87` 已存在** — Oracle Open Question 1 自动 resolved,Phase 2.2 scope 不扩大 |

### 3 大风险

| 风险 | 来源 | 缓解 |
|---|---|---|
| **R1**: `device_api.h` 修改 = 双仓硬失败 | Q5 | drift_check Invariant 1 hard-fail + CppTLM 下次 bump PR 编译断裂 → Phase 2.2/2.3 全程 `git diff --name-only` 守门不含 `device_api.h` |
| **R2**: `set_active_mask` 语义陷阱 | `device_api_impl.cc:105-109` overwrite 注释 | unit test 覆盖 overwrite + e2e 经 `execute_warp_instruction` 验证 thread PC(test-coverage-enforcer 强制) |
| **R3**: README 同步再次遗漏 | §21 违规史 | "README §已实现功能 条目" 写入 change `device-api-delegation` tasks.md 强制 checkbox + archive 前 `ac-verifier` 检查 |

---

## 文件变更总览

| 文件 | 操作 | 责任 |
|---|---|---|
| `AGENTS.md` | 修改 lines 23,26,30,32,33,37-40(7 stale markers) | Phase 1 docs-sync |
| `README.md` | §已实现功能 添加 `IPtxEmuDevice`/`ptxemu_core` 条目(per §21) | Phase 1 docs-sync |
| `openspec/changes/2026-08-24-device-api-delegation/` | **新建** change(proposal/design/3 specs/tasks) | Phase 2 device-api-delegation |
| `src/ptxemu/device_api_impl.cc` | 实施 Phase 2.2 set_scoreboard/set_active_mask/set_next_pc + Phase 2.3 attach_timing(改 lines ~91-137) | Phase 2 实施 |
| `tests/unit/ptxemu/test_device_api_delegation.cpp` | 新增 unit test(overwrite 语义 + delegation) | Phase 2 验证 |
| `tests/integration/warp/test_device_api_delegation_e2e.cc` | 新增 e2e test(经 execute_warp_instruction) | Phase 2 验证 |
| `README.md` | §已实现功能 更新 IPtxEmuDevice 条目(Phase 2.2/2.3 完成) | Phase 2 实施后 |
| `openspec/changes/2026-08-24-device-api-delegation/tasks.md` | Phase 2 archive 前更新 + 添加 `Ref: archive/2026-08-24-ptxemu-public-device-api/` | Phase 2 archive |

---

## Phase 0: Push + 验证当前状态 (10 min)

### Task 0.1: Push 3 commits to origin/main

**Files:**
- 无文件变更(纯 git 操作)

- [ ] **Step 1: 验证 ahead of origin/main**

```bash
cd /workspace/project/PTX-EMU
git status && git log --oneline origin/main..HEAD
# 应显示 3 commits: d5600e89, 8aa72f1d, 530bd6ca
```

- [ ] **Step 2: Push to origin**

```bash
git push origin main
```

- [ ] **Step 3: 验证 CppTLM submodule 自动同步**(可选)

```bash
cd /workspace/project/CppTLM
git submodule update --remote external/PTX-EMU
git diff external/PTX-EMU
# 应为空(CppTLM 已 pin 在 530bd6ca,本次 push 不改变其内容)
```

**Verification gate:**
- `git log origin/main --oneline -3` 显示 `530bd6ca` 为 HEAD
- CppTLM `external/PTX-EMU` 仍 pin 在 `530bd6ca`

---

## Phase 1: docs-sync 单 commit (15 min)

### Task 1.1: 更新 AGENTS.md HSK 链段

**Files:**
- `AGENTS.md`(修改 7 stale markers)

- [ ] **Step 1: 更新 HSK-8 table row** (line 23)

```diff
-| **HSK-8** | 🔄 **PTX-EMU ACK 已发送** (738b412c, ack body 250+ 行) | PTX-EMU `738b412c` + Phase 2 PR `feat/ptxemu-public-device-api` (11 commits ahead) | `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` |
+| **HSK-8** | ✅ **ACCEPTED** (PR #14 merged `fcdad151`, CppTLM bump `beb3db8` 合并) | PTX-EMU `fcdad151` (squash merge of 12 impl commits) + CppTLM `beb3db8` (submodule pin at `530bd6ca`) | `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` + `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` §Postmortem |
```

- [ ] **Step 2: 更新实施进度 Phase 5** (line 30) + 添加 Phase 6 (line 31)

```diff
-- 🔄 Phase 5: 文档同步 (本 phase)
-- ⏳ CppTLM bump PR (待 HSK-8 Phase 2 PR 合入后由 CppTLM 触发)
+- ✅ Phase 5: doc sync (`include/ptxemu/AGENTS.md` + root AGENTS.md HSK chain + audit doc) — commit `3678a0d7`
+- ✅ Phase 6 (本 session): archive prep + postmortem + 4 main specs create — commits `d5600e89` / `8aa72f1d` / `530bd6ca`
+- ✅ CppTLM bump PR — commits `6f408b5` / `09c27d5` / `d035551` + submodule pin `beb3db8`
```

- [ ] **Step 3: 更新跨仓协调顺序步骤 2-5** (lines 37-40)

```diff
-2. 🔄 PTX-EMU Phase 2 PR (当前 feat/ptxemu-public-device-api 分支, 11 commits ahead)
-3. ⏳ PTX-EMU CI 全绿 (drift_check + ctest 全部 PASS)
-4. ⏳ PTX-EMU Phase 2 PR 合入 main (目标 2026-09-19 前, per HSK-8 ack 决策点 4)
-5. ⏳ CppTLM bump PR (submodule pin + add_subdirectory + 桥接残留簇删除)
+2. ✅ PTX-EMU Phase 2 PR — PR #14 merged `fcdad151`(2026-08-24T03:55:14Z by `chisuhua`, 分支已 archive + 删除)
+3. ✅ PTX-EMU CI 全绿 — ctest 246/246 + drift_check 5 invariants PASS
+4. ✅ PTX-EMU Phase 2 PR 合入 main — origin/main HEAD = `fcdad151`,**ahead of 2026-09-19 target by 26 天**
+5. ✅ CppTLM bump PR — submodule pin `beb3db8` → PTX-EMU `530bd6ca`, 5 HSK-8 commits merged (`6f408b5` + `09c27d5` + `d035551` + `12b9e0f` + `beb3db8`)
```

### Task 1.2: 更新 README.md §已实现功能 (per §21)

**Files:**
- `README.md`(在 §已实现功能 现有 5 个 bullet 后添加新 bullet)

- [ ] **Step 1: 添加 IPtxEmuDevice 条目**

```diff
 - **PTX-EMU Image Executor** (`libptxemu_device.so` + `cpptlm_module.h`): ...
 - **PTXIR-Embedded CUBIN/EXE**: ...
+- **PTX-EMU Public Device API** (`include/ptxemu/device_api.h` + `ptxemu_core` STATIC lib): CppTLM 消费入口 (`IPtxEmuDevice` 接口 + `PTXEMU_API_VERSION=1` 冻结 + 5 HSK-8 invariants via `drift_check.yml`)。`cpp 不暴露` 约束: CppTLM 侧 0 PTX-EMU 内部 header includes (commit `09c27d5`)。跨仓协议 HSK-8 ✅ ACCEPTED (PR #14 merged + CppTLM submodule bump `beb3db8`)。详见 [HSK-8 audit](./docs/audits/2026-08-13-hsk8-ptxemu-public-api.md) + [follow-up plan](./docs/superpowers/plans/2026-08-24-hsk8-followup-task-path.md)。
```

### Task 1.3: 提交 + Push

- [ ] **Step 1: 验证 stale markers 已全部清除**

```bash
grep -n "🔄\|⏳" AGENTS.md | grep -E "(HSK-8|Phase 5|Phase 2 PR|CppTLM bump|协调)" || echo "(no stale markers)"
```

- [ ] **Step 2: 提交单 commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: HSK-8 Phase 2 ✅ ACCEPTED — sync AGENTS.md HSK chain + README §已实现功能

Per Oracle consultation (ses_fcbc066a6ffeXfv5kIgmE8UwrB) Hypothesis 1
(docs-sync 单 commit per Phase 5 3678a0d7 先例) + ptx-lessons-learned
§21 Checklist I (重大功能交付 README 同步 4 项缺一不可)。

AGENTS.md HSK 链段 7 stale markers → ✅ ACCEPTED:
- Line 23 HSK-8 row: 🔄 ACK 已发送 → ✅ ACCEPTED (PR #14 fcdad151 +
  CppTLM bump beb3db8)
- Lines 30-31 Phase 5/6: 🔄 本 phase → ✅ 完成 (commits 3678a0d7 +
  d5600e89 + 8aa72f1d + 530bd6ca)
- Line 33 CppTLM bump: ⏳ → ✅ (5 HSK-8 commits merged)
- Lines 37-40 跨仓协调 step 2-5: 🔄/⏳ → ✅ 全部完成

README.md §已实现功能 补 IPtxEmuDevice / ptxemu_core 条目 (per §21):
- Public device API 入口 (HSK-8 spec §3 'cpp 不暴露' 约束)
- PTXEMU_API_VERSION=1 冻结 (drift_check Invariant 1)
- 5 HSK-8 invariants (drift_check.yml)
- CppTLM submodule pin beb3db8 reference
- HSK-8 audit + follow-up plan 链接

Refs:
- Oracle consultation: ses_fcbc066a6ffeXfv5kIgmE8UwrB
- HSK-8 ack: 738b412c
- PR #14 squash: fcdad151
- CppTLM bump commits: 6f408b5 + 09c27d5 + d035551 + 12b9e0f + beb3db8
- Follow-up plan: docs/superpowers/plans/2026-08-24-hsk8-followup-task-path.md
- ptx-lessons-learned §21 Checklist I"

git push origin main
```

**Verification gate:**
- `grep "🔄\|⏳" AGENTS.md | grep -E "(HSK-8|协调)"` 应为空
- `grep "IPtxEmuDevice" README.md` 应显示新增条目
- origin/main HEAD 更新

---

## Phase 2: 新 OpenSpec change `device-api-delegation` (1-2d)

### Task 2.1: 创建 OpenSpec change skeleton

**Files:**
- `openspec/changes/2026-08-24-device-api-delegation/{proposal.md,design.md,specs/,tasks.md}`

- [ ] **Step 1: 创建 change 目录**

```bash
mkdir -p openspec/changes/2026-08-24-device-api-delegation/specs
```

- [ ] **Step 2: 写 proposal.md**(OpenSpec format,引用 `archive/2026-08-24-ptxemu-public-device-api/` 作为前置)

内容要点:
- **Why**: Phase 2.2/2.3 stub 当前返回 false/null, CppTLM facade (`beb3db8` 状态) 已消费 `IPtxEmuDevice` 但 stub 不提供真实 delegation, 阻塞 CppTLM 实际使用
- **What**: 实施 `set_scoreboard` / `set_active_mask` / `set_next_pc` (Phase 2.2) + `attach_timing` (Phase 2.3) delegation to SMContext/WarpContext/ThreadContext
- **Impact**: 
  - 不修改 `include/ptxemu/device_api.h`(签名冻结, PTXEMU_API_VERSION=1 保持)
  - 仅修改 `src/ptxemu/device_api_impl.cc` stub body
  - 不影响 `cpp 不暴露` 约束 (CppTLM 端无需变更)

- [ ] **Step 3: 写 design.md**

内容要点:
- Decision 1: **delegation 顺序** — `set_scoreboard` → `set_active_mask` → `set_next_pc` (per ptx-lessons-learned §1 跨模块状态翻译)
- Decision 2: **overwrite 语义** — `set_active_mask` overwrite NOT OR-merge (per `device_api_impl.cc:105-109` 注释 + BUG-RETHANG/BUG-POSTBARRIER-TWOHALVES)
- Decision 3: **`set_next_pc` 非 `force_set_pc`** — per AGENTS.md ANTI-PATTERNS
- Decision 4: **Phase commit 纪律** — Phase 2.2 (3 methods) + Phase 2.3 (1 method) 独立 commit,各自 ctest 验证 (§3)
- Decision 5: **§21 修复** — "README §已实现功能 条目更新" 写入 tasks.md 强制 checkbox(防 R3 复发)

- [ ] **Step 4: 写 3 个 specs**(OpenSpec ADDED Requirements format)

specs:
1. `specs/ptxemu-device-api-delegation/spec.md` — 包含 4 ADDED requirements (set_scoreboard / set_active_mask / set_next_pc / attach_timing)
2. `specs/delegation-thread-pc-invariants/spec.md` — 包含 2 ADDED requirements (per test-coverage-enforcer)
3. `specs/ci-drift-check-extension/spec.md` — 包含 1 ADDED requirement(drift_check 扩展验证 delegation 不可回归 stub)

- [ ] **Step 5: 写 tasks.md**(§7 + §8 + §9 + 引用 archive)

```markdown
## 1. Pre-Implementation 准备 (per ptx-lessons-learned §4 + §7 + this Oracle plan §Phase 2)
- [ ] 1.1 MUST: 验证基线 worktree `.worktrees/device-api-delegation-baseline` 编译通过(per §4 15-20min)
- [ ] 1.2 MUST: 跑基线 ctest 246/246 PASS(per §4)
- [ ] 1.3 MUST: Metis pre-impl review 审计 4 OpenSpec artifacts (per §7 + §H in ptx-lessons-learned)
- [ ] 1.4 应用 Metis MUST-RESOLVE 列表,重审直至 ⚠️→GO 或 ✅
- [ ] 1.5 NOTE: 4 artifacts 范围数字一致性按 Checklist J 校验
- [ ] 1.6 MUST: `git add openspec/changes/2026-08-24-device-api-delegation/` + commit (per Checklist E)
- [ ] 1.7 创建 `feat/device-api-delegation` 分支从 origin/main (`530bd6ca`) HEAD

## 2. Phase 2.2 delegation 实施 (per design Decision 1 + spec/ptxemu-device-api-delegation)
- [ ] 2.1 实施 `set_scoreboard` delegation (委托到 `SMContext::set_scoreboard(IScoreboard*)` at `sm_context.h:87`)
- [ ] 2.2 实施 `set_active_mask` delegation (overwrite 语义, NOT OR-merge, per BUG-RETHANG)
- [ ] 2.3 实施 `set_next_pc` delegation (NOT `force_set_pc`, per AGENTS.md ANTI-PATTERNS)
- [ ] 2.4 unit test 覆盖 3 个 delegation (tests/unit/ptxemu/test_device_api_delegation.cpp)
- [ ] 2.5 e2e test 经 execute_warp_instruction 验证 thread PC (tests/integration/warp/test_device_api_delegation_e2e.cc per test-coverage-enforcer)
- [ ] 2.6 full clean rebuild + ctest 验证 246/246 + 5/5 drift_check PASS
- [ ] 2.7 commit "feat(ptxemu): phase 2.2 set_scoreboard + set_active_mask + set_next_pc delegation"

## 3. Phase 2.3 attach_timing 实施 (per design Decision 4 + spec/ptxemu-device-api-delegation)
- [ ] 3.1 实施 `attach_timing` HSK-4 vendored interface injection (IScoreboard/IPipelineLatencyProvider/ITensorCoreTiming)
- [ ] 3.2 注入到 SMContext timing hooks (per design.md)
- [ ] 3.3 unit test 覆盖 attach_timing (tests/unit/ptxemu/test_device_api_attach_timing.cpp)
- [ ] 3.4 full clean rebuild + ctest 验证 246/246 + drift_check 5 invariants PASS (无 stub body 残留)
- [ ] 3.5 commit "feat(ptxemu): phase 2.3 attach_timing HSK-4 vendored interface injection"

## 4. README sync + 验证 (per ptx-lessons-learned §21 + this plan Risk R3)
- [ ] 4.1 更新 README.md §已实现功能 IPtxEmuDevice 条目(添加 "Phase 2.2/2.3 delegation 完成" 字样)
- [ ] 4.2 drift_check workflow 验证 delegation 不可回归 stub(grep `return false` 在 device_api_impl.cc 应为 0 except attach_timing)
- [ ] 4.3 ctest PASS(246 + 6 new unit + 1 e2e = ~253 tests)

## 5. PR submission (per archive/2026-08-24-ptxemu-public-device-api §8 先例)
- [ ] 5.1 push feat/device-api-delegation
- [ ] 5.2 PR #15 to main
- [ ] 5.3 merge (squash)
- [ ] 5.4 通知 CppTLM owner 可重新 bump (issue #22 评论)

## 6. Archive + Handoff
- [ ] 6.1 NOT archive change until CppTLM bumps (per HSK-8 ack 决策点 1)
- [ ] 6.2 archive 时 postmortem 段: Phase 2.2/2.3 实施回顾 + BUG-RETHANG 警告(已在 source code 注释体现)
- [ ] 6.3 update HSK-PROTOCOL-NOTES.md §HSK-8 实践示例 (引用本次 delegation 实施作为 HSK-9 准入准备示例)
```

### Task 2.2: Metis pre-impl review (per §7 强制)

- [ ] **Step 1: 调用 Metis 审计**

```bash
# 通过 Task tool 调用 Metis subagent
# 提供 4 artifacts 路径 + 强制引用 file:line 要求
# 要求输出: GO / ⚠️ CONDITIONAL / ❌ NO-GO 决策
# 重点验证:
#   - proposal 引用的 SMContext::set_scoreboard IScoreboard* 真实存在
#   - spec 声称的 4 ADDED requirements 范围与 design Decision 1-3 一致
#   - tasks.md 1.1 worktree 路径真实可创建
```

- [ ] **Step 2: 应用 MUST-RESOLVE 列表**(如有)

- [ ] **Step 3: 4 artifacts 范围数字一致性自检**(Checklist J)

```bash
# proposal 涉及的 openspec 文件清单 vs design.md Migration Plan Phase 列表
# vs 3 specs 涉及的 capability 列表 vs tasks.md Phase 列表 四者交叉一致
```

### Task 2.3: 实施 + 验证 (per Phase 2.2/2.3 tasks)

- [ ] **Step 1**: 建立基线 worktree(per §4 15-20min)

```bash
git worktree add .worktrees/device-api-delegation-baseline 530bd6ca
cd .worktrees/device-api-delegation-baseline
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build  # 必须全量 per §4 陷阱
cd build && ctest --output-on-failure  # 246/246 baseline
```

- [ ] **Step 2**: 创建 feat 分支

```bash
git checkout -b feat/device-api-delegation
```

- [ ] **Step 3**: 实施 Phase 2.2(per tasks.md §2)

- [ ] **Step 4**: Phase 2.2 commit(独立可 revert)

- [ ] **Step 5**: 实施 Phase 2.3(per tasks.md §3)

- [ ] **Step 6**: Phase 2.3 commit(独立可 revert)

- [ ] **Step 7**: README sync commit(per tasks.md §4 + §21)

- [ ] **Step 8**: PR + merge + CppTLM 通知(per tasks.md §5)

### Task 2.4: 验证 gate (per this Oracle plan Risks + R1)

- [ ] **Step 1**: `device_api.h` 未修改守门

```bash
git diff origin/main..feat/device-api-delegation -- include/ptxemu/device_api.h
# 必须为空(签名冻结,PTXEMU_API_VERSION=1 保持)
```

- [ ] **Step 2**: drift_check 5 invariants 全 PASS

```bash
# GitHub Actions UI 或本地: act -j drift-check
# Invariant 1: PTXEMU_API_VERSION == 1
# Invariant 2: IPtxEmuDevice >= 12 pure virtual methods
# Invariant 3: C++17 兼容
# Invariant 4: 4 symbols 存在
# Invariant 5: ptxemu_core STATIC target name
```

- [ ] **Step 3**: ctest PASS

```bash
cd build && ctest --output-on-failure
# 期望 ~253 tests PASS (246 baseline + 6 new unit + 1 e2e)
```

- [ ] **Step 4**: set_active_mask overwrite 语义测试(per Oracle Risk R2)

```bash
# tests/unit/ptxemu/test_device_api_delegation.cpp 必须有 overwrite 测试
# 验证: 假设 active_mask = 0xFF, set_active_mask(0x01) 后 mask 应为 0x01 (overwrite) 而非 0xFF (OR-merge)
```

---

## Phase 3: HSK-9 规划(Q1 Open Question 触发)

### Task 3.1: 监控 CppTLM consumer demand

**触发条件**(任一触发即启动 HSK-9 spec 起草):
- CppTLM 在 issue #22 / 新 issue 提出新虚方法需求(如 set_kernel_args / query_profiler 等)
- CppTLM 提出 C++17→C++20 升级需求(打破 drift_check Invariant 3)
- CppTLM 提出字段变更(如 `WarpStatus` / `DeviceConfig` 添加字段)

**本 task 状态**: ⏸ MONITOR(不在本 plan 内开 change)

### Task 3.2: 推迟 Phase 1.5 namespace 迁移 (per Oracle Hypothesis 3 + MEDIUM conf)

**触发条件**:
- CppTLM 侧出现直接 `#include <ptx_ir/...>` 需求(跨仓 grep 可证)
- PTX-EMU 下个 release 窗口(per task 9.4 `[~] DEFERRED`)

**本 task 状态**: ⏸ DEFERRED(等触发条件)

### Task 3.3: HSK-9 consumer_smoke 归属 (Q1 Open Question)

**当前决策**: 暂未决定(需要用户输入 — 影响 HSK-9 spec 起草归属)

| 选项 | 影响 |
|---|---|
| A) PTX-EMU 侧提供测试夹具 | PTX-EMU `tests/build_cpptlm_consume/` 新建 + consumer_smoke CMake fixture |
| B) CppTLM 侧自建 | CppTLM `tests/consume_ptxemu/` 新建(更接近 CppTLM ctest gate) |
| C) 联合 | 两边都建 — PTX-EMU 提供 fixture + CppTLM 引用 fixture |

**默认推荐**: **B** (CppTLM 侧) — per HSK-8 spec §Decision 2 "consumer_smoke 是 CppTLM 端准入,PTX-EMU 不需要承担测试"

**本 task 状态**: ⏸ AWAIT(需要用户输入)

---

## Open Questions (需要用户输入)

### Q1: HSK-9 consumer_smoke 由谁编写?

- **A**: PTX-EMU 侧提供测试夹具
- **B** (推荐): CppTLM 侧自建
- **C**: 联合

### Q2: CppTLM AGENTS.md HSK-8 条目同步?

- **A** (推荐): 本 cycle 顺手同步(跨仓 docs PR)
- **B**: 留给 CppTLM owner(per 现有 CppTLM AGENTS.md 风格 — 仅 HSK-6 in table)
- **C**: 跳过

---

## Verification Gates Summary

| Phase | Gate | 检查命令 |
|---|---|---|
| Phase 0 | Push 成功 | `git log origin/main --oneline -3` 显示 `530bd6ca` |
| Phase 1 | stale markers 清除 | `grep "🔄\|⏳" AGENTS.md` 应为空(HSK-8 section) |
| Phase 1 | README 同步 | `grep "IPtxEmuDevice" README.md` 显示新条目 |
| Phase 2 | device_api.h 未修改 | `git diff origin/main..feat/device-api-delegation -- include/ptxemu/device_api.h` 为空 |
| Phase 2 | drift_check 全 PASS | 5 invariants PASS |
| Phase 2 | ctest PASS | `ctest` 显示 ~253 tests PASS |
| Phase 2 | overwrite 语义测试 | tests/unit/ptxemu/test_device_api_delegation.cpp 覆盖 |
| Phase 3 | consumer demand 监控 | (持续,无明确 gate) |

---

## 相关链接

- **Oracle consultation**: session `ses_fcbc066a6ffeXfv5kIgmE8UwrB`
- **HSK-8 ack body**: `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` (commit `738b412c`)
- **HSK-8 audit + postmortem**: `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` (commits `3678a0d7` + `8aa72f1d`)
- **HSK-8 archived change**: `openspec/changes/archive/2026-08-24-ptxemu-public-device-api/` (commit `530bd6ca`)
- **PR #14**: HSK-8 Phase 2 squash merge (commit `fcdad151`)
- **CppTLM bump commits**: `6f408b5` / `09c27d5` / `d035551` / `12b9e0f` / `beb3db8`
- **ptx-lessons-learned**: §3 phase commit + §4 baseline worktree + §6 artifacts-first + §7 Metis pre-impl + §8 四件套 + §21 README Checklist I + §H OpenSpec lifecycle
- **test-coverage-enforcer**: 2.2/2.3 验证要求
- **drift_check workflow**: `.github/workflows/drift_check.yml` (5 invariants)

---

**Last updated**: 2026-08-24 (per Oracle consultation `ses_fcbc066a6ffeXfv5kIgmE8UwrB`)
**Owner**: PTX-EMU Architecture Team
**Status**: Phase 0-1 ready to execute (Quick) · Phase 2 pending Metis review (Medium) · Phase 3 awaiting triggers / user input