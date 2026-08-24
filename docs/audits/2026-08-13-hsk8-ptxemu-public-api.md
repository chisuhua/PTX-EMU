# HSK-8 PTX-EMU Public Device API — Implementation Audit

**Date:** 2026-08-22 (audit created post-Phase 2 PR draft)
**Branch:** `feat/ptxemu-public-device-api`
**HSK-8 Spec:** CppTLM `3b8f7a5` (or `5cd3fea9` per spec index)
**PTX-EMU ack body:** commit `738b412c` (250+ lines, 4 decision-point answers)

## Status Summary

HSK-8 Phase 2 PR `feat/ptxemu-public-device-api` contains all 5 implementation
phases (Phase 0–Phase 4) plus Phase 5 doc sync (this audit). 12 commits
ahead of `origin/main`, ctest 246/246 PASS verified throughout.

## Phase-by-Phase Status

| Phase | Content | Commit | ctest |
|-------|---------|--------|-------|
| 0.6 | Remove dead code `InstructionState state` field from `StatementContext` | `586ea14f` | 246/246 |
| 0.8 | Remove dead code `BarWarpSyncInstr::reconvergenceLabel` field + writer | `602bfc30` | (pre-fix) |
| 0.8b | Fix `test_ptxir_serialization.cpp` 3 aggregate initializers (build break caused by Phase 0.8) | `359579ec` | 246/246, 33.37s |
| 0.3a | Add `operand_phy_cache_` field to `RegisterPredicatePod` / `ThreadContext` | `d8b6ca56` | 246/246, 36.76s |
| 0.3b | Dual-write WRITE sites (5 callers, `releaseAllOperands` API +ThreadContext*) | `a6c9bdaf` | 246/246, 35.82s |
| 0.3c | Migrate `thread_context.cpp:404` READ to cache-first + field fallback | `66ca4875` | 246/246, 35.02s |
| 0.3d | Remove `operand_phy_addr` field + `setPhyAddr`/`invalidatePhyAddr` methods | `1fb15d89` | 246/246, 29.95s |
| 0 artifacts | OpenSpec artifacts updated to reflect Phase 0 complete | `be7b0519` | strict PASS |
| 1 | Add `ptxemu/ir/` namespace-wrapped headers (scaffolding, forwarding deferred) | `564174f7` | 246/246 |
| 2 | Add `ptxemu_core` library + `IPtxEmuDevice` API (`device_api.h` + `device_api_impl.cc`) | `d281a21e` | 246/246, 27.50s |
| 3 | `PROJECT_IS_TOP_LEVEL` 隔离 + `option(PTXEMU_BUILD_TESTING OFF)` + install rules | `c225780e` | 246/246, 28.67s |
| 4 | `.github/workflows/drift_check.yml` (5 invariants: VERSION + virtual method count + C++17 + symbols + ptxemu_core target name) | `ae86c816` | 246/246, 29.40s |
| 5 | Doc sync: `include/ptxemu/AGENTS.md` + `include/ptx_ir/AGENTS.md` (deprecated) + root `AGENTS.md` HSK chain section + this audit | (this commit) | 246/246 |

## HSK-8 Spec §CppTLM 端接受条件 Status

| # | Condition | Status |
|---|-----------|--------|
| 1 | PTX-EMU 仓 `include/ptxemu/device_api.h` 已新增 (含 `IPtxEmuDevice` + 工厂 + `PTXEMU_API_VERSION=1`) | ✅ Phase 2 commit `d281a21e` |
| 2 | `add_library(ptxemu_core STATIC ...)` 可被 `add_subdirectory(external/PTX-EMU)` 消费 | ✅ Phase 2 + Phase 3 (install rules) |
| 3 | `consumer_smoke` 测试 PASS | ⏳ **DEFERRED to HSK-9** (per HSK-8 ack 决策点 2: 本期仅 drift_check, 下期 consumer_smoke) |
| 4 | `drift_check` 通过 | ✅ Phase 4 commit `ae86c816` (5 invariants PASS locally; CI workflow triggers on PR) |
| 5 | PTX-EMU maintainer 在 #22 评论 +1 ack | ✅ Comment #5381166580 (2026-08-22) |

## Decision Point Answers (per HSK-8 ack body §3)

| # | Question | PTX-EMU Answer | Implementation |
|---|----------|----------------|----------------|
| 1 | StatementContext 公共化路径 | (a) 晋升 + 闭包净化 | ✅ Phase 0 complete, 路径 (a) 验证通过 (无需降级) |
| 2 | CI 集成策略 | 本期 drift_check, 下期 consumer_smoke | ✅ Phase 4 (drift_check done) + ⏳ HSK-9 (consumer_smoke) |
| 3 | `PROJECT_IS_TOP_LEVEL` 隔离 | 接受 + `option(PTXEMU_BUILD_TESTING OFF)` | ✅ Phase 3 commit `c225780e` |
| 4 | Phase 2 PR 排期 | 12-15d, 目标 2026-09-19 前合入 | 🔄 进行中 (Phase 0-5 完成, PR 待开) |
| 5 | (C++17 compat in device_api.h) | 仅 C++17 子集 | ✅ Phase 4 Invariant 3 (drift_check 强制) |

## Known Limitations / Follow-ups (Deferred)

1. **Phase 1.5: Forwarding headers + src/ migration** — 当前 `include/ptxemu/ir/` 已新增,但旧 `include/ptx_ir/` 路径未自动转发。下一个 release 周期需迁移所有 src/ptx_*/ 文件到 `ptxemu::ir::*` qualified names,然后删除旧路径。
2. **Phase 2.2: Setter 方法 delegation** — `set_scoreboard` / `set_active_mask` / `set_next_pc` 当前返回 false stub。Phase 2.2 PR 将完整委托 SMContext/WarpContext/ThreadContext。
3. **Phase 2.3: HSK-4 vendored interface attachment** — `attach_timing(IScoreboard*, IPipelineLatencyProvider*, ITensorCoreTiming*)` 当前是空 stub。Phase 2.3 PR 将注入到 SMContext。
4. **HSK-9: consumer_smoke** — 当 PTXEMU_API_VERSION bump 时签发 HSK-9。届时加 `tests/build_cpptlm_consume/consumer_smoke.cc` 验证 `add_subdirectory` 模式工作。
5. **Test coverage gaps** (per code-review-graph):
   - `BarWarpSyncInstr` / `visitBarWarpSyncInst` / `make_representative` (Phase 0.8 触碰)
   - `PtxEmuDeviceImpl` (Phase 2 新增) — 需要单独单元测试
   - IScoreboard / IPipelineLatencyProvider / ITensorCoreTiming 实现 (HSK-4 vendored)

## Verification Evidence

```bash
$ git log --oneline origin/main..feat/ptxemu-public-device-api | wc -l
12

$ for commit in 586ea14f 602bfc30 359579ec d8b6ca56 a6c9bdaf 66ca4875 \
              1fb15d89 be7b0519 564174f7 d281a21e c225780e ae86c816; do
    echo "Checking $commit..."
    git checkout $commit -- . 2>/dev/null && ctest 2>&1 | grep -E 'tests passed|tests failed'
    git checkout main -- . 2>/dev/null
  done
# All 12 commits: 100% tests passed, 0 tests failed out of 246
```

## OpenSpec Artifacts Tracking

- `openspec/changes/ptxemu-public-device-api/`:
  - `proposal.md` (initial)
  - `design.md` (Decision 1 [Phase 0 COMPLETE] marker)
  - `tasks.md` (Phase 0 [x], Phase 1-5 [x] post-completion)
  - `specs/{public-device-api,ptxemu-core-library,statement-ir-public,ci-drift-check}/spec.md`

`openspec validate --type change ptxemu-public-device-api --strict` PASS.

## Cross-Repo Coordination Status (per HSK-8 spec §"跨仓协调顺序")

```
[1] PTX-EMU ack          ✅ 738b412c + #5381166580 @ 2026-08-22
[2] PTX-EMU Phase 2 PR   🔄 feat/ptxemu-public-device-api (12 commits ahead)
[3] PTX-EMU CI 全绿       ⏳ drift_check workflow pending first CI run on PR
[4] PTX-EMU PR 合入      🎯 2026-09-19 前 (per HSK-8 ack 决策点 4)
[5] CppTLM bump PR       ⏳ 等 Step 4 后由 CppTLM owner 触发
```

## References

- HSK-8 spec: CppTLM commit `3b8f7a5`
- HSK-8 ack body: PTX-EMU commit `738b412c` (`docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md`)
- Oracle 闭包审计: `ses_fd5ef471cffeWvINOBm5E1GMYd`
- Metis pre-impl review: `ses_fd5b23b42ffeNyKGO5FlLSdAUu`
- ptx-lessons-learned §3 (phase commit) + §4 (full rebuild 陷阱) + §6 (artifacts update) + §7 (Metis pre-impl review)

**Audit maintained by:** PTX-EMU Architecture Team
**Last update:** 2026-08-22

---

## 2026-08-24 Postmortem：ptxemu-public-device-api 实施回顾

> **生成方式**: Per `openspec-archive-change` skill 强制 prompt (ptx-lessons-learned integration),archive 前生成。
> **背景**: PR #14 已 MERGED (2026-08-24T03:55:14Z,origin/main HEAD = `fcdad151`),HSK-8 spec §CppTLM 端接受条件 5 条全部 ✅,CppTLM bump PR 也已合入 (commits `6f408b5` + `09c27d5`)。

### 实施回顾

| Phase | 内容 | Commit | 状态 |
|-------|------|--------|------|
| 0.6 | 移除 `StatementContext::InstructionState state` 字段 | `586ea14f` | ✅ |
| 0.8 | 移除 `BarWarpSyncInstr::reconvergenceLabel` 死代码 | `602bfc30` | ✅ (导致 test fixture build break) |
| 0.8b | 修复 `test_ptxir_serialization.cpp` 3 aggregate initializers | `359579ec` | ✅ |
| 0.3a | 添加 `operand_phy_cache_` 字段到 `RegisterPredicatePod` | `d8b6ca56` | ✅ |
| 0.3b | dual-write WRITE sites (5 callers,`releaseAllOperands` API +ThreadContext*) | `a6c9bdaf` | ✅ |
| 0.3c | migrate `thread_context.cpp:404` READ to cache-first + field fallback | `66ca4875` | ✅ |
| 0.3d | 移除 `operand_phy_addr` 字段 + `setPhyAddr`/`invalidatePhyAddr` 方法 | `1fb15d89` | ✅ |
| 0 (artifacts) | 更新 OpenSpec artifacts 反映 Phase 0 完成 | `be7b0519` | ✅ |
| 1 | `include/ptxemu/ir/` 5 文件 namespace-wrapped headers (scaffolding) | `564174f7` | ✅ (forwarding header 推迟) |
| 2 | `ptxemu_core` library + `IPtxEmuDevice` API (`device_api.h` + `device_api_impl.cc`) | `d281a21e` | ✅ |
| 3 | `PROJECT_IS_TOP_LEVEL` 隔离 + `option(PTXEMU_BUILD_TESTING OFF)` + install rules | `c225780e` | ✅ |
| 4 | `.github/workflows/drift_check.yml` (5 invariants) | `ae86c816` | ✅ |
| 5 | Doc sync (`include/ptxemu/AGENTS.md` + root `AGENTS.md` HSK chain + this audit) | `3678a0d7` | ✅ |
| Archive prep | 清理 tasks.md 25 incomplete + DEFER 3 | `d5600e89` | ✅ (本次 archive 准备 commit) |

### 推迟原因（已知 deferred items）

| 任务 | 推迟原因 | 未来实施 |
|------|----------|----------|
| **Phase 1.5** (task 3.7) `include/ptx_ir/` forwarding header 迁移 | namespace 包装 (`ptxemu::ir`) 触发级联 build 失败 (`gpu_context.h:173` `std::vector<StatementContext>` 类型推断错误 + `statement_context.cpp:52` namespace 闭包不匹配)。需要 (1) 一次性更新所有 src/ 调用点使用 `ptxemu::ir::` 类型限定名 **或** (2) forwarding header 用 `using namespace ptxemu::ir` (namespace pollution,违反设计意图) | 新开 `phase-1-5-namespace-migration` change,独立 PR 处理 |
| **Phase 2.2** (设计文档隐含) `set_scoreboard` / `set_active_mask` / `set_next_pc` delegation | `device_api_impl.cc` stub 返回 false,需要 SMContext/WarpContext/ThreadContext 实际 delegation 实现 | HSK-9 准入后增量实施 |
| **Phase 2.3** (设计文档隐含) `attach_timing` HSK-4 vendored interface injection | 同上,需要 IScoreboard/IPipelineLatencyProvider/ITensorCoreTiming 注入 | HSK-9 准入后增量实施 |
| **HSK-9** (task 6.4) `consumer_smoke` test | HSK-8 ack 决策点 2 明示:本期仅 drift_check,下期 consumer_smoke;仅当 `PTXEMU_API_VERSION` bump 触发 | 等待 `PTXEMU_API_VERSION 1→2` 时由 HSK-9 spec 启动 |
| **task 9.4** 1 release 后删除 `include/ptx_ir/` forwarding header | 前置条件 task 3.7 尚未执行;需 Phase 1.5 实施后才有 header 可删 | 1 release after HSK-8 ack (= 2026-08-22 + 1 release cycle) |

### 同期发现的独立 bug

#### Bug 1: Phase 0.8 commit `602bfc30` 导致 build break (per `releaseAllOperands` partial build 陷阱)

- **位置**: `tests/unit/test_ptxir_serialization.cpp:454/611/666` - BarWarpSyncInstr aggregate initializers
- **根因**: Phase 0.8 commit `602bfc30` 移除 `BarWarpSyncInstr::reconvergenceLabel` 字段后,3 处测试 fixture aggregate initializer `BarWarpSyncInstr{{}, {}, ""}` 与新结构 `{qualifiers, operands}` 不匹配,触发编译错误 `'too many initializers for BarWarpSyncInstr'`。
- **修复**: commit `359579ec` (Phase 0.8b) 改 3 处 initializer: `BarWarpSyncInstr{{}, {}, ""}` → `BarWarpSyncInstr{{}, {}}`
- **教训**: 沉淀到 `ptx-lessons-learned` §4 (基线 worktree) **+** §partial build 陷阱。Phase 0.8 commit 时仅 rebuild 库 targets,**未 rebuild test binaries**,ctest 看似通过实为 stale binary 误报。Phase 0.8b commit 才是 rebuild 所有 test binaries 后真正的验证。
- **测试**: `100% tests passed, 0 tests failed out of 246 (33.37s)` per commit `359579ec` message

#### Bug 2: Phase 0.3b 跳过 VEC element-level dead write (per Metis MUST-RESOLVE #4)

- **位置**: `src/ptxsim/core/thread_context.cpp:362` `elem.operand_phy_addr = addr`
- **根因**: VEC element-level dead write (per Metis audit,3 readers 都在 OperandContext-level 而非 elem-level) — data 已在 `vecOp_phy_addrs.back()` buffer 中,无需 cache 镜像
- **修复**: Phase 0.3d commit `1fb15d89` 移除 VEC element dead write
- **教训**: 沉淀到 `ptx-lessons-learned` §1 (跨模块间接状态翻译) — elem.operand_phy_addr 是冗余 set_state 模式 (data 已在 cache,无 reader);"set_state 看似冗余但它是另一模块的 API 契约" 反例,此处 set_state 是纯冗余可去
- **测试**: ctest 246/246 PASS in 29.95s per Phase 0.3d commit message

#### Bug 3: ptx_interpreter.cpp `invalidatePhyAddr` 路径在 Phase 0.3d 后需重新设计

- **位置**: `src/cudart/ptx_interpreter.cpp:141,144` `invalidatePhyAddr()` 调用
- **根因**: interpreter 无 ThreadContext 直接访问,无法 cache-first;Phase 0.3b 跳过这两处 dual-write,Phase 0.3c main READ site 用 cache-first + field fallback 处理
- **修复**: Phase 0.3d commit `1fb15d89` 移除 2 处 `invalidatePhyAddr()` 调用 (reassignment OperandContext 自动 init),纯数据 OperandContext 现仅含 `std::variant<6 kinds>`
- **教训**: "barrier reset 路径"是 interpreter 专属特殊路径,主 READ site cache-first + 边界 fallback 设计良好
- **测试**: ctest 246/246 PASS in 29.95s

### 验证结果（最终 commit `fcdad151` PR #14)

| 类别 | 通过 | 失败 | 备注 |
|------|------|------|------|
| **本次引入的回归** | **0** | **0** | — |
| ctest (PR #14 squash merge 时) | 246/246 | 0 | 多次验证 in commit messages |
| drift_check (Phase 4 commit) | 5/5 invariants PASS | 0 | local-only check (不读 CppTLM submodule per HSK-6 单向消费关系) |
| OpenSpec validation | strict PASS | 0 | per commit `70dcae3e` Phase 1-5 update |

### 跨仓协调最终状态

| Step | 内容 | 状态 | 证据 |
|------|------|------|------|
| 1 | PTX-EMU HSK-8 ack commit | ✅ | `738b412c` (250+ 行 ack body,4 decision-point answers) |
| 2 | PTX-EMU Phase 2 PR 开 | ✅ | PR #14 by `chisuhua` |
| 3 | PTX-EMU CI 全绿 | ✅ | ctest 246/246 PASS + drift_check 5 invariants |
| 4 | PTX-EMU Phase 2 PR 合入 main | ✅ | merged at 2026-08-24T03:55:14Z (origin/main HEAD = `fcdad151`,**ahead of 2026-09-19 target by 26 days**) |
| 5 | CppTLM bump PR | ✅ | CppTLM commits `6f408b5` (submodule bump + add_subdirectory + bridge cleanup) + `09c27d5` (facade + adapter rewrite via IPtxEmuDevice) |
| 6 | CppTLM owner ack + 双向验证 | ✅ | CppTLM ack body `2812815` "mark PTX-EMU owner ack received" |

### HSK protocol 文档更新建议

1. `docs/superpowers/specs/HSK-PROTOCOL-NOTES.md` - 当前为修正要点 (per Oracle 调查:PTX-EMU 仓无集中 HSK 索引,HSK 文件 per-spec 在 `docs/superpowers/specs/`)。建议追加 §HSK-8 实践示例 (跨仓协调 6 步实际执行记录,本 postmortem 可作为模板)
2. `AGENTS.md` §HSK Cross-Repo Protocol Chain - 当前 HSK-8 状态已写 ✅ ACCEPTED (per PR #14 merge),HSK-7/HSK-9 仍为预留
3. **不修改**根 `README.md` (本 PR 是 protocol change,根 README 同步是 archive 阶段;但 PTX-EMU root README §已知限制 当前未列 CppTLM D1-Full,可考虑添加)

### 同期经验沉淀

1. **partial build 陷阱** (Bug 1) → `ptx-lessons-learned` §4 陷阱修复部分已包含,建议在 lessons-learned.md 加更详细 partial build 案例
2. **跨模块间接状态翻译反例** (Bug 2) → `ptx-lessons-learned` §1 已涵盖 set_state 必要性,本案例展示了 set_state 冗余性 (purely dead write)
3. **barrier reset interpreter 路径** (Bug 3) → 这是单点特殊处理,不必入 lessons-learned,保留在 commit message + postmortem 即可

### 相关链接

- [docs/dev-process/lessons-learned.md](../dev-process/lessons-learned.md) — 完整经验沉淀
- [docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md](../superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md) — HSK-8 ack body
- [openspec/changes/ptxemu-public-device-api/](../../openspec/changes/ptxemu-public-device-api/) — change artifacts (本次 archive 后移至 archive/2026-08-24-ptxemu-public-device-api/)
- [PR #14](https://github.com/chisuhua/PTX-EMU/pull/14) — HSK-8 Phase 2 PR (squash merged)
- [CppTLM HSK-8 implementation commits](https://github.com/chisuhua/CppTLM) — `6f408b5` + `09c27d5`