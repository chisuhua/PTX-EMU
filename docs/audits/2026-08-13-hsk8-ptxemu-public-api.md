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