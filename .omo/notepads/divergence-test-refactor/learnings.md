# Divergence Test Refactor — Learnings

## Conventions & Patterns

- `test_divergence_sync_convergence.cpp` is the **canonical style** in this directory:
  - Uses `ptxsim::testing::make_nop()` / `make_bra_pred()` / `make_bra()` / `make_ret()` (`instruction_helpers.h`)
  - Uses `ptxsim::testing::setup_pred(w, mask)` for per-lane predicates (`predicates.h`)
  - Uses `ptxsim::testing::step_warp(w, v)` to drive execution — **fully simulates** sm_context.cpp scheduler
  - `step_warp` internally loops `check_reconvergence()` until stable
- `makeBarWarpSyncInstr(mask, reconv_pc)` from `ptxir::factory` (ADR-0013)

## Architectural Decisions

- `synchronize_barrier()` (sm_context.cpp:536-637) **does NOT call** `update_active_mask()` — known issue
  - Documented at `src/ptxsim/core/AGENTS.md` KNOWN ISSUES
  - `test_post_barrier_divergence.cpp` is the **only structured doc** of this issue
  - Refactor: keep as known-issue docs, **do NOT flip** assertions to "correct behavior"

## Decisions

- **Redundancy criterion**: same code path coverage + same observable behavior
- **post_barrier**: merge 5→2 (KNOWN-ISSUE + WORKAROUND-VERIFY)
- **strengthen scope**: only rewrite existing assertions, no new TEST_CASE
- **Target**: 6→4 files, 28→~11 TEST_CASE

## Issues & Gotchas

- `ctest -L "integration;divergence"` returns "No tests were found" on this version
- Use `ctest -R "integration_(divergence|nested|shortest|post_barrier)"` instead

## Baseline (Task 1, 2026-06-04)

| File | TEST_CASE | Assertions |
|------|-----------|-----------|
| test_divergence_sync_isolated.cpp | 4 | 48 |
| test_divergence_sync_convergence.cpp | 5 | 118 |
| test_divergence_sync_standalone_integrated.cpp | 6 | 95 |
| test_nested_divergence.cpp | 4 | 95 |
| test_shortest_path_first.cpp | 4 | 12 |
| test_post_barrier_divergence.cpp | 5 | 15 |
| **Total** | **28** | **383** |

**6/6 PASS**

## Task 2 Execution Findings (2026-06-04)

### Files Modified
| File | Action | Lines Changed |
|------|--------|---------------|
| `tests/integration/divergence/test_shortest_path_first.cpp` | DELETED | -N/A (file removed) |
| `tests/integration/CMakeLists.txt` | Removed `add_catch_test` + `set_tests_properties` | -4 lines (97-100) |
| `AGENTS.md` | No change (line 207 had no `shortest_path_first` reference) | 0 |
| `workflow-state.md` | Appended `[DELETED in refactor 2026-06-04]` marker | 0 net, +26 chars |
| `.opencode/skills/ptx-lane-verification/SKILL.md` | Added 5-line blocknote after 参考实现 | +5 lines |

### Unexpected Reference Found
- `openspec/changes/add-sm90-100-bsync-interleave/tasks.md:113` contains:
  `- 创建 tests/test_shortest_path_first.cpp`
- This is a **historical spec change** (already archived/in-progress), NOT in the active build.
- Per MUST NOT DO rules, left untouched. The spec doc is reference-only.

### Verification Results
- `cmake --build build` → `[100%] Built target ...` ✅
- `ctest -R integration_shortest_path_first` → `No tests were found!!!` ✅
- `test -f test_shortest_path_first.cpp` → file does not exist ✅
- `grep -c shortest_path_first CMakeLists.txt` → 0 ✅

### CmakeLists.txt Line Shifts
After removing lines 97-100 (4 lines), all subsequent entries shifted up by 4 lines.
Original line 102 (`integration_post_barrier_divergence`) is now at line 98.
Original line 105 is now at line 101.
No other code references absolute line numbers, so no follow-up changes needed.

### Decision: Skill SKILL.md Documentation
Added inline note in `ptx-lane-verification/SKILL.md` (line 446 area) explaining:
1. File was deleted in 2026-06-04 refactor
2. Reason: weak assertions (4 TEST_CASE, only set ShortestFirst mode)
3. Coverage preserved by `test_divergence_sync_isolated` / `test_nested_divergence`
4. ctest target `integration_shortest_path_first` also removed

## Task 3 Learnings (2026-06-04)

### What was done
- Deleted `test_divergence_sync_isolated.cpp` (-239 lines, file gone)
- Removed 4 lines from `tests/integration/CMakeLists.txt` (target + labels)
- Updated `scripts/sanity.sh:176` regex: `standalone|isolated` → `standalone`
  (also updated description string "(standalone + isolated)" → "(standalone)" for honesty)
- Added 3-line tombstone comment after @file header in standalone file
- Deleted 2 metadata TEST_CASEs from standalone (-43 lines net)
  - "statement sequence structure" (30 lines, 22 type checks)
  - "handler registration" (12 lines, 8 handler null-checks)

### Verification outcome
- Build: `cmake --build build` 100% (no errors/warnings)
- Test: `ctest -R integration_divergence_sync_standalone` PASS (64 assertions in 4 test cases)
- Test count: 6 → 4 (one per remaining execution scenario, all kept)

### Discoveries
- LSP diagnostics noise: `lsp_diagnostics` on the standalone file reports many "undeclared identifier" errors, but these are PRE-EXISTING (CMake-only include path config). Don't be misled by them — `cmake --build` is the authoritative check.
- The `setp+selp` style in iso (vs real `@%p bra` branching) is genuinely redundant with the two kept standalone tests: "barrier releases all threads" and "full warp barrier-then-divergence flow" already cover barrier-then-divergence behavior. So no coverage loss.
- "AGENT MEMO" hook over-triggers on `NOTE:` — tombstones (user-mandated deletion audit comments) are not memos. The plan explicitly mandates this tombstone text verbatim.

### Pattern note for future
When deleting a redundant test file in this codebase:
1. The four touchpoints are always: the .cpp, CMakeLists.txt, sanity.sh regex, downstream tombstone
2. Always check `grep -rn "<filename>" --include="*.sh" --include="CMakeLists.txt" --include="*.md"` first to find all references
3. ctest target naming convention `<type>_<name>` is in tests/integration/CMakeLists.txt; deletion is safe, renaming is not

### Per-task artifact
- Evidence: `.omo/evidence/task-3-iso-deletion.log`

---

## Task 4 Learnings (2026-06-04) — test_post_barrier_divergence.cpp merge 5→2

### Outcome
- TEST_CASE count: 5 → 2 (slim factor 60%)
- File line count: 560 → 317 (delta -243 lines)
- Test result: `All tests passed (8 assertions in 2 test cases)`
- Build: clean (no warnings, no errors)
- Renamed: BUG-REPRODUCTION → KNOWN-ISSUE; FIX VERIFICATION → WORKAROUND-VERIFY
- Deleted (duplicates of the 2 kept TEST_CASEs):
  - "Full cycle: barrier release then post-barrier instruction execution" — duplicate of KNOWN-ISSUE BUG section
  - "BUG-REPRODUCTION-CTA" — duplicate with execute_warp_instruction driver (but the kept KNOWN-ISSUE has both manual wbar simulation and the CTA-level pattern via SECTIONs)
  - "FIX-VERIFICATION-CTA" — exact duplicate of WORKAROUND-VERIFY (same active_mask update, same execute_warp_instruction call)

### Bug Documentation Contract
- The 2 retained TEST_CASEs now form a documented bug/WORKAROUND pair
- File-level Doxygen comment references `src/ptxsim/core/AGENTS.md#48`
- Inline `// KNOWN ISSUE: ...` comment in TEST_CASE #1 prevents future "fix" of flipping the assertion
- The WORKAROUND-VERIFY test serves as both regression check AND a usage example for callers
  (callers must manually `warp.set_active_mask(arrived_mask)` until the bug is fixed)

### Section Structure Preserved
- KNOWN-ISSUE TEST_CASE retained all 3 SECTIONs: Setup, BUG, Verify
- WORKAROUND-VERIFY TEST_CASE retained its 1 SECTION: FIX
- Total SECTIONs: 4 (vs 5 originally — 1 SECTION in deleted "Full cycle" was redundant)

### Constraints Honored
- Did NOT flip any assertion (bug is unfixed)
- Did NOT modify synchronize_barrier() implementation (Scope OUT)
- Did NOT delete SECTIONs inside the 2 retained TEST_CASEs
- Did NOT add new TEST_CASEs
- LSP `include` errors on this file are pre-existing and unrelated (LSP lacks build include paths)

### Next Step Candidates (not in this task scope)
- A future bug fix in synchronize_barrier() will flip KNOWN-ISSUE's assertions (currently REQUIRE(executed_count==1) → REQUIRE(executed_count==32))
- After the fix, KNOWN-ISSUE TEST_CASE can be renamed back to "BUG-VERIFY" or removed
- WORKAROUND-VERIFY can stay as a regression test for the manual API path

## Task 5 — Convergence 合并 (2026-06-04)

### 操作
- Test A + Test C → "scheduler switches at convergence and picks lowest non-blocked PC"
- Test D + Test E → "boundary convergence requires all active_mask lanes"
- Test B 保留不动

### 关键发现

#### 1. Test C 是 Test A 的严格子集
- Test A 已经覆盖了 Test C 的所有断言：
  - `get_lanes_by_pc().size() == 2` (Test A line 121)
  - `step_warp(w, v) == PATH_A_START` (Test A line 124) — 最低 PC 选择
  - 完整 step_warp 序列 (Test A line 124-160)
  - 阻塞/is_blocked 验证 (Test A line 145-146)
  - Path B → BRA_UNI_PC (Test A line 153-160)
  - simt_stack empty (Test A line 165)
- Test C 无任何独占断言
- 合并策略：保留 Test A 完整结构，仅重命名

#### 2. Test E 的 `check_reconvergence() == false` 是关键边界断言
- Test D 缺此断言 → 合并时需补回
- 此断言验证：Path A 到达 PC=14 不触发收敛，因为 active_mask 跟踪的是 taken=0-15 路径
- 是 `Test E` 唯一独占断言，必须保留
- 合并后位置：在 `for (int i = 0; i < 9; i++) step_warp(w, v);` 之后，`step_warp(w, v); // 阻塞` 之前

#### 3. 断言数变化
- 原 5 TEST_CASE / 118 断言（ctest 报告）
- 现 3 TEST_CASE / 100 断言（ctest 报告）
- 减少 18 断言，来源是 Test C（~10 个 step_warp 重复断言）和 Test E（~8 个重复断言）
- grep -c 字符串匹配数从 ~82 → 66，与 ctest 报告差异是嵌套循环内的 CHECK

#### 4. ctest 实测结果
- 100 assertions in 3 test cases
- 100% pass rate
- 0.06 sec

### 风格标杆确认
- 合并后仍完美符合 `ptxsim::testing::step_warp` 风格
- 注释清晰记录合并来源（维护追溯用）
- `build_instrs()` / `setup()` helper 未触碰

---

## Task 6: test_nested_divergence.cpp 清理 (2026-06-04)

### 操作结果
- 226 → 183 lines (-43)
- 4 → 1 TEST_CASE (-3)
- 保留的 TEST_CASE 改名: "test_nested_predication: Full warp execution with nested setp+selp"
- 断言数: 73 assertions in 1 test case (passed)

### 删除的元数据测试
1. "Structure verification" - 验证 build_nested_divergence_statements() 输出语句结构和 instructionText。
   - 删除原因: 与"嵌套分歧"行为验证无关，是构建函数的回归检查。
   - 8 CHECKs 全部删除。
2. "Handler registration" - 验证 InstructionFactory 中 8 个 handler 非空。
   - 删除原因: 测试框架启动已保证 handler 注册；低价值。
3. "Register analysis" - 调用 RegisterAnalyzer::analyze_registers() 并 CHECK size >= 12。
   - 删除原因: 测试的是 RegisterAnalyzer 内部数据，与"嵌套分歧"行为完全无关；属跨界测试。

### 命名修正
- 原名暗示测试嵌套 SIMT 分歧（嵌套 @%p bra + SIMT stack 推送）
- 实际: 两级 setp/selp 谓词化，无分支、无 SIMT stack
- 新名 "nested_predication" 准确描述测试内容

### 关键发现
- 真实嵌套分歧测试需要使用 @%p bra + label 跳转 + SIMT stack 推送
- 该文件使用 setp+selp 是另一种实现"分支"的方式（PTX 编译器常用），是谓词化执行模型
- TODO 注释指向未来工作：添加真正的两级 @%p bra 分歧测试
- ctest 标签 `[nested_divergence]` 保留以避免破坏测试发现机制

### Verification
```bash
grep -c "TEST_CASE" → 1 ✅
grep "nested_predication" → 匹配 ✅
cmake --build → [100%] Built target integration_nested_divergence ✅
ctest -R integration_nested_divergence -V → All tests passed (73 assertions in 1 test case) ✅
```

### 任务 6 总结
- divergence/ 目录清理工作完成（6 个 task 全部通过）
- 该文件保留为谓词化覆盖，但通过命名/注释明确标记非真实 SIMT 分歧

---

## Task 7: "all three modes" 重写为 step_warp 风格 (2026-06-04)

### 操作
- `tests/integration/divergence/test_divergence_sync_standalone_integrated.cpp` 中唯一修改的 TEST_CASE:
  "divergence_sync_standalone: all three modes produce same reconvergence"
- 新增 includes: `ptxsim/testing/scheduler_utils.h`, `instruction_helpers.h`, `predicates.h`, `<array>`
- 新增 using: `ptxsim::testing::step_warp`, `setup_pred`

### 风格转换细节
| 维度 | 之前 | 之后 |
|------|------|------|
| 指令构造 | 21 条手工 makeGenericInstr/BranchInstr/LabelInstr | 35 条: make_nop + make_bra_pred + make_bra |
| 执行驱动 | warp->execute_warp_instruction(s, pc) 直接驱动 | step_warp(w, stmts) 调度器驱动 |
| Predicate | 直接写 r1 寄存器 + setp.eq 计算 p_t0 | setup_pred(w, 0x0000FFFFu) 注入 p1 |
| 指令布局 | divergence_sync_standalone kernel 21-语句 | 对齐 convergence test 35-语句 (4 NOP + 谓词bra + 9 PathA NOP + 1 conv NOP + 6 PathB NOP + 1 bra.uni) |
| 模式断言 | 无 | `REQUIRE(sm.get_divergence_execution_mode() == mode)` |

### 断言数变化
- 之前: 6 CHECK (3 模式 × 2 断言)
- 之后: 70 CHECK + 3 REQUIRE (大幅强化覆盖分歧-汇聚全流程)
- ctest 报告: 244 assertions in 4 test cases (PASS in 0.05 sec)

### 关键发现
- step_warp 自动处理分歧后调度: 选最低 PC，处理阻塞/解阻
- check_reconvergence 在 step_warp 内循环调用，汇聚后自动 unblock 早到 lane
- 对齐 convergence test 指令布局后断言变得很自然: 35 NOP + 1 谓词bra + 1 bra.uni
- 三个 mode (Sequential/Interleaved/ShortestFirst) 都到达相同 final state:
  `simt_stack.empty()` + `exec_mask == 0xFFFFFFFFu` + 所有 32 lane PC=14

### 预存在失败 (与本次重构无关)
- `test_divergence_sync_standalone` (#25) — E2E CUDA bench
- `integration_barrier_divergence_scheduling` (#71) — barrier 调度
- 验证: `git stash` → 重跑同样失败 → `git stash pop`

### 约束遵守
- ✓ 不新增 TEST_CASE
- ✓ 不改变测试宏观意图 (仍测试 "all 3 modes → same reconvergence")
- ✓ 不触碰其他 TEST_CASE
- ✓ 不修改其他 divergence 文件

### Evidence
- `.omo/evidence/task-7-standalone-rewrite.log`

---

## Task 8: External Documentation Update (2026-06-04)

### Files Modified
| File | Lines Changed | Action |
|------|---------------|--------|
| `AGENTS.md` (root) | 1 | Replaced `integration_divergence_sync_isolated` (deleted) with full post-refactor target list |
| `src/ptxsim/core/AGENTS.md` | 2 | Replaced 2 bullet items under "KNOWN ISSUES" with the 2-line `// KNOWN ISSUE` comment block |
| `docs/testing/TEST_DOCUMENTATION.md` | 1 | Extended `test_post_barrier_divergence.cpp` description to note 5→2 merge |
| `docs/adr/0013-statement-factory-test-unification.md` | 8 | 3 rows updated + 1 row added in tables + 1 update record row |

Total lines: 12 lines changed across 4 files.

### Stale Reference in Skill File (LOGGED, not modified)
- File: `.opencode/skills/ptx-lane-verification/SKILL.md`
- Constraint: "禁止修改 .opencode/skills/ 目录下的技能文件"
- Stale references found (require future update):
    * Line 446: `tests/test_divergence_sync_isolated.cpp` in 参考实现 list
    * Line 452: same file referenced in the `test_shortest_path_first` deletion note
- The Task 2/3 5-line blocknote about `test_shortest_path_first.cpp` is in place (verified).

### Verification Results
1. `grep -rn "integration_divergence_sync_isolated" AGENTS.md src/ptxsim/core/AGENTS.md docs/` → no output (exit 1) ✅
2. `grep -rn "integration_shortest_path_first" AGENTS.md src/ptxsim/core/AGENTS.md` → no output (exit 1) ✅
3. `grep "divergence_sync" scripts/sanity.sh` → only `test_divergence_sync_standalone` ✅
4. `ctest -R "integration_(divergence|nested|shortest|post_barrier)"` → 4/4 PASS ✅
    - integration_divergence_sync_convergence: PASS
    - integration_divergence_sync_standalone: PASS (244 assertions in 4 test cases)
    - integration_nested_divergence: PASS (73 assertions in 1 test case)
    - integration_post_barrier_divergence: PASS (8 assertions in 2 test cases)

### Constraint Compliance
- ✓ Did NOT modify `docs/superpowers/plans/*` (historical plan files)
- ✓ Did NOT modify `openspec/changes/*/tasks.md` (historical task records)
- ✓ Did NOT modify `src/ptxsim/**` (source code)
- ✓ Did NOT modify `.opencode/skills/*` (per constraint)
- ✓ Logged 2 stale references in skill file for follow-up

### Pattern Note
The "无历史计划文件修改" + "无源码修改" + "无技能文件修改" 三约束对外部文档更新来说很关键:
文档更新的"爆炸半径"严格限定在 Markdown 描述层和 ADR 元数据层。当上游存在跨文件级联引用时,
采用"记录但不改"的策略,把残留工作打包到下个有权限的 task。

### Evidence
- `.omo/evidence/task-8-docs-update.log`

