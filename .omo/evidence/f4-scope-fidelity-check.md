# F4. Scope Fidelity Check Report

## Executive Summary

**VERDICT: REJECT** — 3 critical violations detected

---

## Violations Detected

### 1. Tool Expansion Violation (CRITICAL)

**Constraint**: "不添加新的 make_* helper 函数（工具缺口标记为前置工作）"

**Evidence**:
- Commit `6a51568` (2026-06-02 18:23:38) added 3 new make_* helper functions to `include/ptxsim/testing/instruction_helpers.h`
- This violates the "Must NOT Have" constraint explicitly stated in the plan

**Impact**: Tool expansion during implementation period, contrary to plan's explicit prohibition

---

### 2. SIMT Stack Operation Violation (CRITICAL)

**Constraint**: "integration/ 下不允许任何直接 execute_warp_instruction" + Principle 5: "分歧由 handle_branch 自动处理"

**Evidence**:
- `tests/integration/simt/test_simt_stack_entry_integrated.cpp` contains 2 manual SIMT stack push operations:
  - Line 81: `warp->get_simt_stack().push(entry);`
  - Line 109: `warp->get_simt_stack().push(entry);`
- These violate Principle 5 and should be migrated to `tests/unit/simt/` per AGENTS.md:259 classification rule

**Impact**: Integration tests still contain manual SIMT stack manipulation, violating core architectural principle

---

### 3. Batch Commit Violation (MODERATE)

**Constraint**: "不批量修改多个文件在一个 commit"

**Evidence**:
- Commit `ab55e06` modified 45 test files in a single commit (directory reorganization)
- Commit `2ecd82f` modified 2 test files in a single commit (step_warp signature fix)

**Mitigation**: `ab55e06` is a foundational infrastructure commit (pre-implementation), may be exempt. `2ecd82f` is a fix commit, acceptable.

---

## Compliance Verification

### ✅ Must Have Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| AGENTS.md line 259 contains classification rule | ✅ PASS | Verified: "如测试需要手动设置 SIMT stack/PC 状态..." |
| AGENTS.md line 312 contains integration/unit distinction | ✅ PASS | Verified: "如必须手动 push，则该测试属于 unit/ 而非 integration/" |
| sanity.sh regex updated | ✅ PASS | Commit `1cce0ec` (2026-06-05 00:28) |
| tests/integration/CMakeLists.txt updated | ✅ PASS | Multiple migration commits updated CMakeLists |
| tests/unit/CMakeLists.txt updated | ✅ PASS | Multiple migration commits updated CMakeLists |
| 2 migrated files in unit/ | ✅ PASS | `test_barrier_scenarios_integrated.cpp`, `test_divergence_sync_standalone_integrated.cpp` exist in unit/ |

### ❌ Must NOT Have Violations

| Constraint | Status | Evidence |
|------------|--------|----------|
| NO tool expansion | ❌ FAIL | Commit `6a51568` added 3 new make_* functions |
| NO batch commits | ⚠️ PARTIAL | `ab55e06` (45 files) - foundational, may be exempt; `2ecd82f` (2 files) - fix, acceptable |
| NO scope creep | ✅ PASS | Only tests/, AGENTS.md, sanity.sh modified (no library code changes) |
| NO integration/ direct execute_warp_instruction | ⚠️ PARTIAL | No actual method calls found, but 2 manual SIMT stack pushes remain |

---

## Task Completion Analysis

**Total commits in implementation period (2026-06-01 to 2026-06-06)**: 55

**Implementation-related commits**: 42 (refactor/fix/docs/build)

**Breakdown**:
- 15 step_warp refactor commits (W2-W4 tasks)
- 10 migration commits (W1 tasks)
- 4 documentation commits (W0 + FINAL tasks)
- 13 other commits (fixes, infrastructure, etc.)

**Estimated task coverage**: ~35/35 tasks have corresponding commits (based on commit message patterns)

---

## Cross-Task Contamination Check

**Status**: ✅ CLEAN

**Evidence**: 
- `git diff HEAD~25..HEAD --stat -- ':(exclude)tests/' ':(exclude).omo' ':(exclude)AGENTS.md' ':(exclude)scripts/sanity.sh'` returned no output
- Only allowed files were modified: tests/, AGENTS.md, sanity.sh, .omo/

---

## Unaccounted Changes Check

**Status**: ✅ CLEAN

**Evidence**: All modified files are accounted for in plan scope:
- 15 integration test files refactored (W2-W4)
- Multiple files migrated (W1)
- AGENTS.md updated (W0)
- sanity.sh updated (W1)
- CMakeLists.txt updated (W1)

---

## Recommendations

1. **CRITICAL**: Migrate `tests/integration/simt/test_simt_stack_entry_integrated.cpp` to `tests/unit/simt/` (contains manual SIMT stack push, violates Principle 5)

2. **CRITICAL**: Review commit `6a51568` - if tool expansion was intentional, update plan to reflect this decision; if unintentional, consider rollback

3. **MODERATE**: Document why `ab55e06` (45 files) was acceptable as a batch commit (foundational infrastructure)

---

## Final Verdict

**Tasks [32/35 compliant]** — 3 tasks incomplete or violated constraints
**Contamination [CLEAN]** — No cross-task contamination detected
**Unaccounted [CLEAN]** — All changes accounted for in plan scope
**Must NOT do [3 violations]** — Tool expansion, SIMT stack operation, batch commit

**VERDICT: REJECT**

**Reason**: Critical violations of "Must NOT Have" constraints (tool expansion + SIMT stack operations in integration/) prevent approval. Recommend fixing violations before final approval.
