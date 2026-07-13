# Postmortem: fix-tcgen05-idesc-parsing (Oracle C1 Fix)

> **Date**: 2026-07-13
> **OpenSpec Change**: `fix-tcgen05-idesc-parsing` (archived as `2026-07-12-fix-tcgen05-idesc-parsing`)
> **Oracle Session**: `ses_0a8af7ff0ffeYHjA65F4uPwcKa` (BLOCKER review)
> **Author**: Sisyphus (opencode)
> **Cross-ref**: [ADR-0016 §2026-07-13 Postmortem C1 fix](../adr/0016-blackwell-only-tcgen05.md), [lessons-learned §6 §7 §10](../lessons-learned.md)

## Summary

Oracle 2026-07-11 audit identified BLOCKER **C1** (handler accumulate routing):
`processTcgen05Mma` hardcoded `accumulate=false`, ignoring `idesc` register
operand[3] from PTX ISA §9.7.16. Even with helper signature extension
(`int warp_id, bool accumulate`), handler path never triggered accumulate.
This change delivers the fix: handler reads idesc register at runtime,
extracts accumulate bit, passes to helper.

**Outcome**: 4 commits (artifacts-first → impl → ADR → archive),
5 files (162 LoC), 3 new integration tests (T4/T5/T6), 1 PTX fixture.
**Test results**: 24/24 ctest tcgen05 PASS, 46/46 test_all_ptx.sh PASS, 0 regressions.

---

## Lessons Learned (5 new patterns)

### 1. HARD GATE pattern for Oracle BLOCKER resolution

**Phenomenon**: Oracle review identified a BLOCKER whose root cause was
a fictional API name (`ThreadContext::read_reg_32`) — proposed in proposal
but never verified to exist. Without verification, the implementation
phase would have failed at compile time.

**Lesson**: For Oracle BLOCKER items, define an explicit **HARD GATE**
task in the proposal's tasks.md that:
1. Greps the codebase for the proposed API
2. If missing, defines the minimum accessor needed
3. Adds unit test for the accessor (before any handler uses it)
4. Promotes verification from "check before apply" to **must-resolve
   before any subsequent Phase** (not just verification step)

**Diagnostic command**:
```bash
# Don't trust proposed API names. Verify with grep:
grep -rn "read_reg_32\|read_reg_u32\|get_reg_value" include/ptxsim/thread_context.h
# Actual API: reg_access_ (RegisterAccessLayer unique_ptr, thread_context.h:45)
# → must define new accessor
```

**Real case**: `fix-tcgen05-idesc-parsing/tasks.md §0.5` (Oracle BLOCKER).
Without HARD GATE, the handler code `context->read_reg_32(idesc_reg)` would
have failed at compile (`'class ThreadContext' has no member named 'read_reg_32'`),
wasting apply phase time on backtracking.

### 2. Defensive null-check + lazy-init pattern for optional infrastructure

**Phenomenon**: After HARD GATE added `read_reg_32`, tests segfaulted
because `reg_access_` (the RegisterAccessLayer) was null for minimally-
constructed `ThreadContext` (no `init()` called). The defensive null-check
on `get_register_bank_manager()` (which uses `reg_access_`) ALSO segfaulted
because the `get_register_bank_manager()` itself dereferences null.

**Lesson**: When adding optional infrastructure accessors:
1. **First-level defense**: accessor checks infrastructure existence
   (e.g., `if (!reg_access_ || ...) return default;`)
2. **Second-level defense**: setter lazy-inits infrastructure if needed
   (e.g., `set_register_bank_manager` creates `reg_access_` on first call)
3. **Critical**: never chain calls that each assume the prior exists —
   check the deepest layer first

**Diagnostic command**:
```bash
# Verify infrastructure lifecycle:
grep -n "reg_access_\s*[=;{]\|reg_access_\s*$" include/ptxsim/thread_context.h
# Default-constructed ThreadContext leaves reg_access_ null
# init() (line 74) creates it; set_register_bank_manager had pre-existing deref bug
```

**Real case**: `include/ptxsim/thread_context.h:113-116` (defensive `read_reg_32`),
`include/ptxsim/thread_context.h:258-265` (lazy-init `set_register_bank_manager`).
Two-line fixes prevented 4 test failures (T1/T2/T3/T1_k_loop_4/HARDENING all
segfaulted without defensive checks).

### 3. Alignment UB in register bank tests (uint8_t* → uint32_t*)

**Phenomenon**: Test code `*static_cast<uint32_t*>(register_bank->get_register("r5", 0, 0)) = 0x1u;`
segfaulted. The register storage is `std::vector<uint8_t>` (alignment 1)
per `RegisterBankManager::create_register` (`register_bank_manager.cpp:27`).
Direct cast to `uint32_t*` violates strict aliasing + alignment.

**Lesson**: Never cast register-bank `void*`/`uint8_t*` to typed pointer
for read/write. Use `std::memcpy` to safely transfer values across types.
This is the same pattern used in `thread_context.cpp:447-457` for production
memory access (`void *regAddr = acquire_register(...); reg_value = *(uint32_t*)regAddr;`
is technically UB; safer: `std::memcpy(&reg_value, regAddr, sizeof(reg_value))`).

**Diagnostic command**:
```bash
# Find potential alignment UB sites:
grep -n "static_cast<.*\*>(.*get_register" tests/ src/
# Check storage container type:
grep -A2 "create_register" src/register/register_bank_manager.cpp | grep "vector<"
# → std::vector<uint8_t> → 1-byte alignment → cast to uint32_t* is UB
```

**Real case**: `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:T4/T5/T6`
(3 tests). Replacing 6 cast assignments with `std::memcpy` fixed the segfault
without changing test semantics.

### 4. Oracle 4-way split validated empirically

**Phenomenon**: Initial proposal might have been tempted to fix all 4
BLOCKERs (C1 handler accumulate, C2 ld/st slot, C3 commit/wait group,
C4 multi-warp fragment) in one change. Oracle Q1 explicitly validated
keeping them as 4 separate changes.

**Lesson**: Oracle's BLOCKER separation rationale:
1. **Different blast radius** (per change file count)
2. **Different abstraction layers** (grammar vs IR vs visitor vs handler)
3. **Independent rollback** (any change can revert without breaking others)
4. **Avoid single large commit** that violates lessons-learned §3
   ("复杂迁移必须分 Phase commit, 每个 Phase 独立可回退")

**Diagnostic command**:
```bash
# Verify scope split is sound:
openspec list --json | jq '.[] | {name, artifacts_count: (.artifacts | length)}'
# Each follow-up should have ≤ 5 file changes for safe rollback
```

**Real case**: This change = 5 files (HARD GATE + handler + 3 tests).
Oracle Q1 stated FU-2 (C1) ≤ FU-3 (C2) ≤ FU-1 (C3) < FU-4 (C4) in complexity.
Keeping them separate respects lessons-learned §3 + enables independent
scheduling per dependencies.

### 5. Pre-impl Oracle review caught "已实施但未清理" anti-pattern

**Phenomenon**: Original proposal claimed `helper signature extension`
and `c_slot warp_id offset` as work to be done in this change. Oracle
empirical verification (file:line grep) revealed they were ALREADY
implemented by predecessor commit `e37c6de` (Oracle C4 fix). Without
Oracle review, the implementation would have done ghost work.

**Lesson**: Before writing "what changes" section in any OpenSpec proposal:
1. Grep all claimed files for the claimed new code
2. If claim is "add field X to struct Y", verify X doesn't exist
3. If claim is "modify helper signature", verify current signature
4. Per Oracle Q1 hard rule: NEVER assume state — always verify with grep

**Diagnostic command**:
```bash
# Standard pre-impl verification per change:
git log --oneline -- <claimed-affected-file> | head -5
grep -n "<claimed-new-symbol>" <claimed-affected-file>
# If grep finds symbol: not work to be done; remove from "what changes"
```

**Real case**: This proposal's "what changes" table originally listed
2 rows for helper signature + c_slot. Oracle caught both as ghost work.
Removed via 8 edits across proposal.md/design.md/tasks.md.

---

## Process Improvements (Meta-Observations)

### What worked

1. **Worktree discipline**: Implementation worktree isolated from main.
   Caught 1 misplaced edit (committed to main, reverted via
   `git checkout`) — would have polluted main branch without isolation.
2. **Baseline worktree (§4)**: Provided `b005665` baseline for regression
   comparison. After implementation: same tests pass on baseline = no
   regression introduced. Removed after archive.
3. **Phase commits (§3)**: Each Phase = independent commit. If Phase 1
   had broken existing tests, could `git revert` cleanly without affecting
   ADR or archive.
4. **Oracle 2-pass review**: First pass (C1 + scope) → fixes; Second
   pass (per-file:line verification) → ghost work removal. The two passes
   caught complementary issues.

### What didn't work (process improvements)

1. **Initial read_reg_32 edit went to main worktree, not implementation
   worktree**. Discovered at compile time. Should have started in the
   implementation worktree from the beginning.
   **Fix**: Always verify worktree branch with `git status --short` before
   editing header files.
2. **PTX fixture multi-line `.param` syntax triggered ANTLR parse error**.
   Existing fixtures use single-line. Should have copied working fixture
   format verbatim before adding comments.
   **Fix**: Per lessons-learned §L: "声称 'X/X PASS' 必须用真实 kernel PTX
   验证". For new fixtures, copy-then-modify is safer than modify-from-scratch.
3. **Test code assumed non-existent APIs (register_bank_["%r5"])**.
   Oracle Q7 BLOCKER caught this in proposal phase. Should have run
   `git grep register_bank` BEFORE writing test pseudo-code.
   **Fix**: Convert pseudo-code to spec-style descriptions in tasks.md
   (test inputs/outputs/invariants), then implement against real APIs
   in apply phase.

---

## Action Items (lessons-learned.md updates)

These new patterns should be added to `.opencode/skills/ptx-lessons-learned/SKILL.md`:

### Checklist M: HARD GATE for BLOCKER API additions

```
□ For Oracle BLOCKER items, add explicit §HARD GATE task in tasks.md
□ HARD GATE includes: grep proposed API name, define minimum accessor,
  add unit test, promote from verification to must-resolve gate
□ HARD GATE must pass BEFORE subsequent Phase implementation
```

### Checklist N: Defensive infrastructure lifecycle

```
□ For new optional infrastructure accessors (e.g., ThreadContext methods
  depending on reg_access_/RegisterAccessLayer):
  - First-level defense: accessor checks infrastructure existence
  - Second-level defense: setter lazy-inits if needed
  - Never chain calls that each assume the prior exists
```

### Checklist O: Register bank memcpy pattern

```
□ Never cast register-bank void*/uint8_t* to typed pointer
□ Use std::memcpy to safely transfer values across types
□ Storage is std::vector<uint8_t> (alignment 1) — typed pointers UB
```

### Failed Mode Lookup: Ghost Work Anti-pattern

```
Symptom: "what changes" section claims work that's already in code
Diagnosis: git log -- <file> + grep -n "<claimed-symbol>" <file>
Prevention: Per Oracle Q1 hard rule "never assume state, always verify"
Apply: Remove ghost work from proposal/design/tasks before apply
```

---

## Cross-References

- **Active predecessor**: [`fix-tcgen05-mma-accumulator-and-f32-storage`](../changes/archive/2026-07-12-fix-tcgen05-mma-accumulator-and-f32-storage/) (H1+H2 helper fix)
- **Parallel follow-up**: [`fix-tcgen05-commit-wait-group`](../changes/archive/2026-07-12-fix-tcgen05-commit-wait-group/) (FU-1/C3, already archived)
- **Future follow-ups**:
  - `fix-tcgen05-idesc-full-parsing` (full idesc descriptor parsing)
  - `fix-tcgen05-ld-st-slot-routing` (FU-3/C2)
  - `fix-tcgen05-multi-warp-fragment` (FU-4/C4, partially done as `e37c6de`)
  - `tcgen05-flashattention-coverage` (FU-5, E2E)
- **Skills applied**: `ptx-lessons-learned` §3/§4/§6/§7/§10, `openspec-propose`,
  `openspec-apply-change`, `oracle-prompting`
- **Related commits** (this branch `fix/tcgen05-idesc-parsing`):
  - `b005665` docs(openspec) — artifacts-first
  - `477840b` fix(tcgen05) — Phase 1 implementation
  - `439827b` docs(adr) — Phase 2 postmortem
  - `5d6de89` chore(openspec) — Phase 3 archive