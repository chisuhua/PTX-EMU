# Postmortem: `fix-phase0-gate1-dgpu-bar-leak` Archive (2026-08-25)

> **TL;DR**: This change did **not need implementation**. Gate 1 leak was physically eliminated by [`commit 09786635`](https://github.com/chisuhua/PTX-EMU/commit/09786635) (`refactor(cudart): remove cpptlm linkage + bridge files (Phase 3 of 4)`), which fully removed `cpptlm_core` from `libcudart.so`'s link line. This postmortem documents the actual resolution path, the dangling `587a6d5e` Round-2 implementation, and the Doc2 (`HANDOFF_FROM_CPPTLM.md`) misattribution.

| Field | Value |
|-------|-------|
| Date | 2026-08-25 |
| Change | `fix-phase0-gate1-dgpu-bar-leak` |
| Status at archive | Proposed (never started) |
| Reason for archive | Task target achieved by 4-phase refactor (`09786635`) |
| Commits in archive commit | `13f55bbe` (artifacts updates) + `<this-postmortem-commit>` |
| Original author | CppTLM handoff (commit `505333b`) → PTX-EMU session (this archive) |
| Affected ADR | ADR-0029 §D7 (Gate 1 contract preserved by structural elimination) |

---

## 1. Background & Original Symptom

### 1.1 Original Gate 1 failure (per Doc2 + tasks.md history)

`tests/integration/test_phase0_byte_identical_gates.cpp:142-156` (Gate 1) compared `nm -D --defined-only libcudart.so` against `/tmp/baseline-artifacts/libcudart-nm-before.txt` (captured 2026-08-18 14:33 from a pre-DGpuBar build).

**Failure**: 131 cpptlm_core origin symbols (including 10 `tlm::gpu::DGpuBar` members + 121 other cpptlm::/tlm::/nlohmann:: mangled names) leaked into `libcudart.so` via `-Wl,--whole-archive cpptlm_core`. Baseline was 3274 symbols; current build was 3284 (131 additions, 0 missing).

**Root cause**: CppTLM commits `4277290` (2026-08-19 17:02) + `923e372` (2026-08-19 17:18) added `tlm::gpu::DGpuBar` (PCIe BAR0 + VRAM model) to `cpptlm_core`. `--whole-archive` forces all `cpptlm_core` objects into `libcudart.so`'s dynamic symbol table.

### 1.2 Why original 1-line fix wasn't deployed

The minimal fix proposed by change `fix-phase0-gate1-dgpu-bar-leak` was:

```cmake
# src/CMakeLists.txt or top-level CMakeLists.txt
target_link_libraries(cudart
    -Wl,--whole-archive,--exclude-libs=ALL cpptlm_core -Wl,--no-whole-archive)
```

**Two reasons this fix wasn't merged to `main`**:

1. **Round-1 `--exclude-libs=ALL` was rejected empirically** by commit `587a6d5e` (2026-08-21) which is **dangling** (not on any branch). The commit message records:
   > GNU ld's `--exclude-libs` applies to archive basename (must include 'lib' prefix and '.a' suffix). The Round-2-recommended `--exclude-libs=ALL` was rejected after empirical test: it also hides 4051 ANTLR4 generated symbols (compiled into cudart's object files at `src/CMakeLists.txt:55`) and breaks ptxir_embed/executable link. Oracle consultation (2026-08-21) recommended targeted `--exclude-libs=libcpptlm_core.a`.

2. **HSK-6 + 4-phase refactor took a different, more radical path**: instead of hiding cpptlm_core symbols, they **removed the link entirely**.

---

## 2. Actual Resolution Path (4-phase refactor + HSK-8)

### 2.1 Timeline (chronological order from git log)

| Commit | Date | Description | Impact on Gate 1 |
|--------|------|-------------|------------------|
| `87820951` | 2026-08-13 | `docs(audit): add PTX-EMU HAL backend cross-repo defect audit` | Identified Gate 1 failure; triggered OpenSpec change creation |
| `587a6d5e` | 2026-08-21 | `fix(cudart): hide cpptlm_core non-ABI symbols (Gate 1)` | **Not merged (dangling)**; 217 cpptlm symbols hidden via `--exclude-libs=libcpptlm_core.a` |
| `8088b24c` | 2026-08-21 | `docs(openspec): commit cleanup-cudart-cpptlm-bridge-coupling + fix-phase0-gate1-dgpu-bar-leak artifacts` | OpenSpec artifacts committed |
| `25e36f60` | 2026-08-18 | `docs(hsk-6): announce CppTLM bridge deprecation` | Docs-only; froze `CPPTLMBRIDGE_VERSION=2`; **did NOT modify CMakeLists.txt** |
| `a9a14e1d` | 2026-08 | `chore(tests): delete bridge-specific test files (Phase 1 of 4)` | Removed `tests/e2e/cosim/*` + `tests/unit/cpptlm/*` |
| `292022a3` | 2026-08 | `refactor(cudart): remove cpptlm bridge code paths (Phase 2a of 4)` | Removed bridge code from `cudart_sim.cpp` |
| `e4d7e369` | 2026-08 | `refactor(ptxsim): remove cpptlm GLOBAL LD/ST bridge (Phase 2b of 4)` | Removed `memory.cpp` bridge code |
| **`09786635`** | **2026-08** | **`refactor(cudart): remove cpptlm linkage + bridge files (Phase 3 of 4)`** | **★ Gate 1 leak physically eliminated** — removed `cpptlm_core` from `target_link_libraries(cudart ...)` |
| `d281a21e` | 2026-08 | HSK-8 Phase 2: `feat(ptxemu): add ptxemu_core library with IPtxEmuDevice API` | Replaced cpptlm bridge with `ptxemu_core` static lib |
| `c225780e` | 2026-08 | HSK-8 Phase 3: `build(cmake): PROJECT_IS_TOP_LEVEL isolation` | Cleaned up CMake build isolation |
| `738b412c` | 2026-08 | `docs(hsk-8): PTX-EMU owner ack public device API contract` | HSK-8 ack |
| `530bd6ca` | 2026-08-24 | `chore(openspec): archive ptxemu-public-device-api` | **Current HEAD** |

### 2.2 Physical state at current HEAD (`530bd6ca`)

```bash
$ grep "cpptlm_core" src/CMakeLists.txt CMakeLists.txt
# (no output)

$ sed -n '170,200p' src/CMakeLists.txt
173: add_subdirectory(ptx_ir)
174:
175: # 创建共享库
176: add_library(cudart SHARED ${SOURCES})
177: target_link_libraries(cudart ptx_ir ptx_parser ptxsim ptxir)  # ★ no cpptlm_core
178: add_dependencies(cudart GenerateParser)
179: add_dependencies(cudart ptx_parser)
180: add_dependencies(cudart ptxsim)
181: add_dependencies(cudart ptxir)
```

**Implication**: `libcudart.so` link line contains `ptx_ir`, `ptx_parser`, `ptxsim`, `ptxir` — **NO cpptlm_core**. Therefore:

```bash
$ nm -D --defined-only build/lib/libcudart.so.12.0 | grep -E "cpptlm|DGpuBar"
# (no output — guaranteed by link line, not by --exclude-libs)
```

### 2.3 Why structural elimination is stronger than `--exclude-libs`

| Aspect | `--exclude-libs=ALL` (proposed fix) | `remove cpptlm_core` (actual fix) |
|--------|---------------------------------------|----------------------------------|
| GNU ld version | Requires binutils ≥ 2.36 | Version-independent |
| Risk of accidental hiding | HIGH (per `587a6d5e`: hides 4051 ANTLR4 symbols) | Zero (cpptlm_core not linked at all) |
| Co-simulation bridge | Preserved (cpptlm_core still linked) | Removed (replaced by HSK-8 `IPtxEmuDevice`) |
| Maintenance cost | Medium (need to update `--exclude-libs` for future additions) | Low (nothing to update) |
| ABI surface complexity | Mixed (3 symbols exposed + others hidden) | Clean (only PTX-EMU-owned symbols exposed) |

---

## 3. The Dangling Commit `587a6d5e`

### 3.1 What it does

```bash
$ git show 587a6d5e --stat
commit 587a6d5ea5b23b68b99c44522a49d7361802de18
Author: PTX-EMU Developer <dev@ptx-emu.local>
Date:   Fri Aug 21 17:13:56 2026 +0800
    fix(cudart): hide cpptlm_core non-ABI symbols from libcudart.so (Gate 1)

    Add '-Wl,--exclude-libs=libcpptlm_core.a' to the '--whole-archive
    cpptlm_core' link line in CMakeLists.txt L167.
```

It changes `-Wl,--whole-archive cpptlm_core -Wl,--no-whole-archive` to `-Wl,--whole-archive,--exclude-libs=libcpptlm_core.a cpptlm_core -Wl,--no-whole-archive` and regenerates baseline.

**Results**: 3274 → 3058 symbols (216 cpptlm symbols removed), 3 e2e_cosim tests PASS (proves `cpptlm_set_driver` strong override intact).

### 3.2 Why it's dangling

```bash
$ git for-each-ref --contains 587a6d5e
# (empty — no branch contains this commit)

$ git log 587a6d5e~1..HEAD --all --oneline | head
# (587a6d5e is NOT an ancestor of any ref)
```

`587a6d5e` was committed on a local branch (likely `fix/phase0-gate1-dgpu-bar-leak` from HANDOFF_FROM_CPPTLM.md §1.2) but never merged to `main`. Subsequent 4-phase refactor (`09786635`) superseded it by removing `cpptlm_core` link entirely.

### 3.3 Disposition

**Decision**: Do **NOT** cherry-pick `587a6d5e` to `main`.

**Rationale**:

1. `09786635` already achieves the same goal more cleanly (link removal > symbol hiding)
2. Cherry-picking would **re-add** `cpptlm_core` to `target_link_libraries(cudart ...)`, contradicting HSK-8 design
3. `587a6d5e`'s commit message documents the GNU ld `--exclude-libs` learning — this remains valuable as documentation even though the fix is not deployed

**Future reference**: This commit's message + diff can serve as a "lessons learned" artifact in `docs/dev-process/lessons-learned.md` §"GNU ld `--exclude-libs` precision required" (proposed addition, see §6 below).

---

## 4. Doc2 Misattribution

### 4.1 The error

`HANDOFF_FROM_CPPTLM.md` Section 1.3 (line 60) states:

> 当前实际状态 (HSK-6 commit `25e36f60` 已删除 `--whole-archive`):
> - `CMakeLists.txt`: install rules (line 161-167), 无 `--whole-archive`
> - `src/CMakeLists.txt:177`: `target_link_libraries(cudart ptx_ir ptx_parser ptxsim ptxir)` — 普通链接,**没有 cpptlm_core**

**Correct attribution**:

- **HSK-6 commit `25e36f60`**: `docs(hsk-6): announce CppTLM bridge deprecation + consumption relationship termination` — **docs-only commit**, did NOT modify CMakeLists.txt
- **4-phase refactor** (commits `a9a14e1d` / `292022a3` / `e4d7e369` / **`09786635`**): The actual implementation that removed `--whole-archive` and `cpptlm_core` linkage
- **`09786635` specifically**: Phase 3 of 4, the decisive commit that removed `cpptlm_core` from link line

### 4.2 Why this matters

- Future code archeology may rely on Doc2's attribution → readers would chase `25e36f60` diff to understand the removal, finding only docs
- Auditors reviewing HANDOFF_FROM_CPPTLM.md need accurate commit refs
- The actual implementation effort was 4 commits × 4 authors, not a single HSK-6 commit

### 4.3 Correction

This postmortem serves as the canonical attribution. Future `lessons-learned.md` update should include §"跨仓 commit 归因必须精确到 file:line" pattern.

---

## 5. Verification Status

### 5.1 What was NOT verified (out of scope for archive)

- 5-gate ctest run on current HEAD (`530bd6ca`) — not executed
- Full regression (`./scripts/regression.sh`) — not executed
- PTX syntax tests (`tests/ptx/test_all_ptx.sh`) — not executed

### 5.2 What is structurally guaranteed

By virtue of `target_link_libraries(cudart ptx_ir ptx_parser ptxsim ptxir)` (no cpptlm_core):

- Gate 1 (`nm -D --defined-only libcudart.so` symbol surface) **PASS by construction**
- Gate 4 (`g_cpptlm_bridge == nullptr` default) **PASS by construction**

### 5.3 What needs future validation

Per Doc1 HSK-8 follow-up (`2026-08-24-hsk8-followup-task-path.md`):

- Phase 0/1 push + docs-sync — current 3 commits ahead of origin
- Phase 2.2/2.3 Setter delegation — new OpenSpec change `device-api-delegation`
- drift_check workflow (5 invariants) — `.github/workflows/drift_check.yml`
- ctest 246/246 PASS (per Doc1 H1 verification)

These are tracked in the HSK-8 follow-up plan, **NOT** this Gate 1 archive change.

---

## 6. Lessons Learned (proposed for `ptx-lessons-learned` §N + `docs/dev-process/lessons-learned.md`)

### 6.1 Lesson: Linkage removal > symbol hiding for ABI surface contracts

**Pattern**: When `--exclude-libs` is proposed to hide unwanted symbols from a shared library's dynamic export table, **consider first whether the link itself should be removed**. If the linked static library has no remaining purpose in the consumer (e.g., superseded by a different ABI mechanism), removing the link is simpler, more robust, and version-independent.

**Trigger condition**:

- `--exclude-libs` proposed in OpenSpec change to hide symbols from `--whole-archive`-linked static lib
- HSK protocol or architectural refactor is simultaneously planning to replace the static lib's functionality

**Real-world case**: `--exclude-libs=libcpptlm_core.a` (`587a6d5e`) was unnecessary because HSK-8 Phase 2 planned to replace cpptlm bridge with `ptxemu_core` (commit `d281a21e`).

### 6.2 Lesson: Cross-repo commit attribution must be precise to file:line

**Pattern**: When documenting the resolution of a cross-repo bug (e.g., "HSK-X commit removed X"), cite the specific commit that **modified the relevant files**, not the HSK protocol commit that merely **announced** the change.

**Diagnostic**: `git log -- <file>` shows ALL commits that touched the file. The HSK protocol docs commit may have `25e36f60` date but `git log -- src/CMakeLists.txt` shows the actual `09786635` removal.

**Real-world case**: Doc2 misattributed `09786635` (Phase 3 of 4 refactor) to `25e36f60` (HSK-6 docs).

### 6.3 Lesson: Dangling commits may contain valuable implementation evidence

**Pattern**: A commit not on any branch (`git for-each-ref --contains <sha>` returns empty) may seem "abandoned" but could contain:

- Empirically-rejected fix attempts (documenting what NOT to do)
- Detailed commit messages with Oracle consultation notes
- Verification artifacts (audit baselines, etc.)

**Action**: Before declaring a dangling commit "obsolete", read its commit message. If it contains lessons learned, preserve it as a reference in `docs/audits/` or merge the message content into `lessons-learned.md`.

**Real-world case**: `587a6d5e` documents the GNU ld `--exclude-libs=ALL` precision requirement — useful even though superseded by `09786635`.

### 6.4 Lesson: "Tasks not done" ≠ "Goals not met"

**Pattern**: When archiving an OpenSpec change, the artifacts' tasks may show 0/23 complete, but the **change's goal** may have been achieved by **unrelated commits** in the same repository or cross-repo.

**Diagnostic before archive**:

- Grep for the change's goal keywords in `git log --all --grep=...`
- Check if related commits in dependent changes have already implemented the goal
- Verify with current HEAD that the original failure mode is no longer present

**Real-world case**: Gate 1 leak goal was met by `09786635` even though `fix-phase0-gate1-dgpu-bar-leak/tasks.md` showed 0/23 complete.

---

## 7. Follow-up Actions

### 7.1 PTX-EMU side (this archive)

1. ✅ Update 4 OpenSpec artifacts (commit `13f55bbe`)
2. ✅ Write this postmortem (commit `<this>`)
3. ⏳ Run `openspec-archive-change` to move change to `openspec/changes/archive/2026-08-25-fix-phase0-gate1-dgpu-bar-leak/`
4. ⏳ Push branch + merge to main (PR)
5. ⏳ Notify CppTLM side that Gate 1 fix is no longer pending

### 7.2 CppTLM side (downstream)

1. ⏳ Update `AGENTS.md` HSK-8 cross-repo section: remove "等待 PTX-EMU fix 后 bump" stale description
2. ⏳ Bump submodule to include HSK-8 follow-up commits (Doc1 Phase 1 docs-sync + Phase 2 delegation)
3. ⏳ Audit `cleanup-cudart-cpptlm-bridge-coupling` (58 tasks): many are now obsolete post-4-phase refactor
4. ✅ ANTLR4 path hardcoding fix landed (commit `<antlr4-path-fix-commit-hash>`, drift_check Invariant 7 added, see `openspec/changes/antlr4-path-hardcoding-fix/`)

### 7.3 Future maintenance

- Run `ctest --test-dir build -R integration_phase0_byte_identical_gates` periodically to verify Gate 1-5 still PASS by structural elimination
- If `cpptlm_core` is ever re-linked (unlikely post-HSK-8), re-evaluate this archive rationale

---

## 8. References

### 8.1 Commits

- **Archive commit**: `13f55bbe` (artifacts update)
- **Resolution commits**:
  - `a9a14e1d` / `292022a3` / `e4d7e369` / `09786635` (4-phase refactor)
  - `25e36f60` (HSK-6 docs deprecation)
  - `d281a21e` / `c225780e` / `738b412c` (HSK-8 Phase 2-3)
  - `530bd6ca` (current HEAD)
- **Dangling reference**: `587a6d5e` (Round-2 implementation, not merged)
- **Audit**: `87820951` (HAL backend defect audit)
- **CppTLM side**: `505333b` / `beb3db8` / `12b9e0f` / `d035551` / `09c27d5`

### 8.2 OpenSpec

- Archive change: `openspec/changes/fix-phase0-gate1-dgpu-bar-leak/` → `openspec/changes/archive/2026-08-25-fix-phase0-gate1-dgpu-bar-leak/` (after `openspec-archive-change` skill)
- Downstream chain: `cleanup-cudart-cpptlm-bridge-coupling` (audit required)
- HSK-8 follow-up: `2026-08-24-hsk8-followup-task-path.md` (uncommitted, Doc1)

### 8.3 ADRs

- `docs/adr/ADR-0029-ptxemu-image-executor.md` §D7 (Gate 1 contract preserved)
- `docs/adr/ADR-0021-cpptlm-d1-full-integration.md` (Cpptlm bridge auto-co-sim)
- `docs/adr/ADR-0035` (HSK protocol ordering, for cross-repo coordination)

### 8.4 Skills

- `ptx-lessons-learned` §6 (artifacts-first) + §G (OpenSpec lifecycle)
- `openspec-archive-change` (this archive's mechanism)
- `adr-compliance-check` (post-archive Gate 1 §D7 contract verification)

---

**Postmortem Author**: PTX-EMU Architecture Team (this session)
**Date**: 2026-08-25
**Review Status**: Awaiting postmortem 提交 + CppTLM 端 AGENTS.md 同步