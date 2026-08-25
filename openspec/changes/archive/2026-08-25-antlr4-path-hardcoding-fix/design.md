# antlr4-path-hardcoding-fix — Design

## Context

PTX-EMU uses **vendored ANTLR4** (`antlr4/antlr-4.13.2-complete.jar` + `antlr4/antlr4-cpp-runtime-4.13.2-source/`) per HSK-2 (`docs/superpowers/hsk-drafts/2026-07-16/HSK-2-antlr4-version.md`). The vendored paths are referenced in `CMakeLists.txt:98-99` via `${CMAKE_SOURCE_DIR}`:

```cmake
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/antlr4/antlr-4.13.2-complete.jar)
set(ANTLR4_RUNTIME_SOURCE_DIR ${CMAKE_SOURCE_DIR}/antlr4/antlr4-cpp-runtime-4.13.2-source)
```

**现状问题**: `CMAKE_SOURCE_DIR` is a CMake built-in variable that resolves to the **top-level project's source directory**. When PTX-EMU is consumed via:
- `add_subdirectory(external/PTX-EMU)` from CppTLM: `CMAKE_SOURCE_DIR` = CppTLM root, `${CMAKE_SOURCE_DIR}/antlr4/` doesn't exist
- `ExternalProject_Add(... PTX-EMU GIT_REPOSITORY ...)` from CppTLM: same problem (CMAKE_SOURCE_DIR = CppTLM root)

This means CppTLM-side chained builds (which is the post-HSK-8 ack integration target per `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md`) **fail at ANTLR4 resolution** in any subproject consumption pattern.

**Current workaround**: PTX-EMU-side standalone build only. CppTLM must vendor ANTLR4 separately (defeats HSK-2 "single source of truth" intent).

**目标状态**: PTX-EMU's `CMakeLists.txt` uses `PROJECT_SOURCE_DIR` (which resolves to **the project that called `project()`**, i.e., PTX-EMU's own root) for vendored path references. CppTLM-side chained builds via `add_subdirectory` / `ExternalProject_Add` succeed with zero ANTLR4 path adjustments.

## Goals / Non-Goals

**Goals:**
- Replace `${CMAKE_SOURCE_DIR}` with `${PROJECT_SOURCE_DIR}` for vendored ANTLR4 path variables in `CMakeLists.txt`
- Add drift_check Invariant 7 to prevent regression to `${CMAKE_SOURCE_DIR}/antlr4` hardcoding
- Standalone build behavior unchanged (zero regressions)
- CppTLM-side chained build via `add_subdirectory` now succeeds (validation: write minimal `CMakeLists.txt` mini-test that consumes PTX-EMU)

**Non-Goals:**
- Full HSK-9 entry (`tests/build_cpptlm_consume/consumer_smoke`) — depends on CppTLM consumer demand (Q1 Open Question)
- Phase 1.5 namespace migration (`include/ptx_ir/` → `include/ptxemu/ir/`) — separate change
- ANTLR4 version upgrade (HSK-2 locks 4.13.2)
- CppTLM-side `ExternalProject_Add` path adjustment (CppTLM-side, not PTX-EMU-side)
- Adding new public methods to `IPtxEmuDevice` (would require HSK-9)
- Removing any vendored content (HSK-2 mandates single source)

## Decisions

### Decision 1: Use `PROJECT_SOURCE_DIR` (not `CMAKE_CURRENT_SOURCE_DIR`)

**Choice**: `${PROJECT_SOURCE_DIR}` for both `ANTLR_EXECUTABLE` and `ANTLR4_RUNTIME_SOURCE_DIR`.

**Rationale**:
- `PROJECT_SOURCE_DIR` is set by `project()` to the directory of the most recently called `project()` command — typically the PTX-EMU root (when PTX-EMU is either standalone or consumed via `add_subdirectory`)
- `CMAKE_CURRENT_SOURCE_DIR` is the directory of the currently-being-processed CMakeLists.txt — usually the same as `PROJECT_SOURCE_DIR` for the top-level file, but differs in nested `add_subdirectory` scenarios (e.g., `${CMAKE_CURRENT_SOURCE_DIR}` in `src/CMakeLists.txt` = `src/` not project root)
- `PROJECT_SOURCE_DIR` is **stable across nested CMakeLists.txt inclusions** — the right semantics for "where vendored content are"

**Alternatives considered**:
- `${CMAKE_CURRENT_SOURCE_DIR}`: rejected — would break in nested CMakeLists.txt subdirectory inclusion
- Relative path (`./antlr4/...`): rejected — fragile, breaks if `cmake -S <other-dir>` is used
- User-supplied `-DANTLR4_ROOT=/abs/path` cache variable: rejected — adds user-facing configuration burden; default must be the vendored location for PTX-EMU standalone

### Decision 2: Add drift_check Invariant 7 (single-line grep)

**Choice**: Invariant 7 is a simple grep on `CMakeLists.txt`:
```bash
! grep -nE "CMAKE_SOURCE_DIR.*antlr4|antlr4.*CMAKE_SOURCE_DIR" CMakeLists.txt
```

**Rationale**:
- Mirrors existing Invariant 6 pattern (regex-based regression guard for `device_api_impl.cc`)
- Single-line, ~50ms runtime, no AST parsing needed
- Catches the specific regression: re-introduction of `CMAKE_SOURCE_DIR` for ANTLR4 paths
- Does NOT catch other `CMAKE_SOURCE_DIR` misuses (out of scope; future invariant if needed)

**Alternatives considered**:
- AST-based check (Python parse of CMake): rejected — overkill for single-variable check; runtime cost
- Multiple-grep covering all vendored dependencies: rejected — out of scope (only ANTLR4 is the known issue)

### Decision 3: Add minimal CppTLM-consume smoke (out of scope, documented)

**Choice**: This change **does NOT** add `tests/build_cpptlm_consume/consumer_smoke` (that requires CppTLM-side coordination). The validation of `${PROJECT_SOURCE_DIR}` correctness is via:
- Standalone build (`cmake -S . -B build`) — passes if change is correct
- Inline assertion in commit message: "Build verified standalone"

**Rationale**: HSK-9 entry is a separate concern requiring CppTLM-side demand (Q1 Open Question). This change makes HSK-9 entry possible but does not perform it.

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| `${PROJECT_SOURCE_DIR}` differs from `${CMAKE_SOURCE_DIR}` in some CMake invocation patterns, causing ANTLR4 path to point to wrong location | LOW | Validate via `cmake -S . -B build` standalone build (must succeed identically). Document expected behavior in commit message |
| Regression: future contributor re-introduces `${CMAKE_SOURCE_DIR}` for vendored paths | MEDIUM | drift_check Invariant 7 prevents; lessons-learned §N added to `docs/dev-process/lessons-learned.md` |
| Other vendored dependencies (json, inipp at CMakeLists.txt:117-118) might have similar issue | LOW | Out of scope for this change; if reported, follow-up change. Currently those paths use `${CMAKE_CURRENT_SOURCE_DIR}` which is correct for subdirectory-relative vendored content |
| PTX-EMU-only build consumers (standalone) might notice no behavioral change but expect one | LOW | drift_check Invariant 7 visible in CI; commit message documents the fix |
| CppTLM-side downstream changes required to consume PTX-EMU via add_subdirectory | LOW | Not required for ANTLR4 fix; CppTLM-side `add_subdirectory(PTX-EMU)` will now find ANTLR4 paths at PTX-EMU's PROJECT_SOURCE_DIR, which is the expected behavior |

## Migration Plan

**Pre-deployment** (per `ptx-lessons-learned` §4):
```bash
# 1. Establish baseline worktree
git worktree add .worktrees/antlr4-path-fix-baseline HEAD

# 2. Verify baseline tests pass
cd .worktrees/antlr4-path-fix-baseline
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
# Expected: 249/249 PASS (per device-api-delegation archive)
```

**Deployment** (single atomic commit):
```bash
# 1. Switch back to main working tree
cd /path/to/PTX-EMU
git checkout main
git pull origin main  # if any new commits since baseline

# 2. Apply change
# Edit CMakeLists.txt:98-99:
#   CMAKE_SOURCE_DIR → PROJECT_SOURCE_DIR (2 instances)

# 3. Add drift_check Invariant 7
# Edit .github/workflows/drift_check.yml (per existing Invariant 6 pattern)

# 4. Add lessons-learned entry
# Edit docs/dev-process/lessons-learned.md (append §N)

# 5. Update AGENTS.md HSK chain §HSK-8 follow-up (note Phase 2.4 entry)

# 6. Verify
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
# Expected: 249/249 PASS, drift_check 7 invariants PASS
```

**Rollback**: `git revert <commit-hash>` (single commit, no inter-dependencies). No special handling needed.

## Open Questions

None. All decisions resolved by existing HSK protocol + CMake best practices. Implementation can proceed after Phase 0.

## Reference

- **HSK-2 contract**: `docs/superpowers/hsk-drafts/2026-07-16/HSK-2-antlr4-version.md`
- **HSK-8 ack spec**: `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md`
- **Postmortem reference**: `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md:265`
- **Skills referenced**:
  - `cmake` (best practices for subproject-consumed libraries)
  - `ptx-lessons-learned` §3 (multi-phase criterion — N/A here, single commit) + §4 (baseline worktree) + §6 (artifacts-first)
- **Affected lines**: `CMakeLists.txt:98-99`
- **Affected workflow**: `.github/workflows/drift_check.yml` (Invariant 7 added)
- **Affected doc**: `docs/dev-process/lessons-learned.md` (new §N appended)
- **Affected AGENTS.md**: §HSK-8 follow-up Phase 2.4 entry