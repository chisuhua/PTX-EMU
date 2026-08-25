# antlr4-path-hardcoding-fix

## Why

PTX-EMU's `CMakeLists.txt:98-99` uses `${CMAKE_SOURCE_DIR}` to locate vendored ANTLR4 (`antlr4/antlr-4.13.2-complete.jar` + `antlr4/antlr4-cpp-runtime-4.13.2-source`). `CMAKE_SOURCE_DIR` always resolves to the **top-level project root**, not PTX-EMU's own source directory. This works for PTX-EMU standalone builds (`CMAKE_SOURCE_DIR` == PTX-EMU root) but breaks when PTX-EMU is consumed as a subproject by CppTLM via `add_subdirectory` or `ExternalProject_Add`: CppTLM's CMakeLists.txt becomes the top-level, so `CMAKE_SOURCE_DIR` resolves to CppTLM's root, and `${CMAKE_SOURCE_DIR}/antlr4/` does not exist. CppTLM-side `cmake --build build` fails with `file(STRINGS .../antlr4-4.13.2-complete.jar)` or similar ANTLR4 path errors.

This blocks Doc2 §8 follow-up list item 4 (per `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md:265` §7.2) and is the only remaining build-system coupling between the two repos that prevents fully-correct CppTLM-side chained builds. Fix is trivial (2-line change `CMAKE_SOURCE_DIR` → `PROJECT_SOURCE_DIR`) but must be gated by a new drift_check invariant to prevent regression.

## What Changes

- **Modify** `CMakeLists.txt:98-99`: change `CMAKE_SOURCE_DIR` → `PROJECT_SOURCE_DIR` for both `ANTLR_EXECUTABLE` and `ANTLR4_RUNTIME_SOURCE_DIR` variables (2-line change)
- **Modify** `.github/workflows/drift_check.yml`: add Invariant 7 — verify `CMakeLists.txt` contains no `${CMAKE_SOURCE_DIR}/antlr4` hardcoded path. **NOTE**: `paths` trigger filter MUST be added in BOTH `pull_request.paths` (L11-17) AND `push.paths` (L20-24) sections.
- **Modify** `docs/dev-process/lessons-learned.md`: append new entry §N on CMake `CMAKE_SOURCE_DIR` vs `PROJECT_SOURCE_DIR` for vendored dependencies

**Out of scope** (explicitly NOT in this change):
- HSK-9 entry (`tests/build_cpptlm_consume/consumer_smoke` full implementation) — depends on CppTLM consumer demand (Q1 Open Question per `2026-08-24-hsk8-followup-task-path.md` Phase 3 Task 3.3). `cmake-antlr4-relative-paths/spec.md` Scenarios 2-3 (CppTLM-side add_subdirectory / ExternalProject_Add consumption succeeds) are deferred to HSK-9 entry; this change's verification is limited to standalone PTX-EMU build.
- Phase 1.5 namespace migration (`include/ptx_ir/` → `include/ptxemu/ir/`) — separate change, triggering condition-driven
- ANTLR4 version upgrade — HSK-2 contract locks 4.13.2 (per `docs/superpowers/hsk-drafts/2026-07-16/HSK-2-antlr4-version.md`)

## Capabilities

### New Capabilities

- `cmake-antlr4-relative-paths`: PTX-EMU's `CMakeLists.txt` MUST use `PROJECT_SOURCE_DIR` (or `CMAKE_CURRENT_SOURCE_DIR`) for all vendored path references, ensuring correct resolution when PTX-EMU is consumed as a subproject via `add_subdirectory` or `ExternalProject_Add`

### Modified Capabilities

- `ci-drift-check`: ADD Invariant 7 — verify `CMakeLists.txt` contains no `${CMAKE_SOURCE_DIR}/antlr4` hardcoded path (extending the 6 invariants from `device-api-delegation` archive to 7 total)

## Impact

**Source code**: 2 lines changed in `CMakeLists.txt:98-99`. Zero new files.

**Build artifact**: `libcudart.so` and `libptxemu_device.so` continue to build identically (the path variables are used only for ANTLR4 parser generation; output is deterministic).

**Test impact**: 
- Standalone build: `cmake -S . -B build` produces identical build (cudart symbol baseline `nm -D build/lib/libcudart.so | wc -l` unchanged)
- CppTLM-side chained build: now succeeds (previously fails at `ANTLR_EXECUTABLE` resolution)
- ctest count unchanged: 249/249 PASS

**CI impact**: drift_check grows from 6 invariants to 7. Runtime: +1 grep invocation (~50ms).

**HSK impact**: None. This is a PTX-EMU single-repo fix. `cpp 不暴露` constraint preserved (no new public API). `PTXEMU_API_VERSION=1` unchanged.

**Documentation**:
- `README.md` §已实现功能: no change (implementation-level fix)
- `AGENTS.md` HSK chain: add brief note in §HSK-8 follow-up (or new §HSK-8 follow-up Phase 2.4) recording this as Doc2 §8 item 4 resolved
- `docs/dev-process/lessons-learned.md`: append entry #N on **CMake `CMAKE_SOURCE_DIR` vs `PROJECT_SOURCE_DIR` for vendored dependencies in subproject-consumed libraries**

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性 (N/A)
- No function migration. Pure build-system variable substitution.
- `CMAKE_SOURCE_DIR` and `PROJECT_SOURCE_DIR` have identical values when PTX-EMU is the top-level project (the only case currently tested). The fix only differs in `add_subdirectory`/`ExternalProject_Add` consumer scenarios — no behavior change for standalone builds.

### 状态修改 (N/A)
- No state modification. Build-system-only change.

### 多 Phase 推进 (N/A)
- Single atomic commit. ~2 LOC change + 1 drift_check invariant + 1 lessons-learned entry.
- No multi-phase split needed (per `ptx-lessons-learned` §3 multi-phase criterion: ≥3 commits OR independent rollback granularity — neither applies).

### 文档同步 (Checklist I)
- [ ] `AGENTS.md` §HSK-8 follow-up: add note "Phase 2.4 ANTLR4 path fix landed (commit `<hash>`, drift_check Invariant 7 added)"
- [ ] `docs/dev-process/lessons-learned.md`: append §N "CMake `CMAKE_SOURCE_DIR` vs `PROJECT_SOURCE_DIR` for vendored dependencies"
- [ ] `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md` §7.2 item 4: mark as ✅ resolved (or add resolution commit reference)
- [ ] README §已实现功能: no change (build infra fix)

## Reference

- **Tracking issue / Doc2 §8 item 4**: `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md:265`
- **Parent HSK-8 follow-up plan**: `2026-08-24-hsk8-followup-task-path.md` (untracked; will commit before this change's Phase 0 per `ptx-lessons-learned` §6 artifacts-first)
- **Antlr4 version contract**: `docs/superpowers/hsk-drafts/2026-07-16/HSK-2-antlr4-version.md` (locks 4.13.2)
- **Affected code**: `CMakeLists.txt:98-99`
- **drift_check workflow**: `.github/workflows/drift_check.yml` (existing 6 invariants → 7 after this change)
- **Skills referenced**: 
  - `ptx-lessons-learned` §3 (multi-phase criterion) + §6 (artifacts-first) + §I (Checklist I doc sync)
  - `cmake` (CMake best practices for subproject-consumed libraries)
- **HSK chain status** (per AGENTS.md): no HSK trigger; PTXEMU_API_VERSION=1 frozen