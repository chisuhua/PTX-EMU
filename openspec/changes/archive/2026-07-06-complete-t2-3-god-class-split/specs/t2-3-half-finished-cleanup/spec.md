# t2-3-half-finished-cleanup Specification

## Purpose
TBD - created by archiving change complete-t2-3-god-class-split. Update Purpose after archive.
## Requirements
### Requirement: T2-3-Stubs-Removed MUST

The PTX-EMU codebase SHALL NOT contain the `src/ptxsim/contexts/` directory or any of its 7 placeholder `.cpp` files (each containing only a `// T2-3: ...` comment).

The build target `ptxsim` MUST NOT include any source file from `src/ptxsim/contexts/`. The CMake configuration MUST NOT reference `contexts` as a subdirectory.

The migration of T2-3 god-class split is acknowledged as future work (separate change), and the half-finished state MUST be cleaned up.

#### Scenario: Directory-Removed
- **WHEN** `ls src/ptxsim/contexts/` is executed
- **THEN** the directory MUST NOT exist (or be empty)

#### Scenario: CMake-Cleanup
- **WHEN** `grep "contexts" src/CMakeLists.txt` is executed
- **THEN** zero matches MUST be returned

### Requirement: T2-3-Unused-POD-Removed MUST

The `WarpContext` class MUST NOT contain the fields `backend_links_` or `warp_identity_` (previously declared at `include/ptxsim/warp_context.h:279-280` as T2-3 placeholders with zero reads/writes).

The fields MUST be removed from `WarpContext` class definition without affecting any other behavior.

#### Scenario: Field-Removed
- **WHEN** grep searches for `backend_links_` or `warp_identity_` in src/, include/, tests/
- **THEN** zero matches MUST be returned (excluding git history)

#### Scenario: No-Behavioral-Change
- **WHEN** running ctest after cleanup
- **THEN** all pre-existing tests MUST pass identically to baseline

### Requirement: Future-T2-3-Work-Recognized MUST

The cleanup MUST explicitly recognize that actual T2-3 god-class split implementation is future work. Specifically:

1. `docs/roadmap/post-phase3-debt-roadmap.md` MUST NOT list T2-3 half-finished state as remaining debt (it's now cleaned)
2. The original `archive/2026-06-24-phase3-t2-3-god-class-split/` MUST NOT be amended (per lessons-learned Checklist G)
3. Future T2-3 implementation MUST be tracked as a separate OpenSpec change

#### Scenario: Documentation-Synced
- **WHEN** reading `docs/roadmap/post-phase3-debt-roadmap.md`
- **THEN** the T2-3 half-finished debt item MUST be removed
- **AND** the original archived change MUST remain unchanged (verified via `git log -- archive/2026-06-24-phase3-t2-3-god-class-split/`)

#### Scenario: Original-Archive-Untouched
- **WHEN** checking `archive/2026-06-24-phase3-t2-3-god-class-split/`
- **THEN** it MUST remain as the historical record (per lessons-learned Checklist G)
- **AND** no new commits MUST reference it as amended