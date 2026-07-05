# dead-active-warp-manager-removal Specification

## Purpose
TBD - created by archiving change remove-dead-active-warp-manager. Update Purpose after archive.
## Requirements
### Requirement: ActiveWarpManager-Module-Deleted MUST

The PTX-EMU codebase SHALL NOT contain the `ActiveWarpManager` class. The file `include/ptxsim/active_warp_manager.h` and `src/ptxsim/core/active_warp_manager.cpp` MUST be removed.

The build target `ptxsim` MUST NOT include `src/ptxsim/core/active_warp_manager.cpp` as a source file.

The authoritative warp scheduler is `WarpScheduler` (with `RoundRobinWarpScheduler` as default and `GreedyWarpScheduler` as alternative), used exclusively via `SMContext::warp_scheduler` (no other production path).

#### Scenario: Header-Removed
- **WHEN** the build system attempts to locate `ptxsim/active_warp_manager.h`
- **THEN** the file MUST NOT exist

#### Scenario: Source-Removed
- **WHEN** CMake configures the `ptxsim` target
- **THEN** `src/CMakeLists.txt` MUST NOT list `ptxsim/core/active_warp_manager.cpp`

#### Scenario: Authoritative-Scheduler-Unchanged
- **WHEN** `SMContext` constructs its scheduler
- **THEN** it MUST use `std::make_unique<RoundRobinWarpScheduler>()` (line 23)
- **AND** no production code path uses `ActiveWarpManager`

### Requirement: ActiveWarpManager-Cleanup-Verified MUST

The cleanup MUST be verified by zero-regression baseline comparison. Specifically:

1. All pre-existing tests MUST pass (`ctest --output-on-failure` 100% PASS)
2. PTX syntax tests MUST pass (`./tests/ptx/test_all_ptx.sh` 100% PASS)
3. SMContext scheduling logic MUST continue to function identically (8 call sites)
4. Documentation MUST be synced per lessons-learned Checklists I

#### Scenario: SMContext-Unchanged
- **WHEN** inspecting `src/ptxsim/core/sm_context.cpp`
- **THEN** all 8 `warp_scheduler->*()` call sites MUST continue to function identically
- **AND** no references to `ActiveWarpManager` MUST exist

#### Scenario: Documentation-Synced
- **WHEN** reading `docs/audits/debt-audit-2026-07-02.md`
- **THEN** the ActiveWarpManager debt item MUST have a "✅ FIXED by commit <hash>" annotation
- **AND** `docs/roadmap/post-phase3-debt-roadmap.md` MUST NOT list ActiveWarpManager in remaining debt