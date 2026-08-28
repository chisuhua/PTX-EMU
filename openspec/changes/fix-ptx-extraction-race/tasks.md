# fix-ptx-extraction-race — Tasks

## Phase 1: Red (write failing test)

- [x] 1.1 Add `tests/unit/cudart/test_cubin_extract_isolation.cu` (N-thread parallel extraction test).
- [x] 1.2 Add `tests/integration/cudart/test_parallel_cubin_extract.cu` (repeated concurrent real-binary extraction test).
- [x] 1.3 Register both tests in the existing unit and integration CMakeLists files.
- [x] 1.4 Verify both tests fail on the current tree while `cubin_utils.cpp:127-154` still uses the shared workspace.
  - Observed: missing extracted PTX, missing PTX list, and cross-call `rm` errors.

## Phase 2: Green (implement fix)

- [x] 2.1 Modify `src/utils/cubin_utils.cpp:127-154` to use `mkdtemp` per call.
  - Created a unique `/tmp/ptxemu-XXXXXX` directory.
  - Preserved cuobjdump argument order inside a private subshell.
  - Added RAII cleanup for the directory on every return path.
- [x] 2.2 Instantiate `src/cudart/cudart_sim.cpp:277` preprocessor output with `mkstemps` per registration.
- [x] 2.3 Verify tests 1.1 and 1.2 pass.

## Phase 3: Regress (verify no new failures)

- [x] 3.1 Run full `ctest -j4`; extraction-related failures did not recur across eight full-suite runs. One unrelated intermittent `integration_libptxemu_device` failure remains.
- [x] 3.2 Run full `ctest -j1`; one unrelated intermittent `integration_libptxemu_device` failure occurred and passed on immediate rerun.
- [x] 3.3 Run the Phase 1.5 scanner; baseline remains unchanged at 212 files and the script exits 0.

## Phase 4: Commit

- [x] 4.1 Commit the implementation with the race mechanisms, unique workspaces, and verification results in the message body.
- [x] 4.2 Verify the commit log shows separate logical phase commits.

## Phase 5: Document

- [x] 5.1 Check off all tasks in this file.
- [x] 5.2 No audit entry added; this remains a focused internal fix.
