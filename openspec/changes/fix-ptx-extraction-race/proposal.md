# fix-ptx-extraction-race

## Why

ctest baseline shows 246/252 PASS, with six pre-existing flaky or SEGFAULT failures. The affected targets are `dummy-float`, `dummy-mul`, `dummy-ldglobal`, `e2e_ldglobal_simple`, `simpleGEMM-double`, and `all-pairs-distance`.

The root cause is `src/utils/cubin_utils.cpp:127-154`, where `extract_ptx_with_cuobjdump` extracts PTX into the shared process current working directory. Parallel `ctest -j4` runs race on the shared workspace: one test can overwrite or remove files while another call is still using them.

This blocks any OpenSpec gate requiring a clean 252/252 baseline and inflates Phase 1.5 sweep verification overhead. The issue is independent of namespace migration because the PTX extraction path does not touch IR types.

## What changes

- Modify `src/utils/cubin_utils.cpp` to use a per-call `mkdtemp()` workspace for PTX listing and extraction.
- Modify `src/cudart/cudart_sim.cpp` to instantiate the PTX preprocessor output path with `mkstemps()`.
- Add `tests/unit/cudart/test_cubin_extract_isolation.cu` for concurrent extraction isolation.
- Add `tests/integration/cudart/test_parallel_cubin_extract.cu` for concurrent real-binary extraction coverage.

## Impact

- Affected code: one file, approximately 30 changed lines.
- Affected tests: two new files, approximately 150 lines total.
- ABI: unchanged; this is an internal helper implementation change.
- HSK: none; this is internal to PTX-EMU.
