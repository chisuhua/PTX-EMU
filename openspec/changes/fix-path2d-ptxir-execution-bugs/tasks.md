# Tasks: Path 2D PTXIR Execution Bug Fixes

## 1. Baseline and reproduction

- [ ] 1.1 Create a baseline worktree from the current HEAD at `.worktrees/baseline-fix-path2d-bugs` and build it (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`).
- [ ] 1.2 Run the existing relevant ctest slices (`ctest -L e2e`, `ctest -L "unit;cudart"`, `ctest -L "integration;cpptlm"`) in the baseline worktree and record the results.
- [ ] 1.3 Preserve the failing NVIDIA CUDA Samples `vectorAdd.cu` path_2D reproduction: capture the exact `ptxemu_image_load`/`execute`/`unload` log lines and the final `InvalidMemoryAccessException` message in `openspec/changes/fix-path2d-ptxir-execution-bugs/baseline-reproduction.md.md`.
- [ ] 1.4 Run `nm -D build/lib/libptxemu_device.so.12.0 | grep ptxemu_` and `nm -D build/lib/libcudart.so.12.0 | grep -E "g_gpu_context|wait_for_completion"` and save the output to `baseline-symbols.txt` for later comparison.

## 2. PTXIR normalization in the image executor

- [ ] 2.1 Write failing unit tests in `tests/unit/cudart/test_cpptlm_module_image_normalization.cpp`: standalone PTXIR returns kernel name and count, embedded PTXIR-Embedded image returns kernel name and count, and malformed embedded footer returns an error.
- [ ] 2.2 Run the new tests and confirm they fail with the current executor (standalone path may pass; embedded path must fail).
- [ ] 2.3 Trace every call site of `read_manifest_from_ptxir_section` and `PTXIRLoader::deserializeForCubin` inside `src/cudart/cpptlm_module.cpp` to confirm the normalization boundary.
- [ ] 2.4 Add a private helper in `src/cudart/cpptlm_module.cpp` that returns a normalized standalone PTXIR byte view: standalone input is returned as-is (as a borrowed span over the stored image); embedded input is extracted via `PTXIRLoader::extractPTXIR` into an owned buffer.
- [ ] 2.5 Route `get_kernel_name`, `kernel_count`, `kernel_name_at`, `execute`, and `execute_named` through the new helper before any `read_manifest_from_ptxir_section` or `PTXIRLoader::deserializeForCubin` call.
- [ ] 2.6 Run `cmake --build build --target ptxemu_device` followed by the new unit tests; they MUST pass.
- [ ] 2.7 Run `ctest -L "unit;cudart"` and `ctest -L "integration;cpptlm"` to confirm no regression; revert this phase on failure.
- [ ] 2.8 Commit Phase 2 changes with message `fix(cudart): normalize PTXIR image bytes before manifest lookup`.

## 3. Synchronous path_2D completion

- [ ] 3.1 Write a failing E2E test in `tests/e2e/path_2D_image_executor/test_image_executor_synchronous.cpp`: load a known store-kernel fixture, execute, and verify a sentinel write is observable before the API returns.
- [ ] 3.2 Run the new test and confirm it fails (writes are not yet observable because the executor returns before `wait_for_completion`).
- [ ] 3.3 Modify `PtxEmuImageExecutor::execute` and `execute_named` in `src/cudart/cpptlm_module.cpp` to call `g_gpu_context->wait_for_completion()` after `PtxInterpreter::launchPtxInterpreter` and before returning success. The call MUST stay inside the existing `exec_mu_` critical section.
- [ ] 3.4 Add an explicit missing-context branch: if `g_gpu_context == nullptr`, return `-EINVAL` and do NOT report successful execution.
- [ ] 3.5 Run `cmake --build build --target ptxemu_device` followed by the new test; it MUST pass.
- [ ] 3.6 Run `ctest -L "unit;cudart"`, `ctest -L "integration;cpptlm"`, and `ctest -L e2e`; no regression. Revert this phase on failure.
- [ ] 3.7 Commit Phase 3 changes with message `fix(cudart): drive GPU completion in path_2D image executor`.

## 4. Global-memory address normalization

- [ ] 4.1 Trace the call path of `ld.global` and `st.global` from instruction handlers through their memory accessor. Record the exact source file(s) and function(s) where the address enters `SimpleMemory` access in `openspec/changes/fix-path2d-ptxir-execution-bugs/memory-trace.md.md`. Do not guess a source location.
- [ ] 4.2 Write failing unit tests for the chosen normalization helper: absolute address inside `[pool_base, pool_base + pool_size)` is converted to the corresponding offset; valid relative address remains unchanged; address outside the pool keeps the existing invalid-memory behavior.
- [ ] 4.3 Run the new tests and confirm they fail (current path rejects absolute addresses).
- [ ] 4.4 Implement the normalization helper at the verified common global-memory access boundary identified in 4.1. The helper MUST keep non-global address spaces untouched and MUST reuse the existing `InvalidMemoryAccessException` path for invalid addresses.
- [ ] 4.5 Run `cmake --build build --target ptxemu_device` followed by the new unit tests; they MUST pass.
- [ ] 4.6 Run `ctest -L "unit;ptx"`, `ctest -L "integration;ptx"`, and `ctest -L e2e`; no shared/local/param regression. Revert this phase on failure.
- [ ] 4.7 Commit Phase 4 changes with message `fix(ptxsim): normalize cudaMalloc pool absolute addresses for global memory access`.

## 5. CUDA Samples path_2D acceptance harness

- [ ] 5.1 Add the harness script `examples/run_cuda_sample.sh` (or its OpenSpec-approved location) that loads `env.sh`, compiles `cpp/0_Introduction/vectorAdd/vectorAdd.cu` with `nvcc -arch=sm_100`, extracts PTX, generates a standalone PTXIR image, loads it through `libptxemu_device.so`, and verifies `abs(A[i] + B[i] - C[i]) < 1e-5` for every element.
- [ ] 5.2 Register the harness as `tests/e2e/path_2D_image_executor/test_cuda_samples_vectorAdd.cpp` (or the agreed E2E test path) with ctest label `e2e;path_2D;cuda_samples`.
- [ ] 5.3 Run `cmake --build build && ctest -L e2e -R cuda_samples_vectorAdd --output-on-failure`; it MUST pass with all `N` elements verified.
- [ ] 5.4 If the harness fails, do NOT extend the fix into other modules. Return the the fix scope to the failing boundary and stop.

## 6. Documentation and governance

- [ ] 6.1 Amend ADR-0029 with a new sub section `Path 2D Synchronous Completion` and `Path 2D PTXIR Image Normalization`. Add the new invariants to the existing `[SINGLE-GPU-INSTANCE]` block. Reference `openspec/changes/fix-path2d-ptxir-execution-bugs/{proposal,design,tasks}.md` and the verified boundary from the memory trace.
- [ ] 6.2 Update `docs/superpowers/specs/2026-08-13-path2d-silent-bugs-design.md` with the verified call path and any decisions that were left as Open Questions at design time. Mark it `Active`.
- [ ] 6.3 Synchronize `README.md` "已实现功能" or equivalent section to mention `path_2D image executor PTXIR normalization + synchronous completion`. Do NOT use stale strings like "stub" or "TODO".
- [ ] 6.4 Verify `git status openspec/changes/fix-path2d-ptxir-execution-bugs/` shows the proposal, design, tasks, and specs are tracked before the implementation commits are merged.

## 7. Final verification

- [ ] 7.1 Run `cmake --build build` for a full rebuild.
- [ ] 7.2 Run `cd build && ctest --output-on-failure`.
- [ ] 7.3 Run `tests/ptx/test_all_ptx.sh`.
- [ ] 7.4 Build and run `bench/dummy-add` (path_1B regression) and the new `path_2D` CUDA samples harness.
- [ ] 7.5 Run `nm -D build/lib/libptxemu_device.so.12.0 | grep ptxemu_` and compare the symbol list with `baseline-symbols.txt`; the public symbol set MUST be unchanged.
- [ ] 7.6 If any gate fails, follow `.opencode/skills/ptx-lessons-learned/SKILL.md` Checklist M and the relevant failure mode in the lessons-learned failure-mode table before changing the plan.

## 8. Rollback rules

- If Phase 2 regresses an existing path_2D test, `git revert` the Phase 2 commit and stop before touching Phase 3.
- If Phase 3 regresses a legacy runtime test (`dummy-add` or other path_1A/1B tests), `git revert` the Phase 3 commit and stop before touching Phase 4.
- If Phase 4 regresses any non-global access path, `git revert` the Phase 4 commit and re-trace the memory access boundary before retrying.
- If Phase 5 fails after Phase 2-4 all pass, the fix is correct; the harness needs adjusting without touching the implementation.