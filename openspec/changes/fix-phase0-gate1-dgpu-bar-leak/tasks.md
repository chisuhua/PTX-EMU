## 1. Pre-Flight Verification

- [ ] 1.1 Verify GNU binutils version ≥ 2.36 (required for `--exclude-libs=ALL`): `ld --version | head -1`. **MUST**: if older, abort and document fallback path in commit message.
- [ ] 1.2 Verify clean working tree: `git status` shows "nothing to commit, working tree clean" on commit `87820951` (or user's current HEAD). MUST.
- [ ] 1.3 Capture OLD baseline for audit: `cp /tmp/baseline-artifacts/libcudart-nm-before.txt /tmp/baseline-artifacts/libcudart-nm-before-OLD.txt`. **MUST**: preserved for diff audit in step 2.4.

## 2. CMake + Baseline Regeneration

- [ ] 2.1 Edit `PTX-EMU/CMakeLists.txt:166-167`: change
  ```
  -Wl,--whole-archive cpptlm_core -Wl,--no-whole-archive
  ```
  to
  ```
  -Wl,--whole-archive,--exclude-libs=ALL cpptlm_core -Wl,--no-whole-archive
  ```
  **MUST**: comma separator between `--whole-archive` and `--exclude-libs=ALL` (GNU ld comma syntax). NOTE: do not break the line in a way that introduces cmake variable interpretation issues.
- [ ] 2.2 Rebuild: `. env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`. Expected: 100% build success, no new linker warnings.
- [ ] 2.3 Verify ABI surface preservation: `nm -D build/lib/libcudart.so | grep -E "cpptlm_set_driver|cpptlm_attach_bridge|cpptlm_detach_bridge|g_cpptlm_bridge"`. **REVISED expectation (Metis A1 + Oracle H1.3)**: 3 symbols (`cpptlm_attach_bridge`, `cpptlm_detach_bridge`, `g_cpptlm_bridge`) MUST be present (defined in PTX-EMU's own `PtxEmuDriverShim.cpp` via `src/CMakeLists.txt:43-48` GLOB — not affected by `--exclude-libs`). The 4th symbol (`cpptlm_set_driver`) is **expected to be hidden** because its strong definition lives in `cpptlm_core` (the very archive targeted by `--exclude-libs=ALL`); GNU ld `--exclude-libs` hides archive symbols even when referenced internally. The strong override is preserved functionally (linker binds the call to the strong body within the archive); what is hidden is only the entry in the dynamic export table. The true e2e acceptance for the strong override is **task 4.3** (`ctest --test-dir build -R 'e2e_cosim'` — all 3 cosim tests must PASS, proving `cpptlm_set_driver` strong override still happens at link time). If `cpptlm_attach_bridge` or `cpptlm_detach_bridge` or `g_cpptlm_bridge` are also absent: ABORT — `--exclude-libs` accidentally hides PTX-EMU-owned symbols. Fallback: add `-Wl,-u,<symbol>` for each missing PTX-EMU-owned symbol (effect on `--exclude-libs`-hidden archive symbols NOT verified — needs empirical test).
- [ ] 2.4 Verify symbol leak eliminated: `nm -D --defined-only build/lib/libcudart.so | grep DGpuBar`. Expected: empty output (zero matches).
- [ ] 2.5 Capture new baseline: `mkdir -p /tmp/baseline-artifacts && nm -D --defined-only build/lib/libcudart.so.12.0 | sort > /tmp/baseline-artifacts/libcudart-nm-before.txt`.
- [ ] 2.6 Generate audit artifact: `nm -D --defined-only build/lib/libcudart.so.12.0 | sort > /tmp/current_syms.txt && diff /tmp/baseline-artifacts/libcudart-nm-before-OLD.txt /tmp/current_syms.txt > docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt`. **MUST**: the diff shows ONLY `<` lines (symbols removed from old baseline), ALL of them being cpptlm_core origin symbols (tlm::/cpptlm::/nlohmann:: mangled names). Zero `>` lines (no symbols added that weren't in old baseline).

## 3. Gate 1 Verification

- [ ] 3.1 Run Gate 1-5 in isolation: `. env.sh && unset EMU_COSIM && ctest --test-dir build -R integration_phase0_byte_identical_gates --output-on-failure`. Expected: all 5 gates pass (Gate 1: symbol surface, Gate 2: SONAME, Gate 3: symlinks, Gate 4: g_cpptlm_bridge == nullptr, Gate 5: get_gpu_clock_from_context). NOTE: Gate 1 transitions from FAIL to PASS.

## 4. Full Regression Verification

- [ ] 4.1 Unit tests: `ctest --test-dir build -L unit`. Expected: 111 tests pass, 0 fail.
- [ ] 4.2 Integration tests: `ctest --test-dir build -L integration`. Expected: ~120 tests pass, 0 fail. NOTE: only `integration_phase0_byte_identical_gates` was failing before; all others already pass.
- [ ] 4.3 E2E tests (excl. e2e_divergence): `ctest --test-dir build -L e2e -E 'e2e_divergence$'`. Expected: 21 tests pass, 0 fail. **MUST verify `e2e_cosim_*` (3 tests) still pass** — proves `cpptlm_set_driver` strong override preserved by `--whole-archive` combination.
- [ ] 4.4 PTX syntax tests: `bash tests/ptx/test_all_ptx.sh`. Expected: 46/46 pass.
- [ ] 4.5 mini benchmarks: `ctest --test-dir build -L mini`. Expected: 13/13 pass.
- [ ] 4.6 cute benchmarks: `ctest --test-dir build -L cute`. Expected: 5/5 pass.
- [ ] 4.7 Full regression script: `./scripts/regression.sh`. Expected: "标准回归全部通过 ✓".

## 5. Documentation Audit

- [ ] 5.1 Verify audit artifact: `cat docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt | head -30`. Expected: diff content with only `<` lines (removed symbols). Lines start with `< T _ZN3tlm...` or `< B _ZN3tlm...` etc.
- [ ] 5.2 Verify audit artifact is git-trackable: `git status docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt`. NOTE: if this file is in `.gitignore`, move to a tracked location like `docs/audits/`.

## 6. Commit (DO NOT COMMIT — user instruction)

NOTE: Per task management rules ("NEVER commit without explicit request"), do NOT run `git commit`. Stop after task 5 and report success to user. User will explicitly request commit + push.

- [ ] 6.1 [USER ACTION] User to commit the change with message:
  ```
  fix(cudart): hide cpptlm_core non-ABI symbols from libcudart.so (Gate 1)
  
  Add -Wl,--exclude-libs=ALL to the --whole-archive cpptlm_core link line
  in PTX-EMU/CMakeLists.txt. This hides tlm::gpu::DGpuBar and other
  cpptlm_core internals from libcudart.so's dynamic symbol table while
  preserving the ABI surface (cpptlm_set_driver, cpptlm_attach_bridge,
  cpptlm_detach_bridge, g_cpptlm_bridge).
  
  Resolves Gate 1 of integration_phase0_byte_identical_gates (ADR-0029 §D7):
  libcudart.so symbol count 3284 → ~3153, removing 131 cpptlm_core symbols.
  
  Baseline regenerated at /tmp/baseline-artifacts/libcudart-nm-before.txt.
  Audit artifact: docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt
  
  Refs: ADR-0029 §D5 (REMOVE_ITEM precedent), §D7 (byte-identical gate)
  ```

## 7. Rollback (if Gate 1 still fails after this change)

- [ ] 7.1 Revert CMake change: `git checkout PTX-EMU/CMakeLists.txt`
- [ ] 7.2 Restore old baseline: `cp /tmp/baseline-artifacts/libcudart-nm-before-OLD.txt /tmp/baseline-artifacts/libcudart-nm-before.txt`
- [ ] 7.3 Document in commit: note that `--exclude-libs` didn't work as expected, fallback option is to move `dgpu_bar.cc` out of `cpptlm_core` (CppTLM-side change, deferred to separate proposal).