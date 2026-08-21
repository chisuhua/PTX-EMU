## ADDED Requirements

### Requirement: `libcudart.so` Phase 0 byte-identical baseline regeneration event

The system SHALL maintain `nm -D --defined-only libcudart.so` byte-identical equality with `/tmp/baseline-artifacts/libcudart-nm-before.txt`, regenerated on 2026-08-21 to remove the 131 cpptlm_core symbols (including 10 `tlm::gpu::DGpuBar`) that leaked via `-Wl,--whole-archive cpptlm_core`. The baseline regeneration MUST be audited by archiving the nm diff to `docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt`. The `kAllowedAdditions` set in `tests/integration/test_phase0_byte_identical_gates.cpp` SHALL remain empty.

#### Scenario: Baseline regeneration audit passes

- **WHEN** `diff /tmp/baseline-artifacts/libcudart-nm-before-OLD.txt /tmp/baseline-artifacts/libcudart-nm-before.txt` is run
- **THEN** the diff contains ONLY `<` lines (removed symbols), ALL of which are cpptlm_core origin symbols (tlm::/cpptlm::/nlohmann:: mangled names)
- **AND** zero `>` lines (no symbols added that were absent from old baseline)

#### Scenario: Gate 1 transitions FAIL to PASS

- **WHEN** `ctest --test-dir build -R integration_phase0_byte_identical_gates` is run after the regeneration
- **THEN** Gate 1 (`nm -D --defined-only libcudart.so symbol surface unchanged`) PASSES
- **AND** Gate 2-5 continue to PASS (SONAME, symlinks, `g_cpptlm_bridge == nullptr`, logger→g_gpu_context)

#### Scenario: ABI surface preservation

- **WHEN** `nm -D build/lib/libcudart.so | grep -E "cpptlm_set_driver|cpptlm_attach_bridge|cpptlm_detach_bridge|g_cpptlm_bridge"` is run after the fix
- **THEN** 3 of 4 symbols are present in the output (`cpptlm_attach_bridge`, `cpptlm_detach_bridge`, `g_cpptlm_bridge` as `T` or `B`)
- **AND** `cpptlm_set_driver` is **expected to be hidden** (not in `nm` output) per `--exclude-libs=ALL` semantics — its strong definition lives in `cpptlm_core` archive which `--exclude-libs` hides from the dynamic export table
- **AND** the strong override is proven functionally by task 4.3: `ctest --test-dir build -R 'e2e_cosim'` SHALL PASS for all 3 e2e cosim tests (`e2e_cosim_vector_add`, `e2e_cosim_infinite_loop_ceiling`, `e2e_cosim_multi_kernel_drain`), proving `cpptlm_set_driver` strong override still happens at link time

#### Scenario: Co-simulation preservation

- **WHEN** `ctest --test-dir build -R 'e2e_cosim'` is run after the fix
- **THEN** all 3 e2e cosim tests (`e2e_cosim_vector_add`, `e2e_cosim_infinite_loop_ceiling`, `e2e_cosim_multi_kernel_drain`) PASS
- **AND** proves the strong `cpptlm_set_driver` override still happens at link time despite `--exclude-libs`

### Requirement: `libcudart.so` ABI surface unchanged for `--exclude-libs` scope

The system SHALL hide from `libcudart.so`'s dynamic symbol table only those symbols originating from `cpptlm_core` (a static lib) and not referenced by PTX-EMU's own source. The 3 PTX-EMU-owned ABI symbols `cpptlm_attach_bridge` / `cpptlm_detach_bridge` / `g_cpptlm_bridge` (defined in `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp`, compiled into libcudart.so via `src/CMakeLists.txt:43-48` GLOB) MUST remain exported. The `cpptlm_set_driver` strong definition (in `cpptlm_core` archive) is expected to be hidden from the dynamic export table per `--exclude-libs=ALL` semantics — internal symbol resolution is preserved by `--whole-archive`, but external visibility is hidden. The functional strong override is verified by e2e cosim tests (task 4.3), not by `nm` output. Implementation mechanism: `-Wl,--exclude-libs=ALL` combined with `-Wl,--whole-archive` for `cpptlm_core` only.

#### Scenario: Hidden symbols are exclusively from cpptlm_core

- **WHEN** `nm -D --defined-only build/lib/libcudart.so` is run after the fix
- **THEN** symbols with mangled names containing `tlm::gpu::` namespace appear ONLY for PTX-EMU-required ABI surfaces (e.g., `tlm::gpu::*` types used by `PtxEmuDriverShim` if any)
- **AND** no `tlm::gpu::DGpuBar::*` symbols appear (131 cpptlm_core symbols removed vs old baseline, including all 10 DGpuBar members)

#### Scenario: PTX-EMU-defined ABI surfaces preserved

- **WHEN** `nm -D build/lib/libcudart.so` is run after the fix
- **THEN** `cpptlm_set_driver` symbol is **NOT** present in `nm -D` output (hidden by `--exclude-libs=ALL`); functional verification via task 4.3 e2e_cosim tests PASSES
- **AND** `cpptlm_attach_bridge`, `cpptlm_detach_bridge` symbols are present
- **AND** `g_cpptlm_bridge` symbol is present (as `B` BSS, since it's a global pointer with default value `nullptr`)