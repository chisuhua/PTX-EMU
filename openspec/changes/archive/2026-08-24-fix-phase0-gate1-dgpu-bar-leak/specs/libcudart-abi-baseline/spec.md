## ADDED Requirements

> **2026-08-25 Archive Notice**: 原 spec 描述 `--exclude-libs=ALL` 方案。实际实施路径是 4-phase refactor `commit 09786635` 完全移除 `cpptlm_core` 链接。本 spec 已更新以反映**实际达成的契约**—— `libcudart.so` 不应链接 cpptlm_core,从而根除 Gate 1 leak 可能性。

### Requirement: `libcudart.so` does not link `cpptlm_core` static library

The system SHALL NOT link `cpptlm_core` (a CppTLM static lib) into `libcudart.so` via `target_link_libraries(cudart ...)`. The co-simulation link between PTX-EMU and CppTLM is mediated by `ptxemu_core` (PTX-EMU-owned static lib, `HSK-8 Phase 2`) and `IPtxEmuDevice` interface, NOT by direct cpptlm_core linkage. This is enforced by:

- `src/CMakeLists.txt:177` line: `target_link_libraries(cudart ptx_ir ptx_parser ptxsim ptxir)` (no `cpptlm_core`)
- Top-level `CMakeLists.txt` does NOT call `add_subdirectory(CppTLM)` for cpptlm_core
- HSK-6 docs commit `25e36f60` froze `CPPTLMBRIDGE_VERSION=2`; CppTLM-side `abi_guards.h` provides 17 `static_assert` verifications

#### Scenario: Physical elimination of Gate 1 leak

- **WHEN** `nm -D --defined-only build/lib/libcudart.so.12.0` is run on any build of `530bd6ca` HEAD or later
- **THEN** the output contains ZERO symbols with mangled names in `tlm::`/`cpptlm::`/`nlohmann::` namespaces
- **AND** specifically zero `tlm::gpu::DGpuBar::*` symbols (10 members)
- **AND** zero `_ZN13DynamicLoader*`, `_ZN12PluginLoader*`, `_ZN11CrossbarTLM*`, `_ZN13ModuleFactory*`, `_Z22get_default_port_specs*` etc.

#### Scenario: HSK-8 public device API integration replaces cpptlm bridge

- **WHEN** `libptxemu_device.so.12.0` is loaded by CppTLM via `dlopen`
- **THEN** `ptxemu_image_load` / `ptxemu_image_execute` / `ptxemu_image_unload` ABI symbols are resolved
- **AND** 5 `extern "C"` ABI entry points (`ptxemu_image_load` / `execute` / `unload` / `kernel_name` / `module_version`) are exported from `libptxemu_device.so.12.0`
- **AND** 12 pure virtual methods of `IPtxEmuDevice` interface are available to CppTLM (`HSK-8 spec §3`)
- **AND** `PTXEMU_API_VERSION=1` is statically asserted (`include/ptxemu/device_api.h:117`)

### Requirement: Gate 1 byte-identical fallback PASSES by structural elimination

The system SHALL pass `integration_phase0_byte_identical_gates` Gate 1 by virtue of having ZERO cpptlm_core symbols to leak (rather than by hiding them via `--exclude-libs`). This is a stronger contract than `--exclude-libs` because:

- It eliminates the `--exclude-libs` GNU ld version dependency (binutils ≥ 2.36)
- It eliminates the risk of accidentally hiding PTX-EMU-owned symbols via `--exclude-libs=ALL` (documented in `587a6d5e` commit message: would hide 4051 ANTLR4 symbols)
- It eliminates the need for `--whole-archive cpptlm_core` in `target_link_libraries(cudart ...)`

#### Scenario: 5-gate test passes post-elimination

- **WHEN** `ctest --test-dir build -R integration_phase0_byte_identical_gates --output-on-failure` is run
- **THEN** Gate 1 (`nm -D --defined-only libcudart.so symbol surface`) PASSES (structural elimination guarantees this)
- **AND** Gate 2 (SONAME `libcudart.so.12.0`) PASSES
- **AND** Gate 3 (symlinks: `lib/libcudart.so.12.0` → `libptxemu_device.so.12.0`) PASSES
- **AND** Gate 4 (`g_cpptlm_bridge == nullptr` default) PASSES (linkage removal makes this vacuously true)
- **AND** Gate 5 (`get_gpu_clock_from_context` logger→g_gpu_context) PASSES

#### Scenario: Co-simulation preservation via new ABI

- **WHEN** `tests/integration/test_ptxemu_device_api_integration.cpp` runs (new test in HSK-8 Phase 2 verification)
- **THEN** `IPtxEmuDevice` methods (`load_ptxir` / `execute` / `query_state` etc.) respond correctly to CppTLM-side stub driver
- **AND** no `tests/e2e/cosim/*` tests exist (deleted in `commit a9a14e1d chore(tests): delete bridge-specific test files (Phase 1 of 4)`); co-simulation is now tested at `tests/integration/` level via `IPtxEmuDevice` interface