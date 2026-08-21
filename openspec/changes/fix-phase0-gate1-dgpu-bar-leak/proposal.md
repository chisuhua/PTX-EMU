## Why

`integration_phase0_byte_identical_gates` Gate 1 currently FAILs: **131 `cpptlm_core` origin symbols** (mangled names in `tlm::`/`cpptlm::`/`nlohmann::` namespaces, including the 10 `tlm::gpu::DGpuBar` members: `init/shutdown/write_reg/read_reg/vram_base/vram_size` + 4 ctors/dtors) leaked into `libcudart.so` via `-Wl,--whole-archive cpptlm_core` in `PTX-EMU/CMakeLists.txt:166-167`. The baseline `/tmp/baseline-artifacts/libcudart-nm-before.txt` (captured 2026-08-18 14:33) has 3274 symbols; current build has 3284 (131 additions, 0 missing). The leak violates ADR-0029 §D7 Gate 1 ("`nm -D --defined-only libcudart.so` 前后 diff 必须为空").

DGpuBar was added to CppTLM in commits `4277290` (2026-08-19 17:02) and `923e372` (2026-08-19 17:18) — AFTER the baseline was captured. CppTLM's `dgpu_bar.cc` is part of `cpptlm_core` (`CppTLM/src/CMakeLists.txt:47`), and `--whole-archive` forces all `cpptlm_core` objects into `libcudart.so`'s dynamic symbol table.

The 5-gate test is HARD-GATED by ADR-0029 Phase 0 acceptance. Without fixing it, no Phase 1 work (and no further Phase commits) can be merged.

## What Changes

- **Modify** `PTX-EMU/CMakeLists.txt:161-167`: add `-Wl,--exclude-libs=ALL` to the `--whole-archive cpptlm_core` line. This hides cpptlm_core's internal symbols (DGpuBar + non-ABI classes) from `libcudart.so`'s export table, while keeping the 3 PTX-EMU-owned ABI symbols (`cpptlm_attach_bridge`, `cpptlm_detach_bridge`, `g_cpptlm_bridge`) via PTX-EMU's own `PtxEmuDriverShim.cpp` (compiled into libcudart.so via `src/CMakeLists.txt:43-48` GLOB, NOT affected by `--exclude-libs`). **REVISED 2026-08-21 (Metis A1 + Oracle H1.3 Round-1)**: `cpptlm_set_driver` strong definition lives in `cpptlm_core` (the archive targeted by `--exclude-libs`); GNU ld hides archive symbols even when referenced internally. So `cpptlm_set_driver` is **expected to be hidden** from the dynamic export table. Its strong override is preserved functionally (linker binds internal calls to the strong body); the true e2e verification gate is task 4.3 (`ctest -R e2e_cosim` — 3 cosim tests must pass).
- **Regenerate** `/tmp/baseline-artifacts/libcudart-nm-before.txt` from the post-fix build.
- **Archive** the nm diff (old vs new baseline) to `docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt` as an audit artifact. The diff MUST be all `<` lines (symbols removed from old baseline; zero `>` lines indicating new symbols added).

**Non-goals (explicitly excluded)**:
- CppTLM-side changes (no moving `dgpu_bar.cc` out of `cpptlm_core`)
- Reversal of PTX-EMU ↔ CppTLM coupling direction (no `PtxEmuSubmodule`, no dlopen of `libptxemu_device.so`)
- Removal of any other bridge-related code (no `g_cpptlm_bridge` consumer cleanup, no `PtxEmuDriverShim.cpp` deletion, no test file deletions)
- ADR-0029 amendment (the byte-identical gate remains HARD)

## Capabilities

### New Capabilities

(none — this is a baseline regeneration event, not a behavior change)

### Modified Capabilities

(none — Gate 1 is a test-level contract; the `cpptlm-d1-full` spec's REQUIREMENTs do not change. The leak fix is implementation-level.)

## Impact

- **Code**: 1 CMakeLists.txt line (`PTX-EMU/CMakeLists.txt:166-167`)
- **Build artifact**: `libcudart.so.12.0` symbol count drops from 3284 → ~3153 (131 cpptlm_core symbols hidden). Lib size unchanged or slightly smaller.
- **ABI surface**: weak `cpptlm_set_driver` + `cpptlm_attach_bridge`/`detach_bridge` + `g_cpptlm_bridge` global remain exported from `libcudart.so` (the strong `cpptlm_set_driver` cross-library override is preserved because `cudart_sim.o` references it, forcing the linker to retain `ptx_emu_driver_shim.o`'s strong definition even under `--exclude-libs`). DGpuBar symbols disappear from export table.
- **Test impact**: Gate 1 transitions FAIL → PASS. All other tests (`unit`, `integration`, `e2e`, `mini`, `cute`, PTX syntax) expected unchanged — verified by full regression in `tasks.md` Phase 1.
- **Documentation**: append `docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt` (single artifact). No ADR amendment; Gate 1 §D7 contract preserved.
- **Risk**: Low. The `--exclude-libs=ALL` flag is supported since binutils 2.36 (Ubuntu 22.04+); verified compatible with the build toolchain. If a transitive symbol resolution issue arises, the fallback is to also add `-Wl,-u,cpptlm_set_driver -Wl,-u,cpptlm_attach_bridge -Wl,-u,cpptlm_detach_bridge` to force the linker to retain those specific symbols.

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性 (N/A)
- No function migration in this change. Only 1 CMake line + baseline regen.

### 状态修改 (N/A)
- No state modification. No mutex/lock/atomic changes.

### 多 Phase 推进 (N/A)
- Single-phase change (1 commit). The follow-up "complete cleanup" change is a separate proposal (deliberately split per user instruction).

### 文档同步
- [x] `docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt` (audit artifact)
- [x] No `AGENTS.md` changes (no semantic code change)
- [x] No ADR amendment (Gate 1 §D7 contract preserved as-is)

## Reference

- **PTX-EMU AGENTS.md** §STRUCTURE: `src/cudart/cudart_sim.cpp` — fake libcudart.so entry
- **ADR-0029 §D7** (lines 317-325): byte-identical fallback 5 gates — Gate 1 spec
- **ADR-0029 §D5** (line 50-54 in src/CMakeLists.txt comment): `cpptlm_module.cpp` REMOVE_ITEM precedent
- **Regression script** `scripts/regression.sh`: ctest labels unit/integration/e2e/mini/cute + `tests/ptx/test_all_ptx.sh`
- **Verification command**: `./scripts/regression.sh` (full) + `ctest --test-dir build -R integration_phase0_byte_identical_gates --output-on-failure` (Gate 1-5 all green)
- **Oracle investigation** (2026-08-21): recommends B1 (dlopen + 8-symbol ABI reuse) and C1 (DGpuBar stays in cpptlm_core, zero PTX-EMU changes). This change implements C1's effect (DGpuBar hidden from libcudart.so exports) via `--exclude-libs=ALL` without touching CppTLM.
- **Metis independent review** (2026-08-21): confirms Gate 1 baseline mechanism, surfaces 6 hidden consumer sites (generate_kernel_id, g_active_streams, g_pending_kernels_mutex, test_stream_sync_loop.cpp, test_abi_stability.cpp, Gate 4) — **all OUT of scope for this change** because they are not the root cause of the Gate 1 failure; addressed in the separate cleanup change.