## ⚠️ 2026-08-25 Archive Notice

**本 change 不再需要实施**。Gate 1 leak 已被 [`commit 09786635`](https://github.com/chisuhua/PTX-EMU/commit/09786635) (`refactor(cudart): remove cpptlm linkage + bridge files (Phase 3 of 4, libcudart.so is sync-only)`) **物理消除** —— 该 commit 及后续 4-phase refactor (`a9a14e1d` / `292022a3` / `e4d7e369` / `09786635`) 完全移除了 `cpptlm_core` 与 `cudart` 的链接。

**物理推论**: 当前 HEAD (`530bd6ca`) 的 `src/CMakeLists.txt:177` 是 `target_link_libraries(cudart ptx_ir ptx_parser ptxsim ptxir)`,**完全不含 cpptlm_core** → `nm -D --defined-only libcudart.so | grep cpptlm` 必然为空 → **Gate 1 必然 PASS**。

**真实消除路径** (post-Archive 文档归档时间序):

| Commit | 日期 | 与 Gate 1 leak 关系 |
|--------|------|--------------------|
| `87820951` | 2026-08-13 | docs(audit): 添加 HAL backend cross-repo defect audit,提出 Gate 1 修复需求(本 change 起源) |
| `587a6d5e` | 2026-08-21 | **未合并 main**: 实际实施 `--whole-archive,--exclude-libs=libcpptlm_core.a`(Round-2 方案,见 design.md Decision 1)。217 symbols 移除,3 e2e_cosim PASS. dangling state,不在任何 branch |
| `8088b24c` | 2026-08-21 | docs(openspec): commit cleanup-cudart-cpptlm-bridge-coupling + fix-phase0-gate1-dgpu-bar-leak artifacts |
| `25e36f60` | 2026-08-18 | docs(hsk-6): announce CppTLM bridge deprecation(本 change Doc2 误归因此 commit) |
| `a9a14e1d` | 2026-08 | Phase 1 of 4: delete bridge-specific test files |
| `292022a3` | 2026-08 | Phase 2a of 4: remove cpptlm bridge code paths from libcudart.so |
| `e4d7e369` | 2026-08 | Phase 2b of 4: remove cpptlm GLOBAL LD/ST bridge from memory.cpp |
| **`09786635`** | 2026-08 | **Phase 3 of 4: remove cpptlm linkage + bridge files** ← **真正的 Gate 1 leak 物理消除** |
| `d281a21e` | 2026-08 | HSK-8 Phase 2: add ptxemu_core library (replace cpptlm bridge with IPtxEmuDevice) |
| `c225780e` | 2026-08 | HSK-8 Phase 3: PROJECT_IS_TOP_LEVEL isolation |
| `738b412c` | 2026-08 | HSK-8 ack |
| `530bd6ca` | 2026-08-24 | current HEAD: archive ptxemu-public-device-api |

**关键认知**: `--exclude-libs=ALL` 方案是"妥协式"修复(保留 cpptlm 链接但隐藏符号);4-phase refactor 是"根治式"修复(直接移除 cpptlm 链接,根本不需要隐藏符号)。后者的优势:

1. **零误伤**: 不依赖 GNU ld `--exclude-libs` 语义的精确理解(`587a6d5e` commit message 实验验证 `--exclude-libs=ALL` 会隐藏 4051 ANTLR4 符号)
2. **简化编译图**: `cudart` 不再依赖 `cpptlm_core`,维护性更好
3. **HSK-8 自然结果**: HSK-8 Phase 2 `ptxemu_core` 替代 cpptlm bridge,导致 cpptlm 链接彻底无意义

**后续工作** (本 change 归档后):

1. **`cleanup-cudart-cpptlm-bridge-coupling`** (下游 chain,58 tasks): 58 tasks 中哪些仍 relevant? 需 audit,因为 4-phase refactor 已完成大部分目标
2. **HSK-8 Phase 2.2/2.3 delegation**: 下一个真 actionable 工作(per `2026-08-24-hsk8-followup-task-path.md`)
3. **CppTLM 端**: `505333b` 已完成,等待 PTX-EMU 推送 HSK-8 follow-up commits 后再 bump(不要再等本 change)

详细分析见 [`docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`](../../../../../docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md)。

---

## Why (原始描述,保留作历史记录)

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