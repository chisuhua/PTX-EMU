# Technical Design: Path 2D PTXIR Execution Bug Fixes

## Context

### 现状问题

The path_2D image executor in `src/cudart/cpptlm_module.cpp` is intended to expose PTXIR images through `libptxemu_device.so`. The current end-to-end NVIDIA CUDA Samples reproduction exposed three separate failures in the same execution path:

1. Image bytes may be a standalone PTXIR image or a PTXIR-Embedded binary. Manifest readers currently need a normalized standalone PTXIR view, but embedded callers can pass the complete host binary.
2. `PtxInterpreter::launchPtxInterpreter()` submits a `KernelLaunchRequest`; the path_2D caller must also drive the shared `GPUContext` until completion. The existing legacy runtime path does this through `cudaLaunchKernel` and `wait_for_completion()`.
3. After execution is actually driven, `ld.global`/`st.global` can reject addresses returned by fake-runtime `cudaMalloc`. The observed address is an absolute address inside the `SimpleMemory` mmap pool, while one access boundary expects a pool-relative representation.

### 目标状态

- All manifest and kernel-selection operations accept both standalone PTXIR and PTXIR-Embedded images.
- `ptxemu_image_execute()` and `ptxemu_image_execute_named()` return only after the kernel has completed or an error is reported.
- Global-memory reads and writes accept the simulator's canonical pool address representation without weakening bounds checks.
- An unmodified NVIDIA CUDA Samples `vectorAdd.cu` can be compiled for `sm_100`, loaded through path_2D, and produce correct output.
- Existing ABI and legacy path behavior remain unchanged.

### Constraints

- `include/cudart/cpptlm_module.h` and `include/cudart/cpptlm_bridge.h` are ABI contracts and must not change.
- PTXIR v1 standalone images and v2 multi-kernel manifests remain backward compatible.
- The path_2D executor is subject to ADR-0029's `[SINGLE-GPU-INSTANCE]` assumptions and existing execution mutex.
- The implementation must follow `.opencode/skills/ptx-lessons-learned/SKILL.md`, including baseline comparison, phase isolation, and explicit regression tests.

## Goals / Non-Goals

**Goals:**

- Normalize PTXIR input at the image-executor boundary.
- Make path_2D execution completion semantics explicit and testable.
- Fix the verified global-memory address mismatch with one centralized helper at the actual access boundary.
- Add regression tests for standalone/embedded images, completion, invalid images, and global-memory reads/writes.
- Add a reproducible NVIDIA CUDA Samples path_2D acceptance harness.
- Amend ADR-0029 and synchronize OpenSpec/documentation artifacts.

**Non-Goals:**

- No modification to NVIDIA CUDA Samples source.
- No fork or replacement of `nvcc`.
- No public ABI or version bump.
- No general rewrite of `CudaDriver`, `SimpleMemory`, or all PTX address spaces.
- No promise that all CUDA Samples APIs or PTX instructions are supported.

## Decisions

### Decision 1: Normalize image bytes in `PtxEmuImageExecutor`

Add a private helper in `src/cudart/cpptlm_module.cpp` that returns an owned standalone PTXIR representation:

```text
if standalone PTXIR magic: return a safe view/copy of the original bytes
if embedded footer: PTXIRLoader::extractPTXIR() and return the extracted section
otherwise: return an invalid-image result
```

Use this helper from `get_kernel_name`, `kernel_count`, `kernel_name_at`, `execute`, and `execute_named` before calling `read_manifest_from_ptxir_section` or PTXIR deserialization.

**Rationale:** The executor already owns a deep copy of the caller image. Normalization at this boundary keeps format classification and ownership in one component, avoids changing `read_manifest_from_ptxir_section` semantics for unrelated legacy callers, and preserves both input formats.

**Rejected alternative:** Teach every low-level manifest reader to detect every image format. That spreads image-format policy into readers and risks changing legacy behavior.

### Decision 2: Use the existing synchronous completion primitive

After submitting a request from `execute` or `execute_named`, call `g_gpu_context->wait_for_completion()` while `exec_mu_` remains held. Return an error if the shared context is unavailable.

**Rationale:** `GPUContext::wait_for_completion()` is the existing owner of the `exe_once()` loop and completion callbacks. Reusing it avoids duplicating scheduler-driving logic in the ABI layer and ensures temporary parameter/global/local allocations are released before the API returns.

**Rejected alternative:** Add a new public `ptxemu_image_synchronize` ABI function and make callers synchronize separately. That would preserve the current silent-success behavior and create a new ordering contract for every consumer.

**Concurrency invariant:** The existing executor mutex serializes launches. The change must not release the mutex while the request is executing, because the simulator uses process-global GPU state.

### Decision 3: Normalize absolute global-pool addresses at one verified memory boundary

Before implementation, trace the actual call path from `ld.global` and `st.global` through their handler and memory helper. Add one helper at the narrowest common boundary:

```text
pool_base = CudaDriver::instance().get_global_pool()
pool_size = CudaDriver::instance().get_global_size()
if pool_base <= address < pool_base + pool_size:
    address = address - pool_base
return address
```

The helper must preserve existing pool-relative addresses and leave non-global address spaces unchanged. Existing `SimpleMemory` validation remains authoritative after normalization.

**Rationale:** The observed failure is an address-representation mismatch, not evidence that bounds checking should be disabled. Centralizing the conversion prevents divergent fixes in individual instruction handlers and keeps rejection behavior for invalid addresses.

**Rejected alternative:** Pass raw host pointers from the launcher. That bypasses the simulator's device-memory contract and would not fix third-party runtime programs using `cudaMalloc`.

**Implementation guard:** The exact file and helper name remain implementation decisions until the call-path trace identifies the shared boundary. The task must not guess a source location from the PTX mnemonic alone.

### Decision 4: Preserve the public ABI and legacy path

No changes are made to `cpptlm_module.h`, `cpptlm_bridge.h`, `CPPTLM_MODULE_VERSION`, or `CPPTLMBRIDGE_VERSION`. The fix is internal to the image executor and memory access implementation.

The legacy path_1A/path_1B byte-identical gates and existing PTXIR compatibility tests remain mandatory regression gates.

### Decision 5: Use a real third-party kernel as the acceptance oracle

The E2E fixture will compile the unmodified NVIDIA CUDA Samples `cpp/0_Introduction/vectorAdd/vectorAdd.cu` with `nvcc -arch=sm_100`, extract PTX, produce a PTXIR image, load it through `libptxemu_device.so`, execute the mangled kernel, copy back the output, and verify:

```text
abs(A[i] + B[i] - C[i]) < 1e-5 for every i
```

This test proves the full path and catches parser, manifest, parameter-space, address-space, and scheduler issues that isolated unit tests cannot expose.

## 影响范围

| 组件 | 影响类型 | 说明 |
|---|---|---|
| `src/cudart/cpptlm_module.cpp` | 修改 | PTXIR normalization and synchronous completion in image executor |
| PTX global-memory access boundary | 修改 | Centralized absolute-pool-address normalization after call-path verification |
| `include/cudart/cpptlm_module.h` | 明确不修改 | Public image-executor ABI remains unchanged |
| `include/cudart/cpptlm_bridge.h` | 明确不修改 | CppTLM ABI remains unchanged |
| `tests/unit/cudart/` | 新增/修改 | Standalone/embedded manifest and invalid-image tests |
| `tests/e2e/path_2D_image_executor/` | 修改 | Completion and output correctness coverage |
| `tests/e2e/cuda_samples/` or approved harness location | 新增 | Unmodified CUDA Samples vectorAdd acceptance |
| `docs/adr/ADR-0029-ptxemu-image-executor.md` | 修改 | Document completion and normalized-image invariants |
| `openspec/changes/fix-path2d-ptxir-execution-bugs/` | 新增 | Proposal, design, specs, and TDD tasks |

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| `wait_for_completion()` can expose an existing non-terminating kernel or scheduler bug | Add a bounded E2E fixture, preserve existing cycle ceiling policy where applicable, and fail with an explicit completion error rather than silently returning success. |
| Embedded-image extraction changes manifest parsing behavior | Add standalone, embedded, malformed-footer, and manifest-mismatch tests; preserve raw image ownership separately from normalized parsing bytes. |
| The global-memory access boundary is misidentified | Make call-path tracing the first implementation task; add a focused regression test at the selected helper and verify both `ld.global` and `st.global`. |
| Absolute-address normalization is applied to an unrelated address space | Keep the helper at the common global-access boundary and assert shared/local/param tests remain unchanged. |
| Existing path_1A/path_1B behavior regresses | Run legacy E2E tests, `dummy-add`, ABI symbol checks, and PTX syntax tests after every phase. |
| Multi-kernel selection regresses | Test both the v1 synthesized manifest entry and v2 named-kernel selection. |
| The CUDA Samples build depends on toolkit-specific headers or helper paths | Keep the sample source unmodified, pass only include/toolchain configuration from the harness, and report unsupported external dependencies explicitly. |

## Migration Plan

### Phase 0: Baseline and reproduction

1. Create a baseline worktree from the current HEAD.
2. Run the existing relevant unit/integration/E2E tests.
3. Preserve the failing CUDA Samples path_2D reproduction and record the exact error/output.
4. Verify current symbols and ABI versions.

If baseline and working-tree behavior differ, stop and resolve the discrepancy before editing.

### Phase 1: PTXIR normalization

1. Write failing tests for standalone and embedded manifest access.
2. Trace all current `read_manifest_from_ptxir_section` callers in the image executor.
3. Add the private normalization helper.
4. Route all image-executor manifest/deserialization operations through it.
5. Run focused tests and legacy PTXIR tests.

This phase is independently revertible.

### Phase 2: Synchronous path_2D completion

1. Write a failing test proving image execution returns only after a store kernel completes.
2. Add the existing `GPUContext::wait_for_completion()` call to both image execute methods, under the existing execution mutex.
3. Add explicit missing-context error handling.
4. Run path_2D and legacy runtime tests.

This phase is independently revertible.

### Phase 3: Global-memory address normalization

1. Trace the `ld.global`/`st.global` path and identify the shared access boundary.
2. Write failing tests for absolute pool addresses, relative offsets, and invalid addresses.
3. Implement the centralized conversion helper.
4. Run the vectorAdd path_2D acceptance test and memory regression suite.

This phase is independently revertible. If the address model reveals a broader architectural mismatch, stop and revise this design instead of broadening the patch opportunistically.

### Phase 4: Documentation and governance

1. Amend ADR-0029 with the verified completion and image-normalization invariants.
2. Update OpenSpec task status and affected capability specs.
3. Synchronize the approved design document and user-facing documentation.
4. Run the full verification matrix.

### Rollback

Each phase must be committed separately. If an existing test regresses, revert only the current phase before investigating the next phase. Do not mix a memory-model redesign into the PTXIR or completion phases.

## Design-Time Checklist

- Baseline worktree and full relevant test status are required before implementation.
- No ThreadContext/WarpState migration is planned; therefore no state-translation migration is introduced. The affected global simulator state is `GPUContext` task scheduling and the `[SINGLE-GPU-INSTANCE]` invariant, which must be tested through path_2D E2E coverage.
- No public ABI signature changes are planned.
- All OpenSpec artifacts must remain tracked before implementation commits.
- ADR-0029 is currently the relevant architectural source; do not amend an archived change. If related work is archived before implementation begins, create this new `fix-*` change with an explicit archive reference.
- Required lessons reference: `.opencode/skills/ptx-lessons-learned/SKILL.md`, especially the phase isolation, baseline worktree, byte-identical fallback, and OpenSpec artifact checklists.

## Open Questions

1. Which concrete source/helper is the narrowest common boundary for `ld.global` and `st.global` address normalization? Resolve by call-path tracing before Phase 3 implementation.
2. What bounded completion/error contract should path_2D use if a kernel never reaches `EXIT`? Align with existing `PTX_EMU_MAX_ADVANCE_CYCLES` and CppTLM advance semantics rather than inventing a second unrelated limit.
