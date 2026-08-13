# Fix Path 2D PTXIR Execution Bugs

## Why

PTX-EMU can compile an unmodified NVIDIA CUDA Sample into PTXIR, but the current path_2D image executor does not reliably execute the resulting image. Embedded PTXIR manifests may be parsed from the wrong byte range, image launches may return before the simulator completes, and global pointers produced by the fake runtime can be rejected as out of bounds. These defects block the intended transparent third-party CUDA workflow and should be fixed together because they occur in one end-to-end path.

## What Changes

- Normalize standalone PTXIR and PTXIR-Embedded inputs before manifest lookup, kernel enumeration, or named execution.
- Make `ptxemu_image_execute()` and `ptxemu_image_execute_named()` synchronous: return only after the submitted kernel has completed or report an execution error.
- Normalize absolute addresses returned from the fake runtime's `CudaDriver` global-memory pool at the verified global-memory access boundary, while preserving existing offset-based addresses and other address spaces.
- Add unit, integration, and E2E regression coverage for standalone images, embedded images, malformed images, completion semantics, and global-memory reads/writes.
- Add an end-to-end acceptance harness using unmodified NVIDIA CUDA Samples `vectorAdd.cu`, compiled with `nvcc -arch=sm_100` and executed through `libptxemu_device.so`.
- Amend ADR-0029 with the image normalization and completion invariants once implementation is verified.
- Keep `cpptlm_module.h`, `cpptlm_bridge.h`, and their ABI version values unchanged.

## Capabilities

### New Capabilities

- `path2d-ptxir-image-execution`: Defines normalized PTXIR image loading, manifest access, named kernel execution, synchronous completion, and error behavior for the path_2D image executor.
- `global-memory-address-normalization`: Defines conversion of fake-runtime global-pool absolute addresses to the address representation consumed by `SimpleMemory` global accesses.
- `cuda-samples-path2d-acceptance`: Defines the transparent third-party CUDA Sample compilation and path_2D execution acceptance workflow.

### Modified Capabilities

- `e2e-image-executor-output-correctness`: Extend the existing path_2D acceptance contract from output-format checks to actual kernel execution and correct device-memory output verification.
- `e2e-path-organized`: Add the reproducible CUDA Samples path_2D workflow and preserve path_2D test naming and isolation conventions.

## Impact

- **PTXIR image executor:** `src/cudart/cpptlm_module.cpp`, `include/cudart/cpptlm_module.h` behavior only; no public ABI signature changes.
- **PTX simulator memory path:** The actual `ld.global`/`st.global` address-normalization helper and its focused tests, after implementation-time call-path verification.
- **Tools and harness:** `tools/ptx_emu_run.cpp`, `tools/CMakeLists.txt`, and the path_2D E2E test/harness.
- **Architecture documentation:** ADR-0029 and the approved design document `docs/superpowers/specs/2026-08-13-path2d-silent-bugs-design.md`.
- **Regression surface:** Existing legacy path_1A/path_1B runtime behavior, PTXIR v1 compatibility, multi-kernel manifest behavior, and `cpptlm_bridge.h` ABI must remain unchanged.
- **Process constraints:** Follow `.opencode/skills/ptx-lessons-learned/SKILL.md`, use a baseline worktree before multi-phase implementation, keep each phase independently revertible, and run unit/integration/E2E/PTX syntax verification before completion.
