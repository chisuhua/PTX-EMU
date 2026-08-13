# Path 2D Baseline Reproduction (2026-08-13)

Captured from `.worktrees/baseline-fix-path2d-bugs` at commit `dc2bf6dd` (plan-done, before any path_2D implementation work).

## Environment

| Component | Value |
|---|---|
| Commit | `dc2bf6dd feat(openspec): plan-done for fix-path2d-ptxir-execution-bugs` |
| Worktree | `.worktrees/baseline-fix-path2d-bugs` (branch `main`) |
| Compiler | gcc 13.3.0 + nvcc 13.0 (sm_100) |
| Build type | Release |
| ANTLR | 4.13.2 |

## Baseline Test Status (after building all relevant test targets)

| Slice | Pass | Fail | Total |
|---|---|---|---|
| `ctest -L e2e` | 20 | 1 (`e2e_ptxir_cubin_embed` — needs `ptxir_extract` rebuilt) | 21 |
| `ctest -L cudart` | 20 | 0 | 20 |
| `ctest -L cpptlm` | 17 | 1 (`integration_cpptlm_module_inflight` — pre-existing, unrelated to path_2D) | 18 |

**Important**: `e2e_image_executor_output` PASSES, but the test only checks a baseline file format — it does not actually execute a kernel through path_2D. The "passing" test is misleading; path_2D kernel execution is untested in baseline.

## Phase 2 Reproduction (PTXIR normalization)

**VectorAdd from NVIDIA CUDA Samples** (unmodified source `cpp/0_Introduction/vectorAdd/vectorAdd.cu`, sm_100):

```bash
nvcc -arch=sm_100 -cubin vectorAdd.cu -I Common -o vectorAdd.cubin   # 5856 B
nvcc -arch=sm_100 -ptx   vectorAdd.cu -I Common -o vectorAdd.ptx       # 1324 B
./build/bin/ptxir_embed --in-cubin vectorAdd.cubin --in-ptx vectorAdd.ptx \
    --kernel-name _Z9vectorAddPKfS0_Pfi --out vectorAdd.embedded.o   # 6917 B
./build/bin/ptxir_extract --in vectorAdd.embedded.o --out-ptxir vectorAdd.ptxir  # 1049 B
```

Loaded via `baseline_repro` launcher (custom; not yet a Phase 5 deliverable):

### Standalone PTXIR (`vectorAdd.ptxir`, 1049 B)

```
[baseline_repro] load OK handle=1
[baseline_repro] module version=2
[baseline_repro] kernel_count=1
[baseline_repro] kernel_name[0]=21 '_Z9vectorAddPKfS0_Pfi'
FAIL: cudaMalloc A
```

- **kernel enumeration works** for standalone PTXIR
- **cudaMalloc fails** (returns `cudaErrorMemoryAllocation`) — separate issue: `CudaDriver::simple_memory_` is never set, so `get_global_pool()` returns nullptr (see `src/cudart/cuda_driver.cpp:89-94`)

### Embedded binary (`vectorAdd.embedded.o`, 6917 B)

```
[baseline_repro] load OK handle=1
[baseline_repro] module version=2
[baseline_repro] kernel_count=0
FAIL: kernel_count=0 (no kernels)
```

- **kernel enumeration FAILS for embedded PTXIR** — confirms Phase 2 bug
- This matches `openspec/changes/fix-path2d-ptxir-execution-bugs/proposal.md`: "Embedded PTXIR manifests may be parsed from the wrong byte range"

### Cute-rmsnorm standalone (`tests/ptxir/fixtures/cute_rmsnorm.ptxir`, 5294 B)

```
[baseline_repro] load OK handle=1
[baseline_repro] kernel_count=1
[baseline_repro] kernel_name[0]=33 '_Z14rmsnorm_kernelIfEvPKT_PS0_iif'
FAIL: cudaMalloc A
```

- Works for standalone PTXIR (already-shimmed cute_rmsnorm fixture), but cudaMalloc still fails — same root cause as above

## Reproduction launcher

`baseline_repro.cpp` in this worktree root (not a Phase 5 deliverable; kept in worktree for re-running baseline verification). To rebuild:

```bash
g++ -std=c++20 -I${CUDA_PATH}/include -I./include baseline_repro.cpp \
   -L./build/lib -lcudart -lptxemu_device \
   -Wl,-rpath,./build/lib -o baseline_repro
./baseline_repro <ptxir_or_embedded_binary>
```

## Conclusion

The Phase 2 bug ("Embedded PTXIR manifests may be parsed from the wrong byte range") is **confirmed reproducible** via the embedded vectorAdd reproduction:
- 6917-byte embedded binary → `kernel_count=0`
- 1049-byte standalone PTXIR → `kernel_count=1` ✓

Phase 3 (`wait_for_completion`) and Phase 4 (`global address normalization`) cannot be exercised in baseline because `cudaMalloc` itself fails — the CudaDriver/SimpleMemory wiring bug blocks any path_2D kernel execution. This is a **pre-existing infrastructure bug**, surfaced by this baseline reproduction, and may need to be addressed before Phase 3/4 can be properly verified.

The baseline captures:
- A direct reproduction of Phase 2's PTXIR normalization bug (kernel_count=0 for embedded)
- A new finding: `CudaDriver::simple_memory_` is never set, blocking all path_2D kernel execution
- 8 ptxemu_ ABI symbols (in `baseline-symbols.txt`)
- g_gpu_context + wait_for_completion symbol references in libcudart.so
- Test pass/fail inventory across e2e, cudart, cpptlm slices