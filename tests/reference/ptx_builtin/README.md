# PTX Builtin Semantic Reference

This directory contains the original `tests/unit/ptx/*.cu` tests that were
**moved here on 2026-06** as part of the A2 plan (see
`docs/developer-guide/KNOWN_ISSUES.md` and the plan in
`.opencode/skills/ptx-debug/`).

## What these tests do

Each test file launches real CUDA kernels using `<<<1, 1>>>` and
`cudaMalloc`/`cudaMemcpy` on the **host's actual NVIDIA GPU**. The kernels
contain `asm()` inline-PTX (e.g. `asm("add.s32 %0, %1, %2;" ...)`) wrapped
in `__device__ __forceinline__` functions.

**Net effect**: these tests verify **NVIDIA's PTX semantics** (i.e. that
the `add.s32` instruction computes `a + b`). They do **not** exercise the
PTX-EMU simulator at all.

## Why they were moved

The PTX-EMU project ships a **fake `libcudart.so`** that intercepts CUDA
runtime calls and runs PTX through the in-house simulator. The simulator
is the actual project under test — not the host GPU. So the
`tests/unit/ptx/*.cu` tests, despite living inside the project, were
**not testing the project**.

They were kept as a "semantic reference" so that when a developer writes
a new simulator-driven test (in `tests/integration/ptx/`), they can
double-check the expected numeric behavior by running the equivalent
real-GPU test.

## Current status

- **NOT built** by ctest. The corresponding `add_catch_test` entries in
  `tests/unit/CMakeLists.txt` (lines 241-281) are commented out.
- **Manually runnable** with `nvcc tests/reference/ptx_builtin/test_ptx_*.cu
  -run` if a developer wants to sanity-check semantics against real NVIDIA
  hardware.
- **To be deleted** if no one references them in 2 months (Oracle A2
  validation: "if zero reads in 2 weeks, delete and switch to A1").

## Where the simulator-driven tests live

| Real-GPU reference (this dir)         | Simulator-driven equivalent              |
|---------------------------------------|------------------------------------------|
| `test_ptx_integer.cu`                 | `tests/integration/ptx/test_integer_arith.cpp` |
| `test_ptx_ld_st.cu`                   | `tests/integration/ptx/test_ld_st_shared.cpp` |
| `test_ptx_bitwise.cu`                 | (not yet added)                          |
| `test_ptx_cvt.cu`                     | (not yet added)                          |
| `test_ptx_float.cu`                   | (not yet added)                          |
| `test_ptx_extended.cu`                | (not yet added)                          |
| `test_ptx_cvta.cu`                    | (not yet added)                          |
