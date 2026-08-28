# fix-ptx-extraction-race — Design

## Root cause (verified by code inspection)

Two shared temporary-file races in the CUDA runtime PTX ingestion path:

**Race 1 — `src/utils/cubin_utils.cpp:127-154`** (`extract_ptx_with_cuobjdump`):
1. Builds `extract_cmd` using `cuobjdump -xptx <ptx_file> <executable_path>` (line 127).
2. Calls `system(extract_cmd)` (line 130); cuobjdump writes extracted PTX to the current working directory.
3. Computes `ptx_file_path = cwd + "/" + ptx_file` (line 136).
4. Opens the file for reading (line 137); this fails if another concurrent call has removed or clobbered it (line 139).
5. After reading, runs `rm ptx_file_path` (line 154); cleanup can race with another call using the shared workspace.

The PTX list file is also fixed at `cwd/__ptx_list_temp__` (lines 102-111), so parallel calls can overwrite or remove each other's list before it is consumed. Under parallel `ctest -j4`, this produces the "Failed to open extracted PTX file" failure.

**Race 2 — `src/cudart/cudart_sim.cpp:276-332`** (`__cudaRegisterFatBinary` preprocessing):
The input file is created with `mkstemps` (unique), but the output file was never instantiated: it remained the literal `/tmp/ptxemu_output_XXXXXX.ptx`. Every concurrent registration wrote and read the same path, so one process overwrote or removed another's preprocessed PTX. This produced the "Kernel not found: <kernel>. Available kernels:" symptom (extraction succeeded but the preprocessed content belonged to a different concurrent registration).

## Fix

1. **`cubin_utils.cpp`** — replace the shared `cwd` workspace with a unique directory created by `mkdtemp("/tmp/ptxemu-XXXXXX")`. Each call creates the directory, writes/reads the list and extracted PTX only inside it, and removes it on scope exit through an RAII cleanup guard. cuobjdump runs in a subshell `cd`'d into the private directory so the parent process cwd (and other threads) is untouched — `chdir` is process-global and would break multi-thread isolation.
2. **`cudart_sim.cpp`** — instantiate the preprocessing output file with `mkstemps` too, so each registration uses its own output path.

The cleanup is best-effort. The workspace is private to each call, so cleanup failure cannot corrupt another extraction.

## Alternatives considered

- **Global mutex on the extraction function**: rejected. A mutex is process-local and does not provide workspace isolation across test processes; it also serializes otherwise independent calls.
- **PID-suffix filenames**: rejected. It does not isolate the fixed PTX list file and still leaves cleanup and filename handling coupled to the shared directory.
- **Per-test process isolation in ctest**: rejected. It is orthogonal; the helper must be correct when called concurrently by threads or processes.

## Test strategy

1. **Unit test** (`tests/unit/cudart/test_cubin_extract_isolation.cu`): start N threads that concurrently call `extract_ptx_with_cuobjdump` on a CUDA test binary containing a real kernel; verify every call returns its own non-empty PTX.
2. **Integration test** (`tests/integration/cudart/test_parallel_cubin_extract.cu`): exercise repeated concurrent extraction against a real CUDA binary and verify all results are complete and independent.
3. **Manual verification**: run `cd build && ctest -j4` and `ctest -j1`; confirm 252/252 PASS.
