# bar-warp-sync-fix learnings

## Translation Logic (src/ptx_parser/ptx_visitor_barrier.cpp)

### Key Pattern: CTA-level vs Warp-level barrier semantics

**Multi-warp CTAs (>= 2 warps = >32 threads):**
- bar.sync MUST stay as S_BAR (CTA-level barrier)
- CTA-level forced reconvergence is required - "未汇合的 Warp 会在此被强制汇合"
- Architecture doc sm90_100.md:294 confirms this behavior

**Single-warp CTAs (<= 32 threads):**
- bar.sync translated to bar.warp.sync (OPTIMIZATION)
- Only one warp exists, so CTA-level = warp-level semantics
- bar.warp.sync is internal instruction (not real PTX ISA)

### Code Pattern
```cpp
// Translation decision based on warp count
if (openum == S_BAR && isWarpLevelBarrier(currentKernel)) {
    // Single-warp: use bar.warp.sync
} else {
    // Multi-warp: keep S_BAR
}
```

## Important: bar.warp.sync is internal only
- bar.warp.sync is NOT a real PTX instruction
- Only used internally for optimization when CTA-level = warp-level semantics
- sm90_100.md documents this as an internal translation optimization

## Verification
- `cmake --build build --target ptxsim -j4` passes
- `lsp_diagnostics` shows no errors
## Task 4: E2E Test for Multi-warp Barrier Divergence

### Test Created
- File: `tests/test_multiwarp_barrier_divergence.cpp`
- CMakeLists.txt: Added at line 135 via `add_catch_test(test_multiwarp_barrier_divergence test_multiwarp_barrier_divergence.cpp)`

### Test Design
- Mode 3C E2E test using existing `test_divergence_sync_standalone` binary
- Uses `popen()` to run binary with fake libcudart.so via LD_LIBRARY_PATH
- Verifies PASS/FAIL result output
- Architecture doc reference: sm90_100.md:294 "bar.sync — 未汇合的 Warp 会在此被强制汇合"

### Key Pattern: E2E Test Structure
```cpp
TEST_CASE("Multi-warp barrier divergence", "[e2e][barrier][divergence]") {
    std::string cmd = "PTX_LOG_LEVEL=error LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH "
                      "timeout 30 "
                      "build/bin/test_divergence_sync_standalone 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    // Read output and check for PASS/FAIL
}
```

### CMake Pattern for Catch2 Tests
```cmake
add_catch_test(test_name test_name.cpp)
```

### Binary Path Reference
- `build/bin/test_divergence_sync_standalone` - compiled from `tests/ptx/test_divergence_sync_standalone.ptx`
