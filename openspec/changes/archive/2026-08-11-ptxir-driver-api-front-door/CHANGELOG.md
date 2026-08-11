# ptxir-driver-api-front-door — Ship Log

Phase 12.3.A complete. libcudart.so now exports 4 CUDA Driver API T symbols:
- `cuModuleLoadData(CUmodule*, const void*)` — load PTXIR module from memory
- `cuModuleGetFunction(CUfunction*, CUmodule, const char*)` — resolve kernel symbol
- `cuLaunchKernel(CUfunction, ...)` — launch kernel via Driver API
- `cuModuleUnload(CUmodule)` — unload module + invalidate child handles

## Implementation
- 6 production files: include/cudart/module_registry.{h,cpp},
  include/cudart/image_classifier.h, src/cudart/image_classifier.cpp,
  cudart_sim.cpp (4 new entries + stub replacement)
- 9 test files: 6 unit + 3 integration suites (28 tests total)
- 6 commits on openspec/ptxir-driver-api-front-door branch

## Oracle Review Conditions (proposal.md)
- C1 lock order: ✓ ModuleRegistry::mutex → per-PtxContext
- C2 6-class image classifier: ✓ unit_image_classifier (6 tests)
- C3 D3 mutation regression: ✓ integration_in_memory_mutation (3 tests)
- C4 single deserialize path: ✓ cudart::PTXIRLoader::deserializeForCubin
- C5 follow-up task: → cuInit / cuCtx* / packed-extra (deferred)
- C7 ABI stability: ✓ integration_abi_stability (3 tests, CPPTLMBRIDGE_VERSION=2)

## Constraints Honored
- cpptlm_bridge.h ABI byte-identical (CPPTLMBRIDGE_VERSION=2, no edits)
- libptxemu_device.so 5 ABI unchanged
- No new WarpContext/ThreadContext/GPUContext dependencies
- Reuses cudart::PTXIRLoader::deserializeForCubin (single deserialization point)
- DL-isolated: PTXIR_MODE/off independence preserved
