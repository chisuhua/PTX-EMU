# CUDA Runtime (fake libcudart.so)

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
Fake CUDA runtime library - intercepts CUDA API calls, extracts PTX via cuobjdump.

## STRUCTURE
```
src/cudart/              # CUDA runtime replacement
include/cudart/          # Public headers
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Entry point | `cudart_sim.cpp` | `__cudaRegisterFatBinary`, `cudaLaunchKernel` |
| PTX extraction | `ptx_parser.cpp` | cuobjdump invocation |
| Kernel launch | `kernel_launch.cpp` | GPUContext dispatch |

## KEY FILES
| File | Purpose |
|------|---------|
| `cudart_sim.cpp` | Main entry - `__cudaRegisterFatBinary`, `cudaLaunchKernel` |
| `fatbin.cpp` | Fat binary handling, cuobjdump extraction |

## CONVENTIONS (this dir)
- Function signatures MUST match CUDA runtime API
- Uses `LD_PRELOAD` / `LD_LIBRARY_PATH` for interception

## ANTI-PATTERNS
- DO NOT implement actual CUDA device code - only intercept calls
- DO NOT assume thread safety without proper synchronization

## COMMANDS
```bash
cmake --build build --target cudart     # Build fake libcudart.so
```
