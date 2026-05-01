# PTX-EMU Test Suite

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
Catch2-based test suite for PTX simulation, instructions, and memory/register subsystems.

## STRUCTURE
```
tests/
├── instructions/        # PTX instruction tests
├── ptx/                # PTX syntax tests (test_all_ptx.sh)
├── warp/               # Warp-level tests
├── three_mode_testing/ # Golden reference tests
├── catch_amalgamated.cpp/hpp  # Catch2 test framework
└── *.cpp               # Unit/integration tests
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| PTX syntax tests | `tests/ptx/test_all_ptx.sh` | Full PTX ISA coverage |
| Instruction tests | `tests/instructions/` | Per-instruction tests |
| Memory tests | `test_memory_manager.cpp` | Memory abstraction |
| Barrier tests | `test_barrier_*.cpp` | Synchronization |

## TEST FRAMEWORK
- **Framework**: Catch2 (`catch_amalgamated.hpp`)
- **CUDA tests**: Compiled with `-keep` flag to preserve PTX
- **Architecture**: `sm_100` (virtual)

## CONVENTIONS (this dir)
- Test files: `test_*.cpp` or `ptx_*.cu`
- Use `ctest -R <name>` to run specific tests
- PTX syntax tests: `./tests/ptx/test_all_ptx.sh` (NOT ctest)

## COMMANDS
```bash
cd build && ctest                      # Run all tests
ctest -R test_memory_manager -V       # Run specific test
./tests/ptx/test_all_ptx.sh           # PTX syntax tests (CRITICAL)
```

## ANTI-PATTERNS
- DO NOT use `ctest` for PTX syntax tests - use `test_all_ptx.sh`
- DO NOT commit test changes without verifying all pass
