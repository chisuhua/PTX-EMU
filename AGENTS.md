# PTX-EMU Agent Instructions

> **开发流程**: 本项目遵循 [文档驱动开发流程](~/.config/opencode/docs/dev-process/README.md)  
> **快速参考**: [quick-reference.md](~/.config/opencode/docs/dev-process/quick-reference.md)  
> **核心规则**: 
> - 🚫 NO IMPLEMENTATION WITHOUT APPROVED DESIGN
> - 🚫 NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST

---

## Build Commands

```bash
# Setup environment (required before building)
. env.sh

# Configure and build (Release)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Or use the build script (does env setup automatically)
./build.sh

# Build specific target
cmake --build build --target cudart
cmake --build build --target ptxsim

# Debug build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Test Commands

```bash
# Run all tests (from build directory)
cd build && ctest

# Run specific test by name
ctest -R test_memory_manager

# Run tests with labels
ctest -L mini      # Mini tests
ctest -L ptx       # PTX instruction tests

# Run with verbose output
ctest -V

# Run single benchmark test (from project root)
make -C build dummy
make -C build RAY
```

## Lint/Format

```bash
# Format code with clang-format
find src include -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

## Code Style Guidelines

### Formatting
- **Style**: LLVM-based (see `.clang-format`)
- **Indent**: 4 spaces, no tabs
- **Line limit**: 80 columns
- **Braces**: Attach style (no newline before braces)
- **Short functions**: Allow on single line
- Use `clang-format` to auto-format before committing

### Naming Conventions
- **Files**: snake_case (e.g., `ptx_parser.cpp`, `instruction_handlers.h`)
- **Functions**: camelCase for most; snake_case for PTX-specific handlers
- **Classes/Structs**: PascalCase (e.g., `GPUContext`, `ThreadContext`)
- **Variables**: camelCase (e.g., `gridDim`, `threadIdx`)
- **Constants/Enums**: UPPER_SNAKE_CASE or enum class with PascalCase
- **Member variables**: Same as variables (no special prefix)
- **PTX instructions**: lowercase (e.g., `mov`, `add`, `ld`, `st`)

### Types & Includes
- **Standard**: C++20 (CUDA code uses C++17)
- **Headers**: Use `#ifndef`/`#define`/`#endif` guards
- **Include order**:
  1. Generated ANTLR headers (if needed)
  2. Project headers (e.g., `"ptxsim/..."`)
  3. Standard library (e.g., `<vector>`, `<string>`)
- Use forward declarations when possible to reduce includes

### Error Handling
- Use assertions (`assert()`) for internal invariants
- Return error codes for recoverable errors
- Use logging macros: `PTX_ERROR()`, `PTX_WARN()`, `PTX_INFO()`
- Fatal errors: print message and exit or throw

## Project Structure

- **src/ptx_ir/**: IR types and semantic context
- **src/ptx_parser/**: ANTLR-based PTX parser (PtxListener)
- **src/ptxsim/**: Execution engine (GPU/SM/CTA/Warp/Thread context)
- **src/ptxsim/instructions/**: PTX instruction implementations
- **src/ptxsim/core/**: Core execution logic
- **src/ptxsim/memory/**: Memory abstractions
- **src/ptxsim/register/**: Register abstractions
- **src/cudart/**: CUDA runtime API replacement (fake libcudart.so)
- **src/grammar/**: ANTLR4 grammar files (ptxLexer.g4, ptxParser.g4)
- **include/**: Public headers
- **tests/**: Catch2 + CUDA PTX tests (forces PTX compilation mode)
- **bench/**: Benchmark programs
- **configs/**: GPU architecture JSON configs and debug INI files

## Key Conventions

### Adding PTX Instructions
1. Update `include/ptx_ir/ptx_op.def` (X-Macro pattern)
2. Implement handler in `src/ptxsim/instructions/`
3. Update grammar in `src/grammar/ptxParser.g4` if needed
4. Regenerate parser: `cmake --build build --target GenerateParser`

### X-Macro Pattern
```cpp
#define X(name, ...) process_##name(__VA_ARGS__);
#include "ptx_op.def"
#undef X
```

### Testing
- Tests use Catch2 framework
- Run specific test: `ctest -R test_name -V`
- Labels: `ctest -L mini` (mini tests), `ctest -L ptx` (PTX instruction tests)
- CUDA files use `.cu` extension, link against fake `libcudart.so`

### Adding CUDA API
- Add implementation in `src/cudart/` directory
- Ensure function signature matches CUDA runtime API
- Rebuild `cudart` target

## Architecture Overview

- **PTX Simulator**: C++/CUDA emulator in `src/` (ptx_ir/, ptxsim/, cudart/)
- **Parser**: ANTLR4-based (v4.13.1), grammar in `src/grammar/`
- **Execution hierarchy**: GPUContext → SMContext → CTAContext → WarpContext → ThreadContext
- **ANTLR runtime**: antlr4/antlr4-cpp-runtime-4.13.1-source

## Debugging & Logging

- Configured via INI files: `configs/config.ini` or `configs/debug_config.ini`
- Component logs: `emu`, `exec`, `mem`, `reg`, `thread`, `func`
- Log levels: trace, debug, info, warning, error, fatal
- See `docs/debugging_guide.md` for detailed setup

## Common Workflows

```bash
# Full rebuild after major changes
. env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build

# Run single test with verbose output
cd build && ctest -R test_name -V

# Rebuild specific target (faster iteration)
cmake --build build --target ptxsim

# Regenerate ANTLR parser (after grammar changes)
cmake --build build --target GenerateParser

# Run benchmark
make -C build RAY

# Run PTX syntax tests (from project root)
./tests/ptx/test_all_ptx.sh
```

## PTX Grammar Modification

**For detailed workflow**: See [docs/skills/ptx-grammar-modification.md](docs/skills/ptx-grammar-modification.md)

**Quick reference**:
1. **Before changes**: Read corresponding chapter in `docs/ptx/`
2. **Verify**: Run `./tests/ptx/test_all_ptx.sh`
3. **Debug** (if tests fail): Extract PTX with `cuobjdump`, add test case, fix parser

**Core principle**: Test-driven development. Read docs first, then fix.

## Important Files

| File | Description |
|------|-------------|
| `include/ptx_ir/ptx_op.def` | Instruction definitions (X-Macro) |
| `src/grammar/ptxLexer.g4` / `ptxParser.g4` | ANTLR grammar |
| `src/cudart/cudart_sim.cpp` | Main CUDA runtime entry point |
| `src/ptxsim/instruction_handlers.h` | Instruction handler declarations |
| `configs/*.json` | GPU architecture configs (ampere_a100.json, hopper_h100.json) |

## Reference Documentation

- `docs/gpgpu_arch.md` - GPU architecture details
- `docs/debugging_guide.md` - Debugging and logging setup
- `docs/arch.md` - System architecture
- `docs/sm90_100.md` - Hopper/Blackwell GPU specifics
- `docs/skills/ptx-grammar-modification.md` - PTX grammar modification skill (TDD workflow)



## ANTI-PATTERNS (THIS PROJECT)

### Critical Limitations (Silent Failures)
- **WMMA/Tensor Core**: Instructions are parsed but implementations are empty stubs — silently do nothing
- **Atomic operations**: No actual atomicity guarantees — stubs return immediately
- **Function calls in PTX**: Call logic not fully implemented
- **Multi-PTX cubins**: Only first PTX extracted (FIXME in ptx_parser.cpp:59)

### Architecture Constraints
- **Hopper (sm_90+) NOT supported**: Thread block cluster abstraction missing
- **Tensor Core (wmma, mma) NOT implemented**: Stubs only
- **Event/Stream APIs**: Fake implementations that log but don't synchronize

### Development Gotchas
- `assert(false)` in multiple places — crashes on unhandled code paths
- TODO/FIXME comments indicate incomplete implementations
- PTX opcode parsing in `ptx_visitor.cpp` has many unimplemented paths

### Safe Assumptions
- Basic PTX arithmetic/logic instructions work
- Memory operations (ld/st) work for global/shared/local
- Control flow (bra, ret) works
- Ampere (sm_80) and earlier architectures supported