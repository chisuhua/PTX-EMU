# PTX-EMU Agent Instructions

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
# Or: make test

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

### Project Structure
- **src/ptx_ir/**: IR types and semantic context
- **src/ptx_parser/**: ANTLR-based PTX parser (PtxListener)
- **src/ptxsim/**: Execution engine (GPU/SM/CTA/Warp/Thread context)
- **src/ptxsim/instructions/**: PTX instruction implementations
- **src/cudart/**: CUDA runtime API replacement (fake libcudart.so)
- **src/memory/**: Memory abstractions
- **src/register/**: Register abstractions
- **include/**: Public headers
- **tests/**: Catch2 + CUDA PTX tests
- **bench/**: Benchmark programs
- **configs/**: GPU architecture JSON configs

## Key Conventions

### Adding PTX Instructions
1. Update `include/ptx_ir/ptx_op.def` (X-Macro pattern)
2. Implement handler in `src/ptxsim/instructions/`
3. Update grammar in `src/grammar/ptxParser.g4` if needed
4. Regenerate parser: `cmake --build build --target GenerateParser`

### Adding Qualifiers
1. Update `include/ptx_ir/ptx_qualifier.def`
2. Used to generate `Qualifier` enum in `ptx_types.h`

### X-Macro Pattern
The project uses X-Macros for code generation:
```cpp
#define X(name, ...) process_##name(__VA_ARGS__);
#include "ptx_op.def"
#undef X
```

### Testing CUDA Code
- Tests use Catch2 framework (`catch_amalgamated.cpp`)
- CUDA files use `.cu` extension and compile to PTX
- Tests link against fake `libcudart.so` (built as `cudart` target)
- Set `LD_LIBRARY_PATH` to include project `lib/` directory

### GPU Architecture
- GPU configs in `configs/*.json` (e.g., `ampere_a100.json`)
- Debug/logging controlled via `configs/config.ini`
- See `docs/gpgpu_arch.md` for architecture details

## Important Files

- `include/ptx_ir/ptx_op.def` - Instruction definitions (X-Macro)
- `include/ptx_ir/ptx_qualifier.def` - Qualifier definitions
- `src/grammar/ptxLexer.g4` / `ptxParser.g4` - ANTLR grammar
- `src/cudart/cudart_sim.cpp` - Main CUDA runtime entry point
- `src/ptxsim/instruction_handlers.h` - Instruction handler declarations
- `docs/debugging_guide.md` - Debugging and logging setup
