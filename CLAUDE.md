# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 强制技能加载规则（必须执行）

**当检测到以下错误模式时，必须立即使用 `Skill` 工具加载对应技能，然后才能开始任何修复工作**：

| 错误模式 | 应加载技能 | 操作流程 |
|---------|-----------|---------|
| `no viable alternative`, `mismatched input`, `extraneous input`, `ANTLR` | `ptx-grammar-modification` | 1. `Skill("ptx-grammar-modification")` 2. 阅读技能文档 3. 按技能流程执行 |
| `segfault`, `SIGSEGV`, `Could not add block`, `core dumped` | `ptx-debug` | 1. `Skill("ptx-debug")` 2. 阅读技能文档 3. 按技能流程执行 |
| `测试失败`, `ctest failed`, `test failed` | `ptx-debug` | 1. `Skill("ptx-debug")` 2. 阅读技能文档 3. 按技能流程执行 |

**禁止行为**：检测到上述错误后，**禁止**直接读取代码或尝试修复。必须先加载技能。

## Build Commands

```bash
# Setup environment (required before building)
. env.sh

# Alternative: use the build script (Debug mode by default)
./build.sh

# Configure and build (Release)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Build specific targets
cmake --build build --target cudart    # Fake CUDA runtime library
cmake --build build --target ptxsim     # PTX simulation engine
cmake --build build --target ptx_parser # PTX parser

# Debug build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# RelWithDebInfo build (for step-by-step debugging)
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

### ANTLR Grammar Development
After sourcing `env.sh`, these aliases are available:
- `antlr4` — runs ANTLR4 tool (java -Xmx500M -cp "$CLASSPATH" org.antlr.v4.Tool)
- `grun` — runs TestRig for grammar debugging

## Test Commands

```bash
# Run all tests
cd build && ctest

# Run specific test by name
ctest -R test_name -V

# Run tests by label
ctest -L mini      # Mini tests
ctest -L ptx       # PTX instruction tests

# Run single benchmark
make -C build RAY
```

## Code Style

- **Style**: LLVM-based (see `.clang-format`)
- **Indent**: 4 spaces, no tabs
- **Line limit**: 80 columns
- **Files**: snake_case (e.g., `ptx_parser.cpp`)
- **Functions**: camelCase (PTX handlers use snake_case)
- **Classes/Structs**: PascalCase
- Run `clang-format -i` on modified files before committing

## Project Structure

```
PTX-EMU/
├── src/
│   ├── ptx_ir/          # IR types, statement/operand contexts
│   ├── ptx_parser/      # ANTLR4-based PTX parser (PtxVisitor)
│   ├── ptxsim/          # Execution engine
│   │   ├── core/        # GPU/SM/CTA/Warp/Thread contexts
│   │   ├── instructions/# PTX instruction implementations
│   │   ├── memory/      # Memory abstractions
│   │   └── register/    # Register abstractions
│   ├── cudart/          # Fake libcudart.so (CUDA runtime replacement)
│   └── grammar/         # ANTLR4 grammar (ptxLexer.g4, ptxParser.g4)
├── include/             # Public headers
├── tests/               # Catch2 + CUDA PTX tests
├── bench/               # Benchmark programs
├── configs/             # GPU architecture JSON + debug INI
├── antlr4/              # ANTLR runtime (4.13.1)
└── external/            # Third-party libs (json, inipp)
```

## Architecture Overview

### Execution Flow
1. `__cudaRegisterFatBinary` (in `src/cudart/cudart_sim.cpp`) → extracts PTX via `cuobjdump` → parses with ANTLR4 → fills `PtxContext`
2. `cudaLaunchKernel` → `PtxInterpreter::launchPtxInterpreter()` → builds symbol tables → submits `KernelLaunchRequest`
3. `GPUContext` dispatches to `SMContext` → builds `CTAContext` → creates `WarpContext`
4. `WarpContext` drives `ThreadContext::execute_thread_instruction()` → dispatched by `InstructionFactory`

### Execution Hierarchy
```
GPUContext (top-level, global memory, SM list)
  └── SMContext (resources, warp scheduler, barriers)
        └── CTAContext (warps, shared/local memory)
              └── WarpContext (32 threads, active mask, divergence)
                    └── ThreadContext (registers, condition codes, PC)
```

### Key Components
- **PTX Parser**: ANTLR4-based, grammar in `src/grammar/`, generated code to `build/antlr4_generated_src/`
- **Instruction Dispatch**: `InstructionFactory::initialize()` registers handlers from `include/ptx_ir/ptx_op.def` (X-macro pattern)
- **Memory Model**: `SimpleMemory` (global), `SharedMemoryManager` (per-SM), `RegisterBankManager` (per-CTA)
- **GPU Configs**: JSON-driven architecture params (`configs/ampere_a100.json`, `hopper_h100.json`, etc.)

## Common Workflows

### Adding a PTX Instruction
1. Add entry to `include/ptx_ir/ptx_op.def` using X-macro pattern
2. Implement handler in `src/ptxsim/instructions/`
3. Update grammar in `src/grammar/ptxParser.g4` if needed
4. Regenerate parser: `cmake --build build --target GenerateParser`
5. Add tests in `tests/`

### Modifying CUDA Runtime API
- Add/edit implementation in `src/cudart/`
- Ensure signature matches CUDA runtime API
- Rebuild: `cmake --build build --target cudart`

### PTXIR Serialization
- API: `include/ptxir/ptxir_serialization.h`
  - `serialize_statements(stmts, path)` / `deserialize_statements(path)` — file I/O
  - `serialize_to_string(stmts)` / `deserialize_from_string(str)` — in-memory
  - Writer/reader: `src/ptx_ir/ptxir_writer.cpp` / `ptxir_reader.cpp`
  - Format: `include/ptx_ir/ptxir_format.h`
  - Rebuild: `cmake --build build --target ptxir`
  - Tests: `test_ptxir_serialization` (10 test cases, 43 assertions)

### Build Output
- Executables: `build/bin/`
- Fake libcudart.so: symlinked to `lib/` (also at `build/lib/`)
- ANTLR generated sources: `build/antlr4_generated_src/`

### Debugging
- Configure via `configs/config.ini` or `configs/debug_config.ini`
- Log components: `emu`, `exec`, `mem`, `reg`, `thread`, `func`
- Log levels: `trace`, `debug`, `info`, `warning`, `error`, `fatal`
- See `docs/debugging_guide.md` for details

### Regenerating ANTLR Parser
```bash
cmake --build build --target GenerateParser
```

## Important Files

| File | Description |
|------|-------------|
| `include/ptx_ir/ptx_op.def` | All PTX instruction definitions (X-macro) |
| `src/grammar/ptxLexer.g4` | ANTLR lexer grammar |
| `src/grammar/ptxParser.g4` | ANTLR parser grammar |
| `src/cudart/cudart_sim.cpp` | Main CUDA runtime entry point |
| `src/ptxsim/instruction_handlers.h` | Instruction handler declarations |
| `src/ptxsim/instruction_handlers.cpp` | Instruction handler implementations |
| `include/ptxir/ptxir_serialization.h` | PTXIR binary serialization API |
| `configs/ampere_a100.json` | Default GPU architecture config |

## Known Limitations

- **WMMA/Tensor Core**: Instructions parsed but implementations are stubs
- **Atomic operations**: No actual atomicity guarantees (stubs)
- **Hopper (sm_90+)**: Thread block cluster abstraction not supported
- **Event/Stream APIs**: Fake implementations (log but don't synchronize)
- **Multi-PTX cubins**: Only first PTX extracted (see `ptx_parser.cpp`)
- `assert(false)` in unhandled code paths — crashes indicate incomplete implementation

## Reference Documentation

- `docs/gpgpu_arch.md` — GPU execution architecture
- `docs/ptx-emu_arch.md` — PTX-EMU system architecture
- `docs/debugging_guide.md` — Debugging and logging setup
- `docs/param_space_and_symbol_table.md` — Parameter space details
- `docs/sm90_100.md` — Hopper/Blackwell specifics

## Environment Dependencies

- CMake ≥ 3.15
- CUDA Toolkit (tested with 11.4+)
- GCC (tested with 10.2+)
- Java (for ANTLR)
- Make ≥ 4.3
