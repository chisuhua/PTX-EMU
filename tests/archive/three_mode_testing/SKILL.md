# Three-Mode Testing Framework

## Overview

This framework provides **four PTX loading modes** for testing the PTX-EMU simulator, enabling debugging from end-to-end to precise unit tests.

| Mode | Description | Use Case |
|------|-------------|----------|
| **Mode 1** | cuobjdump dynamic extraction | End-to-end integration testing |
| **Mode 2** | Pre-extracted PTX file | Static analysis, version control |
| **Mode 3a** | StatementContext BEFORE CFG | Raw parsed state, reconvergence_pc = -1 |
| **Mode 3b** | StatementContext AFTER CFG | **Final execution version**, reconvergence_pc filled |

## Mode 3a vs Mode 3b: CFG Builder Effect

The CFG Builder (`ptx::cfg::CFGBuilder`) modifies StatementContext in-memory:
- **BranchInstr**: `reconvergence_pc` field is set (post-dominator analysis)
- **BarWarpSyncInstr**: `operands[1]` (reconvergence operand) is updated to next instruction

```
Mode 3a (BEFORE CFG):
  BranchInstr { reconvergence_pc = -1 }
  BarWarpSyncInstr { operands[1] = ? }  // original value

Mode 3b (AFTER CFG):
  BranchInstr { reconvergence_pc = 15 }  // filled by CFG builder
  BarWarpSyncInstr { operands[1] = 16 }  // updated to i+1
```

Mode 3b is what actually executes in the simulator. Mode 3a shows the raw parsed state.

## Quick Start

```bash
# Generate four-mode tests for a benchmark
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark your_kernel

# Build specific mode
cmake --build build --target test_your_kernel_mode1 test_your_kernel_mode2 test_your_kernel_mode3a test_your_kernel_mode3b

# Run all modes
ctest -R "your_kernel_mode" -V
```

## Four Modes Explained

### Mode 1: cuobjdump Dynamic Extraction

```bash
# Binary -> cuobjdump -> PTX -> simulator
cuobjdump -ptx -all -arch=sm_100 build/bin/your_kernel
```

Tests end-to-end flow: CUDA binary → PTX extraction → parsing → CFG → execution.

### Mode 2: Pre-extracted PTX File

```bash
# Load from .ptx file directly
# File: tests/three_mode_testing/ptx/your_kernel.ptx
```

Skips cuobjdump extraction, loads pre-saved PTX text for stable reproduction.

### Mode 3a: StatementContext BEFORE CFG

```cpp
// Use parsed StatementContext BEFORE CFGBuilder runs
auto stmts = parse_ptx(ptx_content);
// stmts[0].type == S_BRA
// auto& branch = std::get<BranchInstr>(stmts[0].data);
// branch.reconvergence_pc == -1  // NOT yet set
```

Raw parsed state. Useful for understanding what CFG Builder changes.

### Mode 3b: StatementContext AFTER CFG

```cpp
// Use StatementContext AFTER CFGBuilder has run
auto stmts = parse_ptx(ptx_content);
CFG cfg = CFGBuilder::build(stmts, label2pc);
PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
// ... CFG modifies stmts in-place ...
// stmts[0].type == S_BRA
// auto& branch = std::get<BranchInstr>(stmts[0].data);
// branch.reconvergence_pc == 15  // NOW filled
```

Final execution version. This is what runs in the actual simulator.

## Adding New Tests

### 1. Generate All Four Modes

```bash
python3 docs/skills/three-mode-testing/generate_tests.py \
    --benchmark your_kernel \
    --force
```

### 2. Build

```bash
cmake --build build --target test_your_kernel_mode1 \
    test_your_kernel_mode2 \
    test_your_kernel_mode3a \
    test_your_kernel_mode3b
```

### 3. Run

```bash
ctest -R "your_kernel_mode" -V
```

### 4. Compare Mode3a vs Mode3b

Mode3a and Mode3b should show the CFG builder's effect on reconvergence_pc:

```bash
# Run just mode3 tests
ctest -R "your_kernel_mode3" -V
```

## Directory Structure

```
tests/three_mode_testing/
├── CMakeLists.txt           # Auto-detects all *_mode*.cpp
├── test_helpers.hpp         # Common helpers (StatementContext construction)
...
## test_helpers.hpp Key Functions

| Function | Description |
|----------|-------------|
| `extract_ptx_cuobjdump(binary)` | Mode 1: Extract PTX using cuobjdump |
| `load_ptx_file(path)` | Mode 2: Load PTX from file |
| `parse_ptx_to_statements(ptx)` | Parse PTX to StatementContext (for Mode 3a/3b) |
| `apply_cfg_builder(stmts, label2pc)` | Run CFG builder on statements (for Mode 3b) |

## CMake Auto-Detection

CMakeLists.txt automatically detects all `*_mode*.cpp` files:

```cmake
file(GLOB THREE_MODE_SOURCES CONFIGURE_DEPENDS "*.cpp")
foreach(source IN LISTS THREE_MODE_SOURCES)
    if(basename MATCHES "_mode[0-9]+a?\\.cpp$")
        add_executable(${test_name} ${source} ${THREE_MODE_BASE})
        # ...
    endif()
endforeach()
```

No manual CMake registration needed — just add `test_foo_modeN.cpp` and reconfigure.

## Workflow: Debugging with Four Modes

```
Mode 1 (issue found in end-to-end test)
    ↓ extract PTX
Mode 2 (stable reproduction with PTX file)
    ↓ analyze structure
Mode 3a (raw parsed state, no CFG)
    ↓ understand parsing
Mode 3b (CFG-processed, final version)
    ↓ compare with 3a to see CFG effect
Fix in source
    ↓ verify
Mode 2 (regression test)
    ↓ verify
Mode 1 (end-to-end)
```

## Example: Analyzing Branch Reconvergence

```cpp
// Mode 3a: BEFORE CFG
std::vector<StatementContext> stmts = parse_ptx(ptx);
auto& bra = std::get<BranchInstr>(stmts[5].data);
INFO("Mode3a reconvergence_pc = " << bra.reconvergence_pc);  // -1

// Mode 3b: AFTER CFG
apply_cfg_builder(stmts, label2pc);
auto& bra2 = std::get<BranchInstr>(stmts[5].data);
INFO("Mode3b reconvergence_pc = " << bra2.reconvergence_pc);  // filled
```

## Skill Reference

Generator script: `docs/skills/three-mode-testing/generate_tests.py`

```bash
# All modes (default)
python3 generate_tests.py --benchmark dummy

# Specific mode only
python3 generate_tests.py --benchmark dummy --mode mode3b

# From CUDA source directly
python3 generate_tests.py --cuda-source path/to/kernel.cu

# From binary
python3 generate_tests.py --binary build/bin/kernel

# From existing PTX
python3 generate_tests.py --ptx path/to/kernel.ptx

# Dry run (show what would be generated)
python3 generate_tests.py --benchmark dummy --dry-run

# Force overwrite existing files
python3 generate_tests.py --benchmark dummy --force
```