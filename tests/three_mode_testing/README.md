# Three-Mode PTX Testing Framework

## Overview

This framework provides three PTX loading modes for testing the PTX-EMU simulator, enabling debugging from end-to-end to precise unit tests.

## Three Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Mode 1** | cuobjdump dynamic extraction | Integration testing, CI/CD |
| **Mode 2** | Pre-extracted PTX file | Stable reproduction, version control |
| **Mode 3** | Direct StatementContext | Unit testing, precise debugging |

## Quick Start

```bash
# Build
cmake --build build --target test_mode1 test_mode2 test_mode3

# Run all
ctest -R three_mode -V

# Run specific mode
./build/bin/tests/test_mode3
```

## Adding New Tests

### 1. Extract PTX

```bash
cuobjdump -xptx build/bin/YOUR_BINARY -arch=sm_100 \
    > tests/three_mode_testing/ptx/your_test.ptx
```

### 2. Analyze PTX

```bash
# Check structure
grep -E "bar\.|ld\.shared|st\.shared|bra" tests/three_mode_testing/ptx/your_test.ptx
```

### 3. Create Mode 3 Test

Use `test_helpers.cpp` helpers:

```cpp
std::vector<StatementContext> stmts = {
    make_mov("%r_lane", "%tid.x"),
    make_setp_lt("%p1", "%r_lane", "16"),
    make_bra_pred("L_path_b", "%p1", true),
    make_bar_sync(0),
    // ...
};
```

### 4. Update CMakeLists.txt

```cmake
add_executable(test_your_test_mode3 test_your_test_mode3.cpp ${THREE_MODE_BASE})
# ... same pattern as existing tests
```

## Helper Functions

See `test_helpers.cpp` and `SKILL.md` for complete list.

### Common Patterns

```cpp
// Barrier test
Wbar& wbar = warp.get_warp_state().wbars[0];
wbar.init(0xFFFFFFFF, reconvergence_pc);
for (int i = 0; i < 32; i++) wbar.arrive(i);
warp.set_active_mask(wbar.arrived_mask);

// Shared memory test
void* shmem = allocate_shared(32);
write_shared(shmem, lane, value);
uint32_t val = read_shared(shmem, lane);
```

## Directory Structure

```
tests/three_mode_testing/
├── CMakeLists.txt        # Build config
├── README.md             # This file
├── SKILL.md             # Skill documentation
├── test_helpers.cpp      # Common helpers
├── test_mode1.cpp       # Mode 1 template
├── test_mode2.cpp       # Mode 2 template
├── test_mode3.cpp        # Mode 3 template
├── ptx/                 # Pre-extracted PTX
│   └── *.ptx
└── golden/              # Expected outputs
    └── *.expected
```

## Debugging Workflow

```
Mode 1 (issue found)
    ↓ extract PTX
Mode 2 (stable reproduction)
    ↓ analyze structure
Mode 3 (precise unit test)
    ↓ identify root cause
Fix in source
    ↓ verify
Mode 2 (regression test)
    ↓ verify
Mode 1 (end-to-end)
```

## Skill

See `SKILL.md` for detailed guidance on generating Mode 2/3 tests from Mode 1.
