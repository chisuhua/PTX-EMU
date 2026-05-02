# four-mode-flow-testing

## Overview

End-to-end pipeline tests showing the relationship between Mode 1 (extract), Mode 2 (parse), Mode 3 (StatementContext), and Mode 4 (PTXIR binary).

## Requirements

### R1: Mode 1 → Mode 2 Consistency
- PTX extracted by cuobjdump (Mode 1) MUST match content loaded from file (Mode 2)

### R2: Mode 2 → Mode 3 Consistency
- load_ptx_file() MUST produce StatementContexts equivalent to what PTX parser produces

### R3: Mode 3 → Mode 4 Serialization
- serialize_statements() MUST produce valid .ptxir file from StatementContext vector

### R4: Mode 4 → Mode 3 Deserialization
- deserialize_statements() MUST produce StatementContexts identical to what was serialized

### R5: Pipeline Completeness
- Full pipeline (Mode 1 → 2 → 3 → 4) MUST preserve instruction semantics end-to-end

## Test Structure

```
test_four_mode_flow.cpp
├── TEST: Mode 1 PTX extraction matches Mode 2 file content
├── TEST: Mode 2 parsed statements match Mode 3 hand-written
├── TEST: Mode 3 StatementContext serializes to valid .ptxir
├── TEST: Mode 4 deserialization produces identical StatementContexts
└── TEST: Full pipeline Mode 1→2→3→4 preserves semantics
```

## Dependencies

- All four modes operational
- test_helpers.hpp functions: extract_ptx_cuobjdump, load_ptx_file, load_ptx_statements, serialize_statements, deserialize_statements