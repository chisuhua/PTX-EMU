# ptxir-mode4-testing

## Overview

Mode 4 tests validate the PTXIR binary serialization/deserialization roundtrip.

## Requirements

### R1: Statement Count Preservation
- serialize→deserialize MUST preserve exact statement count

### R2: Statement Type Preservation
- Each StatementContext.type MUST match after roundtrip

### R3: Branch Reconvergence Preservation
- BranchInstr.reconvergence_pc MUST be preserved (if CFG was applied before serialize)

### R4: Operand Value Preservation
- RegOperand names, ImmOperand values MUST be preserved

### R5: No ANTLR Dependency in Mode 4 Load
- deserialize_statements() MUST NOT call ANTLR parser
- load_ptxir() reads pre-serialized .ptxir file directly

## Test Structure

```
test_ptxir_serialization.cpp
├── TEST: Roundtrip preserves count
├── TEST: Roundtrip preserves types
└── TEST: Roundtrip preserves branch reconvergence
```

## Dependencies

- ptxir_writer.so (for serialize)
- ptxir_reader.so (for deserialize)
- StatementContext variant types (S_BRA, S_MOV, S_BAR, etc.)