# PTX Parser Fix - Learnings

## Task 1: Pre-validation and Error Reproduction

### Grammar Change Intent (Commit b157c55)
**Change**: Removed COMMA from `paramList` rule in `src/grammar/ptxDeclarations.g4`
- **OLD**: `LEFT_PAREN paramDecl (COMMA paramDecl)* RIGHT_PAREN`
- **NEW**: `LEFT_PAREN paramDecl (paramDecl)* RIGHT_PAREN`

**Intent**: Support parameter declarations without commas between them (space-separated only).

### Multi-Parameter PTX Functions Found
Location: `build/tests/test_ptx_bra.ptx` (and others in build/tests/)

Example format:
```ptx
.visible .entry _Z15test_bra_kernelPii(
    .param .u64 .ptr .align 1 _Z15test_bra_kernelPii_param_0,
    .param .u32 _Z15test_bra_kernelPii_param_1
)
```

PTX files with multiple parameters use commas between `.param` declarations.

### Error Reproduction
**Command**: `./build/bin/test-ptx build/tests/test_ptx_bra.ptx 2>&1`

**Error observed**:
```
line 29:4 no viable alternative at input '.visible.entry_Z15test_bra_kernelPii(.param.u64.ptr.align1_Z15test_bra_kernelPii_param_0,.param.u32_Z15test_bra_kernelPii_param_1){...'
```

**Error count**: 1 "no viable alternative" error per affected file

### Timestamp Verification
- Grammar file modified: 2026-02-25 20:51:12
- Generated parser created: 2026-02-25 20:43:37
- **Status**: Parser is STALE (generated 8 minutes BEFORE grammar change)

## Key Finding
The grammar change removes comma support from parameter lists, but existing PTX files use commas. This causes parsing failures.

## Pattern Observed
PTX parameter declarations from NVIDIA compiler use format:
```
.param <type> [modifiers] <name>,
.param <type> [modifiers] <name>
```
The comma is required by the NVIDIA toolchain output.
