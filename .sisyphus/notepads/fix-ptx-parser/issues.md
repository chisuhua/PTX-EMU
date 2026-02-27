# PTX Parser Fix - Issues

## Issue 1: Grammar Change Breaks Existing PTX Parsing
**Status**: Confirmed

**Description**: Commit b157c55 removed COMMA from `paramList` grammar rule, causing parsing failures for PTX files with multiple parameters separated by commas.

**Affected Files**: 
- `build/tests/test_ptx_bra.ptx`
- `build/tests/ptx_cvta.ptx`
- `build/tests/ptx_ld_st.ptx`
- And likely all multi-parameter kernel files

**Error Message**:
```
no viable alternative at input '.visible.entry_...(.param..._param_0,.param..._param_1)...'
```

**Impact**: All PTX files with multiple kernel parameters fail to parse.

## Issue 2: Generated Parser is Stale
**Status**: Confirmed

**Description**: The generated ANTLR parser files were created at 20:43:37, but the grammar was modified at 20:51:12 (8 minutes later).

**Parser files location**: `build/antlr4_generated_src/`
- ptxParser.cpp (722KB)
- ptxLexer.cpp (90KB)

**Impact**: Even if grammar is fixed, parser needs regeneration.
