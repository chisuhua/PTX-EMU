---
name: modifying-ptx-grammar
description: Use when modifying PTX grammar files (.g4) in the PTX-EMU project — updating syntax rules, adding new PTX instructions, or fixing parsing errors
---

# Modifying PTX Grammar

This skill provides guidance for modifying ANTLR4 grammar files in the PTX-EMU project. It covers syntax rule updates, parser regeneration, and testing practices for PTX instruction parsing.

## When to Use
- Updating PTX grammar rules in `src/grammar/ptxLexer.g4` or `src/grammar/ptxParser.g4`
- Adding new PTX instruction syntax to support additional operations
- Fixing parsing errors (e.g., "no viable alternative", unexpected token)
- Regenerating the parser after grammar modifications
- Understanding ANTLR4 integration in the PTX-EMU project

## When NOT to Use
- Modifying PTX instruction implementations (execution logic in `src/ptxsim/instructions/`)
- Changing CUDA runtime API behavior (files in `src/cudart/`)
- General C++ development unrelated to grammar files
- Adding new GPU architecture configurations (JSON files in `configs/`)

## Quick Reference

| Task | Location / Notes |
|------|------------------|
| Grammar files | `src/grammar/ptxLexer.g4`, `src/grammar/ptxParser.g4` |
| Regenerate parser | Run via CMake (see AGENTS.md for command) |
| Generated parser output | `build/antlr4_generated_src/` |
| ANTLR version | 4.13.1 (see `CMakeLists.txt`) |
| Test parsing | Use `./build/bin/test-ptx <file.ptx>` |
| Grammar syntax check | Use `antlr4` tool (requires Java) |
| Environment setup | Run `. env.sh` before building (sets CUDA_PATH, LD_LIBRARY_PATH) |

**Note**: See AGENTS.md for full build commands and test workflows.

## Core Concepts

- **ANTLR4 Grammar Structure**: Split into lexer (`ptxLexer.g4`) and parser (`ptxParser.g4`) files
- **Lexer Rules**: Match tokens (uppercase names, e.g., `PARAM: '.param' ;`)
- **Parser Rules**: Define syntax structure (lowercase names, e.g., `paramDecl : PARAM type ... ;`)
- **Project Integration**: Generated parser compiled into `cudart` target
- **Regeneration Process**: CMake invokes ANTLR4 Java tool to regenerate C++ parser code
- **Testing Strategy**: PTX test suite validates parsing correctness after grammar changes
- **PtxListener/PtxVisitor Pattern**: Generated parser uses visitor pattern; semantic analysis in `ptx_visitor.cpp`
- **X-Macro ↔ Grammar Relationship**: Grammar tokens correspond to X-Macro entries in `include/ptx_ir/ptx_op.def`

## Implementation Steps

This section walks through fixing a grammar parsing issue using TDD workflow.

### Example: Fixing `.param .u64 .ptr .align` Declaration

**Scenario**: PTX file contains `.param .u64 .ptr .align 1` but parser fails with "no viable alternative" error.

**RED - Identify the failure:**
1. Run test to see parsing failure
   ```bash
   cd build && ctest -R test_memory_manager -V
   # Look for parser error in output
   ```
2. Verify grammar rule is missing tokens
   ```bash
   # Check paramDecl rule in src/grammar/ptxDeclarations.g4
   ```

**GREEN - Fix the grammar:**
1. Add missing tokens to lexer (`ptxLexer.g4`)
   ```antlr
   PTR   : '.ptr' ;
   ALIGN : '.align' ;
   ```
2. Update parser rule (`ptxDeclarations.g4`)
   ```antlr
   paramDecl
       : PARAM paramTokens
       | typeSpecifier? vectorSpec? ID
       ;

   paramTokens
       : (typeSpecifier | PTR | ALIGN | IMMEDIATE | ID)+
       ;
   ```
3. Regenerate parser
   ```bash
   cmake --build build --target GenerateParser
   ```
4. Rebuild affected targets
   ```bash
   cmake --build build --target cudart
   ```
5. Run tests to verify fix
   ```bash
   cd build && ctest -R test_memory_manager -V
   ```

**REFACTOR - Verify and clean:**
- Run full test suite: `cd build && ctest`
- Check related tests pass: `ctest -L ptx`
- Commit grammar + generated files together

## Common Mistakes

- **Forgetting to regenerate parser**: Grammar changes don't take effect until you run `cmake --build build --target GenerateParser`
- **Breaking existing syntax**: Changing rules used elsewhere can cause regressions across the PTX test suite
- **Incorrect token definitions**: Lexer tokens must match PTX syntax exactly (including leading dots like `.param`)
- **Manual edits to generated files**: Never edit files in `build/antlr4_generated_src/` directly
- **Not testing with full PTX suite**: Some edge cases only appear in specific PTX files; always run `ctest -L ptx`
- **Ignoring AGENTS.md anti-patterns**: Known limitations (WMMA stubs, atomic ops, Hopper not supported) — see AGENTS.md

## Real-World Impact

- **Recent fix** (commit b157c55): Grammar change removed COMMA from `paramList` rule and added `PTR` token to support `.param .u64 .ptr .align 1` declarations
- **What broke**: Parser failed with "no viable alternative" when encountering `.ptr .align` sequence
- **How it was fixed**: Updated lexer with `PTR` token, modified `paramDecl` rule to accept `paramTokens`, regenerated parser
- **Testing emphasis**: Always verify with `./build/bin/test-ptx` and run `ctest -L ptx` before considering change complete

## External Resources

- [ANTLR4 Getting Started](https://github.com/antlr/antlr4/blob/master/doc/getting-started.md) - Official getting started guide
- [ANTLR4 Grammar Structure](https://github.com/antlr/antlr4/blob/master/doc/grammars.md) - Grammar file structure and syntax
- [ANTLR4 Lexer Rules](https://github.com/antlr/antlr4/blob/master/doc/lexer-rules.md) - Lexer rule syntax and patterns
- [ANTLR4 C++ Target Example](https://github.com/teverett/antlr4-cpp-example) - Example project using ANTLR4 with C++
