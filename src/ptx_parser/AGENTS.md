# PTX Parser (ANTLR4)

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
ANTLR4-based PTX parser - tokenizes and parses PTX ISA, builds IR for simulation.

## STRUCTURE
```
src/ptx_parser/          # PTXVisitor implementation, CFGBuilder
include/ptx_parser/      # Public headers
src/grammar/             # ptxLexer.g4, ptxParser.g4
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Grammar files | `src/grammar/` | ptxLexer.g4, ptxParser.g4 |
| Visitor implementation | `src/ptx_parser/` | PtxVisitor |
| CFG builder | `src/ptx_parser/cfg_builder.cpp` | Control flow graph |
| Symbol table | `include/ptx_ir/` | Operand, statement contexts |

## KEY FILES
| File | Purpose |
|------|---------|
| `ptxLexer.g4` | ANTLR lexer grammar |
| `ptxParser.g4` | ANTLR parser grammar |
| `PtxVisitor.cpp` | AST traversal, builds IR |

## CONVENTIONS (this dir)
- Grammar uses ANTLR4 syntax
- X-Macro defines instructions in `include/ptx_ir/ptx_op.def`
- Generated code → `build/antlr4_generated_src/`

## ANTI-PATTERNS
- DO NOT modify grammar without loading `ptx-grammar-modification` skill
- DO NOT skip `test_all_ptx.sh` after grammar changes

## COMMANDS
```bash
cmake --build build --target GenerateParser  # Regenerate ANTLR parser
```
