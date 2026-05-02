# ANTLR4 PTX Grammar

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
ANTLR4 grammar files (5 split `.g4` files) — lexer, parser, declarations, instructions, operands. Generated C++ goes to `build/antlr4_generated_src/`.

## STRUCTURE
```
src/grammar/
├── ptxLexer.g4            # Token definitions (NAN, INFINITY, etc.)
├── ptxParser.g4           # Entry point — imports sub-grammars
├── ptxDeclarations.g4     # PTX declaration rules (.reg, .shared, .entry)
├── ptxInstructions.g4     # PTX instruction rules (~500 lines)
└── ptxOperands.g4         # Operand parsing rules (reg, imm, addr)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Token definitions | `ptxLexer.g4` | Lexer tokens: DIGITS, IDENTIFIER, SIGIL, etc. |
| Parser entry | `ptxParser.g4` | `import ptxDeclarations, ptxInstructions;` |
| Instruction grammar | `ptxInstructions.g4` | One rule per instruction group |
| ANTLR aliases | `. env.sh` → `antlr4`, `grun` | CLI tools for grammar debugging |

## GRAMMAR MODIFICATION CHECKLIST
```
□ 1. 加载技能：docs/skills/ptx-grammar-modification.md
□ 2. 阅读 docs/ptx/ 对应章节
□ 3. 运行基线：./tests/ptx/test_all_ptx.sh（不是 ctest！）
□ 4. cuobjdump -xptx 提取真实 PTX → 复制到 tests/ptx/
□ 5. 修改 .g4 → cmake --build build --target GenerateParser
□ 6. ./tests/ptx/test_all_ptx.sh 全部通过才能交付
```

## CONVENTIONS (this dir)
- ANTLR4 `import` mechanism — sub-grammars share tokens via main grammar
- Token names conflict with `<cmath>` macros (`NAN`, `INFINITY`) — resolved by `#undef` before ANTLR includes
- PTX instruction names: ALL lowercase in grammar (`mov`, `add`, `bra`, `bar.sync`)
- Generated sources → `build/antlr4_generated_src/` (NEVER edit generated files)

## ANTI-PATTERNS
- DO NOT modify grammar without loading `ptx-grammar-modification` skill
- DO NOT use `ctest` for PTX syntax validation — use `./tests/ptx/test_all_ptx.sh`
- DO NOT skip adding test case in `tests/ptx/` before grammar changes
- DO NOT declare test "done" unless `test_all_ptx.sh` passes 100%

## COMMANDS
```bash
cmake --build build --target GenerateParser  # Regenerate ANTLR C++ from .g4
./tests/ptx/test_all_ptx.sh                  # Full syntax test suite
antlr4 src/grammar/ptxParser.g4              # Manual ANTLR invocation (after . env.sh)
grun ptxparser ptxFile -tree < test.ptx      # Parse tree visualization
```
