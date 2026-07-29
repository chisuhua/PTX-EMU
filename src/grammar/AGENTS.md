# src/grammar/ AGENTS.md
**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md; this file only documents grammar-specific content.

## OVERVIEW

ANTLR4 语法文件（5 个 .g4 文件）— 将 PTX ISA 文本解析为 AST，供 ptx_parser 和 ptxsim 消费。

## STRUCTURE

```
src/grammar/
├── ptxLexer.g4          # (513行) 词法分析 — 所有 token 定义
│                        #   标点、关键字、类型、NAN/INFINITY 等
├── ptxParser.g4         # (11行)  解析入口 — ptxFile → (declaration | functionDecl)* EOF
│                        #   仅 1 条规则，其余委托给子语法
├── ptxDeclarations.g4   # (129行) PTX 声明规则 — .version, .target, .reg, .shared, .entry
├── ptxInstructions.g4   # (597行) 指令规则 — ~200 条 PTX 指令语法
└── ptxOperands.g4       # (69行)  操作数规则 — register, immediate, address, vectorRegister
```

**导入链**: `ptxParser` → `ptxDeclarations` + `ptxInstructions` → `ptxOperands`（所有子语法共用 `tokenVocab=ptxLexer`）

## WHERE TO LOOK

| Task | File |
|------|------|
| 新增 token / 关键字 | `ptxLexer.g4` |
| 新增 PTX 指令语法 | `ptxInstructions.g4` |
| 新增声明/变量规则 | `ptxDeclarations.g4` |
| 新增操作数格式 | `ptxOperands.g4` |
| 解析入口 | `ptxParser.g4` |
| 语法修改 skill | `.opencode/skills/ptx-grammar-modification/SKILL.md` |
| PTX 文档参考 | `docs/ptx/README.md` |
| PTX 语法测试 | `tests/ptx/test_all_ptx.sh` |
| 生成后源码 | `build/antlr4_generated_src/`（NEVER 手动编辑） |

## CONVENTIONS

- **Token 命名**: 全大写加下划线（`NAN`, `DOT`, `LEFT_BRACE`）
- **Parser 规则**: 小驼峰（`functionDecl`, `typeSpecifier`）
- **Token 名冲突**: `NAN`, `INFINITY` 与 `<cmath>` 宏冲突 — 在 `ptxLexer.g4` 头部 `#undef` 解决
- **子语法隔离**: 各 `ptxInstructions.g4` 规则加 `instruction` 前缀避免导入其他子语法时冲突
- **循环依赖**: `ptxDeclarations` 只导入 `ptxOperands`（不导入 `ptxInstructions`），`funcBody` 定义在 `ptxInstructions.g4` 中
- **生成目录**: `build/antlr4_generated_src/ptxparser/` — 由 `GenerateParser` CMake target 触发

## ANTI-PATTERNS

- ❌ 直接编辑 `build/antlr4_generated_src/` 下的文件（会被覆盖）
- ❌ 修改 `.g4` 而不加载 `ptx-grammar-modification` skill（缺失强制检查清单）
- ❌ 用 `ctest` 代替 `./tests/ptx/test_all_ptx.sh` 验证语法修改（ctest 跑的是 C++ 单元测试，非语法解析测试）
- ❌ 在 `ptxDeclarations.g4` 中导入 `ptxInstructions`（导致循环依赖）
- ❌ 修改 `.g4` 后忘记运行 `cmake --build build --target GenerateParser`（解析器不更新）

## COMMANDS

```bash
# 改 .g4 后重新生成解析器
cmake --build build --target GenerateParser

# PTX 语法验证（所有 tests/ptx/*.ptx）
./tests/ptx/test_all_ptx.sh

# 新增 .ptx 测试文件后验证
cp my_test.ptx tests/ptx/
./tests/ptx/test_all_ptx.sh
```

### 错误分类（ANTLR 解析错误 vs 运行时错误）

```bash
# 1. 运行失败的测试获取输出
cd build && ctest -R <test_name> -V 2>&1 | tail -50

# 2. 用 grep 区分错误类型
echo <输出> | grep -E "missing|mismatched|no viable|extraneous|ANTLR"
# → 有输出 = ANTLR 解析错误 → 走语法修复流程（本文件 + ptx-grammar-modification）
# → 无输出 = 运行时错误   → 走 ptx-debug 流程
```

**语法修改 checklist**（来自 `ptx-grammar-modification` skill）:
1. 加载 `ptx-grammar-modification` skill
2. 阅读 `docs/ptx/` 对应章节
3. 运行 `./tests/ptx/test_all_ptx.sh` 确认基线
4. 如有真实 binary，`cuobjdump -xptx` 提取 PTX
5. 修改 `.g4` → `cmake --build build --target GenerateParser`
6. `./tests/ptx/test_all_ptx.sh` **全部通过**