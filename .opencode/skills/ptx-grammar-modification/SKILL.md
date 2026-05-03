---
name: "ptx-grammar-modification"
description: "PTX ANTLR4 语法修改流程 — 强制 TDD 流程、错误分类、测试验证"
when_to_use: |
  PTX-EMU 项目中出现以下**任一**匹配时触发：
  - ANTLR 解析错误: "no viable alternative", "mismatched input", "extraneous input"
  - PTX 文件解析失败: "Failed to parse PTX file"
  - 修改了 src/grammar/*.g4 文件
  - 需要添加新的 PTX 指令语法
skills_required: ["test-driven-development"]
---

# PTX 语法修改技能

## 触发决策树

```
遇到错误 → 是 ANTLR 解析错误？ → 是 → 🛑 使用本技能
                           → 否 → 普通 C++ 调试流程
```

## 强制流程（不可跳过）

### 检查清单

- [ ] 确认是 ANTLR 解析错误（`no viable alternative` / `mismatched input`）
- [ ] 已阅读 `docs/ptx/` 对应章节
- [ ] 已运行基线测试：`./tests/ptx/test_all_ptx.sh`
- [ ] 如有真实 binary，已用 `cuobjdump -xptx` 提取 PTX → 复制到 `tests/ptx/`
- [ ] 修改 `.g4` → `cmake --build build --target GenerateParser`
- [ ] 运行 `./tests/ptx/test_all_ptx.sh` **全部通过**才能交付

### 步骤

```bash
# 1. RED - 运行测试确认失败
./tests/ptx/test_all_ptx.sh

# 2. GREEN - 修复语法
# 修改 src/grammar/ptxLexer.g4 和/或 ptxParser.g4
cmake --build build --target GenerateParser
cmake --build build --target cudart

# 3. REFACTOR - 验证
./tests/ptx/test_all_ptx.sh   # 全部通过
cd build && ctest -L ptx       # PTX 测试全绿
```

## 禁止行为

- ❌ 用 `ctest` 代替 `./tests/ptx/test_all_ptx.sh`
- ❌ 未读 `docs/ptx/` 就改语法
- ❌ 未加测试用例就修语法
- ❌ 手动编辑 `build/antlr4_generated_src/` 中的生成文件
- ❌ 测试未全通过就标完成

## 快速参考

| 内容 | 路径 |
|------|------|
| 语法文件 | `src/grammar/ptxLexer.g4`, `ptxParser.g4` |
| 重新生成 | `cmake --build build --target GenerateParser` |
| 生成输出 | `build/antlr4_generated_src/` |
| PTX 文档 | `docs/ptx/README.md` |
| 批量测试 | `./tests/ptx/test_all_ptx.sh` |

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| 忘记重新生成解析器 | 改 .g4 后未运行 GenerateParser | `cmake --build build --target GenerateParser` |
| 破坏现有语法 | 新规则影响已有解析 | 运行 `test_all_ptx.sh` 检查回归 |
| Token 定义错误 | 词法 token 与 PTX 语法不匹配 | Token 必须精确匹配（含前导点） |
| 手动编辑生成文件 | 修改 `build/antlr4_generated_src/` | **永远不要**手动编辑生成文件 |
