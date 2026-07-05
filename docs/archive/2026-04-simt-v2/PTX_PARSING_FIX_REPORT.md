# PTX 解析修复报告

## 问题描述
`dummy-condition.1.sm_100.ptx` 测试失败，错误在 `cvt.u16.u32` 指令处。

## 已完成的修复

### 1. 修复 DECIMAL_INT 未定义问题
**文件**: `ptxDeclarations.g4`, `ptxInstructions.g4`
**问题**: `DECIMAL_INT` token 未在词法文件中定义
**修复**: 将所有 `DECIMAL_INT` 替换为 `IMMEDIATE`

### 2. 修复循环导入问题
**文件**: `ptxDeclarations.g4`, `ptxInstructions.g4`, `ptxParser.g4`
**问题**: `ptxDeclarations` 和 `ptxInstructions` 互相导入，导致 ANTLR 生成解析器失败
**修复**:
- 从 `ptxDeclarations.g4` 移除对 `ptxInstructions` 的导入
- 将 `functionDecl` 相关规则移动到 `ptxInstructions.g4`
- 修改 `ptxParser.g4` 的 `ptxFile` 规则为 `(declaration | functionDecl)* EOF`

### 3. 修复 Visitor 代码
**文件**: `ptx_visitor.cpp`, `ptx_visitor_barrier.cpp`, `test-ptx.cpp`
**问题**: Visitor 代码引用已移除的 `functionDecl` 和 `DECIMAL_INT`
**修复**:
- 更新 `visitPtxFile` 处理 `functionDecl`
- 从 `visitDeclaration` 移除 `functionDecl` 检查
- 将 `DECIMAL_INT` 引用改为 `IMMEDIATE`

### 4. 修复 funcBody 规则
**文件**: `ptxInstructions.g4`
**问题**: `(regDecl | instruction)*` 可能导致解析歧义
**修复**: 改为 `regDecl* instruction*`（先匹配所有寄存器声明，再匹配所有指令）

### 5. CVT 指令问题（进行中）
**问题**: `cvt.u16.u32 %rs1, %r1;` 无法解析
**错误**: 解析器在 `.u32` 处期望看到操作数
**当前状态**: 需要进一步调试

## CVT 指令调试建议

1. 使用 ANTLR `grun` 工具查看解析树：
```bash
cd /workspace/PTX-EMU/src/grammar
antlr4 -Dlanguage=Cpp ptxLexer.g4 ptxParser.g4
grun ptx ptxFile -tokens -tree -gui < /tmp/test_cvt.ptx
```

2. 检查词法分析是否正确标记 `cvt.u16.u32`：
```bash
grun ptx ptxFile -tokens < /tmp/test_cvt.ptx
```

3. 尝试简化 CVT 语法，逐步添加修饰符定位问题

## 测试状态
- `dummy.1.sm_80.ptx`: PASS ✓
- `dummy.1.sm_100.ptx`: PASS ✓
- `dummy-condition.1.sm_100.ptx`: FAIL ✗ (CVT 指令问题)

## 下一步
1. 继续调试 CVT 指令解析问题
2. 考虑联系 ANTLR 专家或查看类似项目
3. 可能需要重新设计 CVT 指令的语法结构
