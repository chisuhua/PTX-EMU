# PTX 解析测试报告

**日期**: 2026-03-13  
**测试工具**: test-ptx  
**测试目录**: tests/ptx/

## 测试结果

| 文件 | 架构 | 状态 | 备注 |
|------|------|------|------|
| dummy.1.sm_80.ptx | sm_80 | PASS | 基础指令 |
| dummy.1.sm_100.ptx | sm_100 | PASS | 新增 .ptr .align 语法 |
| dummy-condition.1.sm_100.ptx | sm_100 | FAIL | 段错误 |

## 发现的问题

### 1. .reg .pred 谓词寄存器声明无法解析

**问题文件**: dummy-condition.1.sm_100.ptx

**错误信息**:
line 26:7 no viable alternative at input
line 26:7 extraneous input .u32 expecting ... .pred
line 26:18 missing type specifier at %

**根本原因**:
- 语法文件 ptxDeclarations.g4 中 typeSpecifier 已包含 PRED
- 但解析器在遇到 .reg .pred %p<2> 时无法正确识别
- 可能是词法分析器中 PRED token 与其他规则冲突

**复现步骤**:
cd /workspace/PTX-EMU
PTX_EMU_PATH=/workspace/PTX-EMU ./build/bin/test-ptx tests/ptx/dummy-condition.1.sm_100.ptx
结果：段错误 (core dumped)

**简化测试用例**:
.version 9.0
.target sm_100
.visible .entry test_pred() {
  .reg .pred %p<2>;
  ret;
}

### 2. 解析器段错误风险

当遇到无法解析的 PTX 语法时，test-ptx 会段错误而不是优雅地报错。

建议:
- 在 PtxVisitor 中添加空指针检查
- 改进错误处理，避免崩溃

## 修复建议

### 短期（绕过问题）
1. 暂时避免在测试中使用 .pred 寄存器
2. 使用 .u16/.u32 等替代谓词逻辑

### 长期（彻底修复）
1. 检查 ptxLexer.g4 中 PRED token 的定义
2. 确认 ptxDeclarations.g4 中 regDecl 规则
3. 重新生成 ANTLR 解析器
4. 在 PtxVisitor 中添加防御性编程

## 测试脚本

批量测试脚本已创建：
./tests/ptx/test_all_ptx.sh

## 附件

- test_all_ptx.sh - PTX 批量解析测试脚本
- dummy.1.sm_100.ptx - cuobjdump 从 dummy 二进制提取
- dummy-condition.1.sm_100.ptx - cuobjdump 从 dummy-condition 二进制提取

