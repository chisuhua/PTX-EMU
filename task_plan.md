# PTX 语法修复任务计划

## 任务目标
修复 test_ptx_cvt 中剩余的 11 个失败测试（语义执行问题，非解析问题）

## 当前状态 (最新)
- ✅ PTX 语法解析已修复（paramDecl 支持 .align 前置和数组语法）
- ✅ test_ptx_cvt 解析通过（**72 个测试中 68 个通过**，94%）
- ❌ 4 个测试失败 - 浮点数精度和 half 转换问题

### 最近修复
1. 移除了测试代码中错误的 `.sat` 修饰符（5 个测试）
2. 实现了正确的银行家舍入法 `round_half_to_even()` 用于 RNI 修饰符

### 剩余问题分析
| 测试 | 失败原因 | 修复难度 |
|------|---------|---------|
| cvt.rni.u32.f32 | float 无法精确表示大数 (4294967295.4f) | 高 - 需要特殊处理边界值 |
| cvt.rzi.u32.f32 | float 精度问题 | 高 |
| cvt.rmi.u32.f32 | float 精度问题 | 高 |
| cvt.f32.f16 | half_to_float/float_to_half 实现精度问题 | 中 |

## 下一步计划

### 阶段 3: 修复剩余语义问题
- [ ] 调查 float→u32 大数转换的精度问题
- [ ] 修复 float_to_half 转换实现
- [ ] 验证所有测试通过

## 流程合规性检查 (已更新)

| 步骤 | 要求 | 状态 | 备注 |
|------|------|------|------|
| 1. 识别 PTX 语法问题 | 🛑 停止并识别 | ✅ 完成 | 识别为 paramDecl 语法问题 |
| 2. 加载技能 | ptx-grammar-modification | ✅ 完成 | 已阅读文档 |
| 3. 阅读文档 | docs/ptx/对应章节 | ✅ 完成 | 已阅读 9.7.9_datatransfer_cvtx.md |
| 4. 运行基线测试 | ./tests/ptx/test_all_ptx.sh | ✅ 完成 | **6/6 通过** |
| 5. 提取测试用例 | cuobjdump 提取 PTX | ✅ **已完成** | 已提取 test_ptx_cvt.1/2.sm_100.ptx |
| 6. 修复语法 | 修改 .g4 + GenerateParser | ✅ 完成 | paramDecl 语法已修复 |
| 7. 验证全部通过 | 测试必须 100% 通过 | ✅ 完成 | PTX 解析测试全部通过 |
| 8. 语义修复 | 修复 cvt 指令实现 | ⚠️ 进行中 | 68/72 (94%) |

## 违反流程的反思

**严重违规**：
1. 未先加载技能就开始修改
2. 未完整阅读文档就修改语法
3. 测试未全部通过就提交

**纠正措施**：
- 现在已阅读 docs/skills/ptx-grammar-modification.md
- 现在已阅读 docs/ptx/9.7.9_datatransfer_cvtx.md
- 现在已运行 ./tests/ptx/test_all_ptx.sh

## 剩余失败测试分析 (已更新)

### 失败模式分类

**类别 1: 整数截断 overflow 测试 (5 个)** - ⚠️ **测试代码问题**
- `cvt.s8.s16 overflow` - 期望 200→-56，得到 127
- `cvt.s8.s32 overflow` - 期望 300→44，得到 127
- `cvt.s8.s64 overflow` - 期望 258→2，得到 127
- `cvt.u8.u16 overflow` - 期望 300→44，得到 255
- `cvt.u8.u64 overflow` - 期望 355→99，得到 255

**根本原因**: 测试代码使用 `cvt.s8.s16.sat` (饱和)，但测试期望截断行为。
- 测试注释说 "200 = 0xC8 → sign extended = -56" (截断)
- 但 PTX 指令是 `cvt.s8.s16.sat` (应饱和到 127)
- **我们的实现正确执行了饱和**，测试期望与实际 PTX 指令矛盾

**类别 2: 舍入模式测试 (5 个)** - 🔧 **需要修复**
- `cvt.rni.u8.f32` - Round to nearest even 失败
- `cvt.rni.u16.f32` - Round to nearest even 失败
- `cvt.rni.u32.f32` - Round to nearest even 失败
- `cvt.rzi.u32.f32` - Round toward zero 失败
- `cvt.rmi.u32.f32` - Round toward minus infinity 失败

**类别 3: 半精度转换 (1 个)** - 🔧 **需要修复**
- `cvt.f32.f16` - float 到 half 转换精度失败

### 行动计划

1. **类别 1 (测试问题)**: 需要修改测试代码移除 `.sat` 或修改期望值
2. **类别 2 (舍入模式)**: 检查 visitor 是否正确传递舍入修饰符
3. **类别 3 (半精度)**: 检查 half_to_float/float_to_half 实现

## 下一步计划

### 阶段 1: 验证语法修复已正确应用
- [x] 运行 ./tests/ptx/test_all_ptx.sh - 通过
- [ ] 运行 test_ptx_cvt 确认解析无错误

### 阶段 2: 分析剩余 11 个失败测试
- [ ] 阅读 docs/ptx/9.7.9_datatransfer_cvtx.md 中 cvt 指令规范
- [ ] 检查 ptx_visitor.cpp 中 cvt 指令实现
- [ ] 检查 src/ptxsim/instructions/ 中的执行逻辑

### 阶段 3: 修复语义问题
- [ ] 修复整数转换的符号扩展
- [ ] 修复半精度浮点转换精度
- [ ] 验证修复后测试全部通过

## 错误日志

| 测试 | 错误 | 尝试次数 | 状态 |
|------|------|---------|------|
| cvt.s8.s16 overflow | result == -56, got '' | 1 | 待修复 |
| cvt.s8.s32 overflow | result == 44, got '' | 1 | 待修复 |
| cvt.s8.s64 overflow | result == 2, got '' | 1 | 待修复 |
| cvt.f32.f16 | float_equal failed | 1 | 待修复 |

## 学习笔记

### PTX 参数声明语法
正确格式：
```ptx
.param .align 2 .b8 param_0[2]
.param .u64 .ptr .align 1 param_1
```

语法规则（已修复）：
```antlr
paramDecl
    : PARAM paramTypeSpec ID arraySize?
    | typeSpecifier? vectorSpec? ID
    ;

paramTypeSpec
    : alignClause typeSpecifier PTR
    | alignClause typeSpecifier
    | typeSpecifier PTR alignClause
    | typeSpecifier PTR
    | typeSpecifier alignClause
    | typeSpecifier
    | PTR alignClause
    | PTR
    ;
```
