# merge-arithmetic-handlers - Proposal

## Why

3 个算术 handler 文件职责重叠，共 **1732 行**，包含大量重复的 operand 处理逻辑：

| 文件 | 行数 | Handler 数 | 职责 |
|------|------|-----------|------|
| `arithmetic.cpp` | 478 | 4 (Add, Sub, Neg, Abs) | 基础算术 + ~350 行注释掉的旧代码 |
| `arithmetic_ext.cpp` | 764 | 5 (Addc, Subc, Mul24, Mad24, Fma) | 扩展算术 |
| `arithmetic_muldiv.cpp` | 490 | 6 (Mul, Div, Mad, Min, Max, Rem) | 乘除运算 |
| **合计** | **1732** | **15 handlers** | |

重复模式包括：
- `getBytes(qualifiers)` + `is_float` + `is_signed` 类型解析（每个 handler 重复）
- `memcpy` 读取/写入 operand（每个 handler 重复）
- f16→f32→f16 转换模式（Mul、Fma、Add 等重复）
- `.cc` 条件码更新逻辑（Add、Sub、Addc、Subc 重复）

X-Macro 分派模式下每个 handler 独立注册，合并后可统一 operand 提取和类型分派。

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-3`

## What Changes

- **合并** 3 个 arithmetic 文件为统一 handler 文件 `arithmetic.cpp`
- **提取** 共享 operand 处理逻辑为 helper 函数
- **统一** 类型分派逻辑（消除每个 handler 中的重复 `switch(bytes)` 模式）
- **删除** `arithmetic.cpp` 中 ~350 行注释掉的旧代码

## Capabilities

### New Capabilities
- `arithmetic-helper-functions`: 共享 operand 提取、类型分派、CC 更新的 helper 函数集

### Modified Capabilities
- `arithmetic-instruction-handlers`: 15 个算术 handler 从 3 文件合并为 1 文件，逻辑不变
- `cmake-source-list`: `src/CMakeLists.txt` 移除 2 个源文件引用

## Impact

**受影响代码**：
- `src/ptxsim/instructions/arithmetic.cpp`（合并目标，重写）
- `src/ptxsim/instructions/arithmetic_ext.cpp`（删除）
- `src/ptxsim/instructions/arithmetic_muldiv.cpp`（删除）
- `src/CMakeLists.txt`（移除 2 行源文件引用，line 125 和 144）
- 可能新增 `src/ptxsim/utils/arithmetic_helpers.h`（共享 helper 声明）

**不受影响**：
- `include/ptx_ir/ptx_op.def`（X-Macro 注册不变）
- `src/ptxsim/instruction_factory.cpp`（分派逻辑不变）
- 测试文件（不动）
- 任何算术指令的计算结果

**依赖**：
- 无前置 change 依赖，可独立执行
- 合并后必须保证所有算术指令测试通过

**工时**: 2-3h（提取 helper + 合并文件 + 验证）

## Design-Time Checklist

- [ ] 确认 15 个 handler 的完整清单和当前文件分布
- [ ] 确认共享 operand 处理逻辑的具体重复模式
- [ ] 确认 `src/CMakeLists.txt` 的源文件引用位置（line 124-125, 144）
- [ ] 确认 `instruction_handlers.h` 中的 handler 类声明不受影响
- [ ] 确认合并策略：按运算类型分组（加减 / 乘除 / 扩展）
- [ ] 确认所有算术指令的现有测试覆盖范围（unit + integration）
