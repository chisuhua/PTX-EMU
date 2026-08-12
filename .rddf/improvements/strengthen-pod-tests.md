# strengthen-pod-tests

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-6
**阶段**: default | **分类**: core-test
**类型**: test

## 架构依据

- `tests/unit/` 中与 context 相关的测试文件偏浅：
  - `test_sm_context.cpp` (126 行)
  - `test_warp_context.cpp` (98 行)
  - `test_cvt_context.cpp` (274 行)
  - `test_smcontext_injection.cpp` (375 行)
- 原始审计指出 tests/unit/contexts/ 下 7 个 <50 行 POD 测试（目录已重组，但浅测试模式仍存在）
- 浅测试仅验证"对象可构造"，不验证行为正确性

## 范围

- **In Scope**:
  - 为现有 context 测试补充行为验证（状态转换、边界条件、错误路径）
  - 确保每个测试文件 ≥ 50 行有效断言
  - 添加 execute_warp_instruction 驱动的集成验证（test-coverage-enforcer 要求）
- **Out Scope**:
  - 不新建测试目录结构
  - 不修改被测源码
  - 不重复已有 integration/e2e 测试覆盖的场景

## 关键场景

- GIVEN 补充行为测试, WHEN ThreadContext 状态变更, THEN 断言验证状态转换正确性（非仅验证构造）
- GIVEN 补充边界测试, WHEN 输入非法参数, THEN 断言验证错误处理路径

## 技术约束

- MUST 遵循 test-coverage-enforcer：新增单元测试必须有对应 execute_warp_instruction 集成测试
- MUST 使用 Catch2 框架
- MUST 测试标签格式 `<type>;<subject>`

## 验收标准

- 每个 context 测试文件 ≥ 50 行有效断言
- 新增测试覆盖至少 1 个错误路径
- 对应集成测试存在且通过
- ctest -L unit 全绿
