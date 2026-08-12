# reduce-memory-test-utils-includes

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-8
**阶段**: default | **分类**: core-test
**类型**: refactor

## 架构依据

- `include/ptxsim/testing/memory_test_utils.h` 有 **18 个 #include**
- 该头文件被多个测试文件包含，include 膨胀直接影响测试编译时间
- 部分 include 可能通过前向声明或移到 .cpp 实现来消除

## 范围

- **In Scope**:
  - 分析 18 个 include 的必要性
  - 对可前向声明的类型改为前向声明
  - 将实现特有的 include 移到头文件对应的 inline 实现区
- **Out Scope**:
  - 不改变 memory_test_utils.h 的任何函数签名
  - 不影响使用该头文件的测试文件

## 关键场景

- GIVEN include 精简后, WHEN 编译所有测试, THEN 编译通过
- GIVEN include 精简后, WHEN 修改被精简的头文件, THEN 不触发测试重编译

## 技术约束

- MUST 所有测试编译通过
- MUST NOT 改变任何函数签名或行为
- SHOULD 保持 inline 函数完整性

## 验收标准

- include 数量减少 ≥ 30%（18 → ≤ 12）
- ctest -L unit 全绿
- 所有使用该头文件的测试文件编译通过
