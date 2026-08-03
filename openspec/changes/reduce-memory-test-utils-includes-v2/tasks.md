# reduce-memory-test-utils-includes-v2 - Tasks

## 1. Phase 1: 基线记录（5 min）

- [ ] 1.1 MUST 记录当前 include 数量：`grep -c '^#include' include/ptxsim/testing/memory_test_utils.h`（应为 18）
- [ ] 1.2 MUST 验证当前可编译：`cmake --build build` 通过
- [ ] 1.3 MUST 运行 `cd build && ctest -L unit --output-on-failure` 记录基线测试结果（全绿）

## 2. Phase 2: 分析函数签名和 inline 状态（15 min）

- [ ] 2.1 MUST 列出 `memory_test_utils.h` 中所有函数声明
- [ ] 2.2 MUST 标注每个函数是否 inline
- [ ] 2.3 MUST 标注每个函数签名中使用的类型（参数和返回值）
- [ ] 2.4 MUST 标注 inline 函数体内使用的类型（调用方法的类型需完整定义）
- [ ] 2.5 MUST 确定分类表：
  - 可前向声明（仅签名指针/引用参数，非 inline 或 inline 但函数体不使用该类型方法）
  - 可移到 .cpp（非 inline 且仅函数体使用）
  - 必须保留（值类型参数、inline 函数体使用方法、Catch2 宏依赖）
- [ ] 2.6 MUST 确认是否需要新建 `.cpp` 文件（如当前仅有头文件）

## 3. Phase 3: 标准库 include 精简（10 min）

- [ ] 3.1 MUST 分析 `<algorithm>` 使用方式：如仅 .cpp 使用 → 移到 .cpp
- [ ] 3.2 MUST 分析 `<cstdlib>` 使用方式：如仅 .cpp 使用 → 移到 .cpp
- [ ] 3.3 MUST 分析 `<map>` 使用方式：如仅 .cpp 使用 → 移到 .cpp
- [ ] 3.4 MUST 验证：`cmake --build build` 通过
- [ ] 3.5 MUST 验证：`ctest -L unit --output-on-failure` 全绿
- [ ] 3.6 git commit -m "refactor(memory_test_utils): reduce standard library includes"

## 4. Phase 4: 项目头文件前向声明替代（20 min）

- [ ] 4.1 MUST 对可前向声明的类型添加 forward declarations 到头文件集中声明区
- [ ] 4.2 MUST 移除可前向声明类型对应的 `#include`
- [ ] 4.3 MUST 保留 inline 函数体使用方法所需类型的完整 include
- [ ] 4.4 MUST 保留 Catch2 `catch_amalgamated.hpp`
- [ ] 4.5 MUST 验证：`cmake --build build` 通过
- [ ] 4.6 MUST 验证：`grep -c '^#include' include/ptxsim/testing/memory_test_utils.h` ≤ 12
- [ ] 4.7 MUST 验证：`ctest -L unit --output-on-failure` 全绿
- [ ] 4.8 git commit -m "refactor(memory_test_utils): replace project headers with forward declarations"

## 5. Phase 5: 最终验证（10 min）

- [ ] 5.1 MUST 验证：`include/ptxsim/testing/memory_test_utils.h` include 数 ≤ 12
- [ ] 5.2 MUST 验证：所有使用 memory_test_utils.h 的测试文件编译通过
- [ ] 5.3 MUST 验证：`cd build && ctest -L unit --output-on-failure` 全绿
- [ ] 5.4 SHOULD 验证测试编译时间缩短（与基线对比）

## 6. 应用阶段

- [ ] 6.1 MUST 运行 `openspec validate reduce-memory-test-utils-includes-v2 --strict`
- [ ] 6.2 MUST 通过所有验证后 archive

## 验收

- include 数量 ≤ 12（18 → ≤12，减少 ≥30%）
- 所有使用 memory_test_utils.h 的测试文件编译通过
- ctest -L unit 全绿
- inline 函数完整性保留

## 关键约束（MUST/MUST NOT）

- MUST 所有测试编译通过
- MUST NOT 改变任何函数签名或行为
- MUST NOT 移除 Catch2 catch_amalgamated.hpp
- SHOULD 保持 inline 函数完整性