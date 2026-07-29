# reduce-memory-test-utils-includes - Tasks

## 1. Phase 1: 基线记录（10 min）

- [ ] 1.1 MUST 记录当前 include 数量：`grep -c '#include' include/ptxsim/testing/memory_test_utils.h`（应为 18）
- [ ] 1.2 MUST 验证当前可编译：`cmake --build build` 通过
- [ ] 1.3 MUST 运行 `cd build && ctest -L unit --output-on-failure` 记录基线测试结果（全绿）
- [ ] 1.4 SHOULD 记录测试编译时间基线：`time cmake --build build`

## 2. Phase 2: 分析函数签名和 inline 状态（20 min）

- [ ] 2.1 MUST 列出 `memory_test_utils.h` 中所有函数声明
- [ ] 2.2 MUST 标注每个函数是否 inline
- [ ] 2.3 MUST 标注每个函数签名中使用的类型（参数和返回值）
- [ ] 2.4 MUST 标注 inline 函数体内使用的类型（调用方法的类型需完整定义）
- [ ] 2.5 MUST 确定分类表：
  - 可前向声明（仅签名指针/引用参数，非 inline 或 inline 但函数体不使用该类型方法）
  - 可移到 .cpp（非 inline 且仅函数体使用）
  - 必须保留（值类型参数、inline 函数体使用方法、Catch2 宏依赖）
- [ ] 2.6 MUST 确认是否需要新建 `.cpp` 文件（如当前仅有头文件）

## 3. Phase 3: 标准库 include 精简（15 min）

- [ ] 3.1 MUST 分析 `<algorithm>` 使用方式：如仅 .cpp 使用 -> 移到 .cpp
- [ ] 3.2 MUST 分析 `<cstdlib>` 使用方式：如仅 .cpp 使用 -> 移到 .cpp
- [ ] 3.3 MUST 分析 `<map>` 使用方式：如仅 .cpp 使用 -> 移到 .cpp
- [ ] 3.4 MUST 验证：`cmake --build build` 通过
- [ ] 3.5 MUST 验证：`ctest -L unit --output-on-failure` 全绿
- [ ] 3.6 git commit -m "refactor(memory_test_utils): reduce standard library includes"

## 4. Phase 4: 项目头文件前向声明/移动（30 min）

- [ ] 4.1 MUST 对可前向声明的类型添加前向声明到头文件：
  ```cpp
  // --- Forward declarations ---
  namespace ptxsim {
  class CTAContext;
  class WarpContext;
  class SMContext;
  class InstructionFactory;
  }  // namespace ptxsim
  ```
- [ ] 4.2 MUST 对可移到 .cpp 的函数：将定义从头文件移到 `.cpp`（去掉 inline）
- [ ] 4.3 MUST 如需新建 `.cpp` 文件：
  - 创建 `src/ptxsim/testing/memory_test_utils.cpp`
  - 在 `src/ptxsim/testing/CMakeLists.txt`（或上级 CMakeLists.txt）添加源文件
- [ ] 4.4 MUST 逐个移除可前向声明/已移动类型对应的 `#include`（每次 2-3 个）
- [ ] 4.5 MUST 每批移除后验证：`cmake --build build` 通过
- [ ] 4.6 MUST 如出现 "incomplete type" 错误：回退该 include（说明为值类型或 inline 函数体使用）
- [ ] 4.7 MUST 验证：`ctest -L unit --output-on-failure` 全绿
- [ ] 4.8 git commit -m "refactor(memory_test_utils): replace includes with forward declarations"

## 5. Phase 5: 最终验证（10 min）

- [ ] 5.1 MUST 验证：include 数量 ≤ 12
  ```bash
  grep -c '#include' include/ptxsim/testing/memory_test_utils.h  # 应 ≤ 12
  ```
- [ ] 5.2 MUST 验证：全量编译通过 `cmake --build build`
- [ ] 5.3 MUST 验证：无新增编译 warning
- [ ] 5.4 MUST 验证：`cd build && ctest -L unit --output-on-failure` 全绿
- [ ] 5.5 MUST 验证：函数签名无变化（diff 对比函数声明）
- [ ] 5.6 SHOULD 验证：所有使用该头文件的测试文件编译通过

## 6. 应用阶段

- [ ] 6.1 MUST 运行 `openspec validate reduce-memory-test-utils-includes --strict`
- [ ] 6.2 MUST 通过所有验证后 archive 此 change

## 验收

- include 数量减少 ≥ 30%（18 -> ≤ 12）
- ctest -L unit 全绿
- 所有使用该头文件的测试文件编译通过
- 无函数签名变更
- 无新增编译 warning

## 关键约束（MUST/MUST NOT）

- MUST 所有测试编译通过
- MUST NOT 改变任何函数签名或行为
- MUST NOT 影响 inline 函数完整性（如移到 .cpp 则正确去 inline）
- SHOULD 保持 inline 函数完整性
- MUST NOT 删除实际需要的 include
