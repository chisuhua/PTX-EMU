# reduce-thread-context-includes-v2 - Tasks

## 1. Phase 1: 基线记录（10 min）

- [ ] 1.1 MUST 记录当前 include 数量：`grep -c '^#include' include/ptxsim/thread_context.h`（应为 25）
- [ ] 1.2 MUST 验证当前可编译：`cmake --build build` 通过
- [ ] 1.3 MUST 运行 `cd build && ctest --output-on-failure` 记录基线测试结果（全绿）
- [ ] 1.4 SHOULD 记录编译时间基线：`time cmake --build build`

## 2. Phase 2: 标准库 include 精简（30 min）

- [ ] 2.1 MUST 分析 `<iostream>` 使用方式：如仅 .cpp 使用 → 移到 .cpp
- [ ] 2.2 MUST 分析 `<any>` 使用方式：若仅 std::any 存储 → 移到 .cpp
- [ ] 2.3 MUST 分析 `<array>` 使用方式：值类型成员 → 保留；仅 .cpp 使用 → 移到 .cpp
- [ ] 2.4 MUST 分析 `<map>`, `<unordered_map>` 使用方式：值类型成员 → 保留
- [ ] 2.5 MUST 分析 `<memory>`（unique_ptr/shared_ptr）使用方式：值类型成员 → 保留
- [ ] 2.6 MUST 分析 `<stack>`, `<string>`, `<vector>` 使用方式：值类型成员 → 保留
- [ ] 2.7 MUST 验证：`cmake --build build` 通过
- [ ] 2.8 MUST 验证：无新增编译 warning
- [ ] 2.9 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 2.10 git commit -m "refactor(thread_context): reduce standard library includes"

## 3. Phase 3: 项目头文件前向声明替代（45 min）

- [ ] 3.1 MUST 分析 14 个项目头文件每个类型的使用方式（值类型成员 vs 指针/引用参数）
- [ ] 3.2 MUST 对指针/引用类型添加前向声明到头文件集中声明区域：
  ```cpp
  // --- Forward declarations ---
  namespace ptx_ir { class StatementContext; }
  namespace ptxsim { class ExecState; }
  // ...
  ```
- [ ] 3.3 MUST 逐个移除可前向声明类型对应的 `#include`（每次 2-3 个 + ctest 验证）
- [ ] 3.4 MUST NOT 将值类型参数的 include 改为前向声明
- [ ] 3.5 MUST NOT 改 inline 函数体内使用的方法所需类型的 include
- [ ] 3.6 MUST 验证：`cmake --build build` 通过 + 无 warning
- [ ] 3.7 MUST 验证：`grep -c '^#include' include/ptxsim/thread_context.h` ≤ 15
- [ ] 3.8 git commit -m "refactor(thread_context): replace project headers with forward declarations"

## 4. Phase 4: 最终验证（10 min）

- [ ] 4.1 MUST 验证：`include/ptxsim/thread_context.h` include 数 ≤ 15
- [ ] 4.2 MUST 验证：所有使用 thread_context.h 的文件编译通过
- [ ] 4.3 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 4.4 MUST 验证：无新增编译 warning
- [ ] 4.5 SHOULD 验证编译时间缩短（与基线对比）

## 5. 应用阶段

- [ ] 5.1 MUST 运行 `openspec validate reduce-thread-context-includes-v2 --strict`
- [ ] 5.2 MUST 通过所有验证后 archive

## 验收

- include 数量 ≤ 15（25 → ≤15，减少 ≥40%）
- 全量编译通过 + 无 warning
- 所有 ctest 通过
- inline 函数体依赖完整保留

## 关键约束（MUST/MUST NOT）

- MUST 编译通过（所有使用 thread_context.h 的文件）
- MUST NOT 将值类型参数/成员的 include 改为前向声明
- MUST NOT 改 inline 函数体方法调用所需类型的 include
- SHOULD 保持 include 分组（标准库 / 项目 / 第三方）
- SHOULD 复用 commit `edb9302e` 已建立的 forward declarations 注释模式