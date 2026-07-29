# reduce-thread-context-includes - Tasks

## 1. Phase 1: 基线记录（10 min）

- [ ] 1.1 MUST 记录当前 include 数量：`grep -c '#include' include/ptxsim/thread_context.h`（应为 25）
- [ ] 1.2 MUST 验证当前可编译：`cmake --build build` 通过
- [ ] 1.3 MUST 运行 `cd build && ctest --output-on-failure` 记录基线测试结果（全绿）
- [ ] 1.4 SHOULD 记录编译时间基线：`time cmake --build build`

## 2. Phase 2: 标准库 include 精简（20 min）

- [ ] 2.1 MUST 分析 `<iostream>` 使用方式：确认头文件中是否直接使用 `std::cout`/`std::cerr` 等
- [ ] 2.2 MUST 如 `<iostream>` 仅在 `.cpp` 使用：从头文件移除，确认 `.cpp` 已包含
- [ ] 2.3 MUST 分析 `<cstdint>` 使用方式：确认是否可被其他项目头文件间接包含
- [ ] 2.4 MUST 逐一确认其他标准库 include（`<any>`, `<array>`, `<map>`, `<memory>`, `<stack>`, `<string>`, `<unordered_map>`, `<vector>`）是否有值类型成员使用
- [ ] 2.5 MUST 验证：`cmake --build build` 通过
- [ ] 2.6 MUST 验证：无新增编译 warning
- [ ] 2.7 git commit -m "refactor(thread_context): reduce standard library includes"

## 3. Phase 3: 项目头文件前向声明替代（30 min）

- [ ] 3.1 MUST 分析 14 个项目头文件每个类型的使用方式（值类型成员 vs 指针/引用参数）
- [ ] 3.2 MUST 对指针/引用类型添加前向声明到头文件集中声明区域：
  ```cpp
  // --- Forward declarations ---
  namespace ptx_ir { class StatementContext; }
  namespace ptxsim { class ExecState; }
  // ...
  ```
- [ ] 3.3 MUST 逐个移除可前向声明类型对应的 `#include`（每次 2-3 个）
- [ ] 3.4 MUST 每批移除后验证：`cmake --build build` 通过
- [ ] 3.5 MUST 如出现 "incomplete type" 错误：回退该 include（说明为值类型，不可前向声明）
- [ ] 3.6 MUST 验证：无新增编译 warning
- [ ] 3.7 git commit -m "refactor(thread_context): replace includes with forward declarations"

## 4. Phase 4: .cpp include 补充（10 min）

- [ ] 4.1 MUST 检查 `src/ptxsim/core/thread_context.cpp` 是否缺少从头文件移除的 include
- [ ] 4.2 MUST 补充 `.cpp` 所需的 include（实现中使用的完整类型）
- [ ] 4.3 MUST 验证：`cmake --build build` 通过
- [ ] 4.4 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 4.5 git commit -m "refactor(thread_context): add moved includes to .cpp"

## 5. Phase 5: 最终验证（10 min）

- [ ] 5.1 MUST 验证：include 数量 ≤ 15
  ```bash
  grep -c '#include' include/ptxsim/thread_context.h  # 应 ≤ 15
  ```
- [ ] 5.2 MUST 验证：全量编译通过 `cmake --build build`
- [ ] 5.3 MUST 验证：无新增编译 warning
- [ ] 5.4 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 5.5 SHOULD 验证：include 分组保持（标准库 / 项目头文件）

## 6. 应用阶段

- [ ] 6.1 MUST 运行 `openspec validate reduce-thread-context-includes --strict`
- [ ] 6.2 MUST 通过所有验证后 archive 此 change

## 验收

- include 数量减少 ≥ 30%（25 -> ≤ 15）
- 全量编译通过
- 无新增编译 warning
- ctest 全绿
- ThreadContext public API 不变

## 关键约束（MUST/MUST NOT）

- MUST 编译通过（所有使用 thread_context.h 的文件）
- MUST NOT 将值类型参数/成员的 include 改为前向声明
- MUST NOT 改变 ThreadContext 的 public API
- MUST NOT 删除实际需要的 include
- MUST NOT 引入新的头文件
- SHOULD 保持 include 分组（标准库 / 项目 / 第三方）
