# replace-assert-false-with-throw - Tasks

## Task List

### Phase 1: 修改 statement_context.cpp（10 min）

- [ ] 1.1 MUST 在 `src/ptx_ir/statement_context.cpp` 新增 `#include "ptxsim/ptx_exceptions.h"`
- [ ] 1.2 MUST 替换 line 19 的 `assert(false && "Unknown StatementType");` 和 line 20 的 `return "invalid";` 为：
  ```cpp
  throw PtxEmuException(
      "Unknown StatementType: " +
      std::to_string(static_cast<int>(s)));
  ```
- [ ] 1.3 MUST 删除 `#include <cassert>`（文件中无其他 assert 使用）
- [ ] 1.4 MUST 编译验证：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build` 通过

### Phase 2: 全量验证（20 min）

- [ ] 2.1 MUST Release 构建编译通过：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] 2.2 MUST 全量测试通过：`cd build && ctest --output-on-failure`
- [ ] 2.3 MUST grep 验证：`grep -rn "assert(false" src/ include/` 返回 0 结果
- [ ] 2.4 SHOULD statement_context 相关测试通过：`cd build && ctest -R "statement_context\|ptxir" --output-on-failure`

### Phase 3: 提交

- [ ] 3.1 git commit -m "refactor(ir): replace assert(false) with PtxEmuException in S2s()"
- [ ] 3.2 MUST 运行 `openspec validate replace-assert-false-with-throw --strict`
- [ ] 3.3 MUST 通过所有验证后 archive 此 change

## 验收

- `grep -rn "assert(false" src/ include/` 返回 0 结果
- Debug + Release 构建均编译通过
- `ctest` 全绿
- 异常消息包含 `StatementType` 数值

## 关键约束（MUST/MUST NOT）

- MUST 复用 `include/ptxsim/ptx_exceptions.h` 中已有异常类型
- MUST NOT 保留任何 `assert(false)` 路径
- MUST NOT 引入新异常类型
- MUST NOT 改变函数签名
- SHOULD 在异常消息中包含 StatementType 的数值
- SHOULD 删除不再使用的 `#include <cassert>`
