# dedupe-ptx-op-def-format - Tasks

## Task List

### Phase 1: 修改 ptx_visitor_atom.cpp 注释（10 min）

- [ ] 1.1 MUST 将 `src/ptx_parser/ptx_visitor_atom.cpp:28` 的硬编码格式引用：
  ```
  * that ptx_op.def (X(S_ATOM, atom, Atom, 3, ATOM_INSTR, atomic)) requires.
  ```
  替换为泛化描述：
  ```
  * that ptx_op.def (S_ATOM entry, op_count defined in ptx_op.def) requires.
  * Note: opcount parameter is passed via X-Macro expansion; do not hardcode
  * the operand count here - it is derived from ptx_op.def at compile time.
  ```
- [ ] 1.2 MUST 编译验证：`cmake --build build` 通过
- [ ] 1.3 MUST grep 验证：`grep -n "X(S_ATOM" src/ptx_parser/ptx_visitor_atom.cpp` 返回 0 结果

### Phase 2: 验证（20 min）

- [ ] 2.1 MUST atom 相关测试通过：`cd build && ctest -R "atom" --output-on-failure`
- [ ] 2.2 MUST PTX 语法测试通过：`./tests/ptx/test_all_ptx.sh`
- [ ] 2.3 MUST 全量回归通过：`cd build && ctest --output-on-failure`

### Phase 3: 提交

- [ ] 3.1 git commit -m "refactor(parser): remove hardcoded ptx_op.def format in atom visitor macro"
- [ ] 3.2 MUST 运行 `openspec validate dedupe-ptx-op-def-format --strict`
- [ ] 3.3 MUST 通过所有验证后 archive 此 change

## 验收

- `grep -n "X(S_ATOM" src/ptx_parser/ptx_visitor_atom.cpp` 返回 0 结果
- 编译通过
- atom 指令解析测试通过
- PTX 语法测试通过

## 关键约束（MUST/MUST NOT）

- MUST 保持 atom 指令解析行为不变
- MUST NOT 修改 `ptx_op.def` 条目
- MUST NOT 修改其他 visitor 文件
- MUST NOT 改变 atom 宏体逻辑
- SHOULD 用 constexpr/static_assert 替代纯注释引用
