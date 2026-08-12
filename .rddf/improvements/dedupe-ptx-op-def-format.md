# dedupe-ptx-op-def-format

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-20
**阶段**: default | **分类**: core-impl
**类型**: refactor

## 架构依据

- `src/ptx_parser/ptx_visitor_atom.cpp:28` 的宏中**硬编码** ptx_op.def 格式：
  ```
  X(S_ATOM, atom, Atom, 3, ATOM_INSTR, atomic)
  ```
- 该宏在注释中引用 ptx_op.def 的格式约定（operand count、instruction kind 等），但格式变更时不会自动同步
- DRY 违反：ptx_op.def 是 SSOT，atom 宏中的格式描述是冗余副本

## 范围

- **In Scope**:
  - 将 atom 宏中的硬编码格式引用替换为从 ptx_op.def 自动派生
  - 或在注释中明确标注"此格式引用 ptx_op.def 第 X 行，变更时需同步"
- **Out Scope**:
  - 不修改 ptx_op.def
  - 不改变 atom 指令的解析逻辑
  - 不修改其他 visitor 文件

## 关键场景

- GIVEN ptx_op.def 格式变更, WHEN atom 宏使用新格式, THEN 编译器能检测不一致（而非静默错误）

## 技术约束

- MUST 保持 atom 指令解析行为不变
- MUST NOT 修改 ptx_op.def 条目
- SHOULD 用 constexpr/static_assert 替代纯注释引用

## 验收标准

- 消除 atom 宏中的硬编码 ptx_op.def 格式引用
- 编译通过
- atom 指令解析测试通过
