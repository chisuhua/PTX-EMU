# dedupe-ptx-op-def-format - Proposal

## Why

`src/ptx_parser/ptx_visitor_atom.cpp:28` 的宏注释中**硬编码**了 `ptx_op.def` 格式引用：

```
* that ptx_op.def (X(S_ATOM, atom, Atom, 3, ATOM_INSTR, atomic)) requires.
```

该引用是 `ptx_op.def` 中 `S_ATOM` 条目的冗余副本。当 `ptx_op.def` 格式变更时
（如 `op_count` 从 3 改为 4），此注释不会自动同步，导致开发者被误导。

DRY 违反：`ptx_op.def` 是单一真值源（SSOT），atom 宏注释中的格式描述是冗余副本。

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-20`

## What Changes

- **替换** `ptx_visitor_atom.cpp:28` 注释中的硬编码格式引用，改为引用行号或泛化描述
- **验证** `ptx_op.def` 格式不变，atom 指令解析行为不变
- **不修改** `ptx_op.def` 条目本身

## Capabilities

### New Capabilities

（无新增能力--注释/文档改进）

### Modified Capabilities

- `atom-instruction-parsing`: `VISITOR_ATOM_INSTR` 宏注释从硬编码格式引用改为
  泛化描述，消除 DRY 违反

## Impact

**受影响代码**：
- `src/ptx_parser/ptx_visitor_atom.cpp`（1 处注释修改：line 28 区域）

**不受影响**：
- `include/ptx_ir/ptx_op.def`（不修改）
- atom 指令的解析逻辑（宏体不变）
- 其他 visitor 文件

**依赖**：
- 无前置 change 依赖，可独立执行

**工时**: 0.5h（注释修改 + 验证）

## Design-Time Checklist

- [ ] 确认 `ptx_op.def` 中 `S_ATOM` 条目的确切格式和位置
- [ ] 确认 `ptx_visitor_atom.cpp` 宏注释中引用 `ptx_op.def` 的具体行
- [ ] 确认 atom 指令解析行为不受注释修改影响
- [ ] 评估是否可用 `static_assert` 替代纯注释引用（improvement 技术约束建议）
