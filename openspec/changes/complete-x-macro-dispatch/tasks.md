# complete-x-macro-dispatch - Tasks

## Task List

### Phase 1: 审计现有展开点（30 min）

- [ ] 1.1 MUST 记录全部 10 个 `ptx_op.def` 展开点的位置和 X 宏参数命名：
  ```bash
  grep -rn '#include.*ptx_op.def' src/ include/ | grep -v 'AGENTS.md'
  ```
- [ ] 1.2 MUST 确认每个展开点都有 `#undef X` 清理
- [ ] 1.3 MUST 记录 TCGEN05_INSTR 特殊跳过逻辑的位置和实现方式
- [ ] 1.4 MUST 确认 `ptx_op.def` 有 106 个条目：`grep -c '^X(' include/ptx_ir/ptx_op.def`

### Phase 2: 统一 X 宏参数命名（45 min）

- [ ] 2.1 MUST 统一 `include/ptx_ir/ptx_types.h:20` 的 X 宏参数：
  - 当前: `X(enum_val, struct_name, str, opcount, _, instr_kind)`
  - 统一为: `X(enum_val, op_name, op_str, op_count, struct_kind, instr_kind)`
- [ ] 2.2 MUST 统一 `include/ptx_parser/ptx_parser.h:162` 的 X 宏参数：
  - 当前: `X(openum, opname, opstr, opcount, struct_kind)` (5 参数，Listener 不需 instr_kind)
  - 统一为: `X(enum_val, op_name, op_str, op_count, struct_kind)` (保持 5 参数)
- [ ] 2.3 MUST 统一 `include/ptx_parser/ptx_visiter.h:104` 的 X 宏参数：
  - 当前: `X(openum, opstr, opname, opcount, struct_kind, instr_kind)`
  - 统一为: `X(enum_val, op_name, op_str, op_count, struct_kind, instr_kind)`
- [ ] 2.4 MUST 统一 `src/ptx_ir/statement_context.cpp:11` 的 X 宏参数：
  - 当前: `X(stype, opkind, opname, count, struct_kind, instr_kind)`
  - 统一为: `X(enum_val, op_name, op_str, op_count, struct_kind, instr_kind)`
- [ ] 2.5 MUST 统一 `src/ptx_parser/ptx_parser.cpp:1046` 的 X 宏参数：
  - 当前: `X(openum, opname, opstr, opcount, struct_kind)` (5 参数)
  - 统一为: `X(enum_val, op_name, op_str, op_count, struct_kind)` (保持 5 参数)
- [ ] 2.6 MUST 统一 `src/ptx_parser/ptx_visitor.cpp:590` 的 X 宏参数：
  - 当前: `X(openum, opstr, opname, opcount, _, instr_kind)`
  - 统一为: `X(enum_val, op_name, op_str, op_count, struct_kind, instr_kind)`
- [ ] 2.7 MUST 统一 `src/ptx_parser/ptx_visitor_dispatch.cpp:44` 的 X 宏参数：
  - 当前: `X(openum, opstr, opname, opcount, struct_kind, instr_kind)`
  - 统一为: `X(enum_val, op_name, op_str, op_count, struct_kind, instr_kind)`
- [ ] 2.8 MUST 每步编译验证：`cmake --build build` 通过

### Phase 3: 文档化 TCGEN05 跳过逻辑（15 min）

- [ ] 3.1 MUST 完善 `include/ptxsim/instruction_handlers.h:132-141` 的 TCGEN05_INSTR 注释：
  - 说明跳过原因（11 个 S_TCGEN05_* 共享 1 个 Tcgen05Handler）
  - 说明手动声明 Tcgen05Handler 的原因
- [ ] 3.2 MUST 完善 `src/ptxsim/instruction_handlers.cpp:163-170` 的 TCGEN05_INSTR 注释：
  - 说明 `IMPLEMENT_TCGEN05_INSTR_HANDLER` 为 no-op 的原因
  - 说明 `processTcgen05Operation` 的 weak stub 机制

### Phase 4: 全量验证（30 min）

- [ ] 4.1 MUST Debug 构建通过：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build`
- [ ] 4.2 MUST Release 构建通过：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] 4.3 MUST 全量测试通过：`cd build && ctest --output-on-failure`
- [ ] 4.4 MUST PTX 语法测试通过：`./tests/ptx/test_all_ptx.sh`
- [ ] 4.5 SHOULD 编译时间不增加（对比修改前后 `time cmake --build build`）
- [ ] 4.6 SHOULD 验证 106 个 handler 正确注册

### Phase 5: 提交

- [ ] 5.1 git commit -m "refactor(xmacro): unify X macro parameter naming across 10 expansion points"
- [ ] 5.2 MUST 运行 `openspec validate complete-x-macro-dispatch --strict`
- [ ] 5.3 MUST 通过所有验证后 archive 此 change

## 验收

- 10 个展开点的 X 宏参数命名统一为 `enum_val, op_name, op_str, op_count, struct_kind, instr_kind`
  （Listener 展开点保持 5 参数，不含 `instr_kind`）
- 所有 106 个 `ptx_op.def` 条目的 handler 正确注册
- `ctest` 全绿
- PTX 语法测试通过
- 编译时间不增加
- TCGEN05_INSTR 跳过逻辑有完善注释

## 关键约束（MUST/MUST NOT）

- MUST 保持 `ptx_op.def` 的 106 个条目全部正确注册
- MUST 保持 `InstructionFactory::get_handler()` 分派逻辑不变
- MUST 保持 TCGEN05_INSTR 跳过逻辑不变
- MUST NOT 修改 `ptx_op.def` 条目本身
- MUST NOT 影响 parser 端的 X-Macro 使用行为（仅统一命名）
- SHOULD 减少 X-Macro 重复展开次数（如可行）
- SHOULD 编译时间不增加
