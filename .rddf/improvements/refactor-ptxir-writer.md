# refactor-ptxir-writer

> **状态**: RESOLVED(2026-08-26 验证)
> **解决 commit**: `b21c875d` refactor(ptxir_writer): extract per-type helper methods
> **验证证据**: `src/ptx_ir/ptxir_writer.cpp:237-267` `write_instruction()` 现为 31 行纯分发逻辑(`stmt.visit` + `if constexpr` 链调用 per-type 函数),满足验收标准 "< 50 行";25 个独立 `write_*()` 序列化函数位于 lines 269-411;`ctest -R ptxir` 实测 14/14 通过(含 `unit_ptxir_serialization` round-trip)。

**优先级**: P2 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-4
**阶段**: default | **分类**: core-impl
**类型**: refactor

## 架构依据

- `src/ptx_ir/ptxir_writer.cpp::write_instruction()` 函数实测 **232 行**（始于 line 129），处理 31 个 `is_same_v` variant 分支
- 函数内联展开所有指令类型的序列化逻辑，单函数承载全部 IR 类型的写入职责
- PTXIR 是预解析 PTX kernel 的快速加载格式，序列化正确性直接影响 kernel 加载正确性

## 范围

- **In Scope**:
  - 将 write_instruction() 的 31 个 variant 分支提取为 per-type 序列化函数
  - 按指令类别分组（branch / arithmetic / memory / barrier / tcgen05 等）
  - 保持 PTXIR 二进制格式不变（writer 格式变更必须同步 reader）
- **Out Scope**:
  - 不修改 PTXIR reader 端
  - 不改变二进制格式（field order / size / endianness）
  - 不动 ptx_op.def X-Macro 定义

## 关键场景

- GIVEN 序列化函数拆分, WHEN 写入任意指令类型, THEN 二进制输出与拆分前逐字节一致
- GIVEN 所有 31 个 variant 分支, WHEN 提取完成, THEN 每个分支有独立函数且被正确调用
- GIVEN PTXIR round-trip, WHEN write→read, THEN IR 结构无损恢复

## 技术约束

- MUST 保持 PTXIR 二进制格式 byte-identical（reader 兼容性）
- MUST NOT 改变 write_u16/write_u32/write_string 等底层写入函数
- SHOULD 按指令类别分组到独立文件或 section
- MUST NOT 修改 ptx_op.def

## 验收标准

- write_instruction() 函数 < 50 行（仅分发逻辑）
- PTXIR round-trip 测试全绿（write→read 无损）
- 所有现有 PTXIR 测试通过
- 每个指令类别的序列化函数可独立测试
