# refactor-ptxir-writer - Proposal

## Why

`src/ptx_ir/ptxir_writer.cpp::write_instruction()` 实测 **232 行**（line 129-360），内联展开全部 IR 类型的序列化逻辑。函数体内含 **31 个 `is_same_v` variant 分支**，每个分支处理一种 `StatementContext` 子类型的二进制写入。

核心问题：
- 单函数承载全部 IR 类型的写入职责，违反 SRP
- 232 行远超 250 LOC 中的有效逻辑行（大量重复的 qualifier/operand 循环模式）
- 新增 IR 类型需修改此巨型函数，错误风险高
- 难以对单个指令类别的序列化逻辑独立测试

PTXIR 是预解析 PTX kernel 的快速加载格式，序列化正确性直接影响 kernel 加载正确性。

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-4`

## What Changes

- **拆分** `write_instruction()` 的 31 个 variant 分支为 per-type 序列化函数
- **按指令类别分组** 提取为独立 helper 方法：
  - branch/control: `BranchInstr`, `LabelInstr`, `VoidInstr`, `AbiDirective`
  - barrier/sync: `BarrierInstr`, `BarWarpSyncInstr`, `MembarInstr`, `FenceInstr`, `ReduxSyncInstr`, `MbarrierInstr`
  - generic/arithmetic: `GenericInstr`, `DeclarationInstr`, `PredicatePrefix`
  - warp-collective: `VoteInstr`, `ShflInstr`
  - memory/atomic: `AtomInstr`, `TextureInstr`, `SurfaceInstr`, `ReductionInstr`, `PrefetchInstr`, `CpAsyncInstr`
  - misc: `PragmaInstr`, `DollarNameInstr`, `CallInstr`
- **保持** PTXIR 二进制格式不变（writer 格式变更必须同步 reader，本 change 不动 reader）
- **write_instruction()** 缩减为纯分发逻辑（< 50 行）

## Capabilities

### New Capabilities
- `ptxir-writer-modular-serialization`: per-type 序列化函数拆分，支持独立测试

### Modified Capabilities
（无现有 spec-level 行为变更。本 change 为纯重构，不修改 PTXIR 二进制格式。）

## Impact

**受影响代码**：
- `src/ptx_ir/ptxir_writer.cpp`（主文件，write_instruction 232 -> < 50 行）
- `include/ptx_ir/ptxir_writer.h`（新增 private helper 方法声明）

**不受影响**：
- `src/ptx_ir/ptxir_reader.cpp`（reader 端不动）
- PTXIR 二进制格式（field order / size / endianness 全部不变）
- `include/ptx_ir/ptx_op.def` X-Macro 定义
- `src/ptxir/ptxir_serialization.cpp`（调用接口不变）
- 所有 PTXIR round-trip 测试

**依赖**：
- 无前置 change 依赖，可独立执行
- 不影响 ADR-PTXIR 序列化架构

**工时**: 2-3h（纯机械重构 + round-trip 验证）
