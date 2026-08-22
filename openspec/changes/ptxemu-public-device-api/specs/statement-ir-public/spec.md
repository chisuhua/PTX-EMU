# Spec: statement-ir-public

**Capability**: StatementContext 公共晋升 (Phase 0 净化 + 5 文件闭包晋升)
**Spec 锁定 commit**: Oracle session `ses_fd5ef471cffeWvINOBm5E1GMYd`
**关联**: ptx-lessons-learned §1 (跨模块状态翻译), CppTLM HSK-8 spec §7 Decision 5 (StatementContext 公共化)

## ADDED Requirements

### Requirement: Phase 0 闭包净化 — 2 污染点 MUST 消除

晋升前 MUST 消除以下 2 个非纯数据污染点:

**污染点 A**: `include/ptx_ir/operand_context.h:59` — `mutable void *operand_phy_addr = nullptr`
- MUST 移出 `OperandContext` 值类型
- 推荐方案 (per Metis MUST-RESOLVE #4): ThreadContext-local index-keyed cache (`std::vector<std::vector<void*>> operand_phy_cache_`), 不使用指针-key (避免 `vector<OperandContext>` 元素地址稳定性问题)
- 8 处 `operand_phy_addr` + `setPhyAddr` + `invalidatePhyAddr` 调用点必须审计迁移 (per Metis audit)

**污染点 B**: `include/ptx_ir/statement_context.h:310` — `InstructionState state = InstructionState::READY`
- MUST 移出 `StatementContext` 值类型
- 关键事实 (per Metis audit): 该字段声明后**从未被读写**, 是 cleanup 必删的 dead code
- 之前 proposal.md 声称的 "8+ 处 set_state() 调用点" 实际是 `ThreadContext::set_state(EXE_STATE)`, 使用 `EXE_STATE` 枚举而非 `InstructionState`, 与本字段无关

#### Scenario: Phase 0 净化后 2 污染点 0 出现
- **WHEN** `git grep "operand_phy_addr" include/ptxemu/ir/` 或 `git grep "InstructionState state =" include/ptxemu/ir/`
- **THEN** 0 matches (字段已移出值类型)

#### Scenario: 调度器 invariant 在 Phase 0 净化后保持
- **WHEN** 实施 Phase 0 净化后跑 PTX-EMU 全量 ctest
- **THEN** 0 回归 (调度器行为不变)

### Requirement: `StatementContext` 晋升为 `ptxemu::ir::Statement`

5 文件闭包 MUST 整体晋升至 `include/ptxemu/ir/`:

| 源路径 | 目标路径 | 命名空间 |
|---|---|---|
| `include/ptx_ir/statement_context.h` | `include/ptxemu/ir/statement.h` | `ptxemu::ir::Statement` |
| `include/ptx_ir/operand_context.h` | `include/ptxemu/ir/operand_context.h` | `ptxemu::ir::OperandContext` |
| `include/ptx_ir/ptx_types.h` | `include/ptxemu/ir/ptx_types.h` | `ptxemu::ir::Qualifier` 等 |
| `include/ptxsim/execution_types.h` | `include/ptxemu/ir/execution_types.h` | `ptxemu::ir::InstructionState` (only this exported) |
| `include/ptx_ir/ptx_qualifier.def` | `include/ptxemu/ir/ptx_qualifier.def` | (X-Macro) |
| `include/ptx_ir/ptx_op.def` | `include/ptxemu/ir/ptx_op.def` | (X-Macro) |

#### Scenario: 5 文件自洽 include
- **WHEN** `g++ -fsyntax-only -I include/ptxemu/ir include/ptxemu/ir/statement.h`
- **THEN** 0 编译错误 (闭包自洽)

#### Scenario: 旧路径 forwarding header 一个 release 周期
- **WHEN** 实施 Phase 1 完成后读 `include/ptx_ir/statement_context.h`
- **THEN** 内容是 `#pragma once + #include <ptxemu/ir/statement.h> + namespace ptx_ir = ptxemu::ir;` (forwarding)

### Requirement: 20 struct 字段零实现层头污染

公共晋升后, 20 个 IR 值类型 struct 的字段类型 MUST 仅引用:
- `ptxemu::ir::Qualifier` / `ptxemu::ir::OperandContext` / `ptxemu::ir::InstructionState`
- std 容器 (`std::string` / `std::vector` / `std::optional`)
- 本地 enum (内嵌定义, 如 `Tcgen05OpKind` / `Tcgen05Dtype`)

禁止引用: `tcgen05_mma_instr` 等实现层类型, 或 `src/ptxsim/instructions/*.h` (handler 类型)

#### Scenario: 20 struct 字段类型 100% 闭包内自洽
- **WHEN** Phase 1 实施后 `git grep -E "(class|struct)\s+\w+\s*\{" include/ptxemu/ir/statement.h | wc -l`
- **THEN** >= 20 (20 struct 全部存在)

### Requirement: `BarWarpSyncInstr::reconvergenceLabel` dead code 删除

晋升前 MUST 删除 `include/ptx_ir/statement_context.h:229` 的 `std::string reconvergenceLabel` 字段 (dead code per Oracle 审计)。

#### Scenario: 0 caller 引用 reconvergenceLabel
- **WHEN** 晋升前 `git grep "reconvergenceLabel" -- ':!docs/' ':!openspec/'`
- **THEN** 0 matches (除定义点外)

## REMOVED Requirements

无 — 本 change 是增量晋升, 不删除任何 IR 行为。

## RENAMED Requirements

无。
