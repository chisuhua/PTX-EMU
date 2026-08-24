# statement-ir-public Specification

## Purpose
TBD - created by archiving change ptxemu-public-device-api. Update Purpose after archive.
## Requirements
### Requirement: Phase 0 闭包净化 — 2 污染点 MUST 消除 ✅ **COMPLETE**

晋升前 MUST 消除以下 2 个非纯数据污染点:

**污染点 A**: `include/ptx_ir/operand_context.h:59` — `mutable void *operand_phy_addr = nullptr` ✅ **REMOVED (commit 1fb15d89)**
- 替代方案 (per Metis MUST-RESOLVE #4): ThreadContext-local index-keyed cache (`std::vector<void *> operand_phy_cache_;`)
- 实施 4 commits (d8b6ca56/a6c9bdaf/66ca4875/1fb15d89): add cache → dual-write → migrate READ → remove field
- OperandContext::setPhyAddr/invalidatePhyAddr methods 已移除
- 8 active sites 全迁移 (5 WRITE dual-write / 3 READ cache-first or removed)

**污染点 B**: `include/ptx_ir/statement_context.h:306` — `InstructionState state = InstructionState::READY` ✅ **REMOVED (commit 586ea14f)**
- 字段声明后**从未被读写** (per Metis audit)
- `src/ptxsim/instruction_base.cpp:100-102` 注释确认 "do not write to stmt.state here to avoid a data race... The state begins as READY and need not be reset"
- InstructionState enum 保留为 schema placeholder

#### Scenario: Phase 0 净化后 2 污染点 0 出现 ✅
- **WHEN** `git grep "operand_phy_addr" include/ptx_ir/` 或 `git grep "InstructionState state =" include/ptx_ir/`
- **THEN** 0 matches (字段已移除)

#### Scenario: 调度器 invariant 在 Phase 0 净化后保持 ✅
- **WHEN** 实施 Phase 0 净化后跑 PTX-EMU 全量 ctest
- **THEN** 0 回归 (246/246 PASS, 多次验证 33.37s/37.57s/35.82s/35.02s/29.95s)

### Requirement: `StatementContext` 晋升为 `ptxemu::ir::Statement`

5 文件闭包 MUST 整体晋升至 `include/ptxemu/ir/`:

| 源路径 | 目标路径 | 命名空间 |
|---|---|---|
| `include/ptx_ir/statement_context.h` | `include/ptxemu/ir/statement.h` | `ptxemu::ir::Statement` |
| `include/ptx_ir/operand_context.h` | `include/ptxemu/ir/operand_context.h` | `ptxemu::ir::OperandContext` |
| `include/ptx_ir/ptx_types.h` | `include/ptxemu/ir/ptx_types.h` | `ptxemu::ir::Qualifier` 等 |
| `include/ptxsim/execution_types.h` | `include/ptxemu/ir/execution_types.h` | `ptxemu::ir::InstructionState` (only this exported) |
| `include/ptx_ir/ptx_qualifier.def` | | (X-Macro) |
| `include/ptx_ir/ptx_op.def` | | (X-Macro) |

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

### Requirement: `BarWarpSyncInstr::reconvergenceLabel` dead code 删除 ✅ **COMPLETE**

Promotes MUST BEFORE 5 file promotion: `reconvergenceLabel` field MUST be removed as it is dead code per Oracle 审计 (commit 602bfc30 + 359579ec).
- ✅ `include/ptx_ir/statement_context.h:229` 字段已删除
- ✅ `src/ptx_parser/ptx_visitor_barrier.cpp:119` writer 已删除
- ✅ `tests/unit/test_ptxir_serialization.cpp` 3 处 aggregate initializer 已修复

#### Scenario: 0 caller 引用 reconvergenceLabel ✅
- **WHEN** `git grep "reconvergenceLabel" -- ':!docs/' ':!openspec/'`
- **THEN** 0 matches ✅

