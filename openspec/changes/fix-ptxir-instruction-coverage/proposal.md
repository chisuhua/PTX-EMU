## Why

PTXIR 序列化工具链存在指令覆盖缺口。审计（2026-07-31，`embed-reconvergence-pc-in-ptxir` 实施后）实测发现：

1. **Reader 仅处理 46/106 个 StatementType enum**：`PtxirReader::read_instruction()` 的 GENERIC_INSTR case 组只覆盖 `S_MOV/S_ADD/S_SUB/S_MUL/S_LD/S_ST/S_SETP/S_CVT` 8 个，其余 45 个 GENERIC_INSTR 指令（`S_FMA, S_DIV, S_SIN, S_AND, S_SHL, S_POPC, S_CVTA, S_MUL24, S_SELP, S_LOP3` 等）反序列化时抛 `Unknown StatementType`。加上 `S_BRX`（BRANCH 类）、`S_TRAP/S_BRK/S_BRKPT`（VOID 类）、`S_ACTIVEMASK`、`S_ST_BULK`，共 **60 个 enum 无法 roundtrip**。
2. **Tcgen05Instr 被 writer 静默丢弃**：`Tcgen05Instr` 在 25 成员 InstrVariant 中，但 `PtxirWriter::write_instruction()` 的 if-constexpr 链（24 分支）无 `write_tcgen05()`。序列化时该指令字节流缺 payload（静默数据丢失），加载时抛异常。11 个 Blackwell tcgen05 指令（ADR-0016 核心特性）全部受影响。
3. **真实 kernel roundtrip 失败**：`bench/cute/cute_rmsnorm.ptx`（含 `cvta/fma/div`）→ `generate_ptxir()` 成功 → `load_ptxir()` 抛 `Unknown StatementType: 28`（S_CVTA）。

**为什么现在修**：`embed-reconvergence-pc-in-ptxir` 使 `generate_ptxir()` 嵌入 CFG 结果、`load_ptxir()` 成为快速加载路径（设计 D4），但该路径对大多数真实 kernel 直接抛异常。PTXIR 快速加载的价值取决于工具链能处理全部受支持指令。此前缺口因 `load_ptxir()` 0 调用方而从未暴露（design.md 已确认），现在成为 active debt。

## What Changes

1. **Reader GENERIC_INSTR 全覆盖**：`PtxirReader::read_instruction()` 的 GENERIC_INSTR case 组从 8 个 enum 扩展为全部 53 个 GENERIC_INSTR enum（统一映射到 GenericInstr 反序列化路径，与 writer 的 GenericInstr 序列化对称）
2. **Reader 其他未覆盖 enum**：`S_BRX` → BranchInstr 路径（与 S_BRA 对称）；`S_TRAP/S_BRK/S_BRKPT` → VoidInstr 路径（与 S_EXIT/S_RET 对称）；`S_ACTIVEMASK`、`S_ST_BULK` → GenericInstr 路径
3. **Tcgen05Instr 序列化支持**：writer 增加 `write_tcgen05()`（序列化 qualifiers + operands），reader 增加 `S_TCGEN05_*` case 组（重建 Tcgen05Instr）
4. **真实 kernel roundtrip 测试**：T4/T5 或新增测试使用真实 PTX fixture（如 `tests/ptx/` 或 `bench/cute/` 下的 kernel），确保 `generate_ptxir() → load_ptxir()` 对含 `cvta/fma/tcgen05` 等指令的 kernel 不抛异常

## Capabilities

### New Capabilities
- `ptxir-full-enum-coverage`: PTXIR reader/writer 覆盖全部 106 个 StatementType enum（含 11 个 tcgen05），真实 kernel 可完整 roundtrip

### Modified Capabilities
- `ptxir-coverage-parity`: Reader 覆盖范围从 24 类型扩展为全部 106 enum（含 Tcgen05Instr 对称序列化）
- `ptxir-format-compliance`: Tcgen05Instr 指令编码格式定义（v3 格式中新增序列化布局）

## Impact

- **Affected files**:
  - `src/ptx_ir/ptxir_reader.cpp` — `read_instruction()` 扩展 case 组（GENERIC_INSTR 45 + S_BRX + S_TRAP/S_BRK/S_BRKPT + S_ACTIVEMASK + S_ST_BULK + S_TCGEN05_* 11）
  - `src/ptx_ir/ptxir_writer.cpp` — 新增 `write_tcgen05()` + `write_instruction()` 注册
  - `tests/unit/test_ptxir_serialization.cpp` — 新增全 enum roundtrip 测试 + 真实 kernel roundtrip 测试
- **Performance**: 无影响（仅序列化覆盖扩展）
- **Backward compatibility**: 完全向后兼容 — v3 格式不变（此 change 只补全已有 enum 的读写，不改变任何现有指令编码）
- **No breaking changes** to simulation output semantics
- **Scope boundary**: 不修改 `BarrierInstr` 的 qualifiers/type 序列化（pre-existing debt，`fix-ptxir-barrier-qualifier-serialization` 单独处理）；不修改 CFGBuilder 算法；不修改 `load_ptxir()` API 签名

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性（如适用）
- [ ] baseline 函数所有 set_*/commit_*/force_*/lock_* 调用已列出 — N/A，无状态迁移
- [x] Writer/Reader 对称性核查：现有 24 类型 writer↔reader 已对称（除 Tcgen05Instr 缺失），本 change 补齐 Tcgen05Instr

### 多 Phase 推进（如适用）
- [x] Phase 拆分方案 + 独立 commit 粒度：Phase 1 = Reader GENERIC_INSTR 全覆盖；Phase 2 = Tcgen05Instr 序列化；Phase 3 = 真实 kernel roundtrip 测试 + 全量回归
- [x] 基线 worktree 命令已记录（如需）：`git worktree add .worktrees/baseline-check <baseline-commit>`
- [x] 失败处理策略：任何已有测试回归 → revert 该 Phase，不混入后续 commit

### 文档同步
- [x] AGENTS.md 同步项：`src/ptx_ir/AGENTS.md` 无结构性变更
- [x] ADR 追加段落：无架构变更，不需要新 ADR
- [x] tasks.md Phase 状态变更已说明
