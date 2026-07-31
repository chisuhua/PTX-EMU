## Why

当前 `load_ptxir()` 在每次加载 PTXIR 时都重新计算 CFG（Control Flow Graph）和后支配树（post-dominator）来填充 `reconvergence_pc`。这导致仿真启动路径中有多余的 O(n×iterations) 计算开销。实际上，PTXIR 生成时 PTX 已经完整解析，可以在序列化阶段一次性计算并嵌入 CFG 结果，使加载路径直接进入仿真，达到最快运行速度。

## What Changes

1. **`generate_ptxir()` 填充 `reconvergence_pc`**：在 PTX 解析后、序列化写入前，调用 CFGBuilder 计算后支配树，填充所有 `S_BRA` 和 `S_BAR` 指令的 `reconvergence_pc` 字段
2. **`write_barrier()` 序列化 `reconvergence_pc`**：Writer 写入 `BarrierInstr` 时增加 `reconvergence_pc` 字段（当前缺失）
3. **`PtxirReader::read_instruction()` 反序列化 `reconvergence_pc`**：Reader 读取 `S_BAR` 时增加 `reconvergence_pc` 字段
4. **`load_ptxir()` 默认跳过 CFG 计算**：`apply_cfg` 参数默认值已为 `false`（代码 SSOT），`load_ptxir(apply_cfg=false)` 直接使用 PTXIR 中嵌入的 `reconvergence_pc`，无需重算
5. **PTXIR 格式版本升级**：`S_BAR` 指令序列化格式变化，需要 bump PTXIR version（v2→v3）或使用兼容格式

## Capabilities

### New Capabilities
- `embed-reconvergence-pc`: PTXIR 二进制格式中嵌入后支配树计算结果，使加载路径直接从二进制恢复 `reconvergence_pc`，无需重新计算 CFG

### Modified Capabilities
- `ptxir-format-compliance`: PTXIR 二进制格式扩展 — `S_BAR` 指令现在包含 `reconvergence_pc` 字段（与 `S_BRA` 对齐）
- `ptxir-tooling-completion`: PTXIR 序列化/反序列化工具链更新，支持新格式

## Impact

- **Affected files**:
  - `src/ptxir/ptxir_serialization.cpp` — `generate_ptxir()` 增加 CFG 计算；`load_ptxir()` 默认跳过重算
  - `src/ptx_ir/ptxir_writer.cpp` — `write_barrier()` 增加 `reconvergence_pc` 序列化
  - `src/ptx_ir/ptxir_reader.cpp` — `read_instruction()` 中 `S_BAR` 分支增加 `reconvergence_pc` 反序列化
  - `include/ptx_ir/ptxir_format.h` — 可能涉及 PTXIR 版本号更新
  - `src/ptx_ir/ptxir_writer.h` — 接口不变
- **Performance**: 加载路径消除 CFG 计算 O(n×iterations) 开销，启动时间更快
- **Backward compatibility**: Reader 需要兼容旧格式（无 `reconvergence_pc` 的 `S_BAR`），或通过版本号分支处理
- **No breaking changes** to simulation output semantics
- **Scope boundary**: S_BAR `qualifiers` 和 `type` 字段在 Writer/Reader 中已丢失（pre-existing debt），不在本 change 范围内。将在后续 `fix-ptxir-barrier-qualifier-serialization` change 中解决

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性（如适用）
- [ ] baseline 函数所有 set_*/commit_*/force_*/lock_* 调用已列出 — N/A，无迁移

### 多 Phase 推进（如适用）
- [ ] Phase 拆分方案 + 独立 commit 粒度已说明
- [ ] 基线 worktree 命令已记录
- [ ] 失败处理策略（revert 该 Phase，不混入后续 commit）已说明

### 文档同步
- [ ] AGENTS.md 同步项已列出
- [ ] ADR 追加段落已规划
- [ ] tasks.md Phase 状态变更已说明