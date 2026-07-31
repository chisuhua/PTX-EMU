## Context

### 现状

当前 PTXIR 加载流程：

```
generate_ptxir(): PTX → ANTLR → kernelStatements → serialize → .ptxir
                                                    ❌ reconvergence_pc 未填充
load_ptxir():    .ptxir → deserialize → kernelStatements → CFG build → postDom → 填充 reconvergence_pc → 仿真
                                        ✅ 快速读盘          ❌ O(n×iterations) 每次重算
```

**两个问题**：
1. `write_barrier()` 不写 `reconvergence_pc`（Reader 也不读），而 `write_branch()` 已写/已读 — 不一致
2. `generate_ptxir()` 不在序列化前填充 `reconvergence_pc`，导致 `load_ptxir()` 每次重算 CFG

### 关键事实

- `BarrierInstr` 结构体已有 `int reconvergence_pc = -1` 字段（`statement_context.h:76`）
- `BranchInstr` 的 `reconvergence_pc` 已正确序列化（Writer `write_branch()`:235 + Reader `read_instruction()` S_BRA:178）
- `BarrierInstr` 的 `reconvergence_pc` 未序列化（Writer `write_barrier()`:244-245 只写 barId；Reader S_BAR:198-207 不读）
- `ptxir_format.h` 中 `BARRIER_ENCODED_SIZE` 为 `sizeof(uint16_t) + sizeof(int32_t)`（缺 `reconvergence_pc` 的 `sizeof(int32_t)`）
- `PTXIR_VERSION = 2`，需要 bump 以兼容旧格式

### 约束

- **向后兼容**：旧 PTXIR v2 文件（不包含 S_BAR 的 reconvergence_pc）必须能被 Reader 正确加载（就当 reconvergence_pc = -1）
- 不改变仿真语义，只改变 PTXIR 加载路径性能
- 不修改 `BranchInstr` 格式（已正确）

## Goals / Non-Goals

**Goals:**
- PTXIR 生成时计算并嵌入 CFG 后支配树到 `reconvergence_pc` 字段（S_BRA + S_BAR）
- Reader 加载时跳过 CFG 重算，直接使用嵌入值
- 旧格式 PTXIR v2 文件兼容加载

**Non-Goals:**
- 不修改 CFGBuilder 算法本身
- 不修改 `load_ptxir()` 的 API 签名（`apply_cfg` 参数默认值已为 `false`，代码 SSOT，无需变更）
- 不修改 Antlr 解析路径（`generate_ptxir()` 之外的入口不变）

## Decisions

### D1: PTXIR 版本升级 v2 → v3

- `PTXIR_VERSION = 3`
- v2 Reader 路径保留兼容：`read_legacy_v1()` 模式扩展为 `read_v2()` 中也处理缺失 reconvergence_pc
- 旧 v2 文件加载时 S_BAR 的 reconvergence_pc 默认为 -1（仿真器已有 fallback 到 `i + 1`）

### D2: BarrierInstr 序列化格式扩展

**方案**：在 `BarrierInstr` 序列化末尾追加 `reconvergence_pc`（int32_t），与 `BranchInstr` 对齐。

```
v2 S_BAR:  opcode(u16) | barId(i32)
v3 S_BAR:  opcode(u16) | barId(i32) | reconvergence_pc(i32)
```

**Writer 变更**：`write_barrier()` 追加 `write_i32(out_, instr.reconvergence_pc)`

**Reader 变更**：`read_instruction()` S_BAR case 追加：
```cpp
if (version_ >= 3) {
    instr.reconvergence_pc = read_i32(in_);
}
```

### D3: `generate_ptxir()` 嵌入 CFG 计算

在 `generate_ptxir()` 的 `serialize_statements()` 调用前，插入与 `load_ptxir(apply_cfg=true)` 相同的 CFG 计算逻辑（复用 CFGBuilder）。

### D4: `load_ptxir()` 默认跳过 CFG

`apply_cfg` 参数默认值已为 `false`（代码 SSOT `ptxir_serialization.h:27`），无需变更。Grep 确认 `load_ptxir(` 在 `src/`、`tests/` 中**0 个调用方**，更改默认值无风险。保留 `apply_cfg=true` 参数用于旧 v2 文件加载时回退重建。

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|-----------|
| 旧 PTXIR v2 文件 S_BAR 无 reconvergence_pc | 仿真器用默认值 -1 运行 | Reader 版本分支：v2 不读，v3 读；仿真器已有 fallback `i+1` |
| CFG 计算结果在生成时与加载时不一致 | 分支收敛行为改变 | 使用相同 CFGBuilder 版本 + 相同语句输入，结果确定 |
| `generate_ptxir()` 增加 CFG 计算时间 | 生成速度变慢 | CFG 是 O(n) 一次过，n=语句数，对单 kernel 生成几乎无感知 |
| 多 kernel PTX 文件 | 只选择指定 kernel 生成 | 当前 `generate_ptxir()` 已支持 kernel_name 参数 |

## Migration Plan

Phase 1: Writer/Reader 格式扩展（S_BAR + version bump）
Phase 2: `generate_ptxir()` 嵌入 CFG 计算
Phase 3: `load_ptxir()` 默认跳过 CFG + 兼容测试

每个 Phase 独立 commit、独立可 revert。

## Open Questions

- ~~`load_ptxir()` 调用方是否都使用 `apply_cfg=true`？需要 grep 确认后决定默认值策略。~~
- **已解决**（2026-07-31）：Grep 确认 `load_ptxir(` 在 `src/`、`tests/`、`include/` 中 **0 调用方**。默认值 `false` 无风险。

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- N/A：无函数迁移，只修改 Writer/Reader 的序列化逻辑

### 多 Phase 推进
- Phase 1/2/3 独立 commit，失败立即 revert 对应 Phase

### 文档同步
- `src/ptx_ir/ptxir_writer.cpp` 注释更新
- `src/ptx_ir/ptxir_reader.cpp` 注释更新
- `include/ptx_ir/ptxir_format.h` 版本号和常量更新
- `src/ptxir/ptxir_serialization.cpp` 注释更新