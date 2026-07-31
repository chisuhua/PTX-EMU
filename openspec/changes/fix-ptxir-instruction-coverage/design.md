## Context

### 现状

PTXIR 工具链（writer/reader）的指令覆盖与仿真器支持的 106 个 StatementType enum 不一致：

```
仿真器支持:  ptx_op.def = 106 enum (53 GENERIC + 11 TCGEN05 + 5 VOID + 4 SURFACE + 5 TEXTURE + 3 MBARRIER + ...)
Writer 覆盖: 24 InstrVariant 类型（Tcgen05Instr 缺失 → 静默丢弃）
Reader 覆盖: 46 enum（GENERIC 组只处理 8/53；无 TCGEN05；无 S_BRX/S_TRAP/S_BRK/S_BRKPT/S_ACTIVEMASK/S_ST_BULK）
```

**审计实测**（2026-07-31）：
- `bench/cute/cute_rmsnorm.ptx` → `generate_ptxir()` OK → `load_ptxir()` 抛 `Unknown StatementType: 28`（S_CVTA）
- `tests/ptx/test_divergence_sync_standalone.ptx` → 同样抛 S_CVTA
- `Tcgen05Instr` 序列化时被 `write_instruction()` 的 if-constexpr 链 no-op → 字节流缺 payload

### 关键事实

- GENERIC_INSTR 的 53 个 enum 在 writer 侧全部走 `write_generic()`（统一编码：qualifiers + dst + operands），reader 侧 case 组只列了 8 个（`S_MOV/S_ADD/S_SUB/S_MUL/S_LD/S_ST/S_SETP/S_CVT`）→ 45 个 enum 落到 `default:` 抛异常
- `Tcgen05Instr` 字段：`op_kind`（11 值）+ `qualifiers` + `operands` + `instructionText` + 4 个便捷字段（`cta_group/dtype/num_regs/has_block_scale`，visitor 从 qualifiers 预提取）
- `S_TCGEN05_*` 11 个 enum 与 `Tcgen05OpKind` 11 值 **1:1 对应**（`S_TCGEN05_MMA_WS` ↔ `MMA_WS`），op_kind 可从 stmt.type 派生
- 便捷字段（cta_group 等）是 visitor 从 qualifiers 提取的派生值，可序列化也可由 handler 从 qualifiers 重推导（lessons-learned 失败模式 13 已确认 handler 侧应扫描 qualifiers，不依赖便捷字段）

### 约束

- **向后兼容**：不改变任何现有指令的编码格式（v3 格式不变）；只补全已有 enum 的读写
- 与 writer 的 GenericInstr 编码保持对称（reader 的 GenericInstr 路径已存在，只需扩展 case 列表）
- Tcgen05Instr 的序列化格式对齐同类指令（qualifiers + operands，参考 MbarrierInstr/VoteInstr 模式）

## Goals / Non-Goals

**Goals:**
- Reader 的 GENERIC_INSTR case 组覆盖全部 53 个 enum（统一映射 GenericInstr 反序列化路径）
- Reader 补齐 `S_BRX`（BranchInstr）、`S_TRAP/S_BRK/S_BRKPT`（VoidInstr）、`S_ACTIVEMASK/S_ST_BULK`（GenericInstr）
- Writer 新增 `write_tcgen05()`，reader 新增 `S_TCGEN05_*` case 组，11 个 tcgen05 指令可完整 roundtrip
- 真实 kernel（含 cvta/fma/tcgen05）通过 `generate_ptxir() → load_ptxir()` 不抛异常

**Non-Goals:**
- 不修改 `BarrierInstr` 的 qualifiers/type 序列化（pre-existing debt，`fix-ptxir-barrier-qualifier-serialization` 处理）
- 不修改 CFGBuilder 算法
- 不修改 `load_ptxir()` / `generate_ptxir()` API 签名
- 不序列化 Tcgen05Instr 的 `instructionText`（派生文本，与现有指令一致均不序列化）
- 不序列化便捷字段 `cta_group/dtype/num_regs/has_block_scale`（派生值，handler 从 qualifiers 重推导；如后续需要再扩展格式）

## Decisions

### D1: Reader GENERIC_INSTR case 组扩展为全部 53 个 enum

**方案**：在 `read_instruction()` 的 GenericInstr case 组中，将 8 个 enum 列表扩展为 53 个（即 GENERIC_INSTR 类别的全部 enum），统一走现有 GenericInstr 反序列化代码（qualifiers + dst_reg_id + operands），与 writer `write_generic()` 对称。

```cpp
// BEFORE: 8 个 enum
case S_MOV: case S_ADD: case S_SUB: case S_MUL:
case S_LD: case S_ST: case S_SETP: case S_CVT: {
    // ... GenericInstr 反序列化
}

// AFTER: 全部 53 个 GENERIC_INSTR enum（含 S_CVTA, S_FMA, S_DIV, S_SIN,
//        S_AND, S_SHL, S_POPC, S_MUL24, S_SELP, S_LOP3, S_ACTIVEMASK, S_ST_BULK, ...）
case S_MOV: case S_ADD: ... case S_CVT:
case S_CVTA: case S_PRMT: case S_ISSPACEP: case S_MAPA: case S_ALLOCA:
case S_MUL24: case S_DIV: case S_REM: case S_MIN: case S_MAX:
case S_NEG: case S_ABS: case S_MAD: case S_MAD24: case S_FMA:
case S_ADDC: case S_SUBC: case S_SAD: case S_COPYSIGN: case S_TESTP:
case S_TANH: case S_AND: case S_OR: case S_XOR: case S_NOT:
case S_SHL: case S_SHR: case S_SHF: case S_BFE: case S_LOP3:
case S_SET: case S_SELP: case S_SLCT: case S_CNOT:
case S_SIN: case S_COS: case S_LG2: case S_EX2: case S_RCP:
case S_RSQRT: case S_SQRT: case S_POPC: case S_CLZ:
case S_ACTIVEMASK: case S_ST_BULK: {
    // ... 同一 GenericInstr 反序列化代码
}
```

### D2: Reader 补齐 Branch/Void 缺失 enum

- `S_BRX` → 加入 S_BRA case 组（BranchInstr 反序列化）
- `S_TRAP/S_BRK/S_BRKPT` → 加入 S_EXIT/S_RET case 组（VoidInstr 反序列化）

### D3: Tcgen05Instr 序列化（writer + reader 对称）

**Writer** 新增 `write_tcgen05()`，序列化 qualifiers + operands（with_imm=true，与 MbarrierInstr 的格式类似但保留立即数，参考现有 `write_mbarrier` 模式中 operands 的处理 —— 统一用 `write_qualifiers()` + `write_operands(instr.operands, true)`，与 CallInstr 一致以保留 ImmOperand）：

```cpp
void PtxirWriter::write_tcgen05(const Tcgen05Instr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, true);
}
```

并在 `write_instruction()` 的 if-constexpr 链注册：
```cpp
else if constexpr (std::is_same_v<T, Tcgen05Instr>) { write_tcgen05(instr); }
```

**Reader** 新增 `S_TCGEN05_*` case 组（11 个 enum → Tcgen05Instr），重建 qualifiers + operands，op_kind 从 stmt.type 派生（1:1 映射）：

```cpp
case S_TCGEN05_ALLOC: case S_TCGEN05_DEALLOC: case S_TCGEN05_RELINQUISH:
case S_TCGEN05_LD: case S_TCGEN05_ST: case S_TCGEN05_CP:
case S_TCGEN05_MMA: case S_TCGEN05_MMA_WS: case S_TCGEN05_COMMIT:
case S_TCGEN05_WAIT: case S_TCGEN05_FENCE: {
    Tcgen05Instr instr;
    instr.op_kind = tcgen05OpKindFromType(type);  // S_TCGEN05_* → Tcgen05OpKind 1:1
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id != 0xFFFFFFFF) {
            std::string name = (id < string_table_.size()) ? string_table_[id] : "";
            instr.operands.emplace_back(RegOperand{name, -1});
        }
    }
    stmt.data = instr;
    break;
}
```

注：operands 反序列化统一按 RegOperand 重建（与现有 GenericInstr/MbarrierInstr 模式一致；ImmOperand 值以 string-id 形式存在 string table 中，可扩展为按需读取，但为对齐现有实现先保持 RegOperand 模式 —— 若现有 writer 以 with_imm=true 写入，reader 需对应读取，具体实现以实际编码为准，测试锁定 roundtrip）。

### D4: 真实 kernel roundtrip 测试

在 `tests/unit/test_ptxir_serialization.cpp` 增加：
- **全 enum roundtrip 测试**：对每个 GENERIC_INSTR enum 构造 GenericInstr 并 roundtrip（用 X-Macro 展开或逐一手写 case）
- **真实 kernel 测试**：使用 `tests/ptx/` 或 `bench/cute/` 下含 `cvta/fma/div` 的 kernel fixture，验证 `generate_ptxir() → load_ptxir()` 不抛异常且语句数 > 0
- **Tcgen05 roundtrip 测试**：构造含 qualifiers + operands 的 `Tcgen05Instr`（如 S_TCGEN05_MMA），roundtrip 后断言 op_kind/qualifiers 正确

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|-----------|
| GENERIC_INSTR case 组手动列出 53 个 enum 易漏 | 编译期不报错，运行时仍抛异常 | 用 `static_assert` 或全 enum roundtrip 测试锁定（测试遍历 ptx_op.def 全部 enum） |
| Tcgen05Instr operands 编码不对称（with_imm 与 RegOperand 重建） | roundtrip 数据不一致 | TDD 测试构造含 ImmOperand 的 Tcgen05Instr 锁定 roundtrip；如不对称则统一编码决策 |
| 便捷字段（cta_group 等）未序列化 | 加载后 handler 读到默认值 | 与 lessons-learned 失败模式 13 一致：handler 必须从 qualifiers 扫描，不依赖便捷字段 |
| `S_TCGEN05_MMA_WS` op_kind 派生错误 | 执行行为错误 | 派生函数用 1:1 显式映射 + 测试断言 op_kind |
| 真实 kernel fixture 解析失败（如 bench PTX 需特殊语法） | 测试脆弱 | 优先使用 `tests/ptx/` 下已验证可解析的 fixture（`test_divergence_sync_standalone.ptx` 含 cvta） |

## Migration Plan

Phase 1: Reader GENERIC_INSTR 全覆盖（45 enum 扩展 + S_BRX + S_TRAP/S_BRK/S_BRKPT + S_ACTIVEMASK + S_ST_BULK）
Phase 2: Tcgen05Instr 序列化（writer write_tcgen05 + reader S_TCGEN05_* case + op_kind 派生）
Phase 3: 全 enum + 真实 kernel roundtrip 测试 + 全量回归

每个 Phase 独立 commit、独立可 revert（失败立即 revert 该 Phase，不混入后续 commit）。

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- N/A：无函数迁移，只扩展 writer/reader 的 case 分派

### 多 Phase 推进
- Phase 1/2/3 独立 commit，失败立即 revert 对应 Phase
- 基线 worktree（如需）：`git worktree add .worktrees/baseline-check <baseline-commit>`

### 文档同步
- `src/ptx_ir/ptxir_writer.cpp` / `ptxir_reader.cpp` 注释更新（case 组覆盖说明）
- `openspec/specs/ptxir-coverage-parity` spec 更新（106 enum 全覆盖要求）
