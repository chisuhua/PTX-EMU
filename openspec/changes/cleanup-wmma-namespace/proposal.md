# Cleanup wmma Namespace (complete removal of pre-Blackwell path)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 changes**:
>   - `archive/2026-07-06-implement-tcgen05-syntax-ir/` (Change-1, archived)
>   - `extend-blackwell-tcgen05-infra` (Change-2, pending)
>   - `implement-tcgen05-handlers` (Change-3, pending)
> **4-Change 拆分**: 本 change 是第 4 步(最终,共 4 步),完全删除 wmma 命名空间
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §6(artifacts-first)+ §20(已实施但未清理)

## Why

Change-1 至 Change-3 建立了独立 tcgen05 命名空间,但 `wmma.cpp` 中**仍残留** pre-Blackwell `wmma.*` 路径(`S_WMMA` 枚举、`WmmaInstr` struct、`makeWmmaInstr` 工厂、`is_tcgen05_*` qualifier helpers、`Q_TCGEN05_*` 4 个 stub、`matrixInst: wmmaInst` grammar rule、4 个 Q_TCGEN05_* qualifier、tests/integration/tcgen05/ 2 个旧测试等)。

per `ptx-lessons-learned` §20 "已实施但未清理模式",这种状态会:
1. 误导 future maintainer(看不出哪个路径是 active)
2. 累积 technical debt(qualifier 命名空间分裂、enum 重复)
3. 阻止 ADR-0016 "Blackwell-only" 约束的彻底执行(虽然 throw 兜底,但代码上仍存在)

Change-4 是 4-change 拆分的**最终清理**,无新功能。

## What Changes

### 删除

| 项 | 位置 | 原因 |
|---|------|------|
| `S_WMMA` enum | `include/ptx_ir/ptx_op.def:127`(原) | wmma 路径不再需要 |
| `WmmaInstr` struct | `include/ptx_ir/statement_context.h` | 同上 |
| `makeWmmaInstr` 工厂 | `include/ptx_ir/statement_factory.h:265-275` | 同上 |
| `WmmaType` enum | `include/ptx_ir/ptx_types.h:30` | 同上 |
| `wmmaInst` / `wmmaOp` / `wmmaLayout` / `wmmaShape` / `wmmaKind` 5 grammar rules | `src/grammar/ptxInstructions.g4:424-433` | 已被 `tcgen05Inst` 替代 |
| `matrixInst: wmmaInst` rule | 同上 | 同上 |
| `Q_TCGEN05_LD/ST/COMMIT/WAIT` 4 stub qualifiers | `include/ptx_ir/ptx_qualifier.def:193-197` | 已被独立 IR 枚举替代 |
| `src/ptxsim/instructions/wmma.cpp` 整个文件 | `src/ptxsim/instructions/` | pre-Blackwell 路径已永久 throw |
| `VISITOR_WMMA_INSTR` macro | `include/ptx_parser/ptx_visitor_categories.h:14-15` | 不再使用 |
| `IMPLEMENT_WMMA_INSTR_HANDLER` macro | `src/ptxsim/instruction_handlers.cpp:110-120` | 同上 |
| `WMMA_INSTR` struct_kind | `include/ptx_ir/ptx_op.def`(原 S_WMMA 行) | 同上 |
| 旧集成测试 `tests/integration/tcgen05/test_tcgen05_*.cpp` | (Change-3 已迁移) | 不需要删除,只需 S_WMMA 引用已换为 S_TCGEN05_* |

### 修改

| 文件 | 范围 |
|------|------|
| `src/grammar/ptxInstructions.g4` | 移除 `wmmaInst` 系列 rule(已在新 `matrixInst: tcgen05Inst` 中) |
| `src/grammar/ptxLexer.g4` | 移除 `WMMA: 'wmma'` token |
| `include/ptx_ir/ptx_op.def` | 移除 `X(S_WMMA, ...)` 行(若未在 Change-1 移除) |
| `include/ptx_ir/ptx_types.h` | 移除 `enum WmmaType` + WmmaType::* 使用 |
| `include/ptx_ir/statement_context.h` | 移除 `struct WmmaInstr` + `InstrVariant` 中的 WmmaInstr |
| `include/ptx_ir/statement_factory.h` | 移除 `makeWmmaInstr` |
| `include/ptx_parser/ptx_visitor_categories.h` | 移除 `VISITOR_WMMA_INSTR` macro |
| `include/ptx_parser/ptx_visiter.h` | 移除 `VISITOR_DECL_WMMA_INSTR` |
| `include/ptx_parser/ptx_parser.h` | 移除 `STATEMENT_DECL_WMMA_INSTR` |
| `src/ptxsim/instruction_handlers.cpp` | 移除 `IMPLEMENT_WMMA_INSTR_HANDLER` |
| `src/ptxsim/instructions/AGENTS.md` | 移除 wmma.cpp 描述 |
| `src/CMakeLists.txt` | 移除 wmma.cpp 引用 |
| 多个 `src/**/CMakeLists.txt` | 移除相关引用 |
| 根 `AGENTS.md` | 移除 "WMMA stub throw" 描述,标注 "pre-Blackwell 永久拒绝" |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 Change-4 archive commit 引用 + 验证 pre-Blackwell 路径完全消失 |

### 不修改

- `S_TCGEN05_*` 11 个枚举(Change-1 已建立)
- `Tcgen05Instr` struct(Change-1 已建立)
- `tcgen05.cpp` handler(Change-3 已实施)
- 4 个基础设施子系统(TMA/TMEM/Cluster/TcQueue)
- `tests/integration/tcgen05/`(已迁移,内容已更新)
- `tests/e2e/kernel/test_blackwell_gemm.cu` 标签(`[e2e][tcgen05][gemm][sm_100]` 仍准确,因 Change-3 真实实现)

## Non-Goals

### 显式拒绝

- ❌ 不实现任何 pre-Blackwell WMMA 路径(ADR-0016 永久拒绝)
- ❌ 不修改 Change-3 实施的 5 个 tcgen05 handler
- ❌ 不修改 4 个基础设施子系统
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8
- ❌ 不重命名 `tests/integration/tcgen05/` 目录(目录名仍准确)

### 范围限制

- 仅做**删除**操作,无新功能
- 不修复 Change-3 残留的 bug(独立 follow-up)
- 不重命名 `tests/ptx/dummy*sm_80.ptx` 等 pre-Blackwell 测试文件(这些不是 WMMA 指令,只是编译目标)
- 不动 `tests/e2e/kernel/test_blackwell_gemm.cu`(用 float,非 tcgen05 路径)

## Goals

### Phase 1: 删除 dead code(独立 commit)

1. 删除 `wmma.cpp` 整个文件
2. 删除 `S_WMMA` enum + `WmmaInstr` + `WmmaType`
3. 删除 5 个 wmma grammar rules + `WMMA` lexer token
4. 删除 4 个 `Q_TCGEN05_*` stub qualifiers
5. 删除 `VISITOR_WMMA_INSTR` / `IMPLEMENT_WMMA_INSTR_HANDLER` / `VISITOR_DECL_WMMA_INSTR` macros
6. 全部 CMakeLists.txt 同步更新

### Phase 2: 验证零回归(独立 commit)

1. `cmake --build build` 全量编译通过
2. `ctest --output-on-failure` 全量通过
3. `test_all_ptx.sh` 13 个 tcgen05 fixtures 通过
4. 现有 4 个 pre-Blackwell .ptx 测试文件不再有 wmma 引用(用 `nm` 或 `grep` 验证)

### Phase 3: 文档 + Archive(独立 commit,per Checklist G + I)

1. 根 `AGENTS.md` 更新:pre-Blackwell 标记为 "永久拒绝,无代码路径"
2. `docs/adr/0016-blackwell-only-tcgen05.md` 追加 Change-4 archive 引用
3. `docs/dev-process/lessons-learned.md` 追加 §24 "命名空间清理" 案例
4. archive

## Capabilities

### Modified Capabilities

- `wmma-tensor-core`: 完全删除(不再是 spec,改为 `docs/audits/` 中的历史记录)
- `tcgen05-grammar`: spec 修订(明确 wmma 路径已删除)
- `tcgen05-ir-types`: spec 修订(明确 S_WMMA / WmmaInstr 已删除)
- `tcgen05-parse-tests`: spec 修订(明确 13 个 fixtures 全部 PASS)

### Removed Capabilities

- `wmma-tensor-core`: 完全移除(per ADR-0016 + Change-1 Decision 4)

## Impact

### 删除的代码(预计)

| 项 | LoC 估计 |
|---|---|
| `src/ptxsim/instructions/wmma.cpp` | -564 |
| `S_WMMA` enum + `WmmaInstr` struct + `WmmaType` enum + `makeWmmaInstr` | -30 |
| 5 个 wmma grammar rules | -10 |
| 4 个 Q_TCGEN05_* stub | -4 |
| WMMA_INSTR struct_kind + macros | -20 |
| 文档更新 | +20 |
| **总计** | **-608** (净删除) |

### 验证

- `git grep "S_WMMA\|WmmaInstr\|WmmaType\|makeWmmaInstr\|VISITOR_WMMA_INSTR\|IMPLEMENT_WMMA_INSTR"` 全部零输出
- `git grep "wmmaInst\|wmmaOp" src/grammar/` 零输出(grammar 已清理)
- `git grep "wmma\.cpp" src/CMakeLists.txt` 零输出
- `tests/ptx/test_all_ptx.sh` 33 → 33(tests 数量不变,因 wmma 不在测试范围)

### 影响的文档

- 根 `AGENTS.md`(更新 wmma 状态)
- `src/grammar/AGENTS.md`(删除 wmma 规则说明)
- `src/ptxsim/instructions/AGENTS.md`(删除 wmma.cpp 描述)
- `docs/adr/0016-blackwell-only-tcgen05.md`(追加更新记录)
- `docs/dev-process/lessons-learned.md`(追加 §24)
- `docs/audits/HEALTH-AUDIT-2026-06-21.md`(可选:更新 wmma 相关条目)

## Design-Time Checklist (Lessons-Learned)

### 删除前审计(per Checklist A 反向)

- [x] 列出所有引用点:`grep -rn "S_WMMA\|WmmaInstr\|WmmaType\|makeWmmaInstr\|wmma.cpp" src/ include/ tests/ docs/`
- [x] 确认无 active caller:Change-3 后,所有引用应已迁移到 `S_TCGEN05_*` + `Tcgen05Instr` + `makeTcgen05Instr`
- [x] 文档同步清单:5 个 AGENTS.md + 2 个 ADR + 1 个 lessons-learned

### 多 Phase 推进(3 个 atomic commits)

- [x] Phase 1: 删除 dead code(独立 commit,可独立 revert)
- [x] Phase 2: 验证零回归(独立 commit)
- [x] Phase 3: 文档 + archive(独立 commit,per Checklist G)
- [x] 基线 worktree 计划:`.worktrees/baseline-wmma-cleanup`
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 Phase

### 文档同步(per Checklist I)

- [x] 根 AGENTS.md 同步项已列出
- [x] ADR 追加段落已规划
- [x] lessons-learned §24 预留

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证删除清单完整性、零回归策略
- [ ] 验证 Change-3 已 archive(tests/integration/tcgen05/ 已迁移)
- [ ] 验证 `grep -rn "S_WMMA" src/ include/ tests/` 仅 0 输出(除本 change 删除目标外)
- [ ] 验证 `ctest --output-on-failure` Change-3 baseline 全绿
