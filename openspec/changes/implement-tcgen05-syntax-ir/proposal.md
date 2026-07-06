# Blackwell tcgen05 独立命名空间(ANTLR 语法 + IR)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **替代路径**: `openspec/changes/archive/2026-07-04-implement-wmma-tensor-core-tcgen05/`
> **4-Change 拆分**: 本 change 是第 1 步(共 4 步),仅交付语法+IR 命名空间;handler 实现在 change-3
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §6(artifacts-first)+ §7(Pre-impl review)+ §20(已实施但未清理)

## Why

`feat/implement-wmma-tensor-core-tcgen05` (archived 2026-07-04) 通过"wmma 路径 + Q_TCGEN05_LD/ST/COMMIT/WAIT qualifier 注入"实现 tcgen05 指令,存在**功能性缺口**(per `ptx-lessons-learned` §20 "已实施但未清理" 表象):

1. **ANTLR grammar 缺少 `tcgen05` lexer token** — 用户写真实 `tcgen05.mma ...` PTX 文本,grammar 拒绝解析,仅能通过测试代码注入 `S_WMMA` 走通
2. **命名空间混淆** — `S_WMMA` / `Q_TCGEN05_*` / `Q_CLUSTER` / `Q_F16` 散落在 wmma 体系,无独立身份;`tests/ptx/test_all_ptx.sh` 零个 tcgen05 用例
3. **PTX 规范与 IR 枚举不匹配** — NVIDIA PTX ISA 8.6+ 实际有 **12 个指令族**(alloc/dealloc/relinquish_alloc_permit/ld/st/cp/mma/commit/wait/fence 等),现有 5 个 IR 枚举(`S_WMMA` + 4 个 qualifier)严重不足
4. **fragment arithmetic UNVERIFIED** — `wmma.cpp:62-93` 32+ 处 `// UNVERIFIED-AGAINST-HARDWARE` 注释,`TmaDescriptor` 128 字节布局(`tma_descriptor.h:1-29` 详尽说明)字段偏移全部是"暂定值"

**本 change 是 4-change 重构的第 1 步**,只交付**语法+IR 命名空间**,**handler 实现留待 change-3**,确保每步独立可回退(per `ptx-lessons-learned` §3 强制要求)。

## What Changes

### 新增 (5 大类)

#### 1. ANTLR Lexer Tokens(`src/grammar/ptxLexer.g4`)

- **6 个主指令 token**:`TCGEN05: 'tcgen05'` + 5 个子操作 token `MMA_ / LD_ / ST_ / COMMIT_ / WAIT_`(用 `_` 后缀避免与 `mma.sync` 等冲突)
- **~25 个 qualifier token**:`CTA_GROUP / KIND / F16 / BF16 / TF32 / F8 / F4 / MXF4 / MXF8 / I8 / MULTICAST / SEM / PACK / M64N*` 等
- **新 sub-op token**:`RELINQUISH / FENCE / SCALE_VEC_SIZE / BLOCK_SCALE / MBARRIER / ALLOC / DEALLOC / CP / SP / WS / AR / LOAD / STORE`
- **3 个 fence 时间 token**:`BEFORE_THREAD_SYNC / AFTER_THREAD_SYNC`

#### 2. ANTLR Parser Rules(`src/grammar/ptxInstructions.g4`)

- **删除**:`tcgenInst: stBulkInst;`(`ptxInstructions.g4:463-465` 被完全替代)
- **删除**:`matrixInst: wmmaInst;`(`ptxInstructions.g4:426-428` 完全替代)
- **删除**:`wmmaInst` / `wmmaOp` / `wmmaLayout` / `wmmaShape` / `wmmaKind` 5 个旧规则
- **新增**:`tcgen05Inst` 完整语法 + 8 个子规则(`tcgen05Op / tcgen05Kind / tcgen05Layout / tcgen05Shape / tcgen05BlockScale / tcgen05WsMask / tcgen05CpShape / tcgen05Qualifier`)
- **保留**:`stBulkInst`(CP 指令单独处理,不属于 tcgen05)

#### 3. IR StatementType(`include/ptx_ir/ptx_op.def`)

- **新增 12 个**:`S_TCGEN05_ALLOC / S_TCGEN05_DEALLOC / S_TCGEN05_RELINQUISH / S_TCGEN05_LD / S_TCGEN05_ST / S_TCGEN05_CP / S_TCGEN05_MMA / S_TCGEN05_MMA_WS / S_TCGEN05_COMMIT / S_TCGEN05_WAIT / S_TCGEN05_FENCE`(共 11 个独立枚举)
- **新增 struct_kind**:`TCGEN05_INSTR`(独立于 `WMMA_INSTR`)
- **删除**:`S_WMMA` 整行(`ptx_op.def:127`),保留给将来真要支持 wmma.* 的 pre-Blackwell change 重建(本 change 不重建)

#### 4. IR Qualifier(`include/ptx_ir/ptx_qualifier.def`)

- **删除**:`Q_TCGEN05_LD / Q_TCGEN05_ST / Q_TCGEN05_COMMIT / Q_TCGEN05_WAIT` 4 个旧 stub(由独立 IR 枚举代替)
- **新增 ~25 个**:`Q_CTA_GROUP / Q_KIND / Q_MULTICAST / Q_SEM / Q_PACK / Q_F16 / Q_BF16 / Q_TF32 / Q_F8 / Q_F4 / Q_MXF4 / Q_MXF8 / Q_I8 / Q_SP / Q_WS / Q_BLOCK_SCALE / Q_SCALE_VEC_SIZE_2X / Q_SCALE_VEC_SIZE_4X / Q_M64N*` 等

#### 5. IR 新结构体(`include/ptx_ir/statement_context.h`)

- **新增**:`struct Tcgen05Instr`(独立于 `WmmaInstr`),包含 `op_kind / qualifiers / operands / instructionText / cta_group / dtype / num_regs / has_block_scale` 字段
- **新增枚举**:`enum class Tcgen05OpKind { ALLOC, DEALLOC, RELINQUISH, LD, ST, CP, MMA, MMA_WS, COMMIT, WAIT, FENCE }`
- **新增枚举**:`enum class Tcgen05Dtype { F16, BF16, TF32, F8, F4, MXF4, MXF8, I8, MXF4NVF4, INVALID }`

### 修改

- `src/ptx_parser/ptx_parser.cpp:751-784` — `WMMA` 分发改用 `tcgen05*` token,`statementType` 改为 `S_TCGEN05_*`
- `src/ptx_parser/ptx_parser.cpp:771` — `ctx->WMMA()` 改为 `ctx->TCGEN05()`
- `src/ptx_parser/ptx_visitor_wmma.cpp` — 改名为 `ptx_visitor_tcgen05.cpp`,X-Macro 改用 `S_TCGEN05_*`
- `include/ptx_ir/statement_factory.h:274` — 新增 `makeTcgen05Instr(...)` 工厂,删除 `makeWmmaInstr(...)`
- `include/ptx_ir/ptx_ir/CMakeLists.txt`(如有)或 `src/ptxsim/CMakeLists.txt` — `IMPLEMENT_TCGEN_INSTR_HANDLER` 改名 `IMPLEMENT_TCGEN05_INSTR_HANDLER`

### 不修改(范围外,留待后续 change)

- ❌ `src/ptxsim/instructions/wmma.cpp`(handler 实际实现,留待 change-3)
- ❌ `src/ptxsim/memory/tma_descriptor.{h,cpp}`(已存在,基础设施审计留待 change-2)
- ❌ `src/ptxsim/memory/tmem.{h,cpp}`(已存在,留待 change-2)
- ❌ `src/ptxsim/cluster/cluster_context.{h,cpp}`(已存在,留待 change-2)
- ❌ `src/ptxsim/async/tc_queue.{h,cpp}`(已存在,留待 change-2)
- ❌ 删除 `wmma` 路径(留待 change-4)
- ❌ 删除 wmma.cpp(留待 change-4)

### Breaking Changes(影响评估)

- `S_WMMA` IR 枚举被删除 → 影响 `src/ptxsim/instructions/wmma.cpp` 的 `IMPLEMENT_WMMA_INSTR_HANDLER` weak symbol(change-3 处理)
- `Q_TCGEN05_*` qualifier 被删除 → 影响 `is_tcgen05_ld/st/commit/wait()` 等 5 个 helper(change-3 重写)
- `ptx_visitor_wmma.cpp` 文件名变更 → 任何引用此文件的 `CMakeLists.txt` 需要更新(本 change 范围)

## Capabilities

### New Capabilities

- `tcgen05-grammar`:ANTLR grammar 完整支持 Blackwell tcgen05 12 个指令族(alloc/dealloc/relinquish/ld/st/cp/mma/mma.ws/commit/wait/fence)
- `tcgen05-ir-types`:StatementType / Qualifier / Tcgen05Instr 独立于 wmma 命名空间,完全替换 `S_WMMA` / `Q_TCGEN05_*`
- `tcgen05-parse-tests`:PTX 语法测试 + 单元/集成测试覆盖新 grammar,验证 12 个指令族在真实 PTX 文本下解析正确

### Modified Capabilities

- `wmma-tensor-core`:本 change **不修改 spec-level behavior**,仅删除 `wmma.*` 相关 spec 段落(替换为指向 `tcgen05-*` 新 spec)。handler 实现 spec 保持不变(因 handler 留待 change-3)。
- `stub-explicit-failure`:本 change **无 delta**。pre-Blackwell 仍抛 `UnsupportedInstructionException`(`wmma.mma.sync` / `wgmma.async` 等仍走原路径,只是走 tcgen05 handler 的判断逻辑也调整)。

## Impact

### 影响的代码

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `src/grammar/ptxLexer.g4` | 新增 ~30 tokens + 6 sub-op tokens | +50/-5 |
| `src/grammar/ptxInstructions.g4` | 删除 wmma 系列 + 新增 tcgen05 系列 | +120/-15 |
| `src/grammar/ptxOperands.g4` | 新增 tcgen05 qualifier 处理 | +40/-0 |
| `include/ptx_ir/ptx_op.def` | 删除 S_WMMA + 新增 11 个 S_TCGEN05_* | +11/-1 |
| `include/ptx_ir/ptx_qualifier.def` | 删除 4 个 stub + 新增 ~25 个 | +25/-4 |
| `include/ptx_ir/statement_context.h` | 新增 Tcgen05Instr + 2 枚举 | +80/-0 |
| `include/ptx_ir/statement_factory.h` | 新增 makeTcgen05Instr + 删除 makeWmmaInstr | +15/-1 |
| `src/ptx_parser/ptx_parser.cpp` | WMMA 分发改 TCGEN05 | +10/-15 |
| `src/ptx_parser/ptx_visitor_wmma.cpp` → `ptx_visitor_tcgen05.cpp` | 改文件 + X-Macro 重命名 | +20/-10 |
| `src/ptxsim/instruction_handlers.cpp` | `IMPLEMENT_TCGEN05_INSTR_HANDLER` macro | +5/-5 |
| `src/ptxsim/instructions/AGENTS.md` | 更新目录结构说明 | +10/-5 |
| `src/grammar/AGENTS.md` | 更新 lexer/parser 规则说明 | +5/-0 |
| `tests/ptx/tcgen05_*.ptx`(12 个新) | 真实 PTX 端到端解析 | +200/-0 |
| `tests/unit/ptx_ir/test_tcgen05_*.cpp` | 单元测试 | +150/-0 |
| `tests/integration/parser/test_tcgen05_*.cpp` | 集成测试 | +200/-0 |
| `CMakeLists.txt`(各) | 注册新文件 | +10/-5 |
| **总计** | | **+951/-76** |

### 影响的依赖

- ANTLR4 4.11.1 — 已就绪,`cmake --build build --target GenerateParser` 重新生成
- Catch2 测试框架 — 已就绪
- `include/ptx_ir/ptx_qualifier.def` X-Macro 模式 — 已就绪
- `include/ptx_ir/ptx_op.def` X-Macro 模式 — 已就绪

### 不影响的依赖(本 change 范围外)

- `src/ptxsim/instructions/wmma.cpp` — change-3 处理
- `src/ptxsim/memory/*` — change-2 审计
- `src/ptxsim/cluster/*` — change-2 审计
- `src/ptxsim/async/*` — change-2 审计

### 影响的文档

- `src/grammar/AGENTS.md` — 更新 lexer/parser 规则说明
- `src/ptxsim/instructions/AGENTS.md` — 更新目录结构说明(`wmma.cpp` → `tcgen05.cpp`)
- 根 `AGENTS.md` 已知限制表 — 标注 "tcgen05 语法已独立"(后续 change 标注 handler 已实现)
- `docs/dev-process/lessons-learned.md` — 追加 §22 "已实施但未清理模式" 案例(本 change 起源)
- `docs/audits/debt-audit-2026-07-02.md` — P0-C 类债务更新(若有)

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- [x] Baseline 函数清单:`wmma.cpp` 中 5 个 `execute_tcgen05_*` + `is_tcgen05_*` helper(change-3 处理,**本 change 不动**)
- [x] 跨模块状态翻译:`S_WMMA` → `S_TCGEN05_*` 在 `instruction_handlers.cpp` 弱符号分发表(change-3 处理)
- [x] 回退策略:本 change 与 change-2 独立 commit,失败立即 revert

### 多 Phase 推进
- [x] Phase 拆分:6 个 atomic commit(`docs(artifacts)` → `feat(grammar)` → `feat(ir)` → `feat(parser)` → `test` → `archive`)
- [x] 基线 worktree 计划:`.worktrees/baseline-tcgen05-syntax` (任务 0.0.1 包含)
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 commit

### 文档同步
- [x] AGENTS.md 同步项已列出
- [x] ADR 追加段落:本 change 不修改 ADR-0016,后续 change-2/3 必要时新建 ADR-0017/0018
- [x] tasks.md Phase 状态变更已说明

### 实施前必跑
- [ ] **Metis pre-implementation review**(per `ptx-lessons-learned` §7):验证 proposal 关键假设(grammar 行数、IR 枚举数量、文件路径)— 提交前必跑
- [ ] 验证 S_WMMA 删除前所有 weak symbol 引用点已列(`grep -rn "S_WMMA" src/ include/`)
- [ ] 验证 wmma.cpp 中无 `// UNVERIFIED-AGAINST-HARDWARE` 残留影响(change-3 处理)
