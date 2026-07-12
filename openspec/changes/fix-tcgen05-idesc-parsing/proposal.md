# Fix tcgen05.mma Handler — Parse idesc.accumulate Bit for Real Accumulate Routing

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) — Blackwell-only tcgen05
> **Oracle 2026-07-11 审计**: BLOCKER **C1**（handler accumulate routing），session `ses_0aefd09c3ffeSqBIAGdxiRBFWC`
> **Predecessor (Active)**: [`fix-tcgen05-mma-accumulator-and-f32-storage`](../fix-tcgen05-mma-accumulator-and-f32-storage/) — Phase 1+2 实施后 helper 已支持 `accumulate` 参数
> **Predecessor (Active)**: [`fix-tcgen05-commit-wait-group`](../fix-tcgen05-commit-wait-group/) — FU-1 (C3) visitTcgen05Inst IMMEDIATE extraction pattern
> **强制 lessons-learned**: `ptx-lessons-learned` §3(分 Phase commit) + §4(基线 worktree) + §6(artifacts-first) + §7(Pre-impl Review)
> **范围**: 2 atomic commits (Phase 1: handler idesc reading; Phase 2: tests + ADR postmortem)

## Why

Oracle 2026-07-11 审计识别 BLOCKER **C1**: 即使 `fix-tcgen05-mma-accumulator-and-f32-storage`（active）Phase 1+2 已扩展 `tcgen05_fragment_mma_f16(Tmem&, bool accumulate=false)` helper 能力，`processTcgen05Mma` (`tcgen05.cpp:383`) 仍**显式硬编码 `accumulate=false`**。原因：handler 无法读取真实 PTX `mma.accumulate::x` 的 `idesc.accumulate` bit（idesc 是 `RegOperand`，非 qualifier）。

后果：FlashAttention QK^T/PV 矩阵乘 `C += A*B` K-loop 沿 128 次累加的语义**仅可通过对 helper 的直接调用验证**；通过 `processTcgen05Mma` handler 路径执行的真实 PTX 永远走 overwrite 路径。helper 层能力增强无法传递到 handler 路径。

## What Changes

### 修改（handler + tests + ADR）
| 文件 | 范围 | Phase |
|------|------|-------|
| `include/ptx_ir/statement_context.h:180-190` | `Tcgen05Instr` 新增 `bool accumulate = false` 字段 | 1 |
| `src/ptxsim/instructions/tcgen05.cpp:355-393` | `processTcgen05Mma` 从 `instr.operands[3]` (idesc RegOperand) 读寄存器 + 提取 accumulate bit → 动态决定 helper 参数 | 1 |
| `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` | 新增 T4 (idesc.accumulate=1 → 真累加) + T5 (idesc.accumulate=0 → overwrite) + T6 (calibration helper) | 2 |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 "2026-07-12 Postmortem: C1 fix" 段（含 idesc bit 位置实测记录） | 2 |

> **Oracle 2026-07-11 review 校准 (session `ses_0a8af7ff0ffeYHjA65F4uPwcKa`)**: 原 proposal 误判 `helper signature + c_slot` 为待实施工作；实证显示二者已通过 active predecessor (`fix-tcgen05-mma-accumulator-and-f32-storage`) 的累积演进落地。已从 "修改" 表删除 2 行已实施项，避免 ghost work（per lessons-learned §7 "已实施但未清理" 反表象）。

### 不修改（明确 Non-Goals）
- ❌ **不修改 grammar**：per active change `design.md:D1.1` 拒绝方案 (b)（新增 `Q_TCGEN_ACCUMULATE` Qualifier），本 change 沿用方案 (c)（handler 运行时从寄存器读）。不引入 lexer token / parser rule / qualifier enum 改动
- ❌ **不解析完整 idesc**：idesc 是 NVIDIA 内部 64-bit 指令描述符（参考 `bench/cute/include/cute/arch/mma_sm100_desc.hpp:478-647` 编译期构造）。本 change **仅**解析 `accumulate` bit（位 0 placeholder，需测试校准），其他 bits (dtype / scale_format / etc.) 留待后续
- ❌ **不实现多 warp slot partition**：FU-2 阶段加 `warp_id` 参数 + `c_slot` 偏移是为了与 FU-4 API 兼容（per Oracle Q4 Option a），但本 change **不实施多 warp 测试**（属于 FU-4 scope）
- ❌ **不修复 C3 (commit/wait group)**：FU-1 (`fix-tcgen05-commit-wait-group`) 独立 change；本 change 不依赖其完成（C1 与 C3 独立，per Oracle Q2）
- ❌ **不修复 C2 (ld/st slot)**：FU-3 独立 change
- ❌ **不实施 E2E FlashAttention kernel**：留待 FU-5 (`tcgen05-flashattention-coverage`)

## Goals

### Phase 1: Handler idesc Reading (commit 1)
1. 添加 `Tcgen05Instr::accumulate` 字段（`statement_context.h:189` 之后，per LearnD3 field-after-has_block_scale 顺序）
2. `processTcgen05Mma` 改造：从 `instr.operands[3]` (idesc RegOperand) 读 `uint32_t` 值 → 提取 `accumulate` bit（位 0 placeholder，符合 PTX ISA §9.7.16 常见布局）→ 调 `tcgen05_fragment_mma_f16(tmem, warp_id, accumulate)`
3. helper 签名扩展：`tcgen05_fragment_mma_f16(Tmem&, int warp_id, bool accumulate = false)`
4. helper body 改：`c_slot = warp_id * 32 + 64 + lane_id`
5. integration 测试：T4 (通过构造 idesc 寄存器值=1) 验证 2 次 mma 后 C == `2 × GOLDEN`；T5 (idesc=0) 验证 overwrite 保留
6. 跑 `ctest -R "tcgen05" --output-on-failure` 全 PASS
7. **commit**: `fix(tcgen05): read accumulate bit from idesc register in processTcgen05Mma (Oracle C1)`

### Phase 2: Tests + ADR Postmortem (commit 2)
1. 完整 PTX fixture regression：`tests/ptx/tcgen05_mma_with_accumulate.ptx` 含 `.accumulate::x` 语法（仅 IR 解析测试，handler 通过直接构造 `instr.accumulate=true` 验证）
2. ADR-0016 追加 "2026-07-12 Postmortem: C1 fix" 段，含 idesc bit 位置实测记录（如发现 bit 0 错误，记录修正过程）
3. **commit**: `test(tcgen05): add idesc.accumulate integration tests + ADR-0016 postmortem`

### Phase 3: Archive (commit 3, per lessons-learned §6 Checklist G)
1. artifacts-first：commit `docs(openspec): fix-tcgen05-idesc-parsing artifacts`
2. ADR-0016 postmortem commit
3. `openspec archive fix-tcgen05-idesc-parsing --yes`
4. **强制 postmortem prompt**：询问用户是否生成 postmortem

## Capabilities

### New Capabilities
- `tcgen05-idesc-parsing`: 处理 PTX `mma.accumulate::x` 语义，从 idesc 寄存器运行时读取 accumulate bit 决定 helper 行为。这是首个端到端 "PTX qualifier → helper parameter" 数据流验证（与 `.ws` 限定符 9 层数据流平行但语义不同：`.ws` 经 grammar/visitor/qualifier 传递，本能力经 operand/registers/handler 运行时读取）。

### Modified Capabilities
- `tcgen05-handlers-extended`: `processTcgen05Mma` 行为修订 —— 显式传 `accumulate=false` → 运行时从 idesc 决定。Spec-level 行为变化（accumulate 语义从硬编码变为 PTX-driven）需要 delta spec 文件。

> 注：`tcgen05_fragment_mma_f16` 签名扩展（加 `int warp_id` 参数）属于 helper API 演进，归入新 capability `tcgen05-idesc-parsing` 的 spec 文档管理（FU-4 `tcgen05-fragment-mma-helper` 独立 capability 暂不创建，避免 helper API 演进记录碎片化）。

## Impact

### 影响的代码
| 文件 | 变更类型 | LoC 估计 |
|------|----------|---------|
| `include/ptx_ir/statement_context.h` | 修改（+accumulate 字段） | +3 / 0 |
| `src/ptxsim/instructions/tcgen05.cpp:355-393` | 修改（运行时 idesc 读取 + 调用点改动态 accumulate） | +18 / -3 |
| `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` | 新增 T4/T5/T6 TCs | +60 / 0 |
| `tests/ptx/tcgen05_mma_with_accumulate.ptx`（新文件）| 新增语法测试 | +15 / 0 |
| `include/ptxsim/thread_context.h` | **新增** `read_reg_32(const RegOperand&) const` accessor（**Phase 1.0 硬性前置门禁**） | +5 / 0 |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 Postmortem 段 | +40 / 0 |
| **总计** | | **+141 / -3** |

> **注**: helper 签名扩展 (`int warp_id`) 与 `c_slot` 偏移已由 active predecessor 累积实施（实证：`tcgen05_helpers.h:70-71`, `tcgen05_helpers.cpp:29,42-44`, `tcgen05.cpp:383` 已传 `warp->get_warp_id()`），不计入本 change 影响范围。

### 影响的依赖
- 无新外部依赖
- 依赖 `WarpContext::get_warp_id()` API（已存在，per `tcgen05_alloc.cpp:68`）
- 依赖 `ThreadContext::register_bank_` 访问（待验证 API；可能需要新 accessor）

### 不影响的依赖
- 11 个 S_TCGEN05_* handler dispatch（仅 `processTcgen05Mma` 内部修改）
- TmemAllocator / TcQueue / Tcgen05PipelineHandler
- Grammar / parser / visitor（per Non-Goals）

### 影响的文档
- `docs/adr/0016-blackwell-only-tcgen05.md` — 追加 C1 Postmortem 段
- 根 `AGENTS.md` 已知限制表 — **不修改**（本 change 是内部精度提升，未改变对外能力声明）
- `src/ptxsim/instructions/AGENTS.md` — 不修改（handler dispatch 表不变）

## Design-Time Checklist (Lessons-Learned, per `ptx-lessons-learned`)

### Checklist A: 函数迁移完整性
- [x] **Baseline 函数清单**: `processTcgen05Mma` (`tcgen05.cpp:355-393`) + `tcgen05_fragment_mma_f16` (`tcgen05_helpers.cpp:15-58`)，2 个文件中 1 个 production caller
- [x] **逐行 diff 计划**: 见 `design.md` Phase 1 + Phase 2 + `tasks.md`
- [x] **跨模块状态翻译表**:
  - PTX idesc RegOperand → `ThreadContext::register_bank_` → `uint32_t idesc_val` → `accumulate = (idesc_val & 0x1u)` → `helper(accumulate)`
  - 不涉及 ThreadContext state / WarpState / 互斥量变化（仅 helper 内部 c_slot 偏移）
- [x] **回退策略**: 每个 Phase 独立 commit，独立可 revert（per lessons-learned §3）

### Checklist B: 重构前
- [x] **基线 worktree 计划** (per lessons-learned §4):
  ```bash
  git worktree add .worktrees/baseline-c1 HEAD  # 当前 HEAD 含 active H1+H2 已 merge
  cd .worktrees/baseline-c1
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # 必须全量 build
  cd build && ctest -L tcgen05 --output-on-failure
  ```
- [x] **Phase 拆分**:
  - Phase 1 (handler idesc reading + helper signature): 独立 commit，1 次行为扩展（动态 accumulate）
  - Phase 2 (tests + ADR postmortem): 独立 commit，测试完备 + 历史记录
  - Phase 3 (Archive): 3 个 commit 总数
- [x] **失败处理策略**: 任何已有测试回归 → 立即 revert 该 Phase（per lessons-learned §3）

### Checklist C: 写注释
- [x] **关键注释**:
  - `tcgen05.cpp:355-393`: "idesc is a RegOperand (PTX ISA §9.7.16, operand[3]). accumulate bit is bit 0 placeholder — verified via T4/T5 in tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp."
  - `tcgen05_helpers.h`: "warp_id parameter added per FU-4 (Oracle Q4 Option a). Single-warp callers should pass `warp->get_warp_id()`. Multi-warp behavior is FU-4 scope."
  - `tcgen05_helpers.cpp:23`: "c_slot per warp: c_slot = warp_id * 32 + 64 + lane_id. Single-warp idiom (warp_id=0) preserves prior layout."

### Checklist D: Commit 前
- [x] **跑过 baseline worktree 对比** (per Phase)
- [x] **AGENTS.md 同步项**: 不需（handler dispatch 表不变）
- [x] **ADR 追加段落**: Phase 2 追加 "2026-07-12 Postmortem: C1 fix" 到 ADR-0016（含 idesc bit 位置实测记录）
- [x] **OpenSpec tasks.md 状态变更**: Phase 3 archive 后 tasks.md 标记 `[x]` 全完成
- [x] **commit message 列出独立 fix 编号**: Phase 1 = "Oracle C1", Phase 2 = "Oracle C1 follow-up test"

### Checklist E: OpenSpec artifacts 提交顺序（2026-07 新增）
- [x] **artifacts-first**: Phase 3 第一步 commit `docs(openspec): ...` 然后才 archive
- [x] **每个 commit 独立可 revert**: Phase 1 / Phase 2 / Phase 3 各自独立

### Checklist G: OpenSpec lifecycle 约束
- [x] **Ref 链接** 到 active predecessor `fix-tcgen05-mma-accumulator-and-f32-storage` + parallel FU-1 `fix-tcgen05-commit-wait-group`
- [x] **禁止 amend 已归档 change**: 本 change 是新建 `fix-*` change，不是 amend

### Checklist H: Pre-implementation Review 强制项（2026-07 新增）
- [x] **Metis pre-implementation review**: ✅ 2026-07-11 (Oracle Q1-Q6 audit session `ses_0aefd09c3ffeSqBIAGdxiRBFWC`)
- [x] **Oracle 决策建议**: ✅ BLOCKER C1 + 4-way split validated (Oracle Q1 信心 HIGH)
- [x] **Design-Time Lessons Applied**: ✅ 当前提案 8 项 Checklist A/B/C/D/E/G/H 全覆盖
- [x] **idesc bit 位置验证计划**: 见 Phase 2 — 通过 T4/T5 fixture 校准 bit 位置；如发现错误，记录修正过程到 ADR postmortem

### Checklist I: 重大功能交付清单
- [x] **本 change 不算"重大功能"**: handler 内部精度提升，未引入新对外指令能力
- [x] **根 README 不需更新**: behavior 对用户透明

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| `fix-tcgen05-mma-accumulator-and-f32-storage` (active) 提供 helper 的 `accumulate` 参数 | **fix-tcgen05-idesc-parsing** | (未来 `tcgen05-flashattention-coverage` FU-5 可基于本 change 验证真实 PTX 路径) |
| `fix-tcgen05-commit-wait-group` (FU-1 active, independent) | | |

- **上游 Active**: `fix-tcgen05-mma-accumulator-and-f32-storage` 必须先 merge 否则 helper 无 `accumulate` 参数（design.md D1.1 已明示）
- **并行 (FU-1)**: `fix-tcgen05-commit-wait-group` 独立 C3 修复，与本 change 无依赖关系（Oracle Q2 验证）
- **下游**: 本 change 是 FlashAttention-readiness 第二前置（第一前置 = H1+H2 helper fix）

## 本 change 特有设计决策

**决策 D1: idesc 读取路径（per Oracle Q1 + active change design.md D1.1）**
- **采纳**: 运行时从 `instr.operands[3]` (idesc RegOperand) 读 `uint32_t` → 提取 accumulate bit（位 0 placeholder）→ 调 helper
- **拒绝的备选**:
  - (a) Grammar 改动引入 `Q_TCGEN_ACCUMULATE` qualifier：per active change design.md D1.1 line 67 已拒绝（PTX 语法不发射 `.accumulate` qualifier；真实 PTX 用 idesc 寄存器携带此信息）
  - (b) 强制 `accumulate=true` 默认：违反 active change design.md D2.1（默认 `false` 是当前所有测试零修改通过的前提）
- **Tradeoff**: idesc bit 布局未公开（NVIDIA 内部）。通过 T4/T5 fixture 校准位 0；如错误，记录修正过程。

**决策 D2: idesc 解码范围（per Oracle H2 + 本 change Non-Goals）**
- **采纳**: 仅解析 accumulate bit（位 0 placeholder），其他 bits (dtype / scale_format / etc.) 全部硬编码或使用当前 handle 的默认行为
- **拒绝的备选**:
  - (a) 解析完整 idesc 64-bit 描述符：超出 scope；CUTLASS `UMMA::make_instr_desc<>()` 是编译期模板，运行时解析需大量位操作
  - (b) 解析 dtype bit 让 helper 支持 multi-dtype：active change D2 (设计决策 D2) 已锁定 f16×f16→f32 dtype 不变
- **Tradeoff**: 未来扩展需后续独立 change（如 `fix-tcgen05-idesc-full-parsing`）

**决策 D3: helper warp_id 参数已存在 — 本 change 不再扩展（per Oracle 2026-07-11 review session `ses_0a8af7ff0ffeYHjA65F4uPwcKa`）**
- **采纳**: 沿用 active predecessor 已实施的 `int warp_id` 参数（实证：`tcgen05_helpers.h:70-71` 三参数签名 + `tcgen05_helpers.cpp:42-44` c_slot `warp_id * 32 + 64 + lane_id` 公式 + `tcgen05.cpp:383` 已传 `warp->get_warp_id()`）
- **拒绝的备选**:
  - (a) 在本 change 再次扩展签名：产生 no-op diff + 风险引入错误（如重复 warp_id 参数）
  - (b) 推迟到 FU-4：FU-4 不再需要改 helper 签名，仅需补多 warp 测试
- **Tradeoff**: 沿用已实施 API（避免 churn + 符合 lessons-learned §7 "不要重做已实施工作"）vs 重新声明已落地工作

**决策 D4: 回退策略（per lessons-learned §3）**
- **采纳**: 2 atomic commits (Phase 1 + Phase 2) + 1 archive commit (Phase 3) = 3 commits
- **拒绝的备选**:
  - (a) 1 combined commit：违反 lessons-learned §3 + 双倍 breakage 调试噩梦
  - (b) 3+ commits (handler | helper signature | tests)：过度拆分，每个 commit diff 过小
- **Tradeoff**: 当前粒度平衡"独立可回退"与"commit 不碎片化"
