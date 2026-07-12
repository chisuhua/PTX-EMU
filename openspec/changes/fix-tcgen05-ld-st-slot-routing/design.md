## Context

Oracle 2026-07-11 审计 (`ses_0b3791d78ffewb52428kJJ2Irz`) 发现 `tcgen05.ld`/`tcgen05.st`/`tcgen05.cp` 三个 handler 均硬编码 TMEM slot `0`，与 mma 写 C 到 `slot[64..95]` (`tcgen05_helpers.cpp:23`) 矛盾。FlashAttention 的 QK^T→softmax→PV 数据流依赖 ld/st 在 mma slot 范围内移动数据 — 当前架构不可能。

`Tmem::read/write` (`src/ptxsim/memory/tmem.h:35-36`) 本身已接受任意 `slot_id`，瓶颈是 handler 内部硬编码常量。`Tcgen05Instr` (`include/ptx_ir/statement_context.h:180-190`) 当前无 `tmem_slot` 字段；`visitTcgen05Inst` (`src/ptx_parser/ptx_visitor.cpp:841-885`) 不提取 slot 操作数。

**Open question**: 真实 PTX `tcgen05.ld` 语法是否包含 TMEM slot 操作数？待 Phase 1 step 0 用 `cuobjdump -xptx` 实证。若 PTX 无 slot 操作数，本 change 必须采用 `.tmem_slot::N` qualifier 路径（per Oracle Q5 Option b pattern）。

## Goals / Non-Goals

**Goals:**
1. `tcgen05.ld` / `.st` / `.cp` handler 接收 `tmem_slot` 参数而非硬编码 `0`
2. `Tcgen05Instr` IR 携带 `tmem_slot` 字段（默认 `0` 保持向后兼容）
3. Grammar + visitor + factory 完整数据通路 — slot 从 PTX 操作数传到 handler
4. Anti-grammar-pattern discipline — 不引入 bare string lexer tokens (per ptx-lessons-learned §9)
5. 行为测试覆盖：ld 写 slot N 后 mma 读 slot N 等于 ld 输入（验证数据流连通）
6. 零回归 — 所有现有测试（包括 FU-1 已修复的 commit/wait 路径）继续通过

**Non-Goals:**
- ❌ 不修改 mma handler（FU-2 `fix-tcgen05-idesc-parsing` 独立 change）
- ❌ 不修复 commit/wait group_id（FU-1 `fix-tcgen05-commit-wait-group` 是**前置**）
- ❌ 不修改 `Tcgen05Instr.cta_group` / `dtype` / `num_regs` / `has_block_scale`（FU-1/C3 处理）
- ❌ 不实现 multi-warp slot 偏移（FU-4 独立 change）
- ❌ 不引入 TmemAllocator API 扩展
- ❌ 不实现 cta_group::2（per ADR-0018）

## Decisions

### D1: tmem_slot 表达路径 — Operand（首选） vs Qualifier（fallback）

**采纳**: PTX 操作数路径（首选），即 `tcgen05Operands` 规则加 slot 操作数。

**理由**:
1. PTX `tcgen05.ld` 当前 2 操作数（smem dst + global src），增加第 3 操作数（slot）是自然扩展（per `ptx_op.def:130` `op_count=2`）
2. 数据流更直观：slot 是"目标地址"，与 smem/global 同级
3. 现有 parse 测试 (`test_tcgen05_ld_parse.cpp`) 验证 `instr.operands.size() == 2`，增加 slot 后 `== 3`，机械化迁移明确

**Fallback**: `.tmem_slot::N` qualifier
- 若 Phase 1 step 0 实证发现真实 PTX 不含 slot 操作数（per `cuobjdump -xptx`）
- 复用 FU-1/C3 建立的 IMMEDIATE walk pattern（`visitTcgen05Inst` 单独 walk `.tmem_slot` qualifier children）
- 决策点：Phase 1 step 0.5 — `cuobjdump` 验证后固化方案

**拒绝**:
- ❌ 把 `tmem_slot` 硬编码为 `0` 的别名（如 `.slot::0`） — 无语义
- ❌ 复用现有 `cta_group::N` qualifier 携带 slot — 语义混淆
- ❌ 新建 `S_TCGEN05_LD_SLOT` 等独立 StatementType — 违反"不引入新 enum" Non-Goal

### D2: ANTLR Grammar 修改 — Anti-bare-token 纪律

**采纳**: 不引入 bare string lexer token；用数字立即数操作数（首选）或 `.tmem_slot::N` qualifier（fallback）

**理由** (per ptx-lessons-learned §9 + §L):
- ❌ `TMEM_SLOT : 'tmem_slot'` bare token 会与 ID 规则 `[a-zA-Z_$][a-zA-Z_0-9$]*` 冲突（commit `ad808e3` 实际案例教训：5 ctest 失败）
- ✅ PTX 数字立即数（`[0-9]+`）作为操作数是安全选择 — ID 规则不匹配纯数字
- ✅ `.tmem_slot::N` qualifier 用 `.` 前缀避开 ID 冲突（参考 `.cta_group::1` 现有 pattern）

**Grammar 修改 (operand 路径)**:
```
// ptxInstructions.g4:488-492 BEFORE
tcgen05Operand : vectorRegister | address | operand;

// AFTER  
tcgen05Operand : tcgen05Slot | vectorRegister | address | operand;

// 新规则
tcgen05Slot : UNSIGNED_INT;

// 或更安全: 限定范围 0..255 (kSlotCount-1)
// BUT — ANTLR 解析时不强制运行时范围，由 handler assert
```

**Grammar 修改 (qualifier 路径, fallback)**:
```
// ptxInstructions.g4:450-468 — tcgen05Qual 规则加 TCGEN_TMEM_SLOT 备选项
| TCGEN_TMEM_SLOT COLONCOLON IMMEDIATE  // .tmem_slot::N

// ptxLexer.g4: 在 .ws 之后 (约 426 行 之后) 加：
// 注意：避免 bare string。改用 . 前缀或 ID 形式
// 实际：lexer 用 ID 形式（不需要新增 bare token），grammar 接受 '.tmem_slot' 作为 ID + IMMEDIATE
```

**最终方案决策点**: Phase 1 step 0 `cuobjdump -xptx` 实证后决定 — operand 路径 OR qualifier 路径。**禁止混合**。

### D3: Tcgen05Instr 字段设计 — 单字段 vs 多字段

**采纳**: 单 `uint32_t tmem_slot = 0` 字段

**理由**:
1. 与现有 `cta_group` / `num_regs` 模式一致（都是 `uint32_t` 字段，默认值保留向后兼容）
2. 单字段最小化 IR 内存开销（per `Tcgen05Instr` 是 instruction-time 创建，频繁构造/析构）
3. 多 warp slot 偏移（FU-4）单独由 `warp_id` 计算，不与 `tmem_slot` 耦合

**拒绝**:
- ❌ `std::optional<uint32_t>` — 增加内存 + 增加 optional 检查路径
- ❌ 拆 `tmem_slot_base` + `tmem_slot_offset` — 过度工程（FU-4 的 warp offset 单独处理）

### D4: 默认值 `tmem_slot = 0` — 向后兼容

**采纳**: 默认 `0`

**理由**:
1. 与当前 handler 硬编码 `0` 一致 → 零静默行为变化
2. `processTcgen05Ld/St` 的现有调用方（包括 FU-2 mma 路径）不感知新字段
3. 只有显式提供非零 slot 的 PTX 才有新行为 — explicit opt-in

**Tradeoff**: 若 PTX 操作数解析失败，默认 `0` 可能掩盖错误 — 通过 `test_all_ptx.sh` 严格验证缓解

### D5: Handler 修改粒度 — 3 文件独立

**采纳**: `tcgen05.cpp:434, 476` + `tcgen05_cp.cpp:138` 三处独立 commit

**理由**:
1. per lessons-learned §3 — "复杂迁移必须分 Phase commit"
2. 单 Phase 1 (grammar + IR + visitor + factory) + 单 Phase 2 (handlers + tests) — 2 commit 粒度清晰
3. 失败处理：handler 改错立刻 revert Phase 2，不污染 Phase 1

**Tradeoff**: 略增 commit 数（2 vs 1）— 严格遵循 lessons-learned §3

### D6: OpenSpec artifacts 结构 — 3 delta specs

**采纳**: 在 `tcgen05-handlers-core` / `tcgen05-handlers-extended` / `tcgen05-ir-types` 三个 capability 下各加 delta spec

**理由**:
1. OpenSpec spec-driven schema 要求 modified capability 必有 delta spec
2. delta spec 文件路径：`openspec/changes/fix-tcgen05-ld-st-slot-routing/specs/<capability>/spec.md`
3. 现有 capability 边界清晰（见 `openspec/specs/` 目录）

**Specs 清单**:
- `specs/tcgen05-handlers-core/spec.md` — delta for LD/ST semantics
- `specs/tcgen05-handlers-extended/spec.md` — delta for CP semantics
- `specs/tcgen05-ir-types/spec.md` — delta for `Tcgen05Instr.tmem_slot` 字段

## Risks / Trade-offs

| ID | 风险 | Severity | Mitigation |
|----|------|----------|------------|
| R1 | Phase 1 grammar 修改引发 ANTLR 预测冲突 / ID 抢占 | HIGH | 严格遵循 ptx-lessons-learned §9/§L：不引入 bare token；改完跑 `./tests/ptx/test_all_ptx.sh` 47/47 |
| R2 | op_count 增加破坏现有 handler 测试（fixtures hardcode `operands.size()==2`） | MEDIUM | Phase 1 跑全量 ctest 验证；若回归立即 revert Phase 1 |
| R3 | FU-1 (C3) 未完成导致 Phase 1 visitor 实现受阻 | HIGH | **前置条件**：本 change apply 前 FU-1 必须已 merge 并 archive |
| R4 | 真实 PTX 不含 slot 操作数 — operand 路径失败 | MEDIUM | Phase 1 step 0 `cuobjdump -xptx` 验证 → fallback 到 `.tmem_slot::N` qualifier 路径 |
| R5 | 默认 `tmem_slot=0` 引发调用方静默行为变化 | LOW | 默认与现有硬编码一致（都是 0），零变化 |
| R6 | Handler `instr.tmem_slot` 引用但 IR 未填充（op_count 改动同步失败） | MEDIUM | `static_assert` 确保 `instr.tmem_slot` 字段存在 + 单元测试覆盖 `makeTcgen05Instr(...).tmem_slot == 0` |
| R7 | mma helper `c_slot = 64 + lane_id` (`tcgen05_helpers.cpp:23`) 与 ld/st slot 不一致 | LOW | 本 change 不动 helper；multi-warp 时由 FU-4 协调；FlashAttention 通常约定 ld 写 slot N, mma 写 slot N+32 等用户控制 |
| R8 | Phase 2 行为测试需要构造多 slot fixture，可能需扩展 `TestRig` | MEDIUM | 测试 helper 复用现有 `TestRig` + 加 `tmem_slot` 字段（per `test_tcgen05_mma_persistence.cpp` 已用模式）|

## Migration Plan

### Phase 1: Grammar + IR + Visitor + Factory（commit 1）

#### Baseline 函数清单

| 函数 | 文件:行 | 调用者 | 当前 slot 来源 |
|------|---------|--------|--------------|
| `processTcgen05Ld` | `src/ptxsim/instructions/tcgen05.cpp:402-439` | `Tcgen05Handler::processTcgen05Operation` (case LD) | 硬编码 `tmem.write(0, ...)` (line 434) |
| `processTcgen05St` | `src/ptxsim/instructions/tcgen05.cpp:448-484` | 同上 (case ST) | 硬编码 `tmem.read(0, ...)` (line 476) |
| `processTcgen05Cp` | `src/ptxsim/instructions/tcgen05_cp.cpp:127-156` | 同上 (case CP) | 硬编码 `kDestSlot = 0` (line 138) |

#### Step 0: 真实 PTX 语法验证（关键决策点）

```bash
# 用 nvcc 编译一个含 tcgen05.ld 的 kernel 并提取 PTX
nvcc -ptx -arch=sm_100 -keep --no-compress tests/ptx/regression_tcgen05_ld_slot.cu -o /tmp/test.ptx
cuobjdump -xptx /tmp/test.ptx | grep -A2 "tcgen05.ld"
# 检查 PTX 是否含 slot 操作数 → 决定 operand 路径 or qualifier 路径
```

#### Step 1.1: Grammar 修改 (operand 路径)

```g4
// src/grammar/ptxInstructions.g4:488-492
// BEFORE: tcgen05Operand : vectorRegister | address | operand;
// AFTER:
tcgen05Operand : tcgen05Slot | vectorRegister | address | operand;
tcgen05Slot : UNSIGNED_INT;  // 0..kSlotCount-1, handler assert 范围
```

```cpp
// include/ptx_ir/ptx_op.def:130-132
// BEFORE: X(S_TCGEN05_LD, ..., 2, ...)
// AFTER:  X(S_TCGEN05_LD, ..., 3, ...)  // +1 slot
X(S_TCGEN05_LD,         tcgen05.ld,         Tcgen05,    3, TCGEN05_INSTR, tensor)
X(S_TCGEN05_ST,         tcgen05.st,         Tcgen05,    3, TCGEN05_INSTR, tensor)
X(S_TCGEN05_CP,         tcgen05.cp,         Tcgen05,    4, TCGEN05_INSTR, tensor)  // +1 slot
```

#### Step 1.2: IR 字段 + factory 改造

```cpp
// include/ptx_ir/statement_context.h:180-190
struct Tcgen05Instr {
    Tcgen05OpKind op_kind = Tcgen05OpKind::MMA;
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
    std::string instructionText;
    uint32_t cta_group = 1;
    Tcgen05Dtype dtype = Tcgen05Dtype::F16;
    uint32_t num_regs = 0;
    bool has_block_scale = false;
    uint32_t tmem_slot = 0;  // NEW: per Oracle C2 fix (PTX ISA §9.7.16)
                              // Default 0 = backward compatible with hardcoded handlers.
};
```

```cpp
// include/ptx_ir/statement_factory.h:265-292 — makeTcgen05Instr 加可选参数
inline StatementContext makeTcgen05Instr(
    Tcgen05OpKind op_kind,
    const std::vector<Qualifier>& qualifiers,
    const std::vector<OperandContext>& operands,
    const std::string& text = "",
    uint32_t tmem_slot = 0);  // NEW
```

#### Step 1.3: visitor 提取（per FU-1 pattern）

```cpp
// src/ptx_parser/ptx_visitor.cpp:841-885 — visitTcgen05Inst
// 现有代码段（已由 FU-1 加上 IMMEDIATE walk for cta_group）：
//   qualifiers = extractQualifiersFromContext(ctx);
//   uint32_t cta_group = 1;
//   if (ctx->tcgen05QualList()) { ... }

// NEW: C2 — extract tmem_slot from operands (position 0 if 4-operand, else fallback to 0)
// 假设 operand 路径：slot 在 operands 最前面
// 假设 qualifier 路径：与 cta_group 同样 walk（但用 .tmem_slot 而不是 .cta_group）
uint32_t tmem_slot = 0;  // default
if (op_kind == Tcgen05OpKind::LD || op_kind == Tcgen05OpKind::ST ||
    op_kind == Tcgen05OpKind::CP) {
    // Operand 路径
    if (!instr.operands.empty() && instr.operands[0].is_imm()) {
        tmem_slot = static_cast<uint32_t>(std::stoul(instr.operands[0].imm.value));
    }
    // Qualifier 路径 (fallback, 若 step 0 实证需要):
    // for (auto* qualCtx : ctx->tcgen05QualList()->tcgen05Qual()) {
    //     if (qualCtx->tmem_slot_qual()) {
    //         tmem_slot = std::stoul(qualCtx->IMMEDIATE()->getText());
    //     }
    // }
}
return makeTcgen05Instr(op_kind, qualifiers, operands, text, tmem_slot);
```

#### Step 1.4: Parser 测试更新

```cpp
// tests/integration/ptx/test_tcgen05_ld_parse.cpp — 新增 TC
TEST_CASE("Tcgen05Instr with tmem_slot operand parses correctly") {
    auto instr = make_tcgen05_ld_with_slot(32);
    REQUIRE(instr.operands.size() == 3);
    REQUIRE(instr.tmem_slot == 32);
    // 验证默认 0 保留向后兼容
    auto instr_default = make_tcgen05_ld_default();
    REQUIRE(instr_default.tmem_slot == 0);
}
```

#### Step 1.5: 验证 + Commit

```bash
cmake --build build --target GenerateParser  # 必须！
./tests/ptx/test_all_ptx.sh  # 47/47 必须
ctest -R "tcgen05" --output-on-failure  # 全 PASS
git add include/ptx_ir/ptx_op.def include/ptx_ir/statement_context.h \
        include/ptx_ir/statement_factory.h src/ptx_parser/ptx_visitor.cpp \
        src/grammar/ptxInstructions.g4 tests/integration/ptx/test_tcgen05_ld_parse.cpp \
        tests/integration/ptx/test_tcgen05_st_parse.cpp tests/integration/ptx/test_tcgen05_cp_parse.cpp
git commit -m "fix(tcgen05): parse tmem_slot operand for ld/st/cp (Oracle C2 Phase 1)"
```

### Phase 2: Handler 路由 + 行为测试（commit 2）

#### Step 2.1: Handler 修改（3 行）

```cpp
// src/ptxsim/instructions/tcgen05.cpp:434
// BEFORE: tmem.write(0, tmp, Tmem::kSlotSize);
// AFTER:
tmem.write(instr.tmem_slot, tmp, Tmem::kSlotSize);
PTX_DEBUG_EMU("tcgen05.ld: ... → TMEM slot %u (%zu bytes)",
              instr.tmem_slot, Tmem::kSlotSize);

// src/ptxsim/instructions/tcgen05.cpp:476
// BEFORE: tmem.read(0, tmp, Tmem::kSlotSize);
// AFTER:
tmem.read(instr.tmem_slot, tmp, Tmem::kSlotSize);

// src/ptxsim/instructions/tcgen05_cp.cpp:130,138
// BEFORE: constexpr size_t kDestSlot = 0; ... tmem.write(kDestSlot, tmp, Tmem::kSlotSize);
// AFTER: 删除 kDestSlot 常量，改用 instr.tmem_slot
tmem.write(instr.tmem_slot, tmp, Tmem::kSlotSize);
```

#### Step 2.2: 行为测试新增

```cpp
// tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp（新文件）
#include "ptxsim/sm_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/testing/scheduler_utils.h"

TEST_CASE("tcgen05.ld to non-zero slot + cp reads same slot: data flow intact") {
    // Setup: 128B golden pattern
    // Step 1: ld → slot 32
    // Step 2: st → slot 32 (= ld destination)
    // Step 3: verify st dst memory == golden
}

TEST_CASE("tcgen05.cp to non-zero slot: round-trip preserves data") {
    // Setup: known pattern in slot 96
    // cp slot 96 → smem dst
    // verify smem dst == pattern
}

TEST_CASE("ld to slot 0 still works (backward compat)") {
    // Replay default scenario — slot=0 行为不变
}
```

#### Step 2.3: 验证 + Commit

```bash
ctest -R "tcgen05" --output-on-failure  # 全 PASS
# 对比 baseline (per ptx-lessons-learned §4):
cd .worktrees/baseline-c2/build && ctest -L tcgen05 --output-on-failure  # baseline 全 PASS
git add src/ptxsim/instructions/tcgen05.cpp src/ptxsim/instructions/tcgen05_cp.cpp \
        tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp
git commit -m "fix(tcgen05): route ld/st/cp to instruction-specified tmem_slot (Oracle C2 Phase 2)"
```

### Phase 3: Archive + ADR Postmortem（commit 3）

per ptx-lessons-learned §6/§G：
1. `git add openspec/changes/fix-tcgen05-ld-st-slot-routing/`
2. commit `docs(openspec): fix-tcgen05-ld-st-slot-routing artifacts`
3. `docs/adr/0016-blackwell-only-tcgen05.md` 追加 "2026-07-12 Postmortem: C2 fix" 段
4. commit ADR
5. `openspec archive fix-tcgen05-ld-st-slot-routing --yes`
6. 强制 postmortem prompt（per openspec-archive-change skill）

#### 回退策略

每个 Phase 独立 commit，独立 revert：
- `git revert <phase-1-commit>` → grammar/IR/visitor 全部回退，handler 重新硬编码 0（**前提：handler 还未改 → 实际 Phase 1 + Phase 2 必须串行，但回退时只 revert Phase 2 就够了**）
- `git revert <phase-2-commit>` → handler 重新硬编码 0，测试 fixture 删除

**警告**：Phase 1 + Phase 2 必须 **串行 commit**（Phase 2 依赖 Phase 1）。若 Phase 1 commit 后立即发现 grammar 问题 → 直接 `git revert <phase-1-commit>`，无需 Phase 2。

## Open Questions

### Q1: 真实 PTX `tcgen05.ld` 语法是否含 slot 操作数？

- **必答来源**：`cuobjdump -xptx` 输出
- **决策点**：Phase 1 step 0
- **若未含**：fallback 到 `.tmem_slot::N` qualifier 路径（per Oracle Q5 Option b pattern）
- **若含**：用 operand 路径（首选）

### Q2: `tcgen05.ld` slot 操作数位置？

- **可能位置**：[tmem_slot, smem_dst, global_src] OR [smem_dst, global_src, tmem_slot] OR 其他
- **待验证**：Phase 1 step 0 实证
- **若模糊**：优先放最前面（slot 是"目标地址"，与 smem/global 同级）

### Q3: FU-1 (C3) merge 时机？

- **本 change apply 的强前置**：FU-1 必须已 archive（其 visitor pattern 是本 change 借鉴基础）
- **若 FU-1 进行中**：本 change 等其 archive 后再 apply
- **若 FU-1 失败被 cancel**：本 change 需在 proposal 阶段重新审视 IMMEDIATE walk pattern，可改用纯 visitor 内联实现

### Q4: `Tcgen05Instr.tmem_slot` 字段是否影响 FU-5 (FlashAttention coverage)？

- **初步答案**：是 — FU-5 的 e2e 测试需要为 ld/st 提供非零 slot 操作数
- **FU-5 跟进**：在 follow-up `tcgen05-flashattention-coverage` proposal 中明确引用本 change 的 `tmem_slot` 字段

## Acceptance Criteria

### Phase 1 (Grammar + IR) Acceptance

1. `tcgen05_fragment_mma_*` / `makeTcgen05Instr` 编译通过
2. `Tcgen05Instr.tmem_slot` 字段存在且默认 `0`
3. parser 测试验证 tmem_slot 操作数解析
4. `./tests/ptx/test_all_ptx.sh` 47/47 PASS
5. `ctest -R "tcgen05"` 全 PASS（除预期 op_count 变化导致的现有 fixture 失败 — 必须修这些 fixture）

### Phase 2 (Handler) Acceptance

1. `tcgen05.ld` / `st` / `cp` handler 引用 `instr.tmem_slot`
2. 行为测试 `test_tcgen05_ld_st_slot_routing.cpp` 全 PASS
3. `ctest -R "tcgen05"` 全 PASS
4. baseline 对比：除 `processTcgen05Ld/St/Cp` 行为变化（slot 来源），其他测试不变

### Phase 3 (Archive) Acceptance

1. 4 artifacts git-tracked
2. ADR-0016 Postmortem 段已追加
3. `ctest --output-on-failure` 全量 PASS
4. `./tests/ptx/test_all_ptx.sh` 47/47 PASS
5. Archive commit 含 Postmortem 引用

## References

- Oracle 2026-07-11 BLOCKER 审计: session `ses_0b3791d78ffewb52428kJJ2Irz`（C2 BLOCKER, HIGH confidence）
- Oracle 2026-07-11 split 验证: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` Q1 + Q5（推荐 Option b IMMEDIATE walk 模式）
- 前置 change: `openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/`
- 关联 change (前置): `openspec/changes/fix-tcgen05-commit-wait-group/` (FU-1, C3)
- 关联 change (后续): `openspec/changes/fix-tcgen05-multi-warp-fragment/` (FU-4, C4) 与 `openspec/changes/tcgen05-flashattention-coverage/` (FU-5)
- Ref (archived): `openspec/changes/archive/2026-07-10-implement-tcgen05-handlers-extended/`
- ADR-0016: `docs/adr/0016-blackwell-only-tcgen05.md`
- ptx-lessons-learned: `.opencode/skills/ptx-lessons-learned/SKILL.md` §3, §4, §6, §7, §9, §L
- ptx-grammar-modification skill: `.opencode/skills/ptx-grammar-modification/SKILL.md`
- ptx-debug skill: `.opencode/skills/ptx-debug/SKILL.md` (Phase 1 step 0 `cuobjdump` 验证)
- PTX ISA §9.7.16 (tcgen05.ld/.st/.cp semantics)
