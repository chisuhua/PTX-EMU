## Context

Oracle 2026-07-11 审计识别 BLOCKER **C1**: `processTcgen05Mma` (`src/ptxsim/instructions/tcgen05.cpp:355-393`) 显式硬编码 `accumulate=false` 调用 helper，无法从真实 PTX `mma.accumulate::x` 语法提取语义。

**根因（per Oracle Q1）**：
- `tcgen05.mma` PTX 语法（PTX ISA §9.7.16）含 `idesc` 寄存器操作数（operand[3]，`RegOperand`）
- idesc 是 64-bit NVIDIA 内部指令描述符，accumulate bit 是其中 1 位（位置未公开）
- 当前 handler (`tcgen05.cpp:383`) 完全忽略 idesc，固定传 `accumulate=false`
- 后果：helper 层面具备累加能力（per active `fix-tcgen05-mma-accumulator-and-f32-storage`），但真实 PTX 路径永不累加

**当前状态**：
- helper `tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false)` 已扩展（active change Phase 1）
- `processTcgen05Mma` 仍传 `accumulate=false` 显式覆盖
- `Tcgen05Instr` 结构体 (`include/ptx_ir/statement_context.h:180-190`) 无 `accumulate` 字段
- `ThreadContext::register_bank_` 访问能力（待验证 API）

**目标状态**：
- `Tcgen05Instr::accumulate` 字段由 handler 在运行时从 idesc 寄存器填充
- `processTcgen05Mma` 调用 `tcgen05_fragment_mma_f16(tmem, warp_id, accumulate)`
- helper 签名扩展为 `tcgen05_fragment_mma_f16(Tmem&, int warp_id, bool accumulate = false)`（同步为 FU-4 铺路）
- `c_slot = warp_id * 32 + 64 + lane_id`（单 warp 调用传 `warp_id=0` 等价当前 layout）

**约束**：
- 不修改 grammar（per active change `design.md:D1.1` 已拒绝方案 b）
- 不解析完整 idesc（per本 change Non-Goals，仅 accumulate bit）
- 不实施多 warp 测试（per FU-4 scope）

## Goals / Non-Goals

**Goals:**
1. Handler 层动态从 idesc 寄存器读 accumulate bit
2. helper 签名扩展 `+int warp_id` 参数（与 FU-4 API 对齐）
3. c_slot 加 warp_id 偏移（单 warp 下等价当前行为）
4. 集成测试 T4/T5 验证 idesc-driven accumulate 语义
5. ADR-0016 追加 C1 Postmortem（含 idesc bit 位置实测记录）
6. 遵循 1 次签名扩展原则（避免 2 次 churn：当前 + FU-4 实施时）

**Non-Goals:**
1. 不修改 grammar（idesc 是 operand，非 qualifier；per active change D1.1）
2. 不解析完整 idesc 64-bit 描述符
3. 不实施多 warp 测试（FU-4 scope）
4. 不修复 C3 (commit/wait group) — 独立 change `fix-tcgen05-commit-wait-group` (FU-1, active)
5. 不修复 C2 (ld/st slot) — 独立 change (FU-3, 未 propose)
6. 不实施 E2E FlashAttention kernel — FU-5 (`tcgen05-flashattention-coverage`, 未 propose)

## Decisions

### D1: idesc 读取路径（per Oracle Q1）

**采纳**: handler 运行时从 `instr.operands[3]` (idesc RegOperand) 读 `uint32_t` 值 → 提取 accumulate bit（位 0 placeholder）→ 调 helper

```cpp
// src/ptxsim/instructions/tcgen05.cpp:355-393
void processTcgen05Mma(ThreadContext& thread, const Tcgen05Instr& instr) {
    // ... 现有 dispatch ...

    // NEW: C1 fix — 运行时从 idesc 读 accumulate bit
    bool accumulate = false;
    if (instr.operands.size() >= 4 &&
        instr.operands[3].type == OperandContext::Type::Reg) {
        const auto& idesc_reg = instr.operands[3].reg;
        uint32_t idesc_val = thread.read_reg_32(idesc_reg);  // 新 accessor
        accumulate = (idesc_val & 0x1u) != 0;  // bit 0 placeholder
    }

    Tmem& tmem = cta->tmem();
    tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), accumulate);
}
```

**拒绝的备选**:
- (a) Grammar 改动引入 `Q_TCGEN_ACCUMULATE` qualifier：per active change `design.md:D1.1` line 67 已拒绝；PTX 语法不发射 `.accumulate` qualifier
- (b) 强制 `accumulate=true` 默认：违反 active change D2.1 默认 false 决策
- (c) 完全不读 idesc，让 helper 默认 `accumulate=false`：当前状态，但 active change D1 标记为 debt

**Tradeoff**: idesc bit 位置未公开 → placeholder bit 0 通过 T4/T5 fixture 验证；如错误，ADR postmortem 记录修正过程

### D2: idesc 解码范围（per本 change Non-Goals）

**采纳**: 仅解析 accumulate bit（位 0），其他 bits (dtype / scale_format / etc.) 使用 helper 现有默认行为

**拒绝的备选**:
- (a) 解析完整 idesc 64-bit 描述符：超出 scope；CUTLASS `UMMA::make_instr_desc<>()` 是编译期模板
- (b) 解析 dtype bit 让 helper 支持 multi-dtype：active change D2 锁定 f16×f16→f32 dtype

**Tradeoff**: 未来扩展需后续独立 change（如 `fix-tcgen05-idesc-full-parsing`）

### D3: helper warp_id 参数同步加入（per Oracle Q4 Option a）

**采纳**: helper signature 同步加 `int warp_id` 参数（即使单 warp 透传 `warp_id=0`），c_slot `= warp_id * 32 + 64 + lane_id`

**拒绝的备选**:
- (a) 仅加 `accumulate` 不加 `warp_id`：未来 FU-4 实施时需再次改 helper signature（2 次 churn）
- (b) 推迟 warp_id 到 FU-4：增加 helper signature 变更次数

**Tradeoff**: 1 次签名扩展同时承载 C1 + 为 C4 铺路 vs 2 次小扩展

### D4: 回退策略（per lessons-learned §3）

**采纳**: 2 atomic commits (Phase 1 + Phase 2) + 1 archive commit (Phase 3) = 3 commits

| Commit | Message | 范围 |
|--------|---------|------|
| Phase 1 | `fix(tcgen05): read accumulate bit from idesc register (Oracle C1)` | handler + helper signature + T4/T5 |
| Phase 2 | `docs(tcgen05): ADR-0016 postmortem C1 fix + idesc bit position record` | ADR 追加段 + 测试 fixture |
| Phase 3 | `chore(openspec): archive fix-tcgen05-idesc-parsing` | 4 个 artifacts git-track + archive |

**拒绝的备选**:
- (a) 1 combined commit：违反 lessons-learned §3
- (b) 3+ commits (handler | helper sig | tests | ADR)：过度拆分

**Tradeoff**: 当前粒度平衡"独立可回退"与"commit 不碎片化"

### D5: idesc bit 位置 placeholder 与校准流程

**采纳**: T4 (idesc 寄存器值=1 触发 accumulate) 与 T5 (idesc=0 触发 overwrite) 编写时**假设 bit 0**；运行后验证 PTX ISA §9.7.16 常见布局（accumulate 通常在低位）

**校准流程**：
1. T4/T5 首次运行：若 FAIL，分析哪个 bit 实际触发 accumulate 语义
2. 调整 `accumulate = (idesc_val & 0x1u)` 中的位掩码（如 `0x2u`、`0x4u`）
3. 重新运行 T4/T5，确认 PASS
4. ADR-0016 Postmortem 段记录最终位掩码 + 校准过程

**拒绝的备选**:
- (a) 跳过 fixture 校准，硬编码位掩码 + 等 real-world 失败时再修：违反 lessons-learned §7（Pre-impl Review 要求实证）
- (b) 直接读 CUTLASS 源码反推 bit 位置：依赖外部参考，本项目无 vendored CUTLASS（仅有 `bench/cute/include/cute/arch/mma_sm100_desc.hpp` 编译期模板，不暴露运行时 bit 位置）

**Tradeoff**: 临时 placeholder + 校准 vs 阻塞实施等外部参考

## Risks / Trade-offs

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R1 | idesc bit 位置 placeholder 错误 → T4/T5 FAIL | HIGH | D5 校准流程 + ADR postmortem 记录 |
| R2 | `ThreadContext::read_reg_32` accessor 不存在 → 编译失败 | HIGH | Phase 1.1 实施前先 grep 验证 API；如不存在，加最小 accessor（与 lessons-learned §1 跨模块状态翻译原则一致） |
| R3 | helper signature 扩展破坏 active change 已 merge 的 accumulate=false 调用 | MEDIUM | active change 已显式传 `accumulate=false`，签名扩展加新参数无破坏；`git blame` 验证 |
| R4 | warp_id 单 warp 下行为与之前不一致 | LOW | `c_slot = 0 * 32 + 64 + lane_id = 64 + lane_id` 数学等价 active change 状态 |
| R5 | 同时改 handler + helper signature 扩大 diff | LOW | Phase 1 单 atomic commit 内 diff 集中于 4 文件，行级 diff ≤ 30 行 |
| R6 | PTX 解析不暴露 idesc（idesc 是内部寄存器名，PTX 用户赋名 `%r5`）→ 无法用真实 PTX 验证 | MEDIUM | T4/T5 测试直接构造 `instr.operands[3].reg = {name="%r5"}` + 设置 `thread.register_bank_["%r5"] = 1` 模拟 PTX 用户视角 |
| R7 | handler 运行时读寄存器与 SM scheduler 顺序执行假设冲突 | LOW | SM scheduler 顺序执行多 warp（per helper header comment "Currently safe because SM scheduler runs one warp at a time"），handler 内同步读寄存器无并发冲突 |

## Migration Plan

### Phase 1: Handler idesc Reading + Helper Signature (commit 1)

#### Baseline 函数清单

| 函数 | 文件:行 | 当前状态 |
|------|---------|---------|
| `processTcgen05Mma` | `src/ptxsim/instructions/tcgen05.cpp:355-393` | 显式 `accumulate=false` |
| `tcgen05_fragment_mma_f16` | `src/ptxsim/instructions/tcgen05_helpers.cpp:15-58` | 已扩展签名 `(Tmem&, bool)` |
| `Tcgen05Instr` struct | `include/ptx_ir/statement_context.h:180-190` | 无 `accumulate` 字段 |

#### 跨模块状态翻译表

**Phase 1 改动链**：
```
PTX: tcgen05.mma.accumulate::x.kind::f16.cta_group::1 [taddr], adesc, bdesc, idesc_reg;
                                                                              ^^^^^^^^^
                                                              OperandContext::Type::Reg (%r5)
                                                                              │
[parser] visitTcgen05Inst → instr.operands[3] = {RegOperand{"%r5"}}  ←  stage 1: grammar → IR (已存在)

[handler] processTcgen05Mma:
    + idesc_val = thread.read_reg_32(instr.operands[3].reg);   ←  stage 2: handler reads runtime register
    + accumulate = (idesc_val & 0x1u);                         ←  stage 3: extract accumulate bit (placeholder)
    + helper(tmem, warp_id, accumulate);                       ←  stage 4: pass to helper

[helper] tcgen05_fragment_mma_f16:
    c_slot = warp_id * 32 + 64 + lane_id;                       ←  stage 5: per-warp slot (single-warp == current)
    if (accumulate) { ... read existing C ... };               ←  stage 6: accumulate pre-load (already in active change)
```

**不涉及 ThreadContext state / WarpState / 互斥量变化**（仅 helper 内部 c_slot 偏移 + handler 调 helper）

#### 逐行 diff 计划

**`include/ptx_ir/statement_context.h`**：
- 第 189 行（`has_block_scale` 之后）新增：
  ```cpp
  bool accumulate = false;  // mma.accumulate::x semantic (per C1 fix)
  ```

**`include/ptxsim/instructions/tcgen05_helpers.h:51`**：
- 第 51 行从 `void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false);`
- 改为 `void tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id, bool accumulate = false);`
- doc comment 加："warp_id parameter added per C1+FU-4 sync. Single-warp callers pass `warp->get_warp_id()`."

**`src/ptxsim/instructions/tcgen05_helpers.cpp:23`**：
- 第 23 行从 `size_t c_slot = static_cast<size_t>(64) + static_cast<size_t>(lane_id);`
- 改为 `size_t c_slot = static_cast<size_t>(warp_id) * 32 + static_cast<size_t>(64) + static_cast<size_t>(lane_id);`

**`src/ptxsim/instructions/tcgen05.cpp:355-393`**：
- 第 355 行函数体起始加 idesc 读 accumulate bit block：
  ```cpp
  bool accumulate = false;
  if (instr.operands.size() >= 4 &&
      instr.operands[3].type == OperandContext::Type::Reg) {
      const auto& idesc_reg = instr.operands[3].reg;
      uint32_t idesc_val = thread.read_reg_32(idesc_reg);
      accumulate = (idesc_val & 0x1u) != 0;  // bit 0 placeholder
  }
  ```
- 第 383 行从 `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false);`
- 改为 `tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), accumulate);`

**`tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp`**：
- 新增 T4 TC（idesc=1 → 2× GOLDEN 累加）
- 新增 T5 TC（idesc=0 → 1× GOLDEN overwrite）

#### 回退策略
- Phase 1 commit 独立可 revert：`git revert <sha>` 后代码回到 active change Phase 1+2 状态
- helper signature 扩展是 1 次性：revert 后 helper 回到 `(Tmem&, bool)` 单调用者 `tcgen05.cpp:383` 传递位置

### Phase 2: Tests + ADR Postmortem (commit 2)

#### 任务清单
1. T4/T5 测试代码完整化（per D5 校准流程）
2. ADR-0016 追加 "2026-07-12 Postmortem: C1 fix" 段（含 idesc bit 位置实测记录）
3. `cd build && ctest -R "tcgen05" --output-on-failure` 验证

#### 回退策略
- ADR 追加段独立可 revert（不改源码）

### Phase 3: Archive (commit 3, per lessons-learned §6 Checklist G)

#### 任务清单（per lessons-learned §6 artifacts-first）
1. `git add openspec/changes/fix-tcgen05-idesc-parsing/{proposal,design,tasks}.md openspec/changes/fix-tcgen05-idesc-parsing/specs/**/*.md`
2. `git commit -m "docs(openspec): fix-tcgen05-idesc-parsing artifacts (Oracle C1, Pre-impl Review)"` (artifacts FIRST)
3. `openspec archive fix-tcgen05-idesc-parsing --yes`
4. **强制 postmortem prompt** 询问用户是否生成 postmortem

#### 回退策略
- archive 是 git-tracked 文件移动；如需修补，建新 `fix-*` change + Ref 链接（per lessons-learned §6/G）

## Open Questions

| ID | Question | Resolution Strategy | Status |
|----|----------|----------------------|--------|
| OQ1 | `ThreadContext::read_reg_32` accessor 是否存在？ | Phase 1.1 grep `register_bank_` API；如不存在，加最小 accessor + 单元测试 | TBD (Phase 1.1 实施前解决) |
| OQ2 | idesc bit 位置准确值（除 placeholder bit 0）？ | D5 校准流程 + ADR-0016 postmortem 记录 | TBD (Phase 1.2 实施时校准) |
| OQ3 | `warp->get_warp_id()` 返回类型与 helper 参数 `int warp_id` 是否一致？ | Phase 1.1 grep API 签名 | TBD (Phase 1.1 实施前验证) |
| OQ4 | active change Phase 1+2 是否已 merge（否则 helper 无 `accumulate` 参数）？ | `git log --oneline -- src/ptxsim/instructions/tcgen05_helpers.cpp` 验证 active commit | TBD (Phase 1.0 实施前检查) |
