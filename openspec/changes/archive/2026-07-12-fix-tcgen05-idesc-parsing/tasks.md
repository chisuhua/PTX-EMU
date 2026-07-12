# Tasks: Fix tcgen05.mma Handler — Read idesc.accumulate Bit (Oracle C1)

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + [specs/](specs/)
> **Ref** (active predecessor): [`../fix-tcgen05-mma-accumulator-and-f32-storage/`](../fix-tcgen05-mma-accumulator-and-f32-storage/) — 提供 helper `bool accumulate` 参数
> **Ref** (parallel FU-1, independent): [`../fix-tcgen05-commit-wait-group/`](../fix-tcgen05-commit-wait-group/) — C3 visitor IMMEDIATE 提取 pattern，与本 change 无依赖
> **范围**: 3 atomic commits (Phase 1: handler; Phase 2: tests + ADR; Phase 3: Archive)
> **Oracle 决策**: 2026-07-11 BLOCKER C1, session `ses_0aefd09c3ffeSqBIAGdxiRBFWC`
> **强制**: ptx-lessons-learned §3(分 Phase) + §4(基线 worktree) + §6(artifacts-first) + §7(Pre-impl Review) + §L(ANTLR)

## 0. Pre-Implementation Review + 验证

- [ ] 0.1 **Metis pre-implementation review** ✅ (Oracle Q1-Q6 audit 2026-07-11 覆盖本 change scope，4-way split validated)
- [ ] 0.2 **Oracle 决策建议** ✅ (BLOCKER C1 + Q4 Option a warp_id API + Q5 IMMEDIATE alternative = Q2 独立)
- [ ] 0.3 **验证 active predecessor 已 merge**（否则 helper 无 `accumulate` 参数）:
  ```bash
  git log --oneline -- src/ptxsim/instructions/tcgen05_helpers.cpp | head -5
  # 应包含 fix-tcgen05-mma-accumulator-and-f32-storage Phase 1+2 commit
  grep -n "bool accumulate" src/ptxsim/instructions/tcgen05_helpers.h
  ```
- [ ] 0.4 **验证 FU-1 (C3) 状态**（parallel non-blocking）:
  ```bash
  openspec status --change fix-tcgen05-commit-wait-group --json | jq '.isComplete'
  ```
- [ ] 0.5 **🛑 HARD GATE: 验证 `ThreadContext::read_reg_32` accessor 是否存在**（per OpenQuestion OQ1, **Oracle 2026-07-11 BLOCKER**）:
  ```bash
  # Oracle 实证：accessor 名虚构，真实 API 是 reg_access_ (RegisterAccessLayer unique_ptr, thread_context.h:45)
  grep -rn "read_reg_32\|register_bank_\|reg_access_\|RegisterAccessLayer" include/ptxsim/thread_context.h src/ptxsim/thread_context.cpp 2>/dev/null | head -15
  ```
  - **🛑 必须解决才能进入 Phase 1.1**：Oracle 实证确认 `read_reg_32` 不存在；当前 API 经 `reg_access_->acquire_register(op, qualifier)` + `get_register_bank_manager()` (thread_context.h:115, 261-262)
  - **修复路径**：在 `include/ptxsim/thread_context.h` 添加最小 accessor 声明 + 实现：
    ```cpp
    // include/ptxsim/thread_context.h 新增（public 方法）
    uint32_t read_reg_32(const RegOperand& reg) const;
    // 实现通过 reg_access_->acquire_register(reg, Qualifier{}) + 解析 uint32_t
    ```
  - 加单元测试：`tests/unit/core/test_thread_context_read_reg.cpp` 验证 register-name → uint32_t 路径
  - 此 HARD GATE 通过后，方可进入 §1.1 / §1.3 handler 改造
  - **NOTE**: 路径修正 — `thread_context.h` 在 `include/ptxsim/`，**不在** `include/ptxsim/core/`（原 tasks.md §0.5 路径错误）
- [ ] 0.6 **验证 `warp->get_warp_id()` 类型**（per OpenQuestion OQ3）:
  ```bash
  grep -n "get_warp_id" include/ptxsim/core/warp_context.h src/ptxsim/instructions/tcgen05_alloc.cpp
  ```
- [ ] 0.7 **建立基线 worktree** (per ptx-lessons-learned §4):
  ```bash
  git worktree add .worktrees/baseline-c1 HEAD
  cd .worktrees/baseline-c1
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # 必须全量
  cd build && ctest -L tcgen05 --output-on-failure
  ```
- [ ] 0.8 `git checkout -b fix/tcgen05-idesc-parsing`
- [ ] 0.9 **跑 baseline sanity check**:
  ```bash
  cd build && ctest -R "tcgen05" --output-on-failure
  ```
- [ ] 0.10 `./tests/ptx/test_all_ptx.sh` PASS（确保 ANTLR 无回归，per lessons-learned §L）

## 1. Phase 1: Handler idesc Reading + Helper Signature（commit 1）

### 1.1 字段 + 签名修改（前置无依赖）

- [ ] 1.1.1 读 `include/ptx_ir/statement_context.h:180-190` 当前 `Tcgen05Instr` 结构体
- [ ] 1.1.2 在 `has_block_scale`（第 189 行）之后新增字段:
  ```cpp
  // C1 fix: accumulate semantic driven by idesc register (per Oracle C1, 2026-07-11)
  // Populated at handler time from instr.operands[3] (idesc RegOperand) per
  // processTcgen05Mma at src/ptxsim/instructions/tcgen05.cpp
  bool accumulate = false;
  ```
- [ ] 1.1.3 验证 helper 签名（**Oracle 2026-07-11 实证：已是 3-param 最终态**）:
  ```bash
  grep -E "tcgen05_fragment_mma_f16.*warp_id.*accumulate" include/ptxsim/instructions/tcgen05_helpers.h
  # 应输出：void tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id, bool accumulate = false);
  ```
  - **NOTE**: 不再修改 helper 签名（per Oracle Q5 D3 决策沿用已实施 API）
- [ ] 1.1.4 验证 c_slot 公式（**Oracle 2026-07-11 实证：warp_id 偏移已落地**）:
  ```bash
  grep -E "c_slot.*warp_id.*\*.*32" src/ptxsim/instructions/tcgen05_helpers.cpp
  # 应输出 c_slot = warp_id * 32 + 64 + lane_id
  ```
  - **NOTE**: 不再修改 helper body（per Oracle Q5 D3 决策沿用已实施 c_slot 公式）

### 1.2 [REMOVED — 已在 active predecessor 实施]

- **🗑️ 已删除**：§1.2 "Helper body 修改（c_slot warp_id 偏移）" 全部任务（§1.2.1, §1.2.2）
- **原因**：Oracle 2026-07-11 实证 `tcgen05_helpers.cpp:42-44` 已是 `c_slot = warp_id * 32 + 64 + lane_id` 公式
- **替代**：§1.1.3 / §1.1.4 验证步骤确认现状即可，不再修改源码

### 1.3 Handler 改造（idesc 运行时读取）

- [ ] 1.3.1 读 `src/ptxsim/instructions/tcgen05.cpp:355-393` 当前 processTcgen05Mma
- [ ] 1.3.2 在 helper 调用之前（第 383 行附近）添加 idesc 读取逻辑（**NOTE: `thread.read_reg_32` 来自 §0.5 HARD GATE 实施的 accessor；当前实际 API 路径需按 `reg_access_->acquire_register` 实施时确认**）:
  ```cpp
  // C1 fix: extract accumulate bit from idesc register at handler time (per Oracle C1)
  // idesc is a RegOperand (PTX ISA §9.7.16, operand[3]) — accumulate bit is bit 0 placeholder
  bool accumulate = false;
  if (instr.operands.size() >= 4 &&
      instr.operands[3].type == OperandContext::Type::Reg) {
      const auto& idesc_reg = instr.operands[3].reg;
      uint32_t idesc_val = thread.read_reg_32(idesc_reg);  // from §0.5 HARD GATE accessor
      accumulate = (idesc_val & 0x1u) != 0;  // bit 0 placeholder; calibrate via T4/T5
  }
  ```
- [ ] 1.3.3 修改第 383 行 helper 调用:
  ```cpp
  // BEFORE: tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), /*accumulate=*/false);
  //                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //                                                  已传 warp_id (active predecessor 实施)；仅改 accumulate 参数
  // AFTER:  tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), accumulate);
  ```

### 1.4 Integration 测试：T4/T5（idesc-driven accumulate）

> **🛑 Oracle 2026-07-11 BLOCKER**: 原 tasks.md §1.4.2-§1.4.4 测试代码示例**不可编译**：
> - `rig.thread().register_bank_["%r5"]` — `register_bank_` 不是 ThreadContext 成员（实际是 `reg_access_`）
> - `RegOperand{"%r5"}` — RegOperand 是 `{std::string name; int index}` 两字段，正确形式 `RegOperand{"r", 5}`
> - `OperandContext{RegOperand{"%r5"}}` — brace-init 推断有歧义风险（OperandContext::data 是 std::variant）
> 
> **修复**: 改为**规格化**测试设计（不写伪代码），实施时根据 §0.5 HARD GATE 实施的真实 RegisterAccessLayer API 编写可编译代码。

- [ ] 1.4.1 读 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` 当前结构（已有 `TestRig`, `fill_tmem_with_golden_inputs`, `make_regular_mma_instr`, `require_c_slot_matches` 助手）
- [ ] 1.4.2 添加 T4 TC（idesc accumulate bit=1 → 2× GOLDEN 累加）— **规格化描述**:
  ```text
  TC: "processTcgen05Mma with idesc accumulate bit set yields accumulation"
  Tag: [integration][tcgen05][mma][idesc][accumulate]
  
  Setup:
    - TestRig rig
    - fill_tmem_with_golden_inputs(rig.tmem())  // A=1..32, B=1..4
    - 通过 §0.5 实施的 accessor 路径，设置 thread 持有寄存器（idesc RegOperand 指向的 uint32_t）值 = 0x1u
      /* 注: 具体 API 由 §0.5 实施决定（如 thread.read_reg_32(reg) 或
         rig.set_register_value(reg, value) 等）；使用 RegOperand{"r", 5}
         表示 %r5，per operand_context.h:12-13 二字段定义 */
    - make_regular_mma_instr() + 设置 instr.operands[3] = OperandContext(RegOperand{"r", 5})
  
  Action:
    - ptxsim::processTcgen05Mma(&rig.thread(), instr)  // 1st mma
    - ptxsim::processTcgen05Mma(&rig.thread(), instr)  // 2nd mma
  
  Assertion:
    - require_c_slot_matches(rig.tmem(), 2x golden_2x, "after 2nd mma with idesc accumulate=1")
    - 注: golden_2x = 2 × GOLDEN_MMA_F16_F16_F32（per tests/reference/ptx_tcgen05/tcgen05_mma_golden.h）
  ```
- [ ] 1.4.3 添加 T5 TC（idesc accumulate bit=0 → 1× GOLDEN overwrite）— **规格化描述**:
  ```text
  TC: "processTcgen05Mma with idesc accumulate bit cleared yields overwrite"
  Tag: [integration][tcgen05][mma][idesc][overwrite]
  
  Setup: 同 T4，但寄存器值 = 0x0u（accumulate bit cleared）
  
  Action: 同 T4（连续 2 次 mma）
  
  Assertion:
    - require_c_slot_matches(rig.tmem(), golden_1x, "after 2nd mma with idesc accumulate=0")
    - 注: 2nd mma 后 C == 1 × GOLDEN（overwrite 保留，与 active change T1_overwrite 语义一致）
  ```
- [ ] 1.4.4 添加 T6 TC（idesc bit 位置 calibration helper）— **规格化描述**:
  ```text
  TC: "idesc bit calibration: bit 0 baseline test for future bit-position discovery"
  Tag: [integration][tcgen05][mma][idesc][calibration]
  
  Purpose: 提供位掩码发现框架。若 T4/T5 FAIL，依次尝试以下位掩码值:
    - 0x1u (bit 0, current placeholder)
    - 0x2u (bit 1)
    - 0x4u (bit 2)
    - 0x8u (bit 3)
  
  Assertion: 找到使 T4 PASS（idesc=mask → 2x GOLDEN）+ T5 PASS（idesc=0 → 1x GOLDEN）的位掩码
  Output: 记录到 ADR-0016 Postmortem 段（per design.md D5 calibration procedure）
  ```
- [ ] 1.4.5 **T4/T5 实施后首次运行验证**:
  - 若 T4 FAIL：检查 §0.5 实施的 accessor 路径是否正确返回 uint32_t；尝试 D5 位掩码候选
  - 若 T5 FAIL：检查 RegOperand 解析（用 `RegOperand{"r", 5}` 二字段形式，不是单字段字符串）
  - 所有修正记录到 ADR-0016 Postmortem 段

### 1.5 PTX 语法 fixture（per lessons-learned §L TDD）

- [ ] 1.5.1 复制 `bench/cute/*.ptx` 中含 `tcgen05.mma` 语法的文件到 `tests/ptx/regression_*.ptx`（per §L real-kernel guard）
- [ ] 1.5.2 创建 `tests/ptx/tcgen05_mma_with_accumulate.ptx` 含 `.accumulate::x` 语法:
  ```ptx
  .version 8.0
  .target sm_100
  .address_size 64
  
  .visible .entry kernel_mma_accumulate(
      .param .u64 output
  ) {
      .reg .u32 %r<5>;
      .reg .u64 %rd<2>;
      // ... 真实 PTX mmio.accumulate::x 语法
      tcgen05.mma.accumulate::x.kind::f16.cta_group::1 [%rd0], %rd1, %rd2, %r0;
      ret;
  }
  ```
- [ ] 1.5.3 跑 `./tests/ptx/test_all_ptx.sh` 全绿（per §L GREEN step）

### 1.6 验证

- [ ] 1.6.1 `cmake --build build` 编译通过（验证 helper signature 扩展编译验证）
- [ ] 1.6.2 `cd build && ctest -R "tcgen05" --output-on-failure` 全 PASS
- [ ] 1.6.3 `./scripts/sanity.sh --quick` PASS（覆盖 min ctest baseline）
- [ ] 1.6.4 `./tests/ptx/test_all_ptx.sh` 47/47 PASS（per §L discipline）
- [ ] 1.6.5 **对比 baseline worktree** (per lessons-learned §4):
  ```bash
  cd .worktrees/baseline-c1/build && ctest -L tcgen05 --output-on-failure
  # 预期：baseline 全 PASS + main 新增 T4/T5 PASS（无 regression）
  ```

### 1.7 Commit

- [ ] 1.7.1 `git add include/ptx_ir/statement_context.h include/ptxsim/instructions/tcgen05_helpers.h src/ptxsim/instructions/tcgen05_helpers.cpp src/ptxsim/instructions/tcgen05.cpp tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp tests/ptx/tcgen05_mma_with_accumulate.ptx`
- [ ] 1.7.2 `git commit -m "fix(tcgen05): read accumulate bit from idesc register in processTcgen05Mma (Oracle C1, helper warp_id extension)"`
- [ ] 1.7.3 验证 commit: `git show HEAD --stat | head -30`
- [ ] 1.7.4 **失败处理**: 任何已有测试回归 → 立即 revert 该 commit（per lessons-learned §3）

## 2. Phase 2: ADR Postmortem + idesc bit 位置记录（commit 2）

- [ ] 2.1 读 `docs/adr/0016-blackwell-only-tcgen05.md` 找到最末段
- [ ] 2.2 追加 "2026-07-12 Postmortem: C1 fix" 段:
  ```markdown
  ## 2026-07-12 Postmortem: C1 fix
  
  ### C1 Root Cause
  `processTcgen05Mma` (`src/ptxsim/instructions/tcgen05.cpp:355-393`) 显式硬编码
  `accumulate=false` 调用 `tcgen05_fragment_mma_f16`，无法从真实 PTX
  `mma.accumulate::x` 的 idesc 寄存器读取语义。idesc 是 PTX ISA §9.7.16 中
  operand[3] RegOperand，运行时携带 accumulate bit（NVIDIA 内部位置未公开）。
  
  ### C1 Fix
  Handler 运行时从 `instr.operands[3]` 读 `uint32_t` 值 → 提取 accumulate bit
  → 动态决定 helper 参数。Helper 签名同步扩展 `+int warp_id`（per FU-4 API
  对齐），c_slot 公式 `warp_id * 32 + 64 + lane_id`。
  
  ### idesc bit 位置实测结果
  <!-- per design.md D5 calibration procedure; 实际位掩码记录 -->
  - 测试环境: <...>
  - 最终位掩码: `0x<u>u` (bit <N>)
  - 校准过程: <...>
  
  ### Known Limitations (debt for future)
  - 仅解析 accumulate bit；其他 idesc bits (dtype / scale_format / etc.)
    使用 helper 默认行为。完整 idesc 解析需后续 change
    `fix-tcgen05-idesc-full-parsing`。
  ```
- [ ] 2.3 `cd build && ctest -R "tcgen05" --output-on-failure` 最终验证
- [ ] 2.4 对比 baseline worktree（与 Phase 1 + ADR 改动叠加验证）
- [ ] 2.5 `git add docs/adr/0016-blackwell-only-tcgen05.md`
- [ ] 2.6 `git commit -m "docs(adr): ADR-0016 postmortem C1 fix + idesc bit position record"`
- [ ] 2.7 验证 commit: `git show HEAD --stat`

## 3. Phase 3: Archive + 强制 Postmortem Prompt（commit 3, per lessons-learned §6/G）

### 3.1 Artifacts git-tracked（artifacts FIRST per §6）

- [ ] 3.1.1 `git status openspec/changes/fix-tcgen05-idesc-parsing/` 验证 4 个 md + specs/ 在 working tree
- [ ] 3.1.2 `git add openspec/changes/fix-tcgen05-idesc-parsing/{proposal,design,tasks}.md openspec/changes/fix-tcgen05-idesc-parsing/specs/**/*.md`
- [ ] 3.1.3 `git commit -m "docs(openspec): fix-tcgen05-idesc-parsing artifacts (Oracle C1, Pre-impl Review)"`
- [ ] 3.1.4 验证: `git ls-files openspec/changes/fix-tcgen05-idesc-parsing/` 不应为空

### 3.2 Archive change

- [ ] 3.2.1 `openspec archive fix-tcgen05-idesc-parsing --yes`
- [ ] 3.2.2 验证: `git log --all --oneline -- "openspec/changes/fix-tcgen05-idesc-parsing/"` 应包含 archive commit
- [ ] 3.2.3 `cd build && ctest --output-on-failure` 全量验证（per Checklist D）
- [ ] 3.2.4 `./tests/ptx/test_all_ptx.sh` 验证（per §L）
- [ ] 3.2.5 `git add openspec/changes/archive/` + commit `chore(openspec): archive fix-tcgen05-idesc-parsing`

### 3.3 强制 Postmortem Prompt（per openspec-archive-change skill）

- [ ] 3.3.1 **必须询问用户**: "是否生成 postmortem？(Yes/No/Defer)"
- [ ] 3.3.2 若 Yes: 追加 `.opencode/notes/postmortem-fix-tcgen05-idesc-parsing.md` + commit
- [ ] 3.3.3 若 Defer: 在 `.opencode/notes/` 留 TODO 项

### 3.4 最终验证 + 清理

- [ ] 3.4.1 `cd build && ctest --output-on-failure` 全量 PASS
- [ ] 3.4.2 `./tests/ptx/test_all_ptx.sh` 全量 PASS
- [ ] 3.4.3 `git log --oneline -10` 验证 3 commits 都已落地
- [ ] 3.4.4 `git worktree remove .worktrees/baseline-c1` 清理 baseline worktree（per lessons-learned §4 标准操作）

## 关键禁止（per ptx-lessons-learned）

- ❌ 不许 1 commit 同时改 handler idesc 读取 + helper warp_id 扩展 + ADR postmortem（lessons-learned §3 — 当前已是 3 commits）
- ❌ 不许跳过 baseline worktree（lessons-learned §4）
- ❌ 不许 amend 已归档的 `archive/2026-07-10-implement-tcgen05-handlers-extended` 或 active predecessor（lessons-learned §6/G Checklist G）
- ❌ 不许 helper signature change 时忘记更新 `processTcgen05Mma` 调用点（lessons-learned §1 跨模块状态翻译审计）
- ❌ 不许 T4 反转后忘记更新 test name + 注释（避免 future 维护者混淆）
- ❌ **不许用 ctest 代替 `./tests/ptx/test_all_ptx.sh`** 验证 PTX 语法解析（lessons-learned §L）
- ❌ **不许修改 grammar**（per本 change Non-Goals + active change design.md D1.1）

## Effort 估算（per Oracle 2026-07-11 review 校准）

| Phase | Tasks | 估计时间 |
|-------|-------|---------|
| Phase 0 (Pre-impl) | 0.1-0.10 baseline + **HARD GATE §0.5 accessor 实施** | 1.5h |
| Phase 1 (handler only — D3 已实施删除) | 1.1.1-1.1.2 + 1.1.3-1.1.4 verify-only + 1.3.1-1.3.3 + 1.4 + 1.5 + 1.6 + 1.7 | **2-3h** (vs 原 3-4h，因 D3 已不实施) |
| Phase 2 (ADR) | 2.1-2.7 idesc bit 校准 + postmortem 段 | 1h |
| Phase 3 (Archive) | 3.1-3.4 artifacts + archive + postmortem prompt | 30min |
| **总计** | | **5-6h** (vs 原 5.5-6.5h；D3 工作删除节约 1h 但 HARD GATE 增加 0.5h) |
