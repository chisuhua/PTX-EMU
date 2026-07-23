# Tasks: Fix tcgen05.ld/.st/.cp Hardcoded TMEM Slot Routing

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + [specs/](specs/)
> **前置**: [`../fix-tcgen05-commit-wait-group/`](../fix-tcgen05-commit-wait-group/) (FU-1, C3) **必须先 archive** — 本 change visitor 借用其 IMMEDIATE walk pattern
> **Ref** (不能 amend 的已归档 change): [`../../archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/)
> **范围**: 3 atomic commits (Phase 1: grammar + IR; Phase 2: handlers + tests; Phase 3: Archive)
> **Oracle 决策**: 2026-07-11 (`ses_0b3791d78ffewb52428kJJ2Irz`) C2 BLOCKER HIGH confidence
> **强制**: ptx-lessons-learned §3 (Phase commits) + §4 (baseline worktree) + §6 (artifacts-first) + §7 (Pre-impl Review) + §9/§L (ANTLR bare-token 反模式)

## 0. Pre-Implementation Review

- [ ] 0.1 验证 FU-1 (`fix-tcgen05-commit-wait-group`) **已 archive**（其 visitor IMMEDIATE walk pattern 是本 change Phase 1.3 借鉴基础）
  ```bash
  test -f ../fix-tcgen05-commit-wait-group/specs/tcgen05-handlers-core/spec.md && echo "FU-1 archived" || echo "FU-1 NOT archived — block"
  ```
- [ ] 0.2 **真实 PTX 语法验证**（关键决策点 — Operand 路径 vs Qualifier 路径）:
  ```bash
  # 编译含 tcgen05.ld 的 kernel 提取 PTX（需 CUDA Toolkit）
  nvcc -ptx -arch=sm_100 -keep --no-compress -o /tmp/cute_rmsnorm.ptx \
      bench/cute/cute_rmsnorm.cu 2>/dev/null || echo "sm_100 tcgen05 not in ptxas — use cutlass source"
  cuobjdump -xptx /tmp/cute_rmsnorm.ptx 2>&1 | grep -A2 -B1 "tcgen05.ld" | head -20
  # 决策:
  #   含数字 slot 操作数 → Operand 路径（D2 默认）
  #   不含 slot 操作数 → Qualifier 路径 (`.tmem_slot::N`)
  ```
- [ ] 0.3 跑 `cd build && ctest -R "tcgen05" --output-on-failure` 确认 baseline
- [ ] 0.4 跑 `./tests/ptx/test_all_ptx.sh` 确认 47/47 baseline
- [ ] 0.5 **建立基线 worktree** (per ptx-lessons-learned §4):
  ```bash
  # 基线 = FU-1 archive 后的最新 commit
  git worktree add .worktrees/baseline-c2 HEAD
  cd .worktrees/baseline-c2
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # 必须全量 build
  cd build && ctest -L tcgen05 --output-on-failure
  ```
- [ ] 0.6 `git checkout -b fix/tcgen05-ld-st-slot-routing`
- [ ] 0.7 验证 `ls .worktrees/baseline-c2/` 目录非空（per ptx-lessons-learned §7 实测验证）

## 1. Phase 1: Grammar + IR + Visitor + Factory (commit 1)

### 1.1 Grammar 修改（per design.md D1 + D2）

- [ ] 1.1.1 读 `src/grammar/ptxInstructions.g4:432-492` 确认 `tcgen05Inst` 和 `tcgen05Operands` 规则
- [ ] 1.1.2 读 `include/ptx_ir/ptx_op.def:130-132` 确认 ld/st/cp 当前 `op_count`
- [ ] 1.1.3 **Pre-decision 验证** (Step 0.2 已决定 Operand 或 Qualifier 路径):
  - **Operand 路径** (D2 首选): 在 `tcgen05Operand` 规则添加 `tcgen05Slot : UNSIGNED_INT;`
  - **Qualifier 路径** (D2 fallback): 在 `tcgen05Qual` 规则添加 `| TCGEN_TMEM_SLOT COLONCOLON IMMEDIATE`（且 lexer 不引入 bare token — 用 `.tmem_slot` ID 形式）
- [ ] 1.1.4 修改 `src/grammar/ptxInstructions.g4` 添加 tmem_slot 规则 — **禁止 bare string token**（per ptx-lessons-learned §9/§L: ad808e3 案例教训）
- [ ] 1.1.5 修改 `include/ptx_ir/ptx_op.def:130-132`:
  ```cpp
  // BEFORE: X(S_TCGEN05_LD,  tcgen05.ld,  Tcgen05, 2, TCGEN05_INSTR, tensor)
  // AFTER:  X(S_TCGEN05_LD,  tcgen05.ld,  Tcgen05, 3, TCGEN05_INSTR, tensor)
  X(S_TCGEN05_LD,         tcgen05.ld,         Tcgen05,    3, TCGEN05_INSTR, tensor)
  X(S_TCGEN05_ST,         tcgen05.st,         Tcgen05,    3, TCGEN05_INSTR, tensor)
  X(S_TCGEN05_CP,         tcgen05.cp,         Tcgen05,    4, TCGEN05_INSTR, tensor)
  ```

### 1.2 IR + Factory 修改

- [ ] 1.2.1 读 `include/ptx_ir/statement_context.h:180-190` 当前 `Tcgen05Instr` 结构体
- [ ] 1.2.2 加 `uint32_t tmem_slot = 0;` 字段（per design.md D3: 单字段，默认 0 保持向后兼容）— 添加注释解释默认值与向后兼容性的关系（per lessons-learned §C）
- [ ] 1.2.3 读 `include/ptx_ir/statement_factory.h:265-292` `makeTcgen05Instr` 当前签名
- [ ] 1.2.4 修改 `makeTcgen05Instr` 加可选 `uint32_t tmem_slot = 0` 参数（per design.md D4）
- [ ] 1.2.5 验证现有 `makeTcgen05Instr` 调用点（grep `makeTcgen05Instr` 全部 src/ + tests/）— 默认参数应保持零行为变化

### 1.3 Visitor 提取 tmem_slot（per FU-1 pattern）

- [ ] 1.3.1 读 `src/ptx_parser/ptx_visitor.cpp:841-885` `visitTcgen05Inst` 当前实现（FU-1 应已加 cta_group walk）
- [ ] 1.3.2 在 `visitTcgen05Inst` 已有 IMMEDIATE walk 后，添加 `tmem_slot` 提取逻辑:
  ```cpp
  // Operand 路径 (D1 首选):
  uint32_t tmem_slot = 0;
  if (op_kind == Tcgen05OpKind::LD || op_kind == Tcgen05OpKind::ST ||
      op_kind == Tcgen05OpKind::CP) {
      if (!instr.operands.empty() && instr.operands[0].type == OperandContext::Type::Imm) {
          tmem_slot = static_cast<uint32_t>(std::stoul(instr.operands[0].imm.value));
      }
  }
  // Qualifier 路径 (fallback): 复用 FU-1 的 qualifier 解析 pattern
  ```
- [ ] 1.3.3 调用 `makeTcgen05Instr(..., /*tmem_slot=*/... )` 传入
- [ ] 1.3.4 加注释解释 slot 提取路径选择（per lessons-learned §C: 3 个月后陌生人能理解的注释）

### 1.4 Parser 测试更新

- [ ] 1.4.1 读 `tests/integration/ptx/test_tcgen05_ld_parse.cpp` 当前 TC
- [ ] 1.4.2 添加新 TC `tcgen05.instr with tmem_slot operand parses correctly`:
  ```cpp
  TEST_CASE("Tcgen05Instr tmem_slot field defaults to 0 (backward compat)") {
      auto instr = make_tcgen05_ld_default();
      REQUIRE(instr.tmem_slot == 0);
  }
  TEST_CASE("Tcgen05Instr tmem_slot=32 after parse") {
      auto instr = make_tcgen05_ld_with_slot(32);
      REQUIRE(instr.operands.size() == 3);
      REQUIRE(instr.tmem_slot == 32);
  }
  ```
- [ ] 1.4.3 同上 for `test_tcgen05_st_parse.cpp`
- [ ] 1.4.4 同上 for `test_tcgen05_cp_parse.cpp`（若存在）
- [ ] 1.4.5 验证 `grep -rn "instr.operands.size() == 2" tests/integration/tcgen05/` 现有假设仍正确（除新加的 slot TC）

### 1.5 验证 + Commit

- [ ] 1.5.1 `cmake --build build --target GenerateParser` 重新生成 ANTLR 解析器（**必须**）
- [ ] 1.5.2 `./tests/ptx/test_all_ptx.sh` 必须 47/47 PASS（per lessons-learned §L — bare token 检测）
  - **若失败**: 立即 revert Phase 1 commit（per lessons-learned §3）
- [ ] 1.5.3 `cd build && ctest -R "tcgen05" --output-on-failure` 全 PASS（除 op_count 变化导致的 fixture 失败 — 必须修）
- [ ] 1.5.4 **对比 baseline worktree** (per ptx-lessons-learned §4):
  ```bash
  cd .worktrees/baseline-c2/build && ctest -L tcgen05 --output-on-failure
  # 预期: baseline 全 PASS, main 全 PASS
  ```
- [ ] 1.5.5 验证 `grep -rn "tmem_slot" include/ptx_ir/statement_context.h` 出现新字段
- [ ] 1.5.6 验证 `grep -rn "TMEM_SLOT\s*:" src/grammar/ptxLexer.g4` **应为空**（per §9 禁止 bare token）
- [ ] 1.5.7 **失败处理**: 任何已有测试回归 → 立即 revert 该 commit（per lessons-learned §3）
- [ ] 1.5.8 Commit:
  ```bash
  git add src/grammar/ptxInstructions.g4 include/ptx_ir/ptx_op.def \
          include/ptx_ir/statement_context.h include/ptx_ir/statement_factory.h \
          src/ptx_parser/ptx_visitor.cpp \
          tests/integration/ptx/test_tcgen05_ld_parse.cpp \
          tests/integration/ptx/test_tcgen05_st_parse.cpp \
          tests/integration/ptx/test_tcgen05_cp_parse.cpp
  git commit -m "fix(tcgen05): parse tmem_slot operand for ld/st/cp (Oracle C2 Phase 1)"
  ```
- [ ] 1.5.9 验证 commit: `git show HEAD --stat`

## 2. Phase 2: Handler 路由 + 行为测试 (commit 2)

### 2.1 Handler 修改（per design.md Phase 2 Step 2.1）

- [ ] 2.1.1 读 `src/ptxsim/instructions/tcgen05.cpp:402-439` `processTcgen05Ld` 当前实现
- [ ] 2.1.2 改 `tcgen05.cpp:434`:
  ```cpp
  // BEFORE: tmem.write(0, tmp, Tmem::kSlotSize);
  // AFTER:
  if (instr.tmem_slot >= Tmem::kSlotCount) {
      throw std::out_of_range(
          "tcgen05.ld: tmem_slot " + std::to_string(instr.tmem_slot) +
          " exceeds kSlotCount " + std::to_string(Tmem::kSlotCount));
  }
  tmem.write(instr.tmem_slot, tmp, Tmem::kSlotSize);
  PTX_DEBUG_EMU("tcgen05.ld: TMA desc global=0x%016lx → TMEM slot %u (%zu bytes)",
                desc->global_address, instr.tmem_slot, Tmem::kSlotSize);
  ```
- [ ] 2.1.3 加注释说明 `// Post-C2: slot comes from instr.tmem_slot, not hardcoded 0`（per lessons-learned §C）
- [ ] 2.1.4 读 `tcgen05.cpp:448-484` `processTcgen05St`
- [ ] 2.1.5 改 `tcgen05.cpp:476`:
  ```cpp
  // BEFORE: tmem.read(0, tmp, Tmem::kSlotSize);
  // AFTER:
  if (instr.tmem_slot >= Tmem::kSlotCount) {
      throw std::out_of_range("tcgen05.st: tmem_slot ...");
  }
  tmem.read(instr.tmem_slot, tmp, Tmem::kSlotSize);
  ```
- [ ] 2.1.6 读 `src/ptxsim/instructions/tcgen05_cp.cpp:127-156` `processTcgen05Cp`
- [ ] 2.1.7 删除 `constexpr size_t kDestSlot = 0;` (line 130) 与所有引用
- [ ] 2.1.8 改 `tcgen05_cp.cpp:138` 使用 `instr.tmem_slot`:
  ```cpp
  if (instr.tmem_slot >= Tmem::kSlotCount) {
      throw std::out_of_range("tcgen05.cp: tmem_slot ...");
  }
  tmem.write(instr.tmem_slot, tmp, Tmem::kSlotSize);
  ```

### 2.2 行为测试新增

- [ ] 2.2.1 读 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:181-203` `TestRig` 模式
- [ ] 2.2.2 创建新文件 `tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp`:
  - 包含 `TestRig` helper（复用现有持久化测试的 setup）
  - TC1: ld 到 slot 32 + st 从 slot 32 round-trip 验证 128B pattern
  - TC2: ld 到 default slot 0 backward compat 验证
  - TC3: invalid tmem_slot=999 throw `std::out_of_range` 验证
- [ ] 2.2.3 在 `tests/integration/tcgen05/CMakeLists.txt` 注册新测试:
  ```cmake
  add_catch_test(integration_ld_st_slot_routing test_tcgen05_ld_st_slot_routing.cpp)
  set_tests_properties(integration_ld_st_slot_routing PROPERTIES LABELS "integration;tcgen05")
  ```
- [ ] 2.2.4 验证 `grep -rn "kDestSlot" src/ptxsim/instructions/` 应为空（确认 cp 常量已删除）

### 2.3 验证 + Commit

- [ ] 2.3.1 `cmake --build build` 编译通过
- [ ] 2.3.2 `ctest -R "tcgen05" --output-on-failure` 全 PASS
- [ ] 2.3.3 `./scripts/sanity.sh --tier 2 --tier 8` PASS（包含 tcgen05 supporting tests）
- [ ] 2.3.4 对比 baseline (per ptx-lessons-learned §4):
  ```bash
  cd .worktrees/baseline-c2/build && ctest -L tcgen05 --output-on-failure
  # 预期: 除新增 test_tcgen05_ld_st_slot_routing 测试外，其他测试不变
  ```
- [ ] 2.3.5 验证 `grep -rn "tmem.write(0," src/ptxsim/instructions/tcgen05.cpp` 应为空
- [ ] 2.3.6 验证 `grep -rn "tmem.read(0," src/ptxsim/instructions/tcgen05.cpp` 应为空
- [ ] 2.3.7 验证 `grep -rn "kDestSlot" src/ptxsim/instructions/` 应为空
- [ ] 2.3.8 **失败处理**: 任何已有测试回归 → 立即 revert Phase 2 commit（per lessons-learned §3）
- [ ] 2.3.9 Commit:
  ```bash
  git add src/ptxsim/instructions/tcgen05.cpp src/ptxsim/instructions/tcgen05_cp.cpp \
          tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp \
          tests/integration/tcgen05/CMakeLists.txt
  git commit -m "fix(tcgen05): route ld/st/cp to instruction-specified tmem_slot (Oracle C2 Phase 2)"
  ```
- [ ] 2.3.10 验证 commit: `git show HEAD --stat`

## 3. Phase 3: Archive + ADR Postmortem (commit 3, per lessons-learned §6 Checklist G)

### 3.1 Artifacts git-tracked (artifacts FIRST per lessons-learned §6)

- [ ] 3.1.1 `git status openspec/changes/fix-tcgen05-ld-st-slot-routing/` 验证 4 个 md + specs/ 在 working tree
- [ ] 3.1.2 `git add openspec/changes/fix-tcgen05-ld-st-slot-routing/`
- [ ] 3.1.3 `git commit -m "docs(openspec): fix-tcgen05-ld-st-slot-routing artifacts (Oracle C2, Metis pre-impl review)"`
- [ ] 3.1.4 验证: `git ls-files openspec/changes/fix-tcgen05-ld-st-slot-routing/` 不应为空

### 3.2 ADR-0016 Postmortem 追加

- [ ] 3.2.1 读 `docs/adr/ADR-0016-blackwell-only-tcgen05.md` 找到最末段
- [ ] 3.2.2 追加 "2026-07-12 Postmortem: C2 fix" 段（per design.md Phase 3）:
  ```markdown
  ## 2026-07-12 Postmortem: C2 ld/st/cp slot routing fix

  ### C2 Root Cause
  `processTcgen05Ld` (tcgen05.cpp:434)、`processTcgen05St` (tcgen05.cpp:476)、
  `processTcgen05Cp` (tcgen05_cp.cpp:138) 三个 handler 均硬编码 TMEM slot `0`,
  与 mma 写 C 到 slot[64..95] 矛盾。FlashAttention 的 QK^T→softmax→PV
  数据流依赖 ld/st 在 mma 消费 slot 范围内移动数据 — 当前架构不可能。

  ### C2 Fix
  1. `Tcgen05Instr` (statement_context.h:180-190) 新增 `uint32_t tmem_slot = 0`
     字段,默认 0 保持向后兼容。
  2. `tcgen05Inst` grammar 规则增加 tmem_slot 操作数(或 `.tmem_slot::N` qualifier)。
  3. `makeTcgen05Instr` (statement_factory.h:265-292) 加可选 tmem_slot 参数。
  4. `visitTcgen05Inst` (ptx_visitor.cpp:841-885) 提取 tmem_slot。
  5. ld/st/cp handler 引用 `instr.tmem_slot` 替代硬编码 0。
  6. 加入 `out_of_range` 验证防止 silent fallback (超过 kSlotCount)。

  ### Known Semantic Gap (debt for future)
  - 多 warp slot 偏移由 FU-4 (`fix-tcgen05-multi-warp-fragment`) 处理
  - .mma.fragment 64x64 warp-cooperative layout 由 post-P2 follow-up 处理
  ```

### 3.3 ADR commit

- [ ] 3.3.1 `git add docs/adr/ADR-0016-blackwell-only-tcgen05.md`
- [ ] 3.3.2 `git commit -m "docs(adr): ADR-0016 postmortem C2 (Oracle 2026-07-11 ld/st/cp slot routing)"`

### 3.4 Archive change

- [ ] 3.4.1 跑 `openspec archive fix-tcgen05-ld-st-slot-routing --yes`
- [ ] 3.4.2 验证: `git log --all --oneline -- "openspec/changes/fix-tcgen05-ld-st-slot-routing/"` 应包含 archive commit
- [ ] 3.4.3 跑 `cd build && ctest --output-on-failure` 全量验证
- [ ] 3.4.4 跑 `./tests/ptx/test_all_ptx.sh` 验证
- [ ] 3.4.5 `git add openspec/changes/archive/` + commit `chore(openspec): archive fix-tcgen05-ld-st-slot-routing`

### 3.5 强制 Postmortem Prompt (per openspec-archive-change skill)

- [ ] 3.5.1 **必须询问用户**: "是否生成 postmortem？(Yes/No/Defer)"
- [ ] 3.5.2 若 Yes: 追加 `.opencode/notes/postmortem-fix-tcgen05-ld-st-slot-routing.md` + commit
- [ ] 3.5.3 若 Defer: 在 `.opencode/notes/` 留 TODO 项

### 3.6 最终验证

- [ ] 3.6.1 `cd build && ctest --output-on-failure` 全量 PASS
- [ ] 3.6.2 `./tests/ptx/test_all_ptx.sh` 全量 PASS (47/47)
- [ ] 3.6.3 `git log --oneline -10` 验证 3 commits 都已落地
- [ ] 3.6.4 `git worktree remove .worktrees/baseline-c2` 清理 baseline worktree

## 关键禁止 (per ptx-lessons-learned §3 + §9)

- ❌ 不许 1 commit 同时改 grammar + handler（Metis C5 + lessons-learned §3）
- ❌ 不许跳过 baseline worktree (lessons-learned §4)
- ❌ 不许 amend 已归档的 `archive/2026-07-10-implement-tcgen05-handlers-extended`
- ❌ 不许在 grammar 修改时引入 bare string lexer tokens (lessons-learned §9/§L — ad808e3 案例)
- ❌ 不许 Phase 2 漏改 readback/test fixtures (Metis C2 mitigation)
- ❌ 不许 invalid tmem_slot 静默 fallback 到 slot 0 (per design.md R6)
- ❌ 不许 FU-1 未 archive 就开始本 change (前置依赖，per design.md Cross-Change Dependencies)

## Effort 估算

| Phase | Tasks | 估计时间 |
|-------|-------|----------|
| Phase 0 (Pre-impl) | 0.1-0.7 (含 real PTX 验证) | 1-2h（含 baseline build） |
| Phase 1 (Grammar+IR) | 1.1-1.5 (~25 sub-tasks) | 3-4h |
| Phase 2 (Handlers+Tests) | 2.1-2.3 (~20 sub-tasks) | 2-3h |
| Phase 3 (Archive) | 3.1-3.6 (~15 sub-tasks) | 30min |
| **总计** | | **7-10h**（1-1.5 work days） |

## Cross-Change Checklist (per skills ptx-lessons-learned)

- [x] **Checklist A**: 函数迁移完整性 — `tcgen05.cpp:434,476` + `tcgen05_cp.cpp:138` 3 个硬编码点已列
- [x] **Checklist B**: 重构前 — baseline worktree + Phase 拆分 + 失败处理策略
- [x] **Checklist C**: 写注释 — handler 加注释解释 slot 来源
- [x] **Checklist D**: Commit 前 — baseline 对比 + ADR 追加 + tasks.md 状态变更 + commit message 编号
- [x] **Checklist E**: artifacts 提交顺序 — Phase 3 artifacts-first
- [x] **Checklist G**: OpenSpec lifecycle — Ref 链接 + 不 amend archived
- [x] **Checklist H**: Pre-impl Review — Oracle 2026-07-11 + 真实 PTX 验证 (Step 0.2)
- [x] **Checklist J**: 4 个 artifacts 内部一致性 — proposal Impact 与 design Migration 与 tasks LoC 数字对齐
- [x] **Checklist L**: ANTLR grammar modification — bare-token 反模式 + TDD (test_all_ptx.sh) + commit 顺序
- [x] **Checklist M**: FU-1 前置强制 — `0.1` 验证
