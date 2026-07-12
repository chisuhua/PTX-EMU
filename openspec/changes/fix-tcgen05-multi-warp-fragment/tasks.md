# Tasks: Fix tcgen05.mma Fragment — Multi-Warp Slot Offset (Oracle C4 BLOCKER)

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + [specs/tcgen05-multi-warp-fragment/spec.md](specs/tcgen05-multi-warp-fragment/spec.md) + [specs/tcgen05-handlers-extended/spec.md](specs/tcgen05-handlers-extended/spec.md)
> **Sister change**（必须先合并）: [`../fix-tcgen05-mma-accumulator-and-f32-storage/`](../fix-tcgen05-mma-accumulator-and-f32-storage/) — 提供带 `accumulate` 参数的 helper signature，本 change 在其基础上加 `warp_id`
> **Foundation change**（推荐先合并）: [`../fix-tcgen05-commit-wait-group/`](../fix-tcgen05-commit-wait-group/) — 4 个 follow-up 的基础前置
> **范围**: 2 atomic commits (Phase 1: helper sig + caller + 测试 + AGENTS.md; Phase 2: ADR postmortem + Archive)
> **Oracle 决策**: 2026-07-11 (`ses_0aefd09c3ffeSqBIAGdxiRBFWC`) C4 HIGH confidence + 4-way split Q1-Q6 验证
> **强制 skills**: `ptx-lessons-learned` §3(分 Phase) + §4(基线 worktree) + §6(artifacts-first) + §7(Pre-impl Review) + Checklist H/J

## 0. Pre-Implementation

- [ ] 0.1 **Metis pre-implementation review** (per ptx-lessons-learned §7/Checklist H — 4 个 artifacts 已 create):
  - 输入：当前目录下 4 个 artifacts + `src/ptxsim/instructions/tcgen05_helpers.h:51` + `src/ptxsim/instructions/tcgen05_helpers.cpp:23` + `src/ptxsim/instructions/tcgen05.cpp:383`
  - 要求：5 项 MUST-RESOLVE 全部解决才能 apply（hidden intentions / ambiguities / AI failure points / missing context）
  - 输出决策：GO / ⚠️ CONDITIONAL / ❌ NO-GO
  - **Note**: Oracle 已完成 Q1-Q6 split 验证（高置信度），本 task 可用 Oracle 结果作为前置输入简化 Metis 范围

- [ ] 0.2 验证 Oracle 引用真实存在:
  ```bash
  grep -n "c_slot = " src/ptxsim/instructions/tcgen05_helpers.cpp   # 验证 line 23 硬编码
  grep -n "tcgen05_fragment_mma_f16" src/ptxsim/instructions/tcgen05.cpp  # 验证 line 383 caller
  grep -n "warp_id\|get_warp_id" src/ptxsim/instructions/tcgen05_alloc.cpp  # 验证 WarpContext::get_warp_id 已存在
  ```

- [ ] 0.3 **验证 sister change 已合并** (per proposal §Dependencies):
  ```bash
  git log --all --oneline -- "src/ptxsim/instructions/tcgen05_helpers.h"
  # 期望看到 sister change H1+H2 commit（带 accumulate 参数的 signature）
  # 如果未合并：STOP — 等 sister change 先合并后从 main rebase
  ```

- [ ] 0.4 验证 baseline tcgen05-tagged 测试通过:
  ```bash
  . env.sh && cmake --build build && cd build && ctest -L "tcgen05" --output-on-failure
  # 预期：所有现有 tcgen05 测试 PASS（包含 sister change H1+H2 实施后状态）
  ```

- [ ] 0.5 跑 `./tests/ptx/test_all_ptx.sh` 确认 47/47 PASS（grammar 不变，预期无影响）

- [ ] 0.6 **建立基线 worktree** (per ptx-lessons-learned §4):
  ```bash
  git worktree add .worktrees/baseline-c4 $(git rev-parse HEAD)
  cd .worktrees/baseline-c4
  . env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # 必须全量 build
  cd build && ctest -L tcgen05 --output-on-failure
  # 预期：所有 tcgen05-tagged 测试 PASS（与 main 一致）
  ```

- [ ] 0.7 创建工作分支:
  ```bash
  cd /workspace/project/PTX-EMU && git checkout -b fix/tcgen05-multi-warp-fragment
  ```

## 1. Phase 0 — Artifacts FIRST (per ptx-lessons-learned §6 Checklist E)

- [ ] 1.1 验证 4 个 OpenSpec artifacts 在 working tree 且 git-tracked:
  ```bash
  git status openspec/changes/fix-tcgen05-multi-warp-fragment/
  ls openspec/changes/fix-tcgen05-multi-warp-fragment/{proposal,design,tasks}.md
  ls openspec/changes/fix-tcgen05-multi-warp-fragment/specs/tcgen05-multi-warp-fragment/spec.md
  ls openspec/changes/fix-tcgen05-multi-warp-fragment/specs/tcgen05-handlers-extended/spec.md
  ```
- [ ] 1.2 `git add openspec/changes/fix-tcgen05-multi-warp-fragment/`
- [ ] 1.3 commit "docs(openspec): fix-tcgen05-multi-warp-fragment artifacts (Oracle C4 BLOCKER, Q1-Q6 split validation)"
  ```bash
  git commit -m "docs(openspec): fix-tcgen05-multi-warp-fragment artifacts (Oracle C4 BLOCKER, Q1-Q6 split validation)

  - proposal.md: Oracle C4 root cause + Decision D1-D6 摘要
  - design.md: Migration Plan + Risks R1-R7 + 影响范围表格
  - specs/tcgen05-multi-warp-fragment/spec.md: NEW capability (multi-warp fragment layout)
  - specs/tcgen05-handlers-extended/spec.md: MODIFIED delta spec (helper signature adds warp_id)

  Refs:
    - Oracle session ses_0aefd09c3ffeSqBIAGdxiRBFWC (C4 HIGH confidence)
    - Sister fix-tcgen05-mma-accumulator-and-f32-storage (H1+H2, must merge first)
    - Foundation fix-tcgen05-commit-wait-group (C3, recommended before)"
  ```
- [ ] 1.4 验证: `git ls-files openspec/changes/fix-tcgen05-multi-warp-fragment/ | wc -l` → 4 个 md 文件

## 2. Phase 1 — Helper + Caller + 测试 + AGENTS.md sync (1 atomic commit)

### 2.1 Helper signature 修改

- [ ] 2.1.1 读 `include/ptxsim/instructions/tcgen05_helpers.h:51` 当前签名
- [ ] 2.1.2 修改签名为:
  ```cpp
  void tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id,
                                bool accumulate = false);
  ```
- [ ] 2.1.3 添加 doc comment（per 设计 D1 + §Checklist C）：
  ```
  // warp_id: per-warp slot offset to prevent multi-warp C slot conflict.
  //          - 0 = single-warp mode (backward compatible)
  //          - N = warp N owns C slots [N*32+64 : N*32+95]
  //          - A/B slots [0..63] remain shared input fragments.
  //          - Caller MUST pass warp->get_warp_id() (or 0 for single-warp code).
  //          - Throws std::invalid_argument if warp_id < 0.
  ```
- [ ] 2.1.4 更新 Layout 段（移除 "Currently safe because SM scheduler runs one warp at a time"）：
  ```
  // Layout (multi-warp aware, post C4 fix):
  //   A slots: [0..63]      (shared input fragments, lane_id * 2)
  //   B slots: [0..63]      (shared input fragments, lane_id * 2 + 1)
  //   C slots: [64..95]     (per-warp owned, warp_id * 32 + 64 + lane_id)
  //   warp 0: [64..95], warp 1: [96..127], warp 2: [128..159], warp 3: [160..191]
  //   Each warp owns 32 unique slots; A/B remain shared input.
  ```

### 2.2 Helper body 修改（slot 计算 + warp_id 校验）

- [ ] 2.2.1 读 `src/ptxsim/instructions/tcgen05_helpers.cpp:15-23` 当前实现
- [ ] 2.2.2 在函数体最顶部加 warp_id 校验:
  ```cpp
  if (warp_id < 0) {
      throw std::invalid_argument(
          "tcgen05_fragment_mma_f16: warp_id must be >= 0 (got "
          + std::to_string(warp_id) + ")");
  }
  ```
- [ ] 2.2.3 改 c_slot 计算（line 23）:
  ```cpp
  // BEFORE: size_t c_slot = static_cast<size_t>(64) + static_cast<size_t>(lane_id);
  // AFTER:
  size_t c_slot = static_cast<size_t>(warp_id) * 32
                + static_cast<size_t>(64)
                + static_cast<size_t>(lane_id);
  ```
- [ ] 2.2.4 A/B slot 公式保持不变（per 设计 D2）：
  ```cpp
  size_t a_slot = static_cast<size_t>(lane_id) * 2;     // 不变
  size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1; // 不变
  ```

### 2.3 调用点更新 (1 个 production caller)

- [ ] 2.3.1 读 `src/ptxsim/instructions/tcgen05.cpp:355-393` 当前 `processTcgen05Mma` 实现
- [ ] 2.3.2 确认 `WarpContext* warp` 在 function scope 内可访问（per `tcgen05.cpp:358-360` 模式）
- [ ] 2.3.3 改 line 383 调用:
  ```cpp
  // BEFORE: tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false);
  // AFTER:  tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), /*accumulate=*/false);
  ```
- [ ] 2.3.4 编译验证: `cmake --build build` 必须 PASS（编译期强制 — Oracle C1 mitigation）

### 2.4 集成测试新增 `test_tcgen05_mma_multi_warp.cpp`

- [ ] 2.4.1 创建 `tests/integration/tcgen05/test_tcgen05_mma_multi_warp.cpp`
- [ ] 2.4.2 包含必要 headers:
  ```cpp
  #include "ptx_ir/statement_factory.h"
  #include "ptxsim/sm_context.h"
  #include "ptxsim/testing/scheduler_utils.h"   // step_warp
  #include "ptxsim/testing/instruction_helpers.h"
  #include "ptxsim/testing/predicates.h"
  #include "ptxsim/instructions/tcgen05_helpers.h"  // helper direct call
  #include "tests/reference/ptx_tcgen05/tcgen05_mma_golden.h"
  ```
- [ ] 2.4.3 TC1: 单 warp 向后兼容
  ```cpp
  TEST_CASE("tcgen05_fragment_mma_f16 with warp_id=0 matches pre-C4 behavior") {
      // 调用: tcgen05_fragment_mma_f16(tmem, /*warp_id=*/0, /*accumulate=*/false)
      // 断言: C slot 64+lane_id 等于 pre-C4 值（向后兼容）
  }
  ```
- [ ] 2.4.4 TC2: 2-warp C slot 写入 warp 0
  ```cpp
  TEST_CASE("processTcgen05Mma from warp 0 writes C to slots [64..95]") {
      // 配置: SMContext(2, 128, 4096, 0) — 2 warps
      // 切换到 warp 0，调 processTcgen05Mma
      // 断言: tmem.read(64, ...) == GOLDEN, tmem.read(96, ...) == 0
  }
  ```
- [ ] 2.4.5 TC3: 2-warp C slot 写入 warp 1
  ```cpp
  TEST_CASE("processTcgen05Mma from warp 1 writes C to slots [96..127]") {
      // 配置: SMContext(2, 128, 4096, 0) — 2 warps
      // 切换到 warp 1，调 processTcgen05Mma
      // 断言: tmem.read(96, ...) == GOLDEN, tmem.read(64, ...) == 0
  }
  ```
- [ ] 2.4.6 TC4: 2-warp 并行 mma 无冲突
  ```cpp
  TEST_CASE("2 warps mma in parallel do not conflict on C slot") {
      // 配置: SMContext(2, 128, 4096, 0)
      // 切换 active_mask 让 2 warps 同时 active
      // 调 step_warp 同时执行两个 warp 的 mma
      // 断言: tmem.read(64..95) == warp 0 GOLDEN, tmem.read(96..127) == warp 1 GOLDEN
  }
  ```
- [ ] 2.4.7 TC5: warp_id 越界异常
  ```cpp
  TEST_CASE("tcgen05_fragment_mma_f16 with warp_id=-1 throws invalid_argument") {
      // 期望: REQUIRE_THROWS_AS(
      //     ptxsim::tcgen05_fragment_mma_f16(tmem, -1, false),
      //     std::invalid_argument)
  }
  ```

### 2.5 CMake 注册

- [ ] 2.5.1 读 `tests/integration/tcgen05/CMakeLists.txt` 当前 add_catch_test 调用
- [ ] 2.5.2 新增 ctest target:
  ```cmake
  add_catch_test(integration_tcgen05_mma_multi_warp
      test_tcgen05_mma_multi_warp.cpp
  )
  set_tests_properties(integration_tcgen05_mma_multi_warp PROPERTIES
      LABELS "integration;tcgen05;mma;multi_warp")
  ```
- [ ] 2.5.3 命名约束验证: target 名带 `integration_` 前缀（per AGENTS.md 命名约束）

### 2.6 AGENTS.md 已知限制表更新（per lessons-learned §8 + Checklist I）

- [ ] 2.6.1 读根 `AGENTS.md` 已知限制表
- [ ] 2.6.2 定位 "single-warp 顺序执行" 描述（per `tcgen05_helpers.h:43-46` 旧注释）
- [ ] 2.6.3 改写为: "Multi-warp fragment layout 已支持（per `fix-tcgen05-multi-warp-fragment` — Oracle C4 fix）"
- [ ] 2.6.4 验证: `grep -n "single-warp" AGENTS.md` 输出为空

### 2.7 验证

- [ ] 2.7.1 `cmake --build build` 编译通过
- [ ] 2.7.2 `ctest -R "tcgen05" --output-on-failure` 全 PASS（含现有 + 新 TC1-5）
- [ ] 2.7.3 `ctest -R "integration_tcgen05_mma_multi_warp" -V` 5 TC 全部 PASS
- [ ] 2.7.4 `./scripts/sanity.sh --tier 2 --tier 8` PASS
- [ ] 2.7.5 `./tests/ptx/test_all_ptx.sh` 47/47 PASS（grammar 不变）
- [ ] 2.7.6 **对比 baseline worktree** (per ptx-lessons-learned §4):
  ```bash
  # baseline 单 warp 路径数值与本 change warp_id=0 路径应该完全一致
  cd .worktrees/baseline-c4/build && ctest -L tcgen05 --output-on-failure
  # 预期：baseline 单 warp 测试 PASS；本 change 单 warp 测试 PASS + 5 个新 multi-warp 测试 PASS
  ```
- [ ] 2.7.7 **失败处理**: 任何已有测试回归 → 立即 revert 该 commit（per lessons-learned §3 + Oracle Q5）
- [ ] 2.7.8 静态验证 grep:
  ```bash
  # 验证 c_slot 公式已变更
  grep -n "size_t c_slot = static_cast<size_t>(64)" src/ptxsim/instructions/tcgen05_helpers.cpp
  # 期望：无匹配（已改为 warp_id * 32 + 64 + lane_id）

  # 验证 caller 已传 warp_id
  grep -n "tcgen05_fragment_mma_f16(tmem," src/ptxsim/instructions/tcgen05.cpp
  # 期望：无匹配（已改为传 warp_id）
  ```

### 2.8 Commit

- [ ] 2.8.1 `git add include/ptxsim/instructions/tcgen05_helpers.h src/ptxsim/instructions/tcgen05_helpers.cpp src/ptxsim/instructions/tcgen05.cpp tests/integration/tcgen05/test_tcgen05_mma_multi_warp.cpp tests/integration/tcgen05/CMakeLists.txt AGENTS.md`
- [ ] 2.8.2 `git commit -m "fix(tcgen05): add warp_id parameter to fragment_mma_f16 helper for multi-warp C slot isolation (Oracle C4)"`
- [ ] 2.8.3 验证 commit: `git show HEAD --stat` 包含 6 个文件

## 3. Phase 2 — ADR Postmortem + Archive (commit 2)

### 3.1 ADR-0016 Postmortem 追加

- [ ] 3.1.1 读 `docs/adr/0016-blackwell-only-tcgen05.md` 最末段
- [ ] 3.1.2 追加 "2026-07-11 Postmortem: Multi-warp fragment (Oracle C4 fix)" 段:
  ```markdown
  ## 2026-07-11 Postmortem: Multi-warp fragment (Oracle C4 fix)
  
  ### C4 Root Cause
  `tcgen05_fragment_mma_f16` (per `tcgen05_helpers.cpp:23`) 使用硬编码
  `c_slot = 64 + lane_id`，单 warp 假设。多 warp 时 warp 0 和 warp 1 都写
  slot 64+lane_id，导致 C slot 冲突。
  
  ### C4 Fix
  helper signature 新增 `int warp_id` 参数。Slot 计算改为
  `c_slot = warp_id * 32 + 64 + lane_id`。Caller `processTcgen05Mma` 传入
  `warp->get_warp_id()`（已存在 API，见 `tcgen05_alloc.cpp:68`）。
  单 warp 路径向后兼容（warp_id=0 等同原公式）。
  
  ### Known Limitations (debt for future)
  A/B slot 保持共享不变（per design.md D2 — minimal fix）。如果未来
  multi-warp A/B partitioning 需求出现，需新增 P2 follow-up。
  >4 warp 当前未测试覆盖（per design.md D3 — Tmem kTotalSize 容量约束）。
  ```

### 3.2 ADR commit

- [ ] 3.2.1 `git add docs/adr/0016-blackwell-only-tcgen05.md`
- [ ] 3.2.2 `git commit -m "docs(adr): ADR-0016 postmortem C4 multi-warp fragment (Oracle 2026-07-11)"`

### 3.3 Archive change

- [ ] 3.3.1 跑 `openspec archive fix-tcgen05-multi-warp-fragment --yes`
- [ ] 3.3.2 验证: `git log --all --oneline -- "openspec/changes/fix-tcgen05-multi-warp-fragment/"` 应包含 3 commits (Phase 0 artifacts + Phase 1 implementation + Phase 2 ADR postmortem + archive commit)
- [ ] 3.3.3 跑 `cd build && ctest --output-on-failure` 全量验证
- [ ] 3.3.4 跑 `./tests/ptx/test_all_ptx.sh` 验证 47/47
- [ ] 3.3.5 `git add openspec/changes/archive/` + commit "chore(openspec): archive fix-tcgen05-multi-warp-fragment"

### 3.4 强制 Postmortem Prompt (per openspec-archive-change skill)

- [ ] 3.4.1 **必须询问用户**: "是否生成 postmortem？(Yes/No/Defer)"
- [ ] 3.4.2 若 Yes: 追加 `.opencode/notes/postmortem-fix-tcgen05-multi-warp-fragment.md` + commit
- [ ] 3.4.3 若 Defer: 在 `.opencode/notes/` 留 TODO 项

### 3.5 最终验证

- [ ] 3.5.1 `cd build && ctest --output-on-failure` 全量 PASS
- [ ] 3.5.2 `./tests/ptx/test_all_ptx.sh` 全量 PASS
- [ ] 3.5.3 `git log --oneline -10` 验证 4 commits 都已落地 (artifacts + impl + ADR + archive)
- [ ] 3.5.4 `git worktree remove .worktrees/baseline-c4` 清理 baseline worktree（per lessons-learned §4 Step 4）

## 关键禁止 (per ptx-lessons-learned §3 + §6 + Checklist E)

- ❌ 不许在 sister change H1+H2 合并前 apply 本 change（signature 冲突 — proposal §Dependencies）
- ❌ 不许跳过 baseline worktree（lessons-learned §4）
- ❌ 不许 amend 已归档 change（lessons-learned §6/G Checklist G）
- ❌ 不许在 helper signature change 时忘记更新 caller（编译期强制会捕获，但需清晰 commit message）
- ❌ 不许 A/B slot 错误 per-warp offset（per design D2 — 保持 shared input）
- ❌ 不许 warp_id 校验缺失（Risk R6 — 抛 std::invalid_argument）

## Effort 估算

| Phase | Tasks | 估计时间 |
|-------|-------|----------|
| Phase 0 | baseline + branch + artifacts-first commit | 15-30 min |
| Phase 1 | helper sig + caller + 5 TC + AGENTS.md sync | 2-3 hours |
| Phase 2 | ADR postmortem + archive + postmortem prompt | 30 min |
| **总计** | | **3-4 hours** |

## 验证命令速查

```bash
# 快速验证（本 change 实施后）
cd build && ctest -R "integration_tcgen05_mma_multi_warp" -V

# 对比 baseline
cd .worktrees/baseline-c4/build && ctest -L tcgen05 --output-on-failure

# 静态验证 c_slot 公式变更
grep -n "c_slot = " src/ptxsim/instructions/tcgen05_helpers.cpp

# 静态验证 caller 传 warp_id
grep -n "tcgen05_fragment_mma_f16(tmem," src/ptxsim/instructions/tcgen05.cpp  # 期望空

# 静态验证 new TC 文件存在
ls tests/integration/tcgen05/test_tcgen05_mma_multi_warp.cpp  # 期望存在
```
