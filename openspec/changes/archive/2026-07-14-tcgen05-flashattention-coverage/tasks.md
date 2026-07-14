# Tasks: FlashAttention 测试覆盖（基于 tcgen05）

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + [specs/](specs/)
> **阻塞前置**: FU-1 (C3) + FU-2 (C1) + FU-3 (C2) + FU-4 (C4) 4 个 follow-up changes 全部 archive
> **Ref**: [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/) + [`fix-tcgen05-mma-accumulator-and-f32-storage`](../fix-tcgen05-mma-accumulator-and-f32-storage/)
> **Oracle 决策**: 2026-07-11 (`ses_0aefd09c3ffeSqBIAGdxiRBFWC`) 7 BLOCKER/IMPORTANT 缺口
> **Metis pre-implementation review**: 2026-07-11 (`ses_0b1a0cdb1ffenbhbciQ1n0x236`) per checklist H
> **强制**: ptx-lessons-learned §3(分 Phase) + §4(基线 worktree) + §6(artifacts-first) + §8(README sync) + Checklists D+E+H+I

## 0. Pre-Implementation Review + 阻塞前置验证

- [ ] 0.1 Metis pre-implementation review ✅ (2026-07-11, 涵盖 Oracle 审计 7 缺口 + Ambiguities 评估)
- [ ] 0.2 Oracle 决策建议 ✅ (2026-07-11, 5 follow-up changes 拆分 + sequencing 验证)
- [ ] 0.3 验证 FU-1..FU-4 全部 archive：
  ```bash
  for c in fix-tcgen05-commit-wait-group fix-tcgen05-idesc-parsing fix-tcgen05-ld-st-slot-routing fix-tcgen05-multi-warp-fragment; do
      git log --all --oneline -- "openspec/changes/$c/" | grep -q "chore(openspec): archive" \
          && echo "✅ $c archived" || echo "❌ $c NOT archived (blocking)"
  done
  ```
  **MUST**: 4 个 follow-up 全部 archive 才能开始 Phase 1
- [ ] 0.4 验证当前 `master` 分支 FU-1..FU-4 实施 commits 完整：
  ```bash
  # 期望每个 fix-* change 至少有: 1 artifacts commit + 1+ impl commit + 1 archive commit
  for c in fix-tcgen05-commit-wait-group fix-tcgen05-idesc-parsing fix-tcgen05-ld-st-slot-routing fix-tcgen05-multi-warp-fragment; do
      echo "=== $c ==="; git log --all --oneline -- "openspec/changes/$c/" | head -10
  done
  ```
- [ ] 0.5 跑当前 baseline 验证：`cd build && ctest -L "integration;tcgen05" --output-on-failure` 全 PASS
- [ ] 0.6 跑 `./tests/ptx/test_all_ptx.sh` 全 PASS
- [ ] 0.7 **建立基线 worktree** (per lessons-learned §4):
  ```bash
  # MUST 用最新 archive commit (FU-4 archive) 作为 baseline
  git worktree add .worktrees/baseline-fa $(git log --all --oneline | grep "archive fix-tcgen05-multi-warp" | head -1 | cut -d' ' -f1)
  cd .worktrees/baseline-fa
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # MUST 全量 build (per lessons-learned §4 陷阱 #2)
  cd build && ctest -L "integration;tcgen05" --output-on-failure  # baseline PASS 记录
  ```
- [ ] 0.8 `git checkout -b test/tcgen05-flashattention-coverage` (从 master 切新分支)
- [ ] 0.9 验证 FU-2 (C1) 实施的 idesc 寄存器读取 API 已可用（如 `warp->get_thread().read_reg_32(idesc_reg)`）

## 1. Phase 1: K=128 Accumulator 测试（commit 1 — FA-B1）

### 1.1 Helper header 编写

- [ ] 1.1.1 创建 `include/ptxsim/testing/tmem_helpers.h`
- [ ] 1.1.2 把 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` 中的 `fill_tmem_with_golden_inputs` + `require_c_slot_matches` 移入 header（namespace `ptxsim::testing::tmem`）
  - **NOTE**: 这只是函数位置迁移，不修改实现；FU-1..FU-4 apply 时可继续修改
- [ ] 1.1.3 新增 `compare_c_slot_to_reference(tmem, expected_floats, tolerance)` 函数
- [ ] 1.1.4 添加 `#ifndef PTXSIM_TESTING_TMEM_HELPERS_H` 头文件守卫

### 1.2 K=128 测试编写

- [ ] 1.2.1 创建 `tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp`
- [ ] 1.2.2 编写 `KLoopAccumulatorFixture`：
  - 持有 `SMContext(1, 32, 4096, 0)` + `WarpContext` + `Tmem&`
  - `setup_loop(n_iterations)` — 初始化 n 次 mma 的 golden inputs
- [ ] 1.2.3 编写 TC1 `"K=128 sequential accumulation produces correct sum"`（per spec FA-B1 scenario 1）：
  ```cpp
  TEST_CASE("K=128 mma accumulator reaches 128 × golden within 1e-3 tolerance", "[integration][tcgen05][mma][flashattention][k-loop]") {
      constexpr int K = 128;
      KLoopAccumulatorFixture fix(K);
      fix.run_loop();  // 调用 128 次 processTcgen05Mma(accumulate=true)
      const std::array<float, 32> expected = scale_golden(K);  // 128 × GOLDEN
      fix.tmem_helpers::compare_c_slot_to_reference(fix.tmem(), expected,
          Catch::Approx::custom().epsilon(1e-3).margin(1e-5));
  }
  ```
- [ ] 1.2.4 编写 TC2 `"K=128 with random inputs validates per-iteration independence"`（per spec FA-B1 scenario 2）：
  - 每次迭代 i 用 `fill_tmem_with_golden_inputs(i)` 提供不同 A, B
  - 验证 `C[i] = sum_{k=0..i} A[k] * B[k]`
- [ ] 1.2.5 添加注释：`// 依赖 FU-2 (C1 idesc parsing) — 此测试通过 FU-2 实施的 idesc 寄存器读取驱动 accumulate=true`

### 1.3 CMake 注册

- [ ] 1.3.1 读 `tests/integration/tcgen05/CMakeLists.txt`
- [ ] 1.3.2 添加 `add_catch_test(integration_tcgen05_mma_k_loop_128 test_tcgen05_mma_k_loop_128.cpp)`
- [ ] 1.3.3 添加标签 `set_tests_properties(integration_tcgen05_mma_k_loop_128 PROPERTIES LABELS "integration;tcgen05;flashattention")`

### 1.4 验证

- [ ] 1.4.1 `cmake --build build` 编译通过
- [ ] 1.4.2 `ctest -R "tcgen05_mma_k_loop_128" -V` PASS
- [ ] 1.4.3 `ctest -L "integration;tcgen05" --output-on-failure` 全 PASS（0 regression）
- [ ] 1.4.4 对比 baseline worktree：`ctest -L "integration;tcgen05" --output-on-failure` 在 `.worktrees/baseline-fa/build/` 也全 PASS（确认 baseline 状态）

### 1.5 Commit

- [ ] 1.5.1 `git add include/ptxsim/testing/tmem_helpers.h tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp tests/integration/tcgen05/CMakeLists.txt`
- [ ] 1.5.2 `git commit -m "test(tcgen05): add K=128 accumulator integration test (FA-B1)"`
- [ ] 1.5.3 验证 commit：`git show HEAD --stat`

## 2. Phase 2: mma → commit → wait → mma 序列测试（commit 2 — FA-B2）

### 2.1 测试编写

- [ ] 2.1.1 创建 `tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp`
- [ ] 2.1.2 编写 TC1 `"mma → commit → wait → mma accumulates 2× golden"`（per spec FA-B2 scenario 1）：
  ```cpp
  TEST_CASE("mma-commit-wait-mma sequence yields 2× golden", "[integration][tcgen05][flashattention][commit-wait]") {
      TestRig rig;
      fill_tmem_with_golden_inputs(rig.tmem());
      auto mma_instr = make_regular_mma_instr(/*accumulate=*/true);
      ptxsim::processTcgen05Mma(&rig.thread(), mma_instr);
      REQUIRE_NOTHROW(ptxsim::processTcgen05Commit(&rig.thread(), /*group_id=*/1));
      REQUIRE_NOTHROW(ptxsim::processTcgen05Wait(&rig.thread(), /*lane_id=*/0, /*group_id=*/1));
      REQUIRE(rig.cta()->tc_queue().pending_count() == 0);  // commit/wait plumbing
      ptxsim::processTcgen05Mma(&rig.thread(), mma_instr);  // 2nd accumulate
      const auto expected = scale_golden(2);
      compare_c_slot_to_reference(rig.tmem(), expected, Catch::Approx::custom().epsilon(1e-6).margin(1e-5));
  }
  ```
- [ ] 2.1.3 编写 TC2 `"mma → commit → wait → ld → st → mma preserves data across stages"`（per spec FA-B2 scenario 2）：
  - 复用 FU-3 (C2) 实施的 tmem_slot 操作数
  - 验证 `ld(slot_X) → st(slot_X)` 后 2nd mma 累加正确
- [ ] 2.1.4 添加注释：`// 强化版覆盖：此测试在 fix-tcgen05-mma-accumulator-and-f32-storage §1.4.5 B2 简化版基础上扩展`

### 2.2 验证

- [ ] 2.2.1 `cmake --build build && cd build && ctest -R "tcgen05_mma_commit_wait_sequence" -V` PASS
- [ ] 2.2.2 全量 `ctest -L "integration;tcgen05;flashattention"` 全 PASS（Phase 1 + Phase 2）

### 2.3 Commit

- [ ] 2.3.1 `git add tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp`
- [ ] 2.3.2 `git commit -m "test(tcgen05): add mma→commit→wait→mma sequence integration test (FA-B2)"`

## 3. Phase 3: cp → mma 数据流测试（commit 3 — FA-B4）

### 3.1 测试编写

- [ ] 3.1.1 创建 `tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp`
- [ ] 3.1.2 编写 TC1 `"cp writes to slot that mma reads from"`（per spec FA-B4 scenario 1）：
  ```cpp
  TEST_CASE("cp loads A matrix to slot that mma reads from", "[integration][tcgen05][flashattention][cp-mma]") {
      TestRig rig;
      // cp 加载已知 A 矩阵到 tmem slot 0 (per FU-3 C2 tmem_slot)
      fill_tmem_with_golden_A_matrix(rig.tmem(), /*slot=*/0);
      auto cp_instr = make_cp_instr(/*src=*/smem_addr, /*dst_tmem_slot=*/0);
      ptxsim::processTcgen05Cp(&rig.thread(), cp_instr);
      // mma 读 slot 0 的 A (per tcgen05_helpers.cpp:21 a_slot = lane_id * 2)
      auto mma_instr = make_regular_mma_instr();
      ptxsim::processTcgen05Mma(&rig.thread(), mma_instr);
      // 数值断言（不是"至少一个变化"）
      require_c_slot_matches(rig.tmem(), GOLDEN_MMA_F16_F16_F32, "after cp→mma");
  }
  ```
- [ ] 3.1.3 编写 TC2 `"cp with multiple slots validates per-slot data integrity"`（per spec FA-B4 scenario 2）

### 3.2 验证 + Commit

- [ ] 3.2.1 `ctest -R "tcgen05_mma_cp_data_flow" -V` PASS
- [ ] 3.2.2 `git commit -m "test(tcgen05): add cp→mma data flow integration test (FA-B4)"`

## 4. Phase 4: 2-warp C slot 隔离测试（commit 4 — FA-B5）

### 4.1 测试编写

- [ ] 4.1.1 创建 `tests/integration/tcgen05/test_tcgen05_multi_warp_isolation.cpp`
- [ ] 4.1.2 编写 TC1 `"warp 0 and warp 1 write disjoint C slot ranges"`（per spec FA-B5 scenario 1）：
  ```cpp
  TEST_CASE("2-warp mma isolates C slots at [64..95] vs [96..127]", "[integration][tcgen05][flashattention][multi-warp]") {
      SMContext sm(/*num_warps=*/2, 128, 4096, 0);
      WarpContext* w0 = sm.cta(0)->warp(0);
      WarpContext* w1 = sm.cta(0)->warp(1);
      // warp 0 mma
      fill_tmem_with_golden_inputs(w0->tmem());
      ptxsim::processTcgen05Mma(&w0->thread(), make_regular_mma_instr());
      // warp 1 mma (相同输入)
      fill_tmem_with_golden_inputs(w1->tmem());
      ptxsim::processTcgen05Mma(&w1->thread(), make_regular_mma_instr());
      // 验证两 warp C slot 在不同 range
      // warp 0: c_slot = 0 * 32 + 64 + lane_id = [64..95]
      // warp 1: c_slot = 1 * 32 + 64 + lane_id = [96..127]
      for (int lane = 0; lane < 32; ++lane) {
          REQUIRE(w0->tmem().read_verify(64 + lane, /*expects=*/golden));
          REQUIRE(w1->tmem().read_verify(96 + lane, /*expects=*/golden));
      }
  }
  ```
- [ ] 4.1.3 编写 TC2 `"simultaneous 2-warp mma produces independent outputs"`（per spec FA-B5 scenario 2）

### 4.2 验证 + Commit

- [ ] 4.2.1 `ctest -R "tcgen05_multi_warp_isolation" -V` PASS
- [ ] 4.2.2 `git commit -m "test(tcgen05): add 2-warp C slot isolation integration test (FA-B5)"`

## 5. Phase 5: E2E FlashAttention Mini-Kernel + README Sync（commit 5 — FA-E2E）

### 5.1 E2E Kernel 编写

- [ ] 5.1.1 创建 `tests/e2e/kernel/test_flashattention_mini.cu`
- [ ] 5.1.2 编写 `__global__ void kernel_fa_mini(...)`：
  ```cuda
  // K=4 blocks, head_dim=64, block_size=32
  // Phase 1: Q @ K^T → S (累加 mma, K-loop)
  //   for k_block = 0..3:
  //     tcgen05.ld K_tile to tmem slot 0
  //     tcgen05.mma Q @ K^T, accumulate=True → S_k
  //     tcgen05.commit group=QK
  //     tcgen05.wait group=QK
  // Phase 2: softmax(S) → P (纯 CUDA C++ fallback)
  // Phase 3: P @ V → O (累加 mma, K-loop)
  //   for v_block = 0..3:
  //     tcgen05.ld V_tile to tmem slot 1
  //     tcgen05.mma P @ V, accumulate=True → O_v
  //     tcgen05.commit group=PV
  //     tcgen05.wait group=PV
  // Output O[i] = sum_{v=0..3} P[i] * V[v]
  ```
- [ ] 5.1.3 编写 host 验证函数：CUDA fallback (纯 C++ Q@K^T → softmax → @V) 作为 reference，对比 O 相对误差 < 1e-3
- [ ] 5.1.4 添加注释：`// E2E kernel 走 dispatcher 真实路径；如 ptxas 不支持 sm_100 tcgen05，降级为 Priority 3 fallback`

### 5.2 CMake 注册

- [ ] 5.2.1 读 `tests/e2e/kernel/CMakeLists.txt`
- [ ] 5.2.2 添加 `add_catch_test(e2e_flashattention_mini kernel/test_flashattention_mini.cu)`
- [ ] 5.2.3 标签：`set_tests_properties(e2e_flashattention_mini PROPERTIES LABELS "e2e;flashattention")`

### 5.3 README 同步（per lessons-learned §8 + Checklist I）

- [ ] 5.3.1 读 `README.md` "已实现功能"章节（line 30-34）
- [ ] 5.3.2 在 "Blackwell tcgen05" 条目后追加：
  ```
  - **FlashAttention mini-kernel** (QK^T → softmax → @V，commit `tcgen05-flashattention-coverage` archive) — 端到端 FA 数据流验证
  ```
- [ ] 5.3.3 跑 lessons-learned Checklist I 验证：
  ```bash
  grep -n "stub\|TODO\|FIXME\|不实现\|未完成" README.md  # 应为空
  grep -nE "[0-9]+%|硬编码" README.md  # 应替换或为空
  ```

### 5.4 验证 + Commit

- [ ] 5.4.1 `ctest -R "e2e_flashattention_mini" -V` PASS（即使 Priority 3 fallback 也要走通）
- [ ] 5.4.2 `ctest -L "e2e;flashattention" --output-on-failure` PASS
- [ ] 5.4.3 `ctest -L "integration;tcgen05;flashattention" --output-on-failure` PASS（Phase 1-4 全）
- [ ] 5.4.4 `git add tests/e2e/kernel/test_flashattention_mini.cu tests/e2e/kernel/CMakeLists.txt README.md`
- [ ] 5.4.5 `git commit -m "test(e2e): add FlashAttention mini-kernel + README sync (FA-E2E)"`

## 6. Phase 6: Artifacts + ADR Postmortem + Archive（commit 6, per lessons-learned §6 Checklist G）

### 6.1 Artifacts git-tracked（artifacts FIRST per lessons-learned §6）

- [ ] 6.1.1 `git status openspec/changes/tcgen05-flashattention-coverage/` 验证 4 个 md + specs/ 在 working tree
- [ ] 6.1.2 `git add openspec/changes/tcgen05-flashattention-coverage/` + 4 个 md + specs/
- [ ] 6.1.3 `git commit -m "docs(openspec): tcgen05-flashattention-coverage artifacts (Oracle 2026-07-11 FA audit, Metis pre-impl)"`
- [ ] 6.1.4 验证：`git ls-files openspec/changes/tcgen05-flashattention-coverage/` 不为空

### 6.2 ADR-0016 Postmortem 追加

- [ ] 6.2.1 读 `docs/adr/0016-blackwell-only-tcgen05.md` 找到最末段
- [ ] 6.2.2 追加 "2026-07-XX Postmortem: FlashAttention coverage" 段：
  ```markdown
  ## 2026-07-XX Postmortem: FlashAttention coverage

  ### Coverage Gap Root Cause
  Oracle 2026-07-11 audit (ses_0aefd09c3ffeSqBIAGdxiRBFWC) identified 7 BLOCKER/IMPORTANT
  test coverage gaps preventing FlashAttention QK^T→softmax→PV end-to-end validation:
  - B1: No K=128+ accumulator stability test
  - B2: No mma→commit→wait→mma sequence test
  - B4: cp→mma data flow only "at least one element changed" assertion
  - B5: All tests single-warp (no 2-warp isolation)
  - B6: No mma(f32)→ld→st→mma persistence test
  - B7: Catch::Approx default epsilon (1.19e-5) too loose for K=128 drift
  - D-E2E: No FlashAttention E2E kernel

  ### Coverage Fix
  5 new test files + 1 helper header in tcgen05-flashattention-coverage change:
  - tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp (B1)
  - tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp (B2)
  - tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp (B4)
  - tests/integration/tcgen05/test_tcgen05_multi_warp_isolation.cpp (B5)
  - tests/e2e/kernel/test_flashattention_mini.cu (D-E2E)
  - include/ptxsim/testing/tmem_helpers.h (helper)

  ### Prerequisites (out-of-scope, archived separately)
  4 fix-* changes completed handler/visitor/slot routing fixes:
  - fix-tcgen05-commit-wait-group (C3) — FU-1
  - fix-tcgen05-idesc-parsing (C1) — FU-2
  - fix-tcgen05-ld-st-slot-routing (C2) — FU-3
  - fix-tcgen05-multi-warp-fragment (C4) — FU-4
  ```

### 6.3 ADR commit

- [ ] 6.3.1 `git add docs/adr/0016-blackwell-only-tcgen05.md`
- [ ] 6.3.2 `git commit -m "docs(adr): ADR-0016 postmortem FlashAttention coverage (Oracle 2026-07-11)"`

### 6.4 Archive change

- [ ] 6.4.1 `openspec archive tcgen05-flashattention-coverage --yes`
- [ ] 6.4.2 验证：`git log --all --oneline -- "openspec/changes/tcgen05-flashattention-coverage/"` 应包含 archive commit
- [ ] 6.4.3 跑 `cd build && ctest --output-on-failure` 全量验证
- [ ] 6.4.4 跑 `./tests/ptx/test_all_ptx.sh` 全量验证
- [ ] 6.4.5 `git add openspec/changes/archive/` + commit "chore(openspec): archive tcgen05-flashattention-coverage"

### 6.5 强制 Postmortem Prompt（per openspec-archive-change skill）

- [ ] 6.5.1 **必须询问用户**: "是否生成 postmortem？(Yes/No/Defer)"
- [ ] 6.5.2 若 Yes: 追加 `.opencode/notes/postmortem-tcgen05-flashattention-coverage.md` + commit
- [ ] 6.5.3 若 Defer: 在 `.opencode/notes/` 留 TODO 项

### 6.6 最终验证 + 清理

- [ ] 6.6.1 `cd build && ctest --output-on-failure` 全量 PASS
- [ ] 6.6.2 `./tests/ptx/test_all_ptx.sh` 全量 PASS
- [ ] 6.6.3 `git log --oneline -10` 验证 6 commits 都已落地
- [ ] 6.6.4 `git worktree remove .worktrees/baseline-fa` 清理 baseline worktree（per lessons-learned §4 Step 4）

## 关键禁止（per ptx-lessons-learned）

- ❌ 不许在 FU-1..FU-4 archive 前开始 Phase 1（task 0.3 强制 gate）
- ❌ 不许跳过 baseline worktree（lessons-learned §4）
- ❌ 不许 1 commit 同时改多个测试文件（lessons-learned §3 + D6 决策）
- ❌ 不许 amend 已归档的 fix-* changes（lessons-learned §6/G）
- ❌ 不许在 helper header 修改后忘记更新现有测试文件复用（5 文件共用 tmem_helpers.h）
- ❌ 不许 README sync 缺失（lessons-learned §8 + Checklist I）
- ❌ 不许 archive 前未跑 lessons-learned §8 grep 验证

## Effort 估算

| Phase | Tasks | 估计时间 |
|-------|-------|---------|
| Phase 0 | FU-* 验证 + baseline worktree + 分支 | 0.5-1h |
| Phase 1 | K=128 test + tmem_helpers.h | 2-3h |
| Phase 2 | commit/wait sequence test | 1-2h |
| Phase 3 | cp→mma data flow test | 1-2h |
| Phase 4 | 2-warp isolation test | 1-2h |
| Phase 5 | E2E kernel + README sync | 3-4h |
| Phase 6 | Archive + ADR postmortem | 30min |
| **总计** | | **9-14h** |