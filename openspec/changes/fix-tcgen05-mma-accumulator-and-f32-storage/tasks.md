# Tasks: Fix tcgen05.mma Fragment Helper — Accumulator + f32 Storage

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + [spec.md](specs/fix-tcgen05-mma-accumulator-and-f32-storage/spec.md)
> **Ref** (不能 amend 的已归档 change): [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/)
> **范围**: 3 atomic commits (Phase 1: H1; Phase 2: H2; Phase 3: Archive)
> **Oracle 决策**: 2026-07-10 (`ses_0b3791d78ffewb52428kJJ2Irz`) H1+H2 HIGH confidence
> **Metis pre-implementation review**: 2026-07-10 (`ses_0b1a0cdb1ffenbhbciQ1n0x236`) 3 MUST-RESOLVE 全部采纳
> **强制**: ptx-lessons-learned §3(分 Phase) + §4(基线 worktree) + §6(artifacts-first) + §7(Pre-impl Review)

## 0. Pre-Implementation Review

- [x] 0.1 Metis pre-implementation review ✅ (2026-07-10, 3 MUST-RESOLVE 全部采纳)
- [x] 0.2 Oracle 决策建议 ✅ (2026-07-10, H1+H2 HIGH confidence)
- [x] 0.3 验证 step 1 commit `d3be589` 已 archive（persistence test 提供 H5 验证基础）
- [x] 0.4 验证 `archive/2026-07-10-implement-tcgen05-handlers-extended/` 已 archive（11/11 handler 已实施）
- [ ] 0.5 跑 `cd build && ctest -R "tcgen05" --output-on-failure` 确认 baseline
- [ ] 0.6 跑 `./tests/ptx/test_all_ptx.sh` 确认 12 fixtures PASS
- [ ] 0.7 **建立基线 worktree** (per ptx-lessons-learned §4):
  ```bash
  # Step 1: 建立 baseline (commit `d3be589` 包含 step 1 persistence test)
  git worktree add .worktrees/baseline-h1h2 d3be589
  cd .worktrees/baseline-h1h2
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # 必须全量 build
  cd build && ctest -L tcgen05 --output-on-failure
  ```
- [ ] 0.8 `git checkout -b fix/tcgen05-mma-accumulator-and-f32-storage`

## 1. Phase 1: H1 — Accumulator 支持（commit 1）

### 1.1 Helper signature 修改

- [ ] 1.1.1 读 `include/ptxsim/instructions/tcgen05_helpers.h:51` 确认当前签名
- [ ] 1.1.2 修改签名为 `void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false);`
- [ ] 1.1.3 添加 doc comment（accumulate 语义 + single-warp execution 要求）

### 1.2 Helper body 修改（accumulate 路径）

- [ ] 1.2.1 读 `src/ptxsim/instructions/tcgen05_helpers.cpp:42-58` 当前实现
- [ ] 1.2.2 添加通用 `load_c_slot` 模板 helper（per Oracle Q4，在 `tcgen05_helpers.cpp` 顶部 `namespace ptxsim` 内）:
  ```cpp
  // internal helper for accumulate pre-load (per Oracle Q4 analysis)
  template <typename T>
  static void load_c_slot(Tmem& tmem, size_t c_slot, T* c_frag, size_t count) {
      alignas(T) std::array<uint8_t, Tmem::kSlotSize> buf{};
      tmem.read(c_slot, buf.data(), Tmem::kSlotSize);
      std::memcpy(c_frag, buf.data(), count * sizeof(T));
  }
  ```
- [ ] 1.2.3 在 `c_frag{}` 初始化后添加 accumulate 分支（使用 `load_c_slot<uint16_t>`，f16 storage）:
  ```cpp
  if (accumulate) {
      load_c_slot<uint16_t>(tmem, c_slot, c_frag.data(), ROWS * COLS_B);
      // c_frag 现在包含现有 C slot 的前 64 字节 f16 bits（后 64 字节零）
  }
  ```
- [ ] 1.2.4 修改累加循环（行 43-52）添加 accumulate 时 sum += existing:
  ```cpp
  for (int i = 0; i < ROWS; ++i) {
      for (int j = 0; j < COLS_B; ++j) {
          float sum = 0.0f;
          if (accumulate) {
              sum += f16_to_f32(c_frag[i * COLS_B + j]);  // 现有值（f16→f32）
          }
          for (int k = 0; k < COLS_A; ++k) {
              sum += a_flat[i * COLS_A + k] * b_flat[k * COLS_B + j];
          }
          c_frag[i * COLS_B + j] = f32_to_f16(sum);  // 本 Phase 保持 f16 storage
      }
  }
  ```

### 1.3 调用点更新

- [ ] 1.3.1 读 `src/ptxsim/instructions/tcgen05.cpp:383` 当前调用
- [ ] 1.3.2 改为 `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false);`（显式 overwrite）

### 1.4 Persistence T1 反转 + 新增 T1_overwrite

- [ ] 1.4.1 读 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:184-203` 当前 T1
- [ ] 1.4.2 T1 反转:
  - TC 名: `"processTcgen05Mma called twice with identical A,B accumulates into C (2nd mma yields 2× golden)"`
  - 断言: 2nd mma 后 C 等于 `2 * GOLDEN_MMA_F16_F16_F32`
- [ ] 1.4.3 新增 `T1_overwrite` TC（同文件，在 T1 之后）:
  ```cpp
  TEST_CASE("processTcgen05Mma with accumulate=false leaves C unchanged (overwrite preserved)") {
      TestRig rig;
      fill_tmem_with_golden_inputs(rig.tmem());
      ptxsim::tcgen05_fragment_mma_f16(rig.tmem(), /*accumulate=*/false);
      ptxsim::tcgen05_fragment_mma_f16(rig.tmem(), /*accumulate=*/false);
      // 断言 C == GOLDEN_MMA_F16_F16_F32（overwrite 保留）
  }
  ```
- [ ] 1.4.4 在 T1 注释中明确说明：H1 实施后 overwrite 行为需显式传 `accumulate=false`

### 1.4bis 加固测试（per Oracle 审计 + Metis pre-impl review — 测试-only，零实现改动）

- [ ] 1.4.5 **B2 hardening**: 新增 `tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp`
  - 构造 4 指令序列：mma(1st) → commit(group_id=1) → wait(group_id=1) → mma(accumulate, 2nd)
  - 调 `ptxsim::testing::step_warp` 驱动（per `tests/integration/divergence/test_divergence_sync_convergence.cpp` 模式）
  - 断言 2nd mma 后 C slot 等于 `2 × GOLDEN_MMA_F16_F16_F32`（验证 H1 累加经 commit/wait 路径仍生效）
  - 断言 `cta->tc_queue().pending_count() == 0`（验证 commit/wait plumbing）
  - **零实现改动**：仅测试现有 handler 路径（`processTcgen05Commit` 硬编码 `group_id=1` + `processTcgen05Wait` 硬编码 `lane_id=0, group_id=1`）
  - **风险**：纯测试，Oracle Q3 验证不构成 scope creep（lessons-learned §3 仅限行为变更）

- [ ] 1.4.6 **T1_k_loop_4 hardening**: 在 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` 同文件新增
  - 构造 `processTcgen05Mma(instr)` × 4 调用（无中间 cp/ld）
  - 断言 C slot 等于 `4 × GOLDEN_MMA_F16_F16_F32`（即使不能做 128 次，4 次可捕获明显累加器 bug）
  - 注释：K=128 完整覆盖留待 `tcgen05-flashattention-coverage` follow-up
  - 复用 `fill_tmem_with_golden_inputs` + `require_c_slot_matches` 助手

### 1.5 验证

- [ ] 1.5.1 `cmake --build build` 编译通过
- [ ] 1.5.2 `ctest -R "tcgen05" --output-on-failure` 全 PASS（除 T1 反转的预期 fail 之外）
- [ ] 1.5.3 `./scripts/sanity.sh --tier 2 --tier 8` PASS（包含 tcgen05 supporting tests per step 1 commit）
- [ ] 1.5.4 **对比 baseline worktree** (per ptx-lessons-learned §4):
  ```bash
  # baseline 在 .worktrees/baseline-h1h2/build/
  cd .worktrees/baseline-h1h2/build && ctest -L tcgen05 --output-on-failure
  # 预期：baseline 22/22 PASS，main 22/22 PASS（含反转 T1 + T1_overwrite）
  ```
- [ ] 1.5.5 **失败处理**: 任何已有测试回归 → 立即 revert 该 commit（per lessons-learned §3）

### 1.6 Commit

- [ ] 1.6.1 `git add include/ptxsim/instructions/tcgen05_helpers.h src/ptxsim/instructions/tcgen05_helpers.cpp src/ptxsim/instructions/tcgen05.cpp tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp`
- [ ] 1.6.2 `git commit -m "fix(tcgen05): add accumulate parameter to fragment_mma_f16 helper (Oracle H1)"`
- [ ] 1.6.3 验证 commit: `git show HEAD --stat`

## 2. Phase 2: H2 — f32 Output Storage（commit 2）

### 2.1 Helper body 修改（f32 storage）

- [ ] 2.1.1 读当前 `tcgen05_helpers.cpp`（Phase 1 后状态）
- [ ] 2.1.2 改 `c_frag` 类型（行 42）:
  ```cpp
  // BEFORE: std::array<uint16_t, ROWS * COLS_B> c_frag{};
  // AFTER:  std::array<float, ROWS * COLS_B> c_frag{};
  ```
- [ ] 2.1.3 修改 accumulate 预加载分支（使用 `load_c_slot<float>`，f32 storage）:
  ```cpp
  if (accumulate) {
      load_c_slot<float>(tmem, c_slot, c_frag.data(), ROWS * COLS_B);
      // c_frag 现在包含现有 C slot 的全部 128 字节 f32 values
  }
  ```
- [ ] 2.1.4 改累加循环（行 51）删除 `f32_to_f16`:
  ```cpp
  // BEFORE: sum += f16_to_f32(c_frag[i * COLS_B + j]);
  // AFTER:  sum += c_frag[i * COLS_B + j];  // 直接 float 累加
  
  // BEFORE: c_frag[i * COLS_B + j] = f32_to_f16(sum);
  // AFTER:  c_frag[i * COLS_B + j] = sum;  // 直接写 float
  ```
- [ ] 2.1.5 改 memcpy size（行 55）:
  ```cpp
  // BEFORE: std::memcpy(c_buf.data(), c_frag.data(), c_frag.size() * sizeof(uint16_t));
  // AFTER:  std::memcpy(c_buf.data(), c_frag.data(), c_frag.size() * sizeof(float));  // 32*4 = 128 bytes
  ```

- [ ] 2.1.6 **f32_to_f16 移除运行时断言** (per Oracle test gap B5 + Metis test gap §3):
  ```cpp
  // 在 tcgen05_fragment_mma_f16 函数体顶部加注释:
  // Post-H2: f32_to_f16 must NOT appear in this function body.
  // Future refactors that re-introduce f32→f16 conversion violate PTX ISA §9.7.16.
  
  // 在 tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp 新增 TC:
  TEST_CASE("tcgen05_fragment_mma_f16 helper has no f32_to_f16 in body") {
      // 静态验证: 编译期字符串检查
      const std::string source = read_file("src/ptxsim/instructions/tcgen05_helpers.cpp");
      const auto body_start = source.find("tcgen05_fragment_mma_f16");
      REQUIRE(body_start != std::string::npos);
      REQUIRE(source.find("f32_to_f16", body_start) == std::string::npos);
  }
  ```

### 2.2 Helper header doc 更新

- [ ] 2.2.1 读 `include/ptxsim/instructions/tcgen05_helpers.h` 当前 doc
- [ ] 2.2.2 在 Layout 段添加:
  ```
  // C output: 32 f32 elements per lane (128 bytes, fills slot completely).
  // Storage format changed from f16 in fix-tcgen05-mma-accumulator-and-f32-storage
  // Phase 2 commit (Oracle H2 fix per PTX ISA §9.7.16).
  ```

### 2.3 Readback 机械修改（per Metis C2 mitigation + Oracle Q3 选项 ii）

- [ ] 2.3.1 跑 `grep -rn "c_buf\[idx \* 2\]\|f16_to_f32" tests/integration/tcgen05/` 列出所有 readback 点
- [ ] 2.3.2 改 `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp:156`（per Oracle Q3 推荐模式）:
  ```cpp
  // BEFORE:
  const uint16_t actual_bits = static_cast<uint16_t>(
      c_buf[idx * 2] | (c_buf[idx * 2 + 1] << 8));
  const float actual = f16_to_f32(actual_bits);
  
  // AFTER (per Oracle Q3 推荐: alignas(16) float[32] + 单次 memcpy):
  alignas(16) float c_arr[32];
  std::memcpy(c_arr, c_buf.data(), sizeof(c_arr));
  const float actual = c_arr[idx];
  ```
- [ ] 2.3.3 改 `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp:192`（同样模式）
- [ ] 2.3.4 改 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:167`（同样模式）
- [ ] 2.3.5 改 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:288`（同样模式）
- [ ] 2.3.6 验证 `grep -rn "c_buf\[idx \* 2\]" tests/` 输出为空（确认无遗漏）
- [ ] 2.3.7 验证 `grep -rn "f16_to_f32" tests/integration/tcgen05/` 输出为空（确认所有 f16 readback 已替换）

- [ ] 2.3.8 **B7 hardening: 收紧数值公差** (per Oracle Section E 审计):
  - 当前: `test_tcgen05_mma_golden.cpp:64` + `test_tcgen05_mma_persistence.cpp:172` + `test_tcgen05_mma_ws.cpp:160` 使用默认 `Catch::Approx` (epsilon ≈ 1.19e-5)
  - 改为: `.epsilon(1e-6)` — 针对 f32 直接读取（无 f16→f32 转换误差），K=128 累加后可检测 ULP 级漂移
  - 例:
    ```cpp
    // BEFORE: REQUIRE(actual == Catch::Approx(expected[idx]));
    // AFTER:  REQUIRE(actual == Catch::Approx(expected[idx]).epsilon(1e-6));
    ```
  - **风险**: 1.0..32.0 golden 数值正好是 f16/f32 可精确表示的值，收紧不会引入 false negative
  - **Forward compat**: K=128 FlashAttention 测试需要此公差作为基础

### 2.4 Golden header 注释更新

- [ ] 2.4.1 读 `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h:6-7`
- [ ] 2.4.2 更新注释:
  ```cpp
  // Layout: 8 rows × 4 cols = 32 f32 elements (per-lane fragment output).
  // Storage format: f32 (per PTX ISA §9.7.16, mma output dtype is f32).
  // Previously stored as f16 with f16→f32 readback; storage changed in
  // fix-tcgen05-mma-accumulator-and-f32-storage Phase 2 commit (Oracle H2).
  ```

### 2.5 验证

- [ ] 2.5.1 `cmake --build build` 编译通过
- [ ] 2.5.2 `ctest -R "tcgen05" --output-on-failure` 全 PASS
- [ ] 2.5.3 `./scripts/sanity.sh --tier 2 --tier 8` PASS
- [ ] 2.5.4 对比 baseline: 22/22 tcgen05-tagged 测试 PASS（数值不变）
- [ ] 2.5.5 失败处理: 任何已有测试回归 → 立即 revert 该 commit

### 2.6 Commit

- [ ] 2.6.1 `git add src/ptxsim/instructions/tcgen05_helpers.cpp include/ptxsim/instructions/tcgen05_helpers.h tests/integration/tcgen05/test_tcgen05_mma_ws.cpp tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`
- [ ] 2.6.2 `git commit -m "fix(tcgen05): store mma C output as f32 per PTX ISA §9.7.16 (Oracle H2)"`
- [ ] 2.6.3 验证 commit: `git show HEAD --stat`

## 3. Phase 3: Archive + ADR Postmortem（commit 3, per lessons-learned §6 Checklist G）

### 3.1 Artifacts git-tracked（artifacts FIRST per lessons-learned §6）

- [ ] 3.1.1 `git status openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/` 验证 4 个 md + specs/ 在 working tree
- [ ] 3.1.2 `git add openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/` + 4 个 md 文件
- [ ] 3.1.3 `git commit -m "docs(openspec): fix-tcgen05-mma-accumulator-and-f32-storage artifacts (Oracle H1+H2, Metis pre-impl review)"`
- [ ] 3.1.4 验证: `git ls-files openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/` 不应为空

### 3.2 ADR-0016 Postmortem 追加

- [ ] 3.2.1 读 `docs/adr/0016-blackwell-only-tcgen05.md` 找到最末段
- [ ] 3.2.2 追加 "2026-07-11 Postmortem: H1+H2 fix" 段（per design.md D7）:
  ```markdown
  ## 2026-07-11 Postmortem: H1+H2 fix
  
  ### H1 Root Cause
  [per design.md D7 内容]
  
  ### H1 Fix
  [per design.md D7 内容]
  
  ### H2 Root Cause
  [per design.md D7 内容]
  
  ### H2 Fix
  [per design.md D7 内容]
  
  ### Known Semantic Gap (debt for future)
  [per design.md D7 内容]
  ```

### 3.3 ADR commit

- [ ] 3.3.1 `git add docs/adr/0016-blackwell-only-tcgen05.md`
- [ ] 3.3.2 `git commit -m "docs(adr): ADR-0016 postmortem H1+H2 (Oracle 2026-07-10 FlashAttention readiness)"`

### 3.4 Archive change

- [ ] 3.4.1 跑 `openspec archive fix-tcgen05-mma-accumulator-and-f32-storage --yes`
- [ ] 3.4.2 验证: `git log --all --oneline -- "openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/"` 应包含 archive commit
- [ ] 3.4.3 跑 `cd build && ctest --output-on-failure` 全量验证
- [ ] 3.4.4 跑 `./tests/ptx/test_all_ptx.sh` 验证
- [ ] 3.4.5 `git add openspec/changes/archive/` + commit "chore(openspec): archive fix-tcgen05-mma-accumulator-and-f32-storage"

### 3.5 强制 Postmortem Prompt（per openspec-archive-change skill）

- [ ] 3.5.1 **必须询问用户**: "是否生成 postmortem？(Yes/No/Defer)"
- [ ] 3.5.2 若 Yes: 追加 `.opencode/notes/postmortem-fix-tcgen05-mma-accumulator-and-f32-storage.md` + commit
- [ ] 3.5.3 若 Defer: 在 `.opencode/notes/` 留 TODO 项

### 3.6 最终验证

- [ ] 3.6.1 `cd build && ctest --output-on-failure` 全量 PASS
- [ ] 3.6.2 `./tests/ptx/test_all_ptx.sh` 全量 PASS
- [ ] 3.6.3 `git log --oneline -10` 验证 3 commits 都已落地
- [ ] 3.6.4 `git worktree remove .worktrees/baseline-h1h2` 清理 baseline worktree

## 关键禁止（per ptx-lessons-learned §3）

- ❌ 不许 1 commit 同时改 helper signature + storage format（Metis C5）
- ❌ 不许跳过 baseline worktree（lessons-learned §4）
- ❌ 不许 amend 已归档的 `archive/2026-07-10-implement-tcgen05-handlers-extended`（lessons-learned §6/G）
- ❌ 不许在 helper signature change 时忘记更新 `processTcgen05Mma` 调用点
- ❌ 不许 H2 commit 漏改 readback（Metis C2 — 静默返回错误值）
- ❌ 不许 T1 反转被误判为 regression（Metis C1 — OpenSpec proposal 已说明）

## Effort 估算

| Phase | Tasks | 估计时间 |
|-------|-------|---------|
| Phase 1 | H1 helper + 调用点 + T1 反转 + T1_overwrite | 2-4h |
| Phase 2 | H2 helper + readback × 4 + golden 注释 | 1-2h |
| Phase 3 | Archive + ADR postmortem | 30min |
| **总计** | | **4-6h** |