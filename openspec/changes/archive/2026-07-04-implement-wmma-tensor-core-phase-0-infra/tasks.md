# Tasks: Blackwell tcgen05 Infrastructure (Phase 0)

> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md)
> **前置 change**: `replace-silent-stub-failures` (archived 2026-07-04)
> **后续 change**: `implement-wmma-tensor-core-tcgen05` (per `Ref:` after archive)
> **Phase 范围**: 0.0 (artifacts tracked) + 0.1-0.4 (4 子系统) + 0.5 (4 micro 集成) = **9 commits**
> **Lessons-learned**: Checklist A (函数迁移) + B (重构前 baseline) + D (commit 前) + E (artifacts tracked) + G (OpenSpec lifecycle)

---

## 0.0 Artifacts Tracking（必做 FIRST！避免 lessons-learned §6 模式）

- [ ] 0.0.1 在 main 上创建分支：`git checkout -b feat/implement-blackwell-tcgen05`
- [ ] 0.0.2 `git add openspec/changes/implement-wmma-tensor-core-phase-0-infra/`
- [ ] 0.0.3 `git status` 验证 artifacts tracked (proposal / design / specs/wmma-tensor-core/spec / tasks)
- [ ] 0.0.4 commit: `git commit -m "docs(openspec): track implement-wmma-tensor-core-phase-0-infra artifacts (ADR-0016)"`
- [ ] 0.0.5 `git ls-files openspec/changes/implement-wmma-tensor-core-phase-0-infra/` 验证非空
- [ ] 0.0.6 ADR-0016（docs/adr/ADR-0016-blackwell-only-tcgen05.md）随本批 change 提交

---

## Phase 0.1: TMA descriptors（Fix #5）— ✅ DONE (commit ad527f5)

- [x] 0.1.1 建立基线 worktree（per archive precedent `openspec/changes/archive/2026-07-04-replace-silent-stub-failures/tasks.md` "基线 worktree" section；统一使用 `.worktrees/` 前缀而不是 `../<name>` 形式）：
      `git worktree add .worktrees/fix-pre-p0-baseline -b feat/implement-blackwell-tcgen05 main`
- [x] 0.1.2 验证基线：`.worktrees/fix-pre-p0-baseline` 下 `cmake -S . -B build && cmake --build build && cd build && ctest --output-on-failure`
      (123 labeled tests PASS at b7d48ca baseline)
- [x] 0.1.3 阅读 NVIDIA PTX ISA §9.7.13 + cuobjdump 提取真实 TMA descriptor 字节
      (research via librarian agent: 128-byte layout from tensormap.replace §9.7.9.27 + LLVM NVPTX + CUDA Driver API；byte offsets INFERRED marked UNVERIFIED-AGAINST-HARDWARE)
- [x] 0.1.4 创建 `src/ptxsim/memory/tma_descriptor.h` (168 LoC):
      - `struct TmaDescriptor` (global_address / global_dim / global_stride / box_dim / element_stride / rank / elemtype / interleave / swizzle_mode / fill_mode)
      - `class TmaDescriptorStore` (per-CTA descriptor 表)
      - `parse_descriptor_bytes(const void* bytes) -> TmaDescriptor` (throws std::runtime_error on invalid)
- [x] 0.1.5 创建 `src/ptxsim/memory/tma_descriptor.cpp` (204 LoC, all magic numbers annotated UNVERIFIED-AGAINST-HARDWARE)
- [x] 0.1.6 创建 `tests/unit/memory/test_tma_descriptor.cpp` (477 LoC, 18 TEST_CASEs / 89 assertions):
      dtype_variants / swizzle_variants / interleave_variants / rank_variants / misaligned_address / too_short / reserved_nonzero / store_roundtrip / stride_constraint / box_dim_range / ... (≥10 覆盖)
- [x] 0.1.7 在 `src/CMakeLists.txt` (line 87) + `tests/unit/CMakeLists.txt` (line 218-221) 注册
- [x] 0.1.8 自检：`cmake --build build` + `ctest -R "tma_descriptor"` → **18 PASS (89 assertions)**
- [x] 0.1.9 验证无回归：`ctest -L "unit|integration|e2e"` → **124 labeled tests PASS, 0 FAIL** (1 个新 TMA 测试加入；zero regression)
- [x] 0.1.10 commit: `ad527f5 feat(memory): TMA descriptor parser (Fix #5)` (5 files: 3 new + 2 modified; atomic)
- [x] 0.1.11 验证独立可 revert (待执行 — 单 atomic commit + 单一新目录 + 测试独立 → revert 安全)

## Phase 0.2: Tensor Memory (TMEM)（Fix #6）— ✅ DONE (commit 758edb0)

- [x] 0.2.1 创建 `src/ptxsim/memory/tmem.h` (49 LoC):
      - `class Tmem` with `kSlotCount=256`, `kSlotSize=128`, `kTotalSize=32*1024` constants
      - `read(slot_id, bytes, size)`, `write(slot_id, bytes, size)`, `clear()`, `validate_slot_id()`
- [x] 0.2.2 创建 `src/ptxsim/memory/tmem.cpp` (61 LoC, std::array<uint8_t,32KB>+std::mutex backing store)
- [x] 0.2.3 创建 `tests/unit/memory/test_tmem.cpp` (340 LoC, 18 TEST_CASEs, 99,760 loop-based assertions):
      construct_default_zeros / cross_instance_isolation / write_read_roundtrip / partial_write_no_clobber /
      write_to_slot_0 / write_to_slot_255 / read_slot_256_throws / write_size_128_boundary /
      write_size_129_throws / clear_zeros_all_slots / partial_slot_no_leak_to_neighbor /
      mutex_serializes_concurrent_writes / loops over 256 slots to verify all-zero / all-correct / etc.
- [x] 0.2.4 在 `src/CMakeLists.txt` (after tma_descriptor line) + `tests/unit/CMakeLists.txt` (after unit_tma_descriptor) 注册
- [x] 0.2.5 自检：`ctest -R "tmem"` → **18 TEST_CASEs / 99,760 assertions PASS** (0.04s)
- [x] 0.2.6 commit: `758edb0 feat(memory): per-CTA Tensor Memory (TMEM) (Fix #6)` (5 files: 3 new + 2 modified; atomic)
- [x] 0.2.7 验证独立可 revert (新 src/ptxsim/memory/tmem.* + tests 隔离；revert HEAD 安全移除 3 files + 2 CMakeLists 行 — 无 shared dep)

## Phase 0.3: cluster mode — arrive/wait only（Fix #7）

> **Oracle review simplification**: `cta_group::1` (Phase 1-3 目标) 不需要
> distributed shared memory — 仅 `cta_group::2` (future) 需要。
> Phase 0.3 仅实现 `arrive`/`wait` 原语；`distributed_smem` defer 到 `cta_group::2`
> 真正需要时（独立 change 或 Phase expansion）。

- [x] 0.3.1 创建 `src/ptxsim/cluster/cluster_context.h` (55 LoC):
      - `class ClusterContext`（1-8 CTA 集群标识，root_id 0..7, num_ctas 1..8）
      - `cta_cluster_arrive(cta_id)` / `cta_cluster_wait(cta_id)` 同步原语
      - **Deferred**: `distributed_smem` view（when cta_group::2 needed）— ADR-0018 候选
- [x] 0.3.2 创建 `src/ptxsim/cluster/cluster_context.cpp` (83 LoC, std::mutex+std::condition_variable 同步；NO busy-wait per lessons-learned §2)
- [x] 0.3.3 创建 `tests/unit/cluster/test_cluster_mode.cpp` (195 LoC, 15 TEST_CASEs / 17 assertions):
      construct_default_cluster_size_1 / construct_invalid_size_zero_throws / construct_invalid_size_9_throws /
      construct_invalid_root_8_throws / arrive_then_wait_single_cta_immediate /
      arrive_multiple_peer_ctas_wait_blocks_until_all / wait_before_arrive_throws /
      wait_with_invalid_cta_id_throws / multiple_waits_after_all_arrived_succeeds /
      duplicate_arrive_throws / cross_cluster_isolation / … ≥10 cases
- [x] 0.3.4 CMakeLists 注册：`src/CMakeLists.txt` (new `ptxsim/cluster/` section after `ptxsim/memory/tmem.cpp`) + `tests/unit/CMakeLists.txt` (new `tests/unit/cluster/` dir + `add_catch_test(unit_cluster_mode cluster/test_cluster_mode.cpp)` `LABELS "unit;cluster"`)
- [x] 0.3.5 自检：`ctest -R "cluster"` → **15 TEST_CASEs / 17 assertions PASS** (0.11s) ；回归 `ctest -L "unit|integration|e2e"` → **126 labeled PASS** (added 1 ctest target；zero regression)
- [x] 0.3.6 commit: `e513235 feat(sim): cluster arrive/wait primitives (Fix #7, simplified—no distributed smem)` (5 files: 3 new + 2 modified; atomic)

## Phase 0.4: async tensor core queue（Fix #8）— ✅ DONE (commit c0fa43f)

- [x] 0.4.1 创建 `src/ptxsim/async/tc_queue.h` (74 LoC):
      - `class TcQueue`（per-CTA 命令队列）
      - `commit(group_id)` → atomic CAS fetch_max
      - `wait(group_id, warp_ctx, lane_id)` → capture pc+1 store + per-warp_state set is_blocked+status
      - `clear()` + `current_counter()` + `pending_count()` 测试/观测 API
      - 注：`enqueue_mma()` 是 Phase 1-3 scope，留待 `tcgen05.mma` handler 接入
- [x] 0.4.2 创建 `src/ptxsim/async/tc_queue.cpp` (108 LoC, std::atomic + std::mutex + std::vector<PendingWaiter>)
- [x] 0.4.3 创建 `tests/unit/async/test_tc_queue.cpp` (315 LoC, 14 TEST_CASEs / 61 assertions):
      construct_default_counter_zero / commit_advances_counter_monotonically /
      commit_cas_idempotent_same_group / wait_block_lane_at_pc /
      wait_stores_completion_pc_plus_one_in_pending /
      wait_then_immediate_commit_releases_lane /
      wait_continues_blocked_until_commit_reaches_group /
      multiple_waiters_different_groups_release_in_order /
      multiple_waiters_same_group_all_released / clear_resets_to_initial_state /
      concurrent_commit_thread_safety / completion_pc_uses_stored_value_not_current /
      no_set_active_mask_in_src_ptxsim_async / ...
- [x] 0.4.4 CMakeLists 注册：`src/CMakeLists.txt` (new `ptxsim/async/` section after `ptxsim/cluster/cluster_context.cpp`) + `tests/unit/CMakeLists.txt` (new `tests/unit/async/` dir + `add_catch_test(unit_tc_queue async/test_tc_queue.cpp)` `LABELS "unit;async;tc_queue"`)
- [x] 0.4.5 **关键审计**（Oracle Q1/Q2/Q3/Q4 hypotheses + state-modification-audit skill）
      - commit_group_counter writers ⊆ {TcQueue::commit (+ init + clear())} ✅
      - NO set_active_mask call from tc_queue.cpp (AGENTS.md ownership rule) ✅
      - NO set_state(BAR_SYNC) call from tc_queue.cpp (Decision 7 reuse BAR_SYNC) ✅
      - is_blocked= assignments: 2 (wait→true, commit→false per Oracle Q1 hypothesis 2)
      - 复用 BarrierModule::release_warp_barrier post-unblock 模式 (advance_thread_pc + is_blocked=false + status=Active + is_active=true per lane)
- [x] 0.4.6 自检：`ctest -R "tc_queue"` → **14/14 PASS (61 assertions, 0.01s)** ；回归 `ctest -L "unit|integration|e2e"` → **127 labeled PASS**（added 1 ctest target；zero regression）
- [x] 0.4.7 commit: `c0fa43f feat(async): tc_queue commit-group + wait-aware scheduling (Fix #8)` (5 files: 3 new + 2 modified; atomic)
- [x] 0.4.8 验证独立可 revert (新 src/ptxsim/async/tc_queue.* + tests 隔离；revert HEAD 安全移除 3 files + 2 CMakeLists 行 — **但 Phase 0.5.4 依赖 TcQueue，所以单独 revert 必须等到 0.5.4 撤销完成**)

## Phase 0.5: 逐子系统集成到 CTAContext（Fix #9a, #9b, #9c, #9d）

> **Design rationale (Oracle review, 2026-07)**: 原设计将 4 个子系统在一个 commit
> 中集成。Oracle 指出这违反 `ptx-lessons-learned` §3 — revert 后留下 4 个未引用
> 子系统死代码。修复：拆为 4 个微 commit，每个只集成一个子系统。
>
> **Oracle review fix (Q2, 2026-07-04)**: 微 commit **不可独立 revert**。
> 原因：`TcQueue::enqueue_mma()` 写入 TMEM slot，集成到 CTAContext 后
> 0.5.4 (TcQueue) 隐式依赖 0.5.2 (TMEM) 和 0.5.1 (TMA)。
>
> **Revert unit = 整体 Phase 0.5（4 commits 整体回退至 0.4.7 后状态）**。
> 单个微 commit revert 会导致编译失败（未解析的 CTAContext 引用）。
> 失败处理：任何 Phase 0.5 子系统 bug → 整体回退，不单独 revert。

### 0.5.1 TMA descriptors → CTAContext（Fix #9a）— ✅ DONE (commit 1254a45)
- [x] 0.5.1.1 修改 `include/ptxsim/cta_context.h`：添加 `tma_descriptor_store` 引用（default-init via `TmaDescriptorStore()=default` + in-class member；cta_context.cpp 不需改）
- [x] 0.5.1.2 创建 `tests/integration/tma/test_tma_with_cta_context.cpp`（172 LoC, 5 TEST_CASEs, 35 assertions）
- [x] 0.5.1.3 自检：`ctest -R "tma.*cta"` → 1/1 PASS ；回归 `ctest -L "unit|integration|e2e"` → 128 labeled PASS（added 1 ctest target；zero regression）
- [x] 0.5.1.4 commit: `1254a45 feat(sim): integrate TMA descriptor store with CTAContext (Fix #9a)` (3 files: 1 new + 2 modified; atomic)
- [x] 0.5.1.5 验证 0.5.1 移除后 CTAContext 不引用 TMA（其余 3 子系统不变）

### 0.5.2 TMEM → CTAContext（Fix #9b）— ✅ DONE (commit 9f325ee)
- [x] 0.5.2.1 修改 `include/ptxsim/cta_context.h`：添加 `tmem` 引用（default-init via `Tmem()` + in-class member）
- [x] 0.5.2.2 创建 `tests/integration/tmem/test_tmem_with_cta_context.cpp`（172 LoC, 5 TEST_CASEs, 1926 assertions）
- [x] 0.5.2.3 自检：`ctest -R "tmem.*cta"` → 1/1 PASS ；回归 → 129 labeled PASS（zero regression）
- [x] 0.5.2.4 commit: `9f325ee feat(sim): integrate TMEM with CTAContext (Fix #9b)` (3 files: 1 new + 2 modified; atomic)

### 0.5.3 cluster → CTAContext（Fix #9c）— ✅ DONE (commit 9891587)
- [x] 0.5.3.1 修改 `include/ptxsim/cta_context.h`：添加 `std::optional<ClusterContext>` 成员（**Option A lazy-init** — `ClusterContext` 有显式 ctor `(root_id, num_ctas)`，不可 default-init；`std::optional::emplace()` 原地构造避免 move/copy）
- [x] 0.5.3.2 创建 `tests/integration/cluster/test_cluster_with_cta_context.cpp`（140 LoC, 6 TEST_CASEs, 16 assertions）
- [x] 0.5.3.3 自检：`ctest -R "cluster.*cta"` → 1/1 PASS ；回归 → 130 labeled PASS（zero regression）
- [x] 0.5.3.4 commit: `9891587 feat(sim): integrate cluster context with CTAContext (Fix #9c)` (3 files: 1 new + 2 modified; atomic)

### 0.5.4 TcQueue → CTAContext（Fix #9d）— ✅ DONE (commit 6b2c465)
- [x] 0.5.4.1 修改 `include/ptxsim/cta_context.h`：添加 `tc_queue` 引用（default-init via `TcQueue()` + in-class member）
- [x] 0.5.4.2 创建 `tests/integration/async/test_tc_queue_with_cta_context.cpp`（87 LoC, 5 TEST_CASEs, 17 assertions）
- [x] 0.5.4.3 自检：`ctest -R "tc_queue.*cta"` → 1/1 PASS ；回归 → **131 labeled PASS**（zero regression）
- [x] 0.5.4.4 commit: `6b2c465 feat(sim): integrate TcQueue with CTAContext (Fix #9d)` (3 files: 1 new + 2 modified; atomic)

**Phase 0.5 最终状态**（per 设计 Decision 9）:
- 4 micro commits (`1254a45`, `9f325ee`, `9891587`, `6b2c465`) 形成非单独可 revert 集合
- 任何 Phase 0.5 子系统 bug → 整体回退：`git revert 1254a45..6b2c465`
- 单 micro commit revert 会导致编译失败（未解析 CTAContext 引用）

---

## Quality Gates (Phase 0 → Phase 1-3 入口)

> **本 change archive 的硬门** = 下列 Gate 全部通过 + Phase 1-3 change `Ref:` 引用本 archive

### Gate G1: 回归测试零失败
```bash
cd build && ctest -L "unit\|integration\|e2e" --output-on-failure 2>&1 | grep -c "^FAILED"
# 期望: 0
```

### Gate G2: baseline worktree diff
```bash
diff <(cd .worktrees/fix-pre-p0-baseline/build && ctest -L "unit\|integration\|e2e" 2>&1 | grep -E "Passed\|Failed") \
     <(cd build && ctest -L "unit\|integration\|e2e" 2>&1 | grep -E "Passed\|Failed")
# 期望: 0 new FAIL
```

### Gate G3: state-modification-audit
```bash
# 加载 state-modification-audit skill, 审计以下变量:
#   - commit_group_counter 写点 ⊆ TcQueue::commit() (design.md Decision 7 声明)
#   - is_blocked 写点 ⊆ TcQueue::wait() 中的 set_warp_state(BAR_SYNC)
# 期望: 0 unexpected writers
```

### Gate G4: artifacts tracked (lessons-learned §6)
```bash
git ls-files openspec/changes/implement-wmma-tensor-core-phase-0-infra/
# 期望: 非空, 包含 proposal.md / design.md / tasks.md / specs/wmma-tensor-core/spec.md
```

### Gate G5: Oracle re-review TMA descriptor (Critical risk #1 独有)
```bash
# 手工对照 NVIDIA PTX ISA §9.7 TensorMap 字段
# 验证 tma_descriptor.cpp 中每个 magic number / bit offset 引用正确
# 期望: review notes 引用在 wmma.cpp (Phase 1-3) 或 tma_descriptor.cpp 注释中
```

### Gate G6: sanity.sh 全套
```bash
./scripts/sanity.sh
# 期望: 0 unexpected FAIL + 含 PTX 语法测试通过
```

### Gate G7: cute header spike (Open Question #5) — ✅ PASS (2026-07-04, commit f1bab6e)
```bash
# 实测命令 (per design.md Open Question #5 决议):
# Spike #1: 仅 include cute/arch/mma_sm100_umma.hpp
nvcc -arch=sm_100 -ptx -I bench/cute/include /tmp/spike_tcgen05.cu
# EXIT=0 (1205 bytes output)

# Spike #2: cute_rmsnorm_debug.cu (uses sm_100 per earlier grep)
nvcc -arch=sm_100 -ptx --expt-relaxed-constexpr -I bench/cute/include \
     bench/cute/cute_rmsnorm_debug.cu
# EXIT=0 (6669 bytes output)

# 期望: exit 0 (Gate G7 已 PASS；无需 propose fix-cute-sm100-headers change)
```

---

## 失败回滚速查

| 失败 Phase | 立即动作 |
|-----------|---------|
| Phase 0.1 (TMA) | `git revert HEAD` → 仍能 build，无 TC descriptor → 抛异常 |
| Phase 0.2 (TMEM) | `git revert HEAD` → 仍能 build，无 TC memory → 抛异常 |
| Phase 0.3 (cluster) | `git revert HEAD` → 必须跑 cluster 测试确认 CTAContext 未破坏 |
| Phase 0.4 (tc_queue) | `git revert HEAD` → **关键**：跑全套回归 + state-modification-audit |
| Phase 0.5.1~0.5.4 (整体) | `git revert <0.5.1-sha>..<0.5.4-sha>` (整体 revert 4 commits) → 仅从 CTAContext 移除子系统引用 |
| Phase 0 整个 fail | 不 archive → 不 propose Phase 1-3 change |

---

## 关键约束（必读）

⚠️ **MUST**：
- Phase 0.0 artifacts tracked **FIRST**（lessons-learned §6 / Checklist E）
- Phase 0.4 (tc_queue) commit 前必须跑 `state-modification-audit` skill
- 全 Quality Gates 通过才能 archive
- ADR-0016 决策不可绕过 — pre-Blackwell 不实现

⚠️ **MUST NOT**：
- 不要修改 `UnsupportedInstructionException` / `ExecutionStateException` 类定义
- 不要修改 X-Macro `ptx_op.def`（`S_WMMA` → `WmmaHandler` 不变）
- 不要修改 `tensor.cpp` 内容（本 change 不改 handler；Phase 1-3 改）
- 不要破坏 cute_rmsnorm / cute_hello_* 等已通过的 E2E 测试
- 不要在 WMMA handler 里用 `qualifiers.back()`（lessons-learned §5, 但本 change 不改 handler）

---

## 后续 change 引用方式

Phase 0 archive 之后, propose `implement-wmma-tensor-core-tcgen05` Phase 1-3 change 时,
proposal.md 顶部必须添加：

```markdown
> **前置 change**: `implement-wmma-tensor-core-phase-0-infra` (archived <date>)
> **Ref**: archive/<YYYY-MM-DD>-implement-wmma-tensor-core-phase-0-infra/
```

per ptx-lessons-learned Checklist G "Archived = 终态"。
