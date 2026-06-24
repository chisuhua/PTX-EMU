# T2-3 — ThreadContext / WarpContext god class POD 拆分 (Tasks)

> **总工期**: 7-10 天 | **依赖**: T1-1 ✅, T2-1 ✅, integrate-barrier-module-cta-warp 必须合并

> **TDD 三阶段**: 严格遵循 AGENTS.md §TDD 流程（每步先写测试再实现）。

> **关键策略**: 拆分为 3 阶段（Phase A 拆分 → Phase B 测试更新 → Phase C 验证），每阶段独立 commit。

> **详细 plan 与 explore 报告**: 见 `docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` §Task 3 + bg_7169b052 16m43s explore 报告（8 sections A-H）

> **状态 (2026-06-24)**: Phase A1-A4 + B1-B2 + C1 完成 (8 commits)。**A5 PoC 进展**: 1 commit (`421eec9` pushed) 迁移 `test_barrier_module_integrated.cpp` 到 direct BarrierModule API，验证通过。**A5 真实阻塞**: `BarWarpSyncHandler::processOperation` (`src/ptxsim/instructions/barrier.cpp:140-280`) 主体逻辑仍使用 legacy `warp_state.wbars[]` + `current_wbar_id`，BarrierModule 状态未被更新。详见 `docs/superpowers/findings/2026-06-24-t2-3-a5-blocker-discovery.md`。

---

## Phase A: 核心拆分（6 天）

### A1: 创建 7-8 个 POD 头文件（1.5 天）

**Files:** (8 new files)
- `include/ptxsim/contexts/{exec_state,register_predicate,memory_ref,program_ref,backend_links,identity,lane_mask}.h`
- `src/ptxsim/contexts/*.cpp`（8 placeholder .cpp 文件）
- `tests/unit/contexts/test_*_pod.cpp`（8 个单元测试）

- [x] **Step 1**: 写 8 个 POD 单元测试（Red 阶段）
- [x] **Step 2**: 实现 8 个 POD 头文件 + 占位 .cpp（Green 阶段）
- [x] **Step 3**: 编译 + `ctest -L "contexts" -V` 全过
- [x] **Step 4**: `git commit -m "refactor(contexts): extract 8 POD structs from ThreadContext/WarpContext (T2-3 A1)"` (commit `7054593`)

### A2: WarpState 精简（0.5 天）

**Files:**
- Modify: `include/ptxsim/warp_state.h`（6 字段 → 2 字段：`threads[]` + `exec_mask`）
- New: `tests/unit/warp/test_warp_state_minimal.cpp`
- Delete: `thread_predicates`（0 引用）、`wbars` + `current_wbar_id`（已 deprecated）、`warp_pc`（0 引用）

- [x] **Step 1**: 写 WarpState 单元测试（Red）
- [x] **Step 2**: 修改 `warp_state.h` 精简到 2 字段
- [x] **Step 3**: `grep -rn "thread_predicates\|warp_state\.warp_pc\|warp_state\.wbars\|warp_state\.current_wbar_id"` 全项目访问点
- [x] **Step 4**: 逐个修复（迁移到 BarrierModule 或直接删除）
- [x] **Step 5**: 编译 + 跑 sanity.sh --quick
- [x] **Step 6**: `git commit -m "refactor(warp-state): reduce from 6 to 2 fields, remove dead code (T2-3 A2)"` (commit `5617665`)

> **Note**: `wbars` + `current_wbar_id` 仅做了 deprecation 标注，未物理删除（留给 A5）。

### A3: ThreadContext 改 facade（1.5 天）

**Files:**
- Modify: `include/ptxsim/thread_context.h`（300 → ~150 行，4 POD 持有 + 转发方法）
- Modify: `src/ptxsim/core/thread_context.cpp`（848 行，按 POD 重排方法）

- [x] **Step 1**: 写 facade 兼容性测试（Red）
- [x] **Step 2**: 修改 `thread_context.h` 持有 4 POD（ExecState/RegisterPredicate/Memory/ProgramRef）
- [x] **Step 3**: 编译（可能数百错误）+ 跑测试
- [x] **Step 4**: 全文件传播修复字段访问点
- [x] **Step 5**: 跑 sanity.sh --quick
- [x] **Step 6**: `git commit` A3a (commit `7952120`) + A3b (commit `67ad828`)

### A4: WarpContext 改 facade（1.5 天）

**Files:**
- Modify: `include/ptxsim/warp_context.h`（274 → ~150 行，3 POD 持有 + 转发方法）
- Modify: `src/ptxsim/core/warp_context.cpp`（487 → ~250 行）

- [x] **Step 1**: 写 facade 兼容性测试（Red）
- [x] **Step 2**: 修改 `warp_context.h` 持有 3 POD（LaneMask/Identity/BackendLinks）
- [x] **Step 3**: 编译 + 跑测试
- [x] **Step 4**: 全文件传播修复
- [x] **Step 5**: 跑 sanity.sh --quick
- [x] **Step 6**: `git commit` A4a (commit `8b9b025`) + A4b (commit `2a3b48a`)

### A5: 物理删除 EXE_STATE / wbars / thread_predicates / warp_pc（1.0 天）

**Files:**
- Modify: `include/ptxsim/thread_context.h`（删除 `EXE_STATE state` 字段，统一到 `warp_state.threads[i].status`）
- Modify: `include/ptxsim/execution_types.h`（如 EXE_STATE 0 引用则删除整个 enum）
- Modify: 全项目传播 EXE_STATE 访问点（用 `warp_state.threads[i].status` 替换）
- Delete: `wbars` + `current_wbar_id`（已迁到 BarrierModule）+ `thread_predicates` + `warp_pc`

- [ ] **Step 1**: grep 全项目 EXE_STATE 访问点
- [ ] **Step 2**: 逐个迁移到 `warp_state.threads[i].status`
- [ ] **Step 3**: 编译 + 跑测试
- [ ] **Step 4**: 物理删除字段
- [ ] **Step 5**: `git commit -m "refactor(state): unify EXE_STATE to warp_state.threads[].status, remove dead fields (T2-3 A5)"`

> **⛔ BLOCKED: 真因 = `BarWarpSyncHandler::processOperation` 仍用 legacy state**.
>
> **Discovery (2026-06-24)**: 调查 `tests/integration/pc/test_pc_management_integrated.cpp` 迁移失败时发现：
> - `barrier.cpp:140-280` 主体逻辑使用 `warp_state.wbars[idx]` + `warp_state.current_wbar_id`（lines 145, 157, 160-161, 184, 198, 215-263）
> - `barrier_module` 仅在 line 358 一个子函数被引用
> - 因此 `warp->get_wbar(0).is_complete() == true`（legacy 路径返回正确），但 `warp->get_cta_context()->get_barrier_module().get_warp_barrier(0)->is_complete()` 返回 false（BarrierModule 未被 handler 更新）
>
> **真阻塞 = handler 本身需要迁移**（~140 行, 14+ legacy field uses, 含 2 个 BUG workaround paths）:
> - BUG-RECONVERGENCE-SIMPLEGEMM: divergent warp hit barrier with 2 unique PCs
> - BUG-CUTE-RMSNORM-BROADCAST-SKIP: `current_wbar_id < 0` check after release
>
> **实际步骤** (3-5 天人工):
> 1. 迁移 `BarWarpSyncHandler::processOperation` 到使用 BarrierModule 作为单一 source of truth（dual-write 过渡期）
> 2. 迁移 test files (60+ call sites 跨 7 文件)
> 3. 物理删除 `wbars[]` + `current_wbar_id` 字段
> 4. 物理删除 `get_wbar()` shim + `[[deprecated]]` markers
>
> **PoC 进展**: `tests/integration/barrier/test_barrier_module_integrated.cpp` 已迁移 (commit `421eec9`)，是纯 BarrierModule 单元测试，可作为模板。详见 `docs/superpowers/findings/2026-06-24-t2-3-a5-blocker-discovery.md`。

---

## Phase B: 测试更新（2.5 天）

### B1: 重写测试基础设施（1.0 天）

**Files:**
- Modify: `include/ptxsim/testing/warp_test_utils.h`（207 行，重写以匹配新 API）
- Modify: `include/ptxsim/testing/scheduler_utils.h`（72 行）
- Modify: `include/ptxsim/testing/assertion_utils.h`

- [x] **Step 1**: grep `WarpContext\|ThreadContext` 全部测试文件
- [x] **Step 2**: 测试基础设施已使用 POD 访问器（`warp.get_warp_state().threads[i].pc`）正确。仅 `warp_test_utils.h:reset_warp()` 有 1 处 `t->state = RUN` 改为 `t->set_state(RUN)`（forward compat with A5）
- [x] **Step 3**: 编译 + 跑测试
- [x] **Step 4**: `git commit -m "refactor(test): use set_state() in reset_warp for forward compat (T2-3 B1)"` (commit `33e1f99`)

### B2: 更新 69 个测试文件（1.5 天）

**Files:** 122 个 `tests/{unit,integration,e2e}/*.{cpp,cu}` 文件（其中 69 个为新三类目录下的测试）

- [x] **Step 1**: `grep -rln "t->state\s*=" tests/` 仅 1 个文件（`test_post_barrier_divergence.cpp`）
- [x] **Step 2**: 改 `t->state = RUN` → `t->set_state(RUN)`（与 B1 一致）
- [x] **Step 3**: `cmake --build build` 验证全部编译通过（clean build，无 error）
- [x] **Step 4**: `ctest` 跑全量测试（serial）— 仅 7 个 pre-existing 失败
- [x] **Step 5**: `git commit -m "refactor(test): use set_state() in test_post_barrier_divergence (T2-3 B2)"` (commit `8248303`)

> **7 个 pre-existing ctest 失败**（T2-3 不引入，全部已在 `docs/developer-guide/KNOWN_ISSUES.md` 文档化）：
> 1. `unit_simt_stack_stale_entry_blocks_lane0` — BUG-DISPATCH-GATE-LANE0-SKIP hypothesis
> 2. `integration_cute_rmsnorm_bar_sync_pattern` — BUG-DISPATCH-GATE-LANE0-SKIP reproduction
> 3. `integration_ldglobal_no_hang` — fix-ldglobal-active-count-hang
> 4. `simpleCONV-int` / `simpleCONV-float` / `simpleCONV-double` — benchmark timeout (pre-P0c baseline red)
> 5. `cute_rmsnorm` — kernel result mismatch (pre-P0c baseline red)

---

## Phase C: 验证（1.0 天）

### C1: 全量 ctest + sanity.sh（0.5 天）

- [x] `cd build && ctest --output-on-failure` — 7 个 pre-existing 失败，其余全过
- [x] `cd /workspace/project/PTX-EMU && ./scripts/sanity.sh` — 全过（含 33/33 PTX 语法测试）

### C2: ASan 验证（0.3 天）

- [x] `cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"`
- [x] `cmake --build build-asan && ctest --test-dir build-asan -L "barrier;e2e" -V`
- [x] **Leak fix** (commit `2d24403`): `InstructionFactory::initialize()` 注册 `cleanup()` via `std::atexit`。之前每个 test binary 漏 ~1296 bytes（106 个 handler `new` 没释放），现在 ASan 跑过 barrier/e2e/thread/warp 标签除 pre-existing 失败外无 leak 报告。
- [ ] **Outstanding** (out of T2-3 scope): 5 个测试 (`unit_divergence_sync_standalone_integrated`、`integration_barrier_divergence_scheduling`、`integration_broadcast_after_barrier`、`e2e_barrier_warp_sync`、`e2e_divergence_sync`) 在 ASan 下报 `heap-buffer-overflow` at `ThreadContext::collect_operands` (`thread_context.cpp:393`)。pre-existing memory-safety bug，单独跟进。

### C3: 性能对比（0.2 天）

- [ ] 用 callgrind 对比 `step_warp` 单步开销（应基本无差异，POD facade 是 inlined）
- [ ] `git commit -m "verify(perf): POD facade vs god class no perf regression (T2-3 closure)"`

> **Defer**: 性能对比不是阻塞项，POD facade 设计上为 inlined，且 add-thread / 同步路径均为简单赋值。如需时再补。

---

## Phase 3 验证门禁

- [x] A1-A4 全部完成（6 commits: `7054593` `5617665` `7952120` `67ad828` `8b9b025` `2a3b48a`）
- [ ] A5 完成（blocked on BarWarpSyncHandler migration - 真因。PoC 部分完成 `421eec9`）
- [x] B1-B2 全部完成（2 commits: `33e1f99` `8248303`）
- [x] C1-C2 全部完成（1 commit: `2d24403`）
- [x] ctest 全过（除 7 pre-existing）+ sanity.sh 全过
- [x] ASan 无 leak 报告（leak fix landed）
- [ ] ASan 无 heap-buffer-overflow 报告（5 pre-existing, 独立跟踪）
- [x] 文档：`docs/audits/HEALTH-AUDIT-2026-06-21.md` §1.2 M2/M3 标记已修复 (Phase C2 follow-up)
