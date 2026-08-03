# god-class-refactor-sm-context-phase3 - Tasks

> **Phase numbering note**: Phase 1+2 already shipped via `2026-07-27-god-class-refactor-sm-context` archive (commits `59df7abf`, `01389c18`). This change covers Phase 3+.

> **Evidence note**: all line references in this file come from a verified `grep` of `src/ptxsim/core/sm_context.cpp` at HEAD `041ae7d1` on 2026-08-03.

## 1. Phase 0: 准备工作

- [ ] 1.1 MUST 验证 C-18 已落地：`ls openspec/changes/archive/*-refactor-warp-context` 存在
- [ ] 1.2 MUST 建立 baseline worktree `.worktrees/baseline-god-class-p3`（lessons-learned Checklist B）
- [ ] 1.3 MUST 建立实施 worktree `.worktrees/refactor-god-class-p3` on branch `openspec/god-class-refactor-sm-context-phase3`
- [ ] 1.4 MUST 记录基线：`wc -l src/ptxsim/core/sm_context.cpp` = 862
- [ ] 1.5 MUST 记录基线：`cd build && ctest --output-on-failure` 全绿
- [ ] 1.6 MUST 验证：`grep -n 'w->update_active_mask()' src/ptxsim/core/sm_context.cpp` 命中 line 362
- [ ] 1.7 MUST 验证：`sed -n '354,359p' src/ptxsim/core/sm_context.cpp` 显示 BUG-001 active_count sync fix 注释（含 "only updated by update_active_mask()" at line 357）
- [ ] 1.8 MUST 验证：`grep -n 'class SMContext::exe_once' src/ptxsim/core/sm_context.h` 记录当前签名
- [ ] 1.9 MUST 验证：`ls src/ptxsim/core/sm_context_reconvergence.{h,cpp} src/ptxsim/core/sm_context_cpptlm_inject.{h,cpp}` 存在（既有 helpers）

## 2. Phase 3: CTA block dispatch 拆分（3h）

- [ ] 2.1 MUST 写 `tests/unit/sm/test_sm_block_dispatch.cpp`（RED）：3 个失败 case — admit happy path / overflow → pending / cleanup releases resources
- [ ] 2.2 MUST 创建 `src/ptxsim/core/sm_block_dispatch.{h,cpp}`（helper namespace `sm_block_dispatch::`）
- [ ] 2.3 MUST 提取 `add_block` (130-204), `try_admit_pending_blocks` (206-258), `cleanup_finished_blocks` (628-643), `free_shared_memory` (645-665), `reserve_resources` (667-689), `release_resources` (691-695) 至 `sm_block_dispatch::`
- [ ] 2.4 MUST NOT 改变 WarpContext public API 调用点
- [ ] 2.5 MUST NOT 改变 `SMContext::exe_once()` 签名
- [ ] 2.6 MUST 更新 `src/ptxsim/core/CMakeLists.txt` 注册新源文件
- [ ] 2.7 MUST 在 `SMContext` 中 friend-declare `sm_block_dispatch::`（最小权限原则）
- [ ] 2.8 MUST 验证：`cd build && cmake --build . && ctest -R unit_sm_sm_block_dispatch --output-on-failure` 3/3 PASS
- [ ] 2.9 MUST 递归锁审计（lessons-learned §2）：`grep -n 'lock_guard\|unique_lock' src/ptxsim/core/sm_block_dispatch.cpp src/ptxsim/core/sm_context.cpp` 比对，无新增同锁 public 调用
- [ ] 2.10 MUST 验证：`wc -l src/ptxsim/core/sm_context.cpp` ≤ 712（≥150 行净减少）
- [ ] 2.11 MUST 验证：`cd build && ctest --output-on-failure` 全绿（无回归）
- [ ] 2.12 git commit -m "refactor(sm): extract CTA block dispatch to sm_block_dispatch.{h,cpp}"

## 3. Phase 4: warp lifecycle 拆分（2.5h）

- [ ] 3.1 MUST 写 `tests/unit/sm/test_sm_warp_lifecycle.cpp`（RED）：3 个失败 case — warp registration / retirement via `update_state` / `get_active_warps_count` after retirement
- [ ] 3.2 MUST 创建 `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}`（helper namespace `sm_warp_lifecycle::`）
- [ ] 3.3 MUST 提取 `update_state` (586-626), `select_next_group` (831-855), `suspend_and_switch` (856-862), `get_active_warps_count` (562-570), `get_active_threads_count` (572-580)
- [ ] 3.4 MUST 行级随迁：`grep -n 'w->update_active_mask()' src/ptxsim/core/sm_context.cpp`（如该调用随 warp lifecycle 迁移，则 :354-359 BUG-001 注释必须同步迁移，lessons-learned §1）
- [ ] 3.5 MUST NOT 改变 BarrierModule 内部实现（lessons-learned §14）
- [ ] 3.6 MUST 更新 `src/ptxsim/core/CMakeLists.txt`
- [ ] 3.7 MUST 在 `SMContext` 中 friend-declare `sm_warp_lifecycle::`
- [ ] 3.8 MUST 验证：`ctest -R unit_sm_sm_warp_lifecycle --output-on-failure` 3/3 PASS
- [ ] 3.9 MUST 递归锁审计（lessons-learned §2）：同 Phase 3 流程
- [ ] 3.10 MUST 验证：`wc -l src/ptxsim/core/sm_context.cpp` ≤ 600
- [ ] 3.11 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 3.12 git commit -m "refactor(sm): extract warp lifecycle to sm_warp_lifecycle.{h,cpp}"

## 4. Phase 5: SM barrier wrapper 拆分（1.5h，含 go/no-go）

- [ ] 4.1 MUST 决策：若 Phase 4 后剩余 SM barrier glue < 50 行 → go/no-go 改为 fold-back，skip new file；本 Phase 转为小补丁直接合并到 `sm_block_dispatch.{h,cpp}`
- [ ] 4.2 若 go：写 `tests/unit/sm/test_sm_barrier_wrapper.cpp`（RED）：2 个失败 case — `cta_context->get_barrier_module()` 委托 / null-barrier fallback
- [ ] 4.3 若 go：创建 `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}` 并迁移 SM-level barrier glue
- [ ] 4.4 若 go：在 `SMContext` 中 friend-declare `sm_barrier_wrapper::`
- [ ] 4.5 若 go：更新 `src/ptxsim/core/CMakeLists.txt`
- [ ] 4.6 若 go：验证 `ctest -R unit_sm_sm_barrier_wrapper --output-on-failure` 2/2 PASS
- [ ] 4.7 MUST 递归锁审计（lessons-learned §2）
- [ ] 4.8 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 4.9 git commit -m "refactor(sm): extract SM barrier wrapper to sm_barrier_wrapper.{h,cpp}" 或 fold-back patch

## 5. Phase 6: 最终验证（1h）

- [ ] 5.1 MUST 验证：`wc -l src/ptxsim/core/sm_context.cpp` ≤ 600
- [ ] 5.2 MUST 验证：`grep -n 'update_active_mask' src/ptxsim/core/sm_context*.cpp` 列出所有 call site，无 orphan
- [ ] 5.3 MUST 验证：`grep -n 'BUG-001' src/ptxsim/core/sm_context*.cpp` 列出所有 comment block，无 orphan
- [ ] 5.4 MUST 验证：`grep -c 'exe_once' src/ptxsim/core/sm_context.h` = 1（签名不变）
- [ ] 5.5 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 5.6 MUST 验证：`tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 分支测试 PASS（lessons-learned §14）
- [ ] 5.7 MUST 验证：`tests/integration/barrier/*` + `tests/integration/divergence/*` 零回归
- [ ] 5.8 MUST 更新 `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` helper cap ≤4 → ≤5 + 指向 follow-up `exe-once-decomposition`
- [ ] 5.9 SHOULD 更新 `src/ptxsim/core/AGENTS.md` 表格包含 `sm_block_dispatch` / `sm_warp_lifecycle` / `sm_barrier_wrapper`（如创建）

## 6. 应用阶段

- [ ] 6.1 MUST 运行 `openspec validate god-class-refactor-sm-context-phase3 --strict`
- [ ] 6.2 MUST commit 4 个 OpenSpec artifacts：`git add openspec/changes/god-class-refactor-sm-context-phase3/ && git commit -m "docs(openspec): god-class-refactor-sm-context-phase3 design adjustments"`
- [ ] 6.3 MUST 通过所有验证后 `openspec archive god-class-refactor-sm-context-phase3 --yes`

## 验收

- `src/ptxsim/core/sm_context.cpp ≤ 600 行`（基线 862，净减 ≥ 262 行；`<250` 为 multi-change end-state，留待 follow-up `exe-once-decomposition`）
- 新增 3 个 helper：`sm_block_dispatch.{h,cpp}` / `sm_warp_lifecycle.{h,cpp}` / `sm_barrier_wrapper.{h,cpp}`（若 Phase 5 走 fold-back 路径则仅 2 个）
- helper cap 由 ≤4 上调至 ≤5，已记录在 `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2`
- 新增 unit tests：`tests/unit/sm/test_sm_block_dispatch.cpp`（3 case）、`test_sm_warp_lifecycle.cpp`（3 case）、`test_sm_barrier_wrapper.cpp`（2 case，若创建）
- `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 分支测试 PASS
- `tests/integration/barrier/*` + `tests/integration/divergence/*` 零回归
- `tests/integration/.../execute_warp_instruction` 路径零回归
- 每个 Phase commit 独立可 revert（lessons-learned §3）
- ptx-lessons-learned §1 (line-level diff), §2 (recursive lock), §14 (byte-identical fallback), Checklist B 全部勾选

## 关键约束（MUST/MUST NOT）

- MUST §1 行级 diff（lessons-learned SKILL.md:48-77）：使用当前 verified sites `sm_context.cpp:362` (call) + `:354-359` (comment block，含 :357 关键句)，不是过时的 379/374
- MUST Checklist B（lessons-learned SKILL.md:474-483）：worktree + 3 Phase commit
- MUST §14 step_b no-op fallback 4 分支测试锁定（lessons-learned SKILL.md:409-455）
- MUST §2 递归锁审计（lessons-learned SKILL.md）：不在持锁方法内调用同锁 public 方法
- MUST NOT 改 `SMContext::exe_once()` 签名、SM/CTA/Warp 三层调用链、WarpContext public API 签名
- MUST NOT 引入新 `Wbar` struct（lessons-learned §14 — 已全部迁移至 BarrierModule）
- SHOULD 复用 `BarrierModule` public API（不引入新 Wbar struct）
- SHOULD 在 Phase 5 走 fold-back path if SM barrier glue 残留 < 50 行（避免过度工程）