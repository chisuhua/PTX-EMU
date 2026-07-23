# Tasks: Fix TmemAllocator Phase 1.x Critical Issues

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec
> **范围**: 1 atomic commit (修订性变更)
> **前置**: Phase 1 commit `486246a` 已存在

## 1. Fix Data Race (Oracle Critical #1)

- [ ] 1.1 修改 `src/ptxsim/memory/tmem_allocator.h`:
  - [ ] 在 read-only methods 前加注释: "持锁, 调用者无需持外锁"
  - [ ] 标注 `is_allocated_start`/`is_allocated`/`active_allocation_count`/`total_allocated_slots` 为 `const` (已是)
- [ ] 1.2 修改 `src/ptxsim/memory/tmem_allocator.cpp`:
  - [ ] `is_allocated_start`: 加 `std::lock_guard<std::mutex> lock(mu_);`
  - [ ] `is_allocated`: 加 `std::lock_guard<std::mutex> lock(mu_);`
  - [ ] `active_allocation_count`: 加 `std::lock_guard<std::mutex> lock(mu_);`
  - [ ] `total_allocated_slots`: 加 `std::lock_guard<std::mutex> lock(mu_);`
  - [ ] 删除错误的 "well-defined for each individual bit/key" 注释
  - [ ] 顶部加 `static_assert(TmemAllocator::kSlotCount == Tmem::kSlotCount, ...)`
- [ ] 1.3 验证单元测试仍通过 (`is_allocated_start` 持锁后行为一致)

## 2. Fix Multi-Threaded Deadlock Detection (Oracle #4)

- [ ] 2.1 修改 `tests/unit/memory/test_tmem_allocator.cpp:196-269`:
  - [ ] 把 `std::thread` 替换为 `std::async(std::launch::async, ...)`
  - [ ] 用 `future.wait_for(30s)` 替代 `th.join()`
  - [ ] 若 `wait_for` 返回 `future_status::timeout`, 主动 `REQUIRE(false, "deadlock suspected")`
  - [ ] 修正常规断言 (`elapsed < 30` 改为 `elapsed.count() < 30` 计算正确的耗时)

## 3. Fix kSlotCount Consistency (Oracle #6)

- [ ] 3.1 在 `src/ptxsim/memory/tmem_allocator.cpp` 顶部加:
  ```cpp
  static_assert(TmemAllocator::kSlotCount == Tmem::kSlotCount,
                "TmemAllocator must mirror Tmem slot count");
  ```
- [ ] 3.2 删除 `tmem_allocator.h:45-46` 关于 "kept as a separate constant to avoid a hard include dependency" 的注释 (static_assert 已强制一致, 无需注释解释)

## 4. Fix dealloc Comment (Oracle #3)

- [ ] 4.1 修改 `src/ptxsim/instructions/tcgen05_alloc.cpp:142-146`:
  - [ ] 注释改为: "Phase 1 simplification: releases the **lowest active slot_id** (first-fit, matches allocate() policy). Per-warp ownership tracking deferred to Phase 2."
  - [ ] 修正代码逻辑注释 `:162-167` (确认 "lowest slot_id" 而非 "most-recent")

## 5. Add 3 Handler Integration Tests (Oracle Critical #2)

- [ ] 5.1 新建 `tests/integration/tcgen05/test_alloc_dispatch.cpp`:
  - [ ] 构造 `SMContext` + `WarpContext` + `CTAContext`
  - [ ] 构造 `Tcgen05Instr{op_kind=Tcgen05OpKind::ALLOC, cta_group=1}`
  - [ ] 调 `execute_warp_instruction` 驱动 dispatch → `processTcgen05Alloc`
  - [ ] 验证 `cta->tmem_allocator().active_allocation_count() == 1`
  - [ ] 验证 `cta->tmem_allocator().is_allocated_start(0) == true`
- [ ] 5.2 新建 `tests/integration/tcgen05/test_dealloc_dispatch.cpp`:
  - [ ] alloc → dealloc round-trip
  - [ ] 验证 `active_allocation_count() == 0` after dealloc
- [ ] 5.3 新建 `tests/integration/tcgen05/test_relinquish_dispatch.cpp`:
  - [ ] relinquishing permit 后调 `processTcgen05Alloc` 期望 `runtime_error`
- [ ] 5.4 在 `tests/integration/tcgen05/CMakeLists.txt` 注册 3 个新测试:
  - [ ] `add_catch_test(integration_tcgen05_alloc_dispatch ...)`
  - [ ] `add_catch_test(integration_tcgen05_dealloc_dispatch ...)`
  - [ ] `add_catch_test(integration_tcgen05_relinquish_dispatch ...)`
  - [ ] 标签 `integration;tcgen05`

## 6. Sync AGENTS.md (Oracle Critical #3)

- [ ] 6.1 根 `AGENTS.md` "已知限制" 表: tcgen05 handler dispatch 行
  - [ ] 旧: "ALLOC/DEALLOC/RELINQUISH/CP/MMA_WS/FENCE 抛 UnsupportedInstructionException (deferred but wired, 见 implement-tcgen05-handlers-extended)"
  - [ ] 新: "CP/MMA_WS/FENCE 抛 UnsupportedInstructionException (deferred but wired, 见 implement-tcgen05-handlers-extended Phase 2/3/4)"
- [ ] 6.2 `src/ptxsim/instructions/AGENTS.md` "TCGEN05 HANDLER DISPATCH" 节:
  - [ ] 旧: "6 deferred op_kinds (ALLOC/DEALLOC/RELINQUISH/CP/MMA_WS/FENCE) throw"
  - [ ] 新: "3 deferred op_kinds (CP/MMA_WS/FENCE) throw; ALLOC/DEALLOC/RELINQUISH implemented (commit 486246a + Phase 1.x)"

## 7. Update ADR-0016 (Per Oracle Q7-A)

- [ ] 7.1 `docs/adr/ADR-0016-blackwell-only-tcgen05.md` 追加段落:
  - [ ] 标题: "2026-07-09: Phase 1 of implement-tcgen05-handlers-extended 完成 + Phase 1.x 修订"
  - [ ] 内容: 3/11 deferred → 0 deferred for alloc-family; cp/mma_ws/fence 待 Phase 2-4

## 8. Verification

- [ ] 8.1 `cmake --build build -j$(nproc)` 成功
- [ ] 8.2 `ctest -L "unit;tcgen05"` PASS (含 unit_tmem_allocator 12/12)
- [ ] 8.3 `ctest -R "integration_tcgen05"` PASS (含 3 个新增 + 5 个旧)
- [ ] 8.4 `./tests/ptx/test_all_ptx.sh` 45/45 PASS
- [ ] 8.5 对比 baseline worktree: PTX 仍 45/45 (per ptx-lessons-learned §4)

## 9. Commit + Archive

- [ ] 9.1 `git add -A`
- [ ] 9.2 `git commit -m "fix(tmem-allocator): Phase 1.x critical issues per Oracle 2026-07-09 review"`
- [ ] 9.3 `openspec archive fix-tmem-allocator-phase1x-critical --yes`
- [ ] 9.4 验证 archive 目录 commit

## Final Validation

- [ ] 10.1 `git log --oneline | head -3` 显示 1 个新 fix commit + 之前的 Phase 1 + artifacts
- [ ] 10.2 `openspec list --changes` 显示此 change 已 archive
- [ ] 10.3 无 regression: 全套 unit + integration 测试通过