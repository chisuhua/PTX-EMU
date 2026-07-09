# Tasks: Implement 6 Extended tcgen05 Handlers

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec
> **前置 changes**(必须 archive): Change-1, 2, 3a, 3b
> **范围**: 6 atomic commits(per Phase 独立)
> **Oracle 决策**: 2026-07-08 采纳 7 关键问题 (Q1-A/Q2-A/Q3-A/Q4-B/Q5-C/Q6-B/Q7-A)
> **强制**: 递归锁审计(Phase 1)+ baseline worktree(per ptx-lessons-learned §4)

## 0. Pre-Implementation Review

- [ ] 0.1 Metis pre-implementation review ✅ (2026-07-08)
- [ ] 0.2 Oracle 决策建议 ✅ (2026-07-08, 7 关键问题已采纳)
- [ ] 0.3 验证 Change-3b 已 archive(5 core handler 已实施)
- [ ] 0.4 验证 Change-3a 已 archive(grammar 100%)
- [ ] 0.5 验证 Change-2 已 archive(infra ≥L2)
- [ ] 0.6 跑 `ctest -L "unit;tcgen05" -V` 确认 baseline
- [ ] 0.7 跑 `./tests/ptx/test_all_ptx.sh` 确认 13 fixtures PASS
- [ ] 0.8 **建立基线 worktree** (per ptx-lessons-learned §4):
  ```bash
  git worktree add .worktrees/baseline-tcgen05-extended <baseline-commit>
  cd .worktrees/baseline-tcgen05-extended
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  cd build && ctest -L tcgen05 --output-on-failure
  ```
- [ ] 0.9 `git checkout -b feat/implement-tcgen05-handlers-extended`

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `git add openspec/changes/implement-tcgen05-handlers-extended/`
- [ ] 1.2 `git commit -m "docs(openspec): add implement-tcgen05-handlers-extended artifacts (ADR-0016, Oracle 决策)"`

## 2. Phase 1: TmemAllocator + alloc/dealloc/relinquish(commit 2)

### 2.1 递归锁审计(必做,per ptx-lessons-learned §2)

- [ ] 2.1.1 读 `tmem.h:46-47` 注释,确认 `mutable std::mutex mu_` 存在
- [ ] 2.1.2 读 `cluster_context.h:50` 确认 `mutable std::mutex mu_` 存在
- [ ] 2.1.3 计划 TmemAllocator 公共方法列表,标注每个方法是否持锁
- [ ] 2.1.4 验证所有"持锁方法调用的其他方法"不持同一锁
- [ ] 2.1.5 Falsification: 写多线程并发 alloc/dealloc 单元测试,确认不死锁

### 2.2 实现 TmemAllocator

- [ ] 2.2.1 新建 `include/ptxsim/memory/tmem_allocator.h`(Q1-A 新抽象层)
  - 公共方法: `allocate(num_cols) -> slot_id`, `deallocate(slot_id)`, `query(slot_id) -> bytes`
  - 内部状态: `std::bitset<256>` 跟踪分配状态,`std::mutex mu_`
- [ ] 2.2.2 新建 `src/ptxsim/memory/tmem_allocator.cpp`
- [ ] 2.2.3 在 `CTAContext::tmem_allocator()` 暴露访问器(per `cta_context.h:101-126`)

### 2.3 实现 3 个 handler

- [ ] 2.3.1 新建 `src/ptxsim/instructions/tcgen05_alloc.cpp`(~150 LoC)
- [ ] 2.3.2 实施 `processTcgen05Alloc(context, ...)`:
  - [ ] 从 `operand[0]` 读 smem_addr
  - [ ] 从 `qualifiers` 读 `num_cols`(via `TCGEN_CTA_GROUP` or `IMMEDIATE`)
  - [ ] 调 `cta->tmem_allocator().allocate(num_cols)`
  - [ ] Q2-A: 检测 `cta_group::2`,抛清晰异常(包含 ADR-0018 引用)
- [ ] 2.3.3 实施 `processTcgen05Dealloc(context, ...)`:
  - [ ] 调 `cta->tmem_allocator().deallocate()`
- [ ] 2.3.4 实施 `processTcgen05Relinquish(context, ...)`:
  - [ ] 调 `warp->set_allocate_permit(false)`

### 2.4 验证

- [ ] 2.4.1 `cmake --build build` 编译通过
- [ ] 2.4.2 `ctest -R tcgen05_alloc -V` PASS
- [ ] 2.4.3 跑 `./tests/ptx/test_all_ptx.sh` 仍 PASS(13 fixtures)
- [ ] 2.4.4 **对比 baseline worktree** (per ptx-lessons-learned §4):
  ```bash
  cd /workspace/project/PTX-EMU/build && ctest -L tcgen05 --output-on-failure
  ```
- [ ] 2.4.5 失败处理:已有测试回归 → 立即 revert 该 commit,不混入后续

### 2.5 Commit

- [ ] 2.5.1 `git add src/ptxsim/memory/tmem_allocator.{h,cpp} src/ptxsim/instructions/tcgen05_alloc.cpp src/ptxsim/instructions/tcgen05.cpp src/ptxsim/cta_context.h`
- [ ] 2.5.2 `git commit -m "feat(handlers): TmemAllocator + tcgen05.alloc/dealloc/relinquish (ADR-0016, Oracle Q1-A/Q2-A)"`

## 3. Phase 2: cp handler(commit 3)

### 3.1 实现

- [ ] 3.1.1 新建 `src/ptxsim/instructions/tcgen05_cp.cpp`(~200 LoC)
- [ ] 3.1.2 实施 `processTcgen05Cp(context, ...)`:
  - [ ] 从 `operand[0]` 读 smem_addr (Q4-B 复用 smem 解析)
  - [ ] 从 `qualifiers` 读 shape (`.128x256b` / `.64x128b` / etc)
  - [ ] 调 `cta->smem().read(addr, bytes)` + `cta->tmem_allocator().write(slot, bytes)`
  - [ ] Q2-A: 检测 `.cta_group::2`,抛清晰异常(包含 ADR-0018 引用)
- [ ] 3.1.3 在 `tcgen05.cpp` 的 `processTcgen05Operation` switch 中添加 `case Tcgen05OpKind::CP`

### 3.2 验证

- [ ] 3.2.1 `cmake --build build && ctest -R tcgen05_cp -V` PASS
- [ ] 3.2.2 E2E `test_tcgen05_cp.cu` (后续 Phase 4 创建) 验证
- [ ] 3.2.3 对比 baseline worktree

### 3.3 Commit

- [ ] 3.3.1 `git commit -m "feat(handlers): implement tcgen05.cp smem→tmem (ADR-0016, Oracle Q4-B/Q2-A)"`

## 4. Phase 3: mma.ws handler(commit 4, Oracle 2026-07-08 A-path)

### 4.0 Pre-Phase 2.5 refactor(commit 3.5, behavior-preserving)

- [x] 4.0.1 抽 `tcgen05_fragment_mma_f16(Tmem&)` 到 `include/ptxsim/instructions/tcgen05_helpers.h`
- [x] 4.0.2 实现 `src/ptxsim/instructions/tcgen05_helpers.cpp` (60 LoC fragment arithmetic)
- [x] 4.0.3 重构 `processTcgen05Mma` 为 helper wrapper
- [x] 4.0.4 注册 `tcgen05_helpers.cpp` 到 `src/CMakeLists.txt`
- [x] 4.0.5 验证 183/183 ctest + 45/45 PTX 一致(behavior preserved)

### 4.1 Phase 3 实现(Oracle A-path,qualifier routing)

- [x] 4.1.1 `processTcgen05Mma` 内 scan `instr.qualifiers` 找 `Q_TCGEN_WS`(不再新建 tcgen05_mma_ws.cpp)
- [x] 4.1.2 Q3-A 范围检查: ws path 要求 `Q_F16` 必备,缺失则抛 `UnsupportedInstructionException`
- [x] 4.1.3 ws path 调 `tcgen05_fragment_mma_f16(cta->tmem())`(与 mma 共用 helper)
- [x] 4.1.4 dispatch 表:`case MMA_WS:` 与 `case MMA:` 共享 throw → 现在改为 shared call to `processTcgen05Mma`
- [x] 4.1.5 `case Tcgen05OpKind::FENCE` 仍 throw(Phase 4)

### 4.2 Golden Value

- [x] 4.2.1 复用 `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h` (handler 与 mma 共用算术,golden 数据相同)
- [x] 4.2.2 集成测试在 `// UNVERIFIED-AGAINST-HARDWARE` 注释下验证

### 4.3 验证

- [x] 4.3.1 `cmake --build build && ctest -R tcgen05_mma_ws -V` PASS (unit 7 + integration 3 cases)
- [x] 4.3.2 对比 baseline worktree: PTX 45/45 一致
- [x] 4.3.3 完整 ctest 195/195 PASS (含 e2e_tcgen05_mma_ws Priority 3 fallback)
- [x] 4.3.4 `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp` 移除 `MMA_WS` 出 deferred list (现在 routed)

### 4.4 Commit

- [x] 4.4.0 Phase 2.5 commit: `refactor(tcgen05): extract fragment_mma_f16 helper` (Oracle Q4-recommendation)
- [x] 4.4.1 Phase 3 commit: `feat(handlers): tcgen05.mma.ws via qualifier routing` (Oracle A-path, ADR-0016)
- [x] 4.4.2 spec.md/design.md 更新:在 spec.md Scenario 中改用 `Q_TCGEN_WS qualifier` + 在 design.md D3 area 加 "Phase 3 实施修订" 注释。注: `proposal.md` 仍保留 Oracle 原话 `.warpspecialized::1`(per lessons-learned §6 + Checklist G "已归档 change 不 amend")

## 5. Phase 4: fence + 混合测试(commit 5)

### 5.1 fence 实现 (Q6-B no-op marker)

- [ ] 5.1.1 新建 `src/ptxsim/instructions/tcgen05_fence.cpp`(~100 LoC)
- [ ] 5.1.2 实施 `processTcgen05Fence(context, ...)`:
  - [ ] 解析 `::before_thread_sync` / `::after_thread_sync`
  - [ ] 调 `warp->record_fence_position(before/after)` (扩展点)
  - [ ] **不调 membar,不触发内存屏障**
- [ ] 5.1.3 在 `tcgen05.cpp` 的 `processTcgen05Operation` switch 中添加 `case Tcgen05OpKind::FENCE`

### 5.2 混合测试策略 (Q5-C)

- [ ] 5.2.1 注册 4 个新源文件到 `src/ptxsim/CMakeLists.txt`
- [ ] 5.2.2 **Unit** (手算 golden): `tests/unit/ptx_ir/test_tcgen05_extended_opkind.cpp`(~80 LoC,6 OpKind)
- [ ] 5.2.3 **Integration**: `tests/integration/parser/test_tcgen05_extended_parse.cpp`(~150 LoC,6 集成)
- [ ] 5.2.4 **E2E (尝试 nvcc)**: `tests/e2e/kernel/test_tcgen05_alloc.cu`(~100 LoC)
- [ ] 5.2.5 **E2E (尝试 nvcc)**: `tests/e2e/kernel/test_tcgen05_cp.cu`(~150 LoC)
- [ ] 5.2.6 注册到 `tests/unit/CMakeLists.txt` + `tests/integration/CMakeLists.txt` + `tests/e2e/CMakeLists.txt`
- [ ] 5.2.7 `ctest -L "unit;tcgen05|integration;tcgen05|e2e;tcgen05" -V` PASS

### 5.3 验证

- [ ] 5.3.1 对比 baseline worktree
- [ ] 5.3.2 跑 `./tests/ptx/test_all_ptx.sh` 仍 PASS(13 fixtures)

### 5.4 Commit

- [ ] 5.4.1 `git commit -m "feat(handlers): fence no-op + mixed oracle tests for 6 extended tcgen05 (ADR-0016, Oracle Q5-C/Q6-B)"`

## 6. Phase 5: 文档同步(commit 6, Q7-A)

- [ ] 6.1 根 `AGENTS.md` 已知限制表:tcgen05 11/11 handler 已实现
- [ ] 6.2 `src/ptxsim/instructions/AGENTS.md`:`tcgen05.cpp` 包含 11 handler
- [ ] 6.3 `docs/ptx/README.md` 状态表更新
- [ ] 6.4 ADR-0016 追加更新记录(本 change archive commit 引用)
- [ ] 6.5 `git commit -m "docs: update AGENTS + ADR for tcgen05 11/11 handler (ADR-0016, Oracle Q7-A)"`

## 7. Phase 6: Archive(commit 7, per Checklist G)

- [ ] 7.1 `openspec archive implement-tcgen05-handlers-extended --yes`
- [ ] 7.2 `ctest --output-on-failure` 全量验证
- [ ] 7.3 `./tests/ptx/test_all_ptx.sh` 验证
- [ ] 7.4 `git add openspec/changes/archive/`
- [ ] 7.5 `git commit -m "chore(openspec): archive implement-tcgen05-handlers-extended (ADR-0016)"`

## 8. Final Validation

- [ ] 8.1 `git log --oneline | head -8` 显示 7 atomic commits
- [ ] 8.2 跨 Change 协调:Change-4 (cleanup wmma) 可基于本 change 完整状态开始
- [ ] 8.3 清理 baseline worktree:
  ```bash
  git worktree remove .worktrees/baseline-tcgen05-extended
  ```

## Risks Recap

| Risk | 等级 | 缓解 |
|------|------|------|
| R1: alloc/dealloc 越界 | 中 | TmemAllocator + `tmem.h:35` `validate_slot_id` |
| R2: cp SMEM 越界 | 中 | SharedMemoryManager bounds check |
| R3: mma.ws fragment 错位 | 中 | 复用 mma + layout 转换 + golden 标记 UNVERIFIED |
| **R4: 递归锁死锁** | **高** | Phase 1 必做 `grep` 审计 + 多线程测试 |
| R5: cta_group::2 误用 | 低 | 清晰异常 + 文档说明 |
| R6: mma.ws 范围扩大 | 中 | 显式异常,其他 collector 模式 reject |
| R7: 6 commit 拆分过细 | 低 | per Phase 独立 revert |
