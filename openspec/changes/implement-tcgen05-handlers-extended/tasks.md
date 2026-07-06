# Tasks: Implement 6 Extended tcgen05 Handlers

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec
> **前置 changes**(必须 archive): Change-1, 2, 3a, 3b
> **范围**: 7 atomic commits(per handler 独立)

## 0. Pre-Implementation Review

- [ ] 0.1 跑 Metis 验证(详见 [proposal.md](proposal.md) Section "实施前必跑")
- [ ] 0.2 基线 worktree:`.worktrees/baseline-tcgen05-handlers-extended`

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `git checkout -b feat/implement-tcgen05-handlers-extended`
- [ ] 1.2 `git add openspec/changes/implement-tcgen05-handlers-extended/`
- [ ] 1.3 `git commit -m "docs(openspec): add implement-tcgen05-handlers-extended artifacts (ADR-0016)"`

## 2. Phase 1: alloc/dealloc/relinquish(commit 2)

- [ ] 2.1 新建 `src/ptxsim/instructions/tcgen05_alloc.cpp`(~150 LoC)
- [ ] 2.2 实施 `processTcgen05Alloc(context, ...)`:
  - [ ] 从 `operand[0]` 读 smem_addr
  - [ ] 从 `qualifiers` 读 `num_cols`(via `TCGEN_CTA_GROUP` or `IMMEDIATE`)
  - [ ] 调 `cta->tmem().allocate(num_cols)`
- [ ] 2.3 实施 `processTcgen05Dealloc(context, ...)`:
  - [ ] 调 `cta->tmem().deallocate()`
- [ ] 2.4 实施 `processTcgen05Relinquish(context, ...)`:
  - [ ] 调 `warp->set_allocate_permit(false)`
- [ ] 2.5 `cmake --build build && ctest -L "unit;tcgen05" -V` 验证
- [ ] 2.6 `git commit -m "feat(handlers): implement tcgen05.alloc/dealloc/relinquish (ADR-0016)"`

## 3. Phase 2: fence(commit 3)

- [ ] 3.1 新建 `src/ptxsim/instructions/tcgen05_fence.cpp`(~100 LoC)
- [ ] 3.2 实施 `processTcgen05Fence(context, ...)`:
  - [ ] 解析 `::before_thread_sync` / `::after_thread_sync`
  - [ ] 调 `warp_scheduler->fence(before/after)`
- [ ] 3.3 `git commit -m "feat(handlers): implement tcgen05.fence (ADR-0016)"`

## 4. Phase 3: cp(commit 4)

- [ ] 4.1 新建 `src/ptxsim/instructions/tcgen05_cp.cpp`(~200 LoC)
- [ ] 4.2 实施 `processTcgen05Cp(context, ...)`:
  - [ ] 从 `operand[0]` 读 sdesc(共享内存描述符)
  - [ ] 从 `qualifiers` 读 shape (`.128x256b` / `.64x128b` / etc)
  - [ ] 调 `cta->smem().read(shape)` + `cta->tmem().write(slot, ...)`
- [ ] 4.3 `git commit -m "feat(handlers): implement tcgen05.cp (ADR-0016)"`

## 5. Phase 4: mma.ws(commit 5)

- [ ] 5.1 新建 `src/ptxsim/instructions/tcgen05_mma_ws.cpp`(~250 LoC)
- [ ] 5.2 实施 `processTcgen05MmaWs(context, ...)`:
  - [ ] 复用 Change-3b 的 `processTcgen05Mma` fragment 算术
  - [ ] 在 layout 上做 weight-stationary 转换
- [ ] 5.3 新建 `tests/ptx/reference/tcgen05_mma_ws_golden.h`(从 PTX ISA §9.7.16 规范)
- [ ] 5.4 `git commit -m "feat(handlers): implement tcgen05.mma.ws weight-stationary variant (ADR-0016)"`

## 6. Phase 5: 测试(commit 6)

- [ ] 6.1 注册 4 个新源文件到 `src/ptxsim/CMakeLists.txt`
- [ ] 6.2 新建 `tests/unit/ptx_ir/test_tcgen05_extended_opkind.cpp`(~80 LoC,测试 6 OpKind)
- [ ] 6.3 新建 `tests/integration/parser/test_tcgen05_extended_parse.cpp`(~150 LoC,测试 6 集成)
- [ ] 6.4 新建 `tests/e2e/kernel/test_tcgen05_alloc.cu`(~100 LoC)
- [ ] 6.5 新建 `tests/e2e/kernel/test_tcgen05_cp.cu`(~150 LoC)
- [ ] 6.6 注册到 `tests/unit/CMakeLists.txt` + `tests/integration/CMakeLists.txt` + `tests/e2e/CMakeLists.txt`
- [ ] 6.7 `ctest -L "unit;tcgen05|integration;tcgen05|e2e;tcgen05" -V` PASS
- [ ] 6.8 `git commit -m "test: add 6 extended tcgen05 handler tests (ADR-0016)"`

## 7. Phase 6: 文档(commit 7)

- [ ] 7.1 根 `AGENTS.md` 已知限制表:tcgen05 11/11 handler 已实现
- [ ] 7.2 `src/ptxsim/instructions/AGENTS.md`:tcgen05.cpp 包含 11 handler
- [ ] 7.3 ADR-0016 追加更新记录
- [ ] 7.4 `git commit -m "docs: update AGENTS + ADR for tcgen05 11/11 handler (ADR-0016)"`

## 8. Phase 7: Archive(commit 8,per Checklist G)

- [ ] 8.1 `openspec archive implement-tcgen05-handlers-extended --yes`
- [ ] 8.2 `ctest --output-on-failure` + `test_all_ptx.sh` 最终验证
- [ ] 8.3 `git add openspec/changes/archive/`
- [ ] 8.4 `git commit -m "chore(openspec): archive implement-tcgen05-handlers-extended (ADR-0016)"`

## Final Validation

- [ ] 9.1 `git log --oneline | head -8` 显示 8 atomic commits
- [ ] 9.2 跨 Change 协调:Change-4 (cleanup wmma) 可基于本 change 完整状态开始

## Risks Recap

| Risk | Mitigation |
|------|------------|
| R1: alloc/dealloc 越界 | Tmem.h `validate_slot_id` |
| R2: cp SMEM 越界 | SharedMemoryManager bounds check |
| R3: mma.ws fragment 错位 | 复用 mma + layout 转换 |
| R4: 6 commit 拆分过细 | per handler 独立 revert |
