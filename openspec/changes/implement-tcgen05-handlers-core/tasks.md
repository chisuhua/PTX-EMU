# Tasks: Implement 5 Core tcgen05 Handlers

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec in [specs/](specs/)
> **前置 changes**(必须 archive): fix-tcgen05-grammar-mr3 + extend-blackwell-tcgen05-infra
> **范围**: 5 atomic commits,每步独立可 revert
> **Lessons-learned**: Checklist E + G + I(重大功能交付清单)

## 0. Pre-Implementation Review

- [ ] 0.1 跑 Metis 验证:
  - [ ] 0.1.1 `wc -l src/ptxsim/instructions/wmma.cpp`(约 564,实际含 4 个 `execute_tcgen05_*` + 1 个 inline mma handler)
  - [ ] 0.1.2 验证 fix-tcgen05-grammar-mr3 已 archive
  - [ ] 0.1.3 验证 extend-blackwell-tcgen05-infra 已 archive
  - [ ] 0.1.4 跑 `ctest -L "unit;tcgen05|integration;tcgen05" -V` 确认 baseline
  - [ ] 0.1.5 ✅ **已确认**:Cutlass 3.x 在 /usr/local /opt ~/cutlass 不可用 → **采用 PTX ISA §9.7.16 手算 + 复用 `wmma.cpp:374-420` 现有 inline mma fragment arithmetic**(per design.md D7)

- [ ] 0.2 基线 worktree:`.worktrees/baseline-tcgen05-handlers-core`

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `git checkout -b feat/implement-tcgen05-handlers-core`
- [ ] 1.2 `git add openspec/changes/implement-tcgen05-handlers-core/`
- [ ] 1.3 `git commit -m "docs(openspec): add implement-tcgen05-handlers-core artifacts (ADR-0016)"`

## 2. Phase 1: visitor operand 提取(commit 2)

- [ ] 2.1 编辑 `src/ptx_parser/ptx_visitor_wmma.cpp` 完善 `visitTcgen05Inst`
- [ ] 2.2 完整提取 qualifiers(用 `extractQualifiersFromContext`)
- [ ] 2.3 完整提取 operands(从 `tcgen05Operand` 规则提取,处理 vectorRegister/address/operand 三种情况)
- [ ] 2.4 `cmake --build build` 验证编译
- [ ] 2.5 `ctest -L "integration;tcgen05" -V` 验证
- [ ] 2.6 `git add src/ptx_parser/` + `git commit -m "fix(parser): complete visitTcgen05Inst operand extraction (ADR-0016)"`

## 3. Phase 2: 5 handler 实施(commit 3-4)

### 3.1 新建 tcgen05.cpp

- [ ] 3.1.1 新建 `src/ptxsim/instructions/tcgen05.cpp`
- [ ] 3.1.2 实施 `processTcgen05Mma(context, op, qualifiers, operands)`:
  - 32 lane × 8x4 fragment 算术(从 `wmma.cpp:374-420` 提取)
  - 读 TMEM slots 0-63(input a + b),写 TMEM slots 64-95(output c)
  - 验证 golden value(per Cutlass 3.x SM100_MMA_F16_F16_F32)
  - 每个 fragment element `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16`
- [ ] 3.1.3 实施 `processTcgen05Ld(context, op, qualifiers, operands)`:
  - 读 `cta->tma_descriptor_store().load(0)->global_address` 128 字节
  - 写 `cta->tmem().write(0, ...)` 128 字节
- [ ] 3.1.4 实施 `processTcgen05St(context, op, qualifiers, operands)`:
  - 读 `cta->tmem().read(0, ...)` 128 字节
  - 写 `cta->tma_descriptor_store().load(0)->global_address` 128 字节
- [ ] 3.1.5 实施 `processTcgen05Commit(context, op, qualifiers, operands)`:
  - `cta->tc_queue().commit(1)`
  - if cluster: `cta->cluster_context().cta_cluster_arrive(cta->blockIdx.x)`
- [ ] 3.1.6 实施 `processTcgen05Wait(context, op, qualifiers, operands)`:
  - `cta->tc_queue().wait(warp, 0, 1)`
  - if cluster: `cta->cluster_context().cta_cluster_wait(cta->blockIdx.x)`
- [ ] 3.1.7 新建 `tests/ptx/reference/tcgen05_mma_golden.h`(Cutlass 3.x 提取)

### 3.2 wmma.cpp 清理

> **2026-07-07 修正**:Day 1 验证发现 `wmma.cpp` 实际只有 **4 个 `execute_tcgen05_*`** 函数(`ld/st/commit/wait`)+ **1 个 inline mma handler**(line 352 `if (!is_tcgen05_mma_f16(qualifiers))` 走原路径)。**不是 5 个**。proposal 中"删除 5 个 execute_tcgen05_*"是错误描述。

- [ ] 3.2.1 读 `src/ptxsim/instructions/wmma.cpp` 当前 4 个 `execute_tcgen05_*`(ld:321/423, st:323/463, commit:325/502, wait:327/534) + 1 个 inline mma(line 352)
- [ ] 3.2.2 删除 4 个 `execute_tcgen05_*` 函数(ld/st/commit/wait)
- [ ] 3.2.3 删除 5 个 `is_tcgen05_*()` helper(line 29-56,所有 5 个,因为 `mma` helper 不再需要)
- [ ] 3.2.4 提取 inline mma handler 到 `tcgen05.cpp::processTcgen05Mma`(从 line 352 + line 374-420 提取 fragment arithmetic)
- [ ] 3.2.5 保留 pre-Blackwell `wmma.mma.sync.*` 路径(`UnsupportedInstructionException`)

### 3.3 注册新文件

- [ ] 3.3.1 编辑 `src/ptxsim/CMakeLists.txt` 注册 `tcgen05.cpp`
- [ ] 3.3.2 编辑 `src/ptxsim/instructions/AGENTS.md` 添加 `tcgen05.cpp` 说明

### 3.4 测试

> **2026-07-07 修正**:Day 1 验证发现 `tests/unit/ptx_ir/` 和 `tests/integration/parser/` 目录不存在,实际目录是 `tests/unit/ptx/` 和 `tests/integration/tcgen05/`。proposal 中路径已修正。

- [ ] 3.4.1 新建 `tests/unit/ptx/test_tcgen05_qualifier.cpp`(~50 LoC)
- [ ] 3.4.2 新建 `tests/unit/ptx/test_tcgen05_opkind.cpp`
- [ ] 3.4.3 新建 `tests/unit/ptx/test_tcgen05_dtype.cpp`
- [ ] 3.4.4 新建 `tests/unit/ptx/test_tcgen05_statement_factory.cpp`
- [ ] 3.4.5 新建 `tests/unit/ptx/test_tcgen05_instr_struct.cpp`
- [ ] 3.4.6 新建 `tests/integration/tcgen05/test_tcgen05_mma_parse.cpp`
- [ ] 3.4.7 新建 `tests/integration/tcgen05/test_tcgen05_ld_parse.cpp`(验证 num_regs)
- [ ] 3.4.8 新建 `tests/integration/tcgen05/test_tcgen05_st_parse.cpp`
- [ ] 3.4.9 新建 `tests/integration/tcgen05/test_tcgen05_commit_parse.cpp`(验证 mbarrier)
- [ ] 3.4.10 新建 `tests/integration/tcgen05/test_tcgen05_wait_parse.cpp`(验证 .load/.store)
- [ ] 3.4.11 编辑 `tests/unit/CMakeLists.txt` + `tests/integration/CMakeLists.txt` 注册 10 个新测试 + 标签 `unit;tcgen05` / `integration;tcgen05;grammar`

### 3.5 验证

- [ ] 3.5.1 `cmake --build build` 全量编译通过
- [ ] 3.5.2 `ctest -L "unit;tcgen05" -V` 5 单元测试 PASS
- [ ] 3.5.3 `ctest -L "integration;tcgen05" -V` 5 集成测试 PASS
- [ ] 3.5.4 `ctest --output-on-failure` 零回归
- [ ] 3.5.5 `./tests/ptx/test_all_ptx.sh` 13/13 fixtures PASS

### 3.6 Commit

- [ ] 3.6.1 `git add src/ptxsim/ tests/unit/ tests/integration/ tests/ptx/reference/`
- [ ] 3.6.2 `git commit -m "feat(handlers): implement 5 core tcgen05 handlers (mma/ld/st/commit/wait) + tests (ADR-0016)"`

## 4. Phase 3: E2E kernel(commit 5)

- [ ] 4.1 新建 `tests/e2e/kernel/test_tcgen05_mma_gemm.cu`(real-style tcgen05.mma GEMM)
- [ ] 4.2 注册到 `tests/e2e/CMakeLists.txt`
- [ ] 4.3 `ctest -L "e2e;tcgen05" -V` 验证 PASS
- [ ] 4.4 `git add tests/e2e/`
- [ ] 4.5 `git commit -m "test(e2e): add tcgen05.mma GEMM E2E kernel (ADR-0016)"`

## 5. Phase 4: 文档(commit 6)

- [ ] 5.1 根 `AGENTS.md` 已知限制表:tcgen05 5 core handler 已实现
- [ ] 5.2 `src/ptxsim/instructions/AGENTS.md`:添加 `tcgen05.cpp` 说明
- [ ] 5.3 `docs/adr/0016-blackwell-only-tcgen05.md` 追加更新记录(本 change commit 引用)
- [ ] 5.4 `git add AGENTS.md src/ptxsim/instructions/AGENTS.md docs/adr/`
- [ ] 5.5 `git commit -m "docs: update AGENTS + ADR for tcgen05 5 core handler (ADR-0016)"`

## 6. Phase 5: Archive(commit 7,per Checklist G)

- [ ] 6.1 `openspec archive implement-tcgen05-handlers-core --yes`
- [ ] 6.2 `ctest --output-on-failure` + `test_all_ptx.sh` 最终验证
- [ ] 6.3 `git add openspec/changes/archive/`
- [ ] 6.4 `git commit -m "chore(openspec): archive implement-tcgen05-handlers-core (ADR-0016)"`

## Final Validation

- [ ] 7.1 `git log --oneline | head -8` 显示 7 个 atomic commits
- [ ] 7.2 `openspec list` 确认 change 已 archive
- [ ] 7.3 跨 Change 协调:Change-3d 可基于本 change 的 5 handler 模式扩展 6 handler

## Risks Recap

| Risk | Mitigation |
|------|------------|
| R1: fragment arithmetic 与硬件不一致 | Golden value 来自 `wmma.cpp:374-420` inline mma + PTX ISA §9.7.16 手算(per design.md D1 修正) |
| R2: visitor 提取 qualifier 错误 | ctest -L "integration;tcgen05" 验证 |
| R3: wmma.cpp 删除后旧测试 fail | Phase 3 前先迁移(change-3a) |
| R4: E2E 真实 PTX 不可用 | Hand-written 真实风格 |
