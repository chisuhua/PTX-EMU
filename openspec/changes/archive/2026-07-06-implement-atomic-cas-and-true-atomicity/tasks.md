## 1. Artifacts Commit (Phase 0 — per lessons-learned Checklists E + H, 强制)

- [ ] 1.1 git add `openspec/changes/implement-atomic-cas-and-true-atomicity/` (4 artifacts: proposal.md + design.md + 2 specs/*.md)
- [ ] 1.2 git commit with message:
  ```
  docs(openspec): add implement-atomic-cas-and-true-atomicity artifacts (Phase 1 scope)

  Metis pre-impl review applied.
  Refs: docs/audits/debt-audit-2026-07-02.md §A-9 + §C-16
        docs/roadmap/post-phase3-debt-roadmap.md §3.2
        docs/dev-process/lessons-learned.md §2 + §5
  ```

## 2. Phase 1 - CAS Handler Implementation (~3h, Tier 1)

### 2.1 实证验证 (MR-1 + MR-2)

- [ ] 2.1.1 验证 `Q_CAS_ATOM` 在 `ptx_qualifier.def:251` 唯一匹配 (no Q_DOTCAS conflict)
- [ ] 2.1.2 验证 `ptx_visitor_atom.cpp:75-77` 循环 `for (i=2; i<min(size, opcount); ++i)` 正确收集 4-operand (opcount=3, operandCtxs.size()=3)
- [ ] 2.1.3 **MUST**: 创建最小 PTX 测试样本 `tests/ptx/atom_cas_basic.ptx`:
      ```ptx
      .version 7.0
      .target sm_70
      .address_size 64
      .global .b32 mem;
      .entry test() { .reg .b32 %r<4>; atom.global.cas.b32 %r0, [mem], %r1, %r2; ret; }
      ```
- [ ] 2.1.4 跑 `./tests/ptx/test_all_ptx.sh` 确认 sample 解析通过 (Q_CAS_ATOM 在 qualifiers 列表)

### 2.2 Handler 实现

- [ ] 2.2.1 修改 `src/ptxsim/instructions/atomic.h`: 新增 `processAtomicCAS` 静态方法签名
      ```cpp
      static void processAtomicCAS(
          ThreadContext* context, void* dst, void* addr,
          void* cmp_buffer, void* val_buffer,
          size_t data_size, MemorySpace space);
      ```
- [ ] 2.2.2 修改 `src/ptxsim/instructions/atomic.cpp`: line ~36 `atom_op` 检测循环添加 `case Qualifier::Q_CAS_ATOM:`
- [ ] 2.2.3 修改 `src/ptxsim/instructions/atomic.cpp`: 删除 line 55-58 "CAS is out-of-scope" 注释块
- [ ] 2.2.4 **MUST NOT**: 不在 atomic.cpp 引入 `lock_guard`/`unique_lock` (Phase 2 才有 mutex);不引入 `qualifiers.back()` 调用
- [ ] 2.2.5 实现 `processAtomicCAS` 函数体 (load → compare → conditional store → write-back dst),per design.md §2.2 算法
- [ ] 2.2.6 修改 dispatcher (`src/ptxsim/instructions/instruction_base.cpp` ~line 195-220): 在识别 `Q_CAS_ATOM` 时路由到 `processAtomicCAS` 而非 `processAtomicOperation`

### 2.3 测试类型一 (unit, 强制 per AGENTS.md)

- [ ] 2.3.1 创建 `tests/unit/atomic/test_cas_handler_basic.cpp`
- [ ] 2.3.2 测试用例 1: 单 lane CAS 成功 (`old=10, cmp=10, val=20` → dst=10, mem=20)
- [ ] 2.3.3 测试用例 2: 单 lane CAS 失败 (`old=10, cmp=5, val=20` → dst=10, mem=10)
- [ ] 2.3.4 测试用例 3: 4 个 data_size (.b8/.b16/.b32/.b64) 全覆盖
- [ ] 2.3.5 ctest target: `add_catch_test(unit_cas_handler test_cas_handler_basic.cpp)` in `tests/unit/atomic/CMakeLists.txt`
- [ ] 2.3.6 Label: `unit;atomic`

### 2.4 测试类型二 (integration, 强制 per AGENTS.md)

- [ ] 2.4.1 创建 `tests/integration/atomic/test_atom_global_cas.cpp`
- [ ] 2.4.2 使用 `ptxsim::testing::step_warp` + `make_atom_global_cas_u32()` helper
- [ ] 2.4.3 测试用例 1: 单 warp 32 lanes + all cmp-match → 验证 dst/memory 语义
- [ ] 2.4.4 测试用例 2: 单 warp 32 lanes + all cmp-mismatch → 验证 dst/memory 语义
- [ ] 2.4.5 测试用例 3: warp 内混合 cmp (前 16 match, 后 16 mismatch) → winner-takes-all 语义
- [ ] 2.4.6 **MUST NOT**: 测试中直接调用 `processAtomicCAS` 绕过调度器 (应通过 `step_warp` + PTX 序列驱动)
- [ ] 2.4.7 ctest target: `add_catch_test(integration_atom_global_cas test_atom_global_cas.cpp)` in `tests/integration/atomic/CMakeLists.txt`
- [ ] 2.4.8 Label: `integration;atomic`

### 2.5 PTX 语法测试 (per AGENTS.md + ptx-grammar-modification skill)

- [ ] 2.5.1 跑 `./tests/ptx/test_all_ptx.sh` 全量 — 必须全部通过
- [ ] 2.5.2 **MUST**: 修复任何因 CAS parser 触发的 ANTLR 解析错误 (Phase 1 scope 内,但 grammar 不修改 → 应无错误)

### 2.6 Phase 1 验证 Gate

- [ ] 2.6.1 **G1 (ctest unit)**: `cd build && ctest -L "unit;atomic" --output-on-failure` — 0 failed
- [ ] 2.6.2 **G2 (ctest integration)**: `ctest -L "integration;atomic" --output-on-failure` — 0 failed
- [ ] 2.6.3 **G3 (PTX syntax)**: `./tests/ptx/test_all_ptx.sh` — exit 0
- [ ] 2.6.4 **G4 (sanity quick)**: `./scripts/sanity.sh --quick` — 0 regression
- [ ] 2.6.5 **G5 (no new back() call)**: `grep -n "qualifiers.back()" src/ptxsim/instructions/atomic.cpp` — 0 变化
- [ ] 2.6.6 **G6 (no new lock)**: `grep -rn "lock_guard\|unique_lock" src/ptxsim/instructions/atomic.cpp` — 0 匹配
- [ ] 2.6.7 **G7 (docs sync)**: `src/ptxsim/instructions/AGENTS.md` 新增 "CAS handler" 章节

## 3. Phase 1 Commit (per lessons-learned #3 multi-Phase commit pattern)

- [ ] 3.1 git stage: 包含所有 .cpp/.h/CMakeLists.txt/test_*.cpp/.ptx + AGENTS.md 改动
- [ ] 3.2 git commit with message:
      ```
      refactor(atomic): implement CAS handler (Fix #1)

      Add atomic.cas and atomic.exch PTX instruction handlers.

      - New processAtomicCAS function (4-operand signature: dst, addr, cmp, val)
      - New case Qualifier::Q_CAS_ATOM in atom_op detection loop
      - Remove "CAS is out-of-scope" stub comment
      - 2 new tests (unit_cas_handler + integration_atom_global_cas)
      - 1 new PTX sample (tests/ptx/atom_cas_basic.ptx)

      Scope: Phase 1 of 3 (Tier 1, ~3h). Phase 2 (mutex) and
      Phase 3 (multi-warp oracle test) deferred to follow-up change.

      Refs: docs/audits/debt-audit-2026-07-02.md §A-9 + §C-16
            openspec/changes/implement-atomic-cas-and-true-atomicity/
            docs/roadmap/post-phase3-debt-roadmap.md §3.2

      Metis pre-impl audit applied (6 MUST-RESOLVE all addressed in design.md).
      ```

## 4. Revert Strategy (per lessons-learned §3 + Checklist B)

- [ ] 4.1 若 G1-G6 任一 gate 失败 → 立即 `git revert HEAD` (回滚单一 Phase 1 commit)
- [ ] 4.2 验证 revert 后状态: `ctest -L "atomic" --output-on-failure` 必须恢复到 pre-Phase 1 状态 (existing 2 atom tests PASS, new tests 不存在)
- [ ] 4.3 调查 root cause → 修复后在新 commit 中重新实施;不混入后续 Phase

## 5. Post-Phase 1 移交 (Phase 2/3 prep)

- [ ] 5.1 更新 `openspec/changes/implement-atomic-cas-and-true-atomicity/tasks.md` — 勾选完成 Phase 1 部分 (2.1-2.6, 3)
- [ ] 5.2 创建后续 change scaffold: `implement-atomic-true-atomicity-phase-2` (单独 OpenSpec change)
  - Phase 2: mutex 实现 + 死锁审计 (per design.md §3)
  - Phase 3: multi-warp oracle test (per design.md §3.3 + specs/atomic-true-atomicity/spec.md)
- [ ] 5.3 **Ref 链接**: 新 change proposal.md 头部添加:
      ```
      Refs: openspec/changes/implement-atomic-cas-and-true-atomicity/ (Phase 1)
      ```
- [ ] 5.4 更新 `docs/roadmap/post-phase3-debt-roadmap.md` §1.1 A-9 行 — 标记 "Phase 1 ✅, Phase 2/3 ⏸"
- [ ] 5.5 (可选) 触发 Skills "OpenSpec archive" 流程对 Phase 1 单独归档? — **不推荐** (整个 change 应作为 1 个单元 archive,3 Phase 完成后统一归档)

---

## 总时间估算

- Phase 1: ~3h (Tier 1 预算内)
- Phase 2 (后续会话): ~3h
- Phase 3 (后续会话): ~2h
- 总: 8h = post-phase3-debt-roadmap.md §3.2 A-9 estimate
