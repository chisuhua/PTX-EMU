# Tasks: Blackwell tcgen05 Handler Implementation (Phase 1-3)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md)
> **前置 change**: `implement-wmma-tensor-core-phase-0-infra` (本批 archive 后)
> **后续**: spec publish (本 change archive 后) → cute_rmsnorm 升级等 follow-up
> **总 commits**: **5 commits** (1.1, 1.2, 2.1, 2.2, 3.1)
> **Lessons-learned**: Checklist A (函数迁移) + D (commit 前) + E (artifacts tracked) + G (OpenSpec lifecycle)

---

## Pre-conditions

⚠️ **前置 change `implement-wmma-tensor-core-phase-0-infra` 必须先 archive**, 否则:
- Phase 1 缺 TcQueue 框架（无法委托 commit/wait）
- Phase 2 缺 TMA + TMEM（无法 ld/st）
- Quality Gate P1-3.G1-3 无法验证

**前置 change archive 后的引用方式** (Checklist G):
```markdown
> **Ref**: archive/<YYYY-MM-DD>-implement-wmma-tensor-core-phase-0-infra/
```

---

## 0.0 Artifacts Tracking（必做 FIRST！）

- [ ] 0.0.1 **等待 phase-0-infra-archive merge 到 main**，然后基于 main 创建 phase-1-3 分支：
      `git checkout main && git checkout -b feat/implement-tcgen05-handlers`
      (前置条件：phase-0-infra-archive 已 merge，per tasks.md:4b.1)
- [ ] 0.0.2 `git add openspec/changes/implement-wmma-tensor-core-tcgen05/`
- [ ] 0.0.3 `git status` 验证 artifacts tracked (proposal / design / specs/wmma-tensor-core/spec / specs/stub-explicit-failure/spec / tasks)
- [ ] 0.0.4 commit: `git commit -m "docs(openspec): track implement-wmma-tensor-core-tcgen05 artifacts (Ref: archive/...phase-0-infra/)"`

---

## Phase 1: tcgen05.mma fragment arithmetic（Fix #10, #11）

### 1.1 实现 tcgen05.mma fragment arithmetic

- [ ] 1.1.1 阅读 `tensor.cpp` 当前实现（throw 异常），列出所有 set_state/commit_pc 调用
      (`ptx-lessons-learned` Checklist A)
- [ ] 1.1.2 git mv `src/ptxsim/instructions/tensor.cpp` → `src/ptxsim/instructions/wmma.cpp`
- [ ] 1.1.3 修改 `src/CMakeLists.txt:103` source path:
      `ptxsim/instructions/tensor.cpp` → `ptxsim/instructions/wmma.cpp`
- [ ] 1.1.4 解析 `tcgen05.mma.cta_group::1.kind::f16` 指令变体（qualifiers 处理）
- [ ] 1.1.5 实现真实 fragment arithmetic（m64nNk 等）：
      - 从 TMEM 读 A / B fragments
      - 复用 `include/ptxsim/utils/half_utils.h::f16_to_f32`
      - 8x4 输出片段写入 TMEM（保留 Blackwell fragment layout）
      - 委托给 `TcQueue::enqueue_mma`
      - **Oracle review fix (Q4)**：每个输出 fragment 元素（32 lane × 8x4 矩阵 = 256 元素）
        必须在 `wmma.cpp` 中添加 `// UNVERIFIED-AGAINST-HARDWARE` 注释，标注：
        - `lane_idx → (row, col)` 映射
        - PTX ISA §9.7.13 章节行号引用（必须人工对照 latest 规范）
- [ ] 1.1.6 验证 divergent warp 行为（per Decision 5：tcgen05 不在 fetch 时 throw，
      wait 时由 TcQueue 处理）
- [ ] 1.1.7 单元测试 PASS：`ctest -R "tcgen05_mma"`
- [ ] 1.1.8 commit: `git commit -m "feat(wmma): implement tcgen05.mma fragment arithmetic (Fix #10)"`
      commit message 必须含 `Fix #10`
- [ ] 1.1.9 验证独立可 revert

### 1.2 集成测试

> **Oracle review fix**: 原描述 "验证 uniform warp + mma + commit + wait 序列"
> 但 commit/wait 在 Phase 2.2 才实现。修正为直接读 TMEM slot 验证 mma 结果。

- [ ] 1.2.1 创建 `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp`：
      - 使用 `execute_warp_instruction` 驱动
      - 验证 uniform warp 执行 mma 后 TMEM slot 值正确（直接读 TMEM, 不经过 commit/wait）
      - 验证 32 lane 输出片段元素正确（≥ 256 fragment 元素全覆盖）
- [ ] 1.2.2 在 `tests/integration/CMakeLists.txt` 注册（含新建 `tests/integration/tcgen05/` 目录）
- [ ] 1.2.3 自检：`ctest -R "tcgen05_mma_sync"` + 全套回归
- [ ] 1.2.4 commit: `git commit -m "test(wmma): integration test verifying tcgen05.mma writes correct TMEM slots (Fix #11)"`
- [ ] 1.2.5 验证独立可 revert
- [ ] 1.2.6 **Quality Gate P1-3.G1 验证**：`grep -c "UNVERIFIED-AGAINST-HARDWARE" src/ptxsim/instructions/wmma.cpp` ≥ 256
- [ ] 1.2.7 **Quality Gate P1-3.G2 验证**：`ctest -R "tcgen05_mma_sync"` PASS

---

## Phase 2: tcgen05.ld / st + commit / wait（Fix #12, #13）

### 2.1 tcgen05.ld / st 与 TMA + TMEM 集成

- [ ] 2.1.1 实现 `tcgen05.ld` 指令：TMA descriptor + TMEM 目标 slot
- [ ] 2.1.2 实现 `tcgen05.st` 指令：TMEM source + TMA descriptor
- [ ] 2.1.3 验证 descriptor 解析 + TMEM 读写一致性
- [ ] 2.1.4 单元测试 PASS
- [ ] 2.1.5 commit: `git commit -m "feat(wmma): tcgen05.ld/st with TMA + TMEM integration (Fix #12)"`
- [ ] 2.1.6 验证独立可 revert

### 2.2 tcgen05.commit / wait 异步流

- [ ] 2.2.1 实现 `tcgen05.commit` → `TcQueue::commit(group_id)` 调用
- [ ] 2.2.2 实现 `tcgen05.wait` → `TcQueue::wait(group_id)` 调用
      (无新 state 翻译 — 复用 Phase 0-archive TcQueue BAR_SYNC path)
- [ ] 2.2.3 集成测试：完整 mma 序列（ld → mma → commit → wait → st）
- [ ] 2.2.4 创建 `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`
- [ ] 2.2.5 自检：`ctest -R "tcgen05_ld_st_commit"` + 全套回归
- [ ] 2.2.6 commit: `git commit -m "feat(wmma): tcgen05.commit/wait async flow (Fix #13)"`
- [ ] 2.2.7 验证独立可 revert
- [ ] 2.2.8 **Quality Gate P1-3.G3 验证**：`ctest -R "tcgen05_ld_st_commit"` PASS

---

## Phase 3: e2e GEMM + AGENTS + spec publish（Fix #14）

- [ ] 3.1 创建 `tests/e2e/kernel/test_blackwell_gemm.cu`：
      - **Cute tcgen05 风格** 16×16 GEMM kernel, target sm_100
      - 使用 vendored Cute headers (`bench/cute/include/`) — 在 `tests/e2e/kernel/CMakeLists.txt`
        添加 include path（参考 `bench/cute/CMakeLists.txt`）
      - 验证 fragment 算术正确：16×16 矩阵乘 `C[i][j] = sum_k A[i][k] * B[k][j]`,
        host 端对比, f32 rounding tolerance
- [ ] 3.2 在 `tests/e2e/kernel/CMakeLists.txt` 注册
- [ ] 3.3 修改 `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS：
      移除 `tensor.cpp (WmmaHandler)` 异常说明，标注 Blackwell tcgen05 已实现
- [ ] 3.4 修改根 `AGENTS.md` 已知限制表：
      WMMA 条目从"抛异常" → "Blackwell tcgen05 已实现；pre-Blackwell 永久抛异常（ADR-0016）"
- [ ] 3.5 自检：`./scripts/sanity.sh --quick`
- [ ] 3.6 完整 sanity：`./scripts/sanity.sh`
- [ ] 3.7 PTX 语法测试：`./tests/ptx/test_all_ptx.sh`
- [ ] 3.8 与 baseline 对比无新增 FAIL
- [ ] 3.9 commit: `git commit -m "docs+test: e2e GEMM + AGENTS sync + spec publish (Fix #14)"`
- [ ] 3.10 验证独立可 revert

---

## Phase 4: 最终验证 + 合并 + 归档

> **Phase 4 跨 phase-0-archive + phase-1-3-archive 两段**。当 phase-1-3 change 完成
> Phase 3 后，phase-0-archive + phase-1-3-archive 顺序执行。

### Phase 4a: phase-0-infra change archive

- [ ] 4a.1 合并 phase-0-infra change 到 main: `git merge --no-ff feat/implement-blackwell-tcgen05`
- [ ] 4a.2 验证 phase-0-infra artifacts 在 main 已 tracked:
      ```bash
      git ls-files openspec/changes/implement-wmma-tensor-core-phase-0-infra/
      ```
- [ ] 4a.3 清理 worktree：`git worktree remove .worktrees/fix-pre-p0-baseline` (Phase 0 baseline；与 tasks.md:0.1.1 创建路径一致)
- [ ] 4a.4 归档：`openspec archive "implement-wmma-tensor-core-phase-0-infra" --yes`

### Phase 4b: phase-1-3 change 实施

- [ ] 4b.1 创建 phase-1-3 分支：`git checkout -b feat/implement-tcgen05-handlers main`
- [ ] 4b.2 完成 Phase 1.0 + Phase 1 + Phase 2 + Phase 3（上述 tasks）

### Phase 4c: phase-1-3 change archive

- [ ] 4c.1 合并 phase-1-3 change 到 main: `git merge --no-ff feat/implement-tcgen05-handlers`
- [ ] 4c.2 验证 phase-1-3 artifacts 在 main 已 tracked:
      ```bash
      git ls-files openspec/changes/implement-wmma-tensor-core-tcgen05/
      ```
- [ ] 4c.3 归档：`openspec archive "implement-wmma-tensor-core-tcgen05" --yes`
      （spec 自动 publish 到 `openspec/specs/wmma-tensor-core/`，含 phase-0-archive 期间保留的 4 基础设施 MUST）
- [ ] 4c.4 验证 stub-explicit-failure delta 被同步到 main specs：
      ```bash
      git ls-files openspec/specs/stub-explicit-failure/
      # 应包含 blackwell-real-arithmetic scenarios
      ```

---

## 失败回滚速查

| 失败 Phase | 立即动作 |
|-----------|---------|
| Phase 1.1 (mma 实现) | `git revert HEAD` → 回到 throw-only (per `replace-silent-stub-failures` 合约) |
| Phase 1.2 (mma 集成测试) | `git revert HEAD` → 仅回滚测试，handler 不变 |
| Phase 2.1 (ld/st) | `git revert HEAD` → mma 仍工作, ld/st 抛异常 |
| Phase 2.2 (commit/wait) | `git revert HEAD` → mma/ld/st 仍工作, commit/wait 抛异常 |
| Phase 3.1 (e2e + AGENTS) | `git revert HEAD` → 仅回滚测试和文档 |

---

## 关键约束（必读）

⚠️ **MUST**：
- Phase 1.0 artifacts tracked **FIRST** (lessons-learned §6 / Checklist E)
- Phase 0 change (前置) 必须先 archive，否则本 change 无法实施
- 复用 `include/ptxsim/utils/half_utils.h`，不重新实现 f16 ↔ f32
- `grep -c "UNVERIFIED-AGAINST-HARDWARE" wmma.cpp` ≥ 256 (Gate P1-3.G1)
- PTX 语法测试必须 `./tests/ptx/test_all_ptx.sh`，**严禁** ctest 代替
- ADR-0016 决策不可绕过 — pre-Blackwell 不实现

⚠️ **MUST NOT**：
- 不要修改 `UnsupportedInstructionException` / `ExecutionStateException` 类定义
- 不要修改 X-Macro `ptx_op.def`（`S_WMMA` → `WmmaHandler` 不变）
- 不要破坏 cute_rmsnorm / cute_hello_* 等已通过的 E2E 测试
- 不要在 WMMA handler 里用 `qualifiers.back()` (lessons-learned §5)
- 不要 amend 已 archive 的 `implement-wmma-tensor-core-phase-0-infra` change
  (Checklist G "Archived = 终态")

---

## 未来 Phases（不在本 change 范围）

- **ADR-0017**：`cuda::tma::create_tensor_map` 拦截策略
- **ADR-0018**：cluster mode distributed_smem (when cta_group::2)
- **ADR-0019**：async queue 与 WarpState 集成 (if needed)
- **sm_120 sparse / FP4 / mxfp8**：每个特性一个 change
- **mma.sp 稀疏变种**：本 change 后单独 change
- **cute_rmsnorm 升级到 tcgen05**：本 change 后 follow-up issue
