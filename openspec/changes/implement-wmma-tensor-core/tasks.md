# Tasks: Implement Blackwell tcgen05 (skip pre-Blackwell WMMA)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md)
> **前置依赖**: `replace-silent-stub-failures` (archived 2026-07-04)
> **总 Phase 数**: 4 (0 + 1 + 2 + 3) — Phase 0 是基础设施, Phase 1-3 是 tcgen05
> **Lessons-learned**: Checklist B (基线) + D (commit 前) + E (artifacts tracked)

---

## 0. Artifacts Tracking（必做！避免 lessons-learned §6 模式）

- [ ] 0.1 在 main 上创建分支：`git checkout -b feat/implement-blackwell-tcgen05`
- [ ] 0.2 `git add openspec/changes/implement-wmma-tensor-core/`
- [ ] 0.3 `git status` 验证 artifacts tracked（proposal / design / specs/* / tasks）
- [ ] 0.4 commit: `git commit -m "docs(openspec): update implement-wmma-tensor-core scope to Blackwell-only (ADR-0016)"`
- [ ] 0.5 `git ls-files openspec/changes/implement-wmma-tensor-core/` 验证非空
- [ ] 0.6 提交 ADR-0016（docs/adr/0016-blackwell-only-tcgen05.md）随本批 change

---

## Phase 0: 基础设施（TMA + TMEM + cluster + async queue）

> 9 commits (was 5), ~3000-5000 LoC, 全部在写任何 tcgen05 handler 之前。
> 每个子系统独立可 revert（ptx-lessons-learned §3）。
> Oracle review 修正：Phase 0.5 拆为 4 个微 commit（0.5.1~0.5.4）以修复
> 集成 commit 与独立 revert 原则的结构性冲突。

### 0.1 TMA descriptors（Fix #5）

- [ ] 0.1.1 建立基线 worktree（与 `proposal.md:161` 路径一致）：
      `git worktree add .worktrees/fix-pre-p0-baseline -b feat/implement-blackwell-tcgen05 main`
      > **Oracle review fix (Q1)**: 原 `../c5-impl` 与 `proposal.md` 不一致；
      > 统一为 `.worktrees/fix-pre-p0-baseline` 以保持 change 命名约定。
      > 路径名 `fix-pre-p0-baseline` 指"Phase 0 实施前的基线快照"。
- [ ] 0.1.2 验证基线：`.worktrees/fix-pre-p0-baseline` 下 `cmake -S . -B build && cmake --build build && cd build && ctest --output-on-failure`
- [ ] 0.1.3 阅读 NVIDIA PTX ISA §9.7.13 + cuobjdump 提取真实 TMA descriptor 字节
- [ ] 0.1.4 创建 `src/ptxsim/memory/tma_descriptor.h`：
      - `struct TmaDescriptor`（TensorMap header + swizzle + strides + dtype）
      - `class TmaDescriptorStore`（per-CTA descriptor 表）
      - `parse_descriptor_bytes(const void* bytes) -> TmaDescriptor`
- [ ] 0.1.5 创建 `src/ptxsim/memory/tma_descriptor.cpp`：实现解析逻辑
- [ ] 0.1.6 创建 `tests/unit/memory/test_tma_descriptor.cpp`：覆盖 ≥ 10 种 swizzle/stride 组合
- [ ] 0.1.7 在 `src/CMakeLists.txt` + `tests/unit/CMakeLists.txt` 注册
- [ ] 0.1.8 自检：`cmake --build build --target ptxsim && ctest -R "tma_descriptor"`
- [ ] 0.1.9 验证无回归：`ctest -L "unit;integration;e2e"`
- [ ] 0.1.10 commit: `git commit -m "feat(memory): TMA descriptor parser (Fix #5)"`
- [ ] 0.1.11 验证独立可 revert

### 0.2 Tensor Memory (TMEM)（Fix #6）

- [ ] 0.2.1 创建 `src/ptxsim/memory/tmem.h`：
      - `class Tmem`（256 slot × 128 byte = 32 KB per CTA）
      - `read(slot_id, bytes)`, `write(slot_id, bytes)`, `clear()`
- [ ] 0.2.2 创建 `src/ptxsim/memory/tmem.cpp`
- [ ] 0.2.3 创建 `tests/unit/memory/test_tmem.cpp`：验证容量 + 读写一致性
- [ ] 0.2.4 在 CMakeLists 注册
- [ ] 0.2.5 自检：`ctest -R "tmem"` + 全套回归
- [ ] 0.2.6 commit: `git commit -m "feat(memory): per-CTA Tensor Memory (TMEM) (Fix #6)"`
- [ ] 0.2.7 验证独立可 revert

### 0.3 cluster mode — arrive/wait only（Fix #7）

> **Oracle review simplification**: `cta_group::1` (Phase 1 target) does
> NOT need distributed shared memory — only `cta_group::2` (future) does.
> Phase 0.3 implements only `arrive`/`wait` primitives; distributed_smem
> deferred to when `cta_group::2` is actually needed.

- [ ] 0.3.1 创建 `src/ptxsim/cluster/cluster_context.h`：
      - `class ClusterContext`（1-8 CTA 集群标识）
      - `cta_cluster_arrive()` / `cta_cluster_wait()` 同步原语
      - **Deferred**: `distributed_smem` view（when cta_group::2 needed）
- [ ] 0.3.2 创建 `src/ptxsim/cluster/cluster_context.cpp`
- [ ] 0.3.3 创建 `tests/unit/cluster/test_cluster_mode.cpp`：验证 arrive/wait 同步
- [ ] 0.3.4 CMakeLists 注册（含新建 `tests/unit/cluster/` 目录模板）
- [ ] 0.3.5 自检：`ctest -R "cluster"` + 全套回归
- [ ] 0.3.6 commit: `git commit -m "feat(sim): cluster arrive/wait primitives (Fix #7, simplified—no distributed smem)"`

### 0.4 async tensor core queue（Fix #8）

- [ ] 0.4.1 创建 `src/ptxsim/async/tc_queue.h`：
      - `class TcQueue`（per-CTA 命令队列）
      - `commit(group_id)` → counter++
      - `wait(group_id)` → 阻塞直到 commit_group_counter >= group_id
      - `enqueue_mma(...)` 抽象
- [ ] 0.4.2 创建 `src/ptxsim/async/tc_queue.cpp`
- [ ] 0.4.3 创建 `tests/unit/async/test_tc_queue.cpp`：commit-group 顺序 + wait-aware 调度
- [ ] 0.4.4 CMakeLists 注册（含新建 `tests/unit/async/` 目录模板）
- [ ] 0.4.5 **关键审计**：使用 `state-modification-audit` skill 检查
      `commit_group_counter` 的所有读写点（`ptx-lessons-learned` §1）
- [ ] 0.4.6 自检：`ctest -R "tc_queue"` + 全套回归
- [ ] 0.4.7 commit: `git commit -m "feat(async): tc_queue commit-group + wait-aware scheduling (Fix #8)"`
- [ ] 0.4.8 验证独立可 revert

### 0.5 逐子系统集成到 CTAContext（Fix #9a, #9b, #9c, #9d）

> **Design rationale (Oracle review, 2026-07)**: 原设计将 4 个子系统在一个 commit
> 中集成。Oracle 指出这违反 `ptx-lessons-learned` §3（独立可 revert）——revert
> 后留下 4 个未引用子系统死代码。修复：拆为 4 个微 commit，每个只集成一个子系统。
>
> **Oracle review fix (Q2, 2026-07-04)**: 微 commit **不可独立 revert**。
> 原因：`TcQueue::enqueue_mma()` 写入 TMEM slot，集成到 CTAContext 后
> 0.5.4 (TcQueue) 隐式依赖 0.5.2 (TMEM) 和 0.5.1 (TMA) — 类比 `cta_context.h:112`
> `BarrierModule` 模式但跨子系统引用破坏独立性。
>
> **Revert 单元 = 整个 Phase 0.5（4 commits 整体回退至 0.4.7 后状态）**。
> 单个微 commit revert 会导致编译失败（未解析的 CTAContext 引用）。
> 失败处理：任何 Phase 0.5 子系统 bug → `git revert <0.5.1-commit>..<0.5.4-commit>`
> 整体回退，不单独 revert。

#### 0.5.1 TMA descriptors → CTAContext（Fix #9a）

- [ ] 0.5.1.1 修改 `src/ptxsim/core/cta_context.{h,cpp}`：添加 `tma_descriptor_store` 引用
- [ ] 0.5.1.2 创建 `tests/integration/tma/test_tma_with_cta_context.cpp`：验证 CTAContext.tma 与独立 TmaDescriptorStore 行为一致
- [ ] 0.5.1.3 自检：`ctest -R "tma.*cta"` + `ctest -L "unit;integration;e2e"` 全套回归
- [ ] 0.5.1.4 commit: `git commit -m "feat(sim): integrate TMA descriptor store with CTAContext (Fix #9a)"`
- [ ] 0.5.1.5 验证独立可 revert（revert 后 CTAContext 不引用 TMA，其余 3 子系统不变）

#### 0.5.2 TMEM → CTAContext（Fix #9b）

- [ ] 0.5.2.1 修改 `src/ptxsim/core/cta_context.{h,cpp}`：添加 `tmem` 引用
- [ ] 0.5.2.2 创建 `tests/integration/tmem/test_tmem_with_cta_context.cpp`：验证 CTAContext.tmem 隔离性
- [ ] 0.5.2.3 自检：`ctest -R "tmem.*cta"` + 全套回归
- [ ] 0.5.2.4 commit: `git commit -m "feat(sim): integrate TMEM with CTAContext (Fix #9b)"`

#### 0.5.3 cluster → CTAContext（Fix #9c）

- [ ] 0.5.3.1 修改 `src/ptxsim/core/cta_context.{h,cpp}`：添加 `cluster_context` 引用
- [ ] 0.5.3.2 创建 `tests/integration/cluster/test_cluster_with_cta_context.cpp`：验证 arrive/wait 同步
- [ ] 0.5.3.3 自检：`ctest -R "cluster.*cta"` + 全套回归
- [ ] 0.5.3.4 commit: `git commit -m "feat(sim): integrate cluster context with CTAContext (Fix #9c)"`

#### 0.5.4 TcQueue → CTAContext（Fix #9d）

- [ ] 0.5.4.1 修改 `src/ptxsim/core/cta_context.{h,cpp}`：添加 `tc_queue` 引用
- [ ] 0.5.4.2 创建 `tests/integration/async/test_tc_queue_with_cta_context.cpp`：验证 commit-group 顺序性
- [ ] 0.5.4.3 自检：`ctest -R "tc_queue.*cta"` + 全套回归
- [ ] 0.5.4.4 commit: `git commit -m "feat(sim): integrate TcQueue with CTAContext (Fix #9d)"`

---

## Phase 1: tcgen05.mma fragment arithmetic（Fix #10, #11）

### 1.1 实现 tcgen05.mma fragment arithmetic

- [ ] 1.1.1 阅读 `tensor.cpp` 当前实现（throw 异常），列出所有 set_state/commit_pc 调用
      （`ptx-lessons-learned` Checklist A）
- [ ] 1.1.2 解析 `tcgen05.mma.cta_group::1.kind::f16` 指令变体（qualifiers 处理）
- [ ] 1.1.3 实现真实 fragment arithmetic（m64nNk 等）：
      - 从 TMEM 读 A / B fragments
      - 复用 `include/ptxsim/utils/half_utils.h::f16_to_f32`
      - 8x4 输出片段写入 TMEM（保留 Blackwell fragment layout）
      - 委托给 `TcQueue::enqueue_mma`
      - **Oracle review fix (Q4)**：每个输出 fragment 元素（32 lane × 8x4 矩阵 = 256 元素）
        必须在 `wmma.cpp` 中添加 `// UNVERIFIED-AGAINST-HARDWARE` 注释，标注：
        - `lane_idx → (row, col)` 映射
        - PTX ISA §9.7.13 章节行号引用（必须人工对照 latest 规范）
      - 单元测试 + 集成测试 PASS
- [ ] 1.1.4 验证 divergent warp 行为（per design.md Decision 5：tcgen05 不在 fetch 时 throw，
      wait 时由 TcQueue 处理）
- [ ] 1.1.5 commit: `git commit -m "feat(wmma): implement tcgen05.mma fragment arithmetic (Fix #10)"`

### 1.2 集成测试

> **Oracle review fix**: 原描述 "验证 uniform warp + mma + commit + wait 序列"
> 但 commit/wait 在 Phase 2.2 才实现。修正为直接读 TMEM slot 验证 mma 结果。

- [ ] 1.2.1 创建 `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp`：
      - 使用 `execute_warp_instruction` 驱动
      - 验证 uniform warp 执行 mma 后 TMEM slot 值正确（直接读 TMEM，不经过 commit/wait）
      - 验证 32 lane 输出片段元素正确
- [ ] 1.2.2 在 `tests/integration/CMakeLists.txt` 注册（含新建 `tests/integration/tcgen05/` 目录模板）
- [ ] 1.2.3 自检：`ctest -R "tcgen05_mma_sync"` + 全套回归
- [ ] 1.2.4 commit: `git commit -m "test(wmma): integration test verifying tm09.mma writes correct TMEM slots (Fix #11)"`

---

## Phase 2: tcgen05.ld / st + commit / wait（Fix #12, #13）

### 2.1 tcgen05.ld / st 与 TMA + TMEM 集成

- [ ] 2.1.1 实现 `tcgen05.ld` 指令：TMA descriptor + TMEM 目标 slot
- [ ] 2.1.2 实现 `tcgen05.st` 指令：TMEM source + TMA descriptor
- [ ] 2.1.3 验证 descriptor 解析 + TMEM 读写一致性
- [ ] 2.1.4 commit: `git commit -m "feat(wmma): tcgen05.ld/st with TMA + TMEM integration (Fix #12)"`

### 2.2 tcgen05.commit / wait 异步流

- [ ] 2.2.1 实现 `tcgen05.commit` → TcQueue::commit(group_id)
- [ ] 2.2.2 实现 `tcgen05.wait` → TcQueue::wait(group_id)
- [ ] 2.2.3 集成测试：完整 mma 序列（ld → mma → commit → wait → st）
- [ ] 2.2.4 创建 `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`
- [ ] 2.2.5 自检：`ctest -R "tcgen05_ld_st_commit"` + 全套回归
- [ ] 2.2.6 commit: `git commit -m "feat(wmma): tcgen05.commit/wait async flow (Fix #13)"`

---

## Phase 3: e2e GEMM + AGENTS + spec publish（Fix #14）

- [ ] 3.1 创建 `tests/e2e/kernel/test_blackwell_gemm.cu`：
      - **Oracle review fix (Q5)**：原"cutlass 3.x 风格"描述不准确。
        cute/cutlass headers 完整 vendored 于 `bench/cute/include/`
        （`cute/arch/mma_sm100_*.hpp` + `cutlass/arch/mma_sm100.h` 已存在），
        但**无现有 e2e 测试使用**——Phase 3 将是首个使用 Cute headers 的 e2e。
      - **Cute tcgen05 风格** 16×16 GEMM kernel，target sm_100
      - 使用 vendored Cute headers (`bench/cute/include/`) — 在 `tests/e2e/kernel/CMakeLists.txt`
        添加 include path（参考 `bench/cute/CMakeLists.txt`）
      - 验证 fragment 算术正确：16×16 矩阵乘 C[i][j] = sum_k A[i][k] * B[k][j]
        host 端对比，f32 rounding tolerance
- [ ] 3.2 在 `tests/e2e/kernel/CMakeLists.txt` 注册
- [ ] 3.3 修改 `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS：
      移除 `tensor.cpp (WmmaHandler)` 异常说明，标注 Blackwell tcgen05 已实现
- [ ] 3.4 修改根 `AGENTS.md` 已知限制表：
      WMMA 条目从"抛异常" → "Blackwell tcgen05 已实现；pre-Blackwell 永久抛异常（ADR-0016）"
- [ ] 3.5 自检：`./scripts/sanity.sh --quick`
- [ ] 3.6 完整 sanity：`./scripts/sanity.sh`
- [ ] 3.7 PTX 语法测试：`./tests/ptx/test_all_ptx.sh`
- [ ] 3.8 与 baseline (`.worktrees/fix-pre-p0-baseline`) 对比无新增 FAIL
- [ ] 3.9 commit: `git commit -m "docs+test: e2e GEMM + AGENTS sync + spec publish (Fix #14)"`

---

## Phase 4: 最终验证 + 合并 + 归档

- [ ] 4.1 合并到 main：`git merge --no-ff feat/implement-blackwell-tcgen05`
- [ ] 4.2 验证 artifacts 在 main 已 tracked：
      ```bash
      git ls-files openspec/changes/implement-wmma-tensor-core/
      ```
- [ ] 4.3 清理 worktree：`git worktree remove ../c5-impl`
- [ ] 4.4 归档：`openspec archive "implement-wmma-tensor-core" --yes`
      （spec 自动 publish 到 `openspec/specs/wmma-tensor-core/`）

---

## 失败回滚速查

| 失败 Phase | 立即动作 |
|-----------|---------|
| Phase 0.1 (TMA) | `git revert HEAD` → 仍能 build，无 TC descriptor → 抛异常 |
| Phase 0.2 (TMEM) | `git revert HEAD` → 仍能 build，无 TC memory → 抛异常 |
| Phase 0.3 (cluster) | `git revert HEAD` → 必须跑 cluster 测试确认 CTAContext 未破坏 |
| Phase 0.4 (tc_queue) | `git revert HEAD` → **关键**：跑全套回归 |
| Phase 0.5.1~0.5.4 (逐子系统集成) | `git revert HEAD` → 仅从 CTAContext 移除该子系统引用；其余 3 个子系统 + 它们自己的 CTAContext 引用不受影响（Oracle fix: 原 0.5 revert 会留 4 个死代码子系统，修复后每个微 revert 只影响一个子系统） |
| Phase 1 (mma 实现) | `git revert HEAD` → 回到 throw-only |
| Phase 2 (ld/st+commit) | `git revert HEAD` → mma 仍工作，ld/st 抛异常 |
| Phase 3 (e2e + AGENTS) | `git revert HEAD` → 仅回滚测试和文档 |

---

## 关键约束（必读）

⚠️ **MUST**：
- Phase 0 子系统必须在 Phase 1 之前完成（每个独立可 revert）
- 复用 `include/ptxsim/utils/half_utils.h`，不重新实现 f16 ↔ f32
- 实施 commits 合并前先 `git add openspec/changes/<name>/`（避免
  lessons-learned §6 模式）
- Phase 0.4 (tc_queue) 必须先跑 `state-modification-audit` skill
- ADR-0016 决策不可绕过 — pre-Blackwell 不实现

⚠️ **MUST NOT**：
- 不要修改 `UnsupportedInstructionException` / `ExecutionStateException` 类定义
- 不要修改 X-Macro `ptx_op.def`（`S_WMMA` → `WmmaHandler` 不变）
- 不要破坏 cute_rmsnorm / cute_hello_* 等已通过的 E2E 测试
- 不要在 WMMA handler 里用 `qualifiers.back()`（lessons-learned §5）

---

## 未来 Phases（不在本 change 范围）

- **ADR-0017**：`cuda::tma::create_tensor_map` 拦截策略
- **ADR-0018**：cluster mode 的 distributed shared memory 模拟策略（如 Phase 0.3 不够）
- **ADR-0019**：async tensor core queue 与现有 WarpState 集成模式（如 Phase 0.4+0.5 不够）
- **sm_120 sparse / FP4 / mxfp8**：每个特性一个 change
- **mma.sp 稀疏变种**：Phase 3 之后单独 change
- **cute_rmsnorm 升级到 tcgen05**：blocked until Phase 0-2 done