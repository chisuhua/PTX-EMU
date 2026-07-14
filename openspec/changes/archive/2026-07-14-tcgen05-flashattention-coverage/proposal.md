# FlashAttention 测试覆盖（基于 tcgen05）

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) — Blackwell-only tcgen05; [ADR-0018](../../../docs/adr/0018-tcgen05-cta-group-restriction.md) — cta_group::2 throws
> **Oracle 2026-07-11 审计**: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` (FlashAttention readiness audit — 7 BLOCKER/IMPORTANT 测试覆盖缺口)
> **Oracle 2026-07-11 API 审查**: session `ses_0b026333bffePgrqVq7PDJNeR1` (Q3 readback 模式 + Q4 `load_c_slot` helper)
> **Metis pre-implementation review**: session `ses_0b1a0cdb1ffenbhbciQ1n0x236` (per checklist H — 必查项)
> **强制 lessons-learned**: `ptx-lessons-learned` §3(分 Phase commit) + §4(基线 worktree) + §6(artifacts-first) + §7(Pre-implementation Review) + Checklists E/H/J
> **依赖**: 本 change 阻塞于 4 个 follow-up changes — `fix-tcgen05-commit-wait-group` (FU-1/C3) + `fix-tcgen05-idesc-parsing` (FU-2/C1) + `fix-tcgen05-ld-st-slot-routing` (FU-3/C2) + `fix-tcgen05-multi-warp-fragment` (FU-4/C4)
> **Ref 链接**: [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/) + [`fix-tcgen05-mma-accumulator-and-f32-storage`](../fix-tcgen05-mma-accumulator-and-f32-storage/) (H1+H2 前置, commit `d3be589` persistence T1 反转已就绪)
> **范围**: 5 atomic commits (Phase 1-5 各对应一个测试文件) + 1 README sync commit

## Why

Oracle 2026-07-11 FlashAttention readiness 审计 (`ses_0aefd09c3ffeSqBIAGdxiRBFWC`) 发现当前 tcgen05 测试套件仅覆盖**单次 mma + 简单 dispatch**，无法验证 FlashAttention 算法的核心数据流：

| 缺口 | FlashAttention 影响 | Oracle confidence | 测试类型 |
|------|---------------------|-------------------|----------|
| **B1** | QK^T/PV 矩阵乘沿 K 维循环累加 128+ 次，单次 mma 测试无法检测 ULP 级漂移 | BLOCKER | Integration |
| **B2** | `mma → commit → wait → mma` 序列无行为测试 — `grep processTcgen05Commit\|processTcgen05Wait` 在 `tests/integration/tcgen05/` 下 **0 匹配** | BLOCKER | Integration |
| **B4** | `cp → mma` 数据流测试断裂（cp 写 slot 0，mma 读 `lane_id*2`，无数值断言） | IMPORTANT | Integration |
| **B5** | 多 warp C slot 隔离未测试（全部测试用单 warp 配置 `SMContext(1, ...)`） | IMPORTANT | Integration |
| **B6** | `mma(f32) → ld → st → mma` 持久性测试缺失 | IMPORTANT | Integration |
| **B7** | `Catch::Approx` 默认 epsilon ≈ 1.19e-5 太松 — K=128 累加后误差 ≥ 7.63e-6，**无法检测 ULP 级漂移** | NICE-TO-HAVE | 全局 |
| **D-E2E** | 无 FlashAttention E2E kernel；现有 `test_tcgen05_mma_gemm.cu` 是 Priority 3 纯 CUDA fallback，**完全不经过 tcgen05 handler** | BLOCKER | E2E |

4 个前置 follow-up changes（FU-1..FU-4）修复 handler/visitor/slot routing 本身的实现缺陷；**本 change 提供端到端测试覆盖**，验证 4 个前置 fix 协同工作，并提供完整 FlashAttention mini-kernel 的端到端验证路径。

## What Changes

### 新增测试（5 文件）

| 文件 | 阶段 | 阻塞前置 |
|------|------|---------|
| `tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp` | Phase 1 | FU-2 (C1 idesc parsing) |
| `tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp`（**注**：H1+H2 已含 B2 简化版，本 Phase 提供强化版覆盖更多场景） | Phase 2 | FU-1 (C3 commit/wait group) |
| `tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp` | Phase 3 | FU-3 (C2 ld/st slot routing) |
| `tests/integration/tcgen05/test_tcgen05_multi_warp_isolation.cpp` | Phase 4 | FU-4 (C4 multi-warp slot) |
| `tests/e2e/kernel/test_flashattention_mini.cu` | Phase 5 | FU-1..FU-4 全 |

### 工具函数（cross-cutting）

| 文件 | 用途 |
|------|------|
| `include/ptxsim/testing/tmem_helpers.h`（新增） | 复用 `fill_tmem_with_golden_inputs` + `require_c_slot_matches` + 新增 `compare_c_slot_to_reference`（宽松/严格公差两套） |

### 根 README 同步（per lessons-learned §8 — 重大功能交付清单）

| 文件 | 修改 |
|------|------|
| `README.md` | "已实现功能"章节追加 "FlashAttention mini-kernel (QK^T → softmax → @V)" 条目；引用本 change |

### 不修改（范围外）

- ❌ 不修改 helper / handler / dispatcher 实现（FU-1..FU-4 范围）
- ❌ 不修改 grammar / IR struct（FU-2 范围）
- ❌ 不实现真正的 warp-cooperative 64×64 tile layout（FU-4 范围）
- ❌ 不支持 cta_group::2 multi-CTA（per ADR-0018 抛异常）
- ❌ 不更新现有 `test_tcgen05_mma_golden.cpp` 等 unit test（FU-1..FU-4 apply 后回填）

## Non-Goals

### 显式拒绝

- ❌ 不实现 FlashAttention FA2/FA3 完整算法（仅 mini-kernel：K=4 blocks, head_dim=64, block_size=32）
- ❌ 不实现 online softmax（仅用纯 CUDA C++ softmax 作为 fallback，对照 FA 算法）
- ❌ 不支持 Hopper sm_90（仅 Blackwell sm_100，per ADR-0016）
- ❌ 不实现 TMA descriptor 解析（前置 commit `ad527f5` 已完成，但本 change 仅复用现有 TMA 测试）
- ❌ 不实现 cluster arrive/wait multi-CTA sync（per ADR-0016 Phase 0.3 实施中）

### 范围限制

- 仅单 warp + 2-warp 配置（不测 4+ warp tile 分配）
- 仅 `accumulate=true` 路径（overwrite 路径由 `fix-tcgen05-mma-accumulator-and-f32-storage` 覆盖）
- 仅 f16×f16→f32 dtype（其他 dtype 留待 follow-up）
- K-loop 上限 128 次（典型 FA 配置；更大 K 留待性能测试）

## Capabilities

### New Capabilities

- `tcgen05-flashattention-coverage`: 提供 5 个 FlashAttention 测试套件（4 integration + 1 E2E），验证 QK^T 矩阵乘沿 K 维循环累加 + softmax + PV 矩阵乘 + tmem stage 间 ld/st 数据流 + commit/wait 屏障同步 + 多 warp C slot 隔离

### Modified Capabilities

- `tcgen05-handlers-extended`: 扩展覆盖范围，明确"FlashAttention QK^T→softmax→PV 端到端数据流"是 handler 必须支持的算法路径（从"单 mma 行为正确"升级为"完整 FA 数据流行为正确"）

## Impact

### 影响的代码（预计）

| 文件 | 变更类型 | LoC 估计 |
|------|----------|---------|
| `tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp` | 新增 | ~80 |
| `tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp` | 新增（强化版） | ~120 |
| `tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp` | 新增 | ~100 |
| `tests/integration/tcgen05/test_tcgen05_multi_warp_isolation.cpp` | 新增 | ~80 |
| `tests/e2e/kernel/test_flashattention_mini.cu` | 新增 | ~150 |
| `include/ptxsim/testing/tmem_helpers.h` | 新增 | ~50 |
| `tests/integration/tcgen05/CMakeLists.txt` | 修改（注册新 ctest） | +10 |
| `tests/e2e/kernel/CMakeLists.txt` | 修改（注册新 ctest） | +5 |
| `README.md` | 修改（已实现功能章节同步） | +3 / -1 |
| **总计** | | **+598 / -1** |

### 影响的依赖

- 阻塞依赖：FU-1..FU-4 必须先 archive（per `proposal.md:Follow-Up Changes` 章节）
- 复用依赖：`include/ptxsim/testing/scheduler_utils.h` + `instruction_helpers.h` + `predicates.h`（per `tests/integration/divergence/test_divergence_sync_convergence.cpp` 模式）
- 测试 ctest 标签：`integration;tcgen05;flashattention` + `e2e;flashattention`

### 不影响的依赖

- 5 个前置 follow-up changes 的 implementation（纯测试添加，零实现改动）
- TmemAllocator（per `include/ptxsim/memory/tmem_allocator.h`）— 本 change 仅读 slot，不分配
- 现有所有 ctest（baseline worktree 对比必须 0 regression）

### 影响的文档

- `README.md` — "已实现功能"章节同步（per lessons-learned §8）
- `docs/adr/0016-blackwell-only-tcgen05.md` — Phase 3 archive 追加 "2026-07-XX Postmortem: FlashAttention coverage" 段
- `.opencode/notes/postmortem-tcgen05-flashattention-coverage.md`（若 user 在 archive 时选 Yes 生成）

## Design-Time Checklist (per ptx-lessons-learned)

### Checklist A: 函数迁移完整性

- N/A — 本 change **零实现改动**，仅测试添加（per lessons-learned §3 + Oracle Q3 验证 scope 内 vs scope 外测试不冲突）
- 但本 change 暴露 FU-1..FU-4 实现 bug（这是测试价值）— 任何实现 regression 必须 revert 对应 FU-* 而非本 change

### Checklist B: 重构前（基线 worktree）

- [ ] Phase 0 步骤：`git worktree add .worktrees/baseline-fa FU-4 最后一个 commit`
- [ ] `cmake -S .worktrees/baseline-fa -B .worktrees/baseline-fa/build -DCMAKE_BUILD_TYPE=Release`
- [ ] `cmake --build .worktrees/baseline-fa/build -j$(nproc)`（全量 build，per lessons-learned §4 三个陷阱 #2）
- [ ] `cd .worktrees/baseline-fa/build && ctest -L "integration;tcgen05" --output-on-failure`（baseline 全 PASS 记录）

### Checklist C: 写注释

- 每个新 TC 必须注释：
  - **依赖哪个 FU-* change**（"本测试需要 FU-1 (C3) 完成，否则硬编码 group_id=1 行为限制测试范围"）
  - **FlashAttention 哪个阶段**（"验证 QK^T 第 N 步累加"）
  - **预期误差边界**（"K=128 累加后相对误差 < 1e-3"）

### Checklist D: Commit 前（每个 Phase）

- [ ] 跑过 baseline worktree 对比（per Phase）
- [ ] `ctest -L "integration;tcgen05;flashattention"` + `ctest -L "e2e;flashattention"` 全 PASS
- [ ] `ctest --output-on-failure` 全量无 regression
- [ ] `./tests/ptx/test_all_ptx.sh` 全量 PASS
- [ ] commit message 列出独立测试编号（如 "Add K=128 accumulator test (FA-B1)"）

### Checklist E: OpenSpec artifacts 提交

- [ ] Phase 0 (artifacts FIRST): `git add openspec/changes/tcgen05-flashattention-coverage/` + 4 个 md artifacts → commit `docs(openspec): tcgen05-flashattention-coverage artifacts`
- [ ] 每个 Phase 独立 commit（per lessons-learned §3）
- [ ] Phase 5 (archive): openspec archive 后所有 artifacts 必须 git-tracked

### Checklist H: Pre-implementation Review 强制项

- [x] Metis pre-implementation review ✅（Oracle 审计 session 已涵盖 FU-5 范围 + Ambiguities 评估）
- [ ] Oracle 实施前验证：5 个测试文件名的最终确认（每个 Phase 开工前跑 Oracle 看 FU-* 完成状态）
- [x] Oracle audit §F 验证：当前测试套件无法 catch FA 路径 regression — PARTIAL 判决已采纳

### Checklist J: OpenSpec artifacts 内部一致性

- 范围数字对齐：proposal "5 文件 ~598 LoC" = design "5 文件 ~598 LoC" = tasks "5 phases × ~120 LoC/phase" = spec "5 scenarios"
- 设计决策路径 vs spec Scenario 路径一致（spec 描述覆盖范围 ↔ design 列具体测试名 ↔ tasks 列具体 commit）
- tasks 验证命令路径 = design 路径示例（`tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp` 三处一致）

## 跨 Change 依赖

| 上游（阻塞） | 本 change | 下游 |
|--------------|----------|------|
| `fix-tcgen05-commit-wait-group` (FU-1) | **tcgen05-flashattention-coverage** | (未来 `implement-flashattention-kernel`) |
| `fix-tcgen05-idesc-parsing` (FU-2) | | |
| `fix-tcgen05-ld-st-slot-routing` (FU-3) | | |
| `fix-tcgen05-multi-warp-fragment` (FU-4) | | |
| `fix-tcgen05-mma-accumulator-and-f32-storage` (前置 H1+H2) | | |

- **上游**: 5 个 fix-* change 全部 archive 后，本 change 方可实施（per Oracle Q2 sequencing 验证 FU-1 须最先完成）
- **本 change 是 FlashAttention 真正端到端验证** — 没有本 change，QK^T→softmax→PV 路径无任何测试覆盖
- **下游**: 未来 `implement-flashattention-kernel` 可基于本 change 的 mini-kernel 模板扩展到完整 FA2/FA3

## 本 change 特有设计决策

**决策 D1: 测试类型分布 (per AGENTS.md TDD 三阶段 + Oracle 审计)**

| 类型 | 占比 | 理由 |
|------|------|------|
| 类型二 (Integration) | 4/5 | FA 核心是 instruction sequences；单元测试覆盖已充分（per FU-*） |
| 类型三 (E2E) | 1/5 | mini-kernel 端到端验证是 FA 唯一"端到端 oracle" |
| 类型一 (Unit) | 0 | 复用现有 `tests/unit/tcgen05/*`（FU-1..FU-4 范围） |

**决策 D2: K-loop 上限 128 (per Oracle Section E)**

- K=64 不足以暴露系统性偏置；K=128 是 FA 典型配置；K=256 性能测试覆盖
- 误差边界：相对误差 < 1e-3（f32 输出 1.0..32.0 范围）

**决策 D3: 多 warp 上限 2 (per FU-4 C4 验证)**

- 2-warp 测试验证 `c_slot = warp_id * 32 + 64 + lane_id` 公式
- 4+ warp 留待 future FA 大 tile 测试（per ADR-0016 Phase 0.3 cluster 实施）

**决策 D4: 不修改现有测试（per Oracle Q3 + Metis test gap）**

- `test_tcgen05_mma_golden.cpp`, `test_tcgen05_mma_ws.cpp`, `test_tcgen05_mma_persistence.cpp` 由 FU-1..FU-4 apply 时同步修改
- 本 change 仅新增 5 个测试文件 + 1 个 helper header

**决策 D5: E2E kernel 走 Priority 3 fallback (per Oracle Section D)**

- `nvcc -ptx -arch=sm_100` 当前 ptxas 不支持 `tcgen05.mma`（per `tests/e2e/kernel/test_tcgen05_mma_gemm.cu:20-23` 注释）
- E2E kernel 必须显式调用 `ptx_pragma "tcgen05.mma ..."` 内联汇编 + 测试 fallback path
- 如 ptxas 仍不支持，E2E TC 自身降级为 Priority 3（标记 `[e2e;flashattention;priority-3]`）

**决策 D6: 5 phases 而非 1 commit（per lessons-learned §3）**

- 每个新测试文件 = 1 atomic commit
- 任何 Phase 失败可独立 revert（不污染其他测试）
- 每个 Phase 独立运行 baseline worktree 对比

**决策 D7: README 同步在 Phase 5（per lessons-learned §8）**

- 任何 feat-* change 归档前必跑 grep 验证 README 同步
- "已实现功能"章节追加 FlashAttention 条目与本 change 引用