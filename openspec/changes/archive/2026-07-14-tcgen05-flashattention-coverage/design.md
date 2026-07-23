## Context

### 现状问题

Oracle 2026-07-11 FlashAttention readiness 审计 (`ses_0aefd09c3ffeSqBIAGdxiRBFWC`) 揭示当前 tcgen05 测试套件的 7 个 BLOCKER/IMPORTANT 缺口（per `proposal.md` Why 章节表格）：

- **B2**: `grep "processTcgen05Commit\|processTcgen05Wait" tests/integration/tcgen05/` → 0 匹配
- **D-E2E**: 无 FlashAttention kernel；现有 GEMM 测试是 Priority 3 纯 CUDA fallback
- **B4**: `test_tcgen05_mma_persistence.cpp:250-294` 的 cp→mma 链 cp 写 slot 0，mma 读 `lane_id*2`，数据流断裂
- **B5**: 全部现有测试用 `SMContext(1, ...)` 单 warp 配置
- **B7**: `Catch::Approx` 默认 epsilon ≈ 1.19e-5 对 K=128 累加勉强通过

**关键架构事实**（已逐行验证）：

| 事实 | 文件:行 | 影响 |
|------|---------|------|
| `tcgen05.ld/st` 硬编码 slot 0 | `tcgen05.cpp:434, 476` | C2 — mma 写 [64..95] 与 ld/st slot 0 不连通 |
| `commit/wait` 硬编码 group_id=1 | `tcgen05.cpp:512, 550` | C3 — 无法区分 QK^T vs PV commit group |
| `c_slot = 64 + lane_id` 单 warp | `tcgen05_helpers.cpp:23` | C4 — 多 warp 时 warp 0/1 冲突 |
| `processTcgen05Mma` 显式 `accumulate=false` | `tcgen05.cpp:383` | C1 — 真实 PTX 累加路径永不执行 |
| `TcQueue` 已支持多 group | `tc_queue.h:53-55` | ✅ — 仅 handler 硬编码限制 |
| `Tmem::read/write(slot_id, ...)` 已支持任意 slot | `tmem.h:35-36` | ✅ — 仅 handler 硬编码限制 |

4 个前置 follow-up changes（FU-1..FU-4）按 Oracle Q2 sequencing 修复实现缺陷：
- **FU-1 (C3)** 必最先做（建立 IMMEDIATE 提取 pattern）
- **FU-2 (C1)** + **FU-3 (C2)** + **FU-4 (C4)** 与 FU-1 后并行

### 目标状态

5 个测试文件 + 1 个 helper header 共同构成 FlashAttention 端到端验证套件：

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: test_tcgen05_mma_k_loop_128.cpp                    │
│   └─ 验证 FU-2 (C1) 后 K=128 累加误差 < 1e-3                │
├─────────────────────────────────────────────────────────────┤
│ Phase 2: test_tcgen05_mma_commit_wait_sequence.cpp          │
│   └─ 验证 FU-1 (C3) 后多 commit group 区分 QK^T vs PV       │
├─────────────────────────────────────────────────────────────┤
│ Phase 3: test_tcgen05_mma_cp_data_flow.cpp                   │
│   └─ 验证 FU-3 (C2) 后 cp 写 mma 实际读的 slot              │
├─────────────────────────────────────────────────────────────┤
│ Phase 4: test_tcgen05_multi_warp_isolation.cpp               │
│   └─ 验证 FU-4 (C4) 后 2-warp C slot 不冲突                 │
├─────────────────────────────────────────────────────────────┤
│ Phase 5: tests/e2e/kernel/test_flashattention_mini.cu        │
│   └─ 完整 FA mini-kernel 端到端 (K=4 blocks, head_dim=64)    │
└─────────────────────────────────────────────────────────────┘
```

每 Phase 独立 commit + 独立 baseline worktree 对比（per lessons-learned §3+§4）。

### 利益相关方

- **tcgen05-flashattention-coverage 用户**：未来 `implement-flashattention-kernel` change 基于本套件
- **当前 fix-* change 实施者**：通过本套件回归验证 FU-1..FU-4 是否正确
- **debug-issue 用户**：本套件提供"已知正确"基准，加速未来 regression bisect

## Goals / Non-Goals

**Goals:**

1. 提供 5 个测试文件覆盖 FA 核心数据流（QK^T 累加、commit/wait 屏障、cp→mma 数据流、多 warp slot 隔离、E2E mini-kernel）
2. 提供 `tmem_helpers.h` 复用现有 `fill_tmem_with_golden_inputs` + `require_c_slot_matches` + 新增 `compare_c_slot_to_reference`（宽松/严格两套公差）
3. K=128 累加后相对误差 < 1e-3（per Oracle Section E 验证 f32 数值稳定性）
4. E2E mini-kernel 走 dispatcher 真实路径（即使 dispatcher 当前 broken，按 Priority 3 fallback 验证 fallback path 也产生 reference output）
5. 5 atomic commits + Phase 5 archive commit + README sync（per lessons-learned §3+§8）

**Non-Goals:**

- ❌ 不修改 helper / handler / dispatcher 实现（FU-1..FU-4 范围，本 change 零实现改动）
- ❌ 不实现 FA2/FA3 完整算法（仅 mini-kernel）
- ❌ 不实现 4+ warp tile 分配（FU-4 仅 2-warp）
- ❌ 不支持 cta_group::2（per ADR-0018）
- ❌ 不实现 TMA 解析（前置 commit `ad527f5` 已完成）
- ❌ 不修改 FU-1..FU-4 实施时已修改的现有测试文件

## Decisions

### D1: 测试类型分布（per AGENTS.md TDD + Oracle 审计）

**采纳**: 4 integration + 1 E2E + 0 unit

**理由**:
- FA 核心是 instruction sequences（unit 测试覆盖已充分于 FU-1..FU-4）
- E2E mini-kernel 是唯一端到端 oracle（无 E2E = 无 FA 正确性最终判定）
- unit 测试与 FU-* 重复（违反 DRY + 增加 maintenance 负担）

**拒绝的备选**:
- (a) 5 unit + 0 integration：unit 直接调 helper 无法验证 dispatcher + scheduler 路径
- (b) 5 E2E：构建时间爆炸（每 E2E 需 nvcc 编译 ~30s，5 个 = 2.5 分钟额外）
- (c) 2 unit + 2 integration + 1 E2E：unit 与 FU-* 范围重叠

### D2: K-loop 上限 128（per Oracle Section E）

**采纳**: `K_MAX = 128` iterations

**理由**:
- K=64 不足以暴露系统性偏置（误差 ≈ 3.8e-6 vs 1.19e-5 epsilon）
- K=128 是 FA 典型配置（per `bench/cute/include/cute/algorithm/*.hpp` 实测）
- K=256+ 性能测试覆盖（per `tests/integration/perf/` 未来扩展）

**Tradeoff**: 测试运行时间 K=128 ≈ 1-2s 单测试（K=256 → 4-5s 不可接受）

### D3: 多 warp 上限 2（per FU-4 C4 验证范围）

**采纳**: `NUM_WARPS = 2`

**理由**:
- 验证 `c_slot = warp_id * 32 + 64 + lane_id` 公式（per Oracle Q4 选项 a）
- 4+ warp 需 cluster arrive/wait 同步（per ADR-0016 Phase 0.3 实施中，未完成）

**Tradeoff**: 不覆盖 4-warp tile 分配（留待 cluster 实施后 follow-up）

### D4: 不修改现有测试（per Oracle Q3 + Metis test gap）

**采纳**: 本 change 仅新增 5 文件 + 1 header，零修改

**理由**:
- `test_tcgen05_mma_golden.cpp`, `test_tcgen05_mma_ws.cpp`, `test_tcgen05_mma_persistence.cpp` 由 FU-1..FU-4 apply 时同步修改（readback 改 `alignas(16) float[32]` + 公差收紧）
- 本 change 测试 FU-* 完成后的端到端正确性，不干预 FU-* 自己的测试调整

**拒绝的备选**:
- (a) 在本 change 中同步收紧所有现有测试公差 → scope creep + 与 FU-* 重复
- (b) 在本 change 中新增 helper function 到现有测试文件 → 制造 churn

### D5: E2E kernel 走 Priority 3 fallback（per Oracle Section D）

**采纳**: `test_flashattention_mini.cu` 显式调用内联 PTX + fallback path

**理由**:
- ptxas 不支持 sm_100 `tcgen05.mma`（per `tests/e2e/kernel/test_tcgen05_mma_gemm.cu:20-23`）
- 内联 PTX 路径走 fake libcudart.so → simulator → dispatcher
- fallback path（纯 CUDA C++ softmax + matmul）作为 oracle 对比

**Tradeoff**: E2E 测试本身有"双路径"复杂度，需清晰标注哪个路径生效

### D6: 5 phases 而非 1 commit（per lessons-learned §3）

**采纳**: Phase 1-5 各对应 1 atomic commit

**理由**:
- 每个新测试文件 = 1 commit（diff 集中，单文件 review）
- 任何 Phase 失败可独立 revert（不污染其他 4 个测试）
- 每个 Phase 独立 baseline worktree 对比（per Checklist D）

**Tradeoff**: 5 commits 比 1 commit 略增 commit 数（5 vs 1），但遵循 lessons-learned §3 强制

### D7: README 同步在 Phase 5（per lessons-learned §8）

**采纳**: Phase 5 archive commit 前完成 README.md "已实现功能"章节同步

**理由**:
- 任何 feat-* change 归档前必跑 grep 验证 README 同步
- "已实现功能"追加："FlashAttention mini-kernel (QK^T → softmax → @V)" + 本 change 引用

**验证** (per lessons-learned Checklist I):
```bash
grep -n "stub\|TODO\|FIXME\|不实现\|未完成" README.md  # 应为空或明确 TODO + plan
grep -nE "[0-9]+%|硬编码" README.md  # 应替换为自动统计链接
```

### D8: tmem_helpers.h 放在 `include/ptxsim/testing/`（per AGENTS.md 测试分类）

**采纳**: 新增 `include/ptxsim/testing/tmem_helpers.h`

**理由**:
- 复用 `include/ptxsim/testing/` 既有约定（per `scheduler_utils.h`, `instruction_helpers.h`, `predicates.h`）
- 5 个新测试文件共用 helper，避免每个测试文件重复实现
- `fill_tmem_with_golden_inputs` + `require_c_slot_matches` 现有实现可平滑升级到 `compare_c_slot_to_reference`

**Tradeoff**: helper header 跨测试类型共享，需在 testing namespace 下严格限定

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `tests/integration/tcgen05/` | 新增 4 文件 + CMakeLists.txt 修改 | Phase 1-4 测试 |
| `tests/e2e/kernel/` | 新增 1 文件 + CMakeLists.txt 修改 | Phase 5 E2E |
| `include/ptxsim/testing/tmem_helpers.h` | 新增 | 复用 helper |
| `README.md` | 修改（Phase 5 archive 前） | 已实现功能章节同步 |
| FU-1..FU-4 现有测试文件 | **不修改** | 由对应 fix-* change 同步修改 |
| FU-1..FU-4 helper / handler | **不修改** | 本 change 零实现改动 |
| Grammar / IR struct | **不修改** | FU-2 范围 |
| Dispatcher | **不修改** | FU-1..FU-4 范围 |

## Migration Plan

### Phase 0: 准备工作 + 阻塞验证（不在 5 atomic commits 内）

```bash
# 1. 验证 FU-1..FU-4 全部 archive（per Oracle Q2 sequencing）
git log --all --oneline -- "openspec/changes/fix-tcgen05-commit-wait-group/"
git log --all --oneline -- "openspec/changes/fix-tcgen05-idesc-parsing/"
git log --all --oneline -- "openspec/changes/fix-tcgen05-ld-st-slot-routing/"
git log --all --oneline -- "openspec/changes/fix-tcgen05-multi-warp-fragment/"
# 每行应包含 archive commit (e.g., "chore(openspec): archive ...")

# 2. 建立 baseline worktree (per lessons-learned §4)
git worktree add .worktrees/baseline-fa FU-4-archive-commit
cd .worktrees/baseline-fa
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)  # 必须全量 build
cd build && ctest -L "integration;tcgen05" --output-on-failure  # baseline 全 PASS 记录

# 3. 切回主分支
cd ../..
git checkout fix/tcgen05-flashattention-coverage  # 或 main 上新建分支
```

### Phase 1: K=128 accumulator test (commit 1)

```bash
# 1. 编写 tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp
# 2. 编写 include/ptxsim/testing/tmem_helpers.h（含 compare_c_slot_to_reference）
# 3. 修改 tests/integration/tcgen05/CMakeLists.txt 注册 ctest
cmake --build build && cd build && ctest -R "tcgen05_mma_k_loop_128" -V
cd ../..
git add tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp \
        tests/integration/tcgen05/CMakeLists.txt \
        include/ptxsim/testing/tmem_helpers.h
git commit -m "test(tcgen05): add K=128 accumulator integration test (FA-B1)"
```

### Phase 2: commit/wait sequence test (commit 2)

```bash
# 编写 tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp
# （强化版：覆盖 mma → commit → wait → mma + mma → commit → wait → ld → st → mma）
cmake --build build && cd build && ctest -R "tcgen05_mma_commit_wait_sequence" -V
cd ../..
git add tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp
git commit -m "test(tcgen05): add mma→commit→wait→mma sequence integration test (FA-B2)"
```

### Phase 3: cp→mma data flow test (commit 3)

```bash
# 编写 tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp
cmake --build build && cd build && ctest -R "tcgen05_mma_cp_data_flow" -V
cd ../..
git add tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp
git commit -m "test(tcgen05): add cp→mma data flow integration test (FA-B4)"
```

### Phase 4: multi-warp isolation test (commit 4)

```bash
# 编写 tests/integration/tcgen05/test_tcgen05_multi_warp_isolation.cpp
cmake --build build && cd build && ctest -R "tcgen05_multi_warp_isolation" -V
cd ../..
git add tests/integration/tcgen05/test_tcgen05_multi_warp_isolation.cpp
git commit -m "test(tcgen05): add 2-warp C slot isolation integration test (FA-B5)"
```

### Phase 5: E2E kernel + README sync (commit 5)

```bash
# 1. 编写 tests/e2e/kernel/test_flashattention_mini.cu
# 2. 修改 tests/e2e/kernel/CMakeLists.txt 注册 ctest
cmake --build build && cd build && ctest -R "e2e_flashattention_mini" -V
cd ../..
# 3. 同步 README.md（per lessons-learned §8 + Checklist I）
git add tests/e2e/kernel/test_flashattention_mini.cu \
        tests/e2e/kernel/CMakeLists.txt \
        README.md
git commit -m "test(e2e): add FlashAttention mini-kernel + README sync (FA-E2E)"
```

### Phase 6: Archive + ADR postmortem (commit 6)

```bash
# 1. artifacts git-tracked（per lessons-learned §6）
git add openspec/changes/tcgen05-flashattention-coverage/
git commit -m "docs(openspec): tcgen05-flashattention-coverage artifacts"

# 2. ADR-0016 追加 postmortem 段
# (per design.md D7 + ptx-lessons-learned §G Checklist)
git add docs/adr/ADR-0016-blackwell-only-tcgen05.md
git commit -m "docs(adr): ADR-0016 postmortem FlashAttention coverage"

# 3. openspec archive
openspec archive tcgen05-flashattention-coverage --yes

# 4. 强制 postmortem prompt（per openspec-archive-change skill）
# → 必须询问用户 "是否生成 postmortem？(Yes/No/Defer)"

# 5. 全量验证
cd build && ctest --output-on-failure  # 全 PASS
./tests/ptx/test_all_ptx.sh  # 全 PASS

# 6. 清理 baseline worktree
cd ../..
git worktree remove .worktrees/baseline-fa
```

### 回退策略

每个 Phase 独立 commit + 独立可 revert：
- `git revert <phase-N-commit>` → 移除对应测试文件
- 5 phases 互不依赖（K=128 test 不依赖 commit/wait test），可任意顺序 revert
- Phase 6 archive commit revert = `openspec unarchive`（per OpenSpec lifecycle）

## 风险与缓解

| ID | 风险 | 严重度 | 缓解 |
|----|------|--------|------|
| R1 | FU-1..FU-4 archive 时间表延误，本 change 阻塞 | 高 | 5 phases 互不依赖，可分批实施；FU-1 最先做（per Oracle Q2） |
| R2 | E2E kernel ptxas 仍不支持 sm_100 tcgen05 | 中 | D5 fallback path：纯 CUDA C++ softmax + matmul 作为 oracle |
| R3 | K=128 测试运行时间过长（>5s 单测试） | 低 | D2 K=64 边界测试 + K=128 性能 profile |
| R4 | 多 warp 测试与 SM scheduler 顺序执行假设冲突 | 中 | FU-4 C4 实施后 helper 才支持多 warp；本 change Phase 4 必须 FU-4 后做 |
| R5 | README sync 与 lessons-learned §8 重复（每次实施都改 README） | 低 | Phase 5 archive 前一次性修改 + grep 验证（per Checklist I） |
| R6 | 5 commits 分散后单个 test regression 定位耗时 | 低 | baseline worktree 对比每个 Phase（per Checklist D） |
| R7 | E2E kernel CUDA fallback path 与真实 PTX 路径结果不一致 | 中 | 内联 PTX 路径优先 + fallback 仅在 dispatcher 抛异常时生效 |
| R8 | `tmem_helpers.h` 与 FU-1..FU-4 引入的 helper 重名冲突 | 低 | 命名空间 `ptxsim::testing::tmem` 限定 + 编译期检测 |

## Open Questions

1. **Q1: E2E kernel dispatcher 实际状态？** 需 Oracle 验证 `tests/e2e/kernel/test_tcgen05_mma_gemm.cu:20-23` 注释 "dispatcher is broken" 在 FU-1..FU-4 完成后是否仍 true。若仍 broken，本 change E2E 必走 Priority 3 fallback。
2. **Q2: K=128 累加后 c_slot 数值范围？** f32 输出 `1.0..32.0` 范围 × 128 = `128..4096`，仍精确表示。需 Oracle 验证实际 FA 输出是否超出此范围（如 softmax 后概率 × 大 K 可能上溢）。
3. **Q3: 多 warp 测试基础设施？** 当前 `SMContext(2, 128, 4096, 0)` 配置是否支持 2 warp `processTcgen05Mma` 调度？需 Oracle 验证 `sm_context.cpp:250-264` scheduler 是否对 tcgen05 指令做特殊处理。
4. **Q4: tmem_helpers.h 与 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` 现有 helper 关系？** 现有 `require_c_slot_matches` 函数可能在 FU-1..FU-4 实施时被改名/重写。需 FU-4 archive 后 final readback。

## Acceptance Criteria

### Phase 1 (FA-B1) Acceptance

- `cd build && ctest -R "tcgen05_mma_k_loop_128" -V` PASS
- 测试文件有 `K=128` accumulator 路径断言（2×...×128 = 128 × GOLDEN 相对误差 < 1e-3）
- baseline worktree 对比：仅本 Phase 1 commit 后的 1 个新测试，零其他 regression

### Phase 2 (FA-B2) Acceptance

- `ctest -R "tcgen05_mma_commit_wait_sequence" -V` PASS
- 测试覆盖 mma → commit → wait → mma + mma → commit → wait → ld → st → mma 两种序列
- `cta->tc_queue().pending_count() == 0` 在所有 wait 后断言

### Phase 3 (FA-B4) Acceptance

- `ctest -R "tcgen05_mma_cp_data_flow" -V` PASS
- cp → mma 数据流：cp 写 tmem slot X → mma 读 slot X → C 等于 expected（与 cp 数据一致的 mma 输出）
- 与现有 `test_tcgen05_mma_persistence.cpp:250-294` 区别：本测试断言**数值正确性**而非"至少一个元素变化"

### Phase 4 (FA-B5) Acceptance

- `ctest -R "tcgen05_multi_warp_isolation" -V` PASS
- 2-warp 配置：warp 0 mma → C slot[64..95]；warp 1 mma → C slot[96..127]（per FU-4 `c_slot = warp_id * 32 + 64 + lane_id` 公式）
- 2-warp 同时 mma 验证两边 C slot 独立（无 race condition）

### Phase 5 (FA-E2E) Acceptance

- `ctest -R "e2e_flashattention_mini" -V` PASS
- K=4 blocks mini-kernel 端到端：Q@K^T → softmax → @V 数值与 CUDA fallback 一致（rel error < 1e-3）
- README.md "已实现功能"追加 "FlashAttention mini-kernel" 条目 + 本 change 引用

### Phase 6 (Archive) Acceptance

- 4 个 md artifacts git-tracked（`git ls-files openspec/changes/tcgen05-flashattention-coverage/` 不为空）
- ADR-0016 postmortem 追加 "2026-07-XX Postmortem: FlashAttention coverage" 段
- `ctest --output-on-failure` 全量 PASS（0 regression）
- `./tests/ptx/test_all_ptx.sh` 全量 PASS
- 强制 postmortem prompt 询问 user → 显式 Yes/No/Defer 决策

## References

- Oracle 2026-07-11 audit: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` (FlashAttention readiness — 7 BLOCKER/IMPORTANT gaps)
- Oracle 2026-07-11 API 审查: session `ses_0b026333bffePgrqVq7PDJNeR1`
- Metis pre-impl review: session `ses_0b1a0cdb1ffenbhbciQ1n0x236` (per checklist H)
- ADR-0016: [docs/adr/ADR-0016-blackwell-only-tcgen05.md](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md)
- ADR-0018: [docs/adr/ADR-0018-tcgen05-cta-group-restriction.md](../../../docs/adr/ADR-0018-tcgen05-cta-group-restriction.md)
- ptx-lessons-learned: [.opencode/skills/ptx-lessons-learned/SKILL.md](../../../.opencode/skills/ptx-lessons-learned/SKILL.md) §3+§4+§6+§8, Checklists D+E+H+I
- test-coverage-enforcer skill: [.opencode/skills/test-coverage-enforcer/SKILL.md](../../../.opencode/skills/test-coverage-enforcer/SKILL.md)
- FU-1..FU-4: [proposal.md §Follow-Up Changes](../fix-tcgen05-mma-accumulator-and-f32-storage/proposal.md)
- Ref: [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/)