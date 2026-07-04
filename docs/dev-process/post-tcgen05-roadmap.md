# Blackwell tcgen05.* 后工作路径规划（Post-Implementation Roadmap）

> **状态**：本 roadmap 在 `implement-wmma-tensor-core` OpenSpec change（拆分为 `phase-0-infra` + `tcgen05` per Oracle Option C）全套交付完毕（2026-07-04 archived）后编写。
>
> **当前 main commit**：`79fc236 archive(phase-1-3): mark tcgen05 handler implementation complete per Checklist G`
>
> **作用域**：ADR-0016（Blackwell-only vision）的后续 forward-path 工作 + 已知 limitation 修复。**不覆盖**：已 archived 状态的修改（per ptx-lessons-learned Checklist G "Archived = 终态"）。

---

## 📍 当前基线（已完成态）

### OpenSpec Changes 已 Archived

| Change | Archive Date | Specs Published |
|---|---|---|
| `implement-wmma-tensor-core-phase-0-infra` | 2026-07-04 | `openspec/specs/wmma-tensor-core/spec.md` (4 Requirements) |
| `implement-wmma-tensor-core-tcgen05` | 2026-07-04 | `openspec/specs/wmma-tensor-core/spec.md` +5 added, `openspec/specs/stub-explicit-failure/spec.md` +2 added / ~2 modified |

### Commits 总计

- **Phase 0**: 14 commits (TMA + TMEM + cluster + TcQueue subsystems + 4 micro CTAContext integration)
- **Phase 1-3**: 5 commits (Fix #10-#14: tcgen05.mma / ld/st / commit/wait / e2e GEMM)
- **Phase 4**: 3 commits (archive phase-0 + merge to main + Ref line + archive phase-1-3)
- **Docs / F2 refactor**: 4 commits (tasks.md updates + validate_u64_zero_terminated rename)
- **Total**: 26 commits on main, all ADR-0016 conformant

### Test State

- **136 labeled tests PASS** (77 unit + 46 integration + 8 e2e + new tcgen05 + wmma)
- **Zero regression** vs baseline (`b7d48ca` pre-split)
- **All Quality Gates PASS**:
  - Phase 0: G1 / G2 / G3 (state-modification-audit) / G4 / G5 (Oracle TMA re-review) / G7 (cute spike)
  - Phase 1-3: G1 (256 UNVERIFIED ≥ 256) / G2 / G3 / G4 (sanity.sh) / G5 (test_all_ptx.sh) / G6 / G7

### ADR-0016 Compliance

| Aspect | Result |
|---|---|
| pre-Blackwell WMMA (`wmma.mma.sync.*`, `wgmma.async.*`, `mma.sync.*`) | ✅ Permanent `UnsupportedInstructionException` |
| Blackwell `tcgen05.*` (sm_100 / sm_120) | ✅ Real fragment arithmetic + ld/st + commit/wait |
| Distributed shared memory | ✅ Deferred to `cta_group::2` (ADR-0018 candidate) |
| sm_120 sparse / FP4 / mxfp8 | ✅ Out of scope (separate changes per feature) |

---

## 🎯 后工作 Tier 分类

### Tier 1：关键修复（correctness/security 类）

| ID | 工作 | 风险 | Scope | 依赖 |
|---|---|---|---|---|
| **F1** | float→f16 精度修复（e2e GEMM kernel）— ANTLR grammar 扩展支持 `ld.global.nc.u16` + `{ }` 代码块 | **High**（grammar 影响所有 PTX parsing） | 1 atomic commit（`fix-*` change） | 需 Oracle consult on ANTLR grammar scope |

**Rationale**：commit `4151268` 的 agent 已明确标记此 limitation（kernel 用 float 而非 f16 due to ANTLR-unsupported constructs）。fix 是直击 known limitation。

---

### Tier 2：架构闭环（complete ADR-0016 vision）

| ID | 工作 | 风险 | Scope | 依赖 |
|---|---|---|---|---|
| **F2** | ADR-0017: TMA host API 拦截（`cuda::tma::create_tensor_map`） | Medium（cudart 拦截层） | 1 OpenSpec change (3-5 commits) | F1 之后 |
| **F3** | ADR-0018: Distributed shared memory for `cta_group::2`（cluster mode 扩展） | **High**（distributed shared memory 复杂） | 1 OpenSpec change (5+ commits) | Phase 0.3 cluster 已有，需要 extension |
| **F4** | ADR-0019: Async queue priority vs scheduler（hardware calibrate 后） | Medium | 1 OpenSpec change | 需要真实 Blackwell GPU 数据 |

**Rationale**：
- F2 让真实 cuda 程序能直接跑（host-side setup 闭环）
- F3 是 cluster mode 的 multi-CTA 扩展（当前 `cta_group::1` only）
- F4 等硬件 calibrate

---

### Tier 3：Blackwell 变体扩展（per ADR-0016 forward path）

| ID | 工作 | 风险 | Scope | 备注 |
|---|---|---|---|---|
| **F5** | sm_120 sparse（`mma.sp.*` 变体） | Medium（不同 fragment layout） | 1 OpenSpec change | 单独的 feature change per ADR-0016 |
| **F6** | FP4 支持（sub-byte packed dtypes） | **High**（ANTLR + TMA + TMEM 三处需更新） | 1 OpenSpec change (multiple commits) | 需要硬件真实数据 |
| **F7** | mxfp8 支持 | High（同 F6） | 1 OpenSpec change | |
| **F8** | cute_rmsnorm 升级到 tcgen05 path（替代 tiled_copy） | Medium | 1 follow-up change | tasks.md 4b.x follow-up 明确提到 |

---

### Tier 4：质量 / 运营清理

| ID | 工作 | 优先级 | 备注 |
|---|---|---|---|
| **F9** | Final worktree cleanup（确认 `.worktrees/` 全清） | Low | 当前 working tree clean（只有 untracked `.opencode/notes/cleanup-barrier-review.md`） |
| **F10** | review `.opencode/notes/cleanup-barrier-review.md` 内容（untracked note） | Low | 可能是 Phase 0 cleanup 残留 — review 后决定归档/删除/合并到 lessons-learned.md |
| **F11** | 真实 Blackwell 硬件 cross-verification（per spec Gate G5） | **Critical when hardware available** | spec 明确要求：256 UNVERIFIED annotations 需硬件 re-validation |

---

### Tier 5：跨切面改进

| ID | 工作 | 优先级 |
|---|---|---|
| F12 | OpenSpec tooling improvements（template for `fix-*` change） | Low |
| F13 | Test infrastructure improvements（reduce 99k-assertion loop tests that overlap） | Low |
| F14 | Documentation sync（archived task.md checkbox patterns） | Low |

---

## 🛤️ 推荐执行路径

```
[Now]
   ↓
F1 (float→f16 修复)            ← lowest risk + highest value
   ↓
F2 (ADR-0017 TMA host API)    ← 架构闭环
   ↓
[分支决策点]
   ├── F3 (ADR-0018 cluster multi-CTA)   ← 如果有 multi-CTA workload 需求
   ├── F8 (cute_rmsnorm upgrade)          ← 如果有 cute cutlass 性能需求
   └── F11 (hardware cross-verify)         ← 当 Blackwell GPU 可用时
   ↓
F5 (sm_120 sparse)             ← 第一个 Blackwell 变体扩展
   ↓
F4 (ADR-0019 scheduler)         ← hardware calibrate 数据驱动
   ↓
F6 / F7 (FP4 / mxfp8)           ← 高级 sub-byte 扩展（需硬件支持）
```

---

## 📂 关键文件路径参考

### Archived OpenSpec Changes

| 内容 | 路径 |
|---|---|
| Phase 0 archive（基础设施） | `openspec/changes/archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/` |
| Phase 1-3 archive（handler 实现） | `openspec/changes/archive/2026-07-04-implement-wmma-tensor-core-tcgen05/` |

### Published Specs

| Spec | 路径 |
|---|---|
| wmma-tensor-core | `openspec/specs/wmma-tensor-core/spec.md` |
| stub-explicit-failure | `openspec/specs/stub-explicit-failure/spec.md` |

### ADRs

| ADR | 路径 | 状态 |
|---|---|---|
| ADR-0016 (Blackwell-only vision) | `docs/adr/0016-blackwell-only-tcgen05.md` | ✅ Accepted 2026-07-04 |
| ADR-0017 (TMA host API) | 尚未存在 | 🟡 Candidate (per F2) |
| ADR-0018 (distributed smem) | 尚未存在 | 🟡 Candidate (per F3) |
| ADR-0019 (async scheduler) | 尚未存在 | 🟡 Candidate (per F4) |

### Implementation Files

| 内容 | 路径 |
|---|---|
| TMA descriptor parser | `src/ptxsim/memory/tma_descriptor.{h,cpp}` |
| TMEM per-CTA storage | `src/ptxsim/memory/tmem.{h,cpp}` |
| Cluster arrive/wait | `src/ptxsim/cluster/cluster_context.{h,cpp}` |
| TcQueue + BAR_SYNC reuse | `src/ptxsim/async/tc_queue.{h,cpp}` |
| WMMA handler (renamed from tensor.cpp) | `src/ptxsim/instructions/wmma.cpp` |
| E2E GEMM kernel (sm_100, known f16 limitation) | `tests/e2e/kernel/test_blackwell_gemm.cu` |

### Known Limitation References

| 限制 | 文档位置 |
|---|---|
| 256 UNVERIFIED-AGAINST-HARDWARE annotations | `src/ptxsim/instructions/wmma.cpp`（每 fragment 元素 + header LAYOUT NOTES） |
| E2E GEMM kernel 用 float 而非 f16 | `tests/e2e/kernel/test_blackwell_gemm.cu` + commit `4151268` body |
| pre-Blackwell WMMA 永久抛异常 | `openspec/specs/stub-explicit-failure/spec.md` Scenario "WmmaHandler-throws-when-invoked-pre-blackwell" |

### Branch / Worktree State

| 项 | 状态 |
|---|---|
| Branch | `main` @ `79fc236`（26 commits ahead of `origin/main`） |
| Worktrees | clean（4a.3 已清理 `.worktrees/fix-pre-p0-baseline`） |
| Working tree | clean（只有 untracked `.opencode/notes/cleanup-barrier-review.md`） |

---

## ⚠️ 重要约束（per OpenSpec 流程 + ptx-lessons-learned）

### OpenSpec 流程约束

1. **新 OpenSpec change 必须 propose first**：直接 git commit 到 main 不符合 OpenSpec 流程
2. **OpenSpec 模板**：每个 fix-*/feat-* change 必须有 `proposal.md` + `design.md` + `tasks.md` + `specs/` 才能 propose
3. **"Archived = 终态"**（per ptx-lessons-learned Checklist G）：archived 状态的修改需要新 change 来覆盖

### ADR-0016 永久约束

1. **pre-Blackwell WMMA 永久抛 `UnsupportedInstructionException`** — 任何 future change **不能 reverse 这点**
2. **per spec scenario "future-change-must-not-add-pre-blackwell-wmma"**：future pre-Blackwell WMMA implementations MUST be rejected at review per ADR-0016

### ptx-lessons-learned 关键 §1/§2/§5

1. **§1 (cross-module state translation)**：Oracle consultation required BEFORE implementation
2. **§2 (recursive lock)**：mutex 复用避免 nested lock death risk（per Phase 0.4 TcQueue 设计）
3. **§5 (qualifier type judgment)**：qualifier 处理用 `has_qualifier()` 遍历，**绝不用 `qualifiers.back()`**（the existing bug pattern）

### Quality Gates 不能跳过

- **Gate G3** (state-modification-audit)：commit 前必须跑（per Phase 0.4 Decision 7）
- **Gate G5** (Oracle re-review TMA)：commit 前必须手动 cross-check magic numbers/bit offsets vs NVIDIA PTX ISA §9.7.13
- **Gate P1-3.G1** (UNVERIFIED annotations)：≥ 256 个 `// UNVERIFIED-AGAINST-HARDWARE` 注释（32 lane × 8x4 matrix）

---

## 🎯 决策选项（preserve for future use）

| 选项 | 描述 | 推荐时机 |
|---|---|---|
| **H1** | 执行 F1（float→f16 e2e GEMM 修复） — propose `fix-tcgen05-e2e-f16-precision` change | Now（lowest risk + highest value） |
| **H2** | 执行 F2（ADR-0017 TMA host API 拦截） — propose `add-cuda-tma-host-api-interception` change | F1 之后 |
| **H3** | 执行 F3（ADR-0018 cluster multi-CTA extension） — propose `add-cluster-distributed-smem` change | F1 + F2 之后 |
| **H4** | review F9/F10 之前先清理（检查 `.opencode/notes/cleanup-barrier-review.md`） | 任意时机 |
| **H5** | 仅规划文档（生成 roadmap reference） | **已执行（本文件）** |

### 当前决策记录

**已选择**：H5 — 仅生成 roadmap reference，无后工作执行。

**理由**：
- OpenSpec change 全套交付完毕（per Checklist G）
- 用户授权 scope 已覆盖（最初 "立即 Phase 0.1" → tasks.md 9 commits → 26 commits total）
- 后工作 Tier 1-5 涉及多方向（precision fix / arch closure / feature extension / cleanup），需要明确 scope decision
- Roadmap 文档化后用户 / 团队可在任意时机选择 H1-H4 启动具体工作

---

## 💡 执行新工作时的标准流程（reference）

当选择 H1-H4 任一选项时，按以下流程：

### Step 1: OpenSpec Propose

```
使用 /openspec-propose skill 或 openspec CLI：
  openspec proposal implement-<name>
  → 生成 proposal.md + design.md + tasks.md + specs/ 模板
```

### Step 2: 设计 Review（per Oracle 触发条件）

| 触发条件 | Action |
|---|---|
| 跨模块 state translation | Oracle consult（per `oracle-prompting` skill 4 rules） |
| 复杂 migration（≥ 3 commits） | Phase 0 baseline worktree + state-modification-audit |
| Critical-risk 改动 | Gate G3 + Gate G5 必须通过 |

### Step 3: 实施（TDD strict）

- 每 Phase 一个 atomic commit
- 单元测试 FIRST（`superpowers/test-driven-development` Iron Law）
- `has_qualifier()` 遍历（NEVER `qualifiers.back()`）
- NO `set_state(BAR_SYNC)` / `set_active_mask` direct call from new handlers

### Step 4: 验证

- `ctest -L "unit|integration|e2e"` → 0 regression
- 适用 Gate 全部通过
- AGENTS.md sync（root + 子模块）

### Step 5: Archive

```
openspec validate implement-<name>     # 必须 valid
openspec archive implement-<name> --yes
git add -A openspec/
git commit -m "archive(<name>): ..."     # single atomic commit
```

---

## 🔗 相关 References

- [`docs/dev-process/lessons-learned.md`](./lessons-learned.md) — 项目核心经验沉淀（含 §1/§2/§5）
- [`docs/dev-process/debugging-strategy.md`](./debugging-strategy.md) — 调试策略
- [`docs/adr/0016-blackwell-only-tcgen05.md`](../adr/0016-blackwell-only-tcgen05.md) — Blackwell-only 基础 ADR
- `openspec/AGENTS.md` — OpenSpec 流程规则
- AGENTS.md — 项目总入口

---

## 📝 Changelog

- **2026-07-04**: Initial version (H5 选择 → 生成 roadmap reference)
- 待更新：当 H1-H4 任一选项被选时更新"当前决策记录"