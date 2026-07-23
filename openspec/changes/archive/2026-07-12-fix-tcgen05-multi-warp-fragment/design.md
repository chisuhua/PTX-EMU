## Context

### 背景

Oracle 2026-07-11 审计（session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` Section C）揭示 `tcgen05_fragment_mma_f16` helper 存在 1 个 HIGH-confidence BLOCKER 级多 warp 缺陷。该缺陷是 4 个 follow-up changes 中之一（Oracle Q1 验证保持 4-way split 正确）。

### 现状问题

**C4 — `c_slot = 64 + lane_id` 硬编码（`tcgen05_helpers.cpp:23`）**

```cpp
// src/ptxsim/instructions/tcgen05_helpers.cpp:20-23
for (int lane_id = 0; lane_id < 32; ++lane_id) {
    size_t a_slot = static_cast<size_t>(lane_id) * 2;
    size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1;
    size_t c_slot = static_cast<size_t>(64) + static_cast<size_t>(lane_id);  // ← 硬编码
}
```

helper header doc（`include/ptxsim/instructions/tcgen05_helpers.h:43-46`）明确写道：
```
Currently safe because SM scheduler runs one warp at a time.
```

实际含义：32 lane × 1 warp 时 `c_slot ∈ [64..95]`，无冲突。但：
- FlashAttention FA3 需要至少 2 个 warp 同时 mma（producer-consumer pipeline）
- 多 warp 配置下 warp 0 和 warp 1 的 `lane_id=0` 都写 slot 64 → **C slot 冲突**
- Tmem 仍是 `cta->tmem()`（per CTA），所有 warp 共享同一 TMEM → 必须人为分片

### 目标状态

`c_slot = warp_id * 32 + 64 + lane_id`，每 warp 独立的 C slot 范围：
- warp 0: `[64..95]`
- warp 1: `[96..127]`
- warp 2: `[128..159]`
- warp 3: `[160..191]`

helper signature 加入 `int warp_id` 参数；调用点 `processTcgen05Mma` 传入 `warp->get_warp_id()`（已存在 API，`tcgen05_alloc.cpp:68` 使用过）。

### Stakeholders

- **sister change `fix-tcgen05-mma-accumulator-and-f32-storage`**（H1+H2）— 同一 helper 的其他修饰
- **foundation change `fix-tcgen05-commit-wait-group`**（C3）— 4 个 follow-up 的基础前置
- **`tcgen05-flashattention-coverage`**（FU-5）— 本 change 的下游消费者

## Goals / Non-Goals

### Goals

- 多 warp mma C slot 不冲突（2-warp baseline）
- 单 warp 路径向后兼容（`warp_id = 0` 时公式等同于原 `c_slot = 64 + lane_id`）
- helper signature compile-time 强制（漏更新 caller 即编译失败 — Oracle C1 mitigation）
- 多 warp mma 测试覆盖（2-warp 隔离 + 2-warp 并行 mma 不冲突）

### Non-Goals

- ❌ **不改 grammar/lexer**（per Oracle Q4 验证 — slot 是 helper 内部数学，非 PTX 语法）
- ❌ **不改 A/B slot layout**（per Oracle 警告 #3 — minimal fix 原则；FA3 实际语义 K shared via cp）
- ❌ **不改 TmemAllocator**（per Oracle Q4 — helper 不走 allocator）
- ❌ **不改 Tmem 容量**（`kTotalSize = 32KB` 保持；4 warp × 4KB = 16KB 仍有 16KB 余量 — design.md D4 论证）
- ❌ **不实现 64×64 warp-cooperative fragment layout**（独立 follow-up，P2 nice-to-have）
- ❌ **不修复 C1/C2/C3**（独立 follow-up changes）
- ❌ **不更新 E2E tests**（Priority 3 fallback 与 helper 改动无关 — per sister change Non-Goals）

## Decisions

### 决策 D1：warp_id 作为 helper 参数显式传入

**采纳方案**: helper signature 加 `int warp_id` 参数（位于 `Tmem&` 之后、`bool accumulate = false` 之前）

```cpp
// include/ptxsim/instructions/tcgen05_helpers.h:51
void tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id,
                              bool accumulate = false);
```

**拒绝的备选**:

| 选项 | 描述 | 拒绝理由 |
|------|------|--------|
| (a) warp_id 通过 Tmem::owner_warp_id() 内部推断 | helper 自动读 tmem 的 owner warp | `tmem.h` 无 `owner_warp_id` 字段（仅 `read/write/clear` 4 个 public methods）；新字段需要 mutex 同步 + 修改 tmem API |
| (b) 通过 warp_id 注册到全局 TmemAllocator | TmemAllocator::allocate 返回 (slot, warp_id) pair | helper 不走 allocator（TmemAllocator 仅用于 `tcgen05.alloc/dealloc`，mma fragment 直接用 helper 内部 slot 计算）；路径错位 |
| **(c) warp_id 作为 helper 参数显式传入**（采纳） | 调用点 `processTcgen05Mma` 传 `warp->get_warp_id()` | `WarpContext::get_warp_id()` API 已存在（`tcgen05_alloc.cpp:68,92,147,191` 使用），零新依赖；helper 内部职责清晰 |

**理由**:

1. `WarpContext::get_warp_id()` 是已存在的 public API（per `tcgen05_alloc.cpp:68` 等 4 处使用）
2. helper 当前调用点（`tcgen05.cpp:383` `processTcgen05Mma`）已有 `WarpContext* warp = context->get_warp_context()`（需验证；如不存在，调用点需要先获取）
3. 编译期强制：漏更新 caller 即编译失败（per Oracle C1 mitigation — function arg 是最强契约）

**Tradeoff**: helper 签名 +1 参数（增加 caller 责任）；4 LoC 成本，可接受。

### 决策 D2：A/B slot 保持共享不变，仅 offset C slot

**采纳方案**: A/B slot 公式不变 `lane_id * 2` / `lane_id * 2 + 1`；仅 C slot 改为 `warp_id * 32 + 64 + lane_id`

**拒绝的备选**:

| 选项 | 描述 | 拒绝理由 |
|------|------|--------|
| (a) 仅 offset C（采纳） | A/B 共享 + C per-warp | minimal fix，符合 FA3 实际语义（Q per-warp + K shared via cp） |
| (b) A/B/C 全 per-warp | 每个 warp 独立 96 slot 范围 | 4 warp × 96 = 384 slot，超出 `kSlotCount=256`；shared input 模型也丢失 |
| (c) A/B per-warp + C shared | 反转 shared/owned | 与实际硬件逻辑不符，impl 复杂 |

**理由**:

1. **minimal fix 原则**（Oracle 警告 #3）：只修 bug 来源，不扩展 scope
2. **FlashAttention FA3 实际语义**：Q tile per-warp（每个 warp 处理不同 head 切片），K tile 共享 via `tcgen05.cp`（cp 加载到 shared tmem slots）→ A per-warp, B shared, C per-warp
3. **scope 隔离**：A/B offset 是 P2 follow-up 的范畴（如 64×64 warp-cooperative fragment）

**Tradeoff**: 如果未来出现 multi-warp A input 需求（如 split-K GEMM），需新增 P2 follow-up。

### 决策 D3：限制为 2-4 warp 支持

**采纳方案**: helper 接受任意 warp_id 但本 change 仅测试 2-warp baseline；4-warp 通过 `kTotalSize` 容量约束隐式支持

**理由**:

1. 当前 NVIDIA Blackwell SM 配置典型为 1-4 warp group
2. Tmem `kTotalSize = 32KB = 256 slots × 128B`（per `tmem.h:30`）→ 每 warp 占 32 slot = 4KB → 4 warp = 16KB（剩余 16KB 余量给 A/B + system）— design.md D4 论证
3. >4 warp 需要更复杂的 FA-3 aware layout（per-warp A/B/C 分片 + barrier 协议）→ 留待真实 PTX 验证

**Tradeoff**: >4 warp 超出当前测试覆盖范围；如真实 FlashAttention kernel 用 5+ warp，需新增 follow-up。

### 决策 D4：不修改 Tmem 容量

**采纳方案**: Tmem `kTotalSize = 32KB` 不变

**理由**:

| Warp count | 每个 warp C slot 范围 | C 总占用 | A/B 占用 | 总占用 | 余量 |
|-----------|---------------------|---------|---------|--------|------|
| 1 | [64..95] (32 slot = 4KB) | 4KB | [0..63] (8KB) | 12KB | 20KB |
| 2 | [64..127] (64 slot) | 8KB | 8KB | 16KB | 16KB |
| 3 | [64..159] (96 slot) | 12KB | 8KB | 20KB | 12KB |
| 4 | [64..223] (128 slot) | 16KB | 8KB | 24KB | 8KB |

8KB 余量足够用于 system metadata + alloc pool。

**Tradeoff**: 如果未来 >4 warp 或更大 fragment layout 需求，需 `kTotalSize` 升级 → 涉及 TmemAllocator 重新设计。

### 决策 D5：单 atomic commit（无 Phase 拆分）

**采纳方案**: 1 atomic commit 包含所有改动（helper sig + slot math + caller + test + AGENTS.md sync + ADR postmortem）

**拒绝的备选**:

| 选项 | 描述 | 拒绝理由 |
|------|------|--------|
| (a) 2 commit: helper sig + caller 测试分两步 | phase 1 = sig + caller, phase 2 = new test | helper 签名变更后无新测试是 broken state（编译过但功能未验证）；2 commit 增加 rebase 复杂度 |
| **(b) 1 commit**（采纳） | 全部一次性提交 | 改动小（+120/-8 LoC），单一 invariant，不易拆分；per lessons-learned §3 "Phase 拆分用于复杂迁移"，非所有改动都分 Phase |

**理由**:

1. 改动范围小（3 文件 + 2 doc）
2. 单一 invariant："warp_id 与 C slot 一一对应"
3. 测试与实现必须同步落地（否则"无新测试的 helper 变更"是 broken state）

**Tradeoff**: 略增 commit 不便 git bisect 定位（但本 change 范围足够小，不需要 bisect）

### 决策 D6：sister change 先合并后 apply 本 change

**采纳方案**: apply 顺序 = sister (H1+H2) → 本 (C4)

**理由**:

1. sister change 添加 `accumulate` 默认参数到 helper signature
2. 本 change 再添加 `warp_id` 参数（在 `tmem` 之后、`accumulate` 之前）
3. 两个连续的 signature 变更如同时 apply 容易产生 rebase 冲突
4. 建议: 等 sister merge 到 main 后，本 change 从 main rebase 出来

**Tradeoff**: 推迟 1 个 merge 周期；可接受（sister change 在提出中，预计先合并）

## Risks / Trade-offs

| ID | 风险 | 影响 | 缓解 | Source |
|----|------|------|------|--------|
| R1 | helper signature 变更导致所有未更新 caller 编译失败 | 编译期 | 编译期强制 — 比 runtime 检查更早发现 | Oracle C1 mitigation |
| R2 | A/B slot 不 per-warp 导致多 warp 共享冲突 | multi-warp GEMM 可能错乱 | 单元测试验证 mma 输出在 warp 间不交叉；下游 follow-up 检测 | Oracle 警告 #3 |
| R3 | 与 sister change `fix-tcgen05-mma-accumulator-and-f32-storage` rebase 冲突 | 推迟合并 | 等 sister merge 后从 main rebase | Decision D6 |
| R4 | >4 warp 未测试覆盖 | 真实 5+ warp FA kernel 可能 broken | 测试限制 2-warp baseline；D3 论证 | Decision D3 |
| R5 | Tmem 容量限制未来可能被突破 | future warp count 增加时需 TmemAllocator 重构 | D4 表格显示 4 warp 仍有 8KB 余量 | Decision D4 |
| R6 | 误将 `warp_id` 参数类型选错（如 `unsigned` vs `int`） | 负 warp_id 导致 UB | 使用 `int` 类型 + helper header 注释明确 "warp_id >= 0 | invalid" → 抛 `std::invalid_argument` | Oracle Q4 建议 |
| R7 | 测试 helper `WarpContext::get_warp_id()` 还未公开 | 测试无法获取 warp_id | 验证 public 状态；如 private，调 public accessor 或加测试 helper | tmem allocator 已有 4 处使用 → 推测 public |

## 影响范围表格（组件 | 影响类型）

| 组件 | 文件 | 影响类型 |
|------|------|----------|
| Helper signature | `include/ptxsim/instructions/tcgen05_helpers.h:51` | API 变更（加参数） |
| Slot math | `src/ptxsim/instructions/tcgen05_helpers.cpp:23` | 行为变更（公式） |
| Caller | `src/ptxsim/instructions/tcgen05.cpp:383` | 调用点（1 行 update） |
| Test | `tests/integration/tcgen05/test_tcgen05_mma_multi_warp.cpp` | 新增文件（4 TC） |
| CMake | `tests/integration/tcgen05/CMakeLists.txt` | 新增 ctest target |
| Helper doc | `include/ptxsim/instructions/tcgen05_helpers.h:43-46` | 注释更新 |
| ADR-0016 | `docs/adr/ADR-0016-blackwell-only-tcgen05.md` | postmortem 段 |
| AGENTS.md | 根 `AGENTS.md` 已知限制表 | 限制更新（s/multi-warp 已支持/g） |

### 不影响的组件（per Oracle Q4 + Q6）

- ❌ `Tcgen05PipelineHandler` 3-stage 架构（dispatch table 不变）
- ❌ `TmemAllocator`（helper 不走 allocator）
- ❌ 11 S_TCGEN05_* handler dispatch（仅 helper 调用点签名变）
- ❌ Grammar/Lexer（per Oracle Q4 验证）
- ❌ sister change H1+H2 范围（除 helper signature 协同）

## Migration Plan（per lessons-learned Checklist A）

### Phase 1: Helper signature + slot math + caller + 测试（1 atomic commit）

#### Baseline 函数清单

| 函数 | 文件:行 | 当前 signature | 计划 signature |
|------|---------|---------------|---------------|
| `tcgen05_fragment_mma_f16` | `include/ptxsim/instructions/tcgen05_helpers.h:51` | `(Tmem&, bool accumulate = false)` | `(Tmem&, int warp_id, bool accumulate = false)` |
| `processTcgen05Mma` (caller) | `src/ptxsim/instructions/tcgen05.cpp:383` | `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false)` | `tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), /*accumulate=*/false)` |

#### 逐行 diff 计划

```cpp
// include/ptxsim/instructions/tcgen05_helpers.h:51
// BEFORE: void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false);
// AFTER:  void tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id, bool accumulate = false);
//         // warp_id: per-warp slot offset to prevent multi-warp C slot conflict.
//         //          - 0 = single-warp mode (backward compatible)
//         //          - N = warp N owns C slots [N*32+64 : N*32+95]
//         //          - A/B slots [0..63] remain shared input fragments.
//         //          - Caller MUST pass warp->get_warp_id() (or 0 for single-warp code).
//         //          - Throws std::invalid_argument if warp_id < 0.
```

```cpp
// src/ptxsim/instructions/tcgen05_helpers.cpp:20-23 (slot math)
// BEFORE:
//   size_t a_slot = static_cast<size_t>(lane_id) * 2;
//   size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1;
//   size_t c_slot = static_cast<size_t>(64) + static_cast<size_t>(lane_id);
// AFTER:
//   size_t a_slot = static_cast<size_t>(lane_id) * 2;
//   size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1;
//   size_t c_slot = static_cast<size_t>(warp_id) * 32
//                 + static_cast<size_t>(64)
//                 + static_cast<size_t>(lane_id);
//   // warp_id validated at entry (>= 0)
```

```cpp
// src/ptxsim/instructions/tcgen05_helpers.cpp (new entry validation, after parameter list)
// BEFORE: (no entry validation)
// AFTER:
//   if (warp_id < 0) {
//       throw std::invalid_argument(
//           "tcgen05_fragment_mma_f16: warp_id must be >= 0 (got " +
//           std::to_string(warp_id) + ")");
//   }
```

```cpp
// src/ptxsim/instructions/tcgen05.cpp:383 (caller update)
// BEFORE: tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false);
// AFTER:  tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), /*accumulate=*/false);
```

#### 跨模块状态翻译表

不适用（per Oracle Q4 — 此 fix 是纯算术变更，不修改 `ThreadContext::state` / 互斥量 / PC / SIMT stack）。

#### 回退策略

- 1 atomic commit 独立 revert（`git revert <commit>` 后 helper 回到单参数版本）
- 回退后所有现有测试应恢复（因为 `warp_id=0` 公式等同原版本 — 但本 commit 包含 signature 变更，所以回退后所有 caller 已更新的 `processTcgen05Mma` 调用点需要相应 revert）

### Phase 2: ADR postmortem + AGENTS.md sync + archive（commit 2，per lessons-learned §6 Checklist E/G）

- [ ] Phase 2.1 `git add openspec/changes/fix-tcgen05-multi-warp-fragment/`
- [ ] Phase 2.2 commit "docs(openspec): fix-tcgen05-multi-warp-fragment artifacts"
- [ ] Phase 2.3 ADR-0016 追加段
- [ ] Phase 2.4 AGENTS.md 已知限制表更新
- [ ] Phase 2.5 `openspec archive fix-tcgen05-multi-warp-fragment --yes`

## Open Questions

1. **A/B slot 是否 per-warp offset**？Decision D2 决定保持共享。需 Oracle 在 design review 阶段确认（session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` 已隐含支持此决定 — Oracle Q4 Option (a) 仅偏移 C slot）。
2. **>4 warp 支持**？Decision D3 决定限制为 2-4。需真实 FlashAttention PTX dump 验证（`cuobjdump -xptx` 提取 FA kernel 看 warp_count）。
3. **`warp->get_warp_id()` API 当前是 public 还是 package-internal**？需验证 `include/ptxsim/core/warp_context.h` 中的访问控制（tcgen05_alloc.cpp 4 处使用的事实表明至少包内可见，测试可能需要 friend 或测试 helper）。

## Acceptance Criteria

### Phase 1 Acceptance

1. ✅ helper signature 包含 `int warp_id` 参数（在 `Tmem&` 后、`bool accumulate = false` 前）
2. ✅ helper body slot 计算改为 `warp_id * 32 + 64 + lane_id`
3. ✅ `processTcgen05Mma` 调用点传入 `warp->get_warp_id()`
4. ✅ `ctest -R "tcgen05_mma_multi_warp"` 4 TC 全部 PASS
5. ✅ 单 warp baseline 路径（如 `unit_tcgen05_mma_ws` 集成测试）保持 PASS（向后兼容 `warp_id=0`）
6. ✅ baseline worktree 对比：本 change 单 warp 路径与 baseline 数值完全一致（`warp_id=0` 等价路径）

### Phase 2 Acceptance

1. ✅ ADR-0016 追加 "2026-07-11 Postmortem: Multi-warp fragment (Oracle C4 fix)" 段
2. ✅ 4 个 md artifacts git-tracked
3. ✅ AGENTS.md 已知限制表更新（"single-warp 顺序执行" 限制取消）
4. ✅ `cd build && ctest --output-on-failure` 全 PASS
5. ✅ `./tests/ptx/test_all_ptx.sh` PASS（grammar 不变，预期 47/47）
6. ✅ archive commit 含 postmortem 引用
7. ✅ 强制 postmortem prompt（per openspec-archive-change skill — 用户明确选择 Yes/No/Defer）

## References

- Oracle 2026-07-11 全局审查: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` (Section C BLOCKER + 4-way split validation Q1-Q6)
- Oracle 2026-07-11 API 审查: session `ses_0af21612bffeKevR9nC1HzBRhL` (Metis pre-impl review for sister H1+H2)
- Sister change: [`../fix-tcgen05-mma-accumulator-and-f32-storage/`](../fix-tcgen05-mma-accumulator-and-f32-storage/) (H1+H2 — helper 内部 fix，已 propose)
- Foundation change: [`../fix-tcgen05-commit-wait-group/`](../fix-tcgen05-commit-wait-group/) (C3 — visitor IMMEDIATE 提取 pattern)
- ADR-0016: [docs/adr/ADR-0016-blackwell-only-tcgen05.md](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md)
- ptx-lessons-learned: [.opencode/skills/ptx-lessons-learned/SKILL.md](../../../.opencode/skills/ptx-lessons-learned/SKILL.md) §3, §4, §6, §7
- proposal.md: [`./proposal.md`](./proposal.md)
- tasks.md: [`./tasks.md`](./tasks.md) (待创建)
- specs/fix-tcgen05-multi-warp-fragment/spec.md: [`./specs/fix-tcgen05-multi-warp-fragment/spec.md`](./specs/fix-tcgen05-multi-warp-fragment/spec.md) (待创建)
- specs/tcgen05-handlers-extended/spec.md: [`./specs/tcgen05-handlers-extended/spec.md`](./specs/tcgen05-handlers-extended/spec.md) (MODIFIED delta spec，待创建)
- PTX ISA §9.7.16 (tcgen05.mma semantics)
