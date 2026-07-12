# Fix tcgen05.mma Fragment Helper — Add Accumulator + f32 Output Storage

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) — Blackwell-only tcgen05
> **Ref** (不能 amend 的已归档 change): [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/) (commit `bc0ef60`)
> **Oracle 2026-07-10 报告**: session `ses_0b3791d78ffewb52428kJJ2Irz` (5 HIGH/MEDIUM confidence blockers)
> **Oracle 2026-07-10 API 审查**: session `ses_0b026333bffePgrqVq7PDJNeR1` (H1/H2 具体 API 细节：idesc/accumulate 默认/readback index)
> **Metis pre-implementation review**: session `ses_0b1a0cdb1ffenbhbciQ1n0x236` (CONDITIONAL GO, 3 MUST-RESOLVE 全部采纳)
> **强制 lessons-learned**: `ptx-lessons-learned` §3(分 Phase commit) + §4(基线 worktree) + §6(artifacts-first) + §7(Pre-implementation Review)
> **范围**: 2 atomic commits (Phase 1: H1 accumulator; Phase 2: H2 f32 storage)

## Why

Oracle 2026-07-10 审计报告 (`ses_0b3791d78ffewb52428kJJ2Irz`) 发现 `tcgen05_fragment_mma_f16` helper 存在 2 个 HIGH confidence 阻塞 FlashAttention 算子开发的缺陷:

| ID | 缺陷 | FlashAttention 影响 | Oracle confidence |
|----|------|---------------------|-------------------|
| **H1** | Helper 无 `+=` 累加器 — `tcgen05_helpers.cpp:42` 零初始化 c_frag + `:57` 覆写写入 | QK^T 和 PV 矩阵乘需要 `C += A*B` 沿 K 维循环累加 | HIGH |
| **H2** | Helper 存为 f16 — `tcgen05_helpers.cpp:50` `f32_to_f16(sum)`，与 PTX ISA §9.7.16 `f16×f16→f32` 矛盾 | 累加器必须 f32 保证数值精度（online softmax 依赖） | HIGH |

加上 step 1 刚提交的 `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` (commit `d3be589`) 验证 H5 假设 — T1 当前断言 overwrite 行为，H1 实施后必然反转。

## What Changes

### 修改（helper + 调用点 + tests）

| 文件 | 范围 | Phase |
|------|------|-------|
| `include/ptxsim/instructions/tcgen05_helpers.h` | 新增 `bool accumulate=false` 参数到 `tcgen05_fragment_mma_f16` | 1 |
| `src/ptxsim/instructions/tcgen05_helpers.cpp` | H1: accumulate 路径读取现有 C slot 累加；H2: c_frag 类型改 `float` 写 f32 | 1 + 2 |
| `src/ptxsim/instructions/tcgen05.cpp:383` | 显式传 `accumulate=false`（保持现有行为） | 1 |
| `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` | T1 反转断言（accumulate）+ 新增 `T1_overwrite` TC | 1 |
| `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp` | readback 模式 `c_buf[idx*2]` → `c_buf[idx*4]` | 2 |
| `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h` | 注释更新（存储格式从 f16 改 f32） | 2 |

### 不修改（范围外）

- ❌ `tcgen05_fragment_mma_f16` 的 32-lane per-lane 布局（per `tcgen05_helpers.cpp:20-23`）— FlashAttention 需要 64×64 warp-cooperative tile（属于 P2 follow-up）
- ❌ `processTcgen05Ld/St/Commit/Wait` 中的 slot 0/group_id/lane_id 硬编码（H3+H4 — 独立 follow-up）
- ❌ 解析真实 PTX `idesc.accumulate` bit（grammar/parser/visitor/handler 全栈修改 — 超出 scope，记录为 ADR-0016 debt）
- ❌ Cluster arrive/wait 的 `cv_.wait()` 阻塞风险（H4 — 独立 follow-up）
- ❌ E2E inline-asm 路径（per Oracle P2 nice-to-have）

## Non-Goals

### 显式拒绝

- ❌ 不修改 grammar（已 archive，无 tcgen05.mma accumulate 语法）
- ❌ 不实现 64×64 warp-cooperative fragment layout（P2 follow-up）
- ❌ 不修复 H3 (ld/st/cp slot 0) 或 H4 (cluster wait 阻塞) — 独立 `fix-*` change
- ❌ 不更新 golden value 数值（保持 1.0..32.0，与 f16→f32 精度等价）
- ❌ 不更新 E2E tests（Priority 3 fallback 与 helper 改动无关）

### 范围限制

- 仅 `tcgen05_fragment_mma_f16` 一个函数（1 个 production caller in `tcgen05.cpp:383`）
- 仅 f16×f16→f32 dtype（其他 dtype 留待 follow-up）
- 仅 single-warp 顺序执行（per helper header comment "Currently safe because SM scheduler runs one warp at a time"）

## Follow-Up Changes（Oracle 2026-07-11 审计 — 独立 OpenSpec）

> **本节记录 Oracle 审计识别的 4 个 BLOCKER 级架构缺口**，均**不在本 change scope 内**，作为独立 `fix-*` change 跟踪。Oracle Q1 + Q6 验证 4-way split 正确，不应合并到本 change（per lessons-learned §3）。

| ID | Follow-up change | Oracle ID | 文件范围 | 依赖 |
|----|------------------|-----------|----------|------|
| FU-1 | `fix-tcgen05-commit-wait-group` | **C3** | 3 文件 ~50 LoC | 基础前置（建立 IMMEDIATE 提取 pattern） |
| FU-2 | `fix-tcgen05-idesc-parsing` | **C1** | 5 文件 ~80 LoC | 与 FU-3, FU-4 并行 |
| FU-3 | `fix-tcgen05-ld-st-slot-routing` | **C2** | 5-7 文件 ~50 LoC | 与 FU-2, FU-4 并行 |
| FU-4 | `fix-tcgen05-multi-warp-fragment` | **C4** | 2 文件 ~20 LoC | 与 FU-2, FU-3 并行 |
| FU-5 | `tcgen05-flashattention-coverage` | **B1+B4+B5+B6+E2E** | 5 文件 ~300 LoC | 在 FU-1..FU-4 完成后 |

### Sequencing 依赖图

```
本 change (H1+H2) + 4 个 in-scope hardening
    │
    ▼
FU-1 (C3) — 基础前置（visitor IMMEDIATE 提取 pattern + handler 读 instr.cta_group）
    │
    ├──► FU-2 (C1 handler accumulate) ──┐
    ├──► FU-3 (C2 ld/st slot) ─────────┼── 可并行
    └──► FU-4 (C4 multi-warp slot) ────┘
         │
         ▼
    FU-5 (FlashAttention coverage + E2E kernel)
```

### 各 Follow-up 简述

**FU-1 `fix-tcgen05-commit-wait-group` (C3) — 基础前置**
- 根因: `extractQualifiersFromContext` (`ptx_visitor.cpp:155-183`) 把 `.cta_group::N` 的 `IMMEDIATE` 值静默丢弃（被 **19 个 call sites** 调用，改返回类型破坏面过大）
- 修复: Option (b) — 在 `visitTcgen05Inst` 加单独 parse tree walk 提取 IMMEDIATE
- 文件: `ptx_visitor.cpp` + `statement_factory.h` + `tcgen05.cpp:493,512,530,550`（handler 读 `instr.cta_group`）

**FU-2 `fix-tcgen05-idesc-parsing` (C1) — handler 累加路由**
- 根因: 即使 H1 修复 helper 累加能力，`processTcgen05Mma` (`tcgen05.cpp:383`) 仍显式传 `accumulate=false`，真实 PTX 路径永不累加
- 修复: idesc 是 `RegOperand`（非 qualifier），运行时从 `ThreadContext::register_bank_` 读 bit 决定 `accumulate` 参数
- 文件: `statement_context.h`（加 `accumulate` 字段）+ `tcgen05.cpp:355-393`（读 idesc）+ `tcgen05_helpers.h:51`（加 `warp_id` 参数）+ `tcgen05_helpers.cpp`（累加路径）+ 新增 `T4/T5` 测试
- 参考模式: `.ws` 限定符 9 层数据流（`ptxLexer.g4:426` → `ptx_qualifier.def:221` → `tcgen05.cpp:340-344`）

**FU-3 `fix-tcgen05-ld-st-slot-routing` (C2) — slot 操作数**
- 根因: `tcgen05.ld` (`tcgen05.cpp:434`)、`tcgen05.st` (`tcgen05.cpp:476`)、`tcgen05.cp` (`tcgen05_cp.cpp:138`) 硬编码 slot 0；mma 写 C 到 `slot[64..95]`，ld/st 永远读不到 mma 输出
- 修复: 添加 `tmem_slot` 字段到 `Tcgen05Instr` + grammar 操作数提取（待 Oracle 确认真实 PTX 语法）
- 文件: `ptxInstructions.g4` + `ptx_op.def:130-132`（op_count 2→3）+ `statement_context.h` + `ptx_visitor.cpp` + `statement_factory.h` + `tcgen05.cpp` + `tcgen05_cp.cpp`

**FU-4 `fix-tcgen05-multi-warp-fragment` (C4) — 多 warp slot 偏移**
- 根因: `c_slot = 64 + lane_id` (`tcgen05_helpers.cpp:23`) 单 warp 假设；多 warp 时 warp 0/1 冲突写 slot 64
- 修复: `c_slot = warp_id * 32 + 64 + lane_id`，helper signature 加 `warp_id` 参数
- 文件: `tcgen05_helpers.h:51` + `tcgen05_helpers.cpp:23` + `tcgen05.cpp:383`（调用点传 `warp->get_warp_id()`）+ 新增 2-warp 集成测试

**FU-5 `tcgen05-flashattention-coverage` (B1+B4+B5+B6+E2E)**
- 测试清单:
  - `test_tcgen05_mma_k_loop_128.cpp` — K=128 累加误差 < 1e-3
  - `test_tcgen05_mma_cp_data_flow.cpp` — cp → mma 完整数据流
  - `test_tcgen05_multi_warp_isolation.cpp` — 2 warp mma slot 不冲突（per FU-4）
  - `test_tcgen05_mma_ld_st_slot_routing.cpp` — ld/st slot 操作数路由（per FU-3）
  - `tests/e2e/kernel/test_flashattention_mini.cu` — 完整 FA mini-kernel

### 当前 change 范围内的 4 个 In-Scope Hardening（per Oracle 审计 + Metis pre-impl review）

| ID | 加固项 | 文件 | Oracle 编号 | 风险 |
|----|--------|------|-------------|------|
| H1+ | B2 `mma → commit → wait → mma` 序列测试 | `tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp`（新文件） | B2 BLOCKER | 零实现改动，纯测试 |
| H2+ | T1_k_loop_4 多迭代 partial | `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp`（新增 TC） | B1 partial | 零实现改动 |
| H3+ | f32_to_f16 移除运行时断言 | `tcgen05_helpers.cpp` 注释 + 新 TC | Metis test gap | 零实现改动 |
| H4+ | 收紧 Catch::Approx epsilon 到 1e-6 | 3 个 readback 测试文件 | B7 | 零实现改动，1.0..32.0 数值不受影响 |

详细任务见 `tasks.md` §1.4.5+§1.4.6（Phase 1 hardening）+ §2.1.6+§2.3.8（Phase 2 hardening）。

## Goals

### Phase 1: H1 — Accumulator 支持（commit 1）

1. 修改 helper signature: `void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false)`
2. accumulate=true 时先 `tmem.read(c_slot)` 预加载 C，f16→f32，与新 sum 累加，写回
3. `processTcgen05Mma` 显式调 `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false)`（保留默认行为）
4. **persistence T1 反转断言** — 2nd mma 后断言 `2 × GOLDEN_MMA_F16_F16_F32`
5. **新增 T1_overwrite TC** — 显式 `accumulate=false` 调用 helper，断言 `1 × GOLDEN`（验证 overwrite 仍可用）
6. 跑 `cd build && ctest -R "tcgen05" --output-on-failure` 验证
7. 跑 baseline worktree 对比（per lessons-learned §4）
8. **commit**: `fix(tcgen05): add accumulate parameter to fragment_mma_f16 helper (Oracle H1)`

### Phase 2: H2 — f32 Output Storage（commit 2）

1. helper body: `c_frag` 类型 `std::array<uint16_t, ROWS * COLS_B>` → `std::array<float, ROWS * COLS_B>`
2. 删除 `f32_to_f16(sum)` 转换，累加循环直接 `c_frag[i*COLS_B + j] = sum`（f32）
3. 写回: `memcpy(c_buf.data(), c_frag.data(), 32 * 4)`（128 字节填满 slot）
4. helper header doc 更新: "C output: 32 f32 elements per lane (128 bytes, fills slot completely)"
5. **readback 机械修改**（per Metis C2 mitigation）:
   - `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp:154-163, 188-194`: `c_buf[idx*2] | (c_buf[idx*2+1] << 8)` + `f16_to_f32` → `memcpy(&val, &c_buf[idx*4], 4)`
   - `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:155-176, 280-292`: 同样 readback 模式更新
6. golden header 注释更新（段 6-7 行加 "f32 storage" 说明）
7. 跑 `cd build && ctest -R "tcgen05" --output-on-failure` 验证
8. **commit**: `fix(tcgen05): store mma C output as f32 per PTX ISA §9.7.16 (Oracle H2)`

### Phase 3: Archive（commit 3，per lessons-learned §6 Checklist G）

1. `git add openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/` + 4 个 md artifacts
2. **先 commit artifacts**（per lessons-learned §6 — artifacts FIRST）: `docs(openspec): fix-tcgen05-mma-accumulator-and-f32-storage artifacts`
3. ADR-0016 追加 "2026-07-11 Postmortem: H1+H2 fix" 段（per lessons-learned §G Checklist）
4. `git add docs/adr/0016-blackwell-only-tcgen05.md` → commit `docs(adr): ADR-0016 postmortem H1+H2`
5. `openspec archive fix-tcgen05-mma-accumulator-and-f32-storage --yes` → 归档
6. 跑 `cd build && ctest --output-on-failure` 全量验证
7. 跑 `./tests/ptx/test_all_ptx.sh` 验证
8. **强制 postmortem prompt**（per openspec-archive-change skill — 用户必须明确选择）

## Capabilities

### Modified Capabilities

- `tcgen05-handlers-extended`:helper 行为修订（accumulator + f32 storage），spec 修订 helper 语义
- `tcgen05-grammar` / `tcgen05-ir-types`:本 change 不修改 grammar/IR；spec 中标注 "simulator function arg ≠ 真实 PTX idesc.accumulate" 的 semantic gap
- `wmma-tensor-core`:本 change 间接修正 mma fragment 算术的 helper（属于同一 fragment 算术链路）

## Impact

### 影响的代码（预计）

| 文件 | 变更类型 | LoC 估计 |
|------|----------|---------|
| `include/ptxsim/instructions/tcgen05_helpers.h` | 修改（加参数 + doc） | +10 |
| `src/ptxsim/instructions/tcgen05_helpers.cpp` | 修改（accumulate + f32 storage） | +20 / -10 |
| `src/ptxsim/instructions/tcgen05.cpp` | 修改（显式传 accumulate=false） | +1 |
| `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` | 修改（T1 反转 + 新增 T1_overwrite） | +30 / -10 |
| `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp` | 修改（readback f16→f32） | +10 / -10 |
| `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h` | 修改（注释更新） | +5 / -2 |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加段（postmortem） | +30 |
| **总计** | | **+106 / -32** |

### 影响的依赖

- 无新外部依赖

### 不影响的依赖

- 11 个 S_TCGEN05_* handler dispatch（per `tcgen05.cpp:574-583`）— 仅 helper body 改动
- TmemAllocator（per `include/ptxsim/memory/tmem_allocator.h`）— 与本 change 无关
- Tcgen05PipelineHandler 3-stage 架构 — 与本 change 无关
- Grammar/parser/visitor — 不修改

### 影响的文档

- `src/ptxsim/instructions/AGENTS.md` — 不需修改（handler dispatch 表不变）
- `src/ptxsim/AGENTS.md` — 不需修改（architecture 不变）
- `docs/adr/0016-blackwell-only-tcgen05.md` — 追加 "2026-07-11 Postmortem: H1+H2 fix" 段
- 根 `AGENTS.md` 已知限制表 — 不需修改（H1+H2 是内部修正，未改变对外能力）

## Design-Time Checklist (Lessons-Learned, per `ptx-lessons-learned`)

### Checklist A: 函数迁移完整性

- [x] **Baseline 函数清单**: `tcgen05_fragment_mma_f16(Tmem&)` 在 `include/ptxsim/instructions/tcgen05_helpers.h:51`，1 个 production caller (`tcgen05.cpp:383`)
- [x] **逐行 diff 计划**: 见 `design.md` §Phase 1 + §Phase 2 + `tasks.md`
- [x] **跨模块状态翻译表**:
  - `tmem.read(c_slot)` → `f16_to_f32(uint16_t bits)` → 与新 `sum` 累加 → `f32_to_f16(sum)`（Phase 1）→ 后续 `tmem.write(c_slot)`
  - Phase 2 后: `tmem.read(c_slot)` → `memcpy<float>` → 累加 → 直接 `memcpy<float>` → `tmem.write(c_slot)`（无 dtype 转换）
- [x] **回退策略**: 每个 Phase 独立 commit，独立可 revert（per lessons-learned §3）

### Checklist B: 重构前

- [x] **基线 worktree 计划** (per lessons-learned §4):
  ```bash
  # Step 1: 建立 baseline (commit `d3be589` 包含 step 1 persistence test)
  git worktree add .worktrees/baseline-h1h2 d3be589
  cd .worktrees/baseline-h1h2
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)  # 必须全量 build
  cd build && ctest -L tcgen05 --output-on-failure
  ```
- [x] **Phase 拆分**:
  - Phase 1 (H1): 独立 commit，1 个 readback 不变（T1 反转是预期契约变更）
  - Phase 2 (H2): 独立 commit，readback 机械修改
  - Phase 3: Archive（含 ADR postmortem）
- [x] **失败处理策略**: 任何已有测试回归 → 立即 revert 该 Phase，不混入后续 commit（per lessons-learned §3）

### Checklist C: 写注释

- [x] **关键注释**（helper header）：
  - Phase 1 添加: "accumulate=true reads existing C slot, f16→f32, accumulates with new sum. Caller must ensure single-warp execution (currently safe per SM scheduler sequential execution)."
  - Phase 2 添加: "C output: 32 f32 elements per lane (128 bytes, fills slot completely). Storage format changed from f16 in commit (Oracle H2 fix per PTX ISA §9.7.16)."
- [x] **golden header 注释**: 更新段 6-7 说明 f32 storage

### Checklist D: Commit 前

- [x] **跑过 baseline worktree 对比** (per Phase)
- [x] **AGENTS.md 同步项**: 不需（handler dispatch 表不变）
- [x] **ADR 追加段落**: Phase 3 追加 "2026-07-11 Postmortem: H1+H2 fix" 到 ADR-0016
- [x] **OpenSpec tasks.md 状态变更**: Phase 3 archive 后 tasks.md 标记 `[x]` 全完成
- [x] **commit message 列出独立 fix 编号**: Phase 1 = "Oracle H1", Phase 2 = "Oracle H2"

### Checklist E: OpenSpec artifacts 提交顺序（2026-07 新增）

- [x] **artifacts-first**: Phase 3 第一步 commit `docs(openspec): ...` 然后才 archive
- [x] **每个 commit 独立可 revert**: Phase 1 / Phase 2 / Phase 3 各自独立

### Checklist G: OpenSpec lifecycle 约束（2026-07 新增）

- [x] **Ref 链接** 到 `archive/2026-07-10-implement-tcgen05-handlers-extended/`
- [x] **禁止 amend 已归档 change**: 本 change 是新建 `fix-*` change，不是 amend

### Checklist H: Pre-implementation Review 强制项（2026-07 新增）

- [x] **Metis pre-implementation review**: ✅ 2026-07-10 (`ses_0b1a0cdb1ffenbhbciQ1n0x236`)，3 MUST-RESOLVE 全部采纳
- [x] **Oracle 决策建议**: ✅ 2026-07-10 (`ses_0b3791d78ffewb52428kJJ2Irz`)，5 个 HIGH/MEDIUM 假设验证
- [x] **Oracle API 细节审查**: ✅ 2026-07-10 (`ses_0b026333bffePgrqVq7PDJNeR1`)，H1/H2 具体 API（idesc=RegOperand、accumulate 默认 false、alignas(16) memcpy readback、load_c_slot helper）
- [x] **persistence T1 反转已规划**: 见 Phase 1 step 4
- [x] **idesc semantic gap 进入 debt**: 见 ADR-0016 postmortem 规划

### Checklist I: 重大功能交付清单

- [x] **本 change 不算"重大功能"**: helper 内部修正，未引入新对外能力
- [x] **根 README 不需更新**: helper 行为修正对用户透明

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| `archive/2026-07-10-implement-tcgen05-handlers-extended` | **fix-tcgen05-mma-accumulator-and-f32-storage** | (未来 `implement-flashattention-kernel` 可基于本 change) |
| `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` (step 1 commit `d3be589`) | | |

- **上游**: 已归档的 implement-tcgen05-handlers-extended 提供 helper 实现的 scope 边界
- **本 change 是 FlashAttention-readiness 的前置** — 没有本 change 修复，QK^T/PV 累加路径无法在 simulator 中验证

## 本 change 特有设计决策（per Metis B 节）

**决策 D1: Accumulator API 形状（per Metis B1）**

- **采纳**: `bool accumulate=false` function arg
- **理由**:
  1. 单调用者（grep 验证 `tcgen05_fragment_mma_f16` 只在 `tcgen05.cpp:383` 调用）
  2. 默认值 `false` 保留所有现有测试的 overwrite 行为（persistence T1 默认走 overwrite 路径）
  3. `(c) Qualifier 路由` 拒绝 — 需 grammar + 5 文件改动，投入产出不成比例
- **Tradeoff**: 不符合真实 PTX `idesc.accumulate` 语义（ADR-0016 debt）

**决策 D2: Output Type 策略（per Metis B2）**

- **采纳**: 无条件 f32 storage（helper body 改 `float[]`）
- **理由**:
  1. PTX ISA §9.7.16 明确 f16×f16→f32（Oracle H2）
  2. slot 利用率从 50% 提升到 100%（A slot/B slot 128B，C slot f32 填满 128B）
  3. `(b) 拆函数` 拒绝 — 创建无意义的"f16 storage"路径，未来删除成本高
  4. `(c) Qualifier 路由` 拒绝 — 输出 dtype 是 hardware-fixed，不需 runtime 分支
- **Tradeoff**: readback 测试需机械修改（per Metis C2 mitigation）

**决策 D3: 现有测试迁移策略（per Metis B3）**

- **采纳**: persistence T1 反转断言（accumulate）+ 新增 T1_overwrite；mma_ws 机械改 readback
- **理由**:
  - T1 反转明确测试 accumulate 语义（2nd mma 后 C 累加）
  - 新增 T1_overwrite 用 `accumulate=false` 显式调用验证 overwrite 仍可用
  - mma_ws readback 是纯机械性（数值不变，存储格式变）
- **Tradeoff**: T1 名字保留但语义反转（注释必须明确）

**决策 D4: Golden 文件名（per Metis B4）**

- **采纳**: 保留 `GOLDEN_MMA_F16_F16_F32` 名字
- **理由**:
  - 名字中的 `F32` 一直指 **output dtype**（per `tcgen05_mma_golden.h:33` 注释 "32 f32 elements"）
  - H2 只是把 storage 与类型名对齐（之前是 f16 storage + f16→f32 readback 掩盖不一致）
- **Tradeoff**: 注释必须明确 "previously stored as f16 with f16→f32 readback; now stored natively as f32"

**决策 D5: Commit 粒度（per Metis B5 + lessons-learned §3）**

- **采纳**: 2 commits (Phase 1 H1, Phase 2 H2)
- **理由**:
  1. lessons-learned §3 强制："复杂迁移必须分 Phase commit"
  2. H1 单独 commit 后 mma_ws 测试 + persistence T2/T3 仍然过（只 T1 overwrite 断言需要改）— Phase 边界清晰
  3. H2 单独 commit 后 readback 改动是 mechanical — diff 集中在 2-3 个文件
  4. 任何 Phase 失败可独立 revert（不污染对方）
- **Tradeoff**: 略增 commit 数（2 vs 1）

**决策 D6: OpenSpec 结构（per Metis C7）**

- **采纳**: 新建 `fix-tcgen05-mma-accumulator-and-f32-storage` change
- **理由**:
  - 已归档的 `implement-tcgen05-handlers-extended` 不能 amend（per lessons-learned §6/G Checklist G）
  - Ref 链接指向 archive 子目录
  - 4 个 artifacts (proposal/design/tasks/spec) 完整
- **Tradeoff**: 比 inline 修改多 4 个 md 文件（强制 OpenSpec 流程）

**决策 D7: ADR 处理（per Metis C8）**

- **采纳**: 在 ADR-0016 追加 "2026-07-11 Postmortem: H1+H2 fix" 段，不开新 ADR
- **理由**:
  - H1+H2 是 helper 行为修正，归 ADR-0016 Blackwell tcgen05 范围
  - 不创建 ADR 制造 churn
- **Tradeoff**: ADR-0016 越来越长（但符合 lessons-learned §G）