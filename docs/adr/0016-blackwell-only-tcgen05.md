# ADR-0016: Skip pre-Blackwell WMMA, only implement Blackwell tcgen05 (with TMA/cluster prerequisites)

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-07-04 |
| **关联任务** | `openspec/changes/implement-wmma-tensor-core/` (proposed, scope now Blackwell-only) |
| **关联 PR** | TBD |
| **作者** | project architect (user-approved 2026-07-04) |
| **审核人** | TBD |
| **Supersedes** | 无 — but supersedes the original `implement-wmma-tensor-core` proposal scope |
| **Related** | `replace-silent-stub-failures` (archived 2026-07-04) — establishes the explicit-failure contract that this ADR builds on |

## 上下文

`replace-silent-stub-failures`（archived 2026-07-04）把 `WmmaHandler::processWmmaOperation`
的 silent no-op 改成 `throw UnsupportedInstructionException`，建立"未实现 stub 必须显式失败"
的合约。该 change 同时提议了 follow-up `implement-wmma-tensor-core` change，原 Phase 1
scope 是 `wmma.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`（pre-Blackwell 同步 WMMA）。

本 ADR 重新定义 follow-up scope：**不实现任何 pre-Blackwell WMMA 指令，仅实现 Blackwell
（sm_100 / sm_120）的 `tcgen05.*` 指令集**。原因为：

1. **pre-Blackwell WMMA 是 legacy 路径**：NVIDIA 在 Hopper 已经引入 `wgmma.async`，
   在 Blackwell 完全切换到 `tcgen05.mma`。继续实现 `wmma.mma.sync` 等于把工程精力
   投入到 NVIDIA 已经标 legacy 的 ISA。

2. **`tcgen05.mma` 需要的前置基础设施是另一个量级的工程**：
   - **TMA descriptors** (`cuda::tma::desc`)：device-side descriptor 构建与解析
   - **Tensor Memory (TMEM)**：per-CTA 的新存储层，pre-Blackwell 没有
   - **Cluster mode + 分布式 shared memory**：跨 CTA 访问 shared memory
   - **Async tensor core queue**：commit-group counter、`tcgen05.wait` 同步原语

   这些基础设施本身是 ~3000-5000 LoC 的独立子系统。它们的第一个 user 就是 tcgen05。

3. **PTX-EMU 当前架构现状不支持 Blackwell-only 是空中楼阁**：
   - 根 `AGENTS.md` 已知限制：`Hopper (sm_90+) | cluster 抽象未实现`
   - `replace-silent-stub-failures` 让所有 wmma.mma.sync 抛异常 — 这是本 ADR 的
     "现状合约"，本 ADR 让该合约长期化（针对 pre-Blackwell），把工程精力转到 Blackwell

4. **cutlass 3.x / cute 模板的 sm_70 / sm_80 / sm_90 fallback 代码路径仍在**：
   即使用户目标 sm_100，链接 cute 时仍可能引入 `mma_sm70.hpp` / `mma_sm80.hpp`。
   这意味着即使我们只 emit sm_100 PTX，工具链可能仍产生 pre-Blackwell 指令。
   本 ADR 接受这个限制 — pre-Blackwell 抛异常是预期行为，不是 bug。

## 决策驱动因素

1. **Vision purity**：放弃 legacy 兼容性换取对 Blackwell 现代范式的全力投入。
2. **Scope discipline**：cutlass / cute 模板矩阵太大，全做 = 永远做不完。
3. **Infrastructure first**：tcgen05 是新子系统的第一个 user，先建基础设施后填指令。
4. **Test pragmatism**：`cute_rmsnorm` 等已通过的 e2e 测试不使用 WMMA（`grep -r "wmma\." tests/ bench/` 零匹配），所以 Blackwell-only 不会立即破坏现有测试通过状态。
5. **Future-readiness**：Blackwell 是 NVIDIA 2024-2026 主推架构，下一代仍在同一范式。

## 考虑的替代方案

### 方案 A：实现所有 WMMA 版本（legacy 兼容）

- **优点**：任何用户的 PTX 代码都能跑（向后兼容性好）。
- **缺点**：scope 巨大（Volta sm_70 + Turing sm_75 + Ampere sm_80/86 + Hopper sm_90
  wgmma 各家 fragment 变种），永远做不完。维护负担重。
- **否决理由**：与项目当前小团队规模不匹配。

### 方案 B：分阶段（pre-Blackwell 优先 + Blackwell 后续）

- 原 `implement-wmma-tensor-core` Phase 1 scope：`wmma.mma.sync.aligned.m8n8k4`
  pre-Blackwell 同步路径（~200 LoC），后续 Phase 2-3 扩展 fragment 变种，
  最后 Blackwell tcgen05。
- **优点**：每一步都可独立交付 + 早期兼容性好 + 风险分散。
- **缺点**：pre-Blackwell 部分投入的精力最终会被 Blackwell 取代（tcgen05 是新抽象层，
  pre-Blackwell fragment 算术无法复用）。cute 3.x 模板里 pre-Blackwell 与
  Blackwell 是两条独立路径，做 pre-Blackwell 不为 Blackwell 做铺垫。
- **否决理由**：pre-Blackwell 工作量主要是"浪费"在将被替代的范式上。

### 方案 C（采纳）：Blackwell-only + 先建 TMA/cluster 基础设施

- **结构**：4 个 Phase，全部 Blackwell 路线
  - **Phase 0**：TMA descriptors + TMEM + cluster mode + async tensor core queue
    基础设施（~3000-5000 LoC，独立子系统）
  - **Phase 1**：`tcgen05.mma.cta_group::1.kind::f16` 真实 fragment arithmetic
  - **Phase 2**：`tcgen05.ld` / `tcgen05.st` + `tcgen05.commit` / `tcgen05.wait`
    完整异步模型
  - **Phase 3**：cutlass 3.x GEMM e2e kernel + AGENTS/spec sync
- **优点**：
  - Vision 清晰 — 全部 Blackwell 路线
  - 基础设施是 tcgen05 的硬前置，先建避免后期架构返工
  - 未来扩展（sm_120 sparse / FP4 / mxfp8）落在同一范式上
- **缺点**：
  - Phase 0 工作量大（~3000-5000 LoC），交付时间晚
  - pre-Blackwell 用户代码永远抛异常 — 显式拒绝兼容
- **采纳理由**：vision purity 优先，scopee discipline 次之。

## 决策

**采用方案 C**。具体承诺：

1. **不再实现任何 pre-Blackwell WMMA 指令**。包括但不限于：
   - `wmma.mma.sync.*` (sm_70 / sm_75 / sm_80 / sm_86)
   - `wgmma.async.*` (sm_90)
   - `mma.sync.*` (sm_70+ 通用路径)
   - 任何依赖上述指令的 cute / cutlass template instantiation
   
   这些路径继续抛 `UnsupportedInstructionException`（与 `replace-silent-stub-failures`
   合约一致），不视为 bug。

2. **TMA / cluster / TMEM / async tensor core queue 必须先建**。这是 tcgen05 的硬前置
   依赖。`openspec/changes/implement-wmma-tensor-core/tasks.md` 的 Phase 0 即此子系统。

3. **`tcgen05.*` 是首个也是唯一一类受支持的 tensor core 指令**。未来扩展（sm_120、
   FP4、sparse variants）落在同一范式上，不需要新 ADR。

4. **现有 cute_rmsnorm 等 e2e 测试不受影响**（实测不依赖 WMMA）。`grep -r "wmma\."
   tests/ bench/` 零匹配。如果未来 cute_rmsnorm 升级到用 tcgen05，需要先确保
   Phase 0-2 已完成。

## 影响

### 项目代码

- `openspec/changes/implement-wmma-tensor-core/` 全部 artifacts 重写：
  - `proposal.md`：Why 改为 Blackwell-only vision；Non-Goals 明确列出 pre-Blackwell
  - `design.md`：Decision 5 个（含 TMA/cluster 前置决策）；Migration Plan 4 个 Phase
  - `tasks.md`：Phase 0 = TMA/cluster infra；Phase 1-3 = tcgen05
  - `specs/wmma-tensor-core/spec.md`：替换 m8n8k4 pre-Blackwell 场景为 tcgen05 m64nNk
  - `specs/stub-explicit-failure/spec.md`：保留 WMMA throw 场景（pre-Blackwell 永久 throw）

### 用户代码

- **不会破坏**：现有 cute_rmsnorm 等测试（不使用 WMMA）
- **会抛异常**：使用 pre-Blackwell WMMA 的用户代码（视为预期行为，不修）
- **会运行**：使用 Blackwell tcgen05 的用户代码（Phase 1-3 完成之后）

### 基础设施依赖

- 必须新增的子系统：
  - `src/ptxsim/memory/tma_descriptor.{h,cpp}` — TMA descriptor 解析
  - `src/ptxsim/memory/tmem.{h,cpp}` — Tensor Memory per-CTA 存储
  - `src/ptxsim/cluster/{h,cpp}` — cluster mode + 分布式 shared memory
  - `src/ptxsim/async/tc_queue.{h,cpp}` — async tensor core queue + commit-group counter

### 文档

- `docs/architecture/sm90_100.md`：在 §4 引用本 ADR，明确 Blackwell 路径是优先路径
- `docs/adr/README.md`：索引添加本 ADR
- 根 `AGENTS.md` 已知限制表：cluster 抽象状态从"未实现"更新为"实施中（ADR-0016）"

## 实施路径（与 tasks.md Phase 0-3 一致）

| Phase | scope | 估算 LoC | commit 粒度 |
|-------|-------|---------|-----------|
| Phase 0 | TMA + TMEM + cluster + async queue | ~3000-5000 | **9 commits** = 4 独立子系统 (TMA/TMEM/cluster/async queue) + 4 逐子系统集成微 commit (0.5.1~0.5.4 解决 ptx-lessons-learned §3) + 1 artifacts commit |
| Phase 1 | tcgen05.mma fragment arithmetic | ~500-800 | 2 commits（impl + tests） |
| Phase 2 | tcgen05.ld/st + commit/wait | ~600-1000 | 2 commits（load/store + commit/wait） |
| Phase 3 | e2e GEMM + AGENTS sync + spec publish | ~300-500 | 1 commit |

每个 Phase commit 独立可 revert（`ptx-lessons-learned` §3 强制要求）。

## 回滚策略

- **Phase 0 子系统 commit**：每个独立 revert。如果某个子系统（如 TMA）出错，
  不影响 TMEM / cluster / async queue 已合并的部分。
- **Phase 1+ tcgen05**：revert 后回到 "pre-Blackwell 抛异常 + Blackwell 也抛异常"
  的现状，与 `replace-silent-stub-failures` 合约一致。
- **整体 ADR 撤销**：如果未来发现 Blackwell-only 决策错误，新建 ADR-XXXX
  将本 ADR 标记为 Superseded，并恢复 pre-Blackwell 实现路线。

## Open Questions

1. **sm_120 sparse + FP4 + mxfp8 是否在同一 change 内？**
   建议：Phase 1-3 完成后单独 propose，每个 sm_120 新特性一个 change。

2. **cutlass 3.x template instantiation 在 cute 端如何处理？**
   即使用户写 `cute::MMA_Atom<cute::SM100_MMA_F16_F16_F32>`，cute 模板可能
   在 PTX 层面 fallback 到 sm_90 wgmma。我们只保证最终 emit 的 PTX 走 Blackwell
   路径，不保证 cute 模板编译时不引用 sm_90 头文件。

3. **TMA descriptor 的 device-side 构建如何模拟？**
   `cuda::tma::create_tensor_map` 是 host API，但 descriptor 存到 device memory
   后被 `tcgen05.ld` 用。PTX-EMU 拦截 `cudaMemcpy` 时是否需要介入？
   建议：先实现 `tcgen05.ld` 接受 fake descriptor（手工填值），再考虑拦截 host API。

## 相关 ADR

- **ADR-0009**（xmacro-instruction-dispatch）：X-Macro + Weak Symbol 模式 — 本 ADR 仍然
  通过该模式分发 `tcgen05.*` handler
- **ADR-0012**（per-thread-pc）：tcgen05 的异步执行模型不冲突 per-thread PC（warp-level
  queue 与 per-thread PC 是两个抽象层）
- **ADR-0014**（independent-thread-scheduling）：ITS 支持与本 ADR 正交 — ITS 是 warp 内
  多路径调度，tcgen05 是 warp 间 / CTA 间异步张量核心
- **未来 ADR 候选**：
  - ADR-0017：`cuda::tma::desc` 拦截策略（如果 Phase 0 实现需要）
  - ADR-0018：cluster mode 的 distributed shared memory 模拟策略
  - ADR-0019：async tensor core queue 与现有 WarpState 集成模式

## 更新记录

### 2026-07-04 — Commit count 修正 (Oracle review C2 fix)

**修订理由**：原 Phase 0 commit 粒度"4-5 个独立 commit"是 Oracle 审查前的初版。Oracle
审查（commit `cf78fe6`，2026-07-04）发现 4 个子系统在单 commit 集成违反 `ptx-lessons-learned §3`
（独立可 revert），拆为 4 个微 commit (0.5.1~0.5.4)。同时按 `experience 6` + Checklist E
新增 1 个 artifacts 前置 commit。

**Phase 0 commit 粒度演进**：
- v1 (ADR 初版)：5 commits = TMA / TMEM / cluster / async queue / 集成
- v2 (Oracle 后)：9 commits = 4 standalone + 4 集成微 commit + 1 artifacts

**变更范围**：仅"实施路径"表格 Phase 0 行文字 + commit 粒度数字。**决策本身**（Blackwell-only、
scope discipline、基础设施优先）**未变**。

**Revert 单元澄清**（同样来自 Oracle Q2 fix）：
- 0.1–0.4 + 1.1/1.2/2.1/2.2/3.1/0.artifacts 可独立 revert
- 0.5.1–0.5.4 **不可独立 revert**（`TcQueue::enqueue_mma()` 写 TMEM slot，跨子系统引用
  破坏独立性 — 见 `cta_context.h:112` `BarrierModule` 模式但 TcQueue 跨 4 子系统）
- 整体 Phase 0.5 revert = `git revert <0.5.1-sha>..<0.5.4-sha>`

## Phase 1-2 完成记录

**2026-07-07**: 5 core handler (mma/ld/st/commit/wait) 从 `wmma.cpp` 提取到独立 `src/ptxsim/instructions/tcgen05.cpp`（commit `df6dde7`，OpenSpec `implement-tcgen05-handlers-core`）。Handler 使用 `Tcgen05Instr::op_kind` 分发替代旧 qualifier-based 检测。`wmma.cpp` 简化为 pre-Blackwell `UnsupportedInstructionException`。

**2026-07-08**: 5 core handler 测试覆盖（commit `fd74261`，OpenSpec `fix-tcgen05-test-coverage-gaps`）。
- 5 integration parse 测试(`tests/integration/ptx/test_tcgen05_*_parse.cpp`)
- 1 unit test + 1 dispatch integration test(`tests/integration/tcgen05/test_tcgen05_dispatch.cpp`)
- 1 unit test `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp` + 1 `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp`
- 1 E2E GEMM kernel(`tests/e2e/kernel/test_tcgen05_mma_gemm.cu`,f32 fallback)
- 1 golden value(`tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`,PTX ISA §9.7.16 手算 f16×f16→f32)
- 测试结果:11 tcgen05-tagged ctest 全 PASS

**2026-07-08**: tcgen05 handler dispatch 管道接入（commit `cc49ae7`，OpenSpec `fix-tcgen05-handler-dispatch`）。
- `Tcgen05Handler::processTcgen05Operation` 统一 dispatch 入口(`tcgen05.cpp` 末段)
- 11 个 `S_TCGEN05_*` X-Macro 注册到 `InstructionFactory`
- `Tcgen05PipelineHandler` 3-stage pipeline stub(`ptxir+ptxsim` X-Macro wiring @ `3a30da8`)
- 5 handler 函数保留兼容(`fix-tcgen05-test-coverage-gaps` dead-code coverage test 需 `&ptxsim::processTcgen05Mma` 函数指针)
- 6 extended op_kind(ALLOC/DEALLOC/RELINQUISH/CP/MMA_WS/FENCE)throw deferred — 待 `implement-tcgen05-handlers-extended`

**2026-07-09**: Phase 1 of `implement-tcgen05-handlers-extended` (commit `486246a`)。
- 新增 `TmemAllocator`(`src/ptxsim/memory/tmem_allocator.{h,cpp}`):per-CTA 256-slot first-fit 分配器,`std::bitset` 跟踪分配状态,严格遵循 ptx-lessons-learned §2 递归锁模式
- 新增 3 个 alloc-family handler(`src/ptxsim/instructions/tcgen05_alloc.cpp`):`processTcgen05Alloc`/`Dealloc`/`Relinquish`
- 接入 dispatch table(`tcgen05.cpp:574-583`):8/11 handler 已实现,3 deferred (CP/MMA_WS/FENCE)
- `cta_group::2` 全部抛 `UnsupportedInstructionException` 含 ADR-0018 引用
- 新增 per-warp `allocate_permit` 字段(`warp_state.h:18`)+ `set/get_allocate_permit` 访问器(`warp_context.h`)
- 12 TmemAllocator 单元测试(`tests/unit/memory/test_tmem_allocator.cpp`)
- AGENTS.md "已知限制" 表同步:8/11 handler 已实现,3 deferred (CP/MMA_WS/FENCE)
- 测试结果:73/73 unit tests PASS,45/45 PTX syntax tests PASS

**2026-07-09**: Phase 1.x critical-issues 修订 (Oracle 2026-07-09 review, OpenSpec `fix-tmem-allocator-phase1x-critical`)。
- 修复 `TmemAllocator` read-only methods 数据竞争 (UB) — `is_allocated_start`/`is_allocated`/`active_allocation_count`/`total_allocated_slots` 加 `lock_guard(mu_)`
- `static_assert(TmemAllocator::kSlotCount == Tmem::kSlotCount)` 强制 256 一致性
- 修复多线程死锁检测(`test_tmem_allocator.cpp`):用 `std::async` + `wait_for(30s)` 替代 `th.join()`
- 修正 `processTcgen05Dealloc` 注释矛盾(原 "most-recent" 实为 "lowest active slot_id")
- 新增 3 个 handler 集成测试(`tests/integration/tcgen05/test_alloc_dealloc_relinquish.cpp`):12 TEST_CASEs / 28 assertions
- AGENTS.md / `src/ptxsim/instructions/AGENTS.md` 同步
- 沉淀:新教训 "read-only methods don't hold mu_" 模式

## Archive 文档一致性声明（2026-07-08）

`openspec/changes/archive/2026-07-07-fix-tcgen05-antlr-prediction-bug/` 的 `proposal.md`/`design.md`/`handoff.md` 声称修复"ANTLR LL(*) 预测冲突",但真正根因是 **lexer bare string token 与 ID 规则冲突**（详见 [`docs/dev-process/lessons-learned.md` §25](../../docs/dev-process/lessons-learned.md#25-antlr4-le)）。按 `ptx-lessons-learned §6 + Checklist G` 铁律"已归档 change 不 amend",**本节为权威 override**: §25 为根因真相,archive 文档保留作为历史。
## Phase 3: tcgen05.mma.ws handler via qualifier routing (2026-07-09, Oracle 2026-07-08 A-path)

**2026-07-09**: Phase 3 of `implement-tcgen05-handlers-extended` 落地(commit `f4b6d58`),基于 Oracle 2026-07-08 critical findings 修正原计划:

### 计划 vs 实现的关键差异

- **原计划**: 写独立 `processTcgen05MmaWs` 函数 + dispatch 表加 `case Tcgen05OpKind::MMA_WS`,沿用 spec.md 的 `.warpspecialized::1` 词汇
- **Oracle review 发现**: grammar `ptxInstructions.g4:436-447` 的 `tcgen05SubOp` 没有 `MMA_WS` sub-op,`.ws` 是 `Q_TCGEN_WS` qualifier 在 MMA sub-op 上。`case MMA_WS:` dispatch 永远从真实 PTX 进不来(dead path)
- **实施**: 删除独立 handler 函数,在 `processTcgen05Mma` 内部 scan `instr.qualifiers` for `Q_TCGEN_WS`,Q3-A 范围检查(要求 `Q_F16` 必备,缺失抛 `UnsupportedInstructionException`),然后调 `tcgen05_fragment_mma_f16` helper(Phase 2.5 抽出,DRY)
- **`case MMA_WS:` 保留但路由到 `processTcgen05Mma`**: 用于直接构造 `Tcgen05Instr{op_kind=MMA_WS, ...}` 的测试场景

### Pre-Phase 2.5 refactor (commit `3b6ead4`)

- 新增 `tcgen05_fragment_mma_f16(Tmem&)` helper 到 `include/ptxsim/instructions/tcgen05_helpers.{h,cpp}`
- 从 `processTcgen05Mma` 抽出 60 LoC 片段算术,行为不变
- 验证 183/183 ctest + 45/45 PTX 一致(behavior-preserving)

### 范围调整 (Oracle 2026-07-08 Q3-A)

- **接受**: ws path 仅支持 `.kind::f16` + `Q_TCGEN_WS` qualifier 存在;非 f16 kind 抛清晰异常
- **defer**: ws-specific weight-stationary layout transform(单 warp 简化下与 mma 算术相同);ws 路径标记 `// UNVERIFIED-AGAINST-HARDWARE`
- **修正**: spec.md `Scenario: weight-stationary mma.ws handler` 改写为 qualifier-based routing 描述;design.md D3 加 "Phase 3 实施修订" 注释解释 grammar 现实

### 测试覆盖

- unit: 7 TEST_CASEs (`tests/unit/tcgen05/test_tcgen05_mma_ws.cpp`)
  - ws+Q_F16 → ws path 执行(无 throw)
  - ws+Q_F32 → throw(Q3-A scope violation)
  - ws+Q_BF16 → throw
  - ws+no kind → throw
  - no ws → regular mma path(无 Q3-A 检查)
  - op_kind=MMA_WS (直接构造) + 空 qualifiers → regular mma path(dispatch trace 验证)
  - op_kind=MMA + 空 qualifiers → regular mma path(negative control)
- integration: 3 TEST_CASEs (`tests/integration/tcgen05/test_tcgen05_mma_ws.cpp`)
  - ws+f16+cta_group qualifier + golden A/B inputs → 32 lanes × golden C 全部命中(`GOLDEN_MMA_F16_F16_F32` from `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`)
  - op_kind=MMA_WS (直接构造) → regular mma result(同 golden)
  - ws+Q_F32 → throw (scope violation 在 helper 调用前)
- e2e: 1 Priority 3 fallback (`tests/e2e/kernel/test_tcgen05_mma_ws.cu`)
  - ptxas 13.0 不支持 sm_100 tcgen05.mma.ws,纯 CUDA C++ fragment 模拟 per-lane output;source-grep oracle 验证 `tcgen05.mma.ws` 引用

### 同步

- spec.md: Scenario rewrite + 移除 `.warpspecialized::1` 词汇
- design.md D3: Phase 3 修订注释 + grammar 现实解释
- tasks.md §4: 全部 `[x]` 标记完成 + Oracle A-path 决策记录
- AGENTS.md: 9/11 → 10/11 handler 已实现
- `src/ptxsim/instructions/AGENTS.md`: FRAGMENT HELPER section + MMA.WS section + FENCE only 仍 deferred
- `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp`: deferred 列表移除 `MMA_WS`(6 → 1,只留 FENCE)

### 沉淀

- 新教训 "dispatch dead path": `processTcgen05OpKind::MMA_WS` 写好但真 PTX 不可达 → 写 dispatch 前 grep grammar sub-op 真存在
- 新教训 "Spec/Design 词汇脱节": spec.md `.warpspecialized::1` 在 grammar 中不存在 → 设计阶段必跑 grep 验证词汇对齐
- 新教训 "IR 便捷字段未连": `Tcgen05Instr::cta_group` 等字段在 visitor 中从未被赋值,handler 检查永远成立 → handler 检查便捷字段前必 grep visitor 验证提取路径
- `docs/dev-process/lessons-learned.md §27` + `.opencode/skills/ptx-lessons-learned/SKILL.md` 失败模式速查表 3 行

### 验证

- 186/186 ctest PASS(原 183;Phase 3 新增 3)
- 45/45 PTX syntax tests PASS(无 grammar 改动)
- baseline worktree(`bb30ea2`)PTX 一致
- 16/16 tcgen05-tagged ctest PASS
- 与 Phase 1.x critical fixes (`0a4358d`) + Phase 2 (`178457d`) + Phase 2.5 (`3b6ead4`) 共存

## Phase 4: tcgen05.fence no-op marker (2026-07-10, Oracle Q6-B / design D8)

**2026-07-10**: Phase 4 of `implement-tcgen05-handlers-extended` 落地(commit `718095a`):
- 新增 `tcgen05_fence.cpp` (~100 LoC):`processTcgen05Fence` 实现为 **no-op marker**(per Oracle Q6-B)
- `WarpState::fence_position` 扩展点(`int8_t`,4 enum 值:`kFenceNone / kFenceBefore / kFenceAfter / kFenceUnknown`)
- `WarpContext::record_fence_position / get_last_fence_position` 访问器
- `cta_group::2` 抛 `UnsupportedInstructionException` 含 ADR-0018 引用(Q2-A consistency)
- 递归锁审计:**0** `lock_guard/unique_lock/std::mutex` matches 在 warp state path(grep 验证)
- 6 unit + 4 integration + 1 e2e 测试(Q5-C 混合 oracle strategy):
    - `unit_ptx_ir_tcgen05_extended_opkind.cpp`(6 cases / 15 assertions)
    - `integration_tcgen05_extended_parse.cpp`(4 cases / 20 assertions)
    - `e2e_tcgen05_alloc.cu`(Priority 3 fallback,ptxas 13.0 不支持 sm_100 alloc)
- AGENTS.md + `src/ptxsim/instructions/AGENTS.md` + `docs/ptx/README.md` 同步 11/11 状态

### 验证

- 189/189 ctest PASS(原 186;Phase 4 新增 3 tests = `unit_extended_opkind` + `integration_extended_parse` + `e2e_tcgen05_alloc`)
- 45/45 PTX syntax tests PASS
- `ctest -L tcgen05` = 22/22 PASS(原 19)
- baseline worktree(`bb30ea2`)PTX 一致
- `tcgen05` 11/11 handler 已实现

## 2026-07-11 Postmortem: H1+H2 fix (fix-tcgen05-mma-accumulator-and-f32-storage)

### H1 Root Cause
`tcgen05_fragment_mma_f16` (per `src/ptxsim/instructions/tcgen05_helpers.cpp:42,45,57`) 零初始化 `c_frag` 并覆写写入，从未读取 `c_slot` 已有值。FlashAttention QK^T/PV 矩阵乘需要 `+=` 累加器（沿 K 维循环），helper 缺乏此能力。

### H1 Fix
新增 `bool accumulate` 参数（默认 `false`）。`accumulate=true` 时先 `tmem.read(c_slot)` 预加载 C，f16→f32 转换，与新 sum 累加，写回。`processTcgen05Mma` (`src/ptxsim/instructions/tcgen05.cpp:383`) 显式传 `accumulate=false` 保持现有行为。

### H2 Root Cause
Helper 输出存为 `uint16_t` (f16)，与 PTX ISA §9.7.16 规定 `f16×f16→f32` 矛盾。`tests/reference/ptx_tcgen05/tcgen05_mma_golden.h:6` 声称 "32 f32 elements" 但实际是 f16 storage + f16→f32 readback 掩盖不一致。

### H2 Fix
Helper body 改 `c_frag` 类型从 `uint16_t` → `float`，删除 `f32_to_f16` 转换。Slot 利用率从 50%（64 bytes / 128 bytes）提升到 100%（128 bytes）。4 处 readback site 迁移到 `alignas(16) float c_arr[32] + std::memcpy` 模式（per Oracle Q3 推荐），数值等价。

### Known Semantic Gap (debt for future)
Helper `accumulate` 参数是 **simulator 内部决策**，**不解析真实 PTX `idesc.accumulate` bit**。完整修复需要 grammar + parser + visitor + handler 全栈修改（Oracle 2026-07-11 审计 C1 BLOCKER）。PTX ISA §9.7.16 中 accumulate 由 `idesc` 第 N 位 bit 控制（NVIDIA 内部微架构细节，未公开）。
Follow-up change: `fix-tcgen05-idesc-parsing` (已 propose)。

### Other BLOCKER Debt (Oracle 2026-07-11 审计)
- **C2** ld/st 硬编码 slot 0 — Follow-up: `fix-tcgen05-ld-st-slot-routing` (已 propose)
- **C3** commit/wait 硬编码 group_id=1 + `extractQualifiersFromContext` 丢弃 IMMEDIATE — Follow-up: `fix-tcgen05-commit-wait-group` (已 propose)
- **C4** 多 warp slot 冲突 `c_slot = 64 + lane_id` — Follow-up: `fix-tcgen05-multi-warp-fragment` (已 propose)

### Validation
- Phase 1 (H1, `df1f6de`): 22/22 tcgen05 PASS (added 4 hardening tests including B2 sequence)
- Phase 2 (H2, `f97863c`): 22/22 tcgen05 PASS (readback migration mechanical, no regressions)
- Zero regressions introduced

## 2026-07-13 Postmortem: C3 fix (fix-tcgen05-commit-wait-group)

### C3 Root Cause
`processTcgen05Commit` (`tcgen05.cpp:512`) + `processTcgen05Wait` (`:550`) hardcoded `group_id=1` + `lane_id=0`. `Tcgen05Instr::cta_group` field (`statement_context.h:186`) was declared but never populated by visitor (`visitTcgen05Inst` at `ptx_visitor.cpp:858` only stored qualifiers as enum values, silently discarding the `IMMEDIATE` child of `TCGEN_CTA_GROUP COLONCOLON IMMEDIATE` per `extractQualifiersFromContext` at `ptx_visitor.cpp:155-183`).

### C3 Fix
1. **Visit-time extraction**: `visitTcgen05Inst` walks parse tree to find `TCGEN_CTA_GROUP` contexts and reads the `IMMEDIATE` child (Option (b) from Oracle 2026-07-11 Q5 — avoids breaking 19 `extractQualifiersFromContext` callers).
2. **Handler reads IR field**: `processTcgen05Commit` now calls `commit(instr.cta_group)` instead of `commit(1)`; `processTcgen05Wait` calls `wait(warp, 0, instr.cta_group)`.
3. **Default `cta_group=1`** preserves backward compatibility for all existing PTX without explicit `.cta_group::N`.
4. **Bonus: TcQueue::wait() early return** (§29): checks `commit_group_counter_ >= group_id` BEFORE pushing to `pending_waiters_`, avoiding stale waiter entries.

### Test Coverage
- 2 new integration tests (`integration_tcgen05_commit_wait_group` — 3 TC: commit routing, commit→wait sequence, wait-block+release)
- 1 new parse test (`integration_ptx_tcgen05_mma_parse` — `cta_group::2` extraction)
- 2 new TcQueue unit tests (early return TC#15 + regression guard TC#16)
- Full ctest: all PASS. PTX grammar: 45/45 PASS.

### Known Semantic Gaps (debt for future)
- `tcgen05.wait N` lane_id operand not parsed (per `ptx_op.def:136` `op_count=0`). Future: `fix-tcgen05-wait-lane-id` (FU-3.5).
- Multi-group synchronization not yet exercised by E2E test. Future: `tcgen05-flashattention-coverage` (FU-5).

### Follow-up Changes
- `fix-tcgen05-idesc-parsing` (FU-2, C1)
- `fix-tcgen05-ld-st-slot-routing` (FU-3, C2)
- `fix-tcgen05-multi-warp-fragment` (FU-4, C4)
- `tcgen05-flashattention-coverage` (FU-5)

### Validation
- 25/25 tcgen05 PASS (added 3 new tests: 1 integration + 1 parse + bonus TcQueue §29)
- Full ctest: ALL PASS. PTX grammar: 45/45 PASS.
- Zero regressions vs baseline `fd0fbb2` (24/24).
