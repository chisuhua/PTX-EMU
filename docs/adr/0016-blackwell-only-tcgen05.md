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

## Archive 文档一致性声明（2026-07-08）

`openspec/changes/archive/2026-07-07-fix-tcgen05-antlr-prediction-bug/` 的 `proposal.md`/`design.md`/`handoff.md` 声称修复"ANTLR LL(*) 预测冲突",但真正根因是 **lexer bare string token 与 ID 规则冲突**（详见 [`docs/dev-process/lessons-learned.md` §25](../../docs/dev-process/lessons-learned.md#25-antlr4-le)）。按 `ptx-lessons-learned §6 + Checklist G` 铁律"已归档 change 不 amend",**本节为权威 override**: §25 为根因真相,archive 文档保留作为历史。