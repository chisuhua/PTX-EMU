## Why

当前 `src/ptxsim/instructions/atomic.cpp` (115 行) 显式排除 CAS 操作 (line 55-58: "CAS is out-of-scope"),导致 PTX 程序中任何 `atom.cas.b32` 等 CAS 指令静默 no-op (无 warning 抛出)。这违反了 [PTX ISA §9.7.12 atomic](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#atomic-instructions-cas) 语义,且让 CUTLASS 类库的 lock-free 累加模式无法在模拟器中正确执行,直接阻塞上层的 H5+ 路线图 (post-tcgen05-roadmap.md §3.3 F8 cute_rmsnorm upgrade)。

本次实施将由原始 audit P1 债务(A-9 + C-16 共 8h)拆为 3 Phase,本 Phase 1 (3h) 仅实施 CAS handler 的解析与算法,Phase 2 (3h) 引入真正原子性 mutex (per-warp serialize + cross-warp mutex),Phase 3 (2h) 添加 multi-warp oracle test。

## What Changes

- **新增** Capability `atomic-cas-handler`: 实现 `atomic.exch` 与 `atomic.compare_and_swap (atomic.cas)` 的 handler (`src/ptxsim/instructions/atomic.cpp`)
- **新增** Capability `atomic-true-atomicity` (Phase 2/3): per-warp 串行化 + cross-warp mutex + 多 warp oracle 测试
- **新增** Capability `atomic-cas-oracle-test` (Phase 3): e2e + integration 测试覆盖 multi-warp CAS 正确性
- **不修改** `.g4` grammar (CAS qualifier `.cas` 已通过 `Q_CAS_ATOM` 在 `ptx_qualifier.def:251` 唯一映射,无需 remap)
- **不修改** `ptx_op.def` (opcount=3 配合 visitor 循环 `for (i=2; i<min(size, opcount); ++i)` 已正确处理 4-operand CAS 收集 — 见 design.md §MR-2)
- **不实现** Memory ordering qualifier (`.relaxed`/`.acq_rel`/`.scope`) 语义 — parser 接受但不强制行为

## Capabilities

### New Capabilities
- `atomic-cas-handler`: 解析 + 算法实现 CAS handler (Phase 1,本会话目标)
- `atomic-true-atomicity`: 引入真正 atomicity mutex,防止 multi-warp race (Phase 2/3,后续会话)

### Modified Capabilities
- 无 (现有 spec 不直接覆盖 atomic;Phase 1 不改变 spec-level 行为,仅扩展 handler 实现)

## Impact

### 受影响代码 (Phase 1)
- **新增** `src/ptxsim/instructions/atomic.h` API: `AtomHandler::processAtomicCAS(...)` 4-operand 签名 (dst, addr, cmp, val)
- **修改** `src/ptxsim/instructions/atomic.cpp` (115→~150 行):
  - line 36-53 `atom_op` 检测循环添加 `case Qualifier::Q_CAS_ATOM:`
  - line 56-58 移除 "CAS is out-of-scope" 排除逻辑
  - 新增 `processAtomicCAS` 函数实现 (load → compare → conditional store → write-back old)
- **新增** 测试:
  - `tests/unit/atomic/test_cas_handler_basic.cpp` (类型一)
  - `tests/integration/atomic/test_atom_global_cas.cpp` (类型二)
  - `tests/ptx/atom_cas_basic.ptx` (PTX 语法样本,test_all_ptx.sh 验证)
- **不修改**: `ptx_op.def` (opcount=3 已足够),ANTLR `.g4`,qualifier enum,parser,visitor

### 不影响 (Phase 2/3,本次不实施)
- 现有 9 个非 CAS atom 操作 (`atom.add`/`atom.and`/.../`atom.max`) — 保持原行为
- 现有 2 个集成测试 `integration_ptx_atom_global_add` + `integration_ptx_atom_global_exch`
- 屏障系统 (`barrier_module`) — 仅在 Phase 2 mutex 设计时需要考虑锁序 (per `lessons-learned.md §2`)
- CUDA Runtime (`cudart_sim.cpp`) — 与本 change 无交集

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- [N/A] Phase 1 不迁移函数,仅在 `AtomHandler` 类新增 `processAtomicCAS` 方法
- [x] design.md §MR-2 已记录 4-operand 传递路径验证 (visitor 循环 line 75-77)
- [N/A] 无 mutex 设计 (Phase 1);Phase 2 design.md §MR-5 已锁定 mutex 必须 audit 现有锁点

### 多 Phase 推进
- [x] Phase 拆分: Phase 1 (handler ~3h, Tier 1) / Phase 2 (mutex ~3h, Tier 2) / Phase 3 (oracle ~2h, Tier 2)
- [x] 基线 worktree 命令已记录 (Phase 2/3 强制): `git worktree add .worktrees/baseline-cas-phase2 main`
- [x] 失败处理策略 (Phase 1): 任何已有测试回归 → 立即 revert Phase 1 commit,不混入后续

### 文档同步
- [x] `src/ptxsim/instructions/AGENTS.md` 在 Phase 1 完成后追加 "CAS handler" 章节
- [x] `openspec/changes/archive/` 路径已建立的 `Ref:` lineage (本 change → future archive/) 将记录 Phase 2/3 顺序
- [x] tasks.md Phase 状态变更明确定义: Phase 1 完成前不允许启动 Phase 2

### Scope 边界 (MR-6 锁定)
- [x] 不实现 `.relaxed`/`.acq_rel`/`.scope` memory ordering 语义
- [x] 不实现 `atom.shared.cas`/`atom.cta.cas` 等非 global 空间 CAS (Phase 1 仅 global)
- [x] 不修改 ANTLR `.g4` grammar
- [x] 不修改 `ptx_op.def` opcount (3 已足够,design.md §MR-2 验证)

### 验证
- [x] **MR-1 (CAS 解析)** — `Q_CAS_ATOM` 在 `ptx_qualifier.def:251` 唯一存在,无需 DOT 冲突 remap (`ptx_visitor_atom.cpp:88-94` 不含 CAS 项)
- [x] **MR-2 (4-operand 传递)** — visitor 循环 `ptx_visitor_atom.cpp:75-77` 已正确收集 4 operands;handler 需新签名
- [x] **MR-3 (baseline worktree)** — 推迟到 Phase 2 启动前;Phase 1 不需要 (scope 明确,无 race 风险)
- [x] **MR-4 (Phase 1 测试验收)** — 见 `specs/atomic-cas-handler/spec.md` 5 个 SHALL Requirements
- [x] **MR-5 (Phase 2 mutex 审计)** — Phase 2 启动时必跑:`grep -rn "mutex_\|lock_guard" src/ptxsim/` + 锁序证明
- [x] **MR-6 (scope 边界)** — 见上方本节 Scope 边界列表

## Refs

- Debt audit: `docs/audits/debt-audit-2026-07-02.md` §2.1 A-9 + §3.2 C-16
- Roadmap: `docs/roadmap/post-phase3-debt-roadmap.md` §1.1 A-9 + §3.2 (8h estimate,Phase 1/2/3 split)
- Lessons-learned: `docs/dev-process/lessons-learned.md` §2 (recursive lock) + §5 (qualifier.back)
- Metis pre-impl audit: `bg_566b7fc3` (parallel agent invocation,详见 session log)
- PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#atomic-instructions-cas>
