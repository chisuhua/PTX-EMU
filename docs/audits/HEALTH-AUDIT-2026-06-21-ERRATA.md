# HEALTH-AUDIT-2026-06-21.md 勘误表 (Errata)

> **官方说明**: 本 Errata 列出 `docs/audits/HEALTH-AUDIT-2026-06-21.md` 中的事实错误与遗漏。原审计作为 commit `baa8c4e` 的历史快照保持不变;本 Errata 是官方补充,供未来复审对比使用。
> **发布日期**: 2026-06-22
> **审查来源**: Oracle 子代理审查记录 `ses_1155c96adffeBJ5SwSGBXUpgYK` (2m 36s, 7 Q&A) + `ses_112932fcdffeSeDXitHeVPQIBG` (15m 58s, 实施计划审查)
> **关联文档**: [HEALTH-AUDIT-2026-06-21.md](./HEALTH-AUDIT-2026-06-21.md) (审计快照) | [phase-1-foundation.md](../../roadmap/phase-1-foundation.md) (修复规划)

---

## 1. 事实错误 (8 项)

### 1.1 ThreadContext public 字段数

| 字段 | 值 |
|------|-----|
| **审计声称** | 108 个 public 字段 |
| **实际情况** | **81** 个 public 字段 (`include/ptxsim/thread_context.h`, grep 实证) |
| **虚增幅度** | 33% |
| **影响范围** | §0.2 第五要点 / §1.2 M2 / §10.1 |
| **建议修正** | 将"108 public 字段"改为"81 public 字段,300 行" |

### 1.2 Symtable 泄漏数

| 字段 | 值 |
|------|-----|
| **审计声称** | 5 处 (仅 `src/cudart/ptx_interpreter.cpp:213,302,443,459,550`) |
| **实际情况** | **7 处** (上述 5 处 + `src/ptxsim/core/cta_context.cpp:74,104` **漏报 2 处**) |
| **影响范围** | §0.2 第三要点 / §2.2.1 |
| **建议修正** | 从"5 处"改为"7 处 (5 in ptx_interpreter.cpp + 2 in cta_context.cpp)" |

### 1.3 `ptx_visiter` 拼写错误影响文件数

| 字段 | 值 |
|------|-----|
| **审计声称** | 14 个 `.cpp` 文件引用 |
| **实际情况** | **18 个文件** 引用 (grep 实证) |
| **影响范围** | §1.2 M1 |
| **建议修正** | 从"14 个 .cpp 引用"改为"18 个文件引用" |

### 1.4 H2 反向依赖严重度过高

| 字段 | 值 |
|------|-----|
| **审计声称** | 🔴 H 级 (必须修复,影响模块边界正确性) |
| **实际情况** | 🟡 M 级 — `include/ptxsim/execution_types.h:8` 是 4 值枚举 `enum EXE_STATE { IDLE, RUN, EXIT, BAR_SYNC }`,是合法的叶子基础类型,非真正违规 |
| **影响范围** | §1.2 H2 |
| **建议修正** | 降为 M 级;若坚持修复,应移到 `include/ptx_ir/execution_state.h` (ptx_ir 是 IR 根) 而非 `include/utils/` (避免 utils 大杂烩陷阱) |

### 1.5 P0-1 membar/fence 工作量低估

| 字段 | 值 |
|------|-----|
| **审计声称** | 2 天 |
| **实际情况** | **2-3 天** (未计入 DUAL STATE MECHANISM 修复时间 + "已知答案测试"编写时间) |
| **影响范围** | §0.4 优先级表 / §3.5 |
| **建议修正** | 从"2 天"改为"2-3 天" |

### 1.6 Phase 1 顺序不当

| 字段 | 值 |
|------|-----|
| **审计声称** | P0-1 (membar) → P0-2 (泄漏) → P0-3 (CI) |
| **实际情况** | **P0-4 (compile_commands 5min) → P0-3 (CI 0.5-1.5d) → P0-2 (泄漏 1d) → P0-1 (membar 2-3d)** — 无 CI = 无回归门禁,CI 必须先于所有正确性修复 |
| **影响范围** | §8 Phase 1 |
| **建议修正** | 调整顺序,把 CI 提升到第 2 位 |

### 1.7 cudaStream_t 性质误判 — **destroy 实现存在但 type-unsafe**

| 字段 | 值 |
|------|-----|
| **审计声称** | "句柄泄漏" 暗示 destroy 是 no-op / "漏写 delete" |
| **实际情况** | **destroy 函数实现存在** — `src/cudart/cudart_sim.cpp:688-696 cudaStreamDestroy` 实际包含 `delete reinterpret_cast<int *>(stream);`,**不是** no-op STUB。审计对 destroy 实现的判断错误。**实际问题是**: (a) `reinterpret_cast<int*>` type-unsafe (cudaStream_t 应为 void*,不是 int*); (b) `cudaStreamSynchronize:698-703` 是 no-op (`return cudaSuccess` 无实际同步); (c) `cudaEventElapsedTime:741-747` 返回硬编码 1.0f。Multi-stream 语义被破坏是 fake synchronization 导致,不是句柄泄漏。 |
| **影响范围** | §2.2.1 cudaStream_t 行 |
| **建议修正** | 区分"漏写 delete"和"destroy 实现 type-unsafe";multi-stream 语义破坏是 fake sync 而非泄漏 |

### 1.8 PTX 8.7+ 占位决策偏差

| 字段 | 值 |
|------|-----|
| **审计声称** | 选项 A (删除) 或 B (3-6 月实现) 推荐,选项 C (维持现状) 作为同等候选 |
| **实际情况** | 选项 C 是**最危险的静默失败** — 17 条占位全部 `IMPLEMENT_SIMPLE_HANDLER`,用户 PTX 含 `cp.async`/`tcgen05.*` 编译通过、运行结果错误、无报错 |
| **影响范围** | §3.5 / §9.1 D-A |
| **建议修正** | 推荐 **A + PTX_WARN** 组合 (visitor 阶段对未识别指令 emit 警告),比"维持现状"更诚实的 graceful degradation |

---

## 2. 严重遗漏 (1 项)

### 2.1 BarWarpSyncHandler 仍用 deprecated `warp_state.wbars[]`

| 字段 | 值 |
|------|-----|
| **审计状态** | 完全未提及 |
| **实际情况** | `src/ptxsim/core/AGENTS.md` 明确记录 `BarWarpSyncHandler` STILL uses `warp_state.wbars[]` (Phase 5 deferred, see `cleanup-deprecated-barrier-apis` change);Wbar struct 已 `[[deprecated]]`。grep 实证仍在 `warp_context.cpp:287`, `barrier.cpp:161,215` 三处使用 |
| **关键约束** | `set_active_mask` MUST NOT be OR-merged with arrived_mask globally — OR logic lives in `BarrierModule::release_warp_barrier` (per AGENTS.md) |
| **影响** | 阻塞 P0-1 (membar/fence 实现) — 不先清理会导致 `bar.warp.sync` 路径仍走旧机制 (双重状态机风险) |
| **建议补充** | 作为**隐藏 P0** 加入 Phase 2,在 T1-4 (membar/fence) 之前完成 |
| **Phase 2 映射** | T1-3 (migrate-warp-barrier-handler),见 `docs/roadmap/phase-2-critical-debt.md` |

---

## 3. 优先级调整建议 (已采纳)

| 审计原顺序 | 修正后顺序 | 理由 |
|---|---|---|
| Phase 1: P0-1→P0-2→P0-3 | **P0-4→P0-3→P0-2→P0-1** | 无 CI = 无回归门禁;CI 是根因解锁项 |
| H2 严重度: 🔴 H | 🟡 M | 4 值枚举是合法叶子类型,非真正违规 |
| PTX 8.7+ 占位: A 或 B | **A + PTX_WARN** | 维持现状 (选项 C) 是最危险的静默失败 |
| CI 首次失败: 未明示 | **xfail 不阻塞 PR** | 避免一次性过载,单独 issue 跟踪修复 |
| P0-1 工作量: 2 d | 2-3 d | 未计 DUAL STATE 修复时间 |

---

## 4. 决策日志 (2026-06-22 用户采纳)

1. ✅ 接受 Tier 0/1 优先级调整 (CI 优先于 membar)
2. ✅ 修正审计文档 8 处事实错误 (本 Errata)
3. ✅ PTX 8.7+ 占位去留: A + PTX_WARN (Phase 3 T2-4 实施)
4. ✅ CI 启用后首次失败处理: xfail 不阻塞 PR
5. ✅ H2 反向依赖降为 M 级

---

## 5. 关联引用

- Oracle 审查会话: `ses_1155c96adffeBJ5SwSGBXUpgYK` (Phase 1 设计审查), `ses_112932fcdffeSeDXitHeVPQIBG` (实施计划审查)
- 审计原文档: `HEALTH-AUDIT-2026-06-21.md` (commit `baa8c4e`)
- Roadmap 实施计划: `docs/roadmap/`
- Phase 1 change: `openspec/changes/phase-1-foundation/`
- 实施计划: `docs/superpowers/plans/2026-06-22-phase-1-foundation.md`

---

**本 Errata 由 Phase 1 T0-3 实施创建,作为审计的官方补充。审计快照保持不变,未来季度复审应创建新 Errata (例如 2026-09-21 Errata v2) 而非修改本文件。**
