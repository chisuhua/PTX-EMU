## Why

PTX-EMU 当前 `SMContext` 仅暴露**一个**外部注入点：`set_warp_scheduler(std::unique_ptr<WarpScheduler>)`。Scoreboard、Pipeline 延迟、TensorCore timing 的注入点**不存在**，外部 timing 模型（如 CppTLM Phase 8.B / gpgpu-sim / custom）无法替换 PTX-EMU 内置实现。

[CppTLM 任务书 §1.2] 审查确认当前隐式机制：

| 组件 | 现状 |
|------|------|
| Scoreboard | 无独立类；用 `WarpState::threads[lane].blocked_cycles_remaining` 隐式管理 |
| Pipeline 延迟 | `InstructionLatencyTable` 全局单例，通过 `load(InstructionLatencyConfig&)` JSON 覆盖 |
| TensorCore timing | 延迟来自 `tcgen05_handler.cpp` 内部硬编码或 JSON config |
| 寄存器信息 | 无 `dest_registers()` 暴露 API（`RegisterAnalyzer::analyze_registers()` 仅返回所有操作数，不区分 src/dst）|

**触发事件链**：

1. **2026-07-03**：[CppTLM 任务书](../../../../CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md)（803 行）发布，请求 PTX-EMU 新增 3 个纯虚接口 + 3 个 SMContext setter + `exe_once()` 改造
2. **2026-07-14**：[CppTLM ADR-NV-02 Status Update](../../../../CppTLM/docs/adr/ADR-NV-02-phase8b-d1-strategy.md) 将策略从 D1-Lite 升级为 D1-Full（WarpScheduler + Scoreboard + Pipeline + TensorCore 全部注入）
3. **2026-07-14**：[CppTLM 协作同步文档](../../../../CppTLM/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md) 追加 §13 D1-Full 双路径协作
4. **2026-07-14**：[CppTLM 实施计划](../../../../CppTLM/docs/superpowers/plans/2026-06-24-gpu-soc-phase8b.md) 修订为 D1-Full（573 行，含 Task 15a 4 个 Adapter）
5. **2026-07-16**：CppTLM 端 P0 归档（commit `b94eccc`）+ P2 AsyncCompletion 占位实施（commit `e69cd1d`）+ Phase 0 对齐 RFC 发送（commit `2b28505`，[`2026-07-16-rfcs-to-ptxemu-p1-injection.md`](../../../../CppTLM/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md)），提供：
   - **RFC-P1-001**: 3 接口签名（与本 change §3.1-§3.3 完全一致）
   - **RFC-P1-003**: 12-endpoint enum 锁定（双端 PipelineId 6 + TcPrecision 6 字字对应）
   - **RFC-P1-004**: Q1-Q5 答复（set_blocked_cycles_for_active 归属 + thread-safety + latency_mnk 退化 + AsyncCompletion 触发时机 + PipelineId 整数 lock）
   - **Phase 0 对齐结果**: PTX-0.1 / PTX-0.2 / PTX-0.4 已通过 CppTLM commit 锁定（详见 `internal-plan.md §5`）

**前置 change 假设**：

- CppTLM `openspec/changes/2026-06-24-gpu-soc-phase8b-core/` OpenSpec change 修订完成（D1-Full）✅
- CppTLM 任务书 `2026-07-03-ptxemu-modification-task.md` 已签发 ✅
- 本 change 与现有 active changes 序列化（详见 tasks.md §序列化考虑）：
  - `cleanup-deprecated-barrier-apis`：**已归档**（2026-06-20 per `openspec/changes/archive/2026-06-20-cleanup-deprecated-barrier-apis/`）— 前置条件已满足 ✅
  - `migrate-bar-warp-sync-to-barrier-module`：**已归档**（2026-07-03 per `openspec/changes/archive/2026-07-03-migrate-bar-warp-sync-to-barrier-module/`）— 并行协调已解除 ✅
  - `god-class-refactor-thread-context-phase3`（**并行**）：需关注 `blocked_cycles` 字段迁移路径

**关联 PTX-EMU 经验沉淀**（来自 `docs/dev-process/lessons-learned.md`）：

- **#4 基线 worktree**：实施前 1 分钟建立，节省数小时争论
- **#3 分 Phase commit**：每个 Phase 独立可回退；任何已有测试回归 → 立即 revert
- **#6 OpenSpec artifacts 2-Phase commit**：artifacts（proposal/design/tasks/spec）必须先 `git add` + commit，再实施代码
- **#1 跨模块状态翻译**：`blocked_cycles` 多处写入需集中审计（用 `state-modification-audit` skill）
- **#7 Pre-implementation Review**：实施前跑 Metis 审计（避免假设错误）

**关联 ADR**：

- [ADR-0020](../../docs/adr/ADR-0020-cpptlm-injection-points.md)：接受 CppTLM D1-Full 注入（核心决策）
- [ADR-0009](../../docs/adr/ADR-0009-xmacro-instruction-dispatch.md)：StatementType 枚举来源（X-Macro）
- [ADR-0008](../../docs/adr/ADR-0008-barrier-semantics.md)：barrier 语义（`blocked_cycles` 与 barrier 交互）
- [ADR-0019](../../docs/adr/ADR-0019-pc-management-extraction.md)：ThreadContext 瘦身（需关注 `blocked_cycles` 字段迁移）

**关联 Skill**：

- `ptx-lessons-learned`：经验沉淀快速决策树 + 4 个 checklist
- `state-modification-audit`：`blocked_cycles` 跨模块写入审计
- `ptx-instruction-pipeline`：`exe_once()` 上下文（指令执行流水线）
- `ptx-barrier-mechanism`：barrier 后 PC 处理与 `blocked_cycles` 交互
- `regression-bisect`：测试回归定位（实施过程中如有回归）

## What Changes

- **新增** `include/ptxsim/scoreboard_interface.h`：`IScoreboard` 纯虚基类（4 方法，零依赖）
- **新增** `include/ptxsim/pipeline_interface.h`：`IPipelineLatencyProvider` + `PipelineId` 枚举（0-5，零依赖）
- **新增** `include/ptxsim/tensor_core_interface.h`：`ITensorCoreTiming` + `TcPrecision` 枚举（0-5，零依赖）
- **修改** `include/ptxsim/sm_context.h`：+3 include + 3 setter + 3 getter + 3 私有成员（裸指针，默认 nullptr）
- **修改** `include/ptxsim/warp_context.h` + `.cpp`：新增 `set_blocked_cycles_for_active(uint32_t cycles)`
- **修改** `include/ptxsim/register_analyzer.h` + `src/ptxsim/register_analyzer.cpp`：新增 `get_dest_registers_as_ids(const StatementContext&) -> vector<uint32_t>`
- **修改** `src/ptxsim/core/sm_context.cpp`：`exe_once()` 三段式注入 + 4 个辅助函数
- **新增** `tests/unit/cpptlm/test_smcontext_injection.cpp`：7 个 Mock 测试用例（任务书 §5.2 完整移植）
- **新增** `tests/integration/cpptlm/test_scoreboard_allocation.cpp`：真实 warp + Mock scoreboard 集成测试
- **修改** `tests/CMakeLists.txt`：注册 2 个新测试目标（`unit_smcontext_injection` + `integration_scoreboard_allocation`）

## Capabilities

### New Capabilities

- `cpptlm-injection-points`: 4 个外部注入接口（`IScoreboard` + `IPipelineLatencyProvider` + `ITensorCoreTiming` + `set_blocked_cycles_for_active`），CppTLM D1-Full 集成必备
- `register-analyzer-dest-extract`: `RegisterAnalyzer::get_dest_registers_as_ids()` 区分 src/dst 寄存器提取

### Modified Capabilities

- `sm-context-exe-once`: `SMContext::exe_once()` 注入 3 处钩子（Scoreboard 检查 → 延迟查询 → Scoreboard 释放），nullptr 完全回退到原行为
- `warp-context-blocked-cycles`: `blocked_cycles_remaining` 设置从 per-thread LD-only 路径扩展为 per-warp 全指令可用
- `sm-context-public-api`: `SMContext` 新增 6 个 public 方法（3 setter + 3 getter），向后兼容

## Impact

| 文件 | 类型 | 工时 | 验证 |
|------|------|:---:|------|
| `include/ptxsim/scoreboard_interface.h` | **新增** | 0.1d | 编译通过 + `grep '#include'` 仅 `<cstdint>` |
| `include/ptxsim/pipeline_interface.h` | **新增** | 0.1d | 同上 + `<string>` |
| `include/ptxsim/tensor_core_interface.h` | **新增** | 0.1d | 同上 |
| `include/ptxsim/sm_context.h` | **修改** | 0.1d | 现有测试 0 回归 |
| `include/ptxsim/warp_context.h` + `.cpp` | **修改** | 0.1d | 编译 + 现有 LD-only 路径 0 回归 |
| `src/ptxsim/register_analyzer.h` + `.cpp` | **修改** | 0.2d | 现有 `analyze_registers()` 0 回归 + 新 PoC 测试 |
| `src/ptxsim/core/sm_context.cpp` | **修改** | 1.0d | nullptr 字节级回退 + `[unit;memory]` 0 回归 |
| `tests/unit/cpptlm/test_smcontext_injection.cpp` | **新增** | 0.5d | 7 个 Mock 测试 PASS |
| `tests/integration/cpptlm/test_scoreboard_allocation.cpp` | **新增** | 0.3d | RAW hazard 测试 PASS |
| `tests/CMakeLists.txt` | **修改** | 0.1d | `add_catch_test` 2 个新目标 |
| **合计** | | **~2.5d** | **9 个文件，~300 LOC** |

**影响类别**：

- **新增公共 API 表面**：SMContext +6 个 public 方法（set_scoreboard / set_pipeline_latency_provider / set_tensor_core_timing + 3 个 getter）+ WarpContext +1 个 public 方法（set_blocked_cycles_for_active）+ RegisterAnalyzer +1 个 public 方法（get_dest_registers_as_ids）
- **现有行为变更**：`exe_once()` 内部插入 3 处 if 分支（仅在 4 个注入点非 nullptr 时激活）
- **依赖关系**：3 个接口头文件仅依赖 `<cstdint>` / `<string>`（无 ANTLR4/Java/CUDA/CppTLM 依赖）

## References

- **CppTLM 任务书**：`CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md`（803 行，PTX-1~PTX-6 完整任务清单）
- **CppTLM 协同计划**：`CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-phase8b-d1full-plan.md`（440 行）
- **CppTLM ADR**：`CppTLM/docs/adr/ADR-NV-02-phase8b-d1-strategy.md` Status Update 2026-07-14
- **CppTLM 协作同步**：`CppTLM/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md` §13
- **CppTLM 实施计划**：`CppTLM/docs/superpowers/plans/2026-06-24-gpu-soc-phase8b.md`（573 行 D1-Full）
- **CppTLM OpenSpec change**：`CppTLM/openspec/changes/2026-06-24-gpu-soc-phase8b-core/`
- **PTX-EMU ADR-0020**：`docs/adr/ADR-0020-cpptlm-injection-points.md`（本 change 决策依据）
- **PTX-EMU 现有 OpenSpec changes**：
  - `cleanup-deprecated-barrier-apis`（**前置**，归档后启动本 change）
  - `god-class-refactor-thread-context-phase3`（**并行**，关注 blocked_cycles 字段迁移）
  - `migrate-bar-warp-sync-to-barrier-module`（**并行**，关注 barrier 后 PC 处理）

## ⚠️ 风险与历史教训（来自 `docs/dev-process/lessons-learned.md`）

1. **#4 基线 worktree 强制**：实施 PTX-1 前 1 分钟建立 baseline.txt，节省数小时争论
2. **#3 分 Phase commit**：每个 PTX-X commit 独立可回退；任何已有测试回归 → 立即 revert
3. **#6 OpenSpec artifacts 2-Phase commit**：本 change 5 个 artifacts 必须先 `git add` + commit，再实施代码
4. **#1 跨模块状态翻译**：`exe_once()` 改造涉及 `blocked_cycles` 多处写入，必须用 `state-modification-audit` skill 集中审计
5. **#7 Pre-implementation Review**：实施 PTX-6 前跑 Metis 审计（避免 `get_dest_registers_as_ids()` 实现假设错误）

## ⚠️ 序列化考虑

- **依赖关系**：`cleanup-deprecated-barrier-apis` 归档 → 本 change 启动 → 与 `god-class-refactor-thread-context-phase3` 并行实施（关注字段迁移）→ 与 `migrate-bar-warp-sync-to-barrier-module` 并行实施（关注 barrier 交互）
- **互斥字段**：本 change 新增 `scoreboard_/pipeline_provider_/tensor_core_timing_` 成员不与现有 barrier 字段冲突
- **测试基线**：必须确认实施前 `[unit;memory]` `[unit;barrier]` `[integration;simt]` 测试基线 100% PASS
