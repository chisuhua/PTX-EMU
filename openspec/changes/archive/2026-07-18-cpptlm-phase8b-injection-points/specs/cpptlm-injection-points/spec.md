## MODIFIED Requirements

### Requirement: SMContext MUST accept CppTLM injection points via pure-virtual interfaces
The `SMContext` MUST provide three new public setters (`set_scoreboard`, `set_pipeline_latency_provider`, `set_tensor_core_timing`) and three getters that accept pure-virtual interfaces (`IScoreboard`, `IPipelineLatencyProvider`, `ITensorCoreTiming`) with zero external dependencies. The three interfaces MUST be defined in three separate header files (`include/ptxsim/{scoreboard,pipeline,tensor_core}_interface.h`), each including only `<cstdint>` and/or `<string>`. Default value of all three pointers MUST be `nullptr`; behavior with all-nullptr MUST be byte-identical to pre-change `SMContext`.

#### Scenario: nullptr injection = byte-identical backward compatibility
- **WHEN** `SMContext` constructed with default nullptr injection points
- **THEN** `get_scoreboard() == nullptr`
- **AND** `get_pipeline_latency_provider() == nullptr`
- **AND** `get_tensor_core_timing() == nullptr`
- **AND** `exe_once()` 输出与改造前**字节级相同**（通过 baseline worktree 对照）

#### Scenario: Scoreboard injection limit concurrent operations
- **WHEN** `MockScoreboardLimited(12 entries)` 通过 `sm.set_scoreboard(&sb)` 注入
- **AND** warp 提交 13 个相同 `dest_reg` 的指令
- **THEN** 前 12 条 `allocate(reg_id, warp_id)` 成功
- **AND** 第 13 条 `allocate(reg_id, warp_id)` 返回 false（RAW hazard）
- **AND** warp 被 stall，`is_blocked == true`

#### Scenario: Scoreboard release after instruction completes
- **WHEN** `MockScoreboardLimited(1 entry)` 注入
- **AND** 发射 1 条指令 → `allocate` 成功 → `used=1`
- **AND** 指令执行完成 → `release` 被调用 → `used=0`
- **THEN** 下一条指令可以正常发射

#### Scenario: Pipeline injection overrides InstructionLatencyTable
- **WHEN** `MockPipelineFixed(4.22 cyc)` 通过 `sm.set_pipeline_latency_provider(&pipe)` 注入
- **AND** warp 执行 FFMA 指令
- **THEN** `blocked_cycles == 5` (ceil of 4.22)
- **AND** **NOT** `InstructionLatencyTable` 的默认值

#### Scenario: TensorCore injection overrides default TC latency
- **WHEN** `MockTensorCoreFixed(29)` 通过 `sm.set_tensor_core_timing(&tc)` 注入
- **AND** warp 执行矩阵乘法指令（如 HMMA）
- **THEN** `blocked_cycles == 29`（TC 注入值）

#### Scenario: TensorCore falls back when Pipeline returns 0
- **WHEN** `MockPipelineFixed(0.0)` 注入 + `MockTensorCoreFixed(29)` 注入
- **AND** warp 执行 TC 指令
- **THEN** pipeline 返回 0.0 → 退回到 TC → `blocked_cycles == 29`

#### Scenario: All three injection points active simultaneously
- **WHEN** `MockScoreboardRAW` + `MockPipelineFixed(2.22)` + `MockTensorCoreFixed(29)` 同时注入
- **AND** warp 执行多种指令（整数 / 浮点 / 矩阵）
- **THEN** 整数指令 → pipeline 延迟生效
- **AND** 浮点指令 → pipeline 延迟 + scoreboard 跟踪
- **AND** 矩阵指令 → pipeline 返回 0 → TC 延迟生效
- **AND** 完整链路无 crash

---

### Requirement: WarpContext MUST provide per-warp blocked_cycles extension
The `WarpContext` MUST provide a new public method `set_blocked_cycles_for_active(uint32_t cycles)` that iterates `warp_state_.threads` and sets `blocked_cycles_remaining = cycles; is_blocked = true` for all threads with `is_active && !is_blocked`. The method MUST NOT modify `ThreadState` layout and MUST preserve existing LD-only per-thread path (`memory.cpp::LdHandler::processOperation()`).

#### Scenario: Set blocked_cycles on all active non-blocked threads
- **WHEN** `warp->set_blocked_cycles_for_active(5)` 调用
- **AND** warp 内 32 threads 状态：`thread[0].is_active=true, is_blocked=false`
- **AND** `thread[1].is_active=true, is_blocked=true`（已阻塞，跳过）
- **AND** `thread[2].is_active=false`（非活跃，跳过）
- **THEN** `thread[0].blocked_cycles_remaining == 5`
- **AND** `thread[0].is_blocked == true`
- **AND** `thread[1].blocked_cycles_remaining` 不变
- **AND** `thread[2].blocked_cycles_remaining` 不变

#### Scenario: Extend blocked_cycles to non-LD instructions
- **WHEN** warp 执行 ADD 指令（非 LD）
- **AND** `pipeline_provider_` 注入返回 `2.22 cyc`
- **THEN** `set_blocked_cycles_for_active(3)` 被调用（ceil of 2.22）
- **AND** 3 个 tick 后 warp 解除阻塞
- **AND** **NOT** 仅依赖 LD-only per-thread 路径

---

### Requirement: RegisterAnalyzer MUST distinguish src and dest registers
The `RegisterAnalyzer` MUST provide a new public static method `get_dest_registers_as_ids(const StatementContext&) -> std::vector<uint32_t>` that extracts only destination (write) register IDs, distinguishing them from source registers. The new method MUST NOT modify the existing `analyze_registers()` method. The implementation MUST use `std::visit` on `StatementContext.data` variant with `if constexpr (requires { ... })` checks for `instr.operands` and `instr.dest` fields.

#### Scenario: Extract dest register from simple arithmetic instruction
- **WHEN** `add.f32 %f1, %f2, %f3;` 语句
- **THEN** `get_dest_registers_as_ids(stmt) == [%f1]`
- **AND** **NOT** 包含 `%f2` 或 `%f3`

#### Scenario: Extract dest register from store instruction
- **WHEN** `st.global.f32 [%rd1], %f1;` 语句
- **THEN** `get_dest_registers_as_ids(stmt) == [%f1]`（store 的 dest 是 value 不是 address）

#### Scenario: Backward compatibility with existing analyze_registers
- **WHEN** 现有代码调用 `RegisterAnalyzer::analyze_registers(statements)`
- **THEN** 返回所有操作数（不区分 src/dst）
- **AND** 现有用户代码 0 回归

---

### Requirement: SMContext::exe_once MUST support three-step CppTLM injection
The `SMContext::exe_once()` MUST insert three injection points: (A) Scoreboard hazard check before instruction execution, (B) latency query (Pipeline → TensorCore → InstructionLatencyTable fallback chain) before execution, (C) Scoreboard release after execution. All three points MUST be no-ops when the corresponding injection pointer is `nullptr`, ensuring byte-identical backward compatibility.

#### Scenario: Step A Scoreboard check (nullptr = skip)
- **WHEN** `scoreboard_ == nullptr`
- **THEN** Step A 完全跳过
- **AND** 不调用 `scoreboard_->tick() / has_free_entry() / allocate()`

#### Scenario: Step A Scoreboard hazard detection
- **WHEN** `scoreboard_ != nullptr` + warp 有 RAW hazard
- **THEN** `has_free_entry()` 返回 false OR `allocate()` 返回 false
- **AND** warp 不发射当前指令
- **AND** `cycle_counter_` 不推进（不消耗功能性 cycle）

#### Scenario: Step B latency priority chain
- **WHEN** 4 个注入点都非 nullptr
- **THEN** 优先级链：`pipeline_provider_ > tensor_core_timing_ > InstructionLatencyTable`
- **AND** Pipeline 返回 >0.0 → 使用 `ceil()` 后的 uint32_t
- **AND** Pipeline 返回 0.0 + TC 指令 → TC 延迟
- **AND** Pipeline 返回 0.0 + 非 TC 指令 → `InstructionLatencyTable` 默认值

#### Scenario: Step C Scoreboard release
- **WHEN** `scoreboard_ != nullptr` + 指令执行完成
- **THEN** 对 `dest_registers` 每个 reg_id 调用 `scoreboard_->release(reg_id, warp_id)`

#### Scenario: Byte-identical backward compatibility
- **WHEN** 4 个注入点全 nullptr
- **THEN** `exe_once()` 输出与改造前**字节级相同**
- **AND** 通过 baseline worktree 对照测试验证

---

### Requirement: Enum values MUST match CppTLM internal types for Adapter static_assert
The `PipelineId` (0-5) and `TcPrecision` (0-5) enums in PTX-EMU MUST exactly match the values of `tlm::PipelineId` and `tlm::TcPrecision` enums in CppTLM. The `StatementType` enum (generated by X-Macro from `include/ptx_ir/ptx_op.def`) MUST remain stable to avoid breaking CppTLM PipelineTLM internal mapping.

**🔒 锁定来源**（2026-07-16 Phase 0 对齐）:
- CppTLM commit `2b28505` (RFC-P1-003 §3.1 §3.2) — 双端 12-endpoint 字字对应
- 完整对照表见 `internal-plan.md §5.1`
- CppTLM 端文档：`CppTLM/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md`

#### Scenario: PipelineId enum value alignment
- **WHEN** CppTLM Adapter 编译时执行 `static_assert(static_cast<uint32_t>(::PipelineId::P0_INT_FP32) == static_cast<uint32_t>(tlm::PipelineId::P0_INT_FP32))`
- **THEN** 编译通过（6 个端点全部对齐）

#### Scenario: TcPrecision enum value alignment
- **WHEN** CppTLM Adapter 编译时执行 `static_assert(static_cast<uint32_t>(::TcPrecision::FP4) == static_cast<uint32_t>(tlm::TcPrecision::FP4))`
- **THEN** 编译通过（6 个精度端点全部对齐）

#### Scenario: StatementType stability
- **WHEN** PTX-EMU 修改 `include/ptx_ir/ptx_op.def` 添加新指令
- **THEN** 必须通知 CppTLM 团队同步更新 PipelineTLM 映射表
- **AND** 在 `tasks.md` 中记录变更条目

---

### Requirement: ADR-0020 status MUST be Accepted before implementation
ADR-0020 (`docs/adr/0020-cpptlm-injection-points.md`) status MUST transition from Proposed → Accepted (2026-07-16) before the cpptlm-phase8b-injection-points change enters implementation. This gates the spec-driven lifecycle per OpenSpec Checkpoint G.

#### Scenario: ADR README reflects Accepted status
- **WHEN** `grep "0020" docs/adr/README.md` is run
- **THEN** ADR-0020 SHALL be listed in the **Accepted** section (not Proposed)
- **AND** the ADR document `docs/adr/0020-cpptlm-injection-points.md` SHALL have `Status: Accepted` (transitioned 2026-07-16)

#### Scenario: Implementation blocked until Accepted
- **WHEN** `openspec apply --change cpptlm-phase8b-injection-points` is invoked
- **AND** ADR-0020 status is still Proposed
- **THEN** the apply command SHALL refuse to proceed until ADR-0020 transitions to Accepted
