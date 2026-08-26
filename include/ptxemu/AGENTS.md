# include/ptxemu/ AGENTS.md

## OVERVIEW

Public device API for PTX-EMU (HSK-8 ack 738b412c) — abstract interface
consumed by external simulator consumers (CppTLM v3.0.0 via
`add_subdirectory(external/PTX-EMU)`). Single entry point: `IPtxEmuDevice`
in `device_api.h`. Internal implementation lives in `src/ptxemu/`,
linked as static library `ptxemu_core`.

## KEY FILES

| File | Role |
|------|------|
| `device_api.h` | Public API: `namespace ptxemu` + `IPtxEmuDevice` interface (12 pure virtual methods covering S1 facade callsites) + DTOs (`DeviceConfig`, `WarpStatus`, `LaneStatus`, `ThreadState` enum) + factory (`create_device` / `destroy_device`) + `PTXEMU_API_VERSION=1` guard macro |
| `ir/ptx_types.h` | IR enum types (`Qualifier`, `StatementType`, `OperandKind`) promoted from `include/ptx_ir/ptx_types.h` with `ptxemu::ir` namespace wrap (Phase 1) |
| `ir/operand_context.h` | `OperandContext` + 6 operand variant types (`RegOperand` / `VariableOperand` / `ImmOperand` / `AddrOperand` / `VecOperand` / `Predicate`) — post-Phase 0.3d clean (no `operand_phy_addr` field) |
| `ir/execution_types.h` | `InstructionState` enum (only public type from original `ptxsim/execution_types.h`) |
| `ir/statement.h` | 20 instruction struct types + `StatementContext` class + `InstrVariant` (std::variant<25>) — promoted from `include/ptx_ir/statement_context.h` |
| `ir/ptx_qualifier.def`, `ir/ptx_op.def` | X-Macro tables (parallel to `include/ptx_ir/` for CppTLM header inclusion path) |

## IPtxEmuDevice METHOD STATUS

All 12 `IPtxEmuDevice` pure virtual methods are wired (12/12 implemented)
as of `phase-2-2-1-3-1-followup` (2026-08-25). Implementation lives in
`src/ptxemu/device_api_impl.cc`.

| # | Method | Delegates to | Implemented |
|---|--------|--------------|-------------|
| 1 | `initialize(config)` | stores config | ✅ Phase 2.1 |
| 2 | `shutdown()` | clears `initialized_` flag | ✅ Phase 2.1 |
| 3 | `exe_once()` | `g_gpu_context->exe_once()` | ✅ Phase 2.1 |
| 4 | `sm_exe_once(sm_id)` | `g_gpu_context->get_sm()->exe_once()` | ✅ Phase 2.1 |
| 5 | `warp_exe_once(sm_id, warp_id)` | `warp->execute_warp_instruction(stmt, pc)` | ✅ Phase 2.2.1 |
| 6 | `set_scoreboard(sm, warp, mask)` | `sm->get_scoreboard()` registration check | ✅ Phase 2.2 |
| 7 | `get_thread_state(sm, warp, lane)` | `thread->get_state()` + `map_state` | ✅ Phase 2.2.1 |
| 8 | `set_active_mask(sm, warp, mask)` | `warp->set_active_mask(mask` (NOT OR-merge) | ✅ Phase 2.2 |
| 9 | `set_next_pc(sm, warp, lane, pc)` | `thread->set_pc() + commit_pc()` | ✅ Phase 2.2 |
| 10 | `get_warp_status(sm, warp)` | `warp->get_warp_state()` + `map_thread_status` | ✅ Phase 2.3.1 |
| 11 | `is_finished()` | `g_gpu_context->get_state() == IDLE` | ✅ Phase 2.1 |
| 12 | `attach_timing(sb, pl, tc)` | `sm->set_*` via `static_cast<void*>` round-trip | ✅ Phase 2.3 |

drift_check workflow Invariant 6 enforces 0 empty-body stubs (exemption
list EMPTY per `phase-2-2-1-3-1-followup` §3.7).

## CONVENTIONS

- **Namespace**: All public types in `ptxemu` (IPtxEmuDevice/DTOs) or `ptxemu::ir` (IR types). Per HSK-8 spec §2: 命名空间.
- **ABI freeze**: `PTXEMU_API_VERSION=1` macro is FROZEN (per HSK-8 spec §Decision 3). Any public signature change requires new HSK-N handshake (not in-PR bump).
- **C++17 compatibility**: `device_api.h` MUST NOT use C++20-only features (`std::format` / `requires` / `concept` / `<=>` / `consteval` / `constinit` / `[[likely]]` / `[[unlikely]]`). Per spec/public-device-api §Requirement C++17.
- **Static_assert lock**: impl layer MUST have `static_assert(static_cast<uint32_t>(ptxemu::ThreadState::kIdle) == static_cast<uint32_t>(::EXE_STATE::IDLE))` series (4 asserts covering kIdle/kRun/kExit/kBarSync, in `src/ptxemu/device_api_impl.cc` anonymous namespace). Per HSK-8 spec §Decision 6. Note: `EXE_STATE` is a global-namespace unscoped enum (`include/ptxsim/execution_types.h:8`), NOT in `ptxsim` namespace.
- **Forward-declared HSK-4 vendored interfaces**: `IScoreboard` / `IPipelineLatencyProvider` / `ITensorCoreTiming` are declared in `device_api.h` only (no duplicate definitions) — full definitions in vendored headers. Per HSK-8 spec §6 HSK-4 复用.

## ANTI-PATTERNS

- ❌ **不要在 `device_api.h` 中使用 C++20 特性** — drift_check workflow `drift_check.yml` Invariant 3 强制检查, 失败 workflow hard-fail
- ❌ **不要 bump `PTXEMU_API_VERSION` 在 PR 中** — 任何公共签名变更必须签发 HSK-N handshake; drift_check Invariant 1 失败
- ❌ **不要重命名 `ptxemu_core` CMake target** — CppTLM consumer 引用 `target_link_libraries(... ptxemu_core)`, drift_check Invariant 5 失败
- ❌ **不要在 `ptxemu::ir` 命名空间内添加实现状态字段** (mutable, lazy-init 等) — 违反 HSK-8 spec Decision 5 "sizeof visibility is mandatory, pure data only"
- ❌ **不要修改 `register_predicate.h` 中的 `RegisterPredicatePod`** — ThreadContext 重新声明了 `operand_collected` 等字段 (duplicated pattern), 修改 register_predicate 不会影响 ThreadContext