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

## CONVENTIONS

- **Namespace**: All public types in `ptxemu` (IPtxEmuDevice/DTOs) or `ptxemu::ir` (IR types). Per HSK-8 spec §2: 命名空间.
- **ABI freeze**: `PTXEMU_API_VERSION=1` macro is FROZEN (per HSK-8 spec §Decision 3). Any public signature change requires new HSK-N handshake (not in-PR bump).
- **C++17 compatibility**: `device_api.h` MUST NOT use C++20-only features (`std::format` / `requires` / `concept` / `<=>` / `consteval` / `constinit` / `[[likely]]` / `[[unlikely]]`). Per spec/public-device-api §Requirement C++17.
- **Static_assert lock**: impl layer MUST have `static_assert(static_cast<uint32_t>(ptxemu::ThreadState::kIdle) == static_cast<uint32_t>(ptxsim::EXE_STATE::IDLE))` series. Per HSK-8 spec §Decision 6.
- **Forward-declared HSK-4 vendored interfaces**: `IScoreboard` / `IPipelineLatencyProvider` / `ITensorCoreTiming` are declared in `device_api.h` only (no duplicate definitions) — full definitions in vendored headers. Per HSK-8 spec §6 HSK-4 复用.

## ANTI-PATTERNS

- ❌ **不要在 `device_api.h` 中使用 C++20 特性** — drift_check workflow `drift_check.yml` Invariant 3 强制检查, 失败 workflow hard-fail
- ❌ **不要 bump `PTXEMU_API_VERSION` 在 PR 中** — 任何公共签名变更必须签发 HSK-N handshake; drift_check Invariant 1 失败
- ❌ **不要重命名 `ptxemu_core` CMake target** — CppTLM consumer 引用 `target_link_libraries(... ptxemu_core)`, drift_check Invariant 5 失败
- ❌ **不要在 `ptxemu::ir` 命名空间内添加实现状态字段** (mutable, lazy-init 等) — 违反 HSK-8 spec Decision 5 "sizeof visibility is mandatory, pure data only"
- ❌ **不要修改 `register_predicate.h` 中的 `RegisterPredicatePod`** — ThreadContext 重新声明了 `operand_collected` 等字段 (duplicated pattern), 修改 register_predicate 不会影响 ThreadContext