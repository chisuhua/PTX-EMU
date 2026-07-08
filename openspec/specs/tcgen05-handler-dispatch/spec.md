# tcgen05-handler-dispatch

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) (Blackwell-only tcgen05)
> **前置 spec**: [tcgen05-ir-types](../tcgen05-ir-types/spec.md)
> **同步来源**: [openspec/changes/fix-tcgen05-handler-dispatch/specs/tcgen05-handler-dispatch/spec.md](../../changes/fix-tcgen05-handler-dispatch/specs/tcgen05-handler-dispatch/spec.md)
> **关键变更**: 把已经定义的"shall dispatch"意图变为可观测的真实 wiring

## Purpose

The system SHALL wire all 11 `S_TCGEN05_*` StatementType enums (defined in `tcgen05-ir-types/spec.md`) to live dispatch handlers, so that PTX kernels containing `tcgen05.*` instructions actually execute their intended fragment arithmetic instead of silently terminating via the `_execute_once()` `nullptr` handler fallback.

This is the **implementation contract** that fulfills the dispatch-layer requirements originally stated as "shalls" in `tcgen05-ir-types/spec.md` (Requirement: "IR X-Macro Dispatch MUST Generate Tcgen05Handler Symbols" + Requirement: "PipelineHandler MUST Route S_TCGEN05_* to Tcgen05PipelineHandler").

## Requirements

### Requirement: 11 S_TCGEN05_* StatementType enums SHALL be registered via X-Macro

The `include/ptx_ir/ptx_op.def` file SHALL contain exactly 11 `X(...)` entries for `S_TCGEN05_*` StatementType enum values, each with `struct_kind = TCGEN05_INSTR` and the same `opstr = "Tcgen05"` literal:

- `S_TCGEN05_ALLOC` (1 operand)
- `S_TCGEN05_DEALLOC` (1 operand)
- `S_TCGEN05_RELINQUISH` (1 operand)
- `S_TCGEN05_LD` (2 operands)
- `S_TCGEN05_ST` (2 operands)
- `S_TCGEN05_CP` (3 operands)
- `S_TCGEN05_MMA` (4 operands)
- `S_TCGEN05_MMA_WS` (4 operands)
- `S_TCGEN05_COMMIT` (0 operands)
- `S_TCGEN05_WAIT` (0 operands)
- `S_TCGEN05_FENCE` (0 operands)

#### Scenario: ptx_op.def contains 11 S_TCGEN05_* X-Macro entries
- **WHEN** `grep -c "^X(S_TCGEN05_" include/ptx_ir/ptx_op.def` is run
- **THEN** output equals `11`

#### Scenario: ptx_types.h S_TCGEN05_* enums come from X-Macro expansion
- **WHEN** `grep -c "S_TCGEN05_" include/ptx_ir/ptx_types.h` is run
- **THEN** output equals `11` (count includes 11 references inside the X-Macro expansion block, NOT 11+11 duplicate definitions)
- **AND** `ptx_types.h` no longer contains a manual `S_TCGEN05_* = ...,` block outside the X-Macro `#include "ptx_op.def"` region

### Requirement: S_TCGEN05_* handlers SHALL be registered in handler_map

After `InstructionFactory::initialize()` returns, the `handler_map` SHALL contain entries for all 11 `S_TCGEN05_*` enums, each pointing to a non-null `InstructionHandler*`. The entries MUST be installed **via the X-Macro loop** in `instruction_factory.cpp:16-19` (i.e., the existing `handler_map[enum_val] = new opstr##Handler()` X-Macro expansion is sufficient — no additional explicit registration block).

#### Scenario: handler_map has 11 S_TCGEN05_* entries
- **WHEN** an integration test queries `InstructionFactory::get_handler(S_TCGEN05_MMA)` through `get_handler(S_TCGEN05_FENCE)` for all 11 enum values
- **THEN** none of the 11 calls returns `nullptr`

#### Scenario: get_handler returns same instance for repeat calls
- **WHEN** `get_handler(S_TCGEN05_MMA)` is called twice in succession
- **THEN** both calls return the same non-null `InstructionHandler*` (registration is consistent)

### Requirement: Tcgen05Handler class SHALL provide processTcgen05Operation method

A class named `Tcgen05Handler` SHALL exist in `namespace ptxsim` and provide a public method:

```cpp
void processTcgen05Operation(ThreadContext *context,
                             void **operands,
                             const std::vector<Qualifier> &qualifiers,
                             const Tcgen05Instr &instr);
```

The `Tcgen05Handler` SHALL be registered with each `S_TCGEN05_*` enum in `handler_map`.

The implementation MUST dispatch on `instr.op_kind` and invoke the corresponding fragment-arithmetic logic for `MMA` / `LD` / `ST` / `COMMIT` / `WAIT` (the 5 handlers implemented in `implement-tcgen05-handlers-core` at commit `df6dde7`).

#### Scenario: Tcgen05Handler declaration exists
- **WHEN** `grep "class Tcgen05Handler" include/ptxsim/instructions/tcgen05.h` is run
- **THEN** the file contains a class declaration

#### Scenario: processTcgen05Operation covers 5 core op_kinds
- **WHEN** the implementation file `src/ptxsim/instructions/tcgen05.cpp` is read
- **THEN** it contains a `switch (instr.op_kind)` with cases for at least: `MMA`, `LD`, `ST`, `COMMIT`, `WAIT`
- **AND** other op_kinds (ALLOC / DEALLOC / RELINQUISH / CP / MMA_WS / FENCE) throw `UnsupportedInstructionException` (per ADR-0016 Deferred-but-Wired)

### Requirement: Tcgen05PipelineHandler 3-phase pipeline SHALL route dispatch correctly

A class named `Tcgen05PipelineHandler : public PipelineHandler` SHALL exist in `include/ptxsim/instruction_base.h` and override the 3-phase pipeline methods:

```cpp
class Tcgen05PipelineHandler : public PipelineHandler {
public:
    bool prepareOperands(ThreadContext*, StatementContext&) override;
    bool executeOperation(ThreadContext*, StatementContext&) override;
    bool commitResults(ThreadContext*, StatementContext&) override;
};
```

The pipeline MUST:
1. In `prepareOperands`, handle empty `Tcgen05Instr.operands` (for zero-operand op_kinds like COMMIT/WAIT/FENCE) by returning `true` immediately
2. In `executeOperation`, call `Tcgen05Handler::processTcgen05Operation`
3. In `commitResults`, call `releaseAllOperands` to release the acquired operands

#### Scenario: pipeline reaches processTcgen05Operation for S_TCGEN05_MMA
- **WHEN** an integration test drives `step_warp` over a sequence that contains a `S_TCGEN05_MMA` statement
- **THEN** `Tcgen05PipelineHandler::executeOperation` is invoked exactly once for that statement
- **AND** `Tcgen05Handler::processTcgen05Operation` is called with the correct `Tcgen05Instr` reference

#### Scenario: pipeline handles zero-operand op_kinds without crash
- **WHEN** a `S_TCGEN05_COMMIT` statement (with empty `instr.operands`) is processed
- **THEN** `prepareOperands` returns `true` without invoking `acquireAllOperands`
- **AND** `executeOperation` is still called with an empty `operands` array
- **AND** `commitResults` skips `commit_operand` (since there is no dst operand)

### Requirement: _execute_once NULL handler fallback SHALL no longer be triggered for S_TCGEN05_*

After this change lands, executing any PTX kernel containing `S_TCGEN05_*` instructions SHALL NOT cause `ThreadContext::state` to be set to `EXIT` as a result of `nullptr` handler fallback.

The "No handler found for statement type: ..." stderr message SHALL NOT appear in test output for any tcgen05 op_kind.

#### Scenario: S_TCGEN05_MMA execution does not set_state(EXIT)
- **WHEN** a warp executes a `tcgen05.mma.kind::f16.cta_group::1 d, a, b, c` instruction
- **THEN** `warp_state.threads[lane].state != EXIT` after the instruction completes
- **AND** the lane's PC advances past the instruction (i.e., `pc` increments by 1)

#### Scenario: no stderr noise for tcgen05 dispatch
- **WHEN** `ctest -R tcgen05_dispatch -V` is run and any tcgen05 op_kind executes
- **THEN** the captured stderr contains no occurrence of "No handler found for statement type"

#### Scenario: all 11 tcgen05 op_kinds reachable through dispatch
- **WHEN** an integration test programatically invokes `step_warp` for each of the 11 `S_TCGEN05_*` statement types
- **THEN** all 11 complete without `set_state(EXIT)`

### Requirement: Existing baseline tests SHALL continue to pass

The 170+ existing tests MUST continue to pass after all phases land. The change in dispatch behavior MAY cause E2E tests (notably `e2e_blackwell_gemm`) to produce numerically different outputs if they were previously silently exiting on tcgen05 paths. Such differences MUST be documented in the relevant commit message.

#### Scenario: full ctest passes with zero regression in count
- **WHEN** `cd build && ctest --output-on-failure` is run after Phase 4
- **THEN** total test count is GREATER than or equal to pre-change baseline (170 + 7 from coverage + 1 from this change = at least 178)
- **AND** no previously-passing test now fails

#### Scenario: PTX syntax baseline still passes
- **WHEN** `./tests/ptx/test_all_ptx.sh` is run
- **THEN** all 12 `tcgen05_*.ptx` fixtures still parse correctly

#### Scenario: 7 dead-code-coverage tests from fix-tcgen05-test-coverage-gaps still pass as real-path tests
- **WHEN** `ctest -L "integration;tcgen05;parse"` and `ctest -L "unit;tcgen05;mma;golden"` are run after Phase 3
- **THEN** all 7 tests still pass — they were intentionally written to survive both dead-code and real-path scenarios
- **AND** the unit golden value test now exercises the real dispatch path (no longer "dead code coverage")

### Requirement: dealloc/cp/mma_ws/fence SHALL throw UnsupportedInstructionException via dispatch

For the 6 op_kinds that are registered in `handler_map` but not yet implemented in fragment arithmetic (ALLOC / DEALLOC / RELINQUISH / CP / MMA_WS / FENCE), executing them MUST throw `UnsupportedInstructionException` (per ADR-0016 §C5 fix #1). This is **intended deferral**, not a bug.

#### Scenario: tcgen05.alloc still throws when executed
- **WHEN** a warp executes a `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` instruction
- **THEN** an `UnsupportedInstructionException` is raised with a message referencing "tcgen05.alloc"
- **AND** the warp state transitions to `EXIT` (consistent with all unsupported instructions)

#### Scenario: deferred op_kinds no longer set_state(EXIT) silently via nullptr fallback
- **WHEN** a deferred op_kind executes
- **THEN** the exception is raised via `Tcgen05Handler::processTcgen05Operation`, NOT via the `_execute_once` nullptr fallback
- **AND** stderr contains a clear "UnsupportedInstructionException" message, NOT a "No handler found" message

## See also

- **Source change**: `openspec/changes/fix-tcgen05-handler-dispatch/`
- **Frontmatter spec**: `openspec/specs/tcgen05-ir-types/spec.md` (the design intent this implements)
- **Test coverage change**: `openspec/changes/fix-tcgen05-test-coverage-gaps/` (dead-code coverage tests that automatically become real-path tests when this change ships)
- **Extended handlers**: `openspec/changes/implement-tcgen05-handlers-extended/` (alloc/dealloc/cp/mma_ws/fence implementation, depends on this change)
