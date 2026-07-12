## ADDED Requirements

### Requirement: Tcgen05Instr.cta_group SHALL be populated from .cta_group::N qualifier IMMEDIATE

The system SHALL modify `visitTcgen05Inst` (in `src/ptx_parser/ptx_visitor.cpp`) to extract the numeric value following `.cta_group::` and store it in `Tcgen05Instr::cta_group` (in `include/ptx_ir/statement_context.h:186`). The extraction SHALL use a separate parse-tree walk after `extractQualifiersFromContext` (Option (b) per design.md D1), preserving the existing `extractQualifiersFromContext` signature so that the 20 other callers remain unchanged. The parse-tree walk SHALL use `ctx->tcgen05Qual()` (NOT `ctx->tcgen05QualList()`, which does not exist — verified at `build/antlr4_generated_src/ptxParser.h:3967`).

#### Scenario: .cta_group::1 populates instr.cta_group = 1

- **WHEN** PTX source contains `tcgen05.mma.kind::f16.cta_group::1 [addr], a, b, i;`
- **THEN** after `visitTcgen05Inst` returns
- **AND** `instr.cta_group == 1u`
- **AND** `instr.qualifiers` contains `Q_TCGEN_CTA_GROUP`

#### Scenario: .cta_group::2 populates instr.cta_group = 2

- **WHEN** PTX source contains `tcgen05.mma.kind::f16.cta_group::2 [addr], a, b, i;`
- **THEN** after `visitTcgen05Inst` returns
- **AND** `instr.cta_group == 2u`

#### Scenario: omitted cta_group qualifier defaults to 1

- **WHEN** PTX source contains `tcgen05.commit;` (no `.cta_group::N` qualifier)
- **THEN** `instr.cta_group == 1u` (the existing default value per `statement_context.h:186`)

#### Scenario: malformed cta_group IMMEDIATE is not gracefully handled

- **WHEN** PTX source contains an empty IMMEDIATE (defensive — should not occur in well-formed PTX as ANTLR grammar `TCGEN_CTA_GROUP COLONCOLON IMMEDIATE` requires IMMEDIATE after `::`)
- **THEN** `std::stoul("")` throws `std::invalid_argument` — this is **expected behavior** (no defensive validation per design.md D1 OpenSpec acceptance criteria "handler 不校验；OpenSpec 接受 PTX 字面量语法限制")
- **AND** the existing exception propagates up — no `tcgen05.cta_group` silent fallback

### Requirement: processTcgen05Commit SHALL use instr.cta_group instead of hardcoded 1

The system SHALL modify `processTcgen05Commit` (in `src/ptxsim/instructions/tcgen05.cpp:493`) to invoke `cta->tc_queue().commit(instr.cta_group)` instead of the current hardcoded `cta->tc_queue().commit(1)`. The `(void)instr;` cast at line 493 SHALL be removed since `instr.cta_group` is now consumed.

#### Scenario: cta_group::1 commit preserves existing behavior

- **WHEN** `processTcgen05Commit` is called with a `Tcgen05Instr{cta_group=1}`
- **THEN** `cta->tc_queue().commit(1)` is invoked
- **AND** the TcQueue counter advances for group 1
- **AND** no regression in `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` T2/T3 (`mma→commit` sequence)

#### Scenario: cta_group::2 commit advances the TcQueue monotonic-max counter

- **WHEN** `processTcgen05Commit` is called with a `Tcgen05Instr{cta_group=2}`
- **THEN** `cta->tc_queue().commit(2)` is invoked
- **AND** the TcQueue `commit_group_counter_` (a single `std::atomic<uint64_t>` per `src/ptxsim/async/tc_queue.h:70`) advances via monotonic-max CAS to value ≥ 2 (per `tc_queue.cpp:54-61`)
- **NOTE**: TcQueue uses **one** monotonic-max counter with **per-waiter** `waited_group_id` tracking, NOT per-group counters. The semantics `commit(N) → wait(M)` succeeds iff `N ≤ M`. This is empirically verified by reading `tc_queue.cpp:54-87` (commit) + `:89-109` (wait).

### Requirement: processTcgen05Wait SHALL use instr.cta_group instead of hardcoded 1

The system SHALL modify `processTcgen05Wait` (in `src/ptxsim/instructions/tcgen05.cpp:530`) to invoke `cta->tc_queue().wait(warp, /*lane_id=*/0, instr.cta_group)` instead of the current hardcoded `cta->tc_queue().wait(warp, 0, 1)`. The `lane_id=0` hardcoding SHALL remain (per Oracle Q5 — multi-lane `wait` belongs to a separate follow-up). The `(void)instr;` cast SHALL be removed.

#### Scenario: cta_group::1 wait preserves existing behavior

- **WHEN** `processTcgen05Wait` is called with a `Tcgen05Instr{cta_group=1}`
- **THEN** `cta->tc_queue().wait(warp, 0, 1)` is invoked
- **AND** the warp waits on group 1 only

#### Scenario: cta_group::2 wait wakes when counter ≥ 2

- **WHEN** `processTcgen05Wait` is called with a `Tcgen05Instr{cta_group=2}` after a `commit(cta_group=2)` has advanced the TcQueue counter to 2 (or higher via monotonic-max)
- **THEN** `cta->tc_queue().wait(warp, 0, 2)` is invoked
- **AND** the warp returns immediately (`waited_group_id=2 ≤ new_counter=2` satisfied, per `tc_queue.cpp:71` predicate)

#### Scenario: cta_group::2 wait blocks when no prior commit has reached 2

- **WHEN** `processTcgen05Wait` is called with a `Tcgen05Instr{cta_group=2}` but no prior `commit(cta_group>=2)` has occurred (counter is still at default 0 or at 1)
- **THEN** the warp is added to `pending_waiters_` (per `tc_queue.cpp:102`) with `waited_group_id=2`
- **AND** the warp sets `is_blocked=true` + `status=Blocked` (per `tc_queue.cpp:106-108`)
- **AND** returns when a subsequent `commit(cta_group>=2)` advances the counter via monotonic-max CAS (per `tc_queue.cpp:54-77`)

#### Scenario: cta_group::2 with this change still throws per ADR-0018

- **NOTE**: This requirement scopes ONLY `cta_group::1` (default) and routing via `instr.cta_group`. For `cta_group::2` semantics, see [ADR-0018](../../../docs/adr/0018-tcgen05-cta-group-restriction.md) (created by this change): handlers throw `UnsupportedInstructionException` with message containing "cluster abstraction not yet implemented (ADR-0018)". The C3 fix does **not** change `cta_group::2` throw behavior — it only adds `cta_group::N` routing for the already-supported `cta_group::1` path.

### Requirement: makeTcgen05Instr SHALL accept optional cta_group parameter

The system SHALL modify `makeTcgen05Instr` (in `include/ptx_ir/statement_factory.h:265`) to accept an additional optional `uint32_t cta_group = 1` parameter. The default value 1 SHALL preserve backward compatibility for all existing call sites.

#### Scenario: existing call sites compile without modification

- **WHEN** existing tests call `makeTcgen05Instr(op_kind, qualifiers, operands, text)` (4 args)
- **THEN** compilation succeeds
- **AND** `instr.cta_group == 1u` (default value)

#### Scenario: visitor passes extracted cta_group

- **WHEN** `visitTcgen05Inst` extracts `cta_group=2` from the parse tree
- **AND** calls `makeTcgen05Instr(op_kind, qualifiers, operands, text, 2)`
- **THEN** `instr.cta_group == 2u`
