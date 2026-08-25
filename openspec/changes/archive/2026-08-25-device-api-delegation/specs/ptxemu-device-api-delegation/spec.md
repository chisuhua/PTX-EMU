# ptxemu-device-api-delegation spec

## ADDED Requirements

### Requirement: `IPtxEmuDevice::set_scoreboard` delegates to SMContext

The `IPtxEmuDevice::set_scoreboard(uint32_t sm_id, uint32_t warp_id, uint64_t mask)` method MUST delegate to `SMContext::set_scoreboard(IScoreboard*)` (`include/ptxsim/sm_context.h:87`) for the given `(sm_id, warp_id)`. The method MUST return `true` if delegation succeeds, `false` if `sm_id`/`warp_id` is invalid or no `IScoreboard` is registered.

#### Scenario: Valid sm_id + warp_id with registered IScoreboard

- **WHEN** `device.set_scoreboard(0, 0, 0xFFFFFFFF)` is called on a GPU with SM 0 containing warp 0 with a registered `IScoreboard`
- **THEN** the method delegates to `SMContext::set_scoreboard(IScoreboard*)` for warp 0
- **AND** returns `true`

#### Scenario: Invalid sm_id

- **WHEN** `device.set_scoreboard(invalid_sm_id, 0, mask)` is called
- **THEN** the method returns `false` without crashing
- **AND** no GPU state is modified

### Requirement: `IPtxEmuDevice::set_active_mask` delegates to WarpContext with overwrite semantics

The `IPtxEmuDevice::set_active_mask(uint32_t sm_id, uint32_t warp_id, uint64_t mask)` method MUST delegate to `WarpContext::set_active_mask(uint32_t)` (`include/ptxsim/warp_context.h:199`) for the given `(sm_id, warp_id)`. The delegation MUST use **overwrite semantics** — the existing `WarpContext::active_mask_` is replaced entirely by `mask`, NOT OR-merged. The method MUST return `true` on success.

> **CRITICAL invariant**: OR-merge semantics in `set_active_mask` are encapsulated in `BarrierModule::release_warp_barrier` (per `ptx-barrier-mechanism` skill). The public `set_active_mask` MUST NOT perform OR-merge. Reintroduction of OR-merge in `device_api_impl.cc::set_active_mask` is the **BUG-RETHANG / BUG-POSTBARRIER-TWOHALVES** regression vector.

#### Scenario: Overwrite replaces entire mask

- **WHEN** warp 0's current `active_mask_` is `0xFF` and `device.set_active_mask(0, 0, 0x01)` is called
- **THEN** warp 0's `active_mask_` becomes `0x01` (overwrite)
- **AND** does NOT become `0xFF` (no-op, wrong) or `0xFF | 0x01 = 0xFF` (OR-merge, wrong)
- **AND** the method returns `true`

#### Scenario: Invalid warp_id

- **WHEN** `device.set_active_mask(0, invalid_warp_id, mask)` is called
- **THEN** the method returns `false` without crashing
- **AND** no warp state is modified

### Requirement: `IPtxEmuDevice::set_next_pc` delegates to ThreadContext via set_pc + commit_pc

The `IPtxEmuDevice::set_next_pc(uint32_t sm_id, uint32_t warp_id, uint32_t lane_id, uint32_t pc)` method MUST delegate to `ThreadContext::set_pc(int)` + `ThreadContext::commit_pc()` (`include/ptxsim/thread_context.h` L227-232) for the given `(sm_id, warp_id, lane_id)`. The delegation MUST use the `set_pc()` + `commit_pc()` pattern (per AGENTS.md ANTI-PATTERNS L85), NOT `force_set_pc()`. The `pc` parameter is narrowed from `uint32_t` to `int` via `static_cast<int>(pc)` (per AGENTS.md PTX PC is 32-bit, no overflow risk for valid PTX programs). The method MUST return `true` on success.

> **CRITICAL invariant**: `force_set_pc` bypasses PC synchronization invariants and is forbidden by AGENTS.md. Using `force_set_pc` in `device_api_impl.cc::set_next_pc` is a hard anti-pattern that MUST NOT occur.

#### Scenario: Valid thread PC update

- **WHEN** `device.set_next_pc(0, 0, 0, 42)` is called on a GPU with thread (sm=0, warp=0, lane=0)
- **THEN** thread (sm=0, warp=0, lane=0)'s PC is set to 42 via `thread->set_pc(42)` + `thread->commit_pc()`
- **AND** the method returns `true`
- **AND** subsequent `execute_warp_instruction` observes the new PC at the next dispatch

#### Scenario: Invalid lane_id

- **WHEN** `device.set_next_pc(0, 0, invalid_lane_id, pc)` is called
- **THEN** the method returns `false` without crashing
- **AND** no thread PC is modified

### Requirement: `IPtxEmuDevice::attach_timing` injects HSK-4 vendored interfaces into SMContext

The `IPtxEmuDevice::attach_timing(IScoreboard* sb, IPipelineLatencyProvider* pl, ITensorCoreTiming* tc)` method MUST store the three HSK-4 vendored interfaces and inject them into `SMContext` timing hooks. The three interfaces are HSK-4 vendored types (NOT redefined in `include/ptxemu/device_api.h`) per HSK-8 spec §6. The method MUST NOT return a status (void return type).

#### Scenario: Successful injection

- **WHEN** `device.attach_timing(sb, pl, tc)` is called with three non-null vendored interfaces
- **THEN** the interfaces are stored in `SMContext` timing hook fields
- **AND** subsequent instruction execution observes the new timing providers
- **AND** the method returns (void)

#### Scenario: Null interface arguments

- **WHEN** `device.attach_timing(nullptr, pl, tc)` is called with a null `IScoreboard*`
- **THEN** the method returns without crashing
- **AND** SMContext timing hooks for IScoreboard remain in their previous state (or default-initialized)