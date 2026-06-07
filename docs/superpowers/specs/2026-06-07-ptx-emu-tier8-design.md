# P1-3: Tier 8 Cross-Component Integration Tests — Design

**Date**: 2026-06-07
**Status:** Draft (pending user review)
**Parent:** [`2026-06-06-ptx-emu-test-coverage-roadmap.md`](./2026-06-06-ptx-emu-test-coverage-roadmap.md) §3
**Estimated effort:** 2-3 days
**Out of scope:** WMMA, kernel launch full flow, multi-SM test infrastructure

---

## 1. Goal

Populate the currently-empty `Tier 8` slot in `scripts/sanity.sh:280-285` with at least 1 full-warp cross-component integration test. After completion:
- `sanity.sh --tier 8` runs ≥1 ctest target and reports PASS
- `sanity.sh` (default) exits 0 with no regressions
- The new test crosses ≥3 simulator components (SM, CTA, barrier, memory, etc.)

## 2. Pre-investigation findings (2026-06-07)

| # | Finding | Impact on design |
|---|---|---|
| 1 | `scripts/sanity.sh:280-285` currently prints "Tier 8 currently empty (reserved for future end-to-end tests)" | P1-3's primary deliverable is replacing this placeholder with ≥1 active test |
| 2 | 6 existing barrier integration tests in `tests/integration/barrier/` (1852 lines total) | Strong test pattern + working `bar.warp.sync` / `bar.sync` infrastructure |
| 3 | 2 barrier tests DISABLED in `tests/integration/CMakeLists.txt:38-49` (Pre-P0 baseline red) | `integration_warp_barrier_memory_visibility` (#84) and `integration_cta_barrier_memory_visibility` (#85) — out of scope for P1-3 |
| 4 | Working barrier test pattern in `test_warp_barrier_integrated.cpp:1-50` | Reuse boilerplate: `init_instruction_factory_once`, `create_warp_with_threads`, `step_warp` |
| 5 | `instruction_factory.cpp` registers handlers via X-macro over `ptx_op.def` | All `bar.*` opcodes are registered. No handler changes needed for P1-3. |
| 6 | `make_bar_warp_sync(mask, reconvergence_pc)` factory exists in `instruction_helpers.h:20` | Use this factory for `bar.warp.sync` statements |
| 7 | `make_bar_sync(bar_id)` factory exists in `instruction_helpers.h:32` | Use this factory for `bar.sync` statements |
| 8 | P1-4 surfaced latent handler bugs (§P1-4.1, §P1-4.2) | P1-3 should use ONLY `bar.warp.sync` and `bar.sync` (no float ops, no cvt) to minimize risk of hitting new handler bugs |

## 3. Selected scenario: `integration_barrier_full_lifecycle`

Among the 3 candidate scenarios from the roadmap:
- `integration_cross_sm_shared` (2 SMs) — high difficulty, multi-SM infrastructure changes needed
- `integration_kernel_launch_flow` (full launch) — very high, largest surface
- **`integration_barrier_full_lifecycle` (2 warps, 1 CTA)** — **selected**, medium difficulty, barrier is well-isolated

This scenario covers the **complete barrier lifecycle**:
1. **Init**: 2 warps initialized in same CTA
2. **Arrive**: Both warps execute `bar.sync 0` and reach the barrier
3. **Release**: After both arrive, the barrier releases both warps to the reconvergence PC
4. **Reset**: After release, the barrier state resets for future use

This crosses 4 simulator components: **SM (warp scheduler) + CTA (warp management) + WBar (barrier state) + Warp (active_mask / PC)**. Pure barrier domain — no float/cvt ops to risk handler bugs.

## 4. File list

### 4.1 New test file (1)
- `tests/integration/barrier/test_barrier_full_lifecycle.cpp` — single file with ≥3 TEST_CASEs covering the lifecycle phases
- ctest target: `integration_barrier_full_lifecycle`
- Labels: `"integration;barrier;lifecycle;tier8"`

### 4.2 Modified files (2)
- `tests/integration/CMakeLists.txt` — add 1 `add_catch_test` + 1 `set_tests_properties` entry
- `scripts/sanity.sh:280-285` — replace the "reserved for future" placeholder with `run_regex_tests "integration_barrier_full_lifecycle" "Barrier full lifecycle (init/arrive/release/reset)"` (or similar)

### 4.3 Untouched files
- All `src/ptxsim/` files — no handler changes needed
- All `src/ptx_ir/` files — no IR changes
- `KNOWN_ISSUES.md` — no new entries (P1-3 uses only existing, working handlers)

## 5. Architecture / data flow

### 5.1 Test setup pattern (mirrors `test_warp_barrier_integrated.cpp`)

```cpp
// 1. Initialize instruction factory
init_instruction_factory_once();

// 2. Create SM with 1 CTA containing 2 warps
SMContext sm(4, 128, 4096, 0);
auto blk = std::make_unique<CTAContext>();
Dim3 g{1, 1, 1}, b{64, 1, 1}, bi{0, 0, 0};
std::map<std::string, int> l2pc;
std::map<std::string, Symtable*> n2s;
blk->init(g, b, bi, stmts, &n2s, l2pc);
sm.add_block(std::move(blk));

// 3. Build minimal statement sequence per warp:
//    PC=0: mov r1, tid.x       (r1 = lane_id)
//    PC=1: bar.sync 0          (arrive at CTA barrier 0)
//    PC=2: add r2, r1, 10      (work after barrier)
//    PC=3: ret

// 4. Set up Wbar at SM level
sm.allocate_cta_barrier(0, expected_arrival_count=64 /* 2 warps * 32 lanes */);

// 5. Drive step_warp on both warps in round-robin
//    - Warp 0 step: PC=0 → 1 (barrier arrive), then PC=1 blocks (waiting for warp 1)
//    - Warp 1 step: PC=0 → 1 (barrier arrive), barrier releases both
//    - Both warps: PC=2 → 3 (ret)

// 6. Verify: r2 == lane_id + 10 for all lanes in both warps
```

### 5.2 Data flow

1. **Warp scheduler** (`SMContext`) selects active warps
2. **Barrier arrival** (`bar.sync 0`) marks each lane as arrived via `Wbar::arrive`
3. When all 64 lanes (2 warps × 32 lanes) arrive, `Wbar::is_complete()` returns true
4. **Barrier release** forces both warps' `reconvergence_pc` to be `PC=2` (post-barrier)
5. **Post-barrier execution** runs the `add r2, r1, 10` statement for all lanes
6. **Commit + ret** terminates both warps

### 5.3 Error handling
- Use `REQUIRE` for setup invariants (warp creation, barrier allocation)
- Use `CHECK` for per-lane value verification
- If the barrier handler has a bug, tests surface it via stalled PC (warp never reaches PC=2)
- No empty catch blocks; no `as any` / `@ts-ignore` equivalents

## 6. Test cases (≥3)

| TEST_CASE | Phase tested | Expected outcome |
|---|---|---|
| `bar_lifecycle_single_warp_init` | Init: barrier created with 1 warp | Warp arrives; barrier remains incomplete (waiting for non-existent 2nd warp) |
| `bar_lifecycle_two_warps_release` | Arrive + Release: both warps arrive at `bar.sync 0` | Both warps released; `r2 == lane_id + 10` for all 64 lanes |
| `bar_lifecycle_reuse_after_release` | Reset: after first barrier, both warps execute a 2nd `bar.sync 0` | 2nd barrier also releases correctly (verifies state was properly reset) |

If 2-warp infrastructure is complex, the test can use a single warp for "init" and "reuse" cases, and 2 warps for "release" case.

## 7. CMake additions

In `tests/integration/CMakeLists.txt`, add after existing `integration_warp_barrier` block:

```cmake
# ============================================================================
# P1-3: Tier 8 cross-component integration test
# (added 2026-06-07 per docs/superpowers/specs/2026-06-07-ptx-emu-tier8-design.md)
# ============================================================================
add_catch_test(integration_barrier_full_lifecycle
    barrier/test_barrier_full_lifecycle.cpp
)
set_tests_properties(integration_barrier_full_lifecycle PROPERTIES LABELS "integration;barrier;lifecycle;tier8")
```

## 8. Sanity.sh update

Replace `scripts/sanity.sh:280-285`:

```bash
# Tier 8: Cross-Component Integration
if ! skip_tier 8; then
    print_header "Tier 8: Cross-Component Integration (full warp flows)"
    run_regex_tests "integration_barrier_full_lifecycle" "Barrier full lifecycle (init/arrive/release/reset)"
fi
```

## 9. Success criteria

- [ ] 1 new test file `test_barrier_full_lifecycle.cpp` with ≥3 TEST_CASEs
- [ ] 1 new ctest entry in `tests/integration/CMakeLists.txt`
- [ ] `scripts/sanity.sh --tier 8` exits 0 (no more "reserved" message)
- [ ] `sanity.sh` (default Tiers 1-9) exits 0, no regressions
- [ ] Pre-P0 baseline red tests (#84, #85) remain DISABLED (out of scope)
- [ ] No `as any` / `@ts-ignore` / C-style casts introduced
- [ ] `clang-format -i` applied to the new test file
- [ ] No new entries needed in `KNOWN_ISSUES.md` (test uses only working handlers)

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 2-warp infrastructure in test harness is incomplete | Medium | Medium | Reference `test_warp_barrier_integrated.cpp`; if 2-warp setup is too complex, fall back to single-warp for "init" + "reuse" and use 2-warp only for "release" |
| `bar.sync` with count != 32 has different behavior than `bar.warp.sync` | Low | Low | Read `barrier.cpp` to verify; if 2-warp `bar.sync` is non-trivial, use `bar.warp.sync` per warp + SM-level coordination |
| Test framework doesn't expose 2-warp scheduler | Low | Medium | Manual scheduling loop in the test (Warp 0 step → Warp 1 step → both proceed) |
| Latent bug in Wbar::reset | Low | Medium | The "reuse" test case surfaces this; if it fails, document in `KNOWN_ISSUES.md` as §P1-3.1 |
| Handler bug in `bar.sync` (untested path) | Low | Medium | If a TEST_CASE fails, apply the same `SKIP()` + `KNOWN_ISSUES.md` pattern used in P1-4 |

## 11. Out of scope (intentional)

- WMMA / MMA tests (independent roadmap item)
- Multi-SM test (requires harness changes, deferred to future roadmap)
- Full kernel launch flow test (too large surface, deferred)
- Float arithmetic / cvt operations (avoiding P1-4 handler bugs)
- Performance benchmarks
- Atomic operations

## 12. Open question

- Should the test use `bar.sync 0` (CTA-level) or `bar.warp.sync` per warp (warp-level) + manual coordination? `bar.sync` is cleaner; `bar.warp.sync` is simpler. The design assumes `bar.sync 0` with both warps in same CTA.
