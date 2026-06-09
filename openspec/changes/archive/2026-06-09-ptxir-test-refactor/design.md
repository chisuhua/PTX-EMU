## Context

The PTX-EMU four-mode testing framework (`tests/three_mode_testing/`) currently contains auto-generated tests that mix concerns:
- Mode 1: cuobjdump extraction
- Mode 2: PTX file loading
- Mode 3a/3b: StatementContext (raw + CFG-processed)
- Mode 3c: Standalone binary execution
- Mode 4: PTXIR serialization (NEW)

Current problems:
1. Each mode has both "StatementContext tests" AND "raw WarpContext tests" - redundancy
2. No tests validate Mode 4 (serialize→deserialize)
3. Tests don't clearly show the pipeline flow (Extract → Parse → Serialize → Deserialize → Execute)

## Goals / Non-Goals

**Goals:**
- Refactor tests to focus on the core pipeline: PTX → StatementContext → .ptxir → Execute
- Add Mode 4 tests that validate serialization roundtrip
- Remove redundant raw WarpContext tests (they belong in unit tests, not integration tests)
- Make tests document the flow: "what goes in, what comes out"

**Non-Goals:**
- Not rewriting all tests from scratch
- Not removing Mode 1/2/3 entirely - they still serve their purpose
- Not adding execution validation (that's covered elsewhere)

## Decisions

### Decision 1: Test Structure - Two Focus Files

**Choice**: Keep test structure focused on two files rather than five.

**Rationale**: Instead of creating `test_extract.cpp`, `test_parse.cpp`, `test_serialize.cpp`, `test_deserialize.cpp`, `test_roundtrip.cpp`, consolidate around:
- `test_ptxir_serialization.cpp` — Mode 4 roundtrip tests (serialize→deserialize correctness, no execution)
- `test_four_mode_flow.cpp` — End-to-end pipeline tests (Mode 1→2→3→4 full flow validation)

**Why not five files**: The five pipeline-stage files create unnecessary fragmentation. The tests already have a clear mode-based organization (Mode 1/2/3/4), and the two-file approach preserves that while adding Mode 4 and removing redundant raw tests.

**Alternatives considered**:
- Pipeline-stage files (rejected: fragments the test suite, modes already provide sufficient granularity)
- One big integration test (rejected: loses granular validation)
- Five files (rejected: over-compartmentalized for the actual testing needs)

### Decision 2: Remove Raw Tests

**Choice**: Remove `raw` tagged tests that validate WarpContext directly.

**Rationale**: These test low-level mechanics, not the PTX-EMU public API. They belong in unit tests for warp scheduler, not integration tests for the four-mode framework.

**Alternatives considered**:
- Keep them for debugging (rejected: can be added back if needed, but not in main integration tests)
- Move to separate `tests/unit/` directory (deferred - not this change)

### Decision 3: Mode 4 Test Focus

**Choice**: Mode 4 tests focus ONLY on serialize→deserialize roundtrip correctness.

**Rationale**: Mode 4's value is fast loading (~5ms vs ~200ms). Tests should validate:
1. Statement count preserved
2. Statement types preserved
3. Branch reconvergence_pc preserved
4. Operand values preserved

**Not** in Mode 4 tests:
- Execution results (covered by Mode 3b)
- PTX parsing (covered by Mode 2)

## Risks / Trade-offs

- **[Risk]** Existing tests that rely on `raw` tag may break
  - **Mitigation**: Document the change, note that raw tests can be re-added to unit tests if needed

- **[Trade-off]** Removing raw tests loses some low-level validation
  - **Mitigation**: WarpContext unit tests already exist in other test files

## Migration Plan

1. Refactor existing test files (do not create new files, modify what's there)
2. Add Mode 4 serialize→deserialize tests to `test_ptxir_serialization.cpp`
3. Create `test_four_mode_flow.cpp` for end-to-end pipeline validation
4. Run both during transition
5. Remove `[raw]` tagged tests once coverage gaps are filled
6. Update `test_helpers.hpp` to support serialization roundtrip testing

## Resolved Decisions

1. **Mode 3c (standalone binary)**: Keep for backwards compatibility. Deprecate in SKILL.md — Mode 4 supersedes it for new tests.
2. **Unit vs integration boundary**: Integration tests cover pipeline flow (PTX → StatementContext → .ptxir → Execute). Unit tests cover individual components (WarpContext, SharedMemoryManager). Raw WarpContext tests belong in unit tests.
3. **Golden reference for Mode 4**: Not needed — roundtrip validation uses comparison of deserialized StatementContexts against in-memory originals, not golden files.