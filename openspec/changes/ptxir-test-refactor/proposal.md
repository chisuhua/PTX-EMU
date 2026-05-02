## Why

The current `tests/three_mode_testing/` tests are auto-generated but contain redundant validation logic (raw WarpContext tests alongside StatementContext tests). With the introduction of Mode 4 (PTXIR binary), the testing workflow should focus on the core serialization/deserialization pipeline rather than low-level warp mechanics.

## What Changes

1. **Refactor Mode 1/2/3 tests** to focus on PTX → StatementContext → execution flow
2. **Add Mode 4 tests** that validate serialize→deserialize roundtrip
3. **Remove redundant raw tests** - raw WarpContext validation belongs in unit tests, not integration tests
4. **Split tests by pipeline stage**:
   - Extract: PTX extraction via cuobjdump (Mode 1)
   - Parse: PTX → StatementContext (Mode 2/3)
   - Serialize: StatementContext → .ptxir (Mode 4 write)
   - Deserialize: .ptxir → StatementContext (Mode 4 read)
   - Execute: StatementContext → result

## Capabilities

### New Capabilities

- **ptxir-mode4-testing**: Mode 4 tests for serialization roundtrip validation
- **four-mode-flow-testing**: End-to-end pipeline tests showing Mode 1→2→3→4 relationship

### Modified Capabilities

- **three-mode-testing**: Refactored to remove redundancy, focus on essential validation

## Impact

- `tests/three_mode_testing/test_divergence_sync_standalone*.cpp` - Refactored
- `tests/three_mode_testing/test_warp_divergence*.cpp` - Refactored
- `tests/three_mode_testing/test_helpers.hpp` - Enhanced with Mode 4 helpers
- `openspec/changes/ptxir-serialization-architecture/` - Existing work to leverage