# extraction-isolation Spec Deltas

## ADDED Requirements

### Requirement: PTX Extraction Isolation

The `extract_ptx_with_cuobjdump` helper in `src/utils/cubin_utils.cpp` SHALL extract PTX content into a unique temporary directory created per call via `mkdtemp`, not into the shared process current working directory.

#### Scenario: Parallel extraction

- Given N concurrent calls to `extract_ptx_with_cuobjdump` with distinct or identical input binaries
- When all N calls complete
- Then each call returns its own complete extracted PTX content
- And no call observes another call's in-flight files

#### Scenario: Cleanup on success

- Given a successful extraction
- When the function returns
- Then the unique temporary directory is removed on a best-effort basis
