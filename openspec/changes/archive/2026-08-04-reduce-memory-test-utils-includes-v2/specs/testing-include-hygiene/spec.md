# testing-include-hygiene

## ADDED Requirements

### Requirement: Testing utility header include surface

The `include/ptxsim/testing/memory_test_utils.h` header SHALL expose its public inline helpers using no more than twelve direct `#include` lines. Function signatures and inline bodies SHALL remain unchanged. Behavior of every helper SHALL be byte-identical before and after the include reduction.

#### Scenario: Include count meets target

- **WHEN** `grep -c '^#include' include/ptxsim/testing/memory_test_utils.h` is run on the merged branch
- **THEN** the count is `<= 12`

#### Scenario: All consumers compile unchanged

- **WHEN** `cmake --build build` is executed
- **THEN** every translation unit that includes `memory_test_utils.h` compiles with zero new warnings

#### Scenario: ctest unit suite stays green

- **WHEN** `ctest -L unit --output-on-failure` is executed
- **THEN** all unit tests pass with the same pass/fail count as the baseline
