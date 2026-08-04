# thread-context-include-hygiene

## ADDED Requirements

### Requirement: ThreadContext header include surface

The `include/ptxsim/thread_context.h` header SHALL expose the `ThreadContext` class using no more than twenty-one direct `#include` lines. Function signatures, inline bodies, member layouts, and public behaviour SHALL remain unchanged. The reduction SHALL preserve the existing Type 1 unit-test contract.

#### Scenario: Include count meets target

- **WHEN** `grep -c '^#include' include/ptxsim/thread_context.h` is run on the merged branch
- **THEN** the count is `<= 21`

#### Scenario: All consumers compile unchanged

- **WHEN** `cmake --build build` is executed
- **THEN** every translation unit that includes `thread_context.h` compiles with zero new warnings

#### Scenario: Forward declarations replace redundant project headers

- **WHEN** `grep -E "class (Symtable|RegisterBankManager);" include/ptxsim/thread_context.h` is run
- **THEN** both forward declarations are present and the corresponding full headers are absent

#### Scenario: Build warning count does not increase

- **WHEN** the baseline warning count is recorded before the change and re-measured after
- **THEN** the post-change warning count is `<=` baseline

#### Scenario: ctest unit suite stays green

- **WHEN** `ctest -L unit --output-on-failure` is executed for any consumer that includes `thread_context.h`
- **THEN** all affected tests pass with the same pass/fail count as the baseline