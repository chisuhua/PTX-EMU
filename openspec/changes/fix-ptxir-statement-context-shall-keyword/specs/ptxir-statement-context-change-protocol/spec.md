## MODIFIED Requirements

### Requirement: Pre-commit hook MUST warn on StatementContext changes without PTXIR sync
A pre-commit hook (e.g., in `.git/hooks/pre-commit` or `.husky/`) MUST detect staged changes to `include/ptx_ir/statement_context.h` and warn if no corresponding changes to `src/ptx_ir/ptxir_writer.cpp` or `src/ptx_ir/ptxir_reader.cpp` are present in the same commit.

#### Scenario: Hook warns on incomplete sync
- **WHEN** a developer runs `git commit` with staged changes to `statement_context.h` but no staged changes to `ptxir_writer.cpp` or `ptxir_reader.cpp`
- **THEN** the pre-commit hook MUST print a warning message suggesting the developer review the modification protocol
- **AND** the hook MUST allow the commit to proceed (warning only, not blocking)

#### Scenario: Hook allows when both updated
- **WHEN** a developer runs `git commit` with staged changes to BOTH `statement_context.h` AND `ptxir_writer.cpp` (or `ptxir_reader.cpp`)
- **THEN** the pre-commit hook MUST NOT print any warning