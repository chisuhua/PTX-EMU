## ADDED Requirements

### Requirement: StatementContext modification protocol MUST be documented
The `src/ptx_ir/AGENTS.md` file MUST contain a dedicated section titled "## StatementContext Modification Protocol" that lists the mandatory checklist for any commit that modifies the `StatementContext` or `OperandContext` structures.

#### Scenario: AGENTS.md contains protocol section
- **WHEN** `src/ptx_ir/AGENTS.md` is read
- **THEN** a section with heading "StatementContext Modification Protocol" (or "StatementContext 修改协议") MUST exist
- **AND** the section MUST include at least 4 checklist items covering: (1) sync with `ptxir_writer.cpp`, (2) sync with `ptxir_reader.cpp`, (3) add roundtrip test, (4) update X-Macro dispatch if new variant alternative added

#### Scenario: Checklist visible in module AGENTS.md
- **WHEN** a developer opens `src/ptx_ir/AGENTS.md`
- **THEN** the modification protocol MUST be in the top 3 sections (before low-level implementation details), so it is read first

### Requirement: OperandContext modification protocol MUST be documented
The same protocol MUST apply to `OperandContext` changes, since `ptxir_writer.cpp` and `ptxir_reader.cpp` also serialize/deserialize operands.

#### Scenario: OperandContext protocol covered
- **WHEN** the "StatementContext Modification Protocol" section in `src/ptx_ir/AGENTS.md` is read
- **THEN** the section MUST explicitly state that `OperandContext` modifications (adding fields, changing enum values) trigger the same checklist

### Requirement: Public header modification protocol MUST exist for include/ptxir
If `include/ptxir/AGENTS.md` does not exist, it MUST be created. If it exists, it MUST contain a parallel "Public Header Modification Protocol" section.

#### Scenario: include/ptxir/AGENTS.md exists
- **WHEN** the `include/ptxir/` directory is inspected
- **THEN** an `AGENTS.md` file MUST be present
- **AND** it MUST contain a "Public Header Modification Protocol" section

#### Scenario: Cross-reference between AGENTS.md files
- **WHEN** `include/ptxir/AGENTS.md` is read
- **THEN** it MUST reference `src/ptx_ir/AGENTS.md` for the underlying implementation modification protocol
- **AND** vice versa: `src/ptx_ir/AGENTS.md` MUST reference `include/ptxir/AGENTS.md` for the public API contract

### Requirement: PR template MUST require StatementContext change acknowledgment
The `.github/PULL_REQUEST_TEMPLATE.md` (or equivalent) MUST include a checkbox question: "Does this PR modify `StatementContext` or `OperandContext`? If yes, list the synchronized PTXIR writer/reader changes below."

#### Scenario: PR template has StatementContext checkbox
- **WHEN** a new PR is opened using the project template
- **THEN** the PR description MUST include the StatementContext/OberandContext acknowledgment checkbox
- **AND** checking the box MUST require listing which PTXIR writer/reader changes were made in the same commit or PR chain

### Requirement: Pre-commit hook SHOULD warn on StatementContext changes without PTXIR sync
A pre-commit hook (e.g., in `.git/hooks/pre-commit` or `.husky/`) SHOULD detect staged changes to `include/ptx_ir/statement_context.h` and warn if no corresponding changes to `src/ptx_ir/ptxir_writer.cpp` or `src/ptx_ir/ptxir_reader.cpp` are present in the same commit.

#### Scenario: Hook warns on incomplete sync
- **WHEN** a developer runs `git commit` with staged changes to `statement_context.h` but no staged changes to `ptxir_writer.cpp` or `ptxir_reader.cpp`
- **THEN** the pre-commit hook MUST print a warning message suggesting the developer review the modification protocol
- **AND** the hook MUST allow the commit to proceed (warning only, not blocking)

#### Scenario: Hook allows when both updated
- **WHEN** a developer runs `git commit` with staged changes to BOTH `statement_context.h` AND `ptxir_writer.cpp` (or `ptxir_reader.cpp`)
- **THEN** the pre-commit hook MUST NOT print any warning
