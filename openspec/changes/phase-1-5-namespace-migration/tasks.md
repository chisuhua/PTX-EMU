# Tasks: phase-1-5-namespace-migration

> **Baseline**: `2cd8449e` (ctest 252/252 verified)
> **Scope**: 218 files with IR callers (src 58, include 40, tests 120), including non-shim `include/ptx_ir/` and `include/ptxir/` headers.
> **Discipline**: Every implementation group is an independent commit and must pass build + ctest 252/252 before the next group (ptx-lessons-learned §3-4).

## 1. Phase 1.5c+d — shim + src/ptx_ir and src/ptxir self-migration

- [ ] 1.1 Replace `include/ptx_ir/ptx_types.h` with a forwarding shim: canonical include, `namespace ptx_ir = ::ptxemu::ir`, explicit `using` for all types and free functions `Q2s`, `S2s`, `Q2bytes`, `extractREG`.
- [ ] 1.2 Replace `include/ptx_ir/operand_context.h` with a forwarding shim and explicit `using` for all operand types.
- [ ] 1.3 Replace `include/ptx_ir/statement_context.h` with a forwarding shim and explicit `using` for all instruction structs, enums, `InstrVariant`, and `StatementContext`.
- [ ] 1.4 Wrap `src/ptx_ir/ptx_types.cpp` in `namespace ptxemu::ir` and migrate all free-function definitions to the canonical declarations (fixes the verified ODR conflict).
- [ ] 1.5 Wrap `src/ptx_ir/{operand_context,statement_context,ptx_syntax_utils,instruction_latency_table,ptxir_reader,ptxir_writer}.cpp` as required; qualify all out-of-line methods and IR types.
- [ ] 1.6 Sweep `src/ptxir/ptxir_serialization.cpp` and `include/ptxir/ptxir_serialization.h` for IR types; migrate any matches to `ptxemu::ir::*`.
- [ ] 1.7 Build + ctest 252/252; run focused PTXIR tests (`ctest -R ptxir --output-on-failure`).
- [ ] 1.8 Commit: `refactor(ptx-1.5c+d): namespace wrap and IR self-migration`.

## 2. Phase 1.5e — src/ptx_parser caller sweep

- [ ] 2.1 Sweep all matching files under `src/ptx_parser/` (13 files measured) and corresponding headers; qualify all IR types with `ptxemu::ir::`.
- [ ] 2.2 Build + ctest 252/252 verification.
- [ ] 2.3 Commit: `refactor(ptx-1.5e): src/ptx_parser caller sweep`.

## 3. Phase 1.5f1 — src/ptxsim/instructions caller sweep

- [ ] 3.1 Sweep all matching files under `src/ptxsim/instructions/` (max 30 files per commit); qualify all IR types.
- [ ] 3.2 Build + ctest 252/252 verification.
- [ ] 3.3 Commit: `refactor(ptx-1.5f1): src/ptxsim instructions caller sweep`.

## 4. Phase 1.5f2 — src/ptxsim/core+utils+debug caller sweep

- [ ] 4.1 Sweep all matching files under `src/ptxsim/core/`, `src/ptxsim/utils/`, and `src/ptxsim/debug/` (remaining src/ptxsim files, max 30 per commit); qualify all IR types.
- [ ] 4.2 Build + ctest 252/252 verification.
- [ ] 4.3 Commit: `refactor(ptx-1.5f2): src/ptxsim core utils debug caller sweep`.

## 5. Phase 1.5g — src/cudart caller sweep

- [ ] 5.1 Sweep all matching files under `src/cudart/` (4 files measured); qualify all IR types.
- [ ] 5.2 Build + ctest 252/252 verification, including CppTLM/image/ PTXIR tests.
- [ ] 5.3 Commit: `refactor(ptx-1.5g): src/cudart caller sweep`.

## 6. Phase 1.5h1 — non-shim include/ptx_ir and include/ptxir sweep

- [ ] 6.1 Sweep non-shim `include/ptx_ir/{kernel_context,param_context,ptx_context,ptxir_reader,ptxir_writer,statement_factory}.h` and all matching `include/ptxir/` headers; qualify all IR types. The three forwarding shims remain the only exempt headers.
- [ ] 6.2 Build + ctest 252/252 verification.
- [ ] 6.3 Commit: `refactor(ptx-1.5h1): include ptx_ir and ptxir caller sweep`.

## 7. Phase 1.5h2 — remaining include/ caller sweep

- [ ] 7.1 Sweep matching headers under `include/ptxsim/`, `include/ptxemu/`, `include/cudart/`, `include/ptx_parser/`, `include/register/`, and `include/utils/` (remaining include files, max 30 per commit); qualify all IR types.
- [ ] 7.2 Build + ctest 252/252 verification.
- [ ] 7.3 Commit: `refactor(ptx-1.5h2): remaining include caller sweep`.

## 8. Phase 1.5i1/i2/i3 — tests caller sweep

- [ ] 8.1 Sweep `tests/unit/` (first sub-batch, max 30 files) and qualify all IR types while preserving fixture construction semantics.
- [ ] 8.2 Build + ctest 252/252 verification.
- [ ] 8.3 Commit: `refactor(ptx-1.5i1): tests unit caller sweep`.
- [ ] 8.4 Sweep `tests/integration/` (second sub-batch, max 30 files), including PTXIR and attach-timing tests.
- [ ] 8.5 Build + ctest 252/252 verification.
- [ ] 8.6 Commit: `refactor(ptx-1.5i2): tests integration caller sweep`.
- [ ] 8.7 Sweep `tests/e2e/` (third sub-batch, max 30 files); do not modify generated artifacts or PTX grammar files.
- [ ] 8.8 Build + ctest 252/252 and run `./tests/ptx/test_all_ptx.sh`.
- [ ] 8.9 Commit: `refactor(ptx-1.5i3): tests e2e caller sweep`.

## 9. Phase 1.5j — GPUContext interface re-sign

- [ ] 9.1 Change the three `std::vector<StatementContext>` signatures in `include/ptxsim/gpu_context.h` (current lines 58/80/173) to `std::vector<ptxemu::ir::StatementContext>`.
- [ ] 9.2 Verify `src/ptxsim/core/gpu_context.cpp` definitions and all callers match.
- [ ] 9.3 Build + ctest 252/252 verification.
- [ ] 9.4 Commit: `refactor(ptx-1.5j): qualify GPUContext StatementContext signatures`.

## 10. Phase 1.5k — drift_check Invariant 8 and closure

- [ ] 10.1 Implement `scripts/check_ptxemu_ir_names.py` token-aware scanner; it MUST skip comments/string literals, exclude only the three forwarding shim headers, scan non-shim `include/ptx_ir/` and `include/ptxir/`, and ignore `ptxemu::ir::`-qualified tokens.
- [ ] 10.2 Add Invariant 8 to `.github/workflows/drift_check.yml` invoking the scanner.
- [ ] 10.3 Run the scanner against a bare-name fixture and a qualified-name fixture; verify bare fails and qualified passes.
- [ ] 10.4 Run all eight local drift checks and ctest 252/252.
- [ ] 10.5 Verify `statement-ir-public`, `ptxemu-ir-namespace-contract`, and `ci-drift-check` scenarios against the implementation.
- [ ] 10.6 Update HSK-8 audit postmortem, root `AGENTS.md`, and `include/ptx_ir/AGENTS.md` with final phase commit list and task 9.4 release-cycle status.
- [ ] 10.7 Commit: `chore(ptx-1.5k): add IR namespace drift invariant and close audit`.

## 11. Push and archive

- [ ] 11.1 Push all Phase 1.5 implementation commits to `origin/main` without force.
- [ ] 11.2 Verify GitHub drift_check completes successfully with all eight invariants.
- [ ] 11.3 Verify `origin/main` equals local HEAD and the three unrelated `.opencode/notes` remain untracked.
- [ ] 11.4 Run `openspec archive phase-1-5-namespace-migration` only after all implementation tasks are complete.
- [ ] 11.5 Verify promoted specs and update the final HSK-8 audit archive reference.

## Per-phase failure discipline

- Any regression means stop, diagnose with `ptx-lessons-learned`, and revert only the failing phase commit.
- Do not amend archived artifacts; update this change or create a follow-up change if scope changes.
- Do not edit ANTLR-generated files under `build/`.
