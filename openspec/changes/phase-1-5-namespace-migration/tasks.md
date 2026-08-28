# Tasks: phase-1-5-namespace-migration

## 0. Phase 0 — freeze scanner and baseline

- [x] 0.1 Create the read-only `scripts/check_ptxemu_ir_names.py` scanner before any caller edits. It must support `--roots`, repeated `--exclude`, `--list-files`, and the token/comment/literal rules specified in design.md; it must exit non-zero on bare caller tokens.
- [x] 0.2 Run `python3 scripts/check_ptxemu_ir_names.py --roots src include tests --list-files`, save the deterministic file list and src/include/tests counts in the implementation notes, and use that list as the phase scope. Confirm the three shims and canonical `include/ptxemu/ir/` are excluded while non-shim `include/ptx_ir/` remains included.
- [x] 0.3 Run `cmake --build build && cd build && ctest --output-on-failure` at baseline `2cd8449e`; record the actual total and failures before continuing.

> **Baseline**: `2cd8449e`; the scanner and ctest results must be recorded by Phase 0 before implementation groups begin.
> **Scope**: the caller file list generated before implementation by `python3 scripts/check_ptxemu_ir_names.py --roots src include tests --list-files` — **~212 files with bare-token matches (src 56, include 35, tests 121)** — verified via scanner match mode (only files with actual bare-token matches; `--list-files` enumerates all 433 files in the roots, src 116/include 108/tests 209); include non-shim `include/ptx_ir/` and `include/ptxir/` headers.
> **Discipline**: Every implementation group is an independent commit and must pass build + ctest with **no NEW failures vs recorded 246/252 dirty baseline** before the next group (ptx-lessons-learned §3-4). The 6 flaky/SEGFAULT tests (`dummy-float`/`dummy-mul`/`dummy-ldglobal`, `e2e_ldglobal_simple`, `simpleGEMM-double`, `all-pairs-distance`) are pre-existing infra races in `cubin_utils.cpp:139` shared workspace clobbering, NOT namespace-related — see Appendix.

## 1. Phase 1.5c+d — shim + src/ptx_ir and src/ptxir self-migration

- [x] 1.1 Replace `include/ptx_ir/ptx_types.h` with a forwarding shim: canonical include, `namespace ptx_ir = ::ptxemu::ir`, and explicit fully-qualified `using` for canonical types. Compatibility policy is fixed to canonical type names only; do not export bare `S_*`/`O_*` enumerators. Legacy enumerator access is covered by the negative fixture and is intentionally outside the shim contract. **Temporary bridge (1.5c+d only):** shim also emits `using enum ::ptxemu::ir::StatementType;` and `using enum ::ptxemu::ir::OperandType;` (C++20) so the ~87 caller files (src 18, include 8, tests 62) keep compiling. The bridge is removed in Phase 1.5k after the caller sweeps complete.
- [x] 1.2 Replace `include/ptx_ir/operand_context.h` with a forwarding shim and explicit fully-qualified `using` for all operand types.
- [x] 1.3 Replace `include/ptx_ir/statement_context.h` with a forwarding shim and explicit fully-qualified `using` for all instruction structs, enums, `InstrVariant`, and `StatementContext`; preserve the type-name-only compatibility policy from 1.1.
- [x] 1.4 Wrap `src/ptx_ir/ptx_types.cpp` in `namespace ptxemu::ir` and migrate all free-function definitions to the canonical declarations (fixes the verified ODR conflict).
- [x] 1.5 Migrate canonical IR out-of-line methods to `ptxemu::ir` where their declarations are canonical; keep `instruction_latency_table.cpp` in `namespace ptxsim` and `ptx_syntax_utils.cpp` in `namespace ptx::syntax`, qualifying only their IR types. Keep reader/writer implementations in the namespace declared by their headers.
- [x] 1.6 Sweep `src/ptxir/ptxir_serialization.cpp`, `include/ptxir/ptxir_serialization.h`, and all related serialization declarations for IR types. When replacing `struct StatementContext` elaborated specifiers, add the canonical include or a valid namespace forward declaration before using the qualified type.
- [x] 1.7 Resolve the duplicate `InstructionState` definition by first including `ptxemu/ir/execution_types.h`, then replacing the global definition in `include/ptxsim/execution_types.h` with `using ::ptxemu::ir::InstructionState;`; preserve unrelated `Dim3`/execution enums and verify `InstructionState::READY` lookup plus `git grep -n "enum class InstructionState" include/` returns only the canonical definition.
- [x] 1.8 Build + ctest 252/252; run focused PTXIR tests (`ctest -R ptxir --output-on-failure`).
- [x] 1.9 Commit: `refactor(ptx-1.5c+d): namespace wrap and IR self-migration`.
- [ ] 1.10 Complete partial migrations in `src/ptx_ir/instruction_latency_table.cpp`, `src/ptxir/ptxir_serialization.cpp`, `src/ptxir/ptxir_reader.cpp`, and `src/ptxir/ptxir_writer.cpp`: replace remaining bare `StatementType`, `Qualifier`, `StatementContext`, `Tcgen05*`, `InstrVariant`, and `OperandContext` references with `ptxemu::ir::`-qualified names. This follow-up is required by design.md **R3** because `StatementType`/qualifier resolution can change under a non-default namespace.

## 2. Phase 1.5e — src/ptx_parser caller sweep

- [x] 2.1 Sweep all matching files under `src/ptx_parser/` (13 files measured) and corresponding headers; qualify all IR types with `ptxemu::ir::`.
- [x] 2.2 Build + ctest 254/254 verification.
- [x] 2.3 Commit: `refactor(ptx-1.5e): src/ptx_parser caller sweep` (commit `7b7f468b`).

## 3. Phase 1.5f1 — src/ptxsim/instructions caller sweep

- [ ] 3.1 Sweep all matching files under `src/ptxsim/instructions/` (max 30 files per commit); qualify all IR types.
- [ ] 3.2 Build + ctest 252/252 verification.
- [ ] 3.3 Commit: `refactor(ptx-1.5f1): src/ptxsim instructions caller sweep`.

## 4. Phase 1.5f2 — src/ptxsim/core+utils+debug caller sweep

- [ ] 4.1 Sweep all matching files under `src/ptxsim/core/`, `src/ptxsim/utils/`, and `src/ptxsim/debug/`, plus the `src/ptxsim/*.cpp` root files (`instruction_base.cpp`, `instruction_factory.cpp`, `instruction_handlers.cpp`, `register_analyzer.cpp`) and `src/utils/*.cpp` (`ptx_lane_verification.cpp`) — no phase previously claimed these 5 caller files, so 1.5f2 is expanded to cover them (**~19 files** total in this phase, max 30 per commit); qualify all IR types. **Rationale (Option A):** the 4 `src/ptxsim/*.cpp` root files + `src/utils/ptx_lane_verification.cpp` carry bare IR tokens but were listed under no phase; folding them into the existing `src/ptxsim` sweep phase keeps the phase map complete without adding a new phase.
- [ ] 4.2 Build + ctest 252/252 verification.
- [ ] 4.3 Commit: `refactor(ptx-1.5f2): src/ptxsim core utils debug caller sweep`.

## 5. Phase 1.5g — src/cudart caller sweep

- [ ] 5.1 Sweep all matching files under `src/cudart/` (4 files measured); qualify all IR types.
- [ ] 5.2 Build + ctest 252/252 verification, including CppTLM/image/ PTXIR tests.
- [ ] 5.3 Commit: `refactor(ptx-1.5g): src/cudart caller sweep`.

## 6. Phase 1.5h1 — non-shim include/ptx_ir and include/ptxir sweep

- [ ] 6.1 Sweep every non-shim `include/ptx_ir/*.h` with a matching scanner hit, including `instruction_latency_table.h`, `kernel_context.h`, `param_context.h`, `ptx_context.h`, `ptxir_reader.h`, `ptxir_writer.h`, `statement_factory.h`, and all matching `include/ptxir/` headers; qualify all IR types. The three forwarding shims and canonical `include/ptxemu/ir/` definitions remain the only exempt paths.
- [ ] 6.2 Build + ctest 252/252 verification.
- [ ] 6.3 Commit: `refactor(ptx-1.5h1): include ptx_ir and ptxir caller sweep`.

## 7. Phase 1.5h2 — remaining include/ caller sweep

- [ ] 7.1 Sweep matching caller headers under `include/ptxsim/`, `include/ptxemu/` excluding canonical `include/ptxemu/ir/`, `include/cudart/`, `include/ptx_parser/`, `include/register/`, and `include/utils/` (remaining include files, max 30 per commit); qualify all IR types. Handle `include/ptxsim/execution_types.h` definition/alias separately per task 1.7.
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

- [ ] 10.1 Implement `scripts/check_ptxemu_ir_names.py` token-aware scanner; it MUST skip comments, char literals, ordinary strings, and C++ raw strings; scan the caller roots from design.md; exclude the three forwarding shims and canonical `include/ptxemu/ir/` definitions; ignore `ptxemu::ir::`-qualified tokens and canonical namespace-block definitions; and cover at least `StatementType`, `OperandType`, `InstructionState`, `Qualifier`, `OperandContext`, `InstrVariant`, `Tcgen05Instr`, `Tcgen05OpKind`, and `Tcgen05Dtype`.
- [ ] 10.2 Add Invariant 8 to `.github/workflows/drift_check.yml` invoking the scanner.
- [ ] 10.3 Run the scanner against bare, qualified, comment, char/string literal, raw-string, canonical-definition, shim, and unscoped-enumerator fixtures; verify only bare caller code fails.
- [ ] 10.4 Record the exact scanner command and `--list-files` output used to derive the caller file count (including src/include/tests breakdown), then run all eight local drift checks and ctest (no NEW failures vs the recorded 246/252 dirty baseline).
- [ ] 10.5 Verify `statement-ir-public`, `ptxemu-ir-namespace-contract`, and `ci-drift-check` scenarios against the implementation.
- [ ] 10.6 Update HSK-8 audit postmortem, root `AGENTS.md`, and `include/ptx_ir/AGENTS.md` with final phase commit list and task 9.4 release-cycle status.
- [ ] 10.7 Commit: `chore(ptx-1.5k): add IR namespace drift invariant and close audit`.

## 11. Push and archive

- [ ] 11.1 Push the implementation branch without force and open the normal PR; do not push directly to `origin/main`.
- [ ] 11.2 Verify the PR's GitHub drift_check completes successfully with all eight invariants.
- [ ] 11.3 Verify `origin/main` equals local HEAD and the three unrelated `.opencode/notes` remain untracked.
- [ ] 11.4 Run `openspec archive phase-1-5-namespace-migration` only after all implementation tasks are complete.
- [ ] 11.5 Verify promoted specs and update the final HSK-8 audit archive reference.

## Per-phase failure discipline

- Any regression means stop, diagnose with `ptx-lessons-learned`, and revert only the failing phase commit.
- Do not amend archived artifacts; update this change or create a follow-up change if scope changes.
- Do not edit ANTLR-generated files under `build/`.

## Appendix: Known Flaky Baseline (Pre-existing, OUT of Phase 1.5 Scope)

Recorded dirty baseline: 246/252 PASS (run-to-run variation 244-246/252).
All 6 failures are pre-existing infra issues in PTX-extraction/kernel-registration path,
NOT namespace-related. To be addressed by independent `fix-ptx-extraction-race` change.

| Test | Symptom | Root cause |
|---|---|---|
| `dummy-float`, `dummy-mul`, `dummy-ldglobal` | "Failed to open extracted PTX file" | `cubin_utils.cpp:139` shared workspace clobbering under parallel `ctest -j` |
| `e2e_ldglobal_simple` | Same | Same |
| `simpleGEMM-double`, `all-pairs-distance` | "Kernel not found … Available kernels: [empty]" | fatbin/PTX registration fails → empty kernel list |

Phase 1.5 acceptance criterion is "no NEW failures vs this baseline" — flakes
that surface during a phase must be diffed against this list before any revert.
