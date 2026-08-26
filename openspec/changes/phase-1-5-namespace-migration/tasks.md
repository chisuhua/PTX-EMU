# Tasks: phase-1-5-namespace-migration

> **Baseline commit**: `2cd8449e` (HEAD at task creation; ctest 252/252 verified, 4 commits ahead of origin/main)
> **Upstream cross-repo state**: HSK-8 ✅ ACCEPTED, CppTLM bump PR merged, PTXEMU_API_VERSION=1 frozen (no public ABI change)
> **Reference**: `openspec/changes/phase-1-5-namespace-migration/{proposal,design}.md`
> **Per-phase commit + ctest 252/252 discipline** (per ptx-lessons-learned §3-4)

## 1. Phase 1.5c+d — shim + src/ptx_ir self-migration (1 commit, ~3h)

- [ ] 1.1 Replace `include/ptx_ir/ptx_types.h` body with 11-line forwarding shim (per design D1): `#include <ptxemu/ir/ptx_types.h>` + `namespace ptx_ir = ::ptxemu::ir;` + 6 `using ::ptxemu::ir::Type;` lines
- [ ] 1.2 Replace `include/ptx_ir/operand_context.h` body with 9-line forwarding shim (per design D1): `#include <ptxemu/ir/operand_context.h>` + 7 `using` lines
- [ ] 1.3 Replace `include/ptx_ir/statement_context.h` body with ~32-line forwarding shim (per design D1): `#include <ptxemu/ir/statement.h>` + `using` for 27 instr structs + Tcgen05OpKind + Tcgen05Dtype + InstrVariant + StatementContext
- [ ] 1.4 Wrap `src/ptx_ir/ptx_types.cpp` body in `namespace ptxemu::ir { ... }` and qualify function definitions as `ptxemu::ir::Q2s` / `ptxemu::ir::S2s` / `ptxemu::ir::Q2bytes` / `ptxemu::ir::extractREG` (per design D2)
- [ ] 1.5 Wrap `src/ptx_ir/operand_context.cpp` body in `namespace ptxemu::ir { ... }` and qualify any out-of-line method definitions
- [ ] 1.6 Wrap `src/ptx_ir/statement_context.cpp` body in `namespace ptxemu::ir { ... }` and qualify `StatementContext::toString()` (and any other out-of-line method definitions) as `ptxemu::ir::StatementContext::toString()`
- [ ] 1.7 Wrap `src/ptxir/ptxir_serialization.cpp` if it includes ptx_ir statement types (verify in design Open Question 2)
- [ ] 1.8 Wrap remaining `src/ptx_ir/*.cpp` files (`ptxir_reader.cpp`, `ptxir_writer.cpp`, `ptx_syntax_utils.cpp`, `instruction_latency_table.cpp`) in `namespace ptxemu::ir { ... }` as needed
- [ ] 1.9 Build + ctest 252/252 verification: `cmake --build build -j$(nproc) && cd build && ctest --output-on-failure` (per ptx-lessons-learned §3)
- [ ] 1.10 Commit: `refactor(ptx-1.5c+d): namespace wrap + shim, src/ptx_ir self-migration` (1.5c+d merged per design D2 + D3 reasoning)

## 2. Phase 1.5e — src/ptx_parser/ caller sweep (1 commit, ~1h)

- [ ] 2.1 Sweep `src/ptx_parser/{cfg_builder,ptx_parser,ptx_visitor,ptx_visitor_atom,ptx_visitor_barrier,ptx_visitor_branch,ptx_visitor_call,ptx_visitor_dispatch,ptx_visitor_generic,ptx_visitor_memory,ptx_visitor_operands,ptx_visitor_simple,ptx_visitor_special,ptx_visitor_tcgen05,ptx_visitor_warp}.cpp` + corresponding headers — replace bare `Qualifier` / `StatementType` / `StatementContext` / `OperandContext` / `InstrVariant` / `Tcgen05Instr` etc with `ptxemu::ir::*` qualified form
- [ ] 2.2 Build + ctest 252/252 verification
- [ ] 2.3 Commit: `refactor(ptx-1.5e): src/ptx_parser caller sweep`

## 3. Phase 1.5f — src/ptxsim/ caller sweep (1 commit, ~2h)

- [ ] 3.1 Sweep `src/ptxsim/instructions/{arithmetic,atomic,barrier,memory,tcgen05_cp,...}.cpp` (~10 files) — replace bare IR type names with `ptxemu::ir::*`
- [ ] 3.2 Sweep `src/ptxsim/core/{cta_context,gpu_context,register_access_layer,sm_context,sm_context_cpptlm_inject,thread_context,warp_context_dispatch}.cpp` (~7 files) — same replacement
- [ ] 3.3 Sweep `src/ptxsim/{debug/ptx_config,instruction_base,instruction_factory,instruction_handlers,utils/qualifier_utils,utils/type_utils}.cpp` (~6 files) — same replacement
- [ ] 3.4 Build + ctest 252/252 verification
- [ ] 3.5 Commit: `refactor(ptx-1.5f): src/ptxsim caller sweep`

## 4. Phase 1.5g — src/cudart/ caller sweep (1 commit, ~1h)

- [ ] 4.1 Sweep `src/cudart/{cpptlm_module,ptx_context_adapter,ptx_interpreter,ptx_interpreter.h,ptxir_loader}.cpp/h` — replace bare IR type names with `ptxemu::ir::*`
- [ ] 4.2 Build + ctest 252/252 verification (including `integration_cpptlm_module*`, `e2e_image_executor*`, `e2e_ptxir_fatbinary*` tests)
- [ ] 4.3 Commit: `refactor(ptx-1.5g): src/cudart caller sweep`

## 5. Phase 1.5h — include/ subdirectory sweep (1 commit, ~1h)

- [ ] 5.1 Sweep `include/ptxsim/{common_types,contexts/program_ref,cta_context,gpu_context.h,instruction_factory,instruction_handlers,instructions/cvt/cvt_strategy,instructions/tcgen05,ptx_config,register_analyzer,simt_pc_manager,sm_context,testing/debug_utils,testing/instruction_helpers,testing/memory_test_utils,testing/scheduler_utils,testing/warp_test_utils,thread_context,utils/qualifier_utils,utils/type_utils}.h` — same replacement
- [ ] 5.2 Sweep `include/cudart/{module_registry,ptx_context_adapter,ptxir_loader}.h` — same replacement
- [ ] 5.3 Sweep `include/ptx_parser/{cfg_builder,ptx_parser,ptx_visiter}.h` — same replacement
- [ ] 5.4 Sweep `include/ptxemu/testing/warp_executor_test_fixture.h` — same replacement
- [ ] 5.5 Sweep `include/register/register_bank_manager.h`, `include/utils/ptx_lane_verification.h` — same replacement
- [ ] 5.6 Build + ctest 252/252 verification
- [ ] 5.7 Commit: `refactor(ptx-1.5h): include/ subdirectory sweep`

## 6. Phase 1.5i — tests/ subdirectory sweep (1 commit, ~2h)

- [ ] 6.1 Sweep `tests/unit/**/*.cpp` (~80 files) — replace bare IR type names with `ptxemu::ir::*`; preserve test fixture semantics (e.g., `StatementContext{type, data}` initializer syntax)
- [ ] 6.2 Sweep `tests/integration/**/*.cpp` (~40 files) — same replacement; pay special attention to `integration_attach_timing_consumer_e2e.cpp`, `integration_ptxir_*` tests
- [ ] 6.3 Sweep `tests/e2e/**/*.cpp` + `tests/ptx/**/*.ptx` driver if applicable
- [ ] 6.4 Build + ctest 252/252 verification; run `./tests/ptx/test_all_ptx.sh` for PTX grammar regression (per ptx-grammar-modification skill, no .g4 change so should be unchanged)
- [ ] 6.5 Commit: `refactor(ptx-1.5i): tests/ subdirectory sweep`

## 7. Phase 1.5j — GPUContext interface re-sign (1 commit, ~30min)

- [ ] 7.1 Modify `include/ptxsim/gpu_context.h:58` `std::vector<StatementContext> *statements;` → `std::vector<ptxemu::ir::StatementContext> *statements;`
- [ ] 7.2 Modify `include/ptxsim/gpu_context.h:80` `std::vector<StatementContext> &_stmts,` → `std::vector<ptxemu::ir::StatementContext> &_stmts,`
- [ ] 7.3 Modify `include/ptxsim/gpu_context.h:173` (or current line per source) `std::vector<StatementContext> &statements,` → `std::vector<ptxemu::ir::StatementContext> &statements,`
- [ ] 7.4 Verify all `src/ptxsim/core/gpu_context.cpp` implementation matches new signature (transitively included via 1.5f sweep but explicit verification required)
- [ ] 7.5 Build + ctest 252/252 verification
- [ ] 7.6 Commit: `refactor(ptx-1.5j): gpu_context.h StatementContext -> ptxemu::ir::StatementContext`

## 8. Phase 1.5k — drift_check Invariant 8 + spec validation + audit postmortem (1 commit, ~30min)

- [ ] 8.1 Add Invariant 8 step to `.github/workflows/drift_check.yml` per `specs/ci-drift-check/spec.md` Scenario: grep for bare `Qualifier` / `StatementContext` / `OperandContext` / `InstrVariant` / `Tcgen05Instr` / `Tcgen05OpKind` / `Tcgen05Dtype` outside `include/ptx_ir/*.h` shim
- [ ] 8.2 Run drift_check locally: simulate Invariant 8 grep against current source; verify 0 hits and exit 0
- [ ] 8.3 Run full local drift_check (1-7 invariants) — all pass
- [ ] 8.4 Verify `openspec/specs/statement-ir-public/spec.md` §46-48 Scenario "旧路径 forwarding header 一个 release 周期" implementation matches (shim form + namespace alias)
- [ ] 8.5 Verify `openspec/specs/ptxemu-ir-namespace-contract/spec.md` ADDED Requirements all met (grep source for forbidden patterns)
- [ ] 8.6 Update `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` §Postmortem "Phase 1.5 deferred" → "Phase 1.5 completed (commits <list>); task 9.4 1 release cycle window opens now"
- [ ] 8.7 Update root `AGENTS.md` HSK chain table: HSK-8 row notes "Phase 1.5 closure: ptxemu::ir canonical namespace + ptx_ir/ forwarding shim (per `phase-1-5-namespace-migration` change)"
- [ ] 8.8 Update `include/ptx_ir/AGENTS.md` "Phase 1.5+1 release cycle" section: 1 release cycle cleanup window now active (per task 9.4)
- [ ] 8.9 Build + ctest 252/252 final verification
- [ ] 8.10 Commit: `chore(ptx-1.5k): drift_check invariant 8 + spec scenario validation + audit postmortem closure`

## 9. Phase 1.5 final — push + archive (independent of code commits)

- [ ] 9.1 Push all 7-9 Phase 1.5 commits to `origin/main` via `git push origin main` (no force; branch protection may require PR — fall back to branch + PR + merge)
- [ ] 9.2 Verify GitHub `drift_check.yml` workflow passes on push (7 + 1 invariants)
- [ ] 9.3 Verify `origin/main` HEAD = local HEAD (commit hash match)
- [ ] 9.4 Run `openspec archive phase-1-5-namespace-migration` to archive the change (per openspec-archive-change skill)
- [ ] 9.5 Verify archive: `openspec/specs/{ptxemu-ir-namespace-contract,statement-ir-public,ci-drift-check}/` content matches the spec delta from this change
- [ ] 9.6 Update `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` "Phase 1.5 完成" entry with final commit list and archive reference

---

## Per-phase discipline reminders

- **Per ptx-lessons-learned §3**: 每 Phase 独立 commit, 任何已有测试回归立即 `git revert HEAD` 不混入后续 commit
- **Per ptx-lessons-learned §4**: 重大修改前 1 分钟可建立 `.worktrees/baseline-check` worktree (本 change 已建立 baseline = `2cd8449e` 全量 ctest 252/252 验证)
- **Per ptx-lessons-learned §6**: artifacts-first — `openspec/changes/phase-1-5-namespace-migration/{proposal,design}.md` + `specs/**` 必须 git-tracked (本 tasks.md 即时提交, 不延后到 archive)
- **Per ptx-lessons-learned §D**: commit message 引用相关 spec ID + ADR 编号 + commit hash; 复合 fix 详细列出每个 fix 原因
- **Per openspec-apply-change skill**: 实施中如发现 design issue, 暂停实施并提示更新 artifacts; 不 amend 已归档 artifacts
