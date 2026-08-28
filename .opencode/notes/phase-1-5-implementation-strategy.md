# Phase 1.5c+d implementation strategy

**Author:** Oracle 2026-08-26 (session `ses_fbe96b30affeZ7oXY8bZ0XARYh`)
**Decision:** CONDITIONAL GO
**Status:** Pending execution
**Branch:** `feat/phase-1-5-namespace-migration` (HEAD = `8c5a07b8`)
**OpenSpec:** `openspec/changes/phase-1-5-namespace-migration/`

## Summary

Phase 1.5c+d is the **shim + canonical cpp namespace wrap + InstructionState
alias + serialization migration** per tasks.md §1 (1.1-1.9). It is the
prerequisite for the per-directory caller sweeps 1.5e-1.5i and the bridge
removal in 1.5k.

The implementation must satisfy ptx-lessons-learned §3-4 (per-phase
commit, each green at 252/252, failure → immediate revert, no contamination
of later commits) and design.md D3 (≤30 files/commit).

## Key insights from empirical investigation

### Why the previous WIP attempt regressed 70+ E2E tests

The earlier single-commit attempt regressed **70+ tests** with undefined
symbols. Investigation isolated **two distinct root causes**, neither
of which is the using-alias mangle drift feared in earlier reviews:

1. **`include/ptxir/ptxir_serialization.h:14-26` declares
   `struct StatementContext` as an elaborated-type-specifier without
   including any definition header.** After shim swap, TUs that include
   ONLY `ptxir_serialization.h` (e.g. `cudart_sim.cpp`) implicitly
   declare a NEW global `::StatementContext` that is a different type
   from `ptxemu::ir::StatementContext`. The serialize_to_string
   definition lives in a TU that includes the shim and resolves to
   canonical, so the two TUs mangle differently → link error.

2. **The canonical `Q2bytes` declaration has no implementation if the
   canonical cpp is not wrapped.** `include/ptxemu/ir/ptx_types.h:25`
   declares `int Q2bytes(Qualifier)` but `src/ptx_ir/ptx_types.cpp`
   still defines `::Q2bytes` (global namespace). If only the cpp is
   wrapped but the declaration in canonical already says
   `ptxemu::ir::Q2bytes`, then `int ::Q2bytes` is undeclared AND
   `int ptxemu::ir::Q2bytes` has no definition. Qualifier-utils.cpp
   had its own duplicate global `Q2bytes` (also declared in
   `include/ptxsim/utils/qualifier_utils.h:10`); both definitions must
   be merged into canonical.

The `Tcgen05Instr` mangle observed in the failed run was a **secondary
symptom** of (1) above (test caller mangling `ptxemu::ir::Tcgen05Instr`
because its TU eventually pulled canonical via the ptxir shim chain;
definition in `src/ptxsim/instructions/tcgen05.cpp:510` sits in
`namespace ptxsim { namespace { ... } }` and resolves via the shim's
`using` alias to global `Tcgen05Instr`). Once (1) is fixed, the secondary
symptom disappears.

### Why `using enum` is a temporary bridge, not policy violation

Tasks.md 1.1 pins the shim policy to "canonical type names only, do not
export bare `S_*`/`O_*` enumerators" — this is the **end-state contract**
for the 1.5k closure. During 1.5c+d the bridge is required:

- After shim swap, ~87 caller files use bare unscoped enumerators
  (tests 62, src 18, include 8). Without bridge, `1.5c+d` cannot be a
  green commit on its own; the bridge holds 1.5e-1.5i3 sweeps
  incremental-green until enumerator usage is swept and qualified.

The bridge is `using enum ::ptxemu::ir::StatementType;` and
`using enum ::ptxemu::ir::OperandType;` at the top of the
`include/ptx_ir/ptx_types.h` shim. C++20. Bridge removal is the
**first task** of 1.5k (after Invariant 8 is in place to prevent
regression) and is the gate that completes HSK-8 spec §33-48.

`using enum` carries **zero ABI impact**: it injects enumerator compile-time
constants into the global namespace; no new symbols are emitted; type
identity is unchanged. The mangle of any class with a using-alias is
always the canonical qualified name (C++ mangling ABI is determined by
the underlying type, not the alias path that brought it into scope).

### Why HSK-8 ABI freeze is narrower than full-dynsym diff

The current `libcudart.so` exports 4654 defined dynsyms. Of these,
**554** contain IR type mangling in their signatures (handler entry
points taking `vector<StatementContext>`, variant vtables for
`InstrVariant`, STL weak instantiations, etc.). After 1.5c+d all 554
must change by design — HSK-8 Decision 3 freezes
`PTXEMU_API_VERSION=1` and `device_api.h`, not the internal mangling of
PTX-EMU-implementation-detail functions in `libcudart.so`.

The actual HSK-8 contract surface per design.md:9,17,36 and
drift_check Invariant 4 (`cpp 不暴露`) is:

- `libcudart.so` exports the `cuda*`/`__cuda*` C ABI (unaffected by
  namespace changes — these are `extern "C"` symbols)
- `libptxemu_device.so` exports 8 `ptxemu_image_*` entry points
- `PTXEMU_API_VERSION=1` macro value

Verification filter:

```bash
nm -D --defined-only build/lib/libcudart.so | grep -E ' T (__cuda|cuda)'
nm -D --defined-only build/lib/libptxemu_device.so | grep ptxemu_image
grep PTXEMU_API_VERSION include/ptxemu/device_api.h
```

All three must match before and after 1.5c+d.

## Commit plan

### Commit C0 — artifacts sync (2 files, docs-only)

Per ptx-lessons-learned §6: artifacts adjustments must be committed
FIRST before implementation commits.

Files:

- `openspec/changes/phase-1-5-namespace-migration/design.md` — add a
  paragraph to D1 noting the temporary enumerator bridge is required
  for 1.5c+d green and is removed in 1.5k
- `openspec/changes/phase-1-5-namespace-migration/tasks.md` — note in
  task 1.1 that the "type-name-only" policy describes the end state;
  the bridge is part of phasing

Acceptance:

- `git status` clean except for the two artifacts files
- `openspec validate phase-1-5-namespace-migration` PASS
- Scanner output unchanged (still reports 433 caller files)

### Commit C1 — shim + canonical wrap + ODR dedup + serialization (10 files)

This commit is **atomic**: any subset of these changes breaks either
compile (shim swap without canonical wrap) or link (canonical wrap
without ODR dedup). Single commit, 10 files, well under the
30-files/commit limit.

Files:

1. `include/ptx_ir/ptx_types.h` — replace with forwarding shim:
   - `#include <ptxemu/ir/ptx_types.h>`
   - `namespace ptx_ir = ::ptxemu::ir;`
   - `using ::ptxemu::ir::Qualifier; using ::ptxemu::ir::StatementType;`
   - `using ::ptxemu::ir::OperandType; using ::ptxemu::ir::OperandKind;`
   - `using ::ptxemu::ir::Q2s; using ::ptxemu::ir::Q2bytes;`
   - `using ::ptxemu::ir::S2s; using ::ptxemu::ir::extractREG;`
   - **TEMPORARY bridge:** `using enum ::ptxemu::ir::StatementType;`
   - **TEMPORARY bridge:** `using enum ::ptxemu::ir::OperandType;`
   - Header guard `#ifndef PTX_TYPES_H`

2. `include/ptx_ir/operand_context.h` — forwarding shim with
   `using ::ptxemu::ir::RegOperand/VariableOperand/ImmOperand/Predicate/AddrOperand/VecOperand/OperandContext;`
   plus canonical include of `<ptxemu/ir/operand_context.h>` and
   `namespace ptx_ir = ::ptxemu::ir;`.

3. `include/ptx_ir/statement_context.h` — forwarding shim with
   using-declarations for **every** instruction struct in canonical
   `ptxemu/ir/statement.h`:
   `DeclarationInstr, DollarNameInstr, PragmaInstr, LabelInstr, VoidInstr, BranchInstr, BarrierInstr, MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, GenericInstr, Tcgen05OpKind, Tcgen05Dtype, Tcgen05Instr, AtomInstr, VoteInstr, ShflInstr, ActivemaskInstr, BarWarpSyncInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, AbiDirective, CpAsyncInstr, InstrVariant, StatementContext`
   plus canonical include of `<ptxemu/ir/statement.h>` and
   `namespace ptx_ir = ::ptxemu::ir;`. The `include/ptxsim/execution_types.h`
   transitive include from the old header should be replicated.

4. `src/ptx_ir/ptx_types.cpp` — wrap free functions in
   `namespace ptxemu { namespace ir { ... } }`. Function bodies:
   `extractREG`, `Q2s`, `Q2bytes`. Qualifier references remain bare
   (resolves to canonical Qualifier inside the namespace).

5. `src/ptx_ir/operand_context.cpp` — wrap `OperandContext::toString`
   in `namespace ptxemu { namespace ir { ... } }`.

6. `src/ptx_ir/statement_context.cpp` — wrap `S2s`, `qualifiersToString`,
   `operandsToString`, `StatementContext::toString` in
   `namespace ptxemu { namespace ir { ... } }`. Match canonical
   signature `StatementContext::toString(int bytes = 0) const`.

7. `include/ptxsim/utils/qualifier_utils.h` — **DELETE** the
   `int Q2bytes(Qualifier q);` declaration at line 10. Replace usages
   of unqualified `Q2bytes` with `ptxemu::ir::Q2bytes` if needed (the
   global alias should suffice; this is just the forward decl that
   conflicts with canonical).

8. `src/ptxsim/utils/qualifier_utils.cpp` — **DELETE** the `int Q2bytes(Qualifier q) { ... }`
   definition at line 16. Replace internal call site (`int bytes = Q2bytes(e);`)
   with `int bytes = ptxemu::ir::Q2bytes(e);` (the qualifiers are
   identical — both implementations return 8/4/2/1 for the standard
   data qualifier set; the canonical version additionally handles
   E4M3/E5M2/E4M3X4/PRED; if any caller relies on the canonical
   extras for non-data qualifiers the existing `default: return 0`
   masks them).

9. `include/ptxir/ptxir_serialization.h` — add
   `#include <ptxemu/ir/statement.h>` and replace every
   `std::vector<struct StatementContext>` with
   `std::vector<ptxemu::ir::StatementContext>`. Drop the `struct`
   elaborated-type-specifier prefix — elaborated-type-specifiers
   cannot refer to using-declared names and would re-introduce the
   link error we are eliminating.

10. `src/ptxir/ptxir_serialization.cpp` — match the header: every
    `std::vector<struct StatementContext>&` becomes
    `std::vector<ptxemu::ir::StatementContext>&`. Internal
    `fill_reconvergence_pc` and `generate_ptxir` block reference
    unscoped enumerators `S_LABEL`, `S_BRA`, `S_BAR` — these resolve
    via the bridge. `Qualifier::Q_U64`, `Qualifier::Q_PTR`, etc. resolve
    via the canonical Qualifier using-declaration.

Acceptance:

- `cmake --build build -j$(nproc)` succeeds (full rebuild, 15-20 min —
  the shim headers are included by 433 TUs).
- `cd build && ctest --output-on-failure` shows 252/252 PASS.
- `cd build && ctest -R ptxir --output-on-failure` shows 252/252 PASS
  (specifically sensitive subset).
- HSK-8 ABI freeze filter: contract symbols byte-identical:
  ```bash
  nm -D --defined-only build/lib/libcudart.so | grep -E ' T (__cuda|cuda)' > /tmp/current-cuda.txt
  diff /tmp/baseline-artifacts/libcudart-cuda-extern.txt /tmp/current-cuda.txt
  # must be empty
  nm -D --defined-only build/lib/libptxemu_device.so | grep ptxemu_image > /tmp/current-image.txt
  diff /tmp/baseline-artifacts/ptxemu_image-symbols.txt /tmp/current-image.txt
  # must be empty
  ```
- `drift_check` workflow Invariant 1-7 PASS (run locally via the
  script the workflow invokes, or skip and let CI verify).
- Scanner still reports 433 caller files with bare tokens (the bridge
  means we are NOT yet enforcement-clean; the 252 ctest result is the
  ground truth for behavior preservation).

Failure handling: per ptx-lessons-learned §3, if any acceptance check
fails, `git revert HEAD` and **do not** proceed to C2.

### Commit C2 — InstructionState alias (1 file)

`include/ptxsim/execution_types.h` — add
`#include <ptxemu/ir/execution_types.h>` at the top and replace the
global `enum class InstructionState { READY, PREPARE, EXECUTE, COMMIT };`
definition with `using ::ptxemu::ir::InstructionState;`. Preserve the
file's other contents (`EXE_STATE`, `BAR_TYPE`, `Dim3`, `CTAId`,
`std::hash<CTAId>`).

Acceptance:

- Incremental rebuild (~5-10 min — `execution_types.h` is widely
  included).
- `cd build && ctest --output-on-failure` shows 252/252 PASS.
- `git grep -n "enum class InstructionState" include/` returns exactly
  one hit (the canonical declaration in `ptxemu/ir/execution_types.h`).
- HSK-8 ABI freeze filter: unchanged (InstructionState is internal).

Failure handling: `git revert HEAD`; investigate whether the bridge
gap causes the regression (some file not picking up the canonical
InstructionState) and either extend the bridge or fix the caller as
part of this commit.

## Bridge removal (NOT in this phase)

The `using enum` bridge in `include/ptx_ir/ptx_types.h` is the first
thing removed in 1.5k. By that point, all 433 caller files have been
qualified to `ptxemu::ir::S_*` / `ptxemu::ir::O_*` via 1.5e-1.5i3
sweeps, and the drift_check Invariant 8 scanner prevents regression.

Bridge removal commit:

- Delete the two `using enum` lines from `include/ptx_ir/ptx_types.h`
- Update `tasks.md` task 9.4 (shim removal) to also remove the bridge
- Verify: `git grep -nE '\bS_[A-Z_]+\b' src include tests | grep -v ptxemu::ir | grep -v ptx_ir/ | grep -v ptx_qualifier.def` should match nothing (modulo comment/string-literal false positives that the scanner filters)

## Pre-flight: baseline nm regeneration

Before C1, regenerate the baseline nm in the correct format. The
existing `/tmp/baseline-artifacts/libcudart-nm-before.txt` is plain
`nm` output (addresses first), but the test invokes `nm -D --defined-only`
which produces a different layout. Mismatch triggers a spurious test
failure unrelated to namespace changes.

```bash
cd build
nm -D --defined-only lib/libcudart.so > /tmp/baseline-artifacts/libcudart-nm-before.txt
nm -D --defined-only lib/libcudart.so | grep -E ' T (__cuda|cuda)' \
  > /tmp/baseline-artifacts/libcudart-cuda-extern.txt
nm -D --defined-only lib/libptxemu_device.so | grep ptxemu_image \
  > /tmp/baseline-artifacts/ptxemu_image-symbols.txt
cd .. && cmake --build build -j$(nproc)  # rebuild to refresh symtab
cd build && ctest  # verify 252/252 still green
```

If `cd build && ctest` regresses after regeneration, the baseline nm
artifact was masking pre-existing failures; investigate those before
starting C1.

## Risk register (for the executor)

- **Q2bytes semantic divergence** between the two current definitions:
  diff `src/ptxir/ptx_types.cpp:54` against
  `src/ptxsim/utils/qualifier_utils.cpp:16` line by line. If any
  qualifier returns a different size, the unified canonical version
  changes behavior for one side. 252 ctest catches net regressions;
  for finer-grained checks add a `tests/unit/common/test_q2bytes.cpp`
  before C1 capturing both implementations' return values for the
  full Qualifier set.

- **ANTLR-generated headers** under `build/antlr4_generated_src/`:
  not in scope (per R4 / design.md D5), but the shim's `using` chain
  must keep them compilable. If generated code fails to compile after
  the shim swap, open a separate `fix-antlr4-namespace` change; do not
  expand C1.

- **CppTLM-as-consumer** (`cpp 不暴露` invariant 4): after C1, do a
  chained build in CppTLM worktree:
  ```bash
  cd /path/to/CppTLM
  git submodule update --init
  cmake --build build -j$(nproc)
  cd build && ctest  # consumer-side gate
  ```
  All consumer-side ctest must stay green; this validates that
  `libptxemu_device.so` ABI did not break for the only external
  consumer.

- **Full rebuild time** is the dominant cost (15-20 min per the
  Phase 0 baseline notes). Plan the work session to allow at least
  two rebuilds (one for C1, one for C2).

- **Per-phase commit discipline**: any acceptance failure in C1 or
  C2 → `git revert HEAD` and **stop**. Do not amend, do not commit
  --fixup, do not push a partial fix on top — the next attempt must
  be a fresh commit on top of the previous successful phase's HEAD.

## Execution checklist

```
[ ] Pre-flight: regenerate baseline nm in `nm -D --defined-only` format
[ ] Pre-flight: diff `Q2bytes` implementations; record divergences
[ ] Pre-flight: rebuild + ctest 252/252 green on baseline
[ ] Commit C0: 2 artifacts files, no build impact
[ ] Commit C1: 10 files, full rebuild, ctest 252/252, ABI filter diff
[ ] Commit C2: 1 file, incremental rebuild, ctest 252/252, InstructionState grep
[ ] Post-C2: scoped build in CppTLM worktree to validate consumer gate
[ ] Update `.opencode/notes/phase-1-5-phase-1-5c-d-postmortem.md` with
    observed acceptances and any deviations from this strategy
```
