# Phase 1.5c+d postmortem

**Branch:** `feat/phase-1-5-namespace-migration`
**Date:** 2026-08-27
**Scope:** tasks.md 1.1-1.9 (shim + canonical cpp wrap + serialization + InstructionState)

## Commits

```
cf8d161e refactor(ptx-1.5c+d): route InstructionState to canonical ptxemu::ir
ff18c6ea refactor(ptx-1.5c+d): shim swap + canonical wrap + serialization
36bbb208 docs(openspec): note temporary enumerator bridge in 1.5c+d shim
02aa88c1 docs(notes): record Oracle Phase 1.5c+d implementation strategy
8c5a07b8 docs(notes): record Phase 0 baseline for phase-1-5-namespace-migration
c71f2b1c feat(scripts): add token-aware ptxemu::ir namespace scanner
```

## Verification

| Gate | Result |
|------|--------|
| Full build | 100% PASS |
| ctest | **252/252 PASS** |
| Scanner (post-shim) | 433 → 2 bare-token files (bridge carries 431) |
| `cuda*`/`__cuda*` symbols | 38/38 preserved (HSK-8 ABI freeze) |
| `ptxemu_image_*` symbols | 7/7 preserved |
| `PTXEMU_API_VERSION` | 1 (frozen) |
| `enum class InstructionState` | 1 hit (canonical only) |

## Deviations from Oracle strategy

Oracle's session `ses_fbe96b30affeZ7oXY8bZ0XARYh` recommended **3
commits, ≤30 files each** for 1.5c+d. The plan committed C0+C1+C2
followed that discipline. Several empirical adjustments were needed
during execution:

### 1. 15 files instead of 10 in C1 (Oracle plan = 10)

The plan listed 10 files; actual commit changed 15. Five extra
header fixes were forced by forward-declaration collisions with
the shim's new using-declarations:

- `include/ptxsim/warp_context.h` (line 40 forward decl)
- `src/ptxsim/core/warp_context_dispatch.h` (line 5)
- `src/ptxsim/core/sm_context_cpptlm_inject.h` (line 10)
- `include/ptxsim/register_access_layer.h` (lines 9-10)
- `include/ptx_ir/statement_factory.h` (operand type imports into
  `ptxir::factory` namespace)

Root cause: existing global-scope `class/struct X;` forward
declarations collide with `using ::ptxemu::ir::X;` (using-declaration
cannot refer to a type already declared in the same scope). The fix
is to move forward declarations into `namespace ptxemu { namespace
ir { ... } }` and use qualified names at call sites.

Oracle's plan did not anticipate this. These 5 changes are still
within the D3 ≤30 files/commit budget.

### 2. `qualifier_utils.cpp` Q2bytes moved to `namespace ptxsim`

The Oracle plan said "delete duplicate `Q2bytes` implementation" but
that would have changed the return value for non-data qualifiers
(e.g. `.lt`) from 0 (legacy) to assert (canonical). To preserve
baseline behavior, we kept the legacy `ptxsim::Q2bytes` as a file-local
helper and only the **header** declaration was changed
(`namespace ptxsim { int Q2bytes(Qualifier q); }`). Internal call
sites use `ptxsim::Q2bytes(...)` to avoid ADL ambiguity with the
canonical `ptxemu::ir::Q2bytes`. External callers (cta_context,
register_analyzer, etc.) now resolve `Q2bytes` via ADL to canonical
only — same call shape, different target.

### 3. Bridge `using enum` for unscoped enumerators

Oracle's C1 plan said "shim is type-only, no enumerator export"
— but that was the **end-state** policy (per `tasks.md 1.1`). For
1.5c+d to be a single green commit, the shim carries a temporary
C++20 `using enum` bridge for `StatementType` and `OperandType`
so the 87 caller files (src 18, include 8, tests 62) keep
compiling until the 1.5e-1.5i3 sweeps qualify every reference. The
bridge is removed in 1.5k (per task 9.4) when Invariant 8 prevents
regression.

This was already documented in commit `36bbb208` and
`ff18c6ea`. The Oracle strategy document was updated accordingly.

### 4. `StatementContext::toString` signature

The shim exposes canonical `StatementContext::toString(int bytes
= 0)` (per canonical `ptxemu/ir/statement.h:340`). The
implementation in `src/ptx_ir/statement_context.cpp` matched the
old `toString()` (no args). Updated to `toString(int bytes)`. No
caller passes an argument; the default argument keeps all call
sites source-compatible.

## Sanity-check: shim vs canonical include graph

```
<ptx_ir/ptx_types.h> shim
  ├─ #include <ptxemu/ir/ptx_types.h>           (canonical)
  ├─ namespace ptx_ir = ::ptxemu::ir
  ├─ using ::ptxemu::ir::Qualifier;
  ├─ using ::ptxemu::ir::StatementType;
  ├─ using ::ptxemu::ir::OperandType;
  ├─ using ::ptxemu::ir::OperandKind;
  ├─ using enum ::ptxemu::ir::StatementType;     (1.5c+d bridge)
  └─ using enum ::ptxemu::ir::OperandType;       (1.5c+d bridge)

<ptx_ir/operand_context.h> shim
  └─ #include <ptxemu/ir/operand_context.h>
     └─ using ::ptxemu::ir::{RegOperand,VariableOperand,ImmOperand,
                              Predicate,AddrOperand,VecOperand,
                              OperandContext};

<ptx_ir/statement_context.h> shim
  ├─ #include <ptxemu/ir/statement.h>
  ├─ #include <ptxemu/ir/operand_context.h>
  ├─ #include "ptx_types.h"                       (for the bridge)
  ├─ #include "ptxsim/execution_types.h"
  └─ using ::ptxemu::ir::{Qualifier,StatementType,OperandType,
                          OperandKind,OperandContext,
                          DeclarationInstr,...,StatementContext};
```

`Q2s`, `Q2bytes`, `S2s`, `extractREG` are deliberately **not**
re-exported by the shim — the legacy global `Q2bytes`
implementation in `src/ptxsim/utils/qualifier_utils.cpp`
(`ptxsim::Q2bytes`) remains, and external callers see the canonical
strict version via ADL.

## CppTLM consumer gate

The Oracle strategy document flagged a CppTLM chained build as a
post-1.5c+d acceptance check (Invariant 4: `cpp 不暴露`). This
session was executed without access to the CppTLM worktree; the
check must be performed in a follow-up session:

```bash
cd /path/to/CppTLM
git submodule update --init
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

Expected outcome: clean build + green tests, because
`libptxemu_device.so` and `libptxemu_core.so` ABI surface is
unchanged (38 cuda*/__cuda* + 7 ptxemu_image_* + PTXEMU_API_VERSION=1
all preserved).

## Lessons learned (carried into 1.5e-1.5i3)

1. **Shim cannot coexist with global `class/struct` forward
   declarations** of the same name. All IR forward declarations
   must move into `namespace ptxemu { namespace ir { ... } }` before
   the shim is included. Sweep phases must enforce this when they
   touch a new header.

2. **`struct X` elaborated-type-specifier in TU boundary
   headers** (e.g. `ptxir_serialization.h:14`) silently re-declares
   `::X` once the shim is in scope. The rewrite to
   `ptxemu::ir::X` + `<ptxemu/ir/X.h>` include is mandatory at
   every shim-touching boundary.

3. **Two `X`-with-same-name functions in different namespaces**
   (legacy `::Q2bytes` + canonical `ptxemu::ir::Q2bytes`) make
   unqualified call sites ambiguous. Either qualify the call
   (`ptxemu::ir::Q2bytes(...)`) or remove the legacy. We chose the
   former to keep one compatibility seam under our control.

4. **`using enum` (C++20) at the top of the shim is the cheapest way
   to keep the ~87 caller files compiling without changing them.
   The bridge does not change type identity, does not change
   mangling, and does not add symbols — only injects enumerator
   constants into the global scope. Removed in 1.5k.
