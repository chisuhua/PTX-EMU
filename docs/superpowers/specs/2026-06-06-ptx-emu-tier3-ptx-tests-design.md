# P1-4: Tier 3 Simulator-Driven Equivalent Tests — Design

**Date**: 2026-06-06
**Status**: Approved (user confirmed 2026-06-06)
**Parent**: [`2026-06-06-ptx-emu-test-coverage-roadmap.md`](./2026-06-06-ptx-emu-test-coverage-roadmap.md) §2
**Estimated effort**: 1 day
**Out of scope**: WMMA/MMA, new PTX instruction implementation, performance benchmarks

---

## 1. Goal

Add 5 simulator-driven integration tests in `tests/integration/ptx/`, one per uncovered
`tests/reference/ptx_builtin/test_ptx_*.cu` family. After completion, `ctest -L "integration;ptx"`
must enumerate 7 targets (2 existing + 5 new), all passing, and `sanity.sh --tier 3` exits 0.

## 2. Pre-investigation findings (verified 2026-06-06)

| # | Finding | Impact on design |
|---|---|---|
| 1 | All 5 instruction families have working handlers in the simulator. Verified file-by-file: `bitwise.cpp` (AndHandler, OrHandler, XorHandler, NotHandler, ShlHandler, ShrHandler), `arithmetic_conversion.cpp` (CvtHandler), `data_transfer.cpp` (CvtaHandler), `arithmetic_ext.cpp` (AddcHandler, SubcHandler, Mul24Handler), `arithmetic_muldiv.cpp` (MadHandler), `arithmetic_utils.h` template (float add/sub/mul/div with Q_F32/Q_F64 qualifiers). | **P1-4 is pure test-writing work.** No instruction implementation is needed. |
| 2 | There is **no separate `S_FADD`/`S_FSUB`/`S_FMUL`/`S_FDIV` opcode**. Float arithmetic reuses `S_ADD`/`S_SUB`/`S_MUL`/`S_DIV` with `Q_F32`/`Q_F64` qualifiers (verified in `include/ptx_ir/ptx_op.def` — no F-prefixed entries for basic arithmetic; only `S_FMA` exists). | The new `make_fadd` etc. factories must set `Q_F32` qualifier, not a different opcode. |
| 3 | Existing `make_*` factories in `include/ptxsim/testing/instruction_helpers.h`: `bar_warp_sync`, `bar_sync`, `mov`, `mov_imm`, `add`, `mul`, `ld_shared`, `st_shared`, `setp_lt`, `bra`, `bra_pred`, `label`, `nop`, `exit`, `ret`. | Missing: `sub` (inlined in test_integer_arith.cpp:53 as local), `and`, `or`, `xor`, `not`, `shl`, `shr`, `cvt`, `cvta_to_global`, `cvta_to_shared`, `addc`, `subc`, `mad`, `mul24`, `fadd`, `fsub`, `fmul`, `fdiv`, `ffma`. **18 new factories + 1 promotion = 19 changes needed.** |
| 4 | Established naming convention (verified in `tests/integration/CMakeLists.txt:113-130`): ctest target = `integration_ptx_<family>`, filename = `test_<family>.cpp`, labels = `"integration;ptx;<family>;<variants>"`. | My naming in §3 follows this convention. |
| 5 | `instruction_factory.cpp` registers handlers via X-macro over `ptx_op.def`. If a handler is missing or stub, the test fails at runtime (not at registration). | TDD red-green works naturally — first test run may surface unimplemented edge cases. |

## 3. File list

### 3.1 New test files (5, all in `tests/integration/ptx/`)

| New file | ctest target | Reference | Instructions tested |
|---|---|---|---|
| `test_bitwise.cpp` | `integration_ptx_bitwise` | `tests/reference/ptx_builtin/test_ptx_bitwise.cu` | `and`, `or`, `xor`, `not`, `shl`, `shr` (B32, B64) |
| `test_cvt.cpp` | `integration_ptx_cvt` | `tests/reference/ptx_builtin/test_ptx_cvt.cu` | `cvt.s32.f32`, `cvt.f32.s32`, `cvt.f64.f32`, `cvt.f32.f64`, `cvt.s64.f64` |
| `test_float_arith.cpp` | `integration_ptx_float_arith` | `tests/reference/ptx_builtin/test_ptx_float.cu` | `fadd`, `fsub`, `fmul`, `fdiv`, `ffma` (F32, F64) |
| `test_extended.cpp` | `integration_ptx_extended` | `tests/reference/ptx_builtin/test_ptx_extended.cu` | `addc`, `subc`, `mad`, `mul24`, `sat` |
| `test_cvta.cpp` | `integration_ptx_cvta` | `tests/reference/ptx_builtin/test_ptx_cvta.cu` | `cvta.to.global`, `cvta.to.shared` |

### 3.2 Modified files (2)

| File | Change |
|---|---|
| `include/ptxsim/testing/instruction_helpers.h` | Add 18 new `make_*` factories (6 bitwise + 1 cvt + 2 cvta + 4 extended + 5 float) + promote `make_sub` from local to header. Estimated +90 lines. |
| `tests/integration/CMakeLists.txt` | Add 5 `add_catch_test` + 5 `set_tests_properties` blocks. Estimated +25 lines. |

### 3.3 Untouched files

- `tests/reference/ptx_builtin/*.cu` — these are NVIDIA PTX semantics reference tests, not simulator-driven. They live in `reference/` and are not built by `sanity.sh`. (See `tests/reference/ptx_builtin/README.md`.)
- All `src/ptxsim/instructions/*.cpp` — handlers exist; no changes needed for P1-4.

## 4. Architecture / data flow / error handling

### 4.1 Per-test pattern (mirrors `tests/integration/ptx/test_integer_arith.cpp`)

```
TEST_CASE("bitwise: and.b32 lane-neutral", "[bitwise][and]") {
    // 1. Build minimal statement sequence
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");         // r1[lane] = lane_id (set by RegisterBankManager below)
    stmts[1] = make_and("r2", "r1", "r1");    // r2[lane] = r1[lane] & r1[lane] == lane_id
    stmts[2] = make_ret();

    // 2. Set up SMContext + WarpContext via existing helper
    //    (see test_integer_arith.cpp:140-160 for reference setup)
    auto v = build_instrs(stmts);
    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, v);

    // 3. Set per-lane r1 values via RegisterBankManager
    auto& rbm = w->get_register_bank_manager();
    for (int lane = 0; lane < 32; ++lane) {
        rbm.set_value<int32_t>(w->get_lane(lane), "r1", lane);
    }

    // 4. Drive execution
    step_warp(w, v);

    // 5. Assert per-lane r2 values
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(rbm.get_value<int32_t>(w->get_lane(lane), "r2") == lane);
    }
}
```

### 4.2 Data flow

1. `RegisterBankManager` sets per-lane input values BEFORE `step_warp`
2. `step_warp(w, stmts)` invokes the scheduler (SM) → `execute_warp_instruction` (Warp) → per-thread `execute_thread_instruction` (Thread) → handler dispatch (Factory → AddHandler/CvtHandler/etc.) → writes back to register bank
3. `RegisterBankManager` reads per-lane output values AFTER `step_warp`
4. Catch2 assertions verify correctness

### 4.3 Error handling

- Per-lane `REQUIRE` with descriptive `INFO`/`CAPTURE` on failure (lane number, expected, actual)
- No empty catch blocks; no `as any` / `@ts-ignore` equivalents in C++ (e.g. no `reinterpret_cast` for unrelated types)
- If a handler is unimplemented or crashes, the test surfaces the issue immediately (TDD red → investigate → fix handler or test)
- All exception types from `step_warp` propagate to Catch2 (do not swallow)

## 5. New factory functions (`instruction_helpers.h`)

### 5.1 Promote `make_sub` (currently local in `test_integer_arith.cpp:53-66`)

```cpp
inline StatementContext make_sub(const std::string& dst, const std::string& src1,
                                  const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_SUB;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "sub.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}
```

### 5.2 Bitwise factories (template: see `make_and`)

```cpp
inline StatementContext make_and(const std::string& dst, const std::string& src1,
                                  const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_AND;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "and.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}
// make_or, make_xor, make_shl, make_shr: same structure, S_OR/S_XOR/S_SHL/S_SHR
// make_not: 2 operands (no src2), S_NOT
```

### 5.3 CVT factory (with dtype/stype qualifiers)

```cpp
inline StatementContext make_cvt(const std::string& dst, const std::string& src,
                                  Qualifier dst_dtype, Qualifier src_dtype) {
    StatementContext ctx;
    ctx.type = S_CVT;
    GenericInstr instr;
    instr.qualifiers = {dst_dtype, src_dtype};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "cvt." + qualifier_name(dst_dtype) + "." +
                          qualifier_name(src_dtype) + " " + dst + ", " + src + ";";
    return ctx;
}
```

### 5.4 CVTA factories

```cpp
inline StatementContext make_cvta_to_global(const std::string& dst, const std::string& src) {
    StatementContext ctx;
    ctx.type = S_CVTA;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_U64, Qualifier::Q_GLOBAL};  // cvta.to.global.u64
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "cvta.to.global.u64 " + dst + ", " + src + ";";
    return ctx;
}
inline StatementContext make_cvta_to_shared(...) { ... }  // similar, Q_SHARED
```

### 5.5 Extended factories

`make_addc`, `make_subc`, `make_mul24` mirror `make_add` (3 operands, Q_B32). `make_mad` takes 4 operands:
```cpp
inline StatementContext make_mad(const std::string& dst,
                                  const std::string& src1, const std::string& src2,
                                  const std::string& src3) {
    StatementContext ctx;
    ctx.type = S_MAD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src3, -1}});
    ctx.data = instr;
    ctx.instructionText = "mad.lo.s32 " + dst + ", " + src1 + ", " + src2 + ", " + src3 + ";";
    return ctx;
}
```

### 5.6 Float factories (reuse S_ADD/S_SUB with Q_F32)

```cpp
inline StatementContext make_fadd(const std::string& dst, const std::string& src1,
                                   const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_ADD;                       // NOTE: shared with integer add
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};  // Q_F32 selects float path in arithmetic_utils.h
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "add.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}
// make_fsub, make_fmul, make_fdiv: same structure, S_SUB/S_MUL/S_DIV + Q_F32
// make_ffma: S_FMA + Q_F32 (or Q_F64)
```

## 6. CMake additions (`tests/integration/CMakeLists.txt`)

Insert after existing `integration_ptx_integer_arith` block (line ~130):

```cmake
# ============================================================================
# P1-4: Tier 3 simulator-driven equivalent tests
# (added 2026-06-06 per docs/superpowers/specs/2026-06-06-ptx-emu-tier3-ptx-tests-design.md)
# ============================================================================
add_catch_test(integration_ptx_bitwise
    ptx/test_bitwise.cpp
)
set_tests_properties(integration_ptx_bitwise PROPERTIES LABELS "integration;ptx;bitwise")

add_catch_test(integration_ptx_cvt
    ptx/test_cvt.cpp
)
set_tests_properties(integration_ptx_cvt PROPERTIES LABELS "integration;ptx;cvt")

add_catch_test(integration_ptx_float_arith
    ptx/test_float_arith.cpp
)
set_tests_properties(integration_ptx_float_arith PROPERTIES LABELS "integration;ptx;float_arith;fadd;fsub;fmul;fdiv;ffma")

add_catch_test(integration_ptx_extended
    ptx/test_extended.cpp
)
set_tests_properties(integration_ptx_extended PROPERTIES LABELS "integration;ptx;extended;addc;subc;mad;mul24")

add_catch_test(integration_ptx_cvta
    ptx/test_cvta.cpp
)
set_tests_properties(integration_ptx_cvta PROPERTIES LABELS "integration;ptx;cvta")
```

## 7. Success criteria

- [ ] 5 new test files in `tests/integration/ptx/`
- [ ] `instruction_helpers.h` has 18 new `make_*` factories + `make_sub` promoted
- [ ] `tests/integration/CMakeLists.txt` has 5 new entries
- [ ] `cd build && ctest -L "integration;ptx" -V` shows 7 targets, all PASS
- [ ] `sanity.sh --tier 3` exits 0
- [ ] `sanity.sh` (default Tiers 1-9) exits 0, no regressions
- [ ] `tests/ptx/test_all_ptx.sh` still passes (orthogonal to Tier 3)
- [ ] All new files use `#ifndef`/`#define` guards if new headers are introduced (N/A here — only existing files modified)
- [ ] No `as any` / `@ts-ignore` / C-style casts introduced
- [ ] `clang-format -i` applied to all modified files

## 8. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `make_cvt` needs multiple qualifiers (dst dtype + src dtype) — API design awkward | Medium | Low | Use `std::vector<Qualifier>` or 2 separate `Qualifier` params (as shown in §5.3) |
| Float NaN/Inf/denormal edge cases fail in handler | Medium | Medium | Restrict tests to finite values (positive, negative, zero, ±small, ±large); flag edge cases as separate follow-up spec |
| `cvta` test needs shared/global memory symbol table setup | Medium | Low | Reference `test_ld_st_shared.cpp` setup pattern (already exists) |
| `make_sub` promotion breaks `test_integer_arith.cpp` (if local helper conflicts) | Low | Low | Remove the local `make_sub` from `test_integer_arith.cpp:53-66` once header version is in place |
| Untested instruction edge case in extended (e.g. carry/borrow for addc/subc) — handler exists but untested | Low | Low | Tests include 2-3 carry/borrow scenarios per instruction |
| Existing handlers have latent bugs exposed by new tests | Low | Medium | TDD red → investigate handler → fix or document as known issue in `KNOWN_ISSUES.md` |

## 9. Implementation order (TDD-aware)

1. **Add `make_sub` to header** → recompile → `ctest -R integration_ptx_integer_arith` still passes (no behavior change)
2. **Add bitwise factories + write `test_bitwise.cpp`** → `ctest -R integration_ptx_bitwise` → expect green
3. **Add CVT factory + write `test_cvt.cpp`** → `ctest -R integration_ptx_cvt` → expect green
4. **Add float factories + write `test_float_arith.cpp`** → `ctest -R integration_ptx_float_arith` → expect green
5. **Add extended factories + write `test_extended.cpp`** → `ctest -R integration_ptx_extended` → expect green
6. **Add CVTA factories + write `test_cvta.cpp`** → `ctest -R integration_ptx_cvta` → expect green
7. **Final validation**: `sanity.sh` (default) exits 0; `git log` shows one commit per logical step

## 10. What this design does NOT cover (intentional)

- WMMA/MMA tests (separate roadmap item, requires WMMA implementation)
- Performance benchmarks
- Atomic operations
- Memory ordering edge cases
- Multi-warp interactions (those belong to Tier 6/7/8)
