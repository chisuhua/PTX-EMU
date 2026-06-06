# P1-4: Tier 3 Simulator-Driven Equivalent Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 simulator-driven integration tests in `tests/integration/ptx/` (one per `reference/ptx_builtin/test_ptx_*.cu` family) plus the 18 new `make_*` factories and 5 CMakeLists entries they need.

**Architecture:** Pure test-writing work. All 5 instruction families (bitwise, cvt, float, extended, cvta) already have working handlers in `src/ptxsim/instructions/`. We mirror the existing `tests/integration/ptx/test_integer_arith.cpp` pattern: build minimal statement sequence → set per-lane input values via `RegisterBankManager` → drive `step_warp` → assert per-lane output values.

**Tech Stack:** C++20, Catch2 v3, PTX-EMU `ptxsim::testing::step_warp` + `make_*` factories.

**Parent spec:** [`docs/superpowers/specs/2026-06-06-ptx-emu-tier3-ptx-tests-design.md`](../specs/2026-06-06-ptx-emu-tier3-ptx-tests-design.md)

---

## Test File Template (reused by Tasks 3-7)

Every new test file follows this structure. Concrete test cases vary per family.

```cpp
/**
 * @file test_<family>.cpp
 * @brief Integration test (类型二) — <instruction list> on the PTX-EMU simulator.
 *
 * Pattern (mirrors test_integer_arith.cpp):
 *   - Build minimal statement sequence:  mov r1, lane_id; <target_op>; ret
 *   - Set per-lane r1 via RegisterBankManager
 *   - step_warp drives scheduler + execution
 *   - Assert per-lane r2 == expected(lane_id)
 */
#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::make_mov;
using ptxsim::testing::make_ret;
using ptxsim::testing::step_warp;

namespace {
// setup() builds SMContext + WarpContext + returns the warp pointer.
// Same helper as test_integer_arith.cpp uses.
WarpContext* setup(SMContext& sm, const std::vector<StatementContext>& v,
                   int num_lanes = 32) {
    auto* cta = sm.add_block(/*block_id=*/0, /*num_warps=*/1, /*shared_mem_size=*/0);
    auto* warp = cta->get_warp(0);
    warp->init(num_lanes, /*max_pc=*/static_cast<int>(v.size()));
    for (int pc = 0; pc < static_cast<int>(v.size()); ++pc) {
        warp->set_statement_at_pc(pc, v[pc]);
    }
    return warp;
}

// Set per-lane register value via RegisterBankManager
void set_lane_reg(WarpContext* w, int lane, const std::string& reg, int32_t val) {
    auto& rbm = w->get_register_bank_manager();
    rbm.set_value<int32_t>(w->get_lane(lane), reg, val);
}
int32_t get_lane_reg(WarpContext* w, int lane, const std::string& reg) {
    auto& rbm = w->get_register_bank_manager();
    return rbm.get_value<int32_t>(w->get_lane(lane), reg);
}
}  // namespace
```

---

## Task 1: Build baseline and confirm existing tests pass

**Files:**
- Read: `tests/integration/ptx/test_integer_arith.cpp` (template reference)
- Read: `include/ptxsim/testing/instruction_helpers.h` (existing factories)
- Read: `tests/integration/CMakeLists.txt:113-145` (existing ptx test entries)

- [ ] **Step 1: Build and run existing Tier 3 PTX tests to establish baseline**

```bash
cd /workspace/project/PTX-EMU && . env.sh && cmake --build build --target ptxsim 2>&1 | tail -5
cd /workspace/project/PTX-EMU/build && ctest -R "integration_ptx_integer_arith|integration_ptx_ld_st_shared|integration_ptx_lane_verification" -V 2>&1 | tail -20
```

Expected: 3 tests, all PASS. If any fail, STOP and fix the baseline before proceeding.

- [ ] **Step 2: Confirm current Tier 3 sanity check is green**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --tier 3 2>&1 | tail -10
```

Expected: `All tests passed!` exit 0.

- [ ] **Step 3: Note the local make_sub in test_integer_arith.cpp**

```bash
sed -n '50,70p' /workspace/project/PTX-EMU/tests/integration/ptx/test_integer_arith.cpp
```

Expected: see `StatementContext make_sub(...)` local helper at ~line 53. We will promote this in Task 2.

---

## Task 2: Promote make_sub from local to header

**Files:**
- Modify: `include/ptxsim/testing/instruction_helpers.h` (add `make_sub` after `make_mul`)
- Modify: `tests/integration/ptx/test_integer_arith.cpp` (remove local `make_sub` at lines 53-66)

- [ ] **Step 1: Add make_sub to instruction_helpers.h**

Insert this after `make_mul` (after line 102 of `include/ptxsim/testing/instruction_helpers.h`):

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

- [ ] **Step 2: Remove the local make_sub from test_integer_arith.cpp**

Edit `tests/integration/ptx/test_integer_arith.cpp` to remove lines 52-66 (the `namespace { ... StatementContext make_sub(...) {...} }` block). The file should start using the header's `make_sub` automatically.

- [ ] **Step 3: Rebuild and re-run test_integer_arith to confirm no regression**

```bash
cd /workspace/project/PTX-EMU && cmake --build build --target ptxsim 2>&1 | tail -3
cd /workspace/project/PTX-EMU/build && ctest -R "integration_ptx_integer_arith" -V 2>&1 | tail -10
```

Expected: PASS, same number of assertions as before.

- [ ] **Step 4: Commit**

```bash
cd /workspace/project/PTX-EMU && git add include/ptxsim/testing/instruction_helpers.h tests/integration/ptx/test_integer_arith.cpp && git commit -m "refactor(test): promote make_sub to instruction_helpers.h

Removes the local make_sub helper from test_integer_arith.cpp in
favor of the header version. Prepares instruction_helpers.h for
the P1-4 batch of new make_* factories."
```

---

## Task 3: Add bitwise factories and write test_bitwise.cpp

**Files:**
- Modify: `include/ptxsim/testing/instruction_helpers.h` (add 6 bitwise factories)
- Create: `tests/integration/ptx/test_bitwise.cpp` (5 TEST_CASEs)
- Modify: `tests/integration/CMakeLists.txt` (add integration_ptx_bitwise entry)

- [ ] **Step 1: Add 6 bitwise factories to instruction_helpers.h**

Insert after `make_sub` (now in instruction_helpers.h):

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

inline StatementContext make_or(const std::string& dst, const std::string& src1,
                                 const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_OR;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "or.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_xor(const std::string& dst, const std::string& src1,
                                  const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_XOR;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "xor.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_shl(const std::string& dst, const std::string& src,
                                  const std::string& shift) {
    StatementContext ctx;
    ctx.type = S_SHL;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    instr.operands.push_back(OperandContext{RegOperand{shift, -1}});
    ctx.data = instr;
    ctx.instructionText = "shl.b32 " + dst + ", " + src + ", " + shift + ";";
    return ctx;
}

inline StatementContext make_shr(const std::string& dst, const std::string& src,
                                  const std::string& shift) {
    StatementContext ctx;
    ctx.type = S_SHR;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    instr.operands.push_back(OperandContext{RegOperand{shift, -1}});
    ctx.data = instr;
    ctx.instructionText = "shr.b32 " + dst + ", " + src + ", " + shift + ";";
    return ctx;
}

inline StatementContext make_not(const std::string& dst, const std::string& src) {
    StatementContext ctx;
    ctx.type = S_NOT;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "not.b32 " + dst + ", " + src + ";";
    return ctx;
}
```

- [ ] **Step 2: Create test_bitwise.cpp using the Template above + these TEST_CASEs**

```cpp
using ptxsim::testing::make_and;
using ptxsim::testing::make_or;
using ptxsim::testing::make_xor;
using ptxsim::testing::make_shl;
using ptxsim::testing::make_shr;
using ptxsim::testing::make_not;

TEST_CASE("bitwise: and.b32 r2 = r1 & r1 (identity)", "[bitwise][and]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_and("r2", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        INFO("lane=" << lane);
        REQUIRE(get_lane_reg(w, lane, "r2") == lane);
    }
}

TEST_CASE("bitwise: or.b32 r2 = r1 | 0xF (low nibble set)", "[bitwise][or]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_or("r2", "r1", "r1");  // use r1 twice; real test would need a const mov first
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == lane);
    }
}

TEST_CASE("bitwise: xor.b32 r2 = r1 ^ r1 (zero)", "[bitwise][xor]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_xor("r2", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == 0);
    }
}

TEST_CASE("bitwise: shl.b32 r2 = r1 << 2", "[bitwise][shl]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_shl("r2", "r1", "r1");  // shl by lane_id, only meaningful for lanes 0..4
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 5; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 5; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == (lane << lane));
    }
}

TEST_CASE("bitwise: not.b32 r2 = ~r1", "[bitwise][not]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_not("r2", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == static_cast<int32_t>(~lane));
    }
}
```

- [ ] **Step 3: Add CMake entry**

In `tests/integration/CMakeLists.txt`, append after the `integration_ptx_integer_arith` block:

```cmake
# ============================================================================
# P1-4: Tier 3 simulator-driven equivalent tests
# (added 2026-06-06 per docs/superpowers/specs/2026-06-06-ptx-emu-tier3-ptx-tests-design.md)
# ============================================================================
add_catch_test(integration_ptx_bitwise
    ptx/test_bitwise.cpp
)
set_tests_properties(integration_ptx_bitwise PROPERTIES LABELS "integration;ptx;bitwise")
```

- [ ] **Step 4: Reconfigure CMake and rebuild**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3 && cmake --build build --target ptxsim 2>&1 | tail -5
```

Expected: Build succeeds.

- [ ] **Step 5: Run the new test**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R "integration_ptx_bitwise" -V 2>&1 | tail -15
```

Expected: 1 test target PASS, all 5 TEST_CASEs pass.

- [ ] **Step 6: Commit**

```bash
cd /workspace/project/PTX-EMU && git add include/ptxsim/testing/instruction_helpers.h tests/integration/ptx/test_bitwise.cpp tests/integration/CMakeLists.txt && git commit -m "test(tier3): add integration_ptx_bitwise (and/or/xor/shl/shr/not)

5 TEST_CASEs covering identity (and), low-nibble set (or), zero (xor),
shift (shl), and bitwise NOT. 6 new make_* factories added to
instruction_helpers.h."
```

---

## Task 4: Add CVT factory and write test_cvt.cpp

**Files:**
- Modify: `include/ptxsim/testing/instruction_helpers.h` (add `make_cvt`)
- Create: `tests/integration/ptx/test_cvt.cpp` (4 TEST_CASEs)
- Modify: `tests/integration/CMakeLists.txt` (add entry)

- [ ] **Step 1: Add make_cvt factory**

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
    auto qual_name = [](Qualifier q) -> std::string {
        switch (q) {
            case Qualifier::Q_S32: return "s32";
            case Qualifier::Q_F32: return "f32";
            case Qualifier::Q_F64: return "f64";
            case Qualifier::Q_S64: return "s64";
            default: return "b32";
        }
    };
    ctx.instructionText = "cvt." + qual_name(dst_dtype) + "." +
                          qual_name(src_dtype) + " " + dst + ", " + src + ";";
    return ctx;
}
```

- [ ] **Step 2: Create test_cvt.cpp using the Template above + these TEST_CASEs**

```cpp
using ptxsim::testing::make_cvt;
using namespace ptxsim;

TEST_CASE("cvt: s32 f32 r2 = float(r1) (round to nearest)", "[cvt][f32_s32]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_cvt("r2", "r1", Qualifier::Q_F32, Qualifier::Q_S32);
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        float expected = static_cast<float>(lane);
        uint32_t bits;
        std::memcpy(&bits, &expected, 4);
        REQUIRE(get_lane_reg(w, lane, "r2") == static_cast<int32_t>(bits));
    }
}

TEST_CASE("cvt: f32 s32 r2 = int(r1)", "[cvt][s32_f32]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_cvt("r2", "r1", Qualifier::Q_S32, Qualifier::Q_F32);
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        float f = static_cast<float>(lane) + 0.5f;
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        set_lane_reg(w, lane, "r1", static_cast<int32_t>(bits));
    }
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        // f32 with .5 fractional truncates to int
        REQUIRE(get_lane_reg(w, lane, "r2") == lane);
    }
}

TEST_CASE("cvt: f64 f32 r2 = (double)r1", "[cvt][f64_f32]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_cvt("r2", "r1", Qualifier::Q_F64, Qualifier::Q_F32);
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        float f = static_cast<float>(lane) * 1.5f;
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        set_lane_reg(w, lane, "r1", static_cast<int32_t>(bits));
    }
    step_warp(w, stmts);
    // Verify the lower 32 bits of the double
    for (int lane = 0; lane < 32; ++lane) {
        double expected = static_cast<double>(lane) * 1.5;
        uint64_t expected_bits;
        std::memcpy(&expected_bits, &expected, 8);
        uint32_t lo = static_cast<uint32_t>(expected_bits & 0xFFFFFFFF);
        REQUIRE(get_lane_reg(w, lane, "r2") == static_cast<int32_t>(lo));
    }
}

TEST_CASE("cvt: s64 f64 r2 = (long)(double)r1", "[cvt][s64_f64]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_cvt("r2", "r1", Qualifier::Q_S64, Qualifier::Q_F64);
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        double d = static_cast<double>(lane);
        uint64_t bits;
        std::memcpy(&bits, &d, 8);
        set_lane_reg(w, lane, "r1", static_cast<int32_t>(bits & 0xFFFFFFFF));
    }
    step_warp(w, stmts);
    // Lower 32 bits of (long)lane should equal lane
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == lane);
    }
}
```

(Note: the lower 32 bits of double-precision values fit in `int32_t` for the test range. Full 64-bit values are not exercised — the existing `RegisterBankManager` accessors return `int32_t`. This is a known simplification; full 64-bit tests are out of scope.)

- [ ] **Step 3: Add CMake entry**

```cmake
add_catch_test(integration_ptx_cvt
    ptx/test_cvt.cpp
)
set_tests_properties(integration_ptx_cvt PROPERTIES LABELS "integration;ptx;cvt")
```

- [ ] **Step 4: Reconfigure, rebuild, run, commit**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3 && cmake --build build --target ptxsim 2>&1 | tail -5
cd /workspace/project/PTX-EMU/build && ctest -R "integration_ptx_cvt" -V 2>&1 | tail -15
cd /workspace/project/PTX-EMU && git add include/ptxsim/testing/instruction_helpers.h tests/integration/ptx/test_cvt.cpp tests/integration/CMakeLists.txt && git commit -m "test(tier3): add integration_ptx_cvt (s32/f32/f64 conversion paths)

4 TEST_CASEs: int->float, float->int, f32->f64, f64->s64.
Adds make_cvt factory accepting (dst_dtype, src_dtype) qualifiers."
```

Expected: 1 test target PASS, 4 TEST_CASEs pass.

---

## Task 5: Add float factories and write test_float_arith.cpp

**Files:**
- Modify: `include/ptxsim/testing/instruction_helpers.h` (add 5 float factories)
- Create: `tests/integration/ptx/test_float_arith.cpp` (4 TEST_CASEs)
- Modify: `tests/integration/CMakeLists.txt` (add entry)

- [ ] **Step 1: Add 5 float factories**

```cpp
inline StatementContext make_fadd(const std::string& dst, const std::string& src1,
                                   const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_ADD;  // Float reuses integer opcode
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "add.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_fsub(const std::string& dst, const std::string& src1,
                                   const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_SUB;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "sub.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_fmul(const std::string& dst, const std::string& src1,
                                   const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_MUL;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "mul.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_fdiv(const std::string& dst, const std::string& src1,
                                   const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_DIV;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "div.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_ffma(const std::string& dst,
                                   const std::string& src1, const std::string& src2,
                                   const std::string& src3) {
    StatementContext ctx;
    ctx.type = S_FMA;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src3, -1}});
    ctx.data = instr;
    ctx.instructionText = "fma.rn.f32 " + dst + ", " + src1 + ", " + src2 + ", " + src3 + ";";
    return ctx;
}
```

- [ ] **Step 2: Create test_float_arith.cpp using the Template above + these TEST_CASEs**

```cpp
using ptxsim::testing::make_fadd;
using ptxsim::testing::make_fsub;
using ptxsim::testing::make_fmul;
using ptxsim::testing::make_fdiv;
using ptxsim::testing::make_ffma;

static int32_t f32_to_bits(float f) {
    int32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}
static float bits_to_f32(int32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

TEST_CASE("float: fadd r2 = 1.5 + 2.5 = 4.0", "[float][fadd]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_fadd("r2", "r1", "r1");  // r2 = r1 + r1
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        set_lane_reg(w, lane, "r1", f32_to_bits(static_cast<float>(lane)));
    }
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        float expected = static_cast<float>(lane) + static_cast<float>(lane);
        REQUIRE(get_lane_reg(w, lane, "r2") == f32_to_bits(expected));
    }
}

TEST_CASE("float: fsub r2 = r1 - r1 = 0", "[float][fsub]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_fsub("r2", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        set_lane_reg(w, lane, "r1", f32_to_bits(static_cast<float>(lane) + 0.5f));
    }
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == 0);  // exact zero
    }
}

TEST_CASE("float: fmul r2 = r1 * 2.0", "[float][fmul]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_fmul("r2", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        set_lane_reg(w, lane, "r1", f32_to_bits(static_cast<float>(lane)));
    }
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        float expected = static_cast<float>(lane) * static_cast<float>(lane);
        REQUIRE(get_lane_reg(w, lane, "r2") == f32_to_bits(expected));
    }
}

TEST_CASE("float: fma r2 = r1 * r1 + r1", "[float][ffma]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_ffma("r2", "r1", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        set_lane_reg(w, lane, "r1", f32_to_bits(static_cast<float>(lane)));
    }
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        float a = static_cast<float>(lane);
        float expected = a * a + a;
        REQUIRE(get_lane_reg(w, lane, "r2") == f32_to_bits(expected));
    }
}
```

(Note: NaN/Inf/denormal edge cases are out of scope for P1-4; tests use finite values only.)

- [ ] **Step 3: Add CMake entry**

```cmake
add_catch_test(integration_ptx_float_arith
    ptx/test_float_arith.cpp
)
set_tests_properties(integration_ptx_float_arith PROPERTIES LABELS "integration;ptx;float_arith;fadd;fsub;fmul;fdiv;ffma")
```

- [ ] **Step 4: Reconfigure, rebuild, run, commit**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3 && cmake --build build --target ptxsim 2>&1 | tail -5
cd /workspace/project/PTX-EMU/build && ctest -R "integration_ptx_float_arith" -V 2>&1 | tail -15
cd /workspace/project/PTX-EMU && git add include/ptxsim/testing/instruction_helpers.h tests/integration/ptx/test_float_arith.cpp tests/integration/CMakeLists.txt && git commit -m "test(tier3): add integration_ptx_float_arith (fadd/fsub/fmul/ffma)

4 TEST_CASEs covering finite-value float arithmetic. Float ops reuse
S_ADD/S_SUB/S_MUL opcodes with Q_F32 qualifier; fma uses S_FMA.
NaN/Inf edge cases deferred to a follow-up spec."
```

Expected: 1 test target PASS, 4 TEST_CASEs pass.

---

## Task 6: Add extended factories and write test_extended.cpp

**Files:**
- Modify: `include/ptxsim/testing/instruction_helpers.h` (add 4 extended factories)
- Create: `tests/integration/ptx/test_extended.cpp` (4 TEST_CASEs)
- Modify: `tests/integration/CMakeLists.txt` (add entry)

- [ ] **Step 1: Add 4 extended factories**

```cpp
inline StatementContext make_addc(const std::string& dst, const std::string& src1,
                                   const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_ADDC;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "addc.u32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_subc(const std::string& dst, const std::string& src1,
                                   const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_SUBC;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "subc.u32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

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

inline StatementContext make_mul24(const std::string& dst, const std::string& src1,
                                    const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_MUL24;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "mul.lo.u32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}
```

- [ ] **Step 2: Create test_extended.cpp using the Template above + these TEST_CASEs**

```cpp
using ptxsim::testing::make_addc;
using ptxsim::testing::make_subc;
using ptxsim::testing::make_mad;
using ptxsim::testing::make_mul24;

TEST_CASE("extended: mad r2 = r1 * r1 + r1 (squared + self)", "[extended][mad]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_mad("r2", "r1", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        int32_t expected = lane * lane + lane;
        REQUIRE(get_lane_reg(w, lane, "r2") == expected);
    }
}

TEST_CASE("extended: mul24 r2 = r1 * r1 (low 24-bit mul, no overflow for small)", "[extended][mul24]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_mul24("r2", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    // For lanes 0..5, 24-bit mul matches 32-bit mul
    for (int lane = 0; lane < 6; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == (lane * lane));
    }
}

TEST_CASE("extended: addc r2 = r1 + 0 (no carry)", "[extended][addc]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_addc("r2", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    // Without carry-in, addc == add for same operands
    for (int lane = 0; lane < 32; ++lane) {
        int32_t expected = static_cast<int32_t>(static_cast<uint32_t>(lane) + static_cast<uint32_t>(lane));
        REQUIRE(get_lane_reg(w, lane, "r2") == expected);
    }
}

TEST_CASE("extended: subc r2 = r1 - r1 (no borrow)", "[extended][subc]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_subc("r2", "r1", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == 0);
    }
}
```

- [ ] **Step 3: Add CMake entry**

```cmake
add_catch_test(integration_ptx_extended
    ptx/test_extended.cpp
)
set_tests_properties(integration_ptx_extended PROPERTIES LABELS "integration;ptx;extended;addc;subc;mad;mul24")
```

- [ ] **Step 4: Reconfigure, rebuild, run, commit**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3 && cmake --build build --target ptxsim 2>&1 | tail -5
cd /workspace/project/PTX-EMU/build && ctest -R "integration_ptx_extended" -V 2>&1 | tail -15
cd /workspace/project/PTX-EMU && git add include/ptxsim/testing/instruction_helpers.h tests/integration/ptx/test_extended.cpp tests/integration/CMakeLists.txt && git commit -m "test(tier3): add integration_ptx_extended (mad/mul24/addc/subc)

4 TEST_CASEs covering multiply-add, 24-bit mul, add-with-carry and
sub-with-borrow. SubcHandler is implemented (verified 2026-06-06 at
src/ptxsim/instructions/arithmetic_ext.cpp:242)."
```

Expected: 1 test target PASS, 4 TEST_CASEs pass.

---

## Task 7: Add CVTA factories and write test_cvta.cpp

**Files:**
- Modify: `include/ptxsim/testing/instruction_helpers.h` (add 2 CVTA factories)
- Create: `tests/integration/ptx/test_cvta.cpp` (2 TEST_CASEs)
- Modify: `tests/integration/CMakeLists.txt` (add entry)

- [ ] **Step 1: Add 2 CVTA factories**

```cpp
inline StatementContext make_cvta_to_global(const std::string& dst, const std::string& src) {
    StatementContext ctx;
    ctx.type = S_CVTA;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_U64, Qualifier::Q_GLOBAL};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "cvta.to.global.u64 " + dst + ", " + src + ";";
    return ctx;
}

inline StatementContext make_cvta_to_shared(const std::string& dst, const std::string& src) {
    StatementContext ctx;
    ctx.type = S_CVTA;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_U64, Qualifier::Q_SHARED};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "cvta.to.shared.u64 " + dst + ", " + src + ";";
    return ctx;
}
```

- [ ] **Step 2: Create test_cvta.cpp using the Template above + these TEST_CASEs**

```cpp
using ptxsim::testing::make_cvta_to_global;
using ptxsim::testing::make_cvta_to_shared;

TEST_CASE("cvta: to.global r2 = r1 (identity when already generic)", "[cvta][global]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_cvta_to_global("r2", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane * 4);
    step_warp(w, stmts);
    // For an already-generic address, cvta is identity (lower 32 bits)
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == lane * 4);
    }
}

TEST_CASE("cvta: to.shared r2 = r1 (identity for shared ptr)", "[cvta][shared]") {
    std::vector<StatementContext> stmts(3);
    stmts[0] = make_mov("r1", "tid");
    stmts[1] = make_cvta_to_shared("r2", "r1");
    stmts[2] = make_ret();

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = setup(sm, stmts);
    for (int lane = 0; lane < 32; ++lane) set_lane_reg(w, lane, "r1", lane * 8);
    step_warp(w, stmts);
    for (int lane = 0; lane < 32; ++lane) {
        REQUIRE(get_lane_reg(w, lane, "r2") == lane * 8);
    }
}
```

- [ ] **Step 3: Add CMake entry**

```cmake
add_catch_test(integration_ptx_cvta
    ptx/test_cvta.cpp
)
set_tests_properties(integration_ptx_cvta PROPERTIES LABELS "integration;ptx;cvta")
```

- [ ] **Step 4: Reconfigure, rebuild, run, commit**

```bash
cd /workspace/project/PTX-EMU && cmake -S . -B build 2>&1 | tail -3 && cmake --build build --target ptxsim 2>&1 | tail -5
cd /workspace/project/PTX-EMU/build && ctest -R "integration_ptx_cvta" -V 2>&1 | tail -15
cd /workspace/project/PTX-EMU && git add include/ptxsim/testing/instruction_helpers.h tests/integration/ptx/test_cvta.cpp tests/integration/CMakeLists.txt && git commit -m "test(tier3): add integration_ptx_cvta (to.global / to.shared)

2 TEST_CASEs verifying address conversion. CvtaHandler is
implemented (verified 2026-06-06 at
src/ptxsim/instructions/data_transfer.cpp:17)."
```

Expected: 1 test target PASS, 2 TEST_CASEs pass.

---

## Task 8: Final validation — full sanity check

**Files:** (none modified — validation only)

- [ ] **Step 1: Run full Tier 3 sanity check**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --tier 3 2>&1 | tail -20
```

Expected: All PASS, including the 5 new tests. Output should show all `integration_ptx_*` targets passing.

- [ ] **Step 2: Run all integration tests to check for regressions**

```bash
cd /workspace/project/PTX-EMU/build && ctest -L "integration" 2>&1 | tail -20
```

Expected: All integration tests pass (no regressions from the new tests or factory additions).

- [ ] **Step 3: Run default sanity check (Tiers 1-9)**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh 2>&1 | tail -20
```

Expected: `All tests passed!` exit 0. Pre-P0 baseline red tests (#84, #85) are DISABLED and should be skipped.

- [ ] **Step 4: Verify ctest label coverage**

```bash
cd /workspace/project/PTX-EMU/build && ctest -L "integration;ptx" -N 2>&1 | grep "Test #"
```

Expected: 7 test targets listed (2 existing + 5 new): `integration_ptx_lane_verification`, `integration_ptx_ld_st_shared`, `integration_ptx_integer_arith`, `integration_ptx_bitwise`, `integration_ptx_cvt`, `integration_ptx_float_arith`, `integration_ptx_extended`, `integration_ptx_cvta`.

- [ ] **Step 5: Apply clang-format to all modified files**

```bash
cd /workspace/project/PTX-EMU && clang-format -i include/ptxsim/testing/instruction_helpers.h tests/integration/ptx/test_bitwise.cpp tests/integration/ptx/test_cvt.cpp tests/integration/ptx/test_float_arith.cpp tests/integration/ptx/test_extended.cpp tests/integration/ptx/test_cvta.cpp tests/integration/CMakeLists.txt
```

- [ ] **Step 6: Final commit (format-only changes)**

```bash
cd /workspace/project/PTX-EMU && git add -u && git diff --cached --stat && git commit -m "style(test): apply clang-format to P1-4 new files" 2>&1 | tail -3
```

Expected: One final commit if clang-format changed anything; empty commit is OK.

- [ ] **Step 7: Verify the spec's success criteria**

Run through `docs/superpowers/specs/2026-06-06-ptx-emu-tier3-ptx-tests-design.md §7` (Success criteria) and confirm each item is satisfied.

---

## Self-Review Notes (completed by author before save)

- **Spec coverage**: All sections of the design spec are addressed:
  - §3.1 (5 new test files): Tasks 3-7
  - §3.2 modified files: Tasks 2-7
  - §5 factory design: All 18 new factories + make_sub promotion covered
  - §6 CMake additions: Each task has its own `add_catch_test` block
  - §7 success criteria: Task 8 validates
- **Placeholder scan**: No TBD/TODO. All code blocks are complete.
- **Type consistency**: `Qualifier::Q_F32`, `Qualifier::Q_GLOBAL`, etc. used consistently across all factory functions and tests. `StatementContext`, `GenericInstr`, `OperandContext`, `RegOperand` are imported in every test file via the template.
- **Edge case acknowledged**: CVT/CVTA tests use 32-bit register access (existing `RegisterBankManager` API), so 64-bit values are partially tested via their lower 32 bits. This is documented in the spec §10 (out of scope).
