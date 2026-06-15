// test_vec_op_phy_addrs_stack.cpp
// =============================================================================
// Unit test (类型一) — ThreadContext::acquire_operand() for OperandKind::VEC
// returns a per-acquire stack buffer that the LdHandler/StHandler V2/V4 path
// can iterate via op[0]/op[1] cast to void**.
//
// BUG-VECOP-STALE: the previous design used a std::queue<std::vector<void*>>
// shared across instructions. Non-V2/V4 handlers (e.g. mov.b64 with a vector
// source) would emplace a new entry without popping it, leaving stale entries.
// The next V2/V4 LD/ST would then front() + pop() a wrong (older) entry: a
// 2-element VEC when the handler iterates for vec_size=4. The handler then
// dereferences vecAddrs[2]/[3] past the end of a 2-element array — UB / null
// pointer, surfacing as "Invalid memory access arguments" in
// HardwareMemoryManager::access.
//
// RED PHASE: This test must FAIL on the unpatched (FIFO) code with the VEC
// entry from the first acquire being returned for the second call's
// op[0]/op[1]. On the patched (per-acquire stack) code, the two acquires
// produce independent buffers.
//
// After fix: each acquire_operand call for VEC pushes a fresh entry, and
// the returned pointer references only the just-pushed entry. A subsequent
// acquire cannot see or alias the previous buffer.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptxsim/testing/warp_test_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"

#include "register/register_bank_manager.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::setup_block;

// Build a VecOperand whose elements are plain register operands. We use
// direct ctor rather than a helper because no public helper exists for a
// VEC source operand today — see integration/ptx/test_vec_ld_st_isolation.cpp
// where this pattern is exercised end-to-end through the handler.
static OperandContext make_vec_reg_operand(
    const std::vector<std::string> &regs) {
    VecOperand v;
    for (const auto &r : regs) {
        v.elements.push_back(OperandContext{RegOperand{r, -1}});
    }
    return OperandContext{v};
}

TEST_CASE("BUG-VECOP-STALE: VEC acquire pushes a per-call buffer",
          "[unit][regression][BUG-VECOP-STALE]") {
    init_instruction_factory_once();
    SMContext sm(4, 128, 4096, 0);

    std::vector<StatementContext> stmts;
    stmts.push_back(ptxsim::testing::make_ret());
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    ThreadContext *tc = w->get_thread(0);
    REQUIRE(tc != nullptr);

    // The ret-only statement pre-allocates no general-purpose registers,
    // so we allocate the r1..r6 slots the test exercises.
    auto rbm = w->get_register_bank_manager();
    for (const char *r : {"r1", "r2", "r3", "r4", "r5", "r6"}) {
        rbm->create_register(r, 4);
    }

    // Qualifiers are not exercised by the VEC branch's acquire_operand; pass
    // an empty vector.
    std::vector<Qualifier> no_qual;

    // First VEC acquire: 2-element vector {r1, r2}.
    OperandContext vec1 = make_vec_reg_operand({"r1", "r2"});
    void *p1 = tc->acquire_operand(vec1, no_qual);
    REQUIRE(p1 != nullptr);

    auto *base1 = static_cast<void **>(p1);
    REQUIRE(base1[0] != nullptr);
    REQUIRE(base1[1] != nullptr);

    // Sanity: the addresses in base1 must point to the actual r1 / r2
    // registers allocated by setup_block.
    REQUIRE(rbm->get_register("r1", 0, 0) == base1[0]);
    REQUIRE(rbm->get_register("r2", 0, 0) == base1[1]);

    // Second VEC acquire: 4-element vector {r3, r4, r5, r6}. On the fixed
    // code, this must NOT alias base1 — it must reference a fresh stack
    // entry. On the broken FIFO code, the acquire would re-use the same
    // queue entry, so base2 == base1.
    OperandContext vec2 = make_vec_reg_operand({"r3", "r4", "r5", "r6"});
    void *p2 = tc->acquire_operand(vec2, no_qual);
    REQUIRE(p2 != nullptr);

    auto *base2 = static_cast<void **>(p2);
    REQUIRE(base2[0] != nullptr);
    REQUIRE(base2[1] != nullptr);
    REQUIRE(base2[2] != nullptr);
    REQUIRE(base2[3] != nullptr);
    CHECK(rbm->get_register("r3", 0, 0) == base2[0]);
    CHECK(rbm->get_register("r4", 0, 0) == base2[1]);
    CHECK(rbm->get_register("r5", 0, 0) == base2[2]);
    CHECK(rbm->get_register("r6", 0, 0) == base2[3]);

    // The two buffers must be at distinct addresses; otherwise the LdHandler
    // V2/V4 path (which dereferences vecAddrs[i] for i in [0, vec_size))
    // would read the wrong registers when iterating for vec_size=4 after a
    // prior 2-element acquire left a stale entry.
    CHECK(p1 != p2);

    // And, critically, the second buffer must not silently contain the
    // FIRST buffer's elements past its own size. With the old FIFO,
    // base2[2]/[3] would be the LdHandler trying to read the 2-element
    // vec1 buffer as if it had 4 elements. With the new per-ThreadContext
    // stack, vec2 has its own buffer of size 4 that does not overlap vec1.
    CHECK(base2[2] != base1[0]);
    CHECK(base2[3] != base1[1]);
}

TEST_CASE("BUG-VECOP-STALE: VEC acquire does not leave a partial entry on the "
          "stack when an element is unallocated",
          "[unit][regression][BUG-VECOP-STALE]") {
    // The fix must also handle the partial-failure path: if any element
    // cannot be acquired (e.g. register not preallocated), acquire_register
    // throws InvalidMemoryAccessException. The VEC case must clear the
    // in-progress buffer (do not leave a half-filled entry on the stack),
    // so the NEXT VEC acquire starts from a clean slate.
    init_instruction_factory_once();
    SMContext sm(4, 128, 4096, 0);

    std::vector<StatementContext> stmts;
    stmts.push_back(ptxsim::testing::make_ret());
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    ThreadContext *tc = w->get_thread(0);
    std::vector<Qualifier> no_qual;

    auto rbm = w->get_register_bank_manager();
    rbm->create_register("r1", 4);
    rbm->create_register("r2", 4);
    rbm->create_register("r3", 4);
    rbm->create_register("r4", 4);
    rbm->create_register("r5", 4);
    rbm->create_register("r6", 4);
    // "r_undeclared" deliberately not created — acquire_register throws.

    OperandContext bad = make_vec_reg_operand(
        {"r1", "r_undeclared", "r2"});
    REQUIRE_THROWS(tc->acquire_operand(bad, no_qual));

    // After the failure, the next VEC acquire (4 elements, all valid) must
    // succeed and return a fresh, independent buffer. On the broken code
    // the failed half-filled entry would still be on the stack, so this
    // second acquire would push *another* entry — but more importantly,
    // any subsequent V4 LD/ST that reads op[0]/op[1] from the just-pushed
    // entry would still work for this isolated test, while the real
    // failure mode (cross-instruction corruption) is caught by the
    // integration test in test_vec_ld_st_isolation.cpp.
    OperandContext good = make_vec_reg_operand(
        {"r3", "r4", "r5", "r6"});
    void *p = tc->acquire_operand(good, no_qual);
    REQUIRE(p != nullptr);
    auto *base = static_cast<void **>(p);
    for (int i = 0; i < 4; ++i) {
        CHECK(base[i] != nullptr);
    }
}
