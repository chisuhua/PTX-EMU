// test_vec_ld_st_isolation.cpp
// =============================================================================
// Integration test (类型二) — Verify that a non-V2/V4 instruction with a
// VEC source operand (e.g. mov.b64 dst, {r1, r2}) does NOT corrupt a
// subsequent ld.shared.v4 / st.shared.v4 round-trip.
//
// BUG-VECOP-STALE: previously, ThreadContext::vecOp_phy_addrs was a
// std::queue<std::vector<void *>>. The VEC case in acquire_operand pushed
// a new entry for every VEC operand, and the LdHandler/StHandler V2/V4
// path did `front() + pop()`. mov.b64 with a VEC source has 2 elements
// (it packs two b32 sources into one b64 dest), so it pushed a 2-element
// vec — but never popped. The next ld.shared.v4 / st.shared.v4 would
// then pop THAT 2-element vec and try to iterate it for vec_size=4,
// reading vecAddrs[2]/[3] past the end of a 2-element array. The
// resulting null/garbage pointers then reached HardwareMemoryManager::access,
// which throws "Invalid memory access arguments".
//
// RED PHASE on unpatched code: the ld.shared.v4 at PC=2 reads junk into
// r3..r6, the st.shared.v4 at PC=3 writes junk back, and the loaded
// round-trip value at PC=4 mismatches the original input. Or, depending
// on memory layout, the access() throws and the test ends in a
// std::invalid_argument catch at the cudart layer.
//
// This test mirrors the scenario in bench/aligned-types/aligned-types.cu
// (the uint3_aligned / uint4_* subtests) but at the unit-test level so
// it runs in milliseconds and can be wired into CI without a CUDA
// toolchain.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"

#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::read_reg_u32;
using ptxsim::testing::setup_block;
using ptxsim::testing::step_warp;

static void write_reg_u32(WarpContext *w, const std::string &reg, int lane,
                          uint32_t value) {
    auto rbm = w->get_register_bank_manager();
    auto *p = rbm->get_register(reg, 0, lane);
    if (!p) {
        rbm->create_register(reg, 4);
        p = rbm->get_register(reg, 0, lane);
    }
    REQUIRE(p != nullptr);
    *static_cast<uint32_t *>(p) = value;
}

// Local VEC builder: a single VEC operand of N REG elements. The StHandler
// / LdHandler V2/V4 paths read op[1] (for ST) or op[0] (for LD) as a
// void** that points to the array of element addresses. For that to work
// the operand must be a VecOperand, NOT 4 separate REG operands (the
// default make_st_shared_addr_v4 helper in memory_test_utils.h pushes
// 4 separate REG operands, which is wrong for the VEC path).
static ptxemu::ir::OperandContext make_vec_reg_operand(
    const std::vector<std::string> &regs) {
    VecOperand v;
    for (const auto &r : regs) {
        v.elements.push_back(ptxemu::ir::OperandContext{RegOperand{r, -1}});
    }
    return ptxemu::ir::OperandContext{v};
}

static ptxemu::ir::StatementContext make_st_shared_v4_vec(
    const std::string &base_sym, const std::string &offset_reg,
    const std::vector<std::string> &srcs) {
    ptxemu::ir::StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {ptxemu::ir::Qualifier::Q_SHARED, ptxemu::ir::Qualifier::Q_B32,
                         ptxemu::ir::Qualifier::Q_V4};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<ptxemu::ir::OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(ptxemu::ir::OperandContext{addr});
    instr.operands.push_back(make_vec_reg_operand(srcs));
    ctx.data = instr;
    std::string text =
        "st.shared.v4.b32 [" + base_sym + "+" + offset_reg + "], {";
    for (size_t i = 0; i < srcs.size(); ++i) {
        if (i > 0) text += ",";
        text += srcs[i];
    }
    text += "};";
    ctx.instructionText = text;
    return ctx;
}

static ptxemu::ir::StatementContext make_ld_shared_v4_vec(
    const std::vector<std::string> &dsts, const std::string &base_sym,
    const std::string &offset_reg) {
    ptxemu::ir::StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {ptxemu::ir::Qualifier::Q_SHARED, ptxemu::ir::Qualifier::Q_B32,
                         ptxemu::ir::Qualifier::Q_V4};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<ptxemu::ir::OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(make_vec_reg_operand(dsts));
    instr.operands.push_back(ptxemu::ir::OperandContext{addr});
    ctx.data = instr;
    std::string text = "ld.shared.v4.b32 {";
    for (size_t i = 0; i < dsts.size(); ++i) {
        if (i > 0) text += ",";
        text += dsts[i];
    }
    text += "}, [" + base_sym + "+" + offset_reg + "];";
    ctx.instructionText = text;
    return ctx;
}

TEST_CASE(
    "BUG-VECOP-STALE: ld/st.shared.v4 round-trip works after the per-ThreadContext "
    "stack fix",
    "[integration][ptx][regression][BUG-VECOP-STALE]") {
    // Minimal V4 round-trip on shared memory. Exercises the per-ThreadContext
    // vecOp_phy_addrs stack path (case VEC in acquire_operand) and the
    // LdHandler/StHandler V4 path (which casts op[0]/op[1] to void** and
    // iterates vec_size entries).
    //
    // On the broken (FIFO) code: the V4 ST pushes a 4-element VEC and
    // pops it, but if any preceding instruction also pushed a VEC that
    // wasn't popped (e.g. mov.b64 with a vector source), the FIFO would
    // hand that older entry to the handler. The handler then dereferences
    // past the older entry's tail, hitting null/garbage and crashing
    // inside HardwareMemoryManager::access.
    //
    // NOTE: a full reproduction of the original BUG-VECOP-STALE scenario
    // (mov.b64 %rd8, {%r8, %r9} followed by ld/st.v4) is blocked by a
    // SEPARATE bug in the mov handler — it memcpy()'s the VEC pointer
    // value rather than packing the source registers. See KNOWN_ISSUES.
    // This test instead verifies that the V4 path itself works under the
    // fixed per-ThreadContext stack, which is the precondition for fixing
    // the original BUG-VECOP-STALE.

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(7);
    stmts.push_back(make_shared_decl("buf", 64, ptxemu::ir::Qualifier::Q_B32)); // PC=0
    stmts.push_back(make_mov_imm("r1", 0x11111111));                 // PC=1
    stmts.push_back(make_mov_imm("r2", 0x22222222));                 // PC=2
    stmts.push_back(make_mov_imm("r3", 0x33333333));                 // PC=3
    stmts.push_back(make_mov_imm("r4", 0x44444444));                 // PC=4
    stmts.push_back(make_mov_imm("r0", 0));                          // PC=5
    stmts.push_back(make_st_shared_v4_vec("buf", "r0",
                                         {"r1", "r2", "r3", "r4"}));  // PC=6
    stmts.push_back(make_ld_shared_v4_vec({"r5", "r6", "r7", "r8"},
                                         "buf", "r0"));              // PC=7
    stmts.push_back(make_ret());                                      // PC=8

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    int step_count = 0;
    int pc = -1;
    while (step_count < 32 && (pc = step_warp(w, stmts)) >= 0) {
        step_count++;
        if (pc == 8) break;
    }
    REQUIRE(pc == 8);

    for (int lane = 0; lane < 32; ++lane) {
        INFO("lane " << lane);
        CHECK(read_reg_u32(w, "r5", lane) == 0x11111111u);
        CHECK(read_reg_u32(w, "r6", lane) == 0x22222222u);
        CHECK(read_reg_u32(w, "r7", lane) == 0x33333333u);
        CHECK(read_reg_u32(w, "r8", lane) == 0x44444444u);
    }
}

TEST_CASE(
    "BUG-VECOP-STALE: a stale VEC push from a prior VEC acquire does NOT "
    "corrupt a subsequent V4 LD/ST round-trip",
    "[integration][ptx][regression][BUG-VECOP-STALE]") {
    // Directly reproduces the FIFO bug class. We do NOT use mov.b64 to
    // push a stale VEC — that path is broken by a SEPARATE mov handler
    // bug. Instead we directly call acquire_operand to push a 2-element
    // VEC entry, mimicking what a VEC-source mov would do. Then we
    // execute a V4 ST + V4 LD round-trip via the pipeline.
    //
    // On the broken (FIFO) code: the manual push leaves a 2-element
    // entry in the queue. The V4 ST pushes a 4-element VEC and the
    // handler pops front → gets the OLD 2-element entry. The handler
    // then dereferences vecAddrs[2] and vecAddrs[3] past the 2-element
    // entry's tail, hitting null/garbage and crashing inside
    // HardwareMemoryManager::access with "Invalid memory access arguments".
    //
    // On the fixed (per-ThreadContext stack) code: the handler reads the
    // V4 destination's own buffer from op[1] (for ST) / op[0] (for LD),
    // never touching the stale manual push. Round-trip succeeds.
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(6);
    stmts.push_back(make_shared_decl("buf", 64, ptxemu::ir::Qualifier::Q_B32)); // PC=0
    stmts.push_back(make_mov_imm("r1", 0xAAAAAAAA));                 // PC=1
    stmts.push_back(make_mov_imm("r2", 0xBBBBBBBB));                 // PC=2
    stmts.push_back(make_mov_imm("r3", 0xCCCCCCCC));                 // PC=3
    stmts.push_back(make_mov_imm("r4", 0xDDDDDDDD));                 // PC=4
    stmts.push_back(make_mov_imm("r0", 0));                          // PC=5
    stmts.push_back(make_st_shared_v4_vec("buf", "r0",
                                         {"r1", "r2", "r3", "r4"}));  // PC=6
    stmts.push_back(make_ld_shared_v4_vec({"r5", "r6", "r7", "r8"},
                                         "buf", "r0"));              // PC=7
    stmts.push_back(make_ret());                                      // PC=8

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    ThreadContext *tc = w->get_thread(0);
    auto rbm = w->get_register_bank_manager();
    // setup_block's prealloc covers r0..r8 from the stmts; no extras needed.

    // Mimic a VEC-source mov.b64 dst, {r1, r2} — push a 2-element VEC
    // entry that the OLD code's FIFO would hand to the next V4 handler.
    std::vector<ptxemu::ir::Qualifier> no_qual;
    ptxemu::ir::OperandContext stale_vec = make_vec_reg_operand({"r1", "r2"});
    void *stale_p = tc->acquire_operand(stale_vec, no_qual);
    REQUIRE(stale_p != nullptr);

    int step_count = 0;
    int pc = -1;
    while (step_count < 32 && (pc = step_warp(w, stmts)) >= 0) {
        step_count++;
        if (pc == 8) break;
    }
    REQUIRE(pc == 8);

    // Round-trip assertion: the V4 ST wrote r1..r4 to buf[0..3], and the
    // V4 LD read them back into r5..r8.
    for (int lane = 0; lane < 32; ++lane) {
        INFO("lane " << lane);
        CHECK(read_reg_u32(w, "r5", lane) == 0xAAAAAAAAu);
        CHECK(read_reg_u32(w, "r6", lane) == 0xBBBBBBBBu);
        CHECK(read_reg_u32(w, "r7", lane) == 0xCCCCCCCCu);
        CHECK(read_reg_u32(w, "r8", lane) == 0xDDDDDDDDu);
    }
}
