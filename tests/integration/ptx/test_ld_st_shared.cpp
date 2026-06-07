// test_ld_st_shared.cpp
// =============================================================================
// Integration test (类型二) — ld.shared / st.shared round-trip on the
// PTX-EMU simulator (NOT real GPU).
//
// Instruction sequence (PC=0..4):
//   PC=0:    S_SHARED .b32 buf[32]  (declaration, consumed by CTAContext::init)
//   PC=1:    mov.b32 %r1, tid.x     ; r1[lane] = lane_id (special register read)
//   PC=2:    st.shared.b32 [buf + r1], r1  ; buf[lane_id] = lane_id
//   PC=3:    ld.shared.b32 r2, [buf + r1]  ; r2 = buf[lane_id]
//   PC=4:    ret
//
// Expected: every lane reads r2 == lane_id (round-trip self-consistency).
// This is the minimal test that exercises the ld/st path WITHOUT divergence
// or barrier, so it isolates the ld/st handler from the bra/bar handlers
// (which have pre-existing bugs — see KNOWN_ISSUES.md).
//
// This test is added per the A2 plan (rewrite tests/unit/ptx/ to drive
// the simulator instead of the real GPU).
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
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
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_ret;
using ptxsim::testing::step_warp;

namespace {

// S_SHARED declaration (e.g., `.shared .b32 buf[32];`)
StatementContext make_shared_decl(const std::string &name, int array_size) {
    StatementContext ctx;
    ctx.type = S_SHARED;
    DeclarationInstr d;
    d.kind = DeclarationInstr::Kind::SHARED;
    d.name = name;
    d.dataType = Qualifier::Q_B32;
    d.array_size = array_size;
    ctx.data = d;
    ctx.instructionText =
        ".shared .b32 " + name + "[" + std::to_string(array_size) + "];";
    return ctx;
}

// st.shared.b8 [baseSym + offset_reg], src_reg  → AddrOperand
// Using b8 (1-byte writes) so 32 lanes writing at offset=lane_id do NOT
// overlap. A b32 write per lane (4 bytes) at offset=lane_id would cause
// inter-lane overlap because lane N writes buf[N..N+3] and lane N+1 writes
// buf[N+1..N+4], so lane N's bytes 1..3 get clobbered by lane N+1.
StatementContext make_st_shared_addr(const std::string &base_sym,
                                     const std::string &offset_reg,
                                     const std::string &src_reg) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B8};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "st.shared.b8 [" + base_sym + "+" + offset_reg + "], " + src_reg + ";";
    return ctx;
}

// ld.shared.b8 dst_reg, [baseSym + offset_reg]  → AddrOperand
StatementContext make_ld_shared_addr(const std::string &dst_reg,
                                     const std::string &base_sym,
                                     const std::string &offset_reg) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B8};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{RegOperand{dst_reg, -1}});
    instr.operands.push_back(OperandContext{addr});
    ctx.data = instr;
    ctx.instructionText =
        "ld.shared.b8 " + dst_reg + ", [" + base_sym + "+" + offset_reg + "];";
    return ctx;
}

void init_instruction_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

WarpContext *setup_block(SMContext &sm,
                         std::vector<StatementContext> &stmts) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{32, 1, 1};
    Dim3 bi{0, 0, 0};
    std::map<std::string, int> l2pc;
    std::map<std::string, Symtable *> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    return sm.get_warp(0);
}

} // namespace

TEST_CASE("integration_ld_st_shared_round_trip",
          "[integration][ptx][shared][ld_st]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(5);
    stmts.push_back(make_shared_decl("buf", 32));        // PC=0
    stmts.push_back(make_mov("r1", "tid.x"));            // PC=1
    stmts.push_back(make_st_shared_addr("buf", "r1", "r1")); // PC=2
    stmts.push_back(make_ld_shared_addr("r2", "buf", "r1")); // PC=3
    stmts.push_back(make_ret());                         // PC=4

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    int ret_pc = -1;
    for (int step = 0; step < 32; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 4) { ret_pc = pc; break; }
    }
    REQUIRE(ret_pc == 4);

    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm != nullptr);

    for (int lane = 0; lane < 32; ++lane) {
        void *p = rbm->get_register("r2", 0, lane);
        REQUIRE(p != nullptr);
        uint32_t v = *static_cast<uint32_t *>(p);
        INFO("lane " << lane << " r2 = 0x" << std::hex << v);
        CHECK(v == static_cast<uint32_t>(lane));
    }
}
