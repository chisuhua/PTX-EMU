// test_local_memory.cpp
// =============================================================================
// Integration test (类型二) — ld.local / st.local round-trip on the
// PTX-EMU simulator.
//
// Local memory is **per-thread** private storage. Each lane has its own
// backing array (allocated in cta_context.cpp:155). This test verifies
// that a per-lane write to local memory is visible when the same lane
// reads it back, and that the value is independent of other lanes'
// writes (no cross-lane aliasing).
//
// Instruction sequence (PC=0..5):
//   PC=0:  S_LOCAL .b32 arr[16]            ; per-thread 16-element array
//   PC=1:  mov.b32 r0, tid.x              ; r0 = lane_id
//   PC=2:  st.local.b32 [arr + r0], r0    ; local[lane_id] = lane_id
//   PC=3:  ld.local.b32 r1, [arr + r0]    ; r1 = local[lane_id]
//   PC=4:  ret
//
// Expected: r1[lane] == lane_id for every lane.
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
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::make_mov;
using ptxsim::testing::make_ret;
using ptxsim::testing::step_warp;

namespace {

void init_instruction_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

StatementContext make_local_decl(const std::string &name, int array_size) {
    StatementContext ctx;
    ctx.type = S_LOCAL;
    DeclarationInstr d;
    d.kind = DeclarationInstr::Kind::LOCAL;
    d.name = name;
    d.dataType = Qualifier::Q_B32;
    d.array_size = array_size;
    ctx.data = d;
    ctx.instructionText =
        ".local .b32 " + name + "[" + std::to_string(array_size) + "];";
    return ctx;
}

StatementContext make_st_local_addr(const std::string &base_sym,
                                    const std::string &offset_reg,
                                    const std::string &src_reg) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_LOCAL, Qualifier::Q_B32};
    AddrOperand addr;
    addr.space = AddrOperand::Space::LOCAL;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "st.local.b32 [" + base_sym + "+" + offset_reg + "], " + src_reg + ";";
    return ctx;
}

StatementContext make_ld_local_addr(const std::string &dst_reg,
                                    const std::string &base_sym,
                                    const std::string &offset_reg) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_LOCAL, Qualifier::Q_B32};
    AddrOperand addr;
    addr.space = AddrOperand::Space::LOCAL;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{RegOperand{dst_reg, -1}});
    instr.operands.push_back(OperandContext{addr});
    ctx.data = instr;
    ctx.instructionText =
        "ld.local.b32 " + dst_reg + ", [" + base_sym + "+" + offset_reg + "];";
    return ctx;
}

WarpContext *setup_block(SMContext &sm, std::vector<StatementContext> &stmts) {
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

uint32_t read_reg_u32(WarpContext *w, const std::string &reg, int lane) {
    auto rbm = w->get_register_bank_manager();
    void *p = rbm->get_register(reg, 0, lane);
    REQUIRE(p != nullptr);
    return *static_cast<uint32_t *>(p);
}

} // namespace

TEST_CASE("integration_local_memory_round_trip",
          "[integration][memory][local][ld_st]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(5);
    stmts.push_back(make_local_decl("arr", 16));            // PC=0
    stmts.push_back(make_mov("r0", "tid.x"));              // PC=1
    stmts.push_back(make_st_local_addr("arr", "r0", "r0")); // PC=2
    stmts.push_back(make_ld_local_addr("r1", "arr", "r0")); // PC=3
    stmts.push_back(make_ret());                            // PC=4

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    int ret_pc = -1;
    for (int step = 0; step < 32; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 4) { ret_pc = pc; break; }
    }
    REQUIRE(ret_pc == 4);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = read_reg_u32(w, "r1", lane);
        CHECK(v == static_cast<uint32_t>(lane));
    }
}
