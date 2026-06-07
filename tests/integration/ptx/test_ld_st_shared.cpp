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
#include "ptxsim/testing/memory_test_utils.h"
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

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::read_reg_u32;
using ptxsim::testing::setup_block;
using ptxsim::testing::step_warp;

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

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = read_reg_u32(w, "r2", lane);
        INFO("lane " << lane << " r2 = 0x" << std::hex << v);
        CHECK(v == static_cast<uint32_t>(lane));
    }
}
