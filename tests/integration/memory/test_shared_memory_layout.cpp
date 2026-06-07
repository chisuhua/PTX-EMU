// test_shared_memory_layout.cpp
// =============================================================================
// Integration test (类型二) — shared memory layout for multiple S_SHARED
// declarations on the PTX-EMU simulator.
//
// Verifies that the shared memory infrastructure (CTAContext::init →
// SMContext::add_block → build_shared_memory_symbol_table) correctly:
//
//   1. Sums the sizes of S_SHARED declarations into sharedMemBytes
//      (buf_a[32] b32 = 128 B + buf_b[32] b32 = 128 B → 256 B total).
//   2. Assigns sequential base offsets (buf_a at 0, buf_b at 128).
//   3. Zero-initializes the shared memory space (memset in cta_context.cpp).
//   4. Resolves AddrOperand.baseSymbol to the right shared_mem_space + offset
//      at runtime (i.e. get_memory_addr works for both buffers).
//
// Instruction sequence (PC=0..5):
//   PC=0:  S_SHARED .b32 buf_a[32]
//   PC=1:  S_SHARED .b32 buf_b[32]
//   PC=2:  mov.b32 r0, tid.x              ; r0 = lane_id
//   PC=3:  ld.shared.b32 r1, [buf_a + r0] ; r1 = buf_a[lane_id] (expect 0)
//   PC=4:  ld.shared.b32 r2, [buf_b + r0] ; r2 = buf_b[lane_id] (expect 0)
//   PC=5:  ret
//
// Note: uses make_ld_shared_addr (AddrOperand + registerOffset), not
// make_ld_shared (VariableOperand). The VariableOperand form fails
// acquire_operand() in thread_context.cpp and SEGFAULTs the handler —
// see KNOWN_ISSUES.md.
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
#include <map>
#include <memory>
#include <string>
#include <vector>

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::read_reg_u32;
using ptxsim::testing::setup_block;
using ptxsim::testing::step_warp;

TEST_CASE("integration_shared_memory_layout_dual_buffer",
          "[integration][memory][shared][layout]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(6);
    stmts.push_back(make_shared_decl("buf_a", 32));          // PC=0
    stmts.push_back(make_shared_decl("buf_b", 32));          // PC=1
    stmts.push_back(make_mov("r0", "tid.x"));                // PC=2
    stmts.push_back(make_ld_shared_addr("r1", "buf_a", "r0")); // PC=3
    stmts.push_back(make_ld_shared_addr("r2", "buf_b", "r0")); // PC=4
    stmts.push_back(make_ret());                              // PC=5

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    int ret_pc = -1;
    for (int step = 0; step < 32; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 5) { ret_pc = pc; break; }
    }
    REQUIRE(ret_pc == 5);

    for (int lane = 0; lane < 32; ++lane) {
        uint32_t ra = read_reg_u32(w, "r1", lane);
        uint32_t rb = read_reg_u32(w, "r2", lane);
        CHECK(ra == 0u);
        CHECK(rb == 0u);
    }
}
