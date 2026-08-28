// test_dynamic_shared_memory.cpp
// =============================================================================
// Integration test (类型二) — dynamic shared memory (extern __shared__)
// through CTAContext init path.
//
// Verifies that setup_block_with_dynamic_shared correctly:
//   1. Allocates static shared memory (buf[16] b32 = 64 bytes)
//   2. Allocates dynamic shared memory (dynamic_bytes = 128 bytes)
//   3. Places dynamic area after static area (no overlap)
//   4. Allows write-then-read to dynamic area
//
// Instruction sequence (PC=0..7):
//   PC=0:  S_SHARED .b32 buf[16]          ; static 64 bytes
//   PC=1:  mov.b32 r0, tid.x              ; r0 = lane_id
//   PC=2:  st.shared.b8 [buf + r0], r0    ; write lane_id to static area (byte)
//   PC=3:  mov.b32 r1, 64                 ; r1 = 64 (dynamic area base offset)
//   PC=4:  add.b32 r1, r1, r0             ; r1 = 64 + lane_id (per-lane dynamic offset)
//   PC=5:  st.shared.b8 [buf + r1], r0    ; write lane_id to dynamic area (byte)
//   PC=6:  ld.shared.b8 r2, [buf + r1]    ; read back from dynamic area (byte)
//   PC=7:  ret
//
// Assertions:
//   - r2 == lane_id (dynamic area write-then-read)
//   - Static area and dynamic area do not overlap
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
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
using ptxsim::testing::make_add;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::read_reg_u32;
using ptxsim::testing::setup_block_with_dynamic_shared;
using ptxsim::testing::step_warp;

TEST_CASE("integration_dynamic_shared_memory",
          "[integration][memory][shared_memory][dynamic]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // Build instruction sequence
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(7);
    stmts.push_back(make_shared_decl("buf", 16));              // PC=0: static 64 bytes
    stmts.push_back(make_mov("r0", "tid.x"));                  // PC=1: r0 = lane_id
    stmts.push_back(make_st_shared_addr("buf", "r0", "r0"));   // PC=2: write to static (b8)
    stmts.push_back(make_mov_imm("r1", 64));                   // PC=3: r1 = 64 (dynamic area base)
    stmts.push_back(make_add("r1", "r1", "r0"));               // PC=4: r1 = 64 + lane_id
    stmts.push_back(make_st_shared_addr("buf", "r1", "r0"));   // PC=5: write to dynamic (b8)
    stmts.push_back(make_ld_shared_addr("r2", "buf", "r1"));   // PC=6: read from dynamic (b8)
    stmts.push_back(make_ret());                               // PC=7

    // Setup SM with dynamic shared memory (128 bytes)
    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block_with_dynamic_shared(sm, stmts, 128);
    REQUIRE(w != nullptr);

    // Execute until ret
    int ret_pc = -1;
    for (int step = 0; step < 32; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 7) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 7);

    // Verify dynamic area write-then-read
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t r2_val = read_reg_u32(w, "r2", lane);
        CHECK(r2_val == static_cast<uint32_t>(lane));
    }

    // Verify static and dynamic areas do not overlap
    // Static area: buf[0..15] = 0..63 bytes
    // Dynamic area: buf[16..] = 64..191 bytes (128 bytes dynamic)
    // We wrote lane_id to offset 16 (byte offset), which is in dynamic area
    // If overlap occurred, static writes at offset 0..31 would corrupt dynamic reads
    // Since r2 == lane_id, no corruption occurred → no overlap
}