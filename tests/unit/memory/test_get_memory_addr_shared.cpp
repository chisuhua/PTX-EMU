// test_get_memory_addr_shared.cpp
// =============================================================================
// Unit regression test for BUG-NAME2SHARE: get_memory_addr SHARED REGISTER path
// doesn't consult name2Share for baseSymbol offset (thread_context.cpp:472-479).
//
// RED PHASE: This test must FAIL on unpatched code.
//
// Bug description:
//   In ThreadContext::get_memory_addr (thread_context.cpp:472-479):
//     if (QvecHasQ(qualifiers, Qualifier::Q_SHARED)) {
//         if (shared_mem_space != nullptr) {
//             ret = (void *)((uint64_t)shared_mem_space + reg_value);
//         }
//     }
//   The code adds reg_value to shared_mem_space, but doesn't add the baseSymbol
//   offset from name2Share. For example, with st.shared.b32 [buf + r1], r0:
//   - buf is at offset 100 in shared memory (from name2Share)
//   - r1 contains 4
//   - Correct address: shared_mem_space + 100 + 4
//   - Bug computes: shared_mem_space + 4 (missing buf offset!)
//
// Expected behavior after fix:
//   get_memory_addr should look up fa.baseSymbol in name2Share and add that
//   offset to the register offset before computing the final address.
//
// Test strategy:
//   1. Declare two shared memory arrays: buf0[32] and buf1[32]
//   2. buf0 should be at offset 0, buf1 at offset 128 (after buf0)
//   3. Write to buf1[0] via st.shared.b32 [buf1 + r0], r1 where r0=0
//   4. Load back from buf1[0] and verify the value matches
//   5. If bug exists, the write goes to buf0[0] instead of buf1[0]
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
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::read_reg_u32;
using ptxsim::testing::setup_block;
using ptxsim::testing::step_warp;

// Helper to write a 32-bit value to a register for a specific lane
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

TEST_CASE("BUG-NAME2SHARE: get_memory_addr adds baseSymbol offset for shared REGISTER path",
          "[unit][regression][BUG-NAME2SHARE]") {
    // RED PHASE: This test must FAIL on unpatched code.
    // Bug: get_memory_addr REGISTER path doesn't consult name2Share for baseSymbol offset.
    // For st.shared.b32 [buf1 + r0], r1, the address should be:
    //   shared_mem_space + buf1_offset + r0_value
    // But the bug computes:
    //   shared_mem_space + r0_value (missing buf1_offset!)

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // Create instruction sequence with TWO shared arrays:
    // PC=0: .shared .b32 buf0[32]  (at offset 0)
    // PC=1: .shared .b32 buf1[32]  (at offset 128, after buf0)
    // PC=2: mov.b32 r0, 0          (offset within buf1)
    // PC=3: mov.b32 r1, 0xAAAAAAAA (value to write)
    // PC=4: st.shared.b32 [buf1 + r0], r1  (write to buf1[0])
    // PC=5: mov.b32 r2, 0xBBBBBBBB (different value for buf0)
    // PC=6: st.shared.b32 [buf0 + r0], r2  (write to buf0[0])
    // PC=7: ld.shared.b32 r3, [buf1 + r0]  (load from buf1[0])
    // PC=8: ld.shared.b32 r4, [buf0 + r0]  (load from buf0[0])
    // PC=9: ret

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(10);
    stmts.push_back(make_shared_decl("buf0", 32, ptxemu::ir::Qualifier::Q_B32)); // PC=0
    stmts.push_back(make_shared_decl("buf1", 32, ptxemu::ir::Qualifier::Q_B32)); // PC=1
    stmts.push_back(make_mov_imm("r0", 0));                         // PC=2: offset=0
    stmts.push_back(make_mov_imm("r1", 0xAAAAAAAA));                // PC=3: value for buf1
    stmts.push_back(make_st_shared_addr("buf1", "r0", "r1", ptxemu::ir::Qualifier::Q_B32)); // PC=4
    stmts.push_back(make_mov_imm("r2", 0xBBBBBBBB));                // PC=5: value for buf0
    stmts.push_back(make_st_shared_addr("buf0", "r0", "r2", ptxemu::ir::Qualifier::Q_B32)); // PC=6
    stmts.push_back(make_ld_shared_addr("r3", "buf1", "r0", ptxemu::ir::Qualifier::Q_B32)); // PC=7
    stmts.push_back(make_ld_shared_addr("r4", "buf0", "r0", ptxemu::ir::Qualifier::Q_B32)); // PC=8
    stmts.push_back(make_ret());                                     // PC=9

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    // Execute the sequence
    int step_count = 0;
    int pc = -1;
    while ((pc = step_warp(w, stmts)) >= 0 && step_count < 30) {
        step_count++;
        if (pc == 9) break; // reached ret
    }

    REQUIRE(step_count > 0);

    // BUG CHECK: If the bug exists, buf1[0] and buf0[0] would have the same value
    // because the st.shared.b32 [buf1 + r0] would write to offset 0 instead of
    // buf1's actual offset (128).
    //
    // Expected (fixed): r3 = 0xAAAAAAAA (buf1[0]), r4 = 0xBBBBBBBB (buf0[0])
    // Bug behavior: r3 = 0xBBBBBBBB (buf0[0] overwrote buf1[0] due to missing offset)

    uint32_t buf1_val = read_reg_u32(w, "r3", 0);
    uint32_t buf0_val = read_reg_u32(w, "r4", 0);

    // Primary assertion: buf1[0] should have 0xAAAAAAAA, not 0xBBBBBBBB
    CHECK(buf1_val == 0xAAAAAAAA);

    // Secondary assertion: buf0[0] should have 0xBBBBBBBB
    CHECK(buf0_val == 0xBBBBBBBB);

    // If both are 0xBBBBBBBB, the bug caused buf0 write to overwrite buf1
    INFO("buf1_val = 0x" << std::hex << buf1_val << ", buf0_val = 0x" << buf0_val);
}

TEST_CASE("BUG-NAME2SHARE: get_memory_addr respects baseSymbol for non-zero offset",
          "[unit][regression][BUG-NAME2SHARE]") {
    // RED PHASE: This test must FAIL on unpatched code.
    // Extended test: verify that baseSymbol offset is added even when register offset is non-zero.

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // Create instruction sequence:
    // PC=0: .shared .b32 buf0[32]
    // PC=1: .shared .b32 buf1[32]
    // PC=2: mov.b32 r0, 4          (offset=4, i.e., buf1[1])
    // PC=3: mov.b32 r1, 0xCCCCCCCC
    // PC=4: st.shared.b32 [buf1 + r0], r1  (write to buf1[1])
    // PC=5: mov.b32 r2, 0xDDDDDDDD
    // PC=6: st.shared.b32 [buf0 + r0], r2  (write to buf0[1])
    // PC=7: ld.shared.b32 r3, [buf1 + r0]  (load from buf1[1])
    // PC=8: ld.shared.b32 r4, [buf0 + r0]  (load from buf0[1])
    // PC=9: ret

    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.reserve(10);
    stmts.push_back(make_shared_decl("buf0", 32, ptxemu::ir::Qualifier::Q_B32)); // PC=0
    stmts.push_back(make_shared_decl("buf1", 32, ptxemu::ir::Qualifier::Q_B32)); // PC=1
    stmts.push_back(make_mov_imm("r0", 4));                         // PC=2: offset=4
    stmts.push_back(make_mov_imm("r1", 0xCCCCCCCC));                // PC=3
    stmts.push_back(make_st_shared_addr("buf1", "r0", "r1", ptxemu::ir::Qualifier::Q_B32)); // PC=4
    stmts.push_back(make_mov_imm("r2", 0xDDDDDDDD));                // PC=5
    stmts.push_back(make_st_shared_addr("buf0", "r0", "r2", ptxemu::ir::Qualifier::Q_B32)); // PC=6
    stmts.push_back(make_ld_shared_addr("r3", "buf1", "r0", ptxemu::ir::Qualifier::Q_B32)); // PC=7
    stmts.push_back(make_ld_shared_addr("r4", "buf0", "r0", ptxemu::ir::Qualifier::Q_B32)); // PC=8
    stmts.push_back(make_ret());                                     // PC=9

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    // Execute
    int step_count = 0;
    int pc = -1;
    while ((pc = step_warp(w, stmts)) >= 0 && step_count < 30) {
        step_count++;
        if (pc == 9) break;
    }

    REQUIRE(step_count > 0);

    uint32_t buf1_val = read_reg_u32(w, "r3", 0);
    uint32_t buf0_val = read_reg_u32(w, "r4", 0);

    // Expected: buf1[1] = 0xCCCCCCCC, buf0[1] = 0xDDDDDDDD
    CHECK(buf1_val == 0xCCCCCCCC);
    CHECK(buf0_val == 0xDDDDDDDD);

    INFO("buf1[1] = 0x" << std::hex << buf1_val << ", buf0[1] = 0x" << buf0_val);
}