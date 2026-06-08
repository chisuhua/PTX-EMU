// test_st_handler_data_size.cpp
// =============================================================================
// Unit regression test for BUG-ST-OVERREAD: StHandler reads uint64_t from
// potentially 4-byte register (memory.cpp:66).
//
// RED PHASE: This test must FAIL on unpatched code.
//
// Bug description:
//   In StHandler::processOperation (memory.cpp:66):
//     uint64_t src_val = *(uint64_t*)src;
//   For st.shared.b32, data_size=4, but the code reads 8 bytes from a 4-byte
//   register. This is undefined behavior - reading past the allocated register
//   memory may read garbage or cause memory corruption.
//
// Expected behavior after fix:
//   StHandler should read only data_size bytes (4 bytes for b32, 8 for b64).
//   The test verifies that st.shared.b32 writes exactly 4 bytes and the
//   read-back value matches the original 4-byte value.
//
// Test strategy:
//   1. Write a known 4-byte value to a register
//   2. Store that register to shared memory via st.shared.b32
//   3. Load back via ld.shared.b32
//   4. Verify the loaded value matches the original (no corruption from 8-byte read)
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

// Helper to read a 64-bit value from a register (to detect overflow)
static uint64_t read_reg_u64(WarpContext *w, const std::string &reg, int lane) {
    auto rbm = w->get_register_bank_manager();
    auto *p = rbm->get_register(reg, 0, lane);
    REQUIRE(p != nullptr);
    return *static_cast<uint64_t *>(p);
}

TEST_CASE("BUG-ST-OVERREAD: StHandler respects declared data_size for b32 write",
          "[unit][regression][BUG-ST-OVERREAD]") {
    // RED PHASE: This test must FAIL on unpatched code.
    // Bug: StHandler::processOperation reads uint64_t src_val = *(uint64_t*)src;
    // For st.shared.b32, data_size=4, but the code reads 8 bytes from a 4-byte
    // register. This is undefined behavior.

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    // Create instruction sequence:
    // PC=0: .shared .b32 buf[32]
    // PC=1: mov.b32 r0, 0x12345678  (write known 32-bit value)
    // PC=2: st.shared.b32 [buf + r1], r0  (store 4 bytes)
    // PC=3: ld.shared.b32 r2, [buf + r1]  (load back)
    // PC=4: ret

    std::vector<StatementContext> stmts;
    stmts.reserve(5);
    stmts.push_back(make_shared_decl("buf", 32, Qualifier::Q_B32)); // PC=0
    stmts.push_back(make_mov_imm("r0", 0x12345678));                 // PC=1
    stmts.push_back(make_mov_imm("r1", 0));                         // PC=2: offset=0
    // Use b32 qualifier for st - this is where the bug manifests
    stmts.push_back(make_st_shared_addr("buf", "r1", "r0", Qualifier::Q_B32)); // PC=3
    stmts.push_back(make_ld_shared_addr("r2", "buf", "r1", Qualifier::Q_B32)); // PC=4
    stmts.push_back(make_ret());                                     // PC=5

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    // Execute the sequence
    int step_count = 0;
    int pc = -1;
    while ((pc = step_warp(w, stmts)) >= 0 && step_count < 20) {
        step_count++;
        if (pc == 5) break; // reached ret
    }

    REQUIRE(step_count > 0);

    // Verify: r2 should contain exactly 0x12345678 (value from mov_imm at PC=1)
    // The StHandler fix ensures that only 4 bytes are read from the 4-byte register,
    // avoiding undefined behavior from reading 8 bytes.
    uint32_t loaded = read_reg_u32(w, "r2", 0);
    CHECK(loaded == 0x12345678);

    // Additional check: verify that adjacent memory wasn't corrupted
    // If the bug writes 8 bytes instead of 4, buf[1] would be corrupted
    // We can check by loading buf[4] (offset=4, which is buf[1] for b32)
    // This should be 0 (uninitialized) if the bug is fixed
    // NOTE: This check requires additional infrastructure to verify
    // For now, the primary assertion is that the round-trip value matches
}

TEST_CASE("BUG-ST-OVERREAD: StHandler b32 write does not corrupt adjacent memory",
          "[unit][regression][BUG-ST-OVERREAD]") {
    // RED PHASE: This test must FAIL on unpatched code.
    // Extended test: verify that writing 4 bytes doesn't corrupt adjacent memory.

    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.reserve(8);
    stmts.push_back(make_shared_decl("buf", 64, Qualifier::Q_B32)); // PC=0: 64 elements
    stmts.push_back(make_mov_imm("r0", 0xAAAAAAAA));                // PC=1: value to write
    stmts.push_back(make_mov_imm("r1", 0));                        // PC=2: offset 0
    stmts.push_back(make_mov_imm("r3", 0xBBBBBBBB));               // PC=3: value for buf[1]
    stmts.push_back(make_mov_imm("r4", 4));                        // PC=4: offset 4 (buf[1])
    stmts.push_back(make_st_shared_addr("buf", "r4", "r3", Qualifier::Q_B32)); // PC=5: write buf[1]
    stmts.push_back(make_st_shared_addr("buf", "r1", "r0", Qualifier::Q_B32)); // PC=6: write buf[0]
    stmts.push_back(make_ld_shared_addr("r2", "buf", "r4", Qualifier::Q_B32)); // PC=7: load buf[1]
    stmts.push_back(make_ret());                                    // PC=8

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);

    // Execute
    int step_count = 0;
    int pc = -1;
    while ((pc = step_warp(w, stmts)) >= 0 && step_count < 20) {
        step_count++;
        if (pc == 8) break;
    }

    REQUIRE(step_count > 0);

    // BUG CHECK: If StHandler writes 8 bytes for b32, buf[1] would be corrupted
    // Expected: r2 == 0xBBBBBBBB (buf[1] should retain its value)
    // Bug behavior: r2 might be garbage or 0 if 8-byte write overwrote it
    uint32_t buf1_val = read_reg_u32(w, "r2", 0);
    CHECK(buf1_val == 0xBBBBBBBB);
}