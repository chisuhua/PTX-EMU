// test_warp_barrier_memory_visibility.cpp
// =============================================================================
// Integration test (类型二) — intra-warp divergent-path shared-memory
// visibility across `bar.warp.sync` (single-warp barrier).
//
// Instruction sequence (PC=0..15):
//   PC=0,1:  S_SHARED .b32 buf_a[32], buf_b[32]  (decls, init-only)
//   PC=2:    mov.b32   %r_tid, %tid.x          ; r_tid[lane] = lane_id
//   PC=3:    setp.lt.u32 %p_lane_lt_16, %r_tid, 16
//   PC=4:    @%p_lane_lt_16 bra L_path_b       ; lanes 0-15 take → PC=7
//   PC=5:    st.shared.b32 [buf_a + %r_tid], %r_val  ; path A: lanes 16-31
//   write PC=6:    bra.uni L_join                    ; → PC=9 PC=7: L_path_b:
//   PC=8:    st.shared.b32 [buf_b + %r_tid], %r_val  ; path B: lanes 0-15 write
//   PC=9:    L_join:
//   PC=10:   bar.warp.sync 0xFFFFFFFF, 11      ; warp-level barrier, release to
//   PC=11 PC=11:   mov.b32 %r_tid, 16                ; uniform offset for
//   path-A read PC=12:   ld.shared.b32 %r_result_a, [buf_a + %r_tid] PC=13:
//   mov.b32 %r_tid, 0                 ; uniform offset for path-B read PC=14:
//   ld.shared.b32 %r_result_b, [buf_b + %r_tid] PC=15:   ret
//
// Per-lane r_val setup:
//   lanes 0-15  (path B)  → 0xBBBB
//   lanes 16-31 (path A)  → 0xAAAA
//
// S_SHARED declarations (PC=0..1) are consumed by CTAContext::init (creates
// name2Share entries); SMContext::add_block then allocates 256 B of shared
// memory and assigns buf_a offset 0, buf_b offset 128 via
// build_shared_memory_symbol_table.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/instruction_helpers.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptxsim/testing/predicates.h"
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

using ptxsim::testing::make_bar_warp_sync;
using ptxsim::testing::make_bra;
using ptxsim::testing::make_bra_pred;
using ptxsim::testing::make_label;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_mov;
using ptxsim::testing::make_mov_imm;
using ptxsim::testing::make_ret;
using ptxsim::testing::make_setp_lt_imm;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::setup_pred;
using ptxsim::testing::step_warp;

namespace {

// -----------------------------------------------------------------------------
// Factory / setup
// -----------------------------------------------------------------------------

static void init_instruction_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

// Set per-lane value of a 32-bit register "rN" via RegisterBankManager.
// Auto-creates the register if it doesn't exist yet.
static void set_reg_per_lane_u32(WarpContext *w, const std::string &reg_name,
                                 std::function<uint32_t(int /*lane*/)> fn) {
    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm != nullptr);
    if (!rbm->get_register(reg_name, 0, 0)) {
        rbm->create_register(reg_name, sizeof(uint32_t));
    }
    for (int i = 0; i < 32; i++) {
        void *p = rbm->get_register(reg_name, 0, i);
        REQUIRE(p != nullptr);
        *static_cast<uint32_t *>(p) = fn(i);
    }
}

static std::vector<StatementContext>
build_statements(std::map<std::string, int> &l2pc) {
    std::vector<StatementContext> stmts;
    stmts.reserve(16);

    // PC=0..1: shared-memory declarations (consumed by CTAContext::init, not
    // by the executor). The DeclarationHandler::ExecPipe just advances PC.
    stmts.push_back(make_shared_decl("buf_a", 32));
    stmts.push_back(make_shared_decl("buf_b", 32));

    // r5 = lane id for the predicate; r1 is preset to lane*4 because
    // PTX [symbol+register] shared-memory addresses are byte offsets.
    stmts.push_back(make_mov("r5", "tid.x"));                   // PC=2
    stmts.push_back(make_setp_lt_imm("p1", "r5", 16));          // PC=3
    stmts.push_back(make_bra_pred("L_path_b", "p1", false, 9)); // PC=4
    stmts.push_back(make_st_shared_addr("buf_a", "r1", "r2", Qualifier::Q_B32));  // PC=5 (path A)
    stmts.push_back(make_bra("L_join"));                        // PC=6
    stmts.push_back(make_label("L_path_b"));                    // PC=7
    stmts.push_back(make_st_shared_addr("buf_b", "r1", "r2", Qualifier::Q_B32));  // PC=8 (path B)
    stmts.push_back(make_label("L_join"));                      // PC=9
    stmts.push_back(make_bar_warp_sync(0xFFFFFFFF, 11));        // PC=10
    stmts.push_back(make_mov_imm("r1", 64));                    // PC=11: buf_a[16] byte offset
    stmts.push_back(make_ld_shared_addr("r3", "buf_a", "r1", Qualifier::Q_B32));  // PC=12
    stmts.push_back(make_mov_imm("r1", 0));                     // PC=13
    stmts.push_back(make_ld_shared_addr("r4", "buf_b", "r1", Qualifier::Q_B32));  // PC=14
    stmts.push_back(make_ret());                                // PC=15

    l2pc.clear();
    l2pc["L_path_b"] = 7;
    l2pc["L_join"] = 9;
    return stmts;
}

static WarpContext *setup_block(SMContext &sm,
                                std::vector<StatementContext> &stmts,
                                std::map<std::string, int> &l2pc) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{32, 1, 1};
    Dim3 bi{0, 0, 0};
    std::map<std::string, std::unique_ptr<Symtable>> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc);
    // sharedMemBytes is auto-computed from S_SHARED declarations by
    // CTAContext::init: 2 * (32 * 4) = 256 bytes. SMContext::add_block will
    // allocate that 256-byte buffer and assign buf_a offset 0, buf_b offset
    // 128 via build_shared_memory_symbol_table.
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    return sm.get_warp(0);
}

} // namespace

// =============================================================================
// TEST_CASE 1: Basic intra-warp divergent-path memory visibility
// =============================================================================
// Verifies that after bar.warp.sync, every lane can read shared-memory values
// written by BOTH divergent paths (path A at lane 16 writes 0xAAAA, path B at
// lane 0 writes 0xBBBB).
// Three-layer assertions (L1 shared memory, L2 registers, L3 active_mask).
// =============================================================================
TEST_CASE("integration_warp_barrier_memory_visibility_basic",
          "[integration][barrier][memory_visibility][warp_barrier]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::map<std::string, int> l2pc;
    auto stmts = build_statements(l2pc);

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts, l2pc);
    REQUIRE(w != nullptr);

    // r5 = lane id for the predicate; r1 is preset to lane*4 because PTX
    // [symbol+register] shared-memory addresses are byte offsets.
    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return static_cast<uint32_t>(lane) * 4u;
    });

    // r_val = 0xBBBB for lanes 0-15 (path B), 0xAAAA for lanes 16-31 (path A)
    set_reg_per_lane_u32(w, "r2", [](int lane) {
        return lane < 16 ? 0x0000BBBBu : 0x0000AAAAu;
    });

    // %p_lane_lt_16 true for lanes 0-15, false for 16-31
    setup_pred(w, 0x0000FFFFu);

    // Drive execution via step_warp until the ret at PC=15 is executed.
    // active_mask must be all-lanes-active right after bar.warp.sync releases
    // (first post-barrier PC=11), because ret will mark every lane exited.
    int ret_pc = -1;
    bool post_barrier_active_mask_ok = false;
    for (int step = 0; step < 128; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 11) {
            post_barrier_active_mask_ok =
                (w->get_active_mask() == 0xFFFFFFFFu);
        }
        if (pc == 15) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 15);
    REQUIRE(post_barrier_active_mask_ok);

    // ------------------------------------------------------------------
    // L1 — Shared memory content (per-lane via shared_mem_space pointer)
    // ------------------------------------------------------------------
    ThreadContext *t0 = w->get_thread(0);
    REQUIRE(t0 != nullptr);
    void *shmem_raw = t0->shared_mem_space;
    REQUIRE(shmem_raw != nullptr);
    auto *shmem_a = reinterpret_cast<uint32_t *>(shmem_raw); // buf_a @ offset 0
    auto *shmem_b = reinterpret_cast<uint32_t *>( // buf_b @ offset 128
        static_cast<char *>(shmem_raw) + 128);

    // Path B (lanes 0-15) wrote 0xBBBB to buf_b[0..15]; buf_a[0..15] untouched
    // (==0)
    for (int i = 0; i < 16; ++i) {
        INFO("L1 buf_a[" << i << "] should be 0 (path A didn't touch it)");
        CHECK(shmem_a[i] == 0u);
        INFO("L1 buf_b[" << i << "] should be 0xBBBB (path B wrote here)");
        CHECK(shmem_b[i] == 0x0000BBBBu);
    }
    // Path A (lanes 16-31) wrote 0xAAAA to buf_a[16..31]; buf_b[16..31]
    // untouched
    for (int i = 16; i < 32; ++i) {
        INFO("L1 buf_a[" << i << "] should be 0xAAAA (path A wrote here)");
        CHECK(shmem_a[i] == 0x0000AAAAu);
        INFO("L1 buf_b[" << i << "] should be 0 (path B didn't touch it)");
        CHECK(shmem_b[i] == 0u);
    }

    // ------------------------------------------------------------------
    // L2 — Register values (32 individual CHECKs per register)
    // ------------------------------------------------------------------
    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm != nullptr);

    // r_result_a = r3 — every lane read buf_a[16] = 0xAAAA (path A's lane-16
    // write)
    for (int i = 0; i < 32; ++i) {
        void *p = rbm->get_register("r3", 0, i);
        REQUIRE(p != nullptr);
        uint32_t v = *static_cast<uint32_t *>(p);
        INFO("L2 r_result_a lane " << i << " = 0x" << std::hex << v);
        CHECK(v == 0x0000AAAAu);
    }
    // r_result_b = r4 — every lane read buf_b[0] = 0xBBBB (path B's lane-0
    // write)
    for (int i = 0; i < 32; ++i) {
        void *p = rbm->get_register("r4", 0, i);
        REQUIRE(p != nullptr);
        uint32_t v = *static_cast<uint32_t *>(p);
        INFO("L2 r_result_b lane " << i << " = 0x" << std::hex << v);
        CHECK(v == 0x0000BBBBu);
    }
}

// =============================================================================
// TEST_CASE 2: Different per-path values (distinguishable markers)
// =============================================================================
// Same divergence pattern but with distinguishable markers (0xCAFE_BABE /
// 0xDEAD_BEEF) to ensure the test actually compares the specific value the
// divergent path wrote, not just "any non-zero".
// =============================================================================
TEST_CASE("integration_warp_barrier_memory_visibility_distinct_values",
          "[integration][barrier][memory_visibility][warp_barrier]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    constexpr uint32_t VAL_A = 0xCAFEBABEu; // path A
    constexpr uint32_t VAL_B = 0xDEADBEEFu; // path B

    std::map<std::string, int> l2pc;
    auto stmts = build_statements(l2pc);

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts, l2pc);
    REQUIRE(w != nullptr);

    set_reg_per_lane_u32(w, "r1", [](int lane) {
        return static_cast<uint32_t>(lane) * 4u;
    });
    set_reg_per_lane_u32(w, "r2", [VAL_A, VAL_B](int lane) {
        return lane < 16 ? VAL_B : VAL_A;
    });
    setup_pred(w, 0x0000FFFFu);

    int ret_pc = -1;
    bool post_barrier_active_mask_ok = false;
    for (int step = 0; step < 128; ++step) {
        int pc = step_warp(w, stmts);
        if (pc == 11) {
            post_barrier_active_mask_ok =
                (w->get_active_mask() == 0xFFFFFFFFu);
        }
        if (pc == 15) {
            ret_pc = pc;
            break;
        }
    }
    REQUIRE(ret_pc == 15);
    REQUIRE(post_barrier_active_mask_ok);

    // L1 — shared memory
    ThreadContext *t0 = w->get_thread(0);
    REQUIRE(t0 != nullptr);
    void *shmem_raw = t0->shared_mem_space;
    REQUIRE(shmem_raw != nullptr);
    auto *shmem_a = reinterpret_cast<uint32_t *>(shmem_raw);
    auto *shmem_b =
        reinterpret_cast<uint32_t *>(static_cast<char *>(shmem_raw) + 128);
    for (int i = 0; i < 16; ++i) {
        CHECK(shmem_a[i] == 0u);
        CHECK(shmem_b[i] == VAL_B);
    }
    for (int i = 16; i < 32; ++i) {
        CHECK(shmem_a[i] == VAL_A);
        CHECK(shmem_b[i] == 0u);
    }

    // L2 — per-lane r_result_a (r3) and r_result_b (r4)
    auto rbm = w->get_register_bank_manager();
    REQUIRE(rbm != nullptr);
    for (int i = 0; i < 32; ++i) {
        uint32_t va = *static_cast<uint32_t *>(rbm->get_register("r3", 0, i));
        uint32_t vb = *static_cast<uint32_t *>(rbm->get_register("r4", 0, i));
        CHECK(va == VAL_A);
        CHECK(vb == VAL_B);
    }
}
