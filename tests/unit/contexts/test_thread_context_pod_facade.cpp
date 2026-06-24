// Unit tests for T2-3 A3a — ThreadContext POD facade: header-level
// verifies that ThreadContext holds 4 POD members (exec_state_, reg_pred_,
// memory_, program_ref_) with default-initialized values. This locks in
// the A3a step BEFORE we begin migrating internal code (A3b).
//
// A3a is purely additive — old fields remain the canonical source.
// These tests ONLY verify the POD members exist and have defaults.
// Tests for behavior migration (A3b) and field removal (A3c) will be
// added in subsequent sub-tasks.

#include "ptxsim/contexts/exec_state.h"
#include "ptxsim/contexts/memory_ref.h"
#include "ptxsim/contexts/program_ref.h"
#include "ptxsim/contexts/register_predicate.h"
#include "ptxsim/thread_context.h"
#include <catch_amalgamated.hpp>

using ptxsim::contexts::ExecStatePod;
using ptxsim::contexts::MemoryPod;
using ptxsim::contexts::ProgramRefPod;
using ptxsim::contexts::RegisterPredicatePod;

TEST_CASE("ThreadContext facade: 4 POD members exist with default values",
          "[thread_context][facade][pod]") {
    ThreadContext tc;

    // ExecStatePod: default state is IDLE, lane_id_=0, warp_id_=0
    REQUIRE(tc.exec_state_.state == IDLE);
    REQUIRE(tc.exec_state_.lane_id_ == 0);
    REQUIRE(tc.exec_state_.warp_id_ == 0);
    REQUIRE(tc.exec_state_.bar_id == 0);
    REQUIRE(tc.exec_state_.BlockIdx.x == 0);
    REQUIRE(tc.exec_state_.BlockIdx.y == 0);
    REQUIRE(tc.exec_state_.BlockIdx.z == 0);
    REQUIRE(tc.exec_state_.ThreadIdx.x == 0);
    REQUIRE(tc.exec_state_.ThreadIdx.y == 0);
    REQUIRE(tc.exec_state_.ThreadIdx.z == 0);
    REQUIRE(tc.exec_state_.GridDim.x == 1);
    REQUIRE(tc.exec_state_.BlockDim.x == 1);

    // RegisterPredicatePod: register_bank_manager_ null, cc_reg default
    REQUIRE(tc.reg_pred_.register_bank_manager_ == nullptr);
    REQUIRE(tc.reg_pred_.operand_collected.empty());
    REQUIRE(tc.reg_pred_.operand_is_immediate_.empty());
    REQUIRE(tc.reg_pred_.vecOp_phy_addrs.empty());
    REQUIRE(tc.reg_pred_.dst_operand_reg_name_.empty());

    // MemoryPod: shared/local mem null, warp/CTA ctx null
    REQUIRE(tc.memory_.shared_mem_space == nullptr);
    REQUIRE(tc.memory_.local_mem_space == nullptr);
    REQUIRE(tc.memory_.warp_context_ == nullptr);
    REQUIRE(tc.memory_.cta_context_ == nullptr);

    // ProgramRefPod: statements null, label2pc empty, call_stack empty
    REQUIRE(tc.program_ref_.statements == nullptr);
    REQUIRE(tc.program_ref_.name2Sym == nullptr);
    REQUIRE(tc.program_ref_.name2Share == nullptr);
    REQUIRE(tc.program_ref_.label2pc.empty());
    REQUIRE(tc.program_ref_.call_stack.empty());
}

TEST_CASE("ThreadContext facade: init() populates 4 PODs from parameters",
          "[thread_context][facade][pod][init]") {
    ThreadContext tc;

    // Prepare init() parameters matching a typical thread in warp 0, lane 5
    Dim3 blockIdx = {0, 1, 0};
    Dim3 threadIdx = {5, 0, 0};
    Dim3 gridDim = {2, 2, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts; // empty
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    label2pc["L1"] = 10;
    std::map<std::string, std::unique_ptr<Symtable>> name2Share;

    tc.init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym, label2pc,
            &name2Share, nullptr);

    // ExecStatePod populated: BlockIdx/ThreadIdx/GridDim/BlockDim/state/bar_id
    REQUIRE(tc.exec_state_.BlockIdx.x == 0);
    REQUIRE(tc.exec_state_.BlockIdx.y == 1);
    REQUIRE(tc.exec_state_.ThreadIdx.x == 5);
    REQUIRE(tc.exec_state_.GridDim.x == 2);
    REQUIRE(tc.exec_state_.GridDim.y == 2);
    REQUIRE(tc.exec_state_.BlockDim.x == 32);
    REQUIRE(tc.exec_state_.state == RUN);
    REQUIRE(tc.exec_state_.bar_id == 0);
    REQUIRE(tc.exec_state_.warp_id_ == 0); // 5/32 = 0
    REQUIRE(tc.exec_state_.lane_id_ == 5);

    // ProgramRefPod populated: statements pointer, name2Sym/Share, label2pc
    REQUIRE(tc.program_ref_.statements == &stmts);
    REQUIRE(tc.program_ref_.name2Sym == &name2Sym);
    REQUIRE(tc.program_ref_.name2Share == &name2Share);
    REQUIRE(tc.program_ref_.label2pc.size() == 1);
    REQUIRE(tc.program_ref_.label2pc["L1"] == 10);

    // MemoryPod populated: cta_context_ (still null since nullptr passed)
    REQUIRE(tc.memory_.cta_context_ == nullptr);
}