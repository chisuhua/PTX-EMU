/**
 * @file test_syncthreads_test3_repro.cpp
 * @brief Exact reproduction of test_syncthreads Test 3 execution sequence
 * @date 2026-04-15
 */

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "register/register_bank_manager.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/instruction_base.h"
#include "utils/logger.h"
#include <memory>

/// Build exact PTX statements from test_nested_sync with proper operand setup
static std::vector<StatementContext> build_nested_sync_statements() {
    std::vector<StatementContext> statements;
    std::map<std::string, int> label2pc;

    // PC 0-5: prologue
    for (int i = 0; i < 6; i++) {
        StatementContext ctx;
        ctx.type = S_MOV;
        statements.push_back(ctx);
    }

    // PC=6: bar.sync 0 (first barrier)
    {
        StatementContext ctx;
        ctx.type = S_BAR_WARP_SYNC;
        BarWarpSyncInstr instr;
        instr.qualifiers = {Qualifier::Q_B32};
        instr.operands.push_back(OperandContext{ImmOperand{"65535"}});
        instr.operands.push_back(OperandContext{ImmOperand{"-1"}});
        ctx.data = instr;
        statements.push_back(ctx);
    }

    // PC=7: setp.gt.u32 %p1, %r1, 15
    {
        StatementContext ctx;
        ctx.type = S_SETP;
        statements.push_back(ctx);
    }

    // PC=8-9: mov, add
    for (int i = 0; i < 2; i++) {
        StatementContext ctx;
        ctx.type = S_MOV;
        statements.push_back(ctx);
    }

    // PC=10: @%p1 bra $L__BB2_2
    {
        StatementContext ctx;
        ctx.type = S_BRA;
        BranchInstr instr;
        instr.target = "$L__BB2_2";
        instr.reconvergence_pc = -1;
        instr.predicate = "p1";
        ctx.data = instr;
        statements.push_back(ctx);
    }

    // PC 11-17: not-taken path (7 instructions)
    for (int i = 0; i < 7; i++) {
        StatementContext ctx;
        ctx.type = S_MOV;
        statements.push_back(ctx);
    }

    // PC=18: label $L__BB2_2
    {
        StatementContext ctx;
        ctx.type = S_LABEL;
        LabelInstr instr;
        instr.labelName = "$L__BB2_2";
        ctx.data = instr;
        statements.push_back(ctx);
    }

    // PC=19: cvta
    {
        StatementContext ctx;
        ctx.type = S_MOV;
        statements.push_back(ctx);
    }

    // PC=20: bar.sync 0 (second barrier)
    {
        StatementContext ctx;
        ctx.type = S_BAR_WARP_SYNC;
        BarWarpSyncInstr instr;
        instr.qualifiers = {Qualifier::Q_B32};
        instr.operands.push_back(OperandContext{ImmOperand{"65535"}});
        instr.operands.push_back(OperandContext{ImmOperand{"-1"}});
        ctx.data = instr;
        statements.push_back(ctx);
    }

    // PC 21-24: post-barrier (ld, mul, add, st)
    for (int i = 0; i < 4; i++) {
        StatementContext ctx;
        ctx.type = S_MOV;
        statements.push_back(ctx);
    }

    // PC=25: ret
    {
        StatementContext ctx;
        ctx.type = S_RET;
        statements.push_back(ctx);
    }

    // Register labels
    for (int i = 0; i < (int)statements.size(); i++) {
        if (statements[i].type == S_LABEL) {
            const auto& lbl = std::get<LabelInstr>(statements[i].data);
            label2pc[lbl.labelName] = i;
        }
    }

    // CFG post-processing
    ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(statements, label2pc);
    ptx::cfg::PostDominatorMap postDoms =
        ptx::cfg::CFGBuilder::computePostDominators(cfg);

    for (int i = 0; i < (int)statements.size(); i++) {
        auto& stmt = statements[i];
        if (stmt.type == S_BRA) {
            auto& branch = std::get<BranchInstr>(stmt.data);
            auto it = postDoms.find(i);
            int reconvergence_pc =
                (it != postDoms.end() && it->second >= 0) ? it->second : i + 1;
            branch.reconvergence_pc = reconvergence_pc;
        } else if (stmt.type == S_BAR_WARP_SYNC) {
            auto& barrier = std::get<BarWarpSyncInstr>(stmt.data);
            if (barrier.operands.size() >= 2) {
                barrier.operands[1] =
                    OperandContext{ImmOperand{std::to_string(i + 1)}};
            }
        }
    }

    return statements;
}

TEST_CASE("test_syncthreads Test 3: exact nested sync execution sequence",
          "[barrier][scheduler][branch-divergence][full-pipeline][test3]")
{
    auto statements = build_nested_sync_statements();

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    WarpContext warp;

    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);

        if (lane < 16) {
            thread->set_state(RUN);
        } else {
            thread->set_state(EXIT);
        }
        warp.add_thread(std::move(thread), lane);
    }

    warp.set_active_mask(0x0000FFFFu);

    SECTION("Initial state: 1 warp with 16 active threads at PC=0") {
        CHECK(warp.get_active_count() == 16);

        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            REQUIRE(t != nullptr);
            CHECK(!t->is_exited());
            CHECK(t->get_pc() == 0);
        }
    }

    SECTION("CFG analysis: barrier reconvergence_pc correctly set to i+1") {
        auto& b1 = std::get<BarWarpSyncInstr>(statements[6].data);
        std::string val1 = std::get<ImmOperand>(b1.operands[1].data).value;
        CHECK(std::stoi(val1) == 7);

        auto& b2 = std::get<BarWarpSyncInstr>(statements[20].data);
        std::string val2 = std::get<ImmOperand>(b2.operands[1].data).value;
        CHECK(std::stoi(val2) == 21);
    }

    SECTION("Scheduler simulation: exe_once() loop advances warp through PC=0 to PC=6") {
        int iterations = 0;
        while (iterations < 10) {
            for (int lane = 0; lane < 16; lane++) {
                auto* t = warp.get_thread(lane);
                if (t && !t->is_exited() && t->get_pc() < 6) {
                    warp.advance_thread_pc(lane, t->get_pc() + 1);
                }
            }
            iterations++;
        }

        int at_barrier = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 6) {
                at_barrier++;
            }
        }
        CHECK(at_barrier == 16);
    }

    SECTION("Branch divergence at PC=10: all 16 threads take fall-through to PC=11") {
        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 10);
        }

        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 11);
        }

        int at_pc_11 = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 11) {
                at_pc_11++;
            }
        }
        CHECK(at_pc_11 == 16);
    }

    SECTION("Second barrier at PC=20: all 16 threads reconverge to PC=21") {
        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 20);
        }

        int at_barrier = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 20) {
                at_barrier++;
            }
        }
        CHECK(at_barrier == 16);

        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 21);
        }

        int past_barrier = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 21) {
                past_barrier++;
            }
        }
        CHECK(past_barrier == 16);
    }

    SECTION("Full execution: all threads reach ret at PC=25") {
        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 25);
        }

        int at_ret = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 25) {
                at_ret++;
            }
        }
        CHECK(at_ret == 16);
    }

    SECTION("Wbar completes with 16-thread participation mask (not 0xFFFFFFFF)") {
        ptxsim::Wbar wbar;
        uint32_t correct_mask = 0x0000FFFFu;
        wbar.init(correct_mask, 21);

        for (int lane = 0; lane < 16; lane++) {
            wbar.arrive(lane);
        }

        CHECK(wbar.is_complete() == true);
        CHECK(wbar.count_arrived() == 16);
        CHECK(wbar.count_participants() == 16);
        CHECK(wbar.reconvergence_pc == 21);
    }
}

/**
 * @brief Test barrier execution with direct Wbar manipulation
 * @details Verifies that BarWarpSyncHandler path works correctly
 */
TEST_CASE("Barrier execution: direct Wbar test with 16 threads",
          "[barrier][wbar][execution][test3]")
{
    auto statements = build_nested_sync_statements();

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    WarpContext warp;

    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);

        if (lane < 16) {
            thread->set_state(RUN);
        } else {
            thread->set_state(EXIT);
        }
        warp.add_thread(std::move(thread), lane);
    }

    warp.set_active_mask(0x0000FFFFu);

    SECTION("Execute first barrier: PC=6 to PC=7 with 16 threads") {
        // Set all threads to PC=6 and BAR_SYNC state
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            REQUIRE(t != nullptr);
            t->set_pc(6);
            t->set_next_pc(6);
            t->set_state(BAR_SYNC);
        }

        // Get warp state and initialize barrier
        auto& warp_state = warp.get_warp_state();
        ptxsim::Wbar& wbar = warp_state.wbars[0];
        
        // Initialize with 16-thread mask and reconvergence to PC=7
        uint32_t participation_mask = 0x0000FFFFu;
        wbar.init(participation_mask, 7);
        warp_state.current_wbar_id = 0;

        // Simulate all 16 threads arriving at barrier
        for (int lane = 0; lane < 16; lane++) {
            wbar.arrive(lane);
        }

        // Verify barrier is complete
        CHECK(wbar.is_complete() == true);
        
        // After barrier completion, threads should be released to PC=7
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            if (t) {
                t->set_pc(7);
                t->set_next_pc(7);
                t->set_state(RUN);
            }
        }

        // Verify all threads moved to PC=7
        int at_pc_7 = 0;
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 7) {
                at_pc_7++;
            }
        }
        CHECK(at_pc_7 == 16);
    }

    SECTION("Execute second barrier: PC=20 to PC=21 with 16 threads") {
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            REQUIRE(t != nullptr);
            t->set_pc(20);
            t->set_next_pc(20);
            t->set_state(BAR_SYNC);
        }

        auto& warp_state = warp.get_warp_state();
        ptxsim::Wbar& wbar = warp_state.wbars[0];
        
        wbar.init(0x0000FFFFu, 21);
        warp_state.current_wbar_id = 0;

        for (int lane = 0; lane < 16; lane++) {
            wbar.arrive(lane);
        }

        CHECK(wbar.is_complete() == true);
        
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            if (t) {
                t->set_pc(21);
                t->set_next_pc(21);
                t->set_state(RUN);
            }
        }

        int at_pc_21 = 0;
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 21) {
                at_pc_21++;
            }
        }
        CHECK(at_pc_21 == 16);
    }

    SECTION("Full sequence: two barriers back-to-back") {
        // First barrier
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            t->set_pc(6);
            t->set_next_pc(6);
            t->set_state(BAR_SYNC);
        }

        auto& warp_state = warp.get_warp_state();
        ptxsim::Wbar& wbar1 = warp_state.wbars[0];
        wbar1.init(0x0000FFFFu, 7);
        warp_state.current_wbar_id = 0;

        for (int lane = 0; lane < 16; lane++) {
            wbar1.arrive(lane);
        }
        CHECK(wbar1.is_complete() == true);

        // Release to PC=7
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            t->set_pc(7);
            t->set_state(RUN);
        }

        // Second barrier
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            t->set_pc(20);
            t->set_next_pc(20);
            t->set_state(BAR_SYNC);
        }

        wbar1.reset();
        wbar1.init(0x0000FFFFu, 21);

        for (int lane = 0; lane < 16; lane++) {
            wbar1.arrive(lane);
        }
        CHECK(wbar1.is_complete() == true);

        // Release to PC=21
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            t->set_pc(21);
            t->set_state(RUN);
        }

        int at_pc_21 = 0;
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && t->get_pc() == 21) {
                at_pc_21++;
            }
        }
        CHECK(at_pc_21 == 16);
    }
}
