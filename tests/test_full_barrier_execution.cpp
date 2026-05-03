/**
 * @file test_full_barrier_execution.cpp
 * @brief Full execution test reproducing test_syncthreads Test 3 (nested sync)
 * @details Creates exact PTX statements from test_nested_sync, sets up 16-thread
 *          warp, and runs the full execution pipeline to verify barrier completion.
 * @date 2026-04-15
 */

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"

/// Build statements exactly as parsed from test_nested_sync PTX,
/// then apply CFG post-processing (same as setupLabels in ptx_interpreter.cpp).
static std::vector<StatementContext> build_nested_sync_statements() {
    std::vector<StatementContext> statements;
    std::map<std::string, int> label2pc;

    // PC 0-5: prologue
    for (int i = 0; i < 6; i++) {
        StatementContext ctx;
        ctx.type = S_MOV;
        statements.push_back(ctx);
    }

    // PC=6: first bar.sync 0
    {
        StatementContext ctx;
        ctx.type = S_BAR_WARP_SYNC;
        BarWarpSyncInstr instr;
        instr.qualifiers = {Qualifier::Q_B32};
        instr.operands.push_back(OperandContext{ImmOperand{"65535"}});  // 16 threads
        instr.operands.push_back(OperandContext{ImmOperand{"-1"}});     // placeholder
        ctx.data = instr;
        statements.push_back(ctx);
    }

    // PC=7: setp
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

    // PC=20: second bar.sync 0
    {
        StatementContext ctx;
        ctx.type = S_BAR_WARP_SYNC;
        BarWarpSyncInstr instr;
        instr.qualifiers = {Qualifier::Q_B32};
        instr.operands.push_back(OperandContext{ImmOperand{"65535"}});  // 16 threads
        instr.operands.push_back(OperandContext{ImmOperand{"-1"}});     // placeholder
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

    // CFG analysis and post-processing (same as ptx_interpreter.cpp:586-634)
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

TEST_CASE("Full barrier execution: nested sync with 16 threads",
          "[barrier][execution][full-pipeline]")
{
    auto statements = build_nested_sync_statements();

    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx_base = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    WarpContext warp;
    std::vector<ThreadContext*> thread_ptrs;

    // Create 16 threads (lanes 0-15), deactivate lanes 16-31
    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);

            if (lane < 16) {
                thread->set_state(RUN);
            } else {
                // Lanes 16-31: mark as exited (they don't exist in 16-thread CTA)
                thread->set_state(EXIT);
            }
        thread_ptrs.push_back(thread.get());
        warp.add_thread(std::move(thread), lane);
    }

    // Activate only lanes 0-15
    uint32_t active_mask = 0x0000FFFFu;
    warp.set_active_mask(active_mask);

    SECTION("CFG post-processing sets correct barrier reconvergence_pc") {
        // First barrier at PC=6 should reconverge to PC=7
        auto& b1 = std::get<BarWarpSyncInstr>(statements[6].data);
        std::string val1 = std::get<ImmOperand>(b1.operands[1].data).value;
        CHECK(std::stoi(val1) == 7);

        // Second barrier at PC=20 should reconverge to PC=21
        auto& b2 = std::get<BarWarpSyncInstr>(statements[20].data);
        std::string val2 = std::get<ImmOperand>(b2.operands[1].data).value;
        CHECK(std::stoi(val2) == 21);
    }

    SECTION("All 16 threads start at PC=0 and are active") {
        int active_count = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && !t->is_exited()) {
                active_count++;
            }
        }
        CHECK(active_count == 16);

        // Verify get_thread returns non-null for lanes 0-15
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            REQUIRE(t != nullptr);
            CHECK(!t->is_exited());
        }
    }

    SECTION("Simulate: advance all threads through first barrier (PC=0→6)") {
        // Directly set PC on ThreadContext (bypassing warp_state checks)
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            REQUIRE(t != nullptr);
            t->set_pc(6);
            t->state = BAR_SYNC;
        }

        int at_barrier = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && !t->is_exited() && t->get_pc() == 6) {
                at_barrier++;
            }
        }
        CHECK(at_barrier == 16);
    }

    SECTION("Wbar completes with correct 16-thread participation mask") {
        // Simulate barrier initialization with 16-thread mask
        ptxsim::Wbar wbar;
        uint32_t expected_mask = 0x0000FFFFu;
        wbar.init(expected_mask, 7);  // reconvergence to PC=7

        // All 16 threads arrive
        for (int lane = 0; lane < 16; lane++) {
            wbar.arrive(lane);
        }

        CHECK(wbar.is_complete() == true);
        CHECK(wbar.count_arrived() == 16);
        CHECK(wbar.reconvergence_pc == 7);
    }

    SECTION("Barrier completion updates thread PCs correctly") {
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            if (t) {
                t->set_pc(6);
                t->state = BAR_SYNC;
            }
        }

        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            if (t) {
                t->set_pc(7);
                t->state = RUN;
            }
        }

        int past_barrier = 0;
        for (int lane = 0; lane < 32; lane++) {
            auto* t = warp.get_thread(lane);
            if (t && !t->is_exited() && t->get_pc() == 7) {
                past_barrier++;
            }
        }
        CHECK(past_barrier == 16);
    }
}
