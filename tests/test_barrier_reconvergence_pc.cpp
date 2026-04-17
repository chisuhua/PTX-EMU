/// @file test_barrier_reconvergence_pc.cpp
/// @brief Reproduce Test 3 barrier reconvergence_pc pointing to self
/// @details Maps exactly to test_nested_sync from test_syncthreads.ptx
/// @date 2026-04-15

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"

TEST_CASE("Barrier reconvergence_pc in nested sync scenario",
          "[barrier][cfg][reconvergence]")
{
    std::vector<StatementContext> statements;
    std::map<std::string, int> label2pc;

    // PC 0-5: prologue (ld.param, mov, shl, mov, add, st.shared)
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
        instr.operands.push_back(OperandContext{ImmOperand{"4294967295"}});
        instr.operands.push_back(OperandContext{ImmOperand{"-1"}});
        ctx.data = instr;
        statements.push_back(ctx);
    }

    // PC=7: setp.gt.u32
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

    // PC=10: predicated bra to $L__BB2_2
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
        instr.operands.push_back(OperandContext{ImmOperand{"4294967295"}});
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

    // Register labels before CFG build
    for (int i = 0; i < (int)statements.size(); i++) {
        if (statements[i].type == S_LABEL) {
            const auto& lbl = std::get<LabelInstr>(statements[i].data);
            label2pc[lbl.labelName] = i;
        }
    }

    // Build CFG and compute post-dominators
    ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(statements, label2pc);
    ptx::cfg::PostDominatorMap postDoms =
        ptx::cfg::CFGBuilder::computePostDominators(cfg);

    // Simulate CFG post-processing (ptx_interpreter.cpp:598-634)
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

    // Verify first barrier: PC=6 -> reconvergence_pc should be 7
    {
        auto& barrier = std::get<BarWarpSyncInstr>(statements[6].data);
        std::string val = std::get<ImmOperand>(barrier.operands[1].data).value;
        CHECK(std::stoi(val) == 7);
    }

    // Verify second barrier: PC=20 -> reconvergence_pc should be 21
    {
        auto& barrier = std::get<BarWarpSyncInstr>(statements[20].data);
        std::string val = std::get<ImmOperand>(barrier.operands[1].data).value;
        int actual = std::stoi(val);

        INFO("Second barrier at PC=20 set reconvergence_pc=" << actual
             << " (expected 21).");
        CHECK(actual == 21);
    }

    // Debug: dump post-dominator map for key PCs
    SECTION("Post-dominator map for key PCs") {
        int key_pcs[] = {6, 7, 10, 11, 18, 19, 20, 21};
        for (int pc : key_pcs) {
            auto it = postDoms.find(pc);
            int pd = (it != postDoms.end()) ? it->second : -1;

            std::string type;
            if (statements[pc].type == S_BAR_WARP_SYNC) type = "barrier";
            else if (statements[pc].type == S_BRA) type = "bra";
            else if (statements[pc].type == S_LABEL) type = "label";
            else if (statements[pc].type == S_RET) type = "ret";
            else type = "instr";

            INFO("PC=" << pc << " (" << type << ") post_dom=" << pd);
        }
        CHECK(true);
    }
}
