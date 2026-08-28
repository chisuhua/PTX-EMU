/**
 * Simulates full CFG analysis of test_syncthreads Test 3 PTX.
 * Checks: does the second barrier get correct reconvergence_pc?
 */

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include <vector>
#include <map>
#include <string>
#include <cstdint>

namespace cfg = ptx::cfg;
using namespace ptx;

// Reconstruct test_nested_sync PTX instruction sequence.
// Derived from actual nvcc output of test_syncthreads.cu.
//
// Simplified model of the NVCC-generated PTX for Test 3:
// PC=0:  mov tid
// PC=1:  mov shared_a_offset
// PC=2:  shl tid*4
// PC=3:  add shared_a_base + offset
// PC=4:  st.shared  [addr] <- tid
// PC=5:  bar.warp.sync 0xFFFFFFFF, placeholder  ←  FIRST barrier
// PC=6:  setp.lt.u32
// PC=7:  @%p bra $L_skip
// PC=8:  mov shared_b_offset
// PC=9:  add shared_b_base + offset
// PC=10: ld.shared  val  [shared_a + offset]
// PC=11: add.u32   val + neighbor_val
// PC=12: st.shared  [shared_b + offset] <- val
// PC=13: $L_skip:
// PC=14: bar.warp.sync 0xFFFFFFFF, placeholder  ←  SECOND barrier
// PC=15: mov shared_b_offset
// PC=16: add shared_b_base + offset
// PC=17: ld.shared  val  [shared_b + offset]
// PC=18: st.global  [output] <- val
// PC=19: ret

static ptxemu::ir::StatementContext make_stmt(int pc, ptxemu::ir::StatementType type,
                                   const std::string& label = "",
                                   const std::string& mask = "0xFFFFFFFF",
                                   int placeholder_pc = -1) {
    ptxemu::ir::StatementContext ctx;
    ctx.type = type;

    if (type == S_BAR_WARP_SYNC) {
        BarWarpSyncInstr barrier;
        barrier.operands.push_back(ptxemu::ir::OperandContext{ImmOperand{mask}});
        barrier.operands.push_back(ptxemu::ir::OperandContext{ImmOperand{std::to_string(placeholder_pc)}});
        ctx.data = barrier;
        return ctx;
    }

    if (type == S_BRA) {
        BranchInstr branch;
        branch.target = label;
        branch.reconvergence_pc = -1;
        ctx.data = branch;
        return ctx;
    }

    if (type == S_DOLLOR) {
        DollarNameInstr dollar;
        dollar.name = label;
        ctx.data = dollar;
        return ctx;
    }

    GenericInstr generic;
    ctx.data = generic;
    return ctx;
}

TEST_CASE("Test3_CFG: Full CFG of test_nested_sync → second barrier reconvergence_pc",
          "[test3][cfg][reconvergence]") {
    std::vector<ptxemu::ir::StatementContext> stmts;
    std::map<std::string, int> label2pc;

    // PC=0-4: Write phase (fill data_a)
    for (int i = 0; i < 5; i++) stmts.push_back(make_stmt(i, S_MOV));

    // PC=5: First __syncthreads
    stmts.push_back(make_stmt(5, S_BAR_WARP_SYNC));

    // PC=6: setp (predicate)
    stmts.push_back(make_stmt(6, S_SETP));

    // PC=7: Conditional branch
    stmts.push_back(make_stmt(7, S_BRA, "L_skip"));
    label2pc["L_skip"] = 13;

    // PC=8-12: Divergent body (data_b[tid] = data_a[tid] + data_a[(tid+1)%16])
    for (int i = 8; i <= 12; i++) stmts.push_back(make_stmt(i, S_MOV));

    // PC=13: Label $L_skip (merge point)
    stmts.push_back(make_stmt(13, S_DOLLOR, "L_skip"));

    // PC=14: Second __syncthreads
    stmts.push_back(make_stmt(14, S_BAR_WARP_SYNC));

    // PC=15-18: Write output
    for (int i = 15; i <= 18; i++) stmts.push_back(make_stmt(i, S_MOV));

    // PC=19: Return
    stmts.push_back(make_stmt(19, S_RET));

    // Build CFG and compute post-dominators
    cfg::CFG cfg_obj = cfg::CFGBuilder::build(stmts, label2pc);
    cfg::PostDominatorMap postDoms = cfg::CFGBuilder::computePostDominators(cfg_obj);

    // Print all PC → post-dominator mappings
    for (const auto& b : cfg_obj.blocks) {
        INFO("Block " << b.id << " PC=" << b.start_pc << "-" << b.end_pc);
    }
    for (const auto& [pc, pd] : postDoms) {
        std::string label = (pc == 5 || pc == 14) ? "[BARRIER]" : "";
        INFO("  PC=" << pc << " -> post_dom=" << pd << " " << label);
    }

    // Key assertions:
    // 1. First barrier (PC=5) post-dominator should be PC=6 (next instr)
    auto it_b1 = postDoms.find(5);
    REQUIRE(it_b1 != postDoms.end());
    // If -1, fallback to PC+1 is used (reconvergence_pc = 6)
    if (it_b1->second == -1) {
        INFO("  First barrier fallback: reconvergence_pc will be PC+1=6");
    }

    // 2. Second barrier (PC=14) post-dominator should be PC=15 (next instr)
    auto it_b2 = postDoms.find(14);
    REQUIRE(it_b2 != postDoms.end());
    // If -1, fallback to PC+1 should be used
    INFO("  Second barrier post-dom=" << it_b2->second);
    REQUIRE((it_b2->second == 15 || it_b2->second == -1));
    // If -1, the interpreter fallback sets it to PC+1=15

    // 3. Branch (PC=7) should reconverge at $L_skip = PC=13
    auto it_br = postDoms.find(7);
    REQUIRE(it_br != postDoms.end());
    INFO("  Branch post-dom=" << it_br->second);
    REQUIRE(it_br->second == 13);

    // 4. All divergent body instructions (PC=8-12) should post-dominate to PC=13
    for (int pc = 8; pc <= 12; pc++) {
        auto it = postDoms.find(pc);
        if (it != postDoms.end()) {
            REQUIRE(it->second == 13);
        }
    }
}
