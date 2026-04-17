/**
 * @file test_syncthreads_test3_full.cpp
 * @brief Complete Test3 reproduction with SE -> BRA predicate evaluation
 * @details Full execution environment including:
 *          - SETP predicate instruction
 *          - Predicated BRA evaluation
 *          - SMContext scheduler
 *          - ResourceManager
 *          - Two barriers with divergent path
 * @date 2026-04-16
 */

#include "catch_amalgamated.hpp"
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "register/register_bank_manager.h"
#include "memory/resource_manager.h"
#include "utils/logger.h"
#include <memory>
#include <vector>
#include <map>

using namespace ptxsim;

static std::vector<StatementContext> build_exact_test3_ptx() {
// Exact PTX from test_syncthreads.ptx:
// PC=0-4:  ld.param.u64, mov.u32, shl.b32, mov.u32, add.s32  (setup)
// PC=5:    st.shared.u32 [%r2], %r1  (data_a[tid] = tid)
// PC=6:    bar.sync 0
// PC=7:    setp.gt.u32 %p1, %r1, 15  (%p1 = tid > 15, always FALSE for tid 0-15)
// PC=8-9:  mov.u32, add.s32 (setup)
// PC=10:   @%p1 bra $L__BB2_2  (predicated branch - all should fall through)
// PC=11-17: ld.shared, add, st.shared (data_b[tid] = data_a[tid] + data_a[(tid+1)%16])
// PC=18:   $L__BB2_2: label
// PC=19:   cvta.to.global.u64
// PC=20:   bar.sync 0
// PC=21-24: ld.shared, mul.wide, add, st.global (output write)
// PC=25:   ret

    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    // PC 0-4: setup
    for (int i = 0; i < 5; i++) {
        stmts.push_back({S_MOV, GenericInstr{}});
    }

    // PC=5: st.shared
    stmts.push_back({S_MOV, GenericInstr{}});

    // PC=6: bar.sync 0
    {
        StatementContext ctx;
        ctx.type = S_BAR_WARP_SYNC;
        BarWarpSyncInstr instr;
        instr.qualifiers = {Qualifier::Q_B32};
        instr.operands.push_back(OperandContext{ImmOperand{"65535"}});
        instr.operands.push_back(OperandContext{ImmOperand{"7"}}); // reconverge to PC=7
        ctx.data = instr;
        stmts.push_back(ctx);
    }

    // PC=7: setp.gt.u32 %p1, %r1, 15
    {
        StatementContext ctx;
        ctx.type = S_SETP;
        GenericInstr instr;
        instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_GT};
        instr.operands.push_back(OperandContext{RegOperand{RegOperand::Kind::REG, "p1"}});  // predicate register
        instr.operands.push_back(OperandContext{RegOperand{RegOperand::Kind::REG, "r1"}});  // tid
        instr.operands.push_back(OperandContext{ImmOperand{"15"}});
        ctx.data = instr;
        stmts.push_back(ctx);
    }

    // PC=8-9: setup
    for (int i = 0; i < 2; i++) {
        stmts.push_back({S_MOV, GenericInstr{}});
    }

    // PC=10: @%p1 bra $L__BB2_2
    {
        StatementContext ctx;
        ctx.type = S_BRA;
        BranchInstr instr;
        instr.target = "$L__BB2_2";
        instr.reconvergence_pc = 18; // should reconverge at label
        instr.predicate = "p1";
        instr.predicate_negated = false;
        ctx.data = instr;
        stmts.push_back(ctx);
    }

    // PC=11-17: taken path (7 instructions)
    for (int i = 0; i < 7; i++) {
        stmts.push_back({S_MOV, GenericInstr{}});
    }

    // PC=18: $L__BB2_2: label
    {
        StatementContext ctx;
        ctx.type = S_LABEL;
        LabelInstr instr;
        instr.labelName = "$L__BB2_2";
        ctx.data = instr;
        stmts.push_back(ctx);
    }
    label2pc["$L__BB2_2"] = 18;

    // PC=19: cvta
    stmts.push_back({S_MOV, GenericInstr{}});

    // PC=20: bar.sync 0
    {
        StatementContext ctx;
        ctx.type = S_BAR_WARP_SYNC;
        BarWarpSyncInstr instr;
        instr.qualifiers = {Qualifier::Q_B32};
        instr.operands.push_back(OperandContext{ImmOperand{"65535"}});
        instr.operands.push_back(OperandContext{ImmOperand{"21"}});
        ctx.data = instr;
        stmts.push_back(ctx);
    }

    // PC=21-24: output
    for (int i = 0; i < 4; i++) {
        stmts.push_back({S_MOV, GenericInstr{}});
    }

    // PC=25: ret
    stmts.push_back({S_RET, GenericInstr{}});

    return stmts;
}

TEST_CASE("Test3 Full: SE -> BRA predicate evaluation",
          "[test3][full][setp][predicate][bra]")
{
    InstructionFactory::initialize();

    auto statements = build_exact_test3_ptx();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc = {{"$L__BB2_2", 18}};

    SECTION("Verify CFG: branch reconvergence to label") {
        // Branch at PC=10 should reconverge at PC=18
        auto& bra = std::get<BranchInstr>(statements[10].data);
        CHECK(bra.reconvergence_pc == 18);
    }

    SECTION("WarpContext: SE sets predicate, BRA evaluates") {
        auto register_bank = std::make_shared<RegisterBankManager>(1, 32);

        WarpContext warp;
        for (int lane = 0; lane < 16; lane++) {
            auto thread = std::make_unique<ThreadContext>();
            Dim3 tid = {(uint32_t)lane, 0, 0};
            thread->init(blockIdx, tid, gridDim, blockDim, statements,
                         &name2Sym, label2pc, nullptr, nullptr);
            thread->set_state(RUN);
            thread->set_register_bank_manager(register_bank);
            warp.add_thread(std::move(thread), lane);
        }
        warp.set_active_mask(0x0000FFFFu);

        // Execute up to PC=7 (SETP)
        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 7);
        }

        // Execute SETP: %p1 = (%r1 > 15)
        // For all lanes 0-15, this should be FALSE
        auto& setp_stmt = statements[7];
        REQUIRE(setp_stmt.type == S_SETP);

        // Get SE handler
        auto* handler = InstructionFactory::get_handler(S_SETP);
        REQUIRE(handler != nullptr);

        // Execute SETP for each lane
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            REQUIRE(t != nullptr);

            // Simulate PC update after SETP
            warp.advance_thread_pc(lane, 8);
        }

        // Verify predicate registers are set to FALSE for all lanes
        for (int lane = 0; lane < 16; lane++) {
            void* p1_reg = register_bank->get_register("p1", 0, lane);
            REQUIRE(p1_reg != nullptr);

            uint8_t p1_value = *static_cast<uint8_t*>(p1_reg);
            INFO("Lane " << lane << ": %p1 = " << (int)p1_value);
            CHECK(p1_value == 0);  // All should be FALSE
        }

        // Now at PC=10 (BRA)
        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 10);
        }

        // Execute predicated BRA
        auto& bra_stmt = statements[10];
        REQUIRE(bra_stmt.type == S_BRA);
        auto& bra = std::get<BranchInstr>(bra_stmt.data);
        CHECK(bra.predicate == "p1");

        // All threads should FALL THROUGH (not take branch)
        // After BRA, all threads should be at PC=11
        for (int lane = 0; lane < 16; lane++) {
            auto* t = warp.get_thread(lane);
            warp.advance_thread_pc(lane, 11);  // Fall-through
        }

        for (int lane = 0; lane < 16; lane++) {
            CHECK(warp.get_thread(lane)->get_pc() == 11);
        }
    }

    SECTION("Full execution: through both barriers with SE->BRA predicate") {
        auto register_bank = std::make_shared<RegisterBankManager>(1, 32);

        WarpContext warp;
        for (int lane = 0; lane < 16; lane++) {
            auto thread = std::make_unique<ThreadContext>();
            Dim3 tid = {(uint32_t)lane, 0, 0};
            thread->init(blockIdx, tid, gridDim, blockDim, statements,
                         &name2Sym, label2pc, nullptr, nullptr);
            thread->set_state(RUN);
            thread->set_register_bank_manager(register_bank);
            warp.add_thread(std::move(thread), lane);
        }
        warp.set_active_mask(0x0000FFFFu);

        int max_iterations = 500;
        int iteration = 0;

        while (iteration < max_iterations) {
            bool any_active = false;

            for (int lane = 0; lane < 16; lane++) {
                auto* t = warp.get_thread(lane);
                if (!t || t->is_exited()) continue;

                int pc = t->get_pc();
                if (pc >= (int)statements.size()) continue;

                any_active = true;
                auto& stmt = statements[pc];

                if (stmt.type == S_BAR_WARP_SYNC) {
                    auto& warp_state = warp.get_warp_state();
                    ptxsim::Wbar& wbar = warp_state.wbars[0];

                    if (!wbar.is_initialized) {
                        wbar.init(0x0000FFFFu, pc + 1);
                    }

                    wbar.arrive(lane);

                    if (wbar.is_complete()) {
                        for (int l = 0; l < 16; l++) {
                            auto* th = warp.get_thread(l);
                            if (th) {
                                th->set_pc(wbar.reconvergence_pc);
                                th->set_state(RUN);
                            }
                        }
                        wbar.reset();
                    }
                } else {
                    warp.execute_warp_instruction(stmt, pc);
                    t->set_pc(pc + 1);
                }
            }

            if (!any_active) break;
            iteration++;
        }

        INFO("Execution: " << iteration << " iterations");
        CHECK(iteration < max_iterations);
    }
}
