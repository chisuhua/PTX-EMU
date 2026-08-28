#include "ptxsim/barrier/barrier_module.h"
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include <vector>
#include <map>

using namespace ptxsim;

static std::vector<ptxemu::ir::StatementContext> build_barrier_statements() {
    std::vector<ptxemu::ir::StatementContext> stmts;
    for (int i = 0; i < 5; i++) stmts.push_back({S_MOV, GenericInstr{}});

    {
        ptxemu::ir::StatementContext ctx;
        ctx.type = S_BAR_WARP_SYNC;
        BarWarpSyncInstr instr;
        instr.qualifiers = {ptxemu::ir::Qualifier::Q_B32};
        instr.operands.push_back(ptxemu::ir::OperandContext{ImmOperand{"65535"}});
        instr.operands.push_back(ptxemu::ir::OperandContext{ImmOperand{"7"}});
        ctx.data = instr;
        stmts.push_back(ctx);
    }

    for (int i = 0; i < 4; i++) stmts.push_back({S_MOV, GenericInstr{}});
    stmts.push_back({S_RET, GenericInstr{}});
    return stmts;
}

static void init_env() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

TEST_CASE("bar_warp_sync_next_pc_not_overwritten", "[barrier][pipeline][bug]") {
    init_env();

    auto statements = build_barrier_statements();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;

    WarpContext warp;
    for (int lane = 0; lane < 16; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements,
                     &name2Sym, label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        warp.add_thread(std::move(thread), lane);
    }
    warp.set_active_mask(0x0000FFFFu);

    for (int pc = 0; pc < 6; pc++) {
        warp.execute_warp_instruction(statements[pc], pc);
    }

    auto& warp_state = warp.get_warp_state();
    ptxsim::BarrierModule bm;
    bm.init_warp_barrier(0, 0x0000FFFFu, 7, 5);
    for (int lane = 0; lane < 16; lane++) {
        bm.get_warp_barrier(0)->arrive(lane);
    }
    REQUIRE(bm.is_warp_barrier_complete(0) == true);

    warp.execute_warp_instruction(statements[6], 6);

    auto* t0 = warp.get_thread(0);
    INFO("thread 0 next_pc = " << t0->get_next_pc() << ", expected 7");
    CHECK(t0->get_next_pc() == 7);
}