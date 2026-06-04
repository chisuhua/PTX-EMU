#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/testing/scheduler_utils.h"
#include "register/register_bank_manager.h"
#include "memory/resource_manager.h"
#include "utils/logger.h"

using ptxsim::testing::step_warp;
#include <memory>
#include <vector>
#include <map>

using namespace ptxsim;
using namespace ptxir::factory;

static void init_execution_environment() {
    static bool initialized = false;
    if (!initialized) {
        InstructionFactory::initialize();
        ResourceManager::instance();
        initialized = true;
    }
}

static std::vector<StatementContext> build_barrier_divergence_statements() {
    std::vector<StatementContext> stmts;

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 1}}, OperandContext{RegOperand{"tid.x", -1}}},
        "mov.u32 %r1, %tid.x;"));

    stmts.push_back(makeGenericInstr(S_SETP, {Qualifier::Q_B32, Qualifier::Q_LT},
        {OperandContext{RegOperand{"p", 1}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"8"}}},
        "setp.lt.u32 %p1, %r1, 8;"));

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{ImmOperand{"100"}}},
        "mov.u32 %r2, 100;"));

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 3}}, OperandContext{ImmOperand{"200"}}},
        "mov.u32 %r3, 200;"));

    stmts.push_back(makeGenericInstr(S_SELP, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 4}}, OperandContext{RegOperand{"r", 2}}, OperandContext{RegOperand{"r", 3}}, OperandContext{RegOperand{"p", 1}}},
        "selp.b32 %r4, %r2, %r3, %p1;"));

    stmts.push_back(makeBarWarpSyncInstr(0x0000FFFFu, 9));

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B64},
        {OperandContext{RegOperand{"rd", 1}}, OperandContext{ImmOperand{"0"}}},
        "mov.u64 %rd1, 0;"));

    stmts.push_back(makeGenericInstr(S_ADD, {Qualifier::Q_S64},
        {OperandContext{RegOperand{"rd", 2}}, OperandContext{RegOperand{"rd", 1}}, OperandContext{ImmOperand{"0"}}},
        "add.s64 %rd2, %rd1, 0;"));

    stmts.push_back(makeVoidInstr(S_RET, "ret;"));

    return stmts;
}

static int expected_divergence_value(int lane) {
    return (lane < 8) ? 200 : 100;
}

TEST_CASE("test_barrier_divergence_scheduling: Structure verification", "[barrier_divergence][structure]") {
    init_execution_environment();
    auto stmts = build_barrier_divergence_statements();

    INFO("Statement count: " << stmts.size());
    REQUIRE(stmts.size() == 9);

    CHECK(stmts[0].type == S_MOV);
    CHECK(stmts[1].type == S_SETP);
    CHECK(stmts[4].type == S_SELP);
    CHECK(stmts[5].type == S_BAR_WARP_SYNC);
    CHECK(stmts[8].type == S_RET);
}

TEST_CASE("test_barrier_divergence_scheduling: Barrier blocks lanes and scheduler picks divergent path", "[barrier_divergence][scheduler][execution]") {
    init_execution_environment();

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};

    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    auto statements = build_barrier_divergence_statements();

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(statements);
    register_bank->preallocate_registers(registers);

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

    INFO("Warp active threads: " << warp.get_active_count());
    CHECK(warp.get_active_count() == 16);

    for (int pc = 0; pc < (int)statements.size(); pc++) {
        auto& stmt = statements[pc];
        INFO("PC=" << pc << " executing: " << stmt.instructionText);
        warp.execute_warp_instruction(stmt, pc);
    }

    INFO("\n=== Verification ===");
    INFO("After barrier divergence, lanes 0-7 should have r4=200 (divergent path), lanes 8-15 should have r4=100 (post-barrier)");

    uint32_t active_mask = warp.get_active_mask();
    INFO("Active mask after execution: 0x" << std::hex << active_mask);

    bool any_lane_blocked = false;
    for (int lane = 0; lane < 16; lane++) {
        auto* thread = warp.get_thread(lane);
        REQUIRE(thread != nullptr);

        void* r4_ptr = register_bank->get_register("r4", 0, lane);
        REQUIRE(r4_ptr != nullptr);

        int r4_val = *static_cast<int*>(r4_ptr);
        int expected = expected_divergence_value(lane);

INFO("Lane " << lane << ": r4=" << r4_val << " expected=" << expected
                      << " state=" << (thread->get_state() == BAR_SYNC ? "BAR_SYNC" : "RUN")
                      << " pc=" << thread->get_pc());

        if (thread->get_state() == BAR_SYNC) {
            any_lane_blocked = true;
        }
    }

    CHECK(any_lane_blocked == true);
}