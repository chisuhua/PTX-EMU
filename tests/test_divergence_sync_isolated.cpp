#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "register/register_bank_manager.h"
#include "memory/resource_manager.h"
#include "utils/logger.h"
#include <memory>
#include <vector>
#include <map>

using namespace ptxsim;
using namespace ptxir::factory;

static std::vector<StatementContext> build_divergence_sync_statements() {
    std::vector<StatementContext> stmts;

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 1}}, OperandContext{RegOperand{"tid.x", -1}}},
        "mov.u32 %r1, %tid.x;"));

    stmts.push_back(makeGenericInstr(S_SETP, {Qualifier::Q_B32, Qualifier::Q_LT},
        {OperandContext{RegOperand{"p", 1}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"4"}}},
        "setp.lt.u32 %p1, %r1, 4;"));

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 3}}, OperandContext{RegOperand{"r", 1}}},
        "mov.u32 %r3, %r1;"));

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 5}}, OperandContext{ImmOperand{"0"}}},
        "mov.u32 %r5, 0;"));

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 6}}, OperandContext{ImmOperand{"1"}}},
        "mov.u32 %r6, 1;"));

    stmts.push_back(makeGenericInstr(S_SELP, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 4}}, OperandContext{RegOperand{"r", 5}}, OperandContext{RegOperand{"r", 6}}, OperandContext{RegOperand{"p", 1}}},
        "selp.b32 %r4, %r5, %r6, %p1;"));

    stmts.push_back(makeBarWarpSyncInstr(0x000000FF, 9));

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B64},
        {OperandContext{RegOperand{"rd", 1}}, OperandContext{ImmOperand{"0"}}},
        "mov.u64 %rd1, 0;"));

    stmts.push_back(makeGenericInstr(S_MUL, {Qualifier::Q_U32, Qualifier::Q_WIDE},
        {OperandContext{RegOperand{"rd", 2}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"4"}}},
        "mul.wide.u32 %rd2, %r1, 4;"));

    stmts.push_back(makeGenericInstr(S_ADD, {Qualifier::Q_S64},
        {OperandContext{RegOperand{"rd", 3}}, OperandContext{RegOperand{"rd", 2}}, OperandContext{ImmOperand{"0"}}},
        "add.s64 %rd3, %rd2, 0;"));

    stmts.push_back(makeGenericInstr(S_ST, {Qualifier::Q_GLOBAL, Qualifier::Q_B32},
        {OperandContext{AddrOperand{}}, OperandContext{RegOperand{"r", 4}}},
        "st.global.u32 [%rd3], %r4;"));

    stmts.push_back(makeVoidInstr(S_RET, "ret;"));

    return stmts;
}

static void init_execution_environment() {
    static bool initialized = false;
    if (!initialized) {
        InstructionFactory::initialize();
        ResourceManager::instance();
        initialized = true;
    }
}

static int expected_divergence_sync(int lane, int p1_val) {
    return p1_val ? 0 : 1;
}

TEST_CASE("test_divergence_sync: Structure verification", "[divergence_sync][structure]") {
    init_execution_environment();
    auto stmts = build_divergence_sync_statements();

    INFO("Statement count: " << stmts.size());
    REQUIRE(stmts.size() == 12);

    CHECK(stmts[0].type == S_MOV);
    CHECK(stmts[1].type == S_SETP);
    CHECK(stmts[5].type == S_SELP);
    CHECK(stmts[6].type == S_BAR_WARP_SYNC);
    CHECK(stmts[11].type == S_RET);
}

TEST_CASE("test_divergence_sync: Handler registration", "[divergence_sync][handlers]") {
    init_execution_environment();

    REQUIRE(InstructionFactory::get_handler(S_MOV) != nullptr);
    REQUIRE(InstructionFactory::get_handler(S_SETP) != nullptr);
    REQUIRE(InstructionFactory::get_handler(S_SELP) != nullptr);
    REQUIRE(InstructionFactory::get_handler(S_ADD) != nullptr);
    REQUIRE(InstructionFactory::get_handler(S_MUL) != nullptr);
    REQUIRE(InstructionFactory::get_handler(S_ST) != nullptr);
    REQUIRE(InstructionFactory::get_handler(S_BAR) != nullptr);
    REQUIRE(InstructionFactory::get_handler(S_RET) != nullptr);
}

TEST_CASE("test_divergence_sync: Full warp execution with barrier", "[divergence_sync][execution][barrier]") {
    init_execution_environment();

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {8, 1, 1};

    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    auto statements = build_divergence_sync_statements();

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(statements);
    register_bank->preallocate_registers(registers);

    WarpContext warp;
    for (int lane = 0; lane < 8; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements,
                     &name2Sym, label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread->set_register_bank_manager(register_bank);
        warp.add_thread(std::move(thread), lane);
    }
    warp.set_active_mask(0x000000FFu);

    INFO("Warp active threads: " << warp.get_active_count());
    CHECK(warp.get_active_count() == 8);

    for (int pc = 0; pc < (int)statements.size(); pc++) {
        auto& stmt = statements[pc];
        INFO("PC=" << pc << " executing: " << stmt.instructionText);
        warp.execute_warp_instruction(stmt, pc);
    }

    INFO("\n=== Verification ===");
    for (int lane = 0; lane < 8; lane++) {
        auto* thread = warp.get_thread(lane);
        REQUIRE(thread != nullptr);

        void* r1_ptr = register_bank->get_register("r1", 0, lane);
        void* r4_ptr = register_bank->get_register("r4", 0, lane);
        void* r5_ptr = register_bank->get_register("r5", 0, lane);
        void* r6_ptr = register_bank->get_register("r6", 0, lane);
        void* p1_ptr = register_bank->get_register("p1", 0, lane);

        REQUIRE(r1_ptr != nullptr);
        REQUIRE(r4_ptr != nullptr);

        int r1_val = *static_cast<int*>(r1_ptr);
        int r4_val = *static_cast<int*>(r4_ptr);
        int r5_val = r5_ptr ? *static_cast<int*>(r5_ptr) : -999;
        int r6_val = r6_ptr ? *static_cast<int*>(r6_ptr) : -999;
        int p1_val = p1_ptr ? *static_cast<int*>(p1_ptr) : -999;
        int expected = expected_divergence_sync(lane, p1_val);

        INFO("Lane " << lane << ": r1=" << r1_val << " r4=" << r4_val
                      << " r5=" << r5_val << " r6=" << r6_val << " p1=" << p1_val
                      << " expected=" << expected);

        CHECK(r4_val == expected);
    }
}

TEST_CASE("test_divergence_sync: Register analysis", "[divergence_sync][register-analysis]") {
    init_execution_environment();
    auto statements = build_divergence_sync_statements();

    auto registers = RegisterAnalyzer::analyze_registers(statements);

    INFO("Analyzed " << registers.size() << " registers:");
    for (const auto& reg : registers) {
        INFO("  " << reg.name << " (size=" << reg.size << ")");
    }

    CHECK(registers.size() >= 8);
}