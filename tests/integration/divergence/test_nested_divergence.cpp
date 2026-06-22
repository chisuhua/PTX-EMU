/**
 * @file test_nested_divergence.cpp
 * @brief Isolated test for nested divergence with SETP/SELP using real execute_warp_instruction
 * @date 2026-05-08
 */
/**
 * @brief 测试嵌套谓词化（setp + selp），非真实 SIMT 嵌套分歧
 * @note  本文件使用两级 setp/selp 谓词选择，不涉及 @%p bra 分支和 SIMT stack
 */

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

static int expected_nested_divergence(int tid) {
    int base = tid;
    int first_branch = (tid % 2 == 0) ? 100 : 200;
    int second_branch = (tid % 4 < 2) ? 10 : 20;
    return base + first_branch + second_branch;
}

static std::vector<StatementContext> build_nested_divergence_statements() {
    std::vector<StatementContext> stmts;
    std::map<std::string, int> label2pc;

    stmts.push_back(makeGenericInstr(S_MOV, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 1}}, OperandContext{RegOperand{"tid.x", -1}}},
        "mov.u32 %r1, %tid.x;"));

    stmts.push_back(makeGenericInstr(S_AND, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"1"}}},
        "and.b32 %r2, %r1, 1;"));

    stmts.push_back(makeGenericInstr(S_SETP, {Qualifier::Q_B32, Qualifier::Q_EQ},
        {OperandContext{RegOperand{"p", 1}}, OperandContext{RegOperand{"r", 2}}, OperandContext{ImmOperand{"1"}}},
        "setp.eq.b32 %p1, %r2, 1;"));

    stmts.push_back(makeGenericInstr(S_SELP, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 3}}, OperandContext{ImmOperand{"200"}}, OperandContext{ImmOperand{"100"}}, OperandContext{RegOperand{"p", 1}}},
        "selp.b32 %r3, 200, 100, %p1;"));

    stmts.push_back(makeGenericInstr(S_ADD, {Qualifier::Q_S32},
        {OperandContext{RegOperand{"r", 4}}, OperandContext{RegOperand{"r", 3}}, OperandContext{RegOperand{"r", 1}}},
        "add.s32 %r4, %r3, %r1;"));

    stmts.push_back(makeGenericInstr(S_AND, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 5}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"2"}}},
        "and.b32 %r5, %r1, 2;"));

    stmts.push_back(makeGenericInstr(S_SETP, {Qualifier::Q_B32, Qualifier::Q_EQ},
        {OperandContext{RegOperand{"p", 2}}, OperandContext{RegOperand{"r", 5}}, OperandContext{ImmOperand{"0"}}},
        "setp.eq.s32 %p2, %r5, 0;"));

    stmts.push_back(makeGenericInstr(S_SELP, {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 6}}, OperandContext{ImmOperand{"10"}}, OperandContext{ImmOperand{"20"}}, OperandContext{RegOperand{"p", 2}}},
        "selp.b32 %r6, 10, 20, %p2;"));

    stmts.push_back(makeGenericInstr(S_ADD, {Qualifier::Q_S32},
        {OperandContext{RegOperand{"r", 7}}, OperandContext{RegOperand{"r", 4}}, OperandContext{RegOperand{"r", 6}}},
        "add.s32 %r7, %r4, %r6;"));

    stmts.push_back(makeGenericInstr(S_MUL, {Qualifier::Q_U32, Qualifier::Q_WIDE},
        {OperandContext{RegOperand{"rd", 3}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"4"}}},
        "mul.wide.u32 %rd3, %r1, 4;"));

    stmts.push_back(makeGenericInstr(S_ADD, {Qualifier::Q_S64},
        {OperandContext{RegOperand{"rd", 4}}, OperandContext{RegOperand{"rd", 2}}, OperandContext{RegOperand{"rd", 3}}},
        "add.s64 %rd4, %rd2, %rd3;"));

    stmts.push_back(makeGenericInstr(S_ST, {Qualifier::Q_GLOBAL, Qualifier::Q_B32},
        {OperandContext{AddrOperand{}}, OperandContext{RegOperand{"r", 7}}},
        "st.global.u32 [%rd4], %r7;"));

    stmts.push_back(makeVoidInstr(S_RET, "ret;"));

    return stmts;
}

static void init_execution_environment() {
    static bool initialized = false;
    if (!initialized) {
        InstructionFactory::initialize();
        auto& rm = ResourceManager::instance();
        initialized = true;
    }
}

// TODO: 添加真正的嵌套分歧测试（两级 @%p bra 推送 SIMT stack entry）

TEST_CASE("test_nested_predication: Full warp execution with nested setp+selp", "[nested_divergence][execution][execute_warp]") {
    init_execution_environment();

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {8, 1, 1};

    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;

    auto statements = build_nested_divergence_statements();

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
        step_warp(&warp, statements);
    }

    INFO("\n=== Verification ===");
    for (int lane = 0; lane < 8; lane++) {
        auto* thread = warp.get_thread(lane);
        REQUIRE(thread != nullptr);

        void* r1_ptr = register_bank->get_register("r1", 0, lane);
        void* r2_ptr = register_bank->get_register("r2", 0, lane);
        void* r3_ptr = register_bank->get_register("r3", 0, lane);
        void* r4_ptr = register_bank->get_register("r4", 0, lane);
        void* r5_ptr = register_bank->get_register("r5", 0, lane);
        void* r6_ptr = register_bank->get_register("r6", 0, lane);
        void* r7_ptr = register_bank->get_register("r7", 0, lane);
        void* p1_ptr = register_bank->get_register("p1", 0, lane);
        void* p2_ptr = register_bank->get_register("p2", 0, lane);

        REQUIRE(r1_ptr != nullptr);
        REQUIRE(r2_ptr != nullptr);
        REQUIRE(r3_ptr != nullptr);
        REQUIRE(r4_ptr != nullptr);
        REQUIRE(r5_ptr != nullptr);
        REQUIRE(r6_ptr != nullptr);
        REQUIRE(r7_ptr != nullptr);

        int r1_val = *static_cast<int*>(r1_ptr);
        int r2_val = *static_cast<int*>(r2_ptr);
        int r3_val = *static_cast<int*>(r3_ptr);
        int r4_val = *static_cast<int*>(r4_ptr);
        int r5_val = *static_cast<int*>(r5_ptr);
        int r6_val = *static_cast<int*>(r6_ptr);
        int r7_val = *static_cast<int*>(r7_ptr);
        int p1_val = p1_ptr ? *static_cast<int*>(p1_ptr) : -1;
        int p2_val = p2_ptr ? *static_cast<int*>(p2_ptr) : -1;

        INFO("Lane " << lane << ": r1=" << r1_val << " r2=" << r2_val
                      << " r3=" << r3_val << " r4=" << r4_val
                      << " r5=" << r5_val << " r6=" << r6_val << " r7=" << r7_val
                      << " p1=" << p1_val << " p2=" << p2_val);

        int expected = expected_nested_divergence(lane);
        CHECK(r7_val == expected);
    }
}

