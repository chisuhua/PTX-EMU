#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"
#include <map>
#include <memory>
#include <vector>
#include <cstdint>

namespace {
using namespace ptxir::factory;

static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

const int PC_BRA_COND   = 4;
const int PATH_A_START   = 5;
const int PATH_A_END     = 13;
const int CONV_PC        = 14;
const int PATH_B_TARGET  = 28;
const int PATH_B_END     = 33;
const int BRA_UNI_PC     = 34;
const int RET_PC         = 27;

static std::vector<StatementContext> build_instrs() {
    std::vector<StatementContext> v;

    v.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 1}}, OperandContext{RegOperand{"tid.x", -1}}},
        "mov.u32 %r_tid, %tid.x;"));

    v.push_back(makeGenericInstr(S_SETP,
        {Qualifier::Q_B32, Qualifier::Q_LT},
        {OperandContext{RegOperand{"p", 1}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"16"}}},
        "setp.lt.u32 %p1, %r_tid, 16;"));

    v.push_back(makeBranchInstr(S_BRA, {}, "L_path_a", "%p1", false, "@%p1 bra L_path_a;"));

    v.push_back(makeBranchInstr(S_BRA, {}, "L_path_b", "", false, "bra L_path_b;"));

    v.push_back(makeLabelInstr("L_entry", "L_entry:"));

    v.push_back(makeLabelInstr("L_path_a", "L_path_a:"));

    v.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{RegOperand{"r", 1}}},
        "mov.u32 %r_val, %r_tid;"));

    v.push_back(makeGenericInstr(S_ADD,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 3}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"0"}}},
        "add.u32 %r_sum, %r_tid, 0;"));

    v.push_back(makeGenericInstr(S_MUL,
        {Qualifier::Q_U32},
        {OperandContext{RegOperand{"r", 4}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"2"}}},
        "mul.u32 %r_mul, %r_tid, 2;"));

    v.push_back(makeGenericInstr(S_SUB,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 5}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"1"}}},
        "sub.u32 %r_sub, %r_tid, 1;"));

    v.push_back(makeGenericInstr(S_AND,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 6}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"255"}}},
        "and.b32 %r_and, %r_tid, 0xFF;"));

    v.push_back(makeGenericInstr(S_OR,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 7}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"0"}}},
        "or.b32 %r_or, %r_tid, 0;"));

    v.push_back(makeGenericInstr(S_XOR,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 8}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"0"}}},
        "xor.b32 %r_xor, %r_tid, 0;"));

    v.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 9}}, OperandContext{RegOperand{"r", 1}}},
        "mov.u32 %r_tmp, %r_tid;"));

    v.push_back(makeLabelInstr("L_join", "L_join:"));

    v.push_back(makeGenericInstr(S_ST,
        {Qualifier::Q_SHARED, Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{RegOperand{"r", 1}}},
        "st.shared.u32 [%r_tid], %r_val;"));

    v.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 10}}, OperandContext{RegOperand{"r", 1}}},
        "mov.u32 %r_st, %r_tid;"));

    v.push_back(makeGenericInstr(S_ST,
        {Qualifier::Q_SHARED, Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 10}}, OperandContext{RegOperand{"r", 9}}},
        "st.shared.u32 [%r_st], %r_tmp;"));

    v.push_back(makeBarWarpSyncInstr(0xFFFFFFFF, 19));

    v.push_back(makeGenericInstr(S_SETP,
        {Qualifier::Q_B32, Qualifier::Q_EQ},
        {OperandContext{RegOperand{"p", 2}}, OperandContext{RegOperand{"r", 1}}, OperandContext{ImmOperand{"0"}}},
        "setp.eq.u32 %p_t0, %r_tid, 0;"));

    v.push_back(makeBranchInstr(S_BRA, {}, "L_reduce", "%p_t0", false, "@%p_t0 bra L_reduce;"));

    v.push_back(makeBranchInstr(S_BRA, {}, "L_exit", "", false, "bra L_exit;"));

    v.push_back(makeLabelInstr("L_reduce", "L_reduce:"));

    v.push_back(makeGenericInstr(S_LD,
        {Qualifier::Q_SHARED, Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 11}}, OperandContext{RegOperand{"r", 1}}},
        "ld.shared.u32 %r_ld, [%r_tid];"));

    v.push_back(makeGenericInstr(S_ADD,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 3}}, OperandContext{RegOperand{"r", 3}}, OperandContext{RegOperand{"r", 11}}},
        "add.u32 %r_sum, %r_sum, %r_ld;"));

    v.push_back(makeBranchInstr(S_BRA, {}, "L_exit", "", false, "bra L_exit;"));

    v.push_back(makeLabelInstr("L_exit", "L_exit:"));

    v.push_back(makeVoidInstr(S_RET, "ret;"));

    v.push_back(makeLabelInstr("L_path_b", "L_path_b:"));

    v.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 2}}, OperandContext{ImmOperand{"1"}}},
        "mov.u32 %r_val, 1;"));

    v.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 12}}, OperandContext{ImmOperand{"2"}}},
        "mov.u32 %r_b, 2;"));

    v.push_back(makeGenericInstr(S_ADD,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 13}}, OperandContext{RegOperand{"r", 12}}, OperandContext{ImmOperand{"3"}}},
        "add.u32 %r_c, %r_b, 3;"));

    v.push_back(makeGenericInstr(S_MUL,
        {Qualifier::Q_U32},
        {OperandContext{RegOperand{"r", 14}}, OperandContext{RegOperand{"r", 13}}, OperandContext{ImmOperand{"4"}}},
        "mul.u32 %r_d, %r_c, 4;"));

    v.push_back(makeGenericInstr(S_MOV,
        {Qualifier::Q_B32},
        {OperandContext{RegOperand{"r", 15}}, OperandContext{RegOperand{"r", 14}}},
        "mov.u32 %r_final, %r_d;"));

    v.push_back(makeBranchInstr(S_BRA, {}, "L_join", "", false, "bra.uni L_join;"));

    return v;
}

static WarpContext* create_warp_with_threads(
    SMContext& sm, std::unique_ptr<CTAContext> block,
    std::shared_ptr<RegisterBankManager> register_bank) {
    block->sharedMemBytes = 128;
    bool ok = sm.add_block(std::move(block));
    REQUIRE(ok == true);
    WarpContext* warp = sm.get_warp(0);
    warp->set_register_bank_manager(register_bank);
    for (int i = 0; i < 32; i++) {
        warp->get_thread(i)->set_register_bank_manager(register_bank);
    }
    return warp;
}

static std::map<std::string, int> build_label2pc(
    const std::vector<StatementContext>& stmts) {
    std::map<std::string, int> map;
    for (int i = 0; i < static_cast<int>(stmts.size()); i++) {
        if (stmts[i].type == S_LABEL) {
            const LabelInstr& li = std::get<LabelInstr>(stmts[i].data);
            map[li.labelName] = i;
        }
    }
    return map;
}

static std::unique_ptr<CTAContext> create_block(
    std::vector<StatementContext>& statements,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    Dim3 blockIdx = {0, 0, 0}) {
    auto block = std::make_unique<CTAContext>();
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc = build_label2pc(statements);
    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);
    return block;
}

static void advance_all_to_pc(WarpContext& warp, int pc) {
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = pc;
        warp.get_warp_state().threads[i].next_pc = pc;
    }
}

static int step_warp(WarpContext* w, std::vector<StatementContext>& stmts) {
    auto state = w->get_warp_state();
    int min_pc = 1024;
    int min_lane = -1;
    for (int i = 0; i < 32; i++) {
        int pc = state.threads[i].pc;
        if (pc < min_pc && w->is_lane_schedulable(i)) {
            min_pc = pc;
            min_lane = i;
        }
    }
    if (min_lane < 0) return -1;
    w->execute_warp_instruction(stmts[min_pc], min_pc);
    return min_pc;
}

}

TEST_CASE("divergence_sync_convergence: Path A blocks at reconvergence PC=14, scheduler switches to Path B", "[divergence_sync][convergence][test_a]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto v = build_instrs();
    REQUIRE(v.size() == 35);

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(v);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = create_warp_with_threads(sm, create_block(v), register_bank);
    w->set_active_mask(0xFFFFFFFF);

    for (int i = 0; i < 32; i++) {
        void* tid_addr = register_bank->get_register("r1", 0, i);
        if (tid_addr) {
            *static_cast<uint32_t*>(tid_addr) = i;
        }
    }

    for (int pc = 0; pc <= PC_BRA_COND; pc++) {
        w->execute_warp_instruction(v[pc], pc);
    }

    CHECK(w->get_thread_pc(0) == PATH_A_START);
    CHECK(w->get_thread_pc(16) == PATH_B_TARGET);

    for (int i = 0; i < 8; i++) {
        step_warp(w, v);
    }

    CHECK(w->get_thread_pc(16) == CONV_PC);
    CHECK(w->get_thread_pc(0) == PATH_B_TARGET);
    CHECK(w->check_reconvergence() == false);
    CHECK(w->get_simt_stack().depth() == 1);

    int pc;
    for (int i = 0; i < 6; i++) {
        pc = step_warp(w, v);
        CHECK(pc >= PATH_B_TARGET);
        CHECK(pc <= PATH_B_END);
    }

    pc = step_warp(w, v);
    CHECK(pc == BRA_UNI_PC);

    CHECK(w->get_simt_stack().empty());
    CHECK(w->get_exec_mask() == 0xFFFFFFFFu);

    for (int i = CONV_PC; i <= 27; i++) {
        pc = step_warp(w, v);
        CHECK(pc == i);
    }
}

TEST_CASE("divergence_sync_convergence: Path B executes first then reconverges", "[divergence_sync][convergence][test_b]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto v = build_instrs();
    REQUIRE(v.size() == 35);

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(v);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = create_warp_with_threads(sm, create_block(v), register_bank);
    w->set_active_mask(0xFFFFFFFF);

    for (int i = 0; i < 32; i++) {
        void* tid_addr = register_bank->get_register("r1", 0, i);
        if (tid_addr) {
            *static_cast<uint32_t*>(tid_addr) = i;
        }
    }

    for (int pc = 0; pc <= PC_BRA_COND; pc++) {
        w->execute_warp_instruction(v[pc], pc);
    }

    CHECK(w->get_thread_pc(0) == PATH_A_START);
    CHECK(w->get_thread_pc(16) == PATH_B_TARGET);

    for (int i = 0; i < 9; i++) {
        step_warp(w, v);
    }

    CHECK(w->get_thread_pc(16) == CONV_PC);
    CHECK(w->get_thread_pc(0) == PATH_B_TARGET);

    int pc;
    for (int i = 0; i < 6; i++) {
        pc = step_warp(w, v);
        CHECK(pc >= PATH_B_TARGET);
        CHECK(pc <= PATH_B_END);
    }

    pc = step_warp(w, v);
    CHECK(pc == BRA_UNI_PC);
    CHECK(w->get_simt_stack().empty());
    CHECK(w->get_exec_mask() == 0xFFFFFFFFu);

    for (int i = CONV_PC; i <= 27; i++) {
        pc = step_warp(w, v);
        CHECK(pc == i);
    }
}

TEST_CASE("divergence_sync_convergence: exec_mask restores to full warp after convergence", "[divergence_sync][convergence][test_c]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto v = build_instrs();
    REQUIRE(v.size() == 35);

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(v);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = create_warp_with_threads(sm, create_block(v), register_bank);
    w->set_active_mask(0xFFFFFFFF);

    for (int i = 0; i < 32; i++) {
        void* tid_addr = register_bank->get_register("r1", 0, i);
        if (tid_addr) {
            *static_cast<uint32_t*>(tid_addr) = i;
        }
    }

    for (int pc = 0; pc <= PC_BRA_COND; pc++) {
        w->execute_warp_instruction(v[pc], pc);
    }

    for (int i = 0; i < 10; i++) {
        step_warp(w, v);
    }

    CHECK(w->get_thread_pc(16) == CONV_PC);
    CHECK(w->get_thread_pc(0) == PATH_B_TARGET);

    int pc;
    for (int i = 0; i < 6; i++) {
        pc = step_warp(w, v);
        CHECK(pc >= PATH_B_TARGET);
        CHECK(pc <= PATH_B_END);
    }

    pc = step_warp(w, v);
    CHECK(pc == BRA_UNI_PC);

    CHECK(w->get_simt_stack().empty());
    CHECK(w->get_exec_mask() == 0xFFFFFFFFu);

    for (int i = CONV_PC; i <= 27; i++) {
        pc = step_warp(w, v);
        CHECK(pc == i);
    }
}

TEST_CASE("divergence_sync_convergence: SIMT stack empty after convergence", "[divergence_sync][convergence][test_d]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto v = build_instrs();
    REQUIRE(v.size() == 35);

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(v);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = create_warp_with_threads(sm, create_block(v), register_bank);
    w->set_active_mask(0xFFFFFFFF);

    for (int i = 0; i < 32; i++) {
        void* tid_addr = register_bank->get_register("r1", 0, i);
        if (tid_addr) {
            *static_cast<uint32_t*>(tid_addr) = i;
        }
    }

    for (int pc = 0; pc <= PC_BRA_COND; pc++) {
        w->execute_warp_instruction(v[pc], pc);
    }

    for (int i = 0; i < 10; i++) {
        step_warp(w, v);
    }

    CHECK(w->get_simt_stack().depth() == 1);

    int pc;
    for (int i = 0; i < 6; i++) {
        pc = step_warp(w, v);
        CHECK(pc >= PATH_B_TARGET);
        CHECK(pc <= PATH_B_END);
    }

    pc = step_warp(w, v);
    CHECK(pc == BRA_UNI_PC);

    CHECK(w->get_simt_stack().empty());

    for (int i = CONV_PC; i <= 27; i++) {
        pc = step_warp(w, v);
        CHECK(pc == i);
    }
}

TEST_CASE("divergence_sync_convergence: unified execution phase correct PC progression", "[divergence_sync][convergence][test_e]") {
    init_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    auto v = build_instrs();
    REQUIRE(v.size() == 35);

    auto register_bank = std::make_shared<RegisterBankManager>(1, 32);
    auto registers = RegisterAnalyzer::analyze_registers(v);
    register_bank->preallocate_registers(registers);

    SMContext sm(4, 128, 4096, 0);
    WarpContext* w = create_warp_with_threads(sm, create_block(v), register_bank);
    w->set_active_mask(0xFFFFFFFF);

    for (int i = 0; i < 32; i++) {
        void* tid_addr = register_bank->get_register("r1", 0, i);
        if (tid_addr) {
            *static_cast<uint32_t*>(tid_addr) = i;
        }
    }

    for (int pc = 0; pc <= PC_BRA_COND; pc++) {
        w->execute_warp_instruction(v[pc], pc);
    }

    for (int i = 0; i < 10; i++) {
        step_warp(w, v);
    }

    int pc;
    for (int i = 0; i < 6; i++) {
        pc = step_warp(w, v);
        CHECK(pc >= PATH_B_TARGET);
        CHECK(pc <= PATH_B_END);
    }

    pc = step_warp(w, v);
    CHECK(pc == BRA_UNI_PC);
    CHECK(w->get_simt_stack().empty());

    pc = step_warp(w, v);
    CHECK(pc == 14);
    pc = step_warp(w, v);
    CHECK(pc == 15);
    pc = step_warp(w, v);
    CHECK(pc == 16);
    pc = step_warp(w, v);
    CHECK(pc == 17);
    pc = step_warp(w, v);
    CHECK(pc == 18);
    pc = step_warp(w, v);
    CHECK(pc == 19);
    pc = step_warp(w, v);
    CHECK(pc == 20);
    pc = step_warp(w, v);
    CHECK(pc == 21);
    pc = step_warp(w, v);
    CHECK(pc == 22);
    pc = step_warp(w, v);
    CHECK(pc == 23);
    pc = step_warp(w, v);
    CHECK(pc == 24);
    pc = step_warp(w, v);
    CHECK(pc == 25);
    pc = step_warp(w, v);
    CHECK(pc == 26);
    pc = step_warp(w, v);
    CHECK(pc == 27);
}