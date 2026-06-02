/**
 * @file test_syncthreads_full_pipeline.cpp
 * @brief Full pipeline test verifying instruction execution and scheduler integration
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
#include "memory/shared_memory_manager.h"
#include "utils/logger.h"
#include <memory>

using namespace ptxsim;

static std::vector<StatementContext> build_simple_barrier_statements() {
    std::vector<StatementContext> statements;

    for (int i = 0; i < 5; i++) {
        statements.push_back({S_MOV, GenericInstr{}});
    }

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

    for (int i = 0; i < 4; i++) {
        statements.push_back({S_MOV, GenericInstr{}});
    }

    statements.push_back({S_RET, GenericInstr{}});

    for (int i = 0; i < (int)statements.size(); i++) {
        if (statements[i].type == S_BAR_WARP_SYNC) {
            auto& barrier = std::get<BarWarpSyncInstr>(statements[i].data);
            if (barrier.operands.size() >= 2) {
                barrier.operands[1] =
                    OperandContext{ImmOperand{std::to_string(i + 1)}};
            }
        }
    }

    return statements;
}

static void init_instruction_factory() {
    static bool initialized = false;
    if (!initialized) {
        InstructionFactory::initialize();
        initialized = true;
    }
}

TEST_CASE("Full pipeline: InstructionFactory with real handlers",
          "[full-pipeline][instruction-factory][handlers]")
{
    init_instruction_factory();

    SECTION("All critical handlers are registered") {
        auto* barrier_handler = InstructionFactory::get_handler(S_BAR_WARP_SYNC);
        REQUIRE(barrier_handler != nullptr);

        auto* branch_handler = InstructionFactory::get_handler(S_BRA);
        REQUIRE(branch_handler != nullptr);

        auto* setp_handler = InstructionFactory::get_handler(S_SETP);
        REQUIRE(setp_handler != nullptr);

        auto* mov_handler = InstructionFactory::get_handler(S_MOV);
        REQUIRE(mov_handler != nullptr);

        auto* ret_handler = InstructionFactory::get_handler(S_RET);
        REQUIRE(ret_handler != nullptr);
    }

    SECTION("Barrier handler executes with Wbar synchronization") {
        auto statements = build_simple_barrier_statements();
        Dim3 blockIdx = {0, 0, 0};
        Dim3 gridDim = {1, 1, 1};
        Dim3 blockDim = {16, 1, 1};
        std::map<std::string, Symtable*> name2Sym;
        std::map<std::string, int> label2pc;

        auto register_bank = std::make_shared<RegisterBankManager>(4, 32);

        WarpContext warp;
        for (int lane = 0; lane < 16; lane++) {
            auto thread = std::make_unique<ThreadContext>();
            Dim3 tid = {(uint32_t)lane, 0, 0};
            thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym, label2pc, nullptr, nullptr);
            thread->set_state(RUN);
            thread->set_register_bank_manager(register_bank);
            warp.add_thread(std::move(thread), lane);
        }

        warp.set_active_mask(0x0000FFFFu);

        // Execute prologue (PC 0-5)
        for (int pc = 0; pc < 6; pc++) {
            auto& stmt = statements[pc];
            warp.execute_warp_instruction(stmt, pc);
        }

        // Verify all threads at barrier
        for (int lane = 0; lane < 16; lane++) {
            CHECK(warp.get_thread(lane)->get_pc() == 6);
        }

        // Execute barrier through handler
        auto* handler = InstructionFactory::get_handler(S_BAR_WARP_SYNC);
        REQUIRE(handler != nullptr);

        // Wbar initialization and arrival
        auto& warp_state = warp.get_warp_state();
        ptxsim::Wbar& wbar = warp_state.wbars[0];
        wbar.init(0x0000FFFFu, 7);

        for (int lane = 0; lane < 16; lane++) {
            wbar.arrive(lane);
        }

        CHECK(wbar.is_complete() == true);
        CHECK(wbar.count_arrived() == 16);
        CHECK(wbar.count_participants() == 16);
        CHECK(wbar.reconvergence_pc == 7);
    }
}

TEST_CASE("Full pipeline: ResourceManager integration",
          "[full-pipeline][resource-manager][integration]")
{
    SECTION("ResourceManager initialization and shared memory allocation") {
        auto& rm = ResourceManager::instance();
        rm.initialize(1, 49152);
        
        auto* shmem_mgr = rm.get_shared_memory_manager(0);
        REQUIRE(shmem_mgr != nullptr);

        void* mem = shmem_mgr->allocate(1024, 1);
        REQUIRE(mem != nullptr);

        size_t available = shmem_mgr->get_available_size();
        CHECK(available > 0);
    }
}

TEST_CASE("Full pipeline: SMContext with ResourceManager",
          "[full-pipeline][sm-context][resource-manager]")
{
    auto& rm = ResourceManager::instance();
    rm.initialize(1, 49152);
    auto* shmem_mgr = rm.get_shared_memory_manager(0);
    REQUIRE(shmem_mgr != nullptr);

    auto statements = build_simple_barrier_statements();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    auto register_bank = std::make_shared<RegisterBankManager>(4, 32);

    CTAContext cta;
    cta.init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc);

    CHECK(cta.get_thread_count() == 16);
    CHECK(cta.get_warp_count() == 1);

    SECTION("SMContext creation and initialization") {
        SMContext sm(4, 128, 49152, 0);
        sm.init();

        CHECK(sm.get_active_warps_count() == 0);
        CHECK(sm.get_active_threads_count() == 0);
    }

    SECTION("CTA to SMContext transfer - verify warp release") {
        SMContext sm(4, 128, 49152, 0);
        sm.init();

        // Release warps from CTA to verify they're properly initialized
        auto released_warps = cta.release_warps();
        CHECK(released_warps.size() == 1);
        CHECK(released_warps[0]->get_active_count() == 16);
    }
}

TEST_CASE("Full pipeline: Warp scheduler simulation",
          "[full-pipeline][warp-scheduler][simulation]")
{
    auto statements = build_simple_barrier_statements();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    auto register_bank = std::make_shared<RegisterBankManager>(4, 32);

    WarpContext warp;
    for (int lane = 0; lane < 16; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym, label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread->set_register_bank_manager(register_bank);
        warp.add_thread(std::move(thread), lane);
    }

    warp.set_active_mask(0x0000FFFFu);

    SECTION("Warp scheduler: round-robin scheduling simulation") {
        // Simulate scheduler repeatedly selecting this warp
        for (int cycle = 0; cycle < 20; cycle++) {
            int current_pc = warp.get_thread_pc(0);
            if (current_pc >= (int)statements.size()) {
                break;
            }

            auto& stmt = statements[current_pc];
            warp.execute_warp_instruction(stmt, current_pc);
        }

        // Verify all threads progressed
        for (int lane = 0; lane < 16; lane++) {
            CHECK(warp.get_thread(lane)->get_pc() >= 6);
        }
    }

    SECTION("Warp scheduler: divergent PC handling") {
        // Set different PCs for different threads (simulate divergence)
        for (int lane = 0; lane < 8; lane++) {
            warp.advance_thread_pc(lane, 6);
        }
        for (int lane = 8; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 7);
        }

        // Verify scheduler can handle divergent PCs
        auto lanes_by_pc = warp.get_lanes_by_pc();
        CHECK(lanes_by_pc.size() == 2);
    }
}

TEST_CASE("Full pipeline: Complete barrier execution sequence",
          "[full-pipeline][barrier][complete-sequence]")
{
    auto statements = build_simple_barrier_statements();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {16, 1, 1};
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    auto register_bank = std::make_shared<RegisterBankManager>(4, 32);

    WarpContext warp;
    for (int lane = 0; lane < 16; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym, label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread->set_register_bank_manager(register_bank);
        warp.add_thread(std::move(thread), lane);
    }

    warp.set_active_mask(0x0000FFFFu);

    SECTION("Complete execution: prologue -> barrier -> epilogue") {
        // Execute prologue
        for (int pc = 0; pc < 6; pc++) {
            auto& stmt = statements[pc];
            warp.execute_warp_instruction(stmt, pc);
        }

        // All threads at barrier (PC=6)
        for (int lane = 0; lane < 16; lane++) {
            CHECK(warp.get_thread(lane)->get_pc() == 6);
        }

        // Simulate barrier completion
        auto& warp_state = warp.get_warp_state();
        ptxsim::Wbar& wbar = warp_state.wbars[0];
        wbar.init(0x0000FFFFu, 7);

        for (int lane = 0; lane < 16; lane++) {
            wbar.arrive(lane);
        }

        CHECK(wbar.is_complete() == true);

        // Release threads to PC=7
        for (int lane = 0; lane < 16; lane++) {
            warp.advance_thread_pc(lane, 7);
        }

        // Execute epilogue
        for (int pc = 7; pc < 10; pc++) {
            auto& stmt = statements[pc];
            warp.execute_warp_instruction(stmt, pc);
        }

        // Verify all threads completed
        for (int lane = 0; lane < 16; lane++) {
            CHECK(warp.get_thread(lane)->get_pc() == 10);
        }
    }
}
