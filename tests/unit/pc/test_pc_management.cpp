#include "catch_amalgamated.hpp"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_scheduler.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"

#include <iostream>
#include <memory>
#include <vector>

using namespace ptxsim;

static StatementContext make_nop_stmt() {
    StatementContext stmt;
    stmt.type = S_MOV;
    GenericInstr instr;
    stmt.data = instr;
    return stmt;
}

static void init_warp_with_threads(WarpContext& warp, int num_threads = 32) {
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};

    std::vector<StatementContext> statements;
    statements.push_back(make_nop_stmt());
    statements.push_back(make_nop_stmt());
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    for (int i = 0; i < num_threads; i++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)i, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        warp.add_thread(std::move(thread), i);
    }
}

TEST_CASE("is_warp_ready_to_fetch: returns true when all threads committed", "[pc][scheduler]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}

TEST_CASE("is_warp_ready_to_fetch: returns false when threads have divergent PC", "[pc][scheduler]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 15;

    REQUIRE(warp.is_warp_ready_to_fetch() == false);
}

TEST_CASE("is_warp_ready_to_fetch: returns false when thread has pc != next_pc", "[pc][scheduler]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 11;

    REQUIRE(warp.is_warp_ready_to_fetch() == false);
}

TEST_CASE("is_warp_ready_to_fetch: inactive threads are skipped", "[pc][scheduler]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 11;
    warp.get_warp_state().threads[0].is_active = false;

    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}

TEST_CASE("force_set_pc: sets pc only, preserves next_pc", "[pc]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 15;

    warp.get_thread(0)->force_set_pc(20);

    REQUIRE(warp.get_warp_state().threads[0].pc == 20);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 15);
}

TEST_CASE("set_thread_pc: sets both pc and next_pc", "[pc]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 15;

    warp.set_thread_pc(0, 20);

    REQUIRE(warp.get_warp_state().threads[0].pc == 20);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 20);
}

TEST_CASE("commit_pc: advances pc to next_pc", "[pc]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 10;
    warp.get_warp_state().threads[0].next_pc = 15;

    warp.get_thread(0)->commit_pc();

    REQUIRE(warp.get_warp_state().threads[0].pc == 15);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 15);
}

TEST_CASE("RoundRobin scheduler skips warp not ready to fetch", "[pc][scheduler]") {
    RoundRobinWarpScheduler scheduler;

    auto warp1 = std::make_unique<WarpContext>();
    auto warp2 = std::make_unique<WarpContext>();
    warp1->set_warp_id(0);
    warp2->set_warp_id(1);

    init_warp_with_threads(*warp1);
    init_warp_with_threads(*warp2);

    warp1->get_warp_state().threads[0].pc = 10;
    warp1->get_warp_state().threads[0].next_pc = 11;

    scheduler.add_warp(warp1.get());
    scheduler.add_warp(warp2.get());

    WarpContext* scheduled = scheduler.schedule_next();
    REQUIRE(scheduled != nullptr);
    REQUIRE(scheduled->get_warp_id() == 1);
}

TEST_CASE("Greedy scheduler skips warp not ready to fetch", "[pc][scheduler]") {
    GreedyWarpScheduler scheduler;

    auto warp1 = std::make_unique<WarpContext>();
    auto warp2 = std::make_unique<WarpContext>();
    warp1->set_warp_id(0);
    warp2->set_warp_id(1);

    init_warp_with_threads(*warp1);
    init_warp_with_threads(*warp2);

    warp1->get_warp_state().threads[0].pc = 10;
    warp1->get_warp_state().threads[0].next_pc = 11;

    scheduler.add_warp(warp1.get());
    scheduler.add_warp(warp2.get());

    WarpContext* scheduled = scheduler.schedule_next();
    REQUIRE(scheduled != nullptr);
    REQUIRE(scheduled->get_warp_id() == 1);
}

TEST_CASE("force_set_pc + set_next_pc + commit_pc: barrier current thread flow", "[pc][barrier][regression]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 5;
    warp.get_warp_state().threads[0].next_pc = 6;

    warp.get_thread(0)->force_set_pc(10);
    warp.get_thread(0)->set_next_pc(10);

    REQUIRE(warp.get_warp_state().threads[0].pc == 10);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 10);

    warp.get_thread(0)->commit_pc();

    REQUIRE(warp.get_warp_state().threads[0].pc == 10);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 10);
    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}

TEST_CASE("Normal instruction: set_next_pc(pc+1) then commit_pc advances correctly", "[pc]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    warp.get_warp_state().threads[0].pc = 5;
    warp.get_warp_state().threads[0].next_pc = 5;

    warp.get_thread(0)->set_next_pc(6);
    REQUIRE(warp.get_warp_state().threads[0].pc == 5);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 6);
    REQUIRE(warp.is_warp_ready_to_fetch() == false);

    warp.get_thread(0)->commit_pc();
    REQUIRE(warp.get_warp_state().threads[0].pc == 6);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 6);
    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}

TEST_CASE("Divergent branch: threads converge to correct reconvergence PC", "[pc][branch][regression]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    const int TARGET_PC = 20;
    const int RECONV_PC = 10;

    warp.get_warp_state().exec_mask = 0xFFFF0000;

    for (int i = 0; i < 16; i++) {
        warp.get_warp_state().threads[i].next_pc = TARGET_PC;
    }
    for (int i = 16; i < 32; i++) {
        warp.get_warp_state().threads[i].next_pc = RECONV_PC;
    }

    for (int i = 0; i < 32; i++) {
        warp.get_thread(i)->commit_pc();
    }

    REQUIRE(warp.get_warp_state().threads[0].pc == TARGET_PC);
    REQUIRE(warp.get_warp_state().threads[16].pc == RECONV_PC);
    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}

TEST_CASE("Barrier completion: all arrived threads set to reconvergence PC", "[pc][barrier][regression]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    const int RECONV_PC = 26;

    warp.get_warp_state().exec_mask = 0xFFFFFFFF;

    warp.get_thread(0)->force_set_pc(RECONV_PC);
    warp.get_thread(0)->set_next_pc(RECONV_PC);
    warp.set_thread_pc(0, RECONV_PC);

    warp.get_thread(1)->force_set_pc(RECONV_PC);
    warp.get_thread(1)->set_next_pc(RECONV_PC);
    warp.set_thread_pc(1, RECONV_PC);

    REQUIRE(warp.get_warp_state().threads[0].pc == RECONV_PC);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == RECONV_PC);
    REQUIRE(warp.get_warp_state().threads[1].pc == RECONV_PC);
    REQUIRE(warp.get_warp_state().threads[1].next_pc == RECONV_PC);
    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}

TEST_CASE("Branch instruction: commit_pc advances to target PC after handle_branch", "[pc][branch]") {
    WarpContext warp;
    init_warp_with_threads(warp);

    const int TARGET_PC = 20;
    const int RECONV_PC = 10;

    warp.get_warp_state().exec_mask = 0xFFFFFFFF;

    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].next_pc = TARGET_PC;
    }

    for (int i = 0; i < 32; i++) {
        warp.get_thread(i)->commit_pc();
    }
    REQUIRE(warp.get_warp_state().threads[0].pc == TARGET_PC);
    REQUIRE(warp.is_warp_ready_to_fetch() == true);
}