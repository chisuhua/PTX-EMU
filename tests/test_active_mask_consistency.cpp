#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/execution_types.h"
#include "ptx_ir/statement_context.h"

#include <memory>

using namespace ptxsim;

namespace {
StatementContext make_nop_stmt() {
    StatementContext stmt;
    stmt.type = S_MOV;
    GenericInstr instr;
    stmt.data = instr;
    return stmt;
}

void add_thread(WarpContext& warp, int lane, bool is_exited = false) {
    auto thread = std::make_unique<ThreadContext>();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {(uint32_t)lane, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts;
    stmts.push_back(make_nop_stmt());
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                 label2pc, nullptr, nullptr);
    thread->set_state(is_exited ? EXIT : RUN);
    warp.add_thread(std::move(thread), lane);
}

void init_full_warp(WarpContext& warp) {
    for (int i = 0; i < 32; i++) {
        add_thread(warp, i);
    }
}
} // namespace

TEST_CASE("J1: default active_mask matches exec_mask", "[active_mask]") {
    WarpContext warp;
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("J2: active_mask unchanged during divergence", "[active_mask]") {
    WarpContext warp;
    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J3: thread exit updates active_mask", "[active_mask]") {
    WarpContext warp;
    init_full_warp(warp);

    // 退出前8个线程
    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
        warp.get_thread(i)->set_state(EXIT);
    }
    warp.update_active_mask();

    uint32_t mask = warp.get_active_mask();
    REQUIRE((mask & 0x000000FF) == 0);
    REQUIRE((mask & 0xFFFFFF00) == 0xFFFFFF00);
}

TEST_CASE("J4: active_mask consistent after convergence", "[active_mask]") {
    WarpContext warp;
    SIMTStackEntry entry;
    entry.branch_pc = 10; entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF; entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J5: active_count matches active_mask bits", "[active_mask]") {
    WarpContext warp;
    init_full_warp(warp);

    // 前16个线程退出
    for (int i = 0; i < 16; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
        warp.get_thread(i)->set_state(EXIT);
    }
    // 接下来8个阻塞在屏障
    for (int i = 16; i < 24; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    warp.update_active_mask();

    // 还剩24-31共8个活跃线程
    REQUIRE(warp.get_active_count() == 8);
}
