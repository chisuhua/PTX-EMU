#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"

#include <map>
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

void init_full_warp(WarpContext& warp) {
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> stmts;
    stmts.push_back(make_nop_stmt());
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    for (int i = 0; i < 32; i++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 threadIdx = {(uint32_t)i, 0, 0};
        thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                     label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        warp.add_thread(std::move(thread), i);
    }
}
} // namespace

static void setup_diverged_warp(WarpContext& warp) {
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    for (int i = 0; i < 16; i++) {
        warp.set_thread_pc(i, 20);
    }
    for (int i = 16; i < 32; i++) {
        warp.set_thread_pc(i, 11);
    }
    warp.set_exec_mask(0x0000FFFF);
}

TEST_CASE("F3: exec_mask restored after reconvergence", "[exec_mask][bug][critical]") {
    WarpContext warp;
    setup_diverged_warp(warp);

    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) {
        warp.set_thread_pc(i, 30);
    }

    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("F1: default exec_mask is full active", "[exec_mask]") {
    WarpContext warp;
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("F2: exec_mask after divergent branch", "[exec_mask][branch]") {
    WarpContext warp;
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);
}

TEST_CASE("F4: set_exec_mask and get_exec_mask roundtrip", "[exec_mask]") {
    WarpContext warp;
    warp.set_exec_mask(0x12345678);
    REQUIRE(warp.get_exec_mask() == 0x12345678);
    warp.set_exec_mask(0xAAAAAAAA);
    REQUIRE(warp.get_exec_mask() == 0xAAAAAAAA);
}

TEST_CASE("F5: nested divergence exec_mask recovery", "[exec_mask][nested]") {
    WarpContext warp;

    SIMTStackEntry outer;
    outer.branch_pc = 10;
    outer.reconvergence_pc = 50;
    outer.active_mask = 0x0000FFFF;
    // check_reconvergence reads return_mask from the NEW top (parent)
    // after popping, so outer.return_mask is what exec_mask will be
    // restored to when inner converges.
    outer.return_mask = 0x0000FFFF;
    outer.return_pc = 50;
    warp.get_simt_stack().push(outer);
    warp.set_exec_mask(0x0000FFFF);

    SIMTStackEntry inner;
    inner.branch_pc = 20;
    inner.reconvergence_pc = 40;
    inner.active_mask = 0x000000FF;
    inner.return_mask = 0x0000FFFF;
    inner.return_pc = 40;
    warp.get_simt_stack().push(inner);
    warp.set_exec_mask(0x000000FF);

    REQUIRE(warp.get_exec_mask() == 0x000000FF);
    REQUIRE(warp.get_simt_stack().depth() == 2);

    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 40);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().depth() == 1);
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);

    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 50);
    warp.check_reconvergence();

    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("F6: exec_mask and active_mask independence", "[exec_mask][concept]") {
    WarpContext warp;
    init_full_warp(warp);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}
