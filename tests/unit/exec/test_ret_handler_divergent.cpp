// test_ret_handler_divergent.cpp
// Regression test for RetHandler::processOperation on divergent warps.
//
// Bug: When a divergent warp reaches `ret`, the RetHandler only marks the
// executing lane as exited. Inactive lanes (those stalled in a divergent
// path) keep state != EXIT, so WarpContext::is_finished() returns false
// and the warp scheduler loops forever.
//
// Fix: RetHandler must mark ALL lanes in the warp as exited when the kernel
// returns (call_stack is empty), so is_all_threads_exited() returns true.

#include "catch_amalgamated.hpp"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/execution_types.h"
#include "ptx_ir/statement_context.h"

#include <memory>

using namespace ptxsim;

namespace {

ptxemu::ir::StatementContext make_void_stmt() {
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_RET;
    GenericInstr instr;
    stmt.data = instr;
    return stmt;
}

void add_thread(WarpContext& warp, int lane) {
    auto thread = std::make_unique<ThreadContext>();
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {(uint32_t)lane, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<ptxemu::ir::StatementContext> stmts;
    stmts.push_back(make_void_stmt());
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    thread->init(blockIdx, threadIdx, gridDim, blockDim, stmts, &name2Sym,
                 label2pc, nullptr, nullptr);
    thread->set_state(RUN);
    warp.add_thread(std::move(thread), lane);
}

void simulate_divergence(WarpContext& warp, uint32_t active_mask) {
    warp.set_active_mask(active_mask);
    warp.update_active_mask();
    warp.set_exec_mask(active_mask);
}

}  // namespace

TEST_CASE("R1: ret on divergent warp marks all lanes exited",
          "[ret][divergent][regression]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        add_thread(warp, i);
    }
    simulate_divergence(warp, 0xFFFF0000u);

    REQUIRE_FALSE(warp.is_finished());
    REQUIRE(warp.get_active_count() == 16);

    ThreadContext* lane16 = warp.get_thread(16);
    REQUIRE(lane16 != nullptr);
    REQUIRE(warp.is_lane_active(16));

    RetHandler handler;
    ptxemu::ir::StatementContext stmt = make_void_stmt();
    handler.processOperation(lane16, stmt);

    CHECK(warp.is_finished());
    CHECK(warp.get_active_count() == 0);

    for (int i = 0; i < 32; i++) {
        CHECK(warp.get_thread(i)->is_exited());
        CHECK_FALSE(warp.is_lane_active(i));
    }
}

TEST_CASE("R2: ret on divergent warp with lower half active still finishes",
          "[ret][divergent][regression]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        add_thread(warp, i);
    }
    simulate_divergence(warp, 0x0000FFFFu);

    REQUIRE_FALSE(warp.is_finished());

    ThreadContext* lane0 = warp.get_thread(0);
    RetHandler handler;
    ptxemu::ir::StatementContext stmt = make_void_stmt();
    handler.processOperation(lane0, stmt);

    CHECK(warp.is_finished());
    CHECK(warp.get_active_count() == 0);
    for (int i = 0; i < 32; i++) {
        CHECK(warp.get_thread(i)->is_exited());
    }
}

TEST_CASE("R3: ret on uniform warp keeps previous behavior",
          "[ret][uniform]") {
    WarpContext warp;
    for (int i = 0; i < 32; i++) {
        add_thread(warp, i);
    }

    REQUIRE_FALSE(warp.is_finished());
    REQUIRE(warp.get_active_count() == 32);

    ThreadContext* lane5 = warp.get_thread(5);
    RetHandler handler;
    ptxemu::ir::StatementContext stmt = make_void_stmt();
    handler.processOperation(lane5, stmt);

    CHECK(warp.is_finished());
    CHECK(warp.get_active_count() == 0);
    for (int i = 0; i < 32; i++) {
        CHECK(warp.get_thread(i)->is_exited());
    }
}
