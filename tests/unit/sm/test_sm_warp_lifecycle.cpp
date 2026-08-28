/**
 * PTX-6 TDD: unit tests for sm_warp_lifecycle::Access helper namespace.
 * Ground truth: a fresh WarpContext has active_count=0 (ctor, warp_context.cpp:174),
 * so a plain add_block registers warps that are NOT yet active.
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/core/sm_warp_lifecycle.h"
#include "memory/resource_manager.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

// NOTE: stmts is passed by REFERENCE (not by value) because
// CTAContext::init() stores a raw pointer to the vector's internal buffer
// (see src/ptxsim/core/cta_context.cpp:43 — this->init_statements = &statements).
// The caller MUST keep the stmts vector alive for the lifetime of the
// CTAContext, otherwise add_block() → build_shared_memory_symbol_table()
// will dereference a dangling pointer (manifests as Debug-build SIGSEGV
// at cta_context.cpp:268). Pattern mirrors setup_two_warps() in
// tests/integration/barrier/test_barrier_full_lifecycle.cpp.
std::unique_ptr<CTAContext> make_block(Dim3 blockIdx, int threads,
                                        size_t shared_mem_bytes,
                                        std::vector<ptxemu::ir::StatementContext> &stmts) {
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {static_cast<uint32_t>(threads), 1, 1};
    auto block = std::make_unique<CTAContext>();
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc,
                nullptr, 0, 0);
    block->sharedMemBytes = shared_mem_bytes;
    return block;
}

}  // namespace

TEST_CASE("sm_warp_lifecycle::Access registers a new warp",
          "[unit][sm][warp_lifecycle]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0, stmts);
    REQUIRE(sm.add_block(std::move(block)) == true);  // public forwarder
    REQUIRE(sm.get_num_warps() == 1);
}

TEST_CASE("sm_warp_lifecycle::Access::get_active_warps_count matches public API after registration",
          "[unit][sm][warp_lifecycle]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0, stmts);
    REQUIRE(sm.add_block(std::move(block)) == true);

    // After add_block, warp_scheduler->add_warp + update_state activate the
    // warp. The helper must return the same count as the public forwarder
    // (byte-identical delegation lock). Ground truth (probe): count == 1.
    REQUIRE(sm_warp_lifecycle::Access::get_active_warps_count(sm) == 1);
    REQUIRE(sm_warp_lifecycle::Access::get_active_warps_count(sm) ==
            sm.get_active_warps_count());
    REQUIRE(sm_warp_lifecycle::Access::get_active_threads_count(sm) ==
            sm.get_active_threads_count());
}

TEST_CASE("sm_warp_lifecycle::Access::update_state transitions empty SM to EXIT",
          "[unit][sm][warp_lifecycle]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);

    // No warps, no managed blocks → EXIT (per update_state body at
    // sm_context.cpp:614-618: if (!has_active_warps && !has_managed_blocks)
    // sm_state = EXIT).
    sm_warp_lifecycle::Access::update_state(sm);
    REQUIRE(sm.get_state() == EXE_STATE::EXIT);
}
