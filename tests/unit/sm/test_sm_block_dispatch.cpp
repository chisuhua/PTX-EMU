/**
 * PTX-6 TDD: unit tests for sm_block_dispatch::Access helper namespace.
 * Covers CTA admission / pending queue / cleanup preservation.
 * Assertions mirror the verified invariants of test_streaming_admission.cpp.
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/core/sm_block_dispatch.h"
#include "memory/resource_manager.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

// Same construction helper as test_streaming_admission.cpp. stmts is
// passed by reference because CTAContext::init() stores a raw pointer
// to the vector; the caller must keep it alive for the lifetime of the
// CTAContext (see test_sm_warp_lifecycle.cpp for the same lifetime
// contract).
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

TEST_CASE("sm_block_dispatch::Access::add_block admits a fresh CTA",
          "[unit][sm][block_dispatch]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0, stmts);
    REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)) == true);
    REQUIRE(sm.get_admitted_block_count() == 1);
    REQUIRE(sm.get_pending_block_count() == 0);
    REQUIRE(sm.get_total_block_count() == 1);
}

TEST_CASE("sm_block_dispatch::Access::add_block overflow → pending",
          "[unit][sm][block_dispatch]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    // 2-warp SM, 1-warp blocks → 2 fit, 2 must queue.
    for (int i = 0; i < 4; i++) {
        Dim3 idx = {static_cast<uint32_t>(i), 0, 0};
        auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0, stmts);
        REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)) == true);
    }
    REQUIRE(sm.get_admitted_block_count() == 2);
    REQUIRE(sm.get_pending_block_count() == 2);
    REQUIRE(sm.get_total_block_count() == 4);
}

TEST_CASE("sm_block_dispatch::Access::cleanup_finished_blocks preserves pending",
          "[unit][sm][block_dispatch]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    for (int i = 0; i < 4; i++) {
        Dim3 idx = {static_cast<uint32_t>(i), 0, 0};
        auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0, stmts);
        REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)));
    }
    REQUIRE(sm.get_pending_block_count() == 2);

    // Pending must NOT silently disappear when cleanup finds no finished warps.
    sm_block_dispatch::Access::cleanup_finished_blocks(sm);
    REQUIRE(sm.get_pending_block_count() == 2);
    REQUIRE(sm.get_total_block_count() == 4);
}

TEST_CASE("sm_block_dispatch::Access::add_block hard-rejects impossible blocks",
          "[unit][sm][block_dispatch][negative]") {
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);
    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    // 128 threads on a 2-warp SM can NEVER fit → hard reject, no pending dump.
    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/128, /*shared_mem=*/0, stmts);
    REQUIRE(sm_block_dispatch::Access::add_block(sm, std::move(block)) == false);
    REQUIRE(sm.get_total_block_count() == 0);
    REQUIRE(sm.get_admitted_block_count() == 0);
    REQUIRE(sm.get_pending_block_count() == 0);
}
