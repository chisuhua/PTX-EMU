// tests/unit/sm/test_streaming_admission.cpp
//
// Type 1 unit test: verifies SMContext::add_block streaming admission
// (BUG-SM-ADMISSION-OVERFLOW: blocks must not be silently dropped when
// an SM is full; overflow goes to a per-SM pending list that is
// refilled on cleanup_finished_blocks).
//
// Pre-fix behavior:
//   add_block returns false when SM is at warp limit → the kernel
//   launcher's outer loop silently drops the block (then re-iterates
//   all SMs N×108 times). For 1953 blocks on 108 SMs with 8 warps/SM
//   max, the 865th block cannot be admitted; 1089 blocks are dropped.
//   aligned-types then hangs in cudaDeviceSynchronize until ctest
//   60s defensive timeout fires.
//
// Post-fix behavior:
//   add_block returns true (block accepted into SM) even when the
//   block is queued in pending_blocks_ rather than admitted into
//   managed_blocks_. The total of (admitted + pending) equals the
//   number of attempted add_block calls.

#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"

#include "memory/resource_manager.h"

#include <map>
#include <memory>
#include <vector>

using namespace ptxsim;

namespace {

// Helper: construct a minimal CTAContext with `warp_count` warps and
// `shared_mem_bytes` shared-memory request. The statement list is
// empty — only resource accounting is exercised.
//
// stmts is passed by reference because CTAContext::init() stores a raw
// pointer to the vector's internal buffer; the caller must keep it alive
// for the lifetime of the CTAContext (see test_sm_warp_lifecycle.cpp for
// the same lifetime contract — pattern matches setup_two_warps in
// tests/integration/barrier/test_barrier_full_lifecycle.cpp).
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

TEST_CASE("SMContext::add_block queues overflow blocks in pending list",
          "[sm][admission][streaming][regression]") {
    // 1 SM with max 2 warps, 64 threads, 8KB shared mem
    // Per-block resource: 1 warp × 32 threads → 2 blocks fit max
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);

    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    constexpr int kTotalBlocks = 4;  // 2 fit, 2 must go to pending
    int admitted_or_pending = 0;
    for (int i = 0; i < kTotalBlocks; i++) {
        Dim3 idx = {static_cast<uint32_t>(i), 0, 0};
        auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0, stmts);
        bool accepted = sm.add_block(std::move(block));
        if (accepted) {
            admitted_or_pending++;
        }
    }

    // No block may be silently dropped.
    REQUIRE(admitted_or_pending == kTotalBlocks);
    REQUIRE(sm.get_total_block_count() == kTotalBlocks);

    // 2 fit (2 warps / 1 warp-per-block), 2 must be in pending.
    REQUIRE(sm.get_admitted_block_count() == 2);
    REQUIRE(sm.get_pending_block_count() == 2);
}

TEST_CASE("SMContext::add_block preserves pending across cleanup_finished_blocks",
          "[sm][admission][streaming][rebalance]") {
    // Same setup: 2-warp SM, 1-warp blocks, 4 launched → 2 admitted + 2 pending.
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);

    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    for (int i = 0; i < 4; i++) {
        Dim3 idx = {static_cast<uint32_t>(i), 0, 0};
        auto block = make_block(idx, /*threads=*/32, /*shared_mem=*/0, stmts);
        REQUIRE(sm.add_block(std::move(block)));
    }
    REQUIRE(sm.get_admitted_block_count() == 2);
    REQUIRE(sm.get_pending_block_count() == 2);

    // Pending must not silently disappear when cleanup_finished_blocks
    // runs without finding any finished warps.
    sm.cleanup_finished_blocks();
    REQUIRE(sm.get_pending_block_count() == 2);
    REQUIRE(sm.get_total_block_count() == 4);
}

TEST_CASE(
    "SMContext::add_block rejects blocks that absolutely cannot fit",
    "[sm][admission][streaming][negative]") {
    // 1 SM, max 2 warps. A single block asking for 4 warps can NEVER
    // fit — add_block must return false, NOT silently queue it forever.
    ResourceManager::instance().initialize(/*num_sms=*/1, /*shared_mem=*/8192);

    SMContext sm(/*max_warps=*/2, /*max_threads=*/64, /*shared_mem=*/8192,
                 /*sm_id=*/0);
    std::vector<ptxemu::ir::StatementContext> stmts;

    Dim3 idx = {0, 0, 0};
    auto block = make_block(idx, /*threads=*/128, /*shared_mem=*/0, stmts);
    bool accepted = sm.add_block(std::move(block));

    // Hard reject — pending must NOT be used as a dumping ground for impossible blocks.
    REQUIRE(accepted == false);
    REQUIRE(sm.get_total_block_count() == 0);
    REQUIRE(sm.get_admitted_block_count() == 0);
    REQUIRE(sm.get_pending_block_count() == 0);
}
