// warp_executor_test_fixture.h
// =============================================================================
// Shared test fixture for IPtxEmuDevice delegation tests.
//
// Used by:
//   - tests/integration/simt/test_set_active_mask_overwrite.cpp
//   - tests/integration/warp/test_warp_status_snapshot.cpp
//   - tests/integration/warp/test_device_api_delegation_e2e.cc
//
// Sets up a fresh g_gpu_context with 1 SM containing 1 warp of 32 threads,
// plus an IPtxEmuDevice instance, so tests can exercise delegation paths
// (set_active_mask / set_next_pc / warp_exe_once / get_thread_state /
// get_warp_status) end-to-end.
//
// Per ptx-lessons-learned §1 (跨模块间接状态翻译): delegation tests
// must drive actual warp setup, NOT just create empty GPUContext (which
// has no warps and triggers early-return guards).
// =============================================================================

#ifndef PTXEMU_TEST_WARP_EXECUTOR_FIXTURE_H
#define PTXEMU_TEST_WARP_EXECUTOR_FIXTURE_H

#include "catch_amalgamated.hpp"
#include "ptxemu/device_api.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

extern std::unique_ptr<GPUContext> g_gpu_context;

namespace ptxemu {
namespace testing {

class WarpExecutorTestFixture {
public:
    // Default `statements = {}` preserves pre-change behavior for the 3 existing
    // fixture-using tests (test_set_active_mask_overwrite / test_warp_status_snapshot
    // / test_device_api_delegation_e2e). Tests that need schedulable instructions
    // for SMContext::exe_once() (e.g. attach-timing-consumer-e2e G4) pass
    // {make_ffma(...)} etc. via this parameter — the warp created by add_block()
    // carries those statements at PC=0 and exe_once() can execute them
    // (sm_context.cpp:266-268 guard passes when statements_size() > 0).
    explicit WarpExecutorTestFixture(
        std::vector<StatementContext> statements = {}) {
        saved_context_ = std::move(g_gpu_context);
        g_gpu_context = std::make_unique<GPUContext>();
        REQUIRE(g_gpu_context != nullptr);

        // init() creates the SMS array (1 SM with default config).
        g_gpu_context->init();

        sm_ = g_gpu_context->get_sm(0);
        REQUIRE(sm_ != nullptr);

        // Add a block with 32 threads = 1 warp to SM 0.
        auto block = std::make_unique<CTAContext>();
        std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
        std::map<std::string, int> label2pc;
        Dim3 grid_dim{1, 1, 1};
        Dim3 block_dim{32, 1, 1};
        Dim3 block_idx{0, 0, 0};
        block->init(grid_dim, block_dim, block_idx, statements, &name2Sym,
                    label2pc);

        bool ok = sm_->add_block(std::move(block));
        REQUIRE(ok == true);

        warp_ = sm_->get_warp(0);
        REQUIRE(warp_ != nullptr);

        // Create the device API implementation.
        dev_ = ptxemu::create_device();
        REQUIRE(dev_ != nullptr);
    }

    ~WarpExecutorTestFixture() {
        dev_.reset();
        g_gpu_context = std::move(saved_context_);
    }

    WarpExecutorTestFixture(const WarpExecutorTestFixture&) = delete;
    WarpExecutorTestFixture& operator=(const WarpExecutorTestFixture&) = delete;

    GPUContext* gpu() const { return g_gpu_context.get(); }
    SMContext* sm() const { return sm_; }
    WarpContext* warp() const { return warp_; }
    IPtxEmuDevice* dev() const { return dev_.get(); }

private:
    std::unique_ptr<GPUContext> saved_context_;
    SMContext* sm_ = nullptr;
    WarpContext* warp_ = nullptr;
    std::unique_ptr<IPtxEmuDevice> dev_;
};

}  // namespace testing
}  // namespace ptxemu

#endif  // PTXEMU_TEST_WARP_EXECUTOR_FIXTURE_H