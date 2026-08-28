#ifndef PTXSIM_TESTING_WARP_TEST_UTILS_H
#define PTXSIM_TESTING_WARP_TEST_UTILS_H

#include "ptxsim/warp_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_factory.h"

#include <memory>
#include <vector>
#include <map>

namespace ptxsim::testing {

// ============================================================================
// TestWarpContext - RAII Warp/CTA Test Helper
// ============================================================================

class TestWarpContext {
public:
    explicit TestWarpContext(int num_lanes = 32);
    ~TestWarpContext();

    WarpContext& warp() { return *warp_; }
    SMContext& sm() { return sm_; }
    CTAContext* cta() { return cta_; }

    void reset();
    void set_active_mask(uint32_t mask);

private:
    SMContext sm_;
    CTAContext* cta_ = nullptr;
    WarpContext* warp_ = nullptr;
    std::vector<std::unique_ptr<ThreadContext>> threads_;
};

// ============================================================================
// Convenience Factory Functions
// ============================================================================

void init_factory_once();

WarpContext* create_warp_with_threads(int num_lanes = 32);

CTAContext* create_block(SMContext& sm, Dim3 grid, Dim3 block, Dim3 blockIdx);

void setup_warp(WarpContext& warp,
                std::vector<std::unique_ptr<ThreadContext>>& threads,
                int num_lanes = 32,
                CTAContext* cta = nullptr);

void reset_warp(WarpContext& warp, int num_lanes = 32);

// ============================================================================
// Inline Implementations
// ============================================================================

inline void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

inline WarpContext* create_warp_with_threads(int num_lanes) {
    WarpContext* warp = new WarpContext();
    warp->set_warp_id(0);

    Dim3 blockIdx{0, 0, 0};
    Dim3 gridDim{1, 1, 1};
    Dim3 blockDim{(uint32_t)num_lanes, 1, 1};

    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    std::vector<ptxemu::ir::StatementContext> stmts;

    std::vector<std::unique_ptr<ThreadContext>> threads;
    for (int i = 0; i < num_lanes; i++) {
        auto t = std::make_unique<ThreadContext>();
        Dim3 tid{(uint32_t)i, 0, 0};
        t->init(blockIdx, tid, gridDim, blockDim, stmts,
                &name2Sym, label2pc, nullptr, nullptr);
        t->set_state(RUN);
        threads.push_back(std::move(t));
    }

    for (int i = 0; i < num_lanes; i++) {
        warp->add_thread(std::move(threads[i]), i);
    }

    warp->set_active_mask(0xFFFFFFFF);
    return warp;
}

inline CTAContext* create_block(SMContext& sm, Dim3 grid, Dim3 block, Dim3 blockIdx) {
    std::vector<ptxemu::ir::StatementContext> stmts;
    std::map<std::string, int> label2pc;
    auto cta = std::make_unique<CTAContext>();
    cta->init(grid, block, blockIdx, stmts,
              nullptr, label2pc, nullptr, 0, 0);
    CTAContext* cta_ptr = cta.get();
    sm.add_block(std::move(cta));
    return cta_ptr;
}

inline void setup_warp(WarpContext& warp,
                        std::vector<std::unique_ptr<ThreadContext>>& threads,
                        int num_lanes,
                        CTAContext* cta) {
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {(uint32_t)num_lanes, 1, 1};

    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    std::vector<ptxemu::ir::StatementContext> stmts;

    threads.clear();

    for (int i = 0; i < num_lanes; i++) {
        auto t = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)i, 0, 0};
        t->init(blockIdx, tid, gridDim, blockDim, stmts,
                &name2Sym, label2pc, nullptr, cta);
        t->set_state(RUN);
        threads.push_back(std::move(t));
    }

    for (int i = 0; i < num_lanes; i++) {
        warp.add_thread(std::move(threads[i]), i);
    }

    warp.set_active_mask(0xFFFFFFFF);
}

inline void reset_warp(WarpContext& warp, int num_lanes) {
    for (int i = 0; i < num_lanes; i++) {
        auto* t = warp.get_thread(i);
        if (!t) continue;
        t->set_pc(0);
        t->set_state(RUN);
        warp.get_warp_state().threads[i].pc = 0;
        warp.get_warp_state().threads[i].next_pc = 0;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
    }
    warp.set_active_mask(0xFFFFFFFF);
}

// ============================================================================
// TestWarpContext Implementation
// ============================================================================

inline TestWarpContext::TestWarpContext(int num_lanes)
    : sm_(1, 32, 4096, 0) {
    sm_.init();

    Dim3 gridDim{1, 1, 1};
    Dim3 blockDim{(uint32_t)num_lanes, 1, 1};
    Dim3 blockIdx{0, 0, 0};

    std::vector<ptxemu::ir::StatementContext> stmts;
    std::map<std::string, int> label2pc;
    auto cta = std::make_unique<CTAContext>();
    cta->init(gridDim, blockDim, blockIdx, stmts,
              nullptr, label2pc, nullptr, 0, 0);
    cta_ = cta.get();
    sm_.add_block(std::move(cta));

    warp_ = sm_.get_warp(0);
    if (warp_) {
        warp_->set_warp_id(0);
    }

    threads_.reserve(num_lanes);
    for (int i = 0; i < num_lanes; i++) {
        threads_.push_back(std::make_unique<ThreadContext>());
    }
}

inline TestWarpContext::~TestWarpContext() = default;

inline void TestWarpContext::reset() {
    if (warp_) {
        reset_warp(*warp_, threads_.size());
    }
}

inline void TestWarpContext::set_active_mask(uint32_t mask) {
    if (warp_) {
        warp_->set_active_mask(mask);
    }
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_WARP_TEST_UTILS_H